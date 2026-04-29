/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#if HSHM_ENABLE_ZMQ
#ifndef _WIN32
#include <unistd.h>
#endif
#include <zmq.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/logging.h"
#include "lightbeam.h"
#include "posix_socket.h"

namespace hshm::lbm {

/** No-op free callback for zmq_msg_init_data zero-copy sends */
static inline void zmq_noop_free(void *data, void *hint) {
  (void)data;
  (void)hint;
}

/** Free zmq_msg_t handles stored in Bulk::desc from zero-copy recv */
static inline void ClearZmqRecvHandles(LbmMeta<> &meta) {
  for (auto &bulk : meta.recv) {
    if (bulk.desc) {
      zmq_msg_t *msg = static_cast<zmq_msg_t*>(bulk.desc);
      zmq_msg_close(msg);
      delete msg;
      bulk.desc = nullptr;
    }
  }
}

/** Action that reads ZMQ_EVENTS when epoll fires on ZMQ_FD.
 *  Required by ZMQ docs: the FD is edge-triggered and won't
 *  re-arm until the application reads ZMQ_EVENTS. */
class ZmqFiredAction : public EventAction {
 public:
  void *socket_;
  explicit ZmqFiredAction(void *socket) : socket_(socket) {}
  void Run(const EventInfo &event) override {
    (void)event;
    int zmq_events = 0;
    ::size_t opt_len = sizeof(zmq_events);  // LCOV_EXCL_LINE
    zmq_getsockopt(socket_, ZMQ_EVENTS, &zmq_events, &opt_len);
  }
};

class ZeroMqTransport : public Transport {
 private:
  static void* GetSharedContext() {
    // CtxOwner holds the shared ZMQ context and destroys it at program exit,
    // ensuring libzmq releases its internal resources and LeakSanitizer is clean.
    struct CtxOwner {
      void* ctx = nullptr;
      std::mutex mtx;
      ~CtxOwner() {
        if (ctx) {
          // zmq_ctx_shutdown() causes all blocking ZMQ calls on open sockets
          // to return immediately with ETERM.  This unblocks any background
          // receive threads (e.g. RecvZmqClientThread) that are polling the
          // socket, allowing them to exit cleanly.  zmq_ctx_destroy() would
          // otherwise block forever if a socket is still open (because the
          // Chimaera singleton is heap-allocated and its destructor -- which
          // calls ClientFinalize / closes the socket -- is never invoked).
          zmq_ctx_shutdown(ctx);
          zmq_ctx_destroy(ctx);
          ctx = nullptr;
        }
      }
    };
    static CtxOwner owner;
    std::lock_guard<std::mutex> lock(owner.mtx);
    if (!owner.ctx) {
      owner.ctx = zmq_ctx_new();
      zmq_ctx_set(owner.ctx, ZMQ_IO_THREADS, 2);
      HLOG(kInfo, "[ZeroMqTransport] Created shared context with 2 I/O threads");
    }
    return owner.ctx;
  }

 public:
  explicit ZeroMqTransport(TransportMode mode, const std::string& addr,
                           const std::string& protocol = "tcp", int port = 8192)
      : Transport(mode),
        addr_(addr),
        protocol_(protocol),
        port_(port),
        zmq_fired_action_(nullptr) {
    type_ = TransportType::kZeroMq;
    sock::InitSocketLib();

    // Optional: pin TCP traffic to a specific local network interface on
    // multi-rail fabrics (e.g. Aurora's Slingshot HSN where each compute
    // node has hsn0..hsn7). When LIGHTBEAM_TCP_DEVICE is set:
    //   - ROUTER (server) binds the device's IP directly: tcp://hsn0:port
    //   - DEALER (client) source-binds outbound traffic to the device
    //     using ZMQ's `<source>;<destination>` endpoint syntax, so
    //     connect() routes via the chosen NIC regardless of how the
    //     destination FQDN resolves. Both ends pinned to the same device
    //     -> symmetric routing -> no asymmetric drops at scale.
    const char *bind_device_env = std::getenv("LIGHTBEAM_TCP_DEVICE");
    std::string bind_device =
        (bind_device_env && *bind_device_env) ? bind_device_env : "";

    // Loopback endpoints must stay on the host's lo interface — clients
    // connecting to 127.0.0.1 / ::1 / localhost can't reach a socket
    // pinned to a routed NIC, and the server's local-only ROUTER (used
    // by same-host wrp_cte / runtime clients) MUST keep listening on
    // 127.0.0.1 even when LIGHTBEAM_TCP_DEVICE is set for cross-node
    // traffic. Otherwise every same-host client times out in
    // WaitForLocalServer.
    auto is_loopback_addr = [](const std::string &a) {
      return a == "127.0.0.1" || a == "::1" || a == "localhost";
    };

    std::string full_url;
    if (protocol_ == "ipc") {
      full_url = "ipc://" + addr_;
    } else if (!bind_device.empty() && !is_loopback_addr(addr_) &&
               mode == TransportMode::kClient) {
      // Client: source-bind outbound traffic to the chosen NIC so
      // connect() routes via that rail regardless of how the
      // destination FQDN resolves. ZMQ syntax:
      //   tcp://<src_dev>:0;<dst_addr>:<dst_port>
      full_url = protocol_ + "://" + bind_device + ":0;" + addr_ + ":" +
                 std::to_string(port_);
    } else {
      // Server: keep the original bind address. Aurora HSN FQDNs are
      // multi-A (one entry per rail); pinning the server to a single
      // rail's IP makes peer DEALERs whose FQDN lookup landed on a
      // different rail unable to reach us. Listening on 0.0.0.0
      // (every interface) lets any rail accept connections; the
      // client-side source-bind above is what makes outbound
      // routing symmetric.
      full_url = protocol_ + "://" + addr_ + ":" + std::to_string(port_);
    }

    if (mode == TransportMode::kClient) {
      // DEALER socket for client
      ctx_ = GetSharedContext();
      owns_ctx_ = false;
      socket_ = zmq_socket(ctx_, ZMQ_DEALER);

      // Set identity for ROUTER identification
      // Use hostname + PID to ensure uniqueness across Docker containers
      // (where multiple processes may have the same PID in different namespaces)
      char hostname_buf[64] = {};
      gethostname(hostname_buf, sizeof(hostname_buf) - 1);
      uint32_t pid = static_cast<uint32_t>(hshm::SystemInfo::GetPid());
      std::string identity = std::string(hostname_buf) + ":" +
                              std::to_string(pid);
      zmq_setsockopt(socket_, ZMQ_IDENTITY, identity.data(),
                      identity.size());

      HLOG(kDebug, "ZeroMqTransport(DEALER) connecting to URL: {}", full_url);

      int immediate = 0;
      zmq_setsockopt(socket_, ZMQ_IMMEDIATE, &immediate, sizeof(immediate));

      int timeout = 5000;
      zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));

      int sndbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_SNDBUF, &sndbuf, sizeof(sndbuf));

      int rcvbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_RCVBUF, &rcvbuf, sizeof(rcvbuf));

      // ZMTP heartbeat: detect dead connections within seconds
      int hb_ivl = 1000;    // Send ZMTP PING every 1 second
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_IVL, &hb_ivl, sizeof(hb_ivl));
      int hb_timeout = 3000; // Consider dead after 3s of no traffic
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TIMEOUT, &hb_timeout, sizeof(hb_timeout));
      int hb_ttl = 3000;     // Tell remote peer: drop me if no traffic for 3s
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TTL, &hb_ttl, sizeof(hb_ttl));

      int rc = zmq_connect(socket_, full_url.c_str());
      if (rc == -1) {
        std::string err = "ZeroMqTransport(DEALER) failed to connect to URL '" +
                          full_url + "': " + zmq_strerror(zmq_errno());
        zmq_close(socket_);
        throw std::runtime_error(err);
      }

      zmq_pollitem_t poll_item = {socket_, 0, ZMQ_POLLOUT, 0};
      int poll_timeout_ms = 5000;
      int poll_rc = zmq_poll(&poll_item, 1, poll_timeout_ms);

      if (poll_rc < 0) {
        HLOG(kError, "[ZeroMqTransport] Poll failed for {}: {}", full_url,
             zmq_strerror(zmq_errno()));
      } else if (poll_rc == 0) {
        HLOG(kWarning,
             "[ZeroMqTransport] Poll timeout - connection to {} may not be ready",
             full_url);
      }

      HLOG(kDebug, "ZeroMqTransport(DEALER) connected to {} (pid={})", full_url, pid);
      zmq_fired_action_.socket_ = socket_;
      StartMonitor("DEALER", full_url);
    } else {
      // ROUTER socket for server
      ctx_ = zmq_ctx_new();
      owns_ctx_ = true;
      zmq_ctx_set(ctx_, ZMQ_IO_THREADS, 2);
      socket_ = zmq_socket(ctx_, ZMQ_ROUTER);

      // Set mandatory routing - reject messages to unknown identities
      int mandatory = 1;
      zmq_setsockopt(socket_, ZMQ_ROUTER_MANDATORY, &mandatory, sizeof(mandatory));

      int rcvbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_RCVBUF, &rcvbuf, sizeof(rcvbuf));

      int sndbuf = 4 * 1024 * 1024;
      zmq_setsockopt(socket_, ZMQ_SNDBUF, &sndbuf, sizeof(sndbuf));

      // ZMTP heartbeat: detect dead client connections
      int hb_ivl = 1000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_IVL, &hb_ivl, sizeof(hb_ivl));
      int hb_timeout = 3000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TIMEOUT, &hb_timeout, sizeof(hb_timeout));
      int hb_ttl = 3000;
      zmq_setsockopt(socket_, ZMQ_HEARTBEAT_TTL, &hb_ttl, sizeof(hb_ttl));

      HLOG(kDebug, "ZeroMqTransport(ROUTER) binding to URL: {}", full_url);
      int rc = zmq_bind(socket_, full_url.c_str());
      if (rc == -1) {
        std::string err = "ZeroMqTransport(ROUTER) failed to bind to URL '" +
                          full_url + "': " + zmq_strerror(zmq_errno());
        zmq_close(socket_);
        zmq_ctx_destroy(ctx_);
        throw std::runtime_error(err);
      }
      HLOG(kDebug, "ZeroMqTransport(ROUTER) bound successfully to {}", full_url);
      zmq_fired_action_.socket_ = socket_;
      StartMonitor("ROUTER", full_url);
    }
  }

  ~ZeroMqTransport() {
    HLOG(kDebug, "ZeroMqTransport destructor - closing socket to {}:{}", addr_,
         port_);

    monitor_running_.store(false);
    if (monitor_thread_.joinable()) {
      monitor_thread_.join();
    }

    int linger = 0;  // Close immediately; don't wait for unsent messages

    zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));

    zmq_close(socket_);
    if (owns_ctx_) {
      zmq_ctx_destroy(ctx_);
    }
    HLOG(kDebug, "ZeroMqTransport destructor - socket closed");
  }

  // ZMQ socket monitor — emits diagnostic events whenever the underlying
  // ZMTP state machine changes (CONNECT / ACCEPT / HANDSHAKE / DISCONNECT
  // etc.). Spawns a reader thread that translates each event into an
  // HLOG(kInfo) line so we can see WHY a peer-to-peer link silently fails
  // to deliver application messages.
  void StartMonitor(const std::string& kind, const std::string& url) {
    static std::atomic<uint64_t> mon_id{0};
    monitor_endpoint_ = "inproc://lbm-mon-" +
                        std::to_string(mon_id.fetch_add(1));
    int rc = zmq_socket_monitor(socket_, monitor_endpoint_.c_str(),
                                ZMQ_EVENT_ALL);
    if (rc < 0) {
      HLOG(kWarning, "[ZMQ Monitor {}] zmq_socket_monitor failed: {}",
           kind, zmq_strerror(zmq_errno()));
      return;
    }
    monitor_running_.store(true);
    monitor_thread_ = std::thread([this, kind, url]() {
      void* mon = zmq_socket(ctx_, ZMQ_PAIR);
      if (!mon) return;
      int linger = 0;
      zmq_setsockopt(mon, ZMQ_LINGER, &linger, sizeof(linger));
      int timeout = 200;
      zmq_setsockopt(mon, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
      if (zmq_connect(mon, monitor_endpoint_.c_str()) < 0) {
        HLOG(kWarning, "[ZMQ Monitor {}] connect failed: {}", kind,
             zmq_strerror(zmq_errno()));
        zmq_close(mon);
        return;
      }
      HLOG(kInfo, "[ZMQ Monitor {}] watching {}", kind, url);
      while (monitor_running_.load()) {
        zmq_msg_t evt_msg;
        zmq_msg_init(&evt_msg);
        int n = zmq_msg_recv(&evt_msg, mon, 0);
        if (n < 0) {
          if (zmq_errno() == EAGAIN) {
            zmq_msg_close(&evt_msg);
            continue;
          }
          if (zmq_errno() == ETERM) {
            zmq_msg_close(&evt_msg);
            break;
          }
          zmq_msg_close(&evt_msg);
          continue;
        }
        if (zmq_msg_size(&evt_msg) < 6) {
          zmq_msg_close(&evt_msg);
          continue;
        }
        const uint8_t* d = static_cast<const uint8_t*>(zmq_msg_data(&evt_msg));
        uint16_t event = static_cast<uint16_t>(d[0] | (d[1] << 8));
        uint32_t value = 0;
        std::memcpy(&value, d + 2, 4);
        zmq_msg_close(&evt_msg);
        zmq_msg_t addr_msg;
        zmq_msg_init(&addr_msg);
        zmq_msg_recv(&addr_msg, mon, 0);
        std::string ep(static_cast<const char*>(zmq_msg_data(&addr_msg)),
                       zmq_msg_size(&addr_msg));
        zmq_msg_close(&addr_msg);
        const char* name = "?";
        switch (event) {
          case ZMQ_EVENT_CONNECTED: name = "CONNECTED"; break;
          case ZMQ_EVENT_CONNECT_DELAYED: name = "CONNECT_DELAYED"; break;
          case ZMQ_EVENT_CONNECT_RETRIED: name = "CONNECT_RETRIED"; break;
          case ZMQ_EVENT_LISTENING: name = "LISTENING"; break;
          case ZMQ_EVENT_BIND_FAILED: name = "BIND_FAILED"; break;
          case ZMQ_EVENT_ACCEPTED: name = "ACCEPTED"; break;
          case ZMQ_EVENT_ACCEPT_FAILED: name = "ACCEPT_FAILED"; break;
          case ZMQ_EVENT_CLOSED: name = "CLOSED"; break;
          case ZMQ_EVENT_CLOSE_FAILED: name = "CLOSE_FAILED"; break;
          case ZMQ_EVENT_DISCONNECTED: name = "DISCONNECTED"; break;
          case ZMQ_EVENT_MONITOR_STOPPED: name = "MONITOR_STOPPED"; break;
          case ZMQ_EVENT_HANDSHAKE_FAILED_NO_DETAIL:
            name = "HANDSHAKE_FAILED_NO_DETAIL"; break;
          case ZMQ_EVENT_HANDSHAKE_SUCCEEDED:
            name = "HANDSHAKE_SUCCEEDED"; break;
          case ZMQ_EVENT_HANDSHAKE_FAILED_PROTOCOL:
            name = "HANDSHAKE_FAILED_PROTOCOL"; break;
          case ZMQ_EVENT_HANDSHAKE_FAILED_AUTH:
            name = "HANDSHAKE_FAILED_AUTH"; break;
          default: name = "UNKNOWN"; break;
        }
        HLOG(kInfo, "[ZMQ Monitor {}] {} value={} ep={}", kind, name,
             value, ep);
        if (event == ZMQ_EVENT_MONITOR_STOPPED) break;
      }
      zmq_close(mon);
    });
  }

  Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size,
              u32 flags) {
    Bulk bulk;
    bulk.data = ptr;
    bulk.size = data_size;
    bulk.flags = hshm::bitfield32_t(flags);
    return bulk;
  }

  template <typename MetaT>
  int Send(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    // Serialize the multipart send so frames from different threads
    // don't interleave. Without this the ZMTP frame stream is racy:
    // identityA, delimiter, identityB (oops), metaA, ... — receiver
    // drops the message and our SWIM probe never gets a reply.
    std::lock_guard<std::mutex> lock(send_mtx_);

    // Compute send_bulks before serialization so receiver knows how many
    meta.send_bulks = 0;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (meta.send[i].flags.Any(BULK_XFER)) {
        meta.send_bulks++;
      }
    }

    std::vector<char> meta_buf;
    {
      hshm::ipc::GlobalSerialize<std::vector<char>> ar(meta_buf);
      ar(meta);
      ar.Finalize();
    }
    std::string meta_str(meta_buf.begin(), meta_buf.end());
    size_t write_bulk_count = meta.send_bulks;

    // ROUTER mode: prepend identity frame + empty delimiter
#if !HSHM_IS_GPU
    if (IsServer() && !meta.client_info_.identity_.empty()) {
      // Send identity frame
      int rc = zmq_send(socket_, meta.client_info_.identity_.data(),
                        meta.client_info_.identity_.size(), ZMQ_SNDMORE);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send(ROUTER) - identity frame FAILED: {}",
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
      // Send empty delimiter frame
      rc = zmq_send(socket_, "", 0, ZMQ_SNDMORE);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send(ROUTER) - delimiter frame FAILED: {}",
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
    } else
#endif
    if (IsClient()) {
      // DEALER mode: send empty delimiter frame
      int rc = zmq_send(socket_, "", 0, ZMQ_SNDMORE);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send(DEALER) - delimiter frame FAILED: {}",
             zmq_strerror(zmq_errno()));
        return zmq_errno();
      }
    }

    int base_flags = 0;
    int flags = base_flags;
    if (write_bulk_count > 0) {
      flags |= ZMQ_SNDMORE;
    }

    int rc = zmq_send(socket_, meta_str.data(), meta_str.size(), flags);
    if (rc == -1) {
      HLOG(kError, "ZeroMqTransport::Send - meta FAILED: {}",
           zmq_strerror(zmq_errno()));
      return zmq_errno();
    }

    size_t sent_count = 0;
    for (size_t i = 0; i < meta.send.size(); ++i) {
      if (!meta.send[i].flags.Any(BULK_XFER)) {
        continue;
      }

      flags = base_flags;
      sent_count++;
      if (sent_count < write_bulk_count) {
        flags |= ZMQ_SNDMORE;
      }

      zmq_msg_t msg;
      zmq_msg_init_data(&msg, meta.send[i].data.ptr_, meta.send[i].size,
                         zmq_noop_free, nullptr);
      rc = zmq_msg_send(&msg, socket_, flags);
      if (rc == -1) {
        HLOG(kError, "ZeroMqTransport::Send - bulk {} FAILED: {}", i,
             zmq_strerror(zmq_errno()));
        zmq_msg_close(&msg);
        return zmq_errno();
      }
    }
    return 0;
  }

  template <typename MetaT>
  ClientInfo Recv(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    // Mirror of Send's locking: a multipart Recv must consume identity /
    // delimiter / meta / bulk frames atomically or threads can take
    // each other's frames mid-message.
    std::lock_guard<std::mutex> lock(recv_mtx_);
    ClientInfo info;
    info.rc = RecvMetadata(meta, ctx);
    if (info.rc != 0) return info;
#if !HSHM_IS_GPU
    // Copy identity from recv into ClientInfo
    info.identity_ = meta.client_info_.identity_;
#endif
    // Set up recv entries from send descriptors
    for (const auto& send_bulk : meta.send) {
      Bulk recv_bulk;
      recv_bulk.size = send_bulk.size;
      recv_bulk.flags = send_bulk.flags;
      recv_bulk.data = hipc::FullPtr<char>::GetNull();
      meta.recv.push_back(recv_bulk);
    }
    info.rc = RecvBulks(meta, ctx);
    return info;
  }

 private:
  template <typename MetaT>
  int RecvMetadata(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;

    // ROUTER mode: receive identity frame first
    if (IsServer()) {
      zmq_msg_t identity_msg;
      zmq_msg_init(&identity_msg);
      int rc = zmq_msg_recv(&identity_msg, socket_, ZMQ_DONTWAIT);
      if (rc == -1) {
        int err = zmq_errno();
        zmq_msg_close(&identity_msg);
        return err;
      }
      // Store identity in meta for targeted Send responses
#if !HSHM_IS_GPU
      meta.client_info_.identity_ = std::string(
          static_cast<char*>(zmq_msg_data(&identity_msg)),
          zmq_msg_size(&identity_msg));
#endif
      zmq_msg_close(&identity_msg);

      // Receive and discard empty delimiter frame
      zmq_msg_t delim_msg;
      zmq_msg_init(&delim_msg);
      rc = zmq_msg_recv(&delim_msg, socket_, 0);
      zmq_msg_close(&delim_msg);
      if (rc == -1) {
        return zmq_errno();
      }
    } else {
      // DEALER mode: receive and discard empty delimiter frame
      zmq_msg_t delim_msg;
      zmq_msg_init(&delim_msg);
      int rc = zmq_msg_recv(&delim_msg, socket_, ZMQ_DONTWAIT);
      if (rc == -1) {
        int err = zmq_errno();
        zmq_msg_close(&delim_msg);
        return err;
      }
      zmq_msg_close(&delim_msg);
    }

    // Receive metadata frame
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int rc = zmq_msg_recv(&msg, socket_, 0);

    if (rc == -1) {
      int err = zmq_errno();
      zmq_msg_close(&msg);
      return err;
    }

    size_t msg_size = zmq_msg_size(&msg);
    try {
      std::vector<char> meta_buf(static_cast<char*>(zmq_msg_data(&msg)),
                                  static_cast<char*>(zmq_msg_data(&msg)) + msg_size);
      hshm::ipc::GlobalDeserialize<std::vector<char>> ar(meta_buf);
      ar(meta);
    } catch (const std::exception& e) {
      HLOG(kFatal,
           "ZeroMQ RecvMetadata: Deserialization failed - {} (msg_size={})",
           e.what(), msg_size);
      zmq_msg_close(&msg);
      return -1;
    }
    zmq_msg_close(&msg);
    return 0;
  }

  template <typename MetaT>
  int RecvBulks(MetaT& meta, const LbmContext& ctx = LbmContext()) {
    (void)ctx;
    size_t recv_count = 0;
    for (size_t i = 0; i < meta.recv.size(); ++i) {
      if (!meta.recv[i].flags.Any(BULK_XFER)) {
        continue;
      }
      recv_count++;
      int flags = (recv_count < meta.send_bulks) ? ZMQ_RCVMORE : 0;

      if (meta.recv[i].data.ptr_) {
        zmq_msg_t zmq_msg;
        zmq_msg_init(&zmq_msg);
        int rc = zmq_msg_recv(&zmq_msg, socket_, flags);
        if (rc == -1) {
          int err = zmq_errno();
          zmq_msg_close(&zmq_msg);
          return err;
        }
        memcpy(meta.recv[i].data.ptr_,
               zmq_msg_data(&zmq_msg), meta.recv[i].size);
        zmq_msg_close(&zmq_msg);
      } else {
        zmq_msg_t *zmq_msg = new zmq_msg_t;
        zmq_msg_init(zmq_msg);
        int rc = zmq_msg_recv(zmq_msg, socket_, flags);
        if (rc == -1) {
          int err = zmq_errno();
          zmq_msg_close(zmq_msg);
          delete zmq_msg;
          return err;
        }
        char *zmq_data = static_cast<char*>(zmq_msg_data(zmq_msg));
        meta.recv[i].data.ptr_ = zmq_data;
        meta.recv[i].data.shm_.alloc_id_ = hipc::AllocatorId::GetNull();
        meta.recv[i].data.shm_.off_ = reinterpret_cast<size_t>(zmq_data);
        meta.recv[i].desc = zmq_msg;
      }
    }
    return 0;
  }

 public:
  void ClearRecvHandles(LbmMeta<>& meta) {
    ClearZmqRecvHandles(meta);
  }

  void RegisterEventManager(EventManager &em) {
    int fd;
    size_t fd_size = sizeof(fd);
    zmq_getsockopt(socket_, ZMQ_FD, &fd, reinterpret_cast<::size_t *>(&fd_size));
    if (fd >= 0) {
      em.AddEvent(fd, kDefaultReadEvent, nullptr);
    }
  }

  std::string GetAddress() const { return addr_; }

  /** Check if the server is still alive via a TCP connect probe. */
  bool IsServerAlive(const LbmContext& ctx = LbmContext()) const {
    (void)ctx;
    if (protocol_ == "ipc") {
      // Unix domain socket — try connect
      sock::socket_t fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
      if (fd == sock::kInvalidSocket) return false;
      struct sockaddr_un sun;
      std::memset(&sun, 0, sizeof(sun));
      sun.sun_family = AF_UNIX;
      std::strncpy(sun.sun_path, addr_.c_str(), sizeof(sun.sun_path) - 1);
      int rc = ::connect(fd, reinterpret_cast<struct sockaddr*>(&sun),
                         sizeof(sun));
      sock::Close(fd);
      return rc == 0;
    }
    // TCP — probe addr_:port_
    sock::socket_t fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == sock::kInvalidSocket) return false;
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 500000;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    struct sockaddr_in sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(static_cast<uint16_t>(port_));
    ::inet_pton(AF_INET, addr_.c_str(), &sa.sin_addr);
    int rc = ::connect(fd, reinterpret_cast<struct sockaddr*>(&sa),
                       sizeof(sa));
    sock::Close(fd);
    return rc == 0;
  }

 private:
  std::string addr_;
  std::string protocol_;
  int port_;
  void* ctx_;
  bool owns_ctx_;
  void* socket_;
  ZmqFiredAction zmq_fired_action_;
  // ZMQ sockets are not thread-safe (per the ZeroMQ guide). Multipart
  // sends (identity / delimiter / meta / N bulk frames) issued from
  // multiple chimaera worker threads on the same socket can interleave,
  // corrupting the ZMTP frame stream. Receivers then silently drop
  // unparseable messages — appearing as e.g. SWIM HeartbeatProbe
  // timeouts on multi-node deployments. Hold these around Send/Recv so
  // each multipart message is atomic.
  std::mutex send_mtx_;
  std::mutex recv_mtx_;
  // ZMQ socket monitor — diagnostic for ZMTP-layer connection events.
  std::thread monitor_thread_;
  std::atomic<bool> monitor_running_{false};
  std::string monitor_endpoint_;
};

}  // namespace hshm::lbm

#endif  // HSHM_ENABLE_ZMQ
