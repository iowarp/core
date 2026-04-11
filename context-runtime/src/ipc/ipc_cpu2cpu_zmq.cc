/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "chimaera/ipc_manager.h"
#include "chimaera/singletons.h"

namespace chi {

//==============================================================================
// RuntimeRecv: poll ZMQ transports for incoming client tasks
//==============================================================================

bool IpcCpu2CpuZmq::RuntimeRecv(IpcManager *ipc, u32 &tasks_received) {
  auto *pool_manager = CHI_POOL_MANAGER;
  bool did_work = false;
  tasks_received = 0;

  // Process both TCP and IPC servers
  for (int mode_idx = 0; mode_idx < 2; ++mode_idx) {
    IpcMode mode = (mode_idx == 0) ? IpcMode::kTcp : IpcMode::kIpc;
    hshm::lbm::Transport *transport = ipc->GetClientTransport(mode);
    if (!transport) continue;

    // Drain all pending messages from this transport
    while (true) {
      LoadTaskArchive archive;
      auto recv_info = transport->Recv(archive);
      int rc = recv_info.rc;
      if (rc == EAGAIN) break;
      if (rc != 0) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeRecv: Recv failed: {}", rc);
        break;
      }

      const auto &task_infos = archive.GetTaskInfos();
      if (task_infos.empty()) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeRecv: No task_infos in message");
        continue;
      }

      const auto &info = task_infos[0];
      PoolId pool_id = info.pool_id_;
      u32 method_id = info.method_id_;

      // Get container for deserialization
      Container *container = pool_manager->GetStaticContainer(pool_id);
      if (!container) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeRecv: Container not found "
             "for pool_id {}", pool_id);
        continue;
      }

      // Allocate and deserialize the task
      hipc::FullPtr<Task> task_ptr =
          container->AllocLoadTask(method_id, archive);
      if (task_ptr.IsNull()) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeRecv: Failed to deserialize task");
        continue;
      }

      // Create FutureShm for the task (server-side)
      hipc::FullPtr<FutureShm> future_shm = ipc->NewObj<FutureShm>();
      future_shm->pool_id_ = pool_id;
      future_shm->method_id_ = method_id;
      future_shm->origin_ = (mode == IpcMode::kTcp)
                                ? FutureShm::FUTURE_CLIENT_TCP
                                : FutureShm::FUTURE_CLIENT_IPC;
      future_shm->client_task_vaddr_ = info.task_id_.net_key_;
      future_shm->client_pid_ = info.task_id_.pid_;
      // Store transport and routing info for response
      future_shm->response_transport_ = transport;
      future_shm->response_fd_ = recv_info.fd_;
      // Store ZMQ identity from recv frame for response routing
      if (!recv_info.identity_.empty() &&
          recv_info.identity_.size() <=
              sizeof(future_shm->response_identity_)) {
        std::memcpy(future_shm->response_identity_,
                    recv_info.identity_.data(),
                    recv_info.identity_.size());
        future_shm->response_identity_len_ =
            static_cast<u32>(recv_info.identity_.size());
      }
      // Mark as copied so EndTask routes back via lightbeam
      future_shm->flags_.SetBits(FutureShm::FUTURE_WAS_COPIED);

      // Create Future and enqueue to worker lane
      Future<Task> future(future_shm.shm_, task_ptr);
      LaneId lane_id =
          ipc->GetScheduler()->ClientMapTask(ipc, future);
      auto *worker_queues = ipc->GetTaskQueue();
      auto &lane_ref = worker_queues->GetLane(lane_id, 0);
      bool was_empty = lane_ref.Empty();
      lane_ref.Push(future);
      if (was_empty) {
        ipc->AwakenWorker(&lane_ref);
      }

      did_work = true;
      tasks_received++;
    }
  }

  return did_work;
}

//==============================================================================
// EnqueueRuntimeSend: worker-inline enqueue to net_queue_
//==============================================================================

void IpcCpu2CpuZmq::EnqueueRuntimeSend(IpcManager *ipc, RunContext *run_ctx,
                                         u32 origin) {
  if (origin == FutureShm::FUTURE_CLIENT_TCP) {
    ipc->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendTcp);
  } else {
    ipc->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendIpc);
  }
}

//==============================================================================
// RuntimeSend: net-worker serialize and send response via ZMQ
//==============================================================================

bool IpcCpu2CpuZmq::RuntimeSend(
    IpcManager *ipc, u32 &tasks_sent,
    std::vector<hipc::FullPtr<Task>> &deferred_deletes) {
  auto *pool_manager = CHI_POOL_MANAGER;
  bool did_work = false;
  tasks_sent = 0;

  // Flush deferred deletes from previous invocation.
  // Zero-copy send (zmq_msg_init_data) lets ZMQ's IO thread read from the
  // task buffer after zmq_msg_send returns. Deferring DelTask by one
  // invocation guarantees the IO thread has flushed the message.
  for (auto &t : deferred_deletes) {
    auto *del_container = pool_manager->GetStaticContainer(t->pool_id_);
    if (del_container) {
      del_container->DelTask(t->method_, t);
    }
  }
  deferred_deletes.clear();

  // Process both TCP and IPC queues
  for (int mode_idx = 0; mode_idx < 2; ++mode_idx) {
    NetQueuePriority priority =
        (mode_idx == 0) ? NetQueuePriority::kClientSendTcp
                        : NetQueuePriority::kClientSendIpc;
    IpcMode mode =
        (mode_idx == 0) ? IpcMode::kTcp : IpcMode::kIpc;

    Future<Task> queued_future;
    while (ipc->TryPopNetTask(priority, queued_future)) {
      auto origin_task = queued_future.GetTaskPtr();
      if (origin_task.IsNull()) continue;

      auto future_shm = queued_future.GetFutureShm();
      if (future_shm.IsNull()) continue;

      // Get container to serialize outputs
      Container *container =
          pool_manager->GetStaticContainer(origin_task->pool_id_);
      if (!container) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeSend: Container not found "
             "for pool_id {}", origin_task->pool_id_);
        continue;
      }

      // Get response transport and routing info from FutureShm
      hshm::lbm::Transport *response_transport =
          future_shm->response_transport_;
      if (!response_transport) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeSend: No response transport "
             "for mode {} pid {}", mode_idx, future_shm->client_pid_);
        continue;
      }

      // Preserve client's net_key for response routing
      origin_task->task_id_.net_key_ = future_shm->client_task_vaddr_;

      // Serialize task outputs
      SaveTaskArchive archive(MsgType::kSerializeOut, response_transport);
      container->SaveTask(origin_task->method_, archive, origin_task);

      // Set routing info for the response
      if (mode == IpcMode::kTcp) {
        if (future_shm->response_identity_len_ > 0) {
          archive.client_info_.identity_ =
              std::string(future_shm->response_identity_,
                          future_shm->response_identity_len_);
        } else {
          u32 client_pid = future_shm->client_pid_;
          archive.client_info_.identity_ = std::string(
              reinterpret_cast<const char *>(&client_pid), sizeof(client_pid));
        }
      } else if (mode == IpcMode::kIpc) {
        archive.client_info_.fd_ = future_shm->response_fd_;
      }

      // Send via lightbeam
      int rc = response_transport->Send(archive, hshm::lbm::LbmContext());
      if (rc != 0) {
        HLOG(kError, "IpcCpu2CpuZmq::RuntimeSend: Send failed: {}", rc);
      }

      // Defer task deletion for zero-copy send safety
      origin_task->ClearFlags(TASK_DATA_OWNER);
      deferred_deletes.push_back(origin_task);

      did_work = true;
      tasks_sent++;
    }
  }

  return did_work;
}

}  // namespace chi
