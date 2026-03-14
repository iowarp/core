#ifndef CHIMAERA_ADMIN_AUTOGEN_METHODS_H_
#define CHIMAERA_ADMIN_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for admin
 */

namespace chimaera::admin {

namespace Method {

enum : chi::u32 {
  // Inherited methods
  kCreate = 0,
  kDestroy = 1,
  kMonitor = 9,

  // admin-specific methods
  kGetOrCreatePool = 10,
  kDestroyPool = 11,
  kStopRuntime = 12,
  kFlush = 13,
  kSend = 14,
  kRecv = 15,
  kClientConnect = 16,
  kSubmitBatch = 18,
  kWreapDeadIpcs = 19,
  kClientRecv = 20,
  kClientSend = 21,
  kRegisterMemory = 22,
  kRestartContainers = 23,
  kAddNode = 24,
  kChangeAddressTable = 25,
  kMigrateContainers = 26,
  kHeartbeat = 27,
  kHeartbeatProbe = 28,
  kProbeRequest = 29,
  kRecoverContainers = 30,
  kSystemMonitor = 31,
  kAnnounceShutdown = 32,
  kRegisterGpuContainer = 33,

  kMaxMethodId = 34,
};

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "GetOrCreatePool";
    v[11] = "DestroyPool";
    v[12] = "StopRuntime";
    v[13] = "Flush";
    v[14] = "Send";
    v[15] = "Recv";
    v[16] = "ClientConnect";
    v[18] = "SubmitBatch";
    v[19] = "WreapDeadIpcs";
    v[20] = "ClientRecv";
    v[21] = "ClientSend";
    v[22] = "RegisterMemory";
    v[23] = "RestartContainers";
    v[24] = "AddNode";
    v[25] = "ChangeAddressTable";
    v[26] = "MigrateContainers";
    v[27] = "Heartbeat";
    v[28] = "HeartbeatProbe";
    v[29] = "ProbeRequest";
    v[30] = "RecoverContainers";
    v[31] = "SystemMonitor";
    v[32] = "AnnounceShutdown";
    v[33] = "RegisterGpuContainer";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace chimaera::admin

#endif  // ADMIN_AUTOGEN_METHODS_H_
