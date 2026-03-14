#ifndef WRP_CTE_CORE_AUTOGEN_METHODS_H_
#define WRP_CTE_CORE_AUTOGEN_METHODS_H_

#include <chimaera/chimaera.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for core
 */

namespace wrp_cte::core {

namespace Method {

enum : chi::u32 {
  // Inherited methods
  kCreate = 0,
  kDestroy = 1,
  kMonitor = 9,

  // core-specific methods
  kRegisterTarget = 10,
  kUnregisterTarget = 11,
  kListTargets = 12,
  kStatTargets = 13,
  kGetOrCreateTag = 14,
  kPutBlob = 15,
  kGetBlob = 16,
  kReorganizeBlob = 17,
  kDelBlob = 18,
  kDelTag = 19,
  kGetTagSize = 20,
  kPollTelemetryLog = 21,
  kGetBlobScore = 22,
  kGetBlobSize = 23,
  kGetContainedBlobs = 24,
  kGetBlobInfo = 25,
  kTagQuery = 30,
  kBlobQuery = 31,
  kGetTargetInfo = 32,
  kFlushMetadata = 33,
  kFlushData = 34,

  kMaxMethodId = 35,
};

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "RegisterTarget";
    v[11] = "UnregisterTarget";
    v[12] = "ListTargets";
    v[13] = "StatTargets";
    v[14] = "GetOrCreateTag";
    v[15] = "PutBlob";
    v[16] = "GetBlob";
    v[17] = "ReorganizeBlob";
    v[18] = "DelBlob";
    v[19] = "DelTag";
    v[20] = "GetTagSize";
    v[21] = "PollTelemetryLog";
    v[22] = "GetBlobScore";
    v[23] = "GetBlobSize";
    v[24] = "GetContainedBlobs";
    v[25] = "GetBlobInfo";
    v[30] = "TagQuery";
    v[31] = "BlobQuery";
    v[32] = "GetTargetInfo";
    v[33] = "FlushMetadata";
    v[34] = "FlushData";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace wrp_cte::core

#endif  // CORE_AUTOGEN_METHODS_H_
