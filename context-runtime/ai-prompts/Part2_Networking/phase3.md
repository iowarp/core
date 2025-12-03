Use the incremental logic builder to initially implement this spec. Make sure to review @doc/MODULE_DEVELOPMENT_GUIDE.md when augmenting the chimod.

# Remote Queue Tasks

This will be adding several new functions and features to the admin chimod and other parts of the chimaera runtime to support distributed task scheduling.

## Configuration Changes

Add a hostfile parameter to the chimaera configuration. If the hostfile is empty, assume this host is the only node on the system. Use hshm::ParseHostfile for this.
Make sure to use hshm::ConfigParse::ExpandPath to expand the hostfile path before using ParseHostfile.

```cpp
// Parse a hostfile with multiple formats
std::vector<std::string> ParseHostfile(const std::string& hostfile_path) {
    std::vector<std::string> all_hosts = hshm::ConfigParse::ParseHostfile(hostfile_path);
    
    // Process and validate hosts
    std::vector<std::string> valid_hosts;
    for (const auto& host : all_hosts) {
        if (IsValidHostname(host)) {
            valid_hosts.push_back(host);
        } else {
            fprintf(stderr, "Warning: Invalid hostname '%s' skipped\n", host.c_str());
        }
    }
    
    return valid_hosts;
}

bool IsValidHostname(const std::string& hostname) {
    // Basic validation
    if (hostname.empty() || hostname.length() > 255) {
        return false;
    }
    
    // Check for valid characters
    for (char c : hostname) {
        if (!std::isalnum(c) && c != '-' && c != '.') {
            return false;
        }
    }
    
    return true;
}

// Example hostfile content:
/*
# Compute nodes
compute[001-064]-ib
compute[065-128]-ib

# GPU nodes  
gpu[01-16]-40g

# Special nodes
login1
login2
scheduler
storage[01-04]
*/
```

```cpp
// Expand environment variables in paths
std::string ExpandConfigPath(const std::string& template_path) {
    return hshm::ConfigParse::ExpandPath(template_path);
}

// Examples
std::string home_config = ExpandConfigPath("${HOME}/.config/myapp");
std::string data_path = ExpandConfigPath("${XDG_DATA_HOME}/myapp/data");
std::string temp_file = ExpandConfigPath("${TMPDIR}/myapp_${USER}.tmp");

// Complex expansion with multiple variables
std::string complex = ExpandConfigPath(
    "${HOME}/.cache/${APPLICATION_NAME}-${VERSION}/data"
);

// Set up environment and expand
hshm::SystemInfo::Setenv("APP_ROOT", "/opt/myapp", 1);
hshm::SystemInfo::Setenv("APP_VERSION", "2.1.0", 1);
std::string app_config = ExpandConfigPath("${APP_ROOT}/config-${APP_VERSION}.yaml");
```

## Container Server
The container server class should be updated to support serializing and copying tasks. Like Run, Monitor, and Del, these tasks should be structure with switch-case statements. The override functions will be placed in autogen/admin_lib_exec.h. Make sure to update admin chimod and MOD_NAME accordingly
```cpp
namespace chi {

/**
 * Represents a custom operation to perform.
 * Tasks are independent of Hermes.
 * */
#ifdef CHIMAERA_RUNTIME
class ContainerRuntime {
public:
  PoolId pool_id_;           /**< The unique name of a pool */
  std::string pool_name_;    /**< The unique semantic name of a pool */
  ContainerId container_id_; /**< The logical id of a container */

  /** Create a lane group */
  void CreateQueue(QueueId queue_id, u32 num_lanes, chi::IntFlag flags);

  /** Get lane */
  Lane *GetLane(QueueId queue_id, LaneId lane_id);

  /** Get lane */
  Lane *GetLaneByHash(QueueId queue_id, u32 hash);

  /** Virtual destructor */
  HSHM_DLL virtual ~Module() = default;

  /** Run a method of the task */
  HSHM_DLL virtual void Run(u32 method, Task *task, RunContext &rctx) = 0;

  /** Monitor a method of the task */
  HSHM_DLL virtual void Monitor(MonitorModeId mode, u32 method, hipc::FullPtr<Task> task,
                                RunContext &rctx) = 0;

  /** Delete a task */
  HSHM_DLL virtual void Del(const hipc::MemContext &ctx, u32 method,
                            hipc::FullPtr<Task> task) = 0;

  /** Duplicate a task into a new task */
  HSHM_DLL virtual void NewCopy(u32 method, 
                                const hipc::FullPtr<Task> &orig_task,
                                hipc::FullPtr<Task> &dup_task, bool deep) = 0;

  /** Serialize a task inputs */
  HSHM_DLL virtual void SaveIn(u32 method, chi::TaskOutputArchiveIN &ar,
                               Task *task) = 0;

  /** Deserialize task inputs */
  HSHM_DLL virtual TaskPointer LoadIn(u32 method,
                                      chi::TaskInputArchiveIN &ar) = 0;

  /** Serialize task inputs */
  HSHM_DLL virtual void SaveOut(u32 method, chi::TaskOutputArchiveOUT &ar,
                                Task *task) = 0;

  /** Deserialize task outputs */
  HSHM_DLL virtual void LoadOut(u32 method, chi::TaskInputArchiveOUT &ar,
                                Task *task) = 0;
};
#endif // CHIMAERA_RUNTIME
} // namespace chi
```

