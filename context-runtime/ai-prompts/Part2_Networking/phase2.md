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

## Detecting the current host
We should have a function in the initialization of the chimaera runtime that identifies this host in the set of hosts on the provided hostfile. This can be done by iterating over the set of hosts and spawning a lightbeam tcp server. Check @ai-prompts/hshm-context.md for details on lightbeam. Make sure to catch exception if the tcp server does not start. If none of the servers start, then exit the runtime

The 64-bit representation of the host string should be stored in the main allocator's shared memory header as "node ID".

## Core Functionality

**Inter-Node Communication**: Handles task distribution and result collection across the distributed system
**Task Serialization**: Manages efficient serialization/deserialization of task parameters and data
**Bulk Data Transfer**: Supports large binary data movement with optimized transfer mechanisms
**Archive Management**: Provides four distinct archive types for different serialization needs

## Task Serialization

Implement serializers that serialize different parts of the task. All tasks  should implement methods named SerializeIn and SerializeOut. Make sure all existing tasks do this.
- **SerializeIn**: (De)serializes task entries labeled "IN" or "INOUT"
- **SerializeOut**: (De)serializes task parameters labeled "OUT" or "INOUT"

The base class Task should implement BaseSerializeIn and BaseSerializeOut. This will serialize the parts of the task that every task contains. Derived classes should not call BaseSerializeIn. 

## Task Archivers

### Here is the general flow:

#### NODE A sends tasks to NODE B:
1. We want to serialize a set of task inputs. A TaskOutputArchiveIN is created. Initially the number of tasks being serialized is passed to the archive. ``(ar << num_tasks)``. Since this is not a base class of type Task, default cereal is used.
2. Next we begin serializing the tasks. container->SaveIn(TaskOutputArchiveIN &ar, task) is called. Container is the container that the task is designated to.
3. SaveIn has a switch-case to type-cast the task to its concrete task type. E.g., it will convert Task to CreateTask and then use the serialize operator for the archive: ``(ar << cast<(CreateTask&)>task)``
4 . Internally, ar will detect the type is derived from Task and first call BaseSerializeIn and then SerializeIn. The task is expected to have been casted to its concrete type during the switch-case.
5. After all tasks have been serialized, they resulting std::string from cereal will be exposed to the client and then transferred using Send.

#### NODE B receives tasks from NODE A
On the node receiving a set of tasks:
* Essentially the reverse of those operations, except it uses a TaskInputArchiveIN and LoadIn functions.

#### NODE B finishes tasks and sends outputs to A
After task completes on the remote:
1. Essentially the same as when sending before except it uses TaskOutputArchiveOUT and SaveOut functions.

#### NODE A recieves outputs from NODE B
After task completion received on the remote:
1. Essentially the same as when recieving expcept it uses TaskInputArchiveOUT and LoadOut functions.

### Basic Serialization Operations
Main operators
* ``ar <<`` serialize (only for TaskOutput* archives)
* ``ar >>`` deserialize (only for TaskInput* archives)
* ``ar(a, b, c)`` serialize or deserialize depending on the archive
* ``ar.bulk(hipc::Pointer p, size_t size, u32 flags)``: Bulk transfers

### Bulk Data Transfer Function

```cpp
bulk(hipc::Pointer p, size_t size, u32 flags);
```

**Transfer Flags**:
- **CHI_WRITE**: The data of pointer p should be copied to the remote location
- **CHI_EXPOSE**: The pointer p should be copied to the remote so the remote can write to it

This should internally 

### Archive Types

Four distinct archive types handle different serialization scenarios:
- **TaskOutputArchiveIN**: Serialize IN params of task using SerializeIn
- **TaskInputArchiveIN**: Deserialize IN params of task using SerializeIn
- **TaskOutputArchiveOUT**: Serialize OUT params of task using SerializeOut  
- **TaskInputArchiveOUT**: Deserialize OUT params of task using SerializeOut

## Admin Chimod Changes
Create a local queue for SendIn and SendOut. 
1. Implement a ClientSendTaskIn function. This function will iterate over the CHI_CUR_LANE and pop all current tasks on that lane. It will create a map of archives where the key is a physical DomainId and the value is a BinaryOutputArchiveIN.  use a for loop using a variable storing current lane size, not a while loop. We should add a new parameter to the base task called net_key_ that uniquely identifies the task in the network queue. This should just be a (u64) of the task pointer since that is unique.
2. Implement a ServerRecvTaskIn function. This function is a periodc task that will receive task inputs and deserialize them. The resulting tasks will be scheduled in the local runtime.
3. Implement a ServerSendTaskOut function. Similar to 1, but BinaryOutputArchiveOUT.
4. Implement a ClientRecvTaskOut function. This is a periodic task that will receive task outputs. The period should be a configurable parameter for now. It deserializes outputs to the original task structures based on the net_key_.

## Container Server
The container server class should be updated to support serializing and copying tasks. Like Run, Monitor, and Del, these tasks should be structure with switch-case statements.
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
  HSHM_DLL virtual void SerializeOut(u32 method, chi::TaskOutputArchiveOUT &ar,
                                Task *task) = 0;

  /** Deserialize task outputs */
  HSHM_DLL virtual void LoadOut(u32 method, chi::TaskInputArchiveOUT &ar,
                                Task *task) = 0;
};
#endif // CHIMAERA_RUNTIME
} // namespace chi
```

## Worker
Resolving a task should be updated to support distributed scheduling.

There are a few cases. 
1. If GetDynamic was used, then get the local container and call the Monitor function using the MonitorMode kGlobalSchedule. This will replace the domain query with something more concrete. Proceed to 2 and 3.
2. If the task does not resolve to kLocal addresses, then send the task to the local admin container for scheduling using the updated chimaera admin client API (ClientSendTask). 
3. Otherwise, if the task is local, then get the container to send this task to. Call the Monitor function with the kLocalSchedule MonitorMode to route the task to a specific lane. If the lane was initially empty, then the worker processing it likely will ignore it. 
