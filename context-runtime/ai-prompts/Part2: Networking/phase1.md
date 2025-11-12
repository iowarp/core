Use the incremental logic builder to initially implement this spec. 

## Task Serialization

Implement serializers that serialize different parts of the task. All tasks  should implement methods named SerializeIn and SerializeOut. Make sure all existing tasks do this.
- **SerializeIn**: (De)serializes task entries labeled "IN" or "INOUT"
- **SerializeOut**: (De)serializes task parameters labeled "OUT" or "INOUT"

The base class Task should implement BaseSerializeIn and BaseSerializeOut. This will serialize the parts of the task that every task contains. Derived classes should not call BaseSerializeIn. 

## Task Archivers
These use cereal for serialization. They serialize non-task objects using the traditional cereal path. For objects inheriting from class Task, they will call the specific SerializeIn and SerializeOut methods of the tasks. Tasks are required to have these methods implemented.

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
