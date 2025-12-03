Do the following:
1. Use the incremental-logic-builder agent to build this specification
2. Use the code-compilation-reviewer to ensure the produced code is correct and compiles


Let's use the paradigm ServerInit and ClientInit for initializing the managers. Manager that only execute server-side should be ServerInit. Ones that do both should be ClientInit and ServerInit.

The process queue should store hipc::Pointer instead of u32. The pointer represents the .shm component of a FullPtr. It represents the shared-memory address of a Task.

Ensure that tasks have an "emplace constructor". Also ensure to use HIPC_CONTAINER_TEMPLATE for the task and use it as documented in hshm.

## Module Manager
The module manager is responsible for dynamically loading all modules on this node. It uses hshm::SharedLibrary for loading shared library symbols. It uses the environment variable LD_LIBRARY_PATH and CHI_REPO_PATH to scan for libraries. It will scan all files in each directory specified and check if they have the entrypoints needed to be a chimaera task. If they do, then they will be loaded and registered. 

ChiMods should have functions to query the name of the chimod and allocate a ChiContainer from the ChiMod. A table should be stored mapping chimod names to their hshm::SharedLibrary.

This will execute only in the runtime.

## ChiMod Specification

There is a client and a server part to the ChiMod. The server executes in the runtime. The client executes in user processes. Client code should be minimal. Client code essentially allocates tasks and places them in a queue using the IPC Manager. Client code should not perform networking, or any complex logic. Logic should be handled in the runtime code. Runtime objects should include methods for scheduling individual tasks locally (within the lanes) and globally. For each task, there should be a function for executing the task and another function for monitoring the task. For example, a task named CompressTask would have a run function named Compress and a monitor function called MonitorCompress. MonitorCompress should be a switch-case style design, and have different monitoring modes (e.g., kLocalSchedule, kGlobalSchedule).

Each ChiMod should have a function for creating a ChiPool and destroying it. When clients create the chipool, it should store the ID internally. 

Each ChiMod for a project should be located in a single directory called the ChiMod repo. ChiMod repos should have a strict structure that is ideal for code autogeneration. 
For example:
```bash
my_mod_repo
├── chimaera_repo.yaml  # Repo metadata
├── CMakeLists.txt      # Repo cmake
└── mod_name
    ├── chimaera_mod.yaml  # Module metadata, including task names
    ├── CMakeLists.txt     # Module cmake
    ├── autogen
    │   └── mod_name_lib_exec.h  
    │   └── mod_name_methods.h  
    ├── include
    │   └── mod_name
    │       ├── mod_name_client.h      # Client API 
    │       └── mod_name_tasks.h       # Task struct definitions 
    └── src
        ├── CMakeLists.txt          # Builds mod_name_client and runtime  
        ├── mod_name_client.cc    # Client API source
        └── mod_name_runtime.cc   # Runtime API source
```

### Container Server
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
  void CreateLocalQueue(QueueId queue_id, u32 num_lanes, chi::IntFlag flags);

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
};
#endif // CHIMAERA_RUNTIME
} // namespace chi

extern "C" {
/** Allocate a state (no construction) */
typedef Container *(*alloc_state_t)();
/** New state (with construction) */
typedef Container *(*new_state_t)(const chi::PoolId *pool_id,
                                  const char *pool_name);
/** Get the name of a task */
typedef const char *(*get_module_name_t)(void);
} // extern c

/** Used internally by task source file */
#define CHI_TASK_CC(TRAIT_CLASS, MOD_NAME)                                     \
  extern "C" {                                                                 \
  HSHM_DLL void *alloc_state(const chi::PoolId *pool_id,                       \
                             const char *pool_name) {                          \
    chi::Container *exec =                                                     \
        reinterpret_cast<chi::Container *>(new TYPE_UNWRAP(TRAIT_CLASS)());    \
    return exec;                                                               \
  }                                                                            \
  HSHM_DLL void *new_state(const chi::PoolId *pool_id,                         \
                           const char *pool_name) {                            \
    chi::Container *exec =                                                     \
        reinterpret_cast<chi::Container *>(new TYPE_UNWRAP(TRAIT_CLASS)());    \
    exec->Init(*pool_id, pool_name);                                           \
    return exec;                                                               \
  }                                                                            \
  HSHM_DLL const char *get_module_name(void) { return MOD_NAME; }              \
  HSHM_DLL bool is_chimaera_task_ = true;                                      \
  }
```

Internally, servers expose a queue stored in private memory. Tasks are routed from the process queue to lanes of the local queue. This routing is done in the Monitor method.
Monitor contains a switch-case statement that can be used to enact different phases of scheduling. Currently, there should be:
* MonitorModeId::kLocalSchedule: Route a task to a lane of the container's queue.

### Container Client
```cpp
namespace chi {

/** Represents the Module client-side */
class ContainerClient {
public:
  PoolId pool_id_; /**< The unique name of a pool */

  template <typename Ar> void serialize(Ar &ar) { ar(pool_id_); }
};
} // namespace chi
```

### Module Repo

Below is an example file tree of a module repo (my_mod_repo) containing one module (mod_name). 
```bash
my_mod_repo
├── chimaera_repo.yaml  # Repo metadata
├── CMakeLists.txt      # Repo cmake
└── mod_name
    ├── chimaera_mod.yaml  # Module metadata, including task names
    ├── CMakeLists.txt     # Module cmake
    ├── include
    │   └── mod_name
    │       ├── autogen
    │         └── mod_name_lib_exec.h  
    │         └── mod_name_methods.h  
    │       ├── mod_name_client.h      # Client API 
    │       └── mod_name_tasks.h       # Task struct definitions 
    └── src
        ├── CMakeLists.txt          # Builds mod_name_client and runtime  
        ├── mod_name_client.cc    # Client API source
        └── mod_name_runtime.cc   # Runtime API source
```

A module repo should have a namespace. This is used to affect how external libraries link to our targets and how the targets are named. E.g., if namespace "example" is chosen for this repo, the targets that get exported should be something like ``example::mod_name_client`` and "``example::mod_name_runtime``. The namespace should be stored in chimaera_repo.yaml and in the repo cmake. In addition, this namespace is used in the C++ code to make namespace commands. Aliases should be made so these targets can be linked internally in the project's cmake as well. 

Make sure to follow the naming convention [REPO_NAME]:[MOD_NAME] in the modules you build for namespaces.

#### chimeara_mod.yaml
This will include all methods that the container exposes. For example:
```yaml
# Inherited Methods
kCreate: 0        # 0
kDestroy: 1       # 1
kNodeFailure: -1  # 2
kRecover: -1      # 3
kMigrate: -1      # 4
kUpgrade: -1       # 5

# Custom Methods (start from 10)
kCompress: 10
kDecompress: 11
```

Here values of -1 mean that this container should not support those methods.

#### include/mod_name/mod_name_tasks.h
This contains all task struct definitions. For example:
```cpp
namespace example::mod_name {
/** The CompressTask task */
struct CompressTask : public chi::Task {
  /** SHM default constructor */
  explicit CompressTask(
      const hipc::CtxAllocator<CHI_ALLOC_T> &alloc)
      : chi::Task(alloc) {}

  /** Emplace constructor */
  explicit CompressTask(
      const hipc::CtxAllocator<CHI_ALLOC_T> &alloc, const chi::TaskNode &task_node,
      const chi::PoolId &pool_id, const chi::DomainQuery &pool_query)
      : chi::Task(alloc) {
    // Initialize task
    task_node_ = task_node;
    pool_ = pool_id;
    method_ = Method::kCompress;
    task_flags_.SetBits(0);
    pool_query_ = pool_query;

    // Custom
  }
};
}
```

IN, INOUT, and OUT are empty macros used just for helping visualize which parameters are inputs and which are outputs.

Tasks should be compatible with shared memory. Use hipc::strings and vectors for storing information within tasks.

#### include/mod_name/mod_name_client.h and cc
This will expose methods for external programs to send tasks to the chimaera runtime. This includes tasks for creating a pool of this container type.
For example, here is an example client code.
```cpp
namespace example::mod_name {
class Client : public ContainerClient {
  public:
    // Create a pool of mod_name
    void Compress(const hipc::MemContext &mctx,
                const chi::DomainQuery &pool_query) {
        // allocate the Create
        auto *ipc_manager = CHI_IPC_MANAGER;
        hipc::FullPtr<CompressTask> task = AsyncCreate(args)
        task->Wait();
        ipc_manager->DelTask<CompressTask>(task);
    }
    void AsyncCreate(const hipc::MemContext &mctx,
                     const chi::DomainQuery &pool_query) {
        auto *ipc_manager = CHI_IPC_MANAGER;
        FullPtr<CompressTask> task = ipc_manager->NewTask<CompressTask>(mctx, pool_query, create_ctx);
        ipc_manager->Enqueue(task);
    }
}
}
```

#### src/mod_name_client.cc
This is mainly for global variables and singletons mainly. Most of the client should be implemented in the header.

#### src/mod_name_runtime.cc
Contains the runtime task processing implementation. E.g.,
```cpp
namespace example::mod_name {
class Runtime : public ContainerRuntime {
 public:
   void Compress(hipc::FullPtr<CompressTask> task, chi::RunContext &rctx) {
     // Compress data
   }
   void MonitorCompress(chi::MonitorModeId mode, hipc::FullPtr<CompressTask> task, chi::RunContext &rctx) {
     switch (mode) {
        
     }
   }
}
}

CHI_TASK_CC(mod_name);
```

#### autogen/mod_name_lib_exec.h
A switch-case lambda function for every implemented method.
```cpp
void Run(Method method, hipc::FullPtr<Task> task, chi::RunContext &rctx) {
    switch (method) {
        case Method::kCreate: {
            Create(task.Cast<CreateTask>, rctx);
        }
        case Method::kCompress: {
            Compress(task.Cast<CompressTask>, rctx)
        }
    }
}

// Similar switch-case for other override functions
```

#### autogen/mod_name_methods.h

Defines the set of methods the module implements in C++. This should be autogenerated from the methods.yaml file.
```cpp
namespace example::mod_name {
class Method {
  CLS_CONST int kCreate = 1,
  CLS_CONST int kCompress = 10;
  CLS_CONST int kDecompress = 11;
}
}
```

## CMakeLists.txt

For chimods, we should create a CMakeLists.txt in a directory called CMake. It should have all find_packages for chimeara to work. We should include this CMake in our original CMakeLists.txt. In addition, this common cmake should include add_chimod_runtime and add_chimod_client functions. 

## Create the initial module repo

Create the module repo for chimaera for all modules that are automatically provided. Name this chimod repo chimods. The namespace should be chimaera. Build a module named MOD_NAME with kCreate and kCustom methods. 

## Documentation

In a fold named doc, document the coding style, structure, and style of creating a module. It should be detailed enough that a new module with its own methods (i.e., tasks) could be easily programmed by another AI. 