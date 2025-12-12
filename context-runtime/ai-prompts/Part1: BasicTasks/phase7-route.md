Create a function called RouteTask in worker.cc. I want you to take the code from the Run function where it was calling ResolvePoolQuery and put it in here. This function  should detect if this is a local schedule and call the scheduling monitor functions (e.g., kLocalSchedule and kGlobalSchedule) like the Run function did. We should remove the functions from worker.cc dedicated to this (e.g., CallMonitorForLocalSchedule)

Add a flag to the base task called TASK_ROUTED. This bit is set immediately after kLocalSchedule is called in RouteTask. This indicates the task should not undergo additional re-routing. This bit should be checked at the beginning of RouteTask. If the bit is true, then return true. Otherwise continue with the function.

@CLAUDE.md

# LocalSerialize

In hshm, we have the class context-transport-primitives/include/hermes_shm/data_structures/serialization/local_serialize.h

I want you to write some unit tests verifying that it works for hshm::priv::string and hshm::priv::vector in their respective unit tests.

In addition, I want you to write a separate unit test verifying that it works for just basic types like std::vector and std::string and int.
Place this under test/unit/data_structures/serialization/test_local_serialize.cc. Add to the cmakes.


@CLAUDE.md

# LocalTaskArchive

context-runtime/include/chimaera/task_archives.h provides a serialization using cereal. 

We want to have something similar, but for local. We should create a new set of classes analagous to those.
Also make a new file called: context-runtime/include/chimaera/local_task_archives.h.
This will use hshm::LocalSerialize instead of cereal. These tasks are only in the node, not outside.

For local, bulk is handled differently. If the object is a ShmPtr, just serialize the ShmPtr value.
If the object is a FullPtr, just serialize the shm_ part of it. If it is a raw pointer, just
serialize the data as a full memory copy of the data.

Write a unit test to verify that the new methods created can correctly serialize and deserialize tasks.

@CLUADE.md

ipc_manager currently has a function called Enqueue to place a task in the worker queues from clients or locally.
I want to change this design paradigm to be a little more flexible.
Instead, we should implement Send and Recv.
Here is how this change will need to be applied.

# Container & chi_refresh_repo

We will need to update Container to include methods for
serializing the task output using LocalSerialize.

We should add the methods:
1. LocalSaveIn
2. LocalLoadIn
3. LocalSaveOut
4. LocalLoadOut

These are effectively the same as their counterparts Saven

# Task futures

Async* operations will need to return a ``Future<Task>`` object instead of Task*.

Future will store:
1. A raw pointer to the Task (e.g., CreateTask*)
2. A FullPtr to a FutreShm object, which contains a hipc::vector representing the serialized task and an atomic is_complete_ bool.

Two constructors:
1. With AllocT* as input. It will allocate the FutureShm object. The FutureShm should inherit from ShmContainer.
2. With AllocT* and ShmPtr<FutureShm> as input.

# IpcManager Send & Recv

We will need to replace ipc_manager->Enqueue with Send / Recv

## Send(TaskT *task)

1. Create Future on the stack.
2. Serialize the TaskT using a LocalTaskInArchive object. Let's use a std::vector for the serialization buffer.
3. Copy the std::vector into the FutureShm's hipc::vector.
4. Enqueue the Future in the worker queue.
5. Return: Future<TaskT>

# Worker 

## Run
1. Pop will pop a future 

## EndTask

