# Locking and Synchronization

We should create two new types of mutexes used for the chimaera runtime: CoMutex and CoRwLock. These two represent "coroutine" mutex and "coroutine" reader-writer lock.

These locks mainly use boost fiber to function, though some external synchronization using std::mutex is required. 

These should be two separate files: comutex.h and corwlock.h. These will be used only within runtime code and have no client code.

## CoMutex

Let's say 3 tasks try to acquire the mutex. Let's say that all three tasks come from different TaskNodes. At least one of the tasks will win. However, tasks do not exactly own comutex. Instead, a TaskNode holds the lock. If two tasks belonging to the same TaskNode (i.e., they only differ in minor number) then both tasks will be allowed to continue. This prevents deadlocks.

Internally, comutex should store an unordered_map[TaskNode] -> list<FullPtr<Task>>. TaskNode should has based on everything except minor number. This way all tasks waiting for this comutex will be processed simultaneously.

During an unlock operation, the next TaskNode group will be used. list<FullPtr<Task>> will be iterated over. Each task in the list will be sent back to its lane (stored in their task->run_ctx_). 

## CoRwLock

This exposes ReadLock, ReadUnlock, WriteLock, and WriteUnlock.

This is very similar to CoMutex. However, if the CoMutex is held by a reader, then all ReadLock requests will continue. If a WriteLock was called during a ReadLock, then it will be added to the block map. 

For a CoMutex held by writes, it will behave exactly the same as CoMutex. Any task not belonging to the TaskNode will be blocked. During WriteUnlock, the next TaskNode group will be unblocked by adding them back to their assigned lane (stored in their task->run_ctx_).

## Scope locks

Implement ScopedCoMutex and ScodeRwMutex. These mutexes are simple 
