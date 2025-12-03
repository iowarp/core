@CLAUDE.md Use incremental agent

Create a new data structure called TaskStat. It has two fields:
```
struct TaskStat {
    size_t io_size_(0);   // I/O size in bytes
    size_t compute_(0);   // Normalized compute time.
}
```

Add the TaskStat to the task base class and call it stat_. This
will be used for ensuring efficient mapping of tasks to threads 
in the runtime and estimating wait times. It is not mandatory for 
tasks to set them.

Expose a new function in the base class for tasks called
size_t EstCpuTime(). It simply performs the following calculation:
io_size / 4GBPs + compute_ + 5. The time returned should be
in microseconds.

## WorkOrchestrator (work_orchestrator.cc)

The work orchestrator should track three different vectors of workers:
* all workers
* scheduler workers
* slow workers

When spawning, it will initially spawn all workers the same exact way and store in all.
But then it will assign each worker to one of the two other vectors.

## Estimating block time (task.cc)

Currently, the blocked time is simply set as a constant in Task::Wait. Let's 
change it to use these parameters. For now, let's do 
min(EstCpuTime, 50). Max 50us wait.

### AssignToThreadType
We will have a new functional called AssignToThreadType(ThreadType, FullPtr<Task>).
This will emplace into the worker's lane. For now, a simple round-robin algorithm
is fine. Store a static counter in the function to do this. Look at the Run
function to see how it polls the lane. You will use emplace instead of poll


## RouteLocal (worker.cc)

Increase the complexity of this function. If the EstCpuTime for the task is less than
50, then keep and return true. Otherwise, if not already a kSlow worker,
AssignToThreadType(kSlow, task).

## Configuration

Add a new configuration parameter to the workers key called slow_workers. The default
value should be 4. Let's update the default value for scheduler workers to also be 4.
Update jarvis to support setting and generating this new key.

## Bdev Write, Bdev Read, 

Use the I/O size parameter to update the stat struct.

## SendIn, RecvIn, SendOut, RecvOut

Hardcode the I/O size as 1MB. This should result in the execution on the slow workers.
