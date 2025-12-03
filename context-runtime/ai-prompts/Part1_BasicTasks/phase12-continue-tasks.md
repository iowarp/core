@CLAUDE.md

Tasks will wait for two reasons:
1. A time constraint (periodic polling)
2. It spawned a subtask and needs to wait for its completion (cooperative)

Right now, workers have one unified queue for holding both. We should have
two queues. 
1. periodic_queue: A priority_queue for periodic tasks. Lower times should be first in the queue.
2. blocked_queue: A set of hshm::spsc_queue<RunContext> representing tasks waiting for other tasks

## AddToBlockedQueue

## ProcessEventQueue

Let's say the worker has 4 hshm::spsc_queue<RunContext> data structures.
Each are 1024. 
This happens in the constructor, not this function.
Each queue stores tasks based on the number of times they have been blocked.

[0] stores tasks blocked <=2 (checked every % 2 iterations)
[1] stores tasks blocked <= 4 (checked every % 4 iterations)
[2] stores tasks blocked <= 8 (checked every % 8  iterations)
[3] stores tasks blocked > 8 (checked every % 16 iterations)

## ProcessPeriodicQueue

Let's say just one priority_queue<RunContext*>. I'm not expecting a billion of these.

The RunContext stores the time the task began blocking in AddToBlockedQueue.

If the time since the block began surpasses the time threshold, then execute the task.
