@CLAUDE.md Now that Create takes as input the PoolId, we can do some caching.

Let's make it recommended to use PoolQuery::Dynamic() instead for Create operations.
If you recall, Dynamic will be routed to a container's Monitor method with kGlobalSchedule as input.
In this case, it will be admin_runtime.cc MonitorGetOrCreatePool.
The global schedule should work as follows:
1. Check if the pool exists locally. If it does, mark the task as completed.
2. Otherwise, set the pool query for the task to Bcast.

Update the code using logic builder agent and the documentation. Update all unit tests 
to ensure the Dynamic pool query is used for Create methods.