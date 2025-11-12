@CLAUDE.md Use incremental builder. Fix the equality operator for PoolId. It should not support equality to an  int. It should only support another PoolId. In addition, we should not support PoolId creation from just a single number. Use IsNull instead of == 0 for PoolId validity checks.

@CLAUDE.md There's an infinite loop of tasks calling AddToBlockedQueue during Wait and ContinueBlockedTasks continuously rechecking it.

The main problem is that task->Wait does not add the "this" task to the "current" task's subtask 
structure in RunContext. AreSubtasksCompleted always completes despite it not actually being 
complete. 