# Flushing

@CLAUDE.md We need to develop a task to flush the runtime. This algorithm should be apart of the admin chimod. Just call the task FlushTask. The task will have no additional inputs outside basic task inputs and will output the total amount of work done. 

The flush task should work as follows:
1. Create a virtual method called GetWorkRemaining as part of the Container base class. This should return a u64 indiciating the amount of work left to do in this container. This should be implemented in each chimod, so make it a pure virtual function. 
2. Create a virtual method called UpdateWork as part of the Container base class. It takes as input a FullPtr to a task, the RunContext, and an integer increment value. 
3. The flush task in the runtime code should call the GetWorkRemaining for each Container on the system. If the total work is 0, flushing should return. Otherwise, flushing should be false.

Flush should check the work remaining in a while loop that calls a new WorkOrchestrator method described next. We should add a method to the WorkOrchestrator called HasWorkRemaining that iterates over the containers and calculates the sum instead. 