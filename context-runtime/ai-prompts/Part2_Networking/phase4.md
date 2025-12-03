# Domain Resolution
We now need to focus on distributed scheduling. We can assume that a task has a PoolQuery object representing how to distribute the task among the pool. Right now, we have several options such as send to local container, directly hashing to a container, and broadcasting across all containers.


## Resolution Algorithm:
First check if GetDynamic was used in the PoolQuery. If so, then get the local container and call the Monitor function using the MonitorMode kGlobalSchedule. This will replace the domain query with something more concrete.

The resolved domain query should be stored in the RuntimeContext for the task. 

### Case 1: The task is hashed to a container
We locate the domain table for the pool. 
We then hash by module number containers get the container ID.
We then get the node ID that container is located on.
We create a physical PoolQuery to that node id.
We should add a helper to the pool manager to get a mapping of container id to physical addresss.

### Case 2: The task is directed to a specific container
Same as case 1. 


### Case 3: The task is broadcasted to a range of containers

The PoolQuery contains a range_offset and range_count. Two cases:

If the range is less than a certain configurable maximum, then we divide into physical PoolQuery objects for each container in the range. We resolve the container id to an address similar to case 1. 

Otherwise, we divide into smaller PoolQuery range objects that each cover a smaller range. There should be a configurable maximum number of PoolQueries produced. For now, let's say 16. If there are 256 containers, then there will be 16 PoolQueries produced, each that broadcasts to a subset of those containers.

### Case 4: The task is broadcasted across all containers

Calls Case 3 but with range_offset 0 and range_count equal to the number of containers.

## Worker Route
If the ResolvedPoolQuery object exactly one entry and the resolved node ID is this node, then we schedule the task as-is. Otherwise, the task is sent to the chimaera admin using the ClientSendTask method. The PoolQuery used should be LocalHash.

Otherwise, the task should be scheduled like it is now.
