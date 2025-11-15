@CLAUDE.md

We need to implement boundary cases to resolve the following to Local in certain instances.
Update IsTaskLocal to acount for these.

ResolveDirectIdQuery: Is local if the container with this id is on this pool manager.
ResolveDirectHashQuery: Is local if the container with the id % num_containers is on this pool manager.
ResolveRangeQuery: Is local if the range has size 1 and the offset % num_containers is on this pool manager.

We may need to augment PoolManager to have a function to query if a container exists on this node.
