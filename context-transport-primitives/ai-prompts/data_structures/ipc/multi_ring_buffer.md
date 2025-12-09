@CLAUDE.md

Implement a new queue type called multi_ring_buffer. It should be placed under context-transport-primitives/include/hermes_shm/data_structures/ipc/multi_ring_buffer.h

It is essentially a vector<ring_buffer<T>>. 

The multi_ring_buffer class should have the same exact template parameters as ring_buffer.h

It should also implement effectively the same typedefs as ring_buffer.h

It only implements two methods.

## multi_ring_buffer(AllocT *alloc, int num_lanes, int num_prios, int depth). 

The constructor.
This should intialize a vector of num_lanes * num_prios queues. Each queue should have initial depth ``depth``.

## GetLane(int lane_id, int prio).

Returns vec[lane_id * num_lanes + prio]. It should verify that lane_id and prio are within the acceptable values.
