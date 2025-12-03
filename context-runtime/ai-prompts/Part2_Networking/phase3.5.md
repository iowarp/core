# Addressing Containers

Containers are uniquely identified by an integer within a pool. 

Tasks are sent to containers, rather than to nodes or processes.

However, we must have a way to Address containers.

Implement this plan.

## Pool Query

Rename DomainQuery to PoolQuery.

PoolQuery is used to route a task to one or more containers. Containers can have one or more addresses.

## Container Addresses

Addresses have three components:
* PoolId: The pool the address is for
* GroupId: The container group for the address. Containers can be divided into groups within the pool. Currently there should be three groups: Physical, Local and Global. Local containers represents the containers on THIS node. Global containers represents the set of all containers. Physical address is a wrapper around this node_id.
* MinorId: The unique integer ID of an element in the group. This can be a node id or container id.

## AddressTable

You should have two unordered_maps. Both maps are from Address -> Address. One map is for converting Local addresses to Global addresses. Another map is for converting Global addresses to Physical addresses.



