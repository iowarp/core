
Use the incremental code building agent to implement this. Then verify it compiles with the code compilation agent.

### Admin ChiMod

This is a special chimod that the chimaera runtime should always find. If it is not found, then a fatal error should occur. This chimod is responsible for creating chipools, destroying them, and stopping the runtime. Processes initially send tasks containing the parameters to the chimod they want to instantiate to the admin chimod, which then distributes the chipool. It should use the PoolManager singleton to create containers locally. The chimod has three main tasks:
1. CreatePool
2. DestroyPool
3. StopRuntime

When creating a container, a table should be built mapping DomainIds to either node ids or other DomainIds. These are referred to as domain tables. These tables should be stored as part of the pool metadata in PoolInfo. Two domains should be stored: kLocal and kGlobal. Local domain maps containers on this node to the global DomainId. Global maps DomainId to physical DomainIds, representing node Ids. The global domain table should be consistent across all nodes. 

For now, set the ContainerId to 0.

#### Create Method
The admin chimod should have a templated BaseCreateTask class. It takes as input a CreateParamsT. This data structure should be defined for each chimod. It should contain 
a static constant named chimod_lib_name, which holds ${namespace}_${chimod}. This is used by the module manager to locate the chimod associated with the container. E.g., it may search the path lib${namespace}_${chimod}.so. This should correspond to the names output by the CMakeLists.txt. Namespace is the namespace stored in chimaera_repo.yaml.

The CreateTask for all chimods should inherit from this base class, including the admin chimod's CreateTask. The parameters to this class should essentially be the same as CreateTask, but it should also have variable arguments to instantiate the CreateParamsT. The BaseCreateTask should have a  hipc::string for storing the serialized CreateParamsT. The string is initially unsized. 

TheTask data structure should be augmented to have templated ``Serialize(hipc::string &, args..)`` and ``OutT Deserialize(hipc::string &)``. These funtions internally use the cereal library's BinaryOutputArchive for serializing and deserializing a set of data structures.

When creating a pool, the Container for the specific class should be created based on the chimod_lib_name variable. The specific Create function for the container is then called with the CreateTask.

#### Destroy Method
The DestroyTask for each chimod should be a simple typedef of the Admin's DestroyTask. It should not be defined for each chimod uniquely.