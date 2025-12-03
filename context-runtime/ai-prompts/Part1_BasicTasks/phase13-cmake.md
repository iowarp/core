@CLAUDE.md 
 
I want to re_invision the cmake infrastructure. The cmake are complicated and not easily used in external projects.

## cmake/ChimaeraCommon.cmake

This file contains all code that is common between code that links to chimaera and the chimaera code itself.

### find packages

This section should find all packages needed to compile the chimaera code, mainly HermesShm and boost.

### add_chimod_client

This function should compile a chimod's client library. It is primarily a wrapper around add_library. It takes as input the following:
SOURCES
COMPILE_DEFINITIONS
LINK_LIBRARIES
LINK_DIRECTORIES
INCLUDE_LIBRARIES
INCLUDE_DIRECTORIES

It will read the chimaera_mod.yaml file located in the current source directory. 
It is assumed that the cmake that invokes this function is in the same directory as a file called chimaera_mod.yaml.
chimaera_mod.yaml contains the following keys: module_name and namespace.
The main target produced by this function should be: namespace_module_name_client
In addition, an alias target namespace::module_name_client should be produced
Internally, it will automatically link the targets to the chimaera core library.

This will also install the targets to an export set.
When external projects want to link to this project, they should do find_package(namespace_module_name REQUIRED).

### add_chimod_runtime

This function will take as input the same sources as the client in addition to the runtime sources. It has the same parameters as add_chimod_client and does a similar task.

However, in this function, we produce the targets: namespace_module_name_runtime and namespace::module_name_runtime.

## cmake/ChimaeraConfig.cmake

The main config needs to include the common config and the main export configuration used by the core chimaera library. This way, when a project does find_package(chimaera_core), it will get the chimaera targets, its dependencies, and the ability to create external chimods.

