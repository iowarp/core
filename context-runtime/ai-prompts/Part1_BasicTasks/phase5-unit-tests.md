# Build unit tests

Use the unit test agent to build a basic unit test that starts the chimaera runtime and client, and then schedules a MOD_NAME custom task. The task should wait for completion. Place unit tests in a subdirectory called test/unit. 

Use the code reviewer and compiler agent to build CMakeList.txt for each subdirectory created. Use catch2 for tests, which is included by hshm.

# MOD_NAME

Use the incremental logic builder agent to augment the MOD_NAME chimod client to support periodic and fire & forget tasks.

