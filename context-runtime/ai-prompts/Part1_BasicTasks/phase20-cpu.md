@CLAUDE.md 

Add to the chimaera configuration the new parameters: 
1. first_busy_wait_: When there is no work for a worker, this is the amount of time we busy wait before sleeping. Default 15us.
2. sleep_increment_: How much do we sleep? On every iteration we will linearly increment the amount of sleep when there is no work. Default 20us.
2. max_sleep_: the maximum sleep increment can go. Default 100us

Add these configuration parameters to the src/config_manager.cc and implement the algorithm in worker.cc