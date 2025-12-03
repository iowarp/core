@CLAUDE.md In worker.cc, split up RouteTask should call the following sub-functions:

1. Instead of having a for loop in RouteTask that checks if we should process locally, create a separate function called ``IsTaskLocal`` that returns bool if it should be processed locally. 
2. Call ``RouteLocal`` if IsTaskLocal is true. RouteLocal is essentially everything within ``if (should_process_locally)``. 
3. Call ``RouteGlobal`` if IsTaskLocal is false. Link to the admin client library. Add a pointer variable singleton for the admin client. Initialize this singleton in the start of the runtime. In worker.cc, call this singleton with using the method ``ClientSendTaskIn``. 

@CLAUDE.md stop creating archives in the ClientSendIn, ClientRecvOut, ServerRecvIn, and ServerRecvOut methods (or their Async counterparts) in the client. The client code does not perform logic. They should take as input the original task, not serialized in any way, and just pass that to the runtime. These functions should be called only from within the runtime. Do not serialize the task in these methods. Do not create archives in these methods. Just build the task and submit. 

@CLAUDE.md Let's revert the changes just made, and assume TaskNode is passed in to NewTask. Update the task node to have: pid (process id), tid (thread id), major (32 bit), and minor (32 bit). pid should be acquired from ``HSHM_SYSTEM_INFO->pid_``, except don't dereference singleton directly. tid should be acquired using ``HSHM_THREAD_MODEL->GetTid``. When initializing the client code, create a thread-local storage block using hshm @ai-prompts/hshm-context.md storing a 32-bit counter. This counter is used to get the major number and is monotonically increasing. 
