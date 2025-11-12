# Fire and Forget tasks

A task should support being marked as FIRE_AND_FORGET. This should be a task flag in a bitfield. 

Fire and forget means that the task, upon its completion, will be deleted automatically by the runtime. The deletion of a task should be handled by its container, since the task will need to be typecasted. Containers expose a Del method for this purpose. Client code for these tasks do not typically have return values.

Build a unit test for testing fire & forget tasks. Add a new method to the MOD_NAME module to test this.