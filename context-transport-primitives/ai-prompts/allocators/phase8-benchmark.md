@CLAUDE.md

# ZeroMQ benchmark

Let's create a benchmark for lightbeam. Client and server.

The benchmark takes as input the message size, number of threads, and time.

Spawn a server thread that creates the lightbeam server with Zmq type.
It should use IPC for the communication, not tcp.

Spawn client threads. 
Each client should 