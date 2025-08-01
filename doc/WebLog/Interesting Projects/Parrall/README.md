## Core concepts
1. Distributed Systems Basics
    1. What is a node, cluster, master/slave vs peer-to-peer, task scheduling
    2. Concepts like latency, throughput, fault tolerance, and synchronization

2. Networking Basics
        TCP/IP, sockets, message passing
        Protocols like HTTP, gRPC, or MPI

3. Parallelism vs Distribution
    1. Understand how multithreading/multiprocessing (e.g. OpenMP, multiprocessing in Python) differs from distributed systems
    2. Learn how tasks are coordinated across machines



### Topics to Explore

- MPI basics: mpirun, MPI_Send, MPI_Recv, collective ops
- Sockets and networking protocols
- Load balancing and distributed job scheduling
- CUDA-aware MPI or NCCL
- Fault tolerance & resilience (optional but good for production)


### Python
Python Tools

1. MPI for Python (mpi4py)

   Most mature option for distributed parallelism in scientific computing; wraps MPI (Message Passing Interface)

### C/C++ Tools

1. MPI (e.g. OpenMPI or MPICH)
   
   Industry standard for C/C++ distributed computing

2. ZeroMQ or nanomsg
   
   For more flexible messaging between C/C++ apps

3. gRPC
   
   Modern, performant way to do cross-language RPC (great for C++ ↔ Python communication)


## Recommended First Steps

Learn mpi4py and run an MPI-based Python script across two machines on your network.
Implement basic socket-based message passing in Python and C.
Later, replace CPU computation with CUDA kernels and integrate NCCL or MPI for GPU-to-GPU communication.