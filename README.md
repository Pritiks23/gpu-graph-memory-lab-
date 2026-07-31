# GPU Graph Memory Optimization Lab
<img width="1024" height="577" alt="image" src="https://github.com/user-attachments/assets/1a3c7984-8b67-405f-9686-454a9fe521ae" />


A CUDA-based benchmark framework that studies how GPU memory allocation strategies impact graph analytics workloads.

The project implements PageRank on real-world graph datasets and compares:

1. cudaMalloc
2. cudaMallocAsync
3. Custom GPU Memory Pool
4. Unified Memory


A CUDA-based benchmark framework that studies GPU memory allocation strategies for large-scale graph analytics workloads using PageRank and graph traversal algorithms.

Real-world motivation
Companies process huge graphs:

Examples:

Social networks → user connections
Fraud detection → transaction networks
Recommendation systems → user-item graphs
Scientific computing → mesh graphs
Cybersecurity → network graphs
A graph workload has a unique GPU challenge:

memory access dominates computation.

Unlike matrix multiplication:

A × B
where memory access is predictable:

Graphs look like:

Node 0
 |
 +---- Node 45
 |
 +---- Node 9203
 |
 +---- Node 17
The GPU constantly jumps around memory. 

Core idea
All four versions run:

SAME GRAPH

SAME PAGE RANK ALGORITHM

SAME GPU KERNEL


ONLY MEMORY ALLOCATION CHANGES
This is important.

Your experiment is controlled.





This project investigates:

> How does GPU memory allocation strategy affect graph analytics performance?


---

# Workload

The benchmark uses PageRank.

PageRank repeatedly updates node importance scores by distributing rank values across graph edges.





What the finished project does
You will run:

./build/graph_benchmark --allocator cudaMalloc
then:

./build/graph_benchmark --allocator cudaMallocAsync
then:

./build/graph_benchmark --allocator memory_pool
then:

./build/graph_benchmark --allocator unified_memory
All four will run:

same graph
same PageRank kernel
same number of iterations
same GPU

Only memory management changes.








Benchmark Metrics

The framework measures:

Runtime
total execution time
average iteration time
Memory
initial GPU memory
peak memory usage
Allocation
allocation count
allocated bytes
allocation overhead
GPU Metrics

