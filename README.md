# GPU Graph Memory Optimization Lab

A CUDA-based benchmark framework that studies how GPU memory allocation strategies impact graph analytics workloads.

The project implements PageRank on real-world graph datasets and compares:

1. cudaMalloc
2. cudaMallocAsync
3. Custom GPU Memory Pool
4. Unified Memory


## Motivation

Modern GPU workloads are often limited not only by computation, but by memory management.

Graph analytics workloads are especially challenging because:

- graph structures are irregular
- memory access patterns are unpredictable
- temporary buffers are frequently created
- GPU memory allocation overhead can reduce utilization


This project investigates:

> How does GPU memory allocation strategy affect graph analytics performance?


---

# Workload

The benchmark uses PageRank.

PageRank repeatedly updates node importance scores by distributing rank values across graph edges.

The computation:
