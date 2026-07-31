# GPU Graph Memory Allocation Benchmark Findings

<div align="center">

<table>
<tr>
<td align="center">
<b>Benchmark Execution</b><br>
<img src="https://github.com/user-attachments/assets/71a44b9f-278b-4dfc-9b9b-acc25f2a05a2" width="500"/>
</td>

<td align="center">
<b>Runtime Results</b><br>
<img src="https://github.com/user-attachments/assets/95445bb3-ac94-4ad2-821e-831057d17861" width="500"/>
</td>

<td align="center">
<b>GPU Memory Analysis</b><br>
<img src="https://github.com/user-attachments/assets/330ef268-9e8e-42a3-a66a-991fb0147484" width="500"/>
</td>
</tr>
</table>

</div>

## Overview

This benchmark evaluates how different GPU memory allocation strategies affect graph analytics performance.

The workload uses PageRank on a real-world citation network while keeping the computation identical across experiments.

## Workload Configuration

- Algorithm: PageRank
- Dataset: SNAP cit-HepTh citation graph
- Nodes: 27,770
- Edges: 352,807
- Iterations: 100
- GPU: NVIDIA GeForce RTX 3090
- CUDA Version: 13.0.88

The benchmark compares four GPU memory allocation strategies:

- `cudaMalloc`
- `cudaMallocAsync`
- Custom GPU Memory Pool
- Unified Memory

All experiments use the same:

- Graph representation (CSR format)
- PageRank CUDA kernel
- Number of iterations
- GPU device

Only the memory allocation strategy changes.

---

# Benchmark Results

| Allocator | GPU Memory After Allocation (MB) | Total Runtime (ms) | Average Iteration (ms) |
|-----------|----------------------------------|--------------------|------------------------|
| cudaMalloc | 267.125 | 5.18144 | 0.0518144 |
| cudaMallocAsync | 297.125 | 4.83533 | 0.0483533 |
| memory_pool | 521.125 | 4.99200 | 0.0499200 |
| unified_memory | 265.125 | 5.49990 | 0.0549990 |

---

# Key Findings

## 1. cudaMallocAsync achieved the best performance

`cudaMallocAsync` produced the lowest runtime:

```
Total Runtime: 4.83533 ms
Average Iteration: 0.0483533 ms
```

Compared with traditional `cudaMalloc`:

```
cudaMalloc:
5.18144 ms

cudaMallocAsync:
4.83533 ms
```

`cudaMallocAsync` improved runtime by approximately:

```
6.7%
```

This indicates that stream-ordered asynchronous allocation can reduce memory allocation overhead and improve execution efficiency for GPU workloads with repeated allocation patterns.

---

# 2. Memory Pool improved performance but increased memory usage

The custom memory pool achieved the second-best runtime:

```
Total Runtime:
4.992 ms
```

However, it consumed significantly more GPU memory:

```
GPU Memory:
521.125 MB
```

The increased memory usage comes from preallocating:

```
256 MB GPU memory pool
```

This demonstrates the common GPU systems tradeoff:

> Preallocated memory pools can improve performance consistency but require additional memory capacity.

Memory pools are often beneficial in production GPU systems where predictable allocation behavior is more important than minimizing memory footprint.

---

# 3. Unified Memory introduced additional overhead

Unified Memory achieved:

```
Total Runtime:
5.4999 ms
```

making it the slowest configuration.

Graph workloads often contain irregular memory access patterns:

```
node -> neighbor -> random memory location
```

Unlike dense matrix workloads, graph traversal creates unpredictable memory access behavior. Unified Memory may introduce additional overhead due to page migration between CPU and GPU memory.

---

# Performance Ranking

Based on total runtime:

| Rank | Allocator | Runtime |
|------|-----------|---------|
| 1 | cudaMallocAsync | 4.83533 ms |
| 2 | memory_pool | 4.99200 ms |
| 3 | cudaMalloc | 5.18144 ms |
| 4 | unified_memory | 5.49990 ms |

---

# Conclusion

For this PageRank graph analytics workload, `cudaMallocAsync` provided the best overall performance.

The benchmark demonstrates that GPU memory allocation strategy directly impacts workload performance.

Key conclusions:

- `cudaMallocAsync` provided the lowest runtime with moderate memory overhead.
- Memory pools improved performance consistency but required larger memory reservations.
- Unified Memory simplified programming but introduced overhead for irregular graph workloads.
- Traditional `cudaMalloc` remains functional but can introduce additional allocation overhead.

These results demonstrate how memory management decisions influence GPU application performance, especially for memory-bound graph analytics workloads.

---

# Reproducibility

Benchmark artifacts:

```
benchmarks/
│
├── results.csv
├── benchmark_log.txt
├── FINDINGS.md
├── plot.py
└── benchmark.png
```

Contents:

- `benchmark_log.txt`  
  Raw benchmark execution output.

- `results.csv`  
  Structured benchmark measurements.

- `FINDINGS.md`  
  Analysis and conclusions.

- `plot.py`  
  Visualization script.

- `benchmark.png`  
  Generated runtime comparison chart.

The benchmark can be reproduced by running the CUDA benchmark executable and regenerating the results visualization.
