import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv(
    "results.csv"
)


plt.figure(
    figsize=(8,5)
)


plt.bar(
    df["allocator"],
    df["runtime_ms"]
)


plt.ylabel(
    "Runtime (ms)"
)


plt.title(
    "CUDA Graph Memory Allocation Benchmark"
)


plt.xticks(
    rotation=30
)


plt.tight_layout()


plt.savefig(
    "benchmark.png"
)
