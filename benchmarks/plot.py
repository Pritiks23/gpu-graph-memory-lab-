import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(
    "results.csv"
)


plt.figure(figsize=(8,5))


plt.bar(
    data["allocator"],
    data["total_time_ms"]
)


plt.xlabel(
    "Memory Strategy"
)


plt.ylabel(
    "Runtime (ms)"
)


plt.title(
    "GPU Graph Memory Optimization Benchmark"
)


plt.xticks(
    rotation=30
)


plt.tight_layout()


plt.savefig(
    "runtime_comparison.png"
)
