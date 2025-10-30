import matplotlib.pyplot as plt

import pandas as pd

# Example data (replace 4.0 with your real sequential time)
data = [
    ["Sequential", 1, 1, 1, 4.0],
    ["Parallel (Threads)", 1, 4, 4, 1.04],
    ["Distributed (MPI)", 2, 1, 2, 1.92],
    ["Hybrid (MPI+Threads)", 2, 4, 8, 1.55],
]

df = pd.DataFrame(data, columns=["Version", "Nodes", "Threads", "Total Cores", "Time (s)"])

# Compute Speedup and Efficiency
df["Speedup"] = df["Time (s)"].iloc[0] / df["Time (s)"]
df["Efficiency (%)"] = (df["Speedup"] / df["Total Cores"]) * 100



plt.bar(df["Version"], df["Speedup"])
plt.title("Speedup Comparison")
plt.ylabel("Speedup (T_seq / T_parallel)")
plt.show()