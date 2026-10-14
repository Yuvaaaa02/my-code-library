import numpy as np
import matplotlib.pyplot as plt

dataset1 = np.array([40, 45, 50, 55, 60, 65, 70])
dataset2 = np.array([42, 48, 52, 57, 63, 68, 75])

quantiles1 = np.sort(dataset1)
quantiles2 = np.sort(dataset2)

plt.figure(figsize=(7, 5))
plt.scatter(quantiles1, quantiles2, label="Quantiles")

min_val = min(quantiles1.min(), quantiles2.min())
max_val = max(quantiles1.max(), quantiles2.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Reference line")

plt.title("Q-Q Plot: Dataset 1 vs Dataset 2")
plt.xlabel("Dataset 1 Quantiles")
plt.ylabel("Dataset 2 Quantiles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
