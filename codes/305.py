import numpy as np
import matplotlib.pyplot as plt

salary = np.array([50000, 60000, 45000, 70000, 40000, 55000])

x = np.sort(salary)
y = np.arange(1, len(x) + 1) / len(x)

plt.figure(figsize=(8, 5))
plt.step(x, y, where="post", label="ECDF")
plt.scatter(x, y)
plt.title("ECDF Plot of Employee Salaries")
plt.xlabel("Salary")
plt.ylabel("Cumulative Probability")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
