import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.DataFrame({
    "emp_name": ["Ravi", "Sita", "Kiran", "Anu", "Mohan", "Latha"],
    "salary": [50000, 60000, 45000, 70000, 40000, 55000],
    "department": ["Sales", "IT", "HR", "IT", "Sales", "HR"]
})

# Single-variable density plot
plt.figure(figsize=(8, 5))
sns.kdeplot(data=data, x="salary", fill=True)
plt.title("Density Plot of Employee Salaries")
plt.xlabel("Salary")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# Multiple-variable density plot
plt.figure(figsize=(8, 5))
sns.kdeplot(data=data, x="salary", hue="department", fill=False)
plt.title("Salary Density by Department")
plt.xlabel("Salary")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
