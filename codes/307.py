import matplotlib.pyplot as plt

advertising_cost = [10, 20, 30, 40, 50, 60, 70, 80]
sales = [15, 25, 35, 30, 50, 60, 65, 85]

plt.figure(figsize=(8, 5))
plt.scatter(advertising_cost, sales)
plt.title("Scatter Plot: Advertising Cost vs Sales")
plt.xlabel("Advertising Cost")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
