import matplotlib.pyplot as plt
import numpy as np

items = ["Laptop", "Phone", "Tablet"]
store1 = np.array([2700, 1100, 750])
store2 = np.array([1500, 300, 800])
store3 = np.array([2000, 700, 0])

x = np.arange(len(items))
width = 0.25

# Grouped bar chart
plt.figure(figsize=(8, 5))
plt.bar(x - width, store1, width, label="Store1")
plt.bar(x, store2, width, label="Store2")
plt.bar(x + width, store3, width, label="Store3")
plt.xlabel("Items")
plt.ylabel("Sales Amount")
plt.title("Grouped Bar Chart - Item-wise Sales by Store")
plt.xticks(x, items)
plt.legend()
plt.tight_layout()
plt.show()

# Stacked bar chart
plt.figure(figsize=(8, 5))
plt.bar(items, store1, label="Store1")
plt.bar(items, store2, bottom=store1, label="Store2")
plt.bar(items, store3, bottom=store1 + store2, label="Store3")
plt.xlabel("Items")
plt.ylabel("Sales Amount")
plt.title("Stacked Bar Chart - Item-wise Sales by Store")
plt.legend()
plt.tight_layout()
plt.show()
