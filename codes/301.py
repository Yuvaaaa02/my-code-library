import matplotlib.pyplot as plt

items = ["Laptop", "Phone", "Tablet"]
sales = [6200, 2100, 1550]

# Vertical bar chart
plt.figure(figsize=(7, 5))
plt.bar(items, sales)
plt.title("Vertical Bar Chart - Item-wise Sales")
plt.xlabel("Items")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Horizontal bar chart
plt.figure(figsize=(7, 5))
plt.barh(items, sales)
plt.title("Horizontal Bar Chart - Item-wise Sales")
plt.xlabel("Total Sales")
plt.ylabel("Items")
plt.tight_layout()
plt.show()
