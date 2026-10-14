import matplotlib.pyplot as plt

# Sales amounts from sample transactions
sales_amount = [1000, 500, 1500, 750, 300, 2000, 700, 800, 1700, 600]

# Single-variable histogram
plt.figure(figsize=(7, 5))
plt.hist(sales_amount, bins=5, edgecolor="black")
plt.title("Histogram of Sales Amount")
plt.xlabel("Sales Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Multiple-distribution histogram
laptop_sales = [1000, 1500, 2000, 1700]
phone_sales = [500, 300, 700, 600]
tablet_sales = [750, 800]

plt.figure(figsize=(7, 5))
plt.hist(laptop_sales, bins=5, alpha=0.5, label="Laptop", edgecolor="black")
plt.hist(phone_sales, bins=5, alpha=0.5, label="Phone", edgecolor="black")
plt.hist(tablet_sales, bins=5, alpha=0.5, label="Tablet", edgecolor="black")
plt.title("Histogram of Sales Distribution by Item")
plt.xlabel("Sales Amount")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
