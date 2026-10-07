import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Advertising.csv")

# Input and output
X = df["TV"].values
y = df["Sales"].values

# Initial values
m = 0
b = 0

# Learning rate
lr = 0.0001

# Gradient Descent
for i in range(1000):
    # Prediction
    y_pred = m * X + b

    # Error
    error = y_pred - y

    # Gradients
    dm = (error * X).mean()
    db = error.mean()

    # Update parameters
    m = m - lr * dm
    b = b - lr * db

# Print final values
print("Slope:", m)
print("Intercept:", b)
print("Gradient of m:", dm)
print("Gradient of b:", db)

# Final prediction
y_pred = m * X + b

print("Predicted Sales:")
print(y_pred)

# Graph
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("Gradient Descent - TV vs Sales")
plt.show()
