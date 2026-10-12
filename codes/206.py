# Experiment 206: Gradient Descent for Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-6\salary.csv")
print(df.head())

X = df['Years_of_Experience'].values.astype(float)
y = df['Salary_Lakhs'].values.astype(float)

# Standardize data
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()

b = 0.0
w = 0.0
alpha = 0.01
n = len(X)

def predict(X, b, w):
    return b + w * X

def get_cost(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def update_beta(X, y, y_hat, b, w, alpha):
    db = (-2 / n) * np.sum(y - y_hat)
    dw = (-2 / n) * np.sum(X * (y - y_hat))
    b = b - alpha * db
    w = w - alpha * dw
    return b, w

# a) One gradient descent update
y_hat = predict(X, b, w)
b, w = update_beta(X, y, y_hat, b, w, alpha)
print("Updated Bias:", b)
print("Updated Weight:", w)
print("Cost:", get_cost(y, predict(X, b, w)))

# b) Train for multiple iterations and plot cost
b = 0.0
w = 0.0
alpha = 0.01
iterations = 1000
cost = []

for i in range(iterations):
    y_hat = b + w * X
    db = (-2 / n) * np.sum(y - y_hat)
    dw = (-2 / n) * np.sum(X * (y - y_hat))
    b -= alpha * db
    w -= alpha * dw
    mse = get_cost(y, b + w * X)
    cost.append(mse)

print("Final Bias:", b)
print("Final Weight:", w)
print("Final Cost:", cost[-1])

plt.plot(range(1, iterations + 1), cost)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.title("Cost Function vs Iterations")
plt.show()
