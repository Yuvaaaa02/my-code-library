import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

m = 0
c = 0

learning_rate = 0.01
iterations = 1000

n = len(X)

for i in range(iterations):
    y_pred = m * X + c

    dm = (-2/n) * sum(X * (y - y_pred))
    dc = (-2/n) * sum(y - y_pred)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

print("Slope (m):", m)
print("Intercept (c):", c)

y_pred = m * X + c

print("Predicted values:", y_pred)

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Gradient Descent")
plt.show()