import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("MBA Salary.csv")

# Display first 5 rows
print(df.head())

# Select input and output
X = df[["Percentage in Grade 10"]]
y = df["Salary"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Display predicted values
print("\nPredicted Salaries:")
print(y_pred)

# Display regression equation
print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("\nRegression Equation:")
print("Salary =", model.intercept_, "+", model.coef_[0], "* Percentage")

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Percentage in Grade 10")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.show()