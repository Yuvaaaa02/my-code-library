import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Salary_Data.csv")

# Independent and dependent variables
X = df[["YearsExperience"]]
y = df["Salary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Visualization
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.show()

# Predict new value
years = [[5]]
print("Predicted salary:", model.predict(years)[0])
