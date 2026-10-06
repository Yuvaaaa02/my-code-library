import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Salary_Data.csv")

# Suppose the dataset contains:
# YearsExperience, Age, EducationLevel, Salary
X = df[["YearsExperience", "Age", "EducationLevel"]]
y = df["Salary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Predict a new employee
new_employee = [[5, 25, 3]]
prediction = model.predict(new_employee)
print("Predicted Salary:", prediction[0])
