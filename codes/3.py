import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("house_price.csv")

# Select multiple features and target
X = df[["Area", "Bedrooms", "BuiltUp", "Age"]]
y = df["Price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict prices
y_pred = model.predict(X_test)

# Display predictions
print("Predicted Prices:")
print(y_pred)

# Display coefficients and intercept
print("\nCoefficients:")
print(model.coef_)
print("\nIntercept:")
print(model.intercept_)