import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("GermanCredit(1).csv")

# Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)

# Select input and output
X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display predictions
print("Predicted values:")
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)