import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("adult.csv")

# Convert categorical columns into numbers
df = pd.get_dummies(df)

# Input and output
X = df.drop("Target_ >50K", axis=1)
y = df["Target_ >50K"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Classification Tree
model = DecisionTreeClassifier()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Predicted values:")
print(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))