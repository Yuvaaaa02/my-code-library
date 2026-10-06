import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")

# Display first 5 rows
print(df.head())

# Display information
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing numerical values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Check again
print("\nAfter cleaning:")
print(df.isnull().sum())

# Statistical summary
print(df.describe())

# Visualization
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()
