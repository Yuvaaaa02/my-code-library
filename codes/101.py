
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

print(df.isnull().sum())
print(df.describe())

sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()