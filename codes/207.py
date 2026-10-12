# Experiment 207: K-Means Clustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-7,8\Mall_Customers.csv")
print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("K-Means clustering completed.")

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering (k=2)')
plt.show()

for i in range(2):
    print(f"\\nCustomers in Cluster {i}:")
    print(df[df['Cluster'] == i].to_string(index=False))
