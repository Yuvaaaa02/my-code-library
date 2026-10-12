# Experiment 208: Hierarchical / Agglomerative Clustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv(r"C:\Users\HP\Downloads\ML Lab Exp's\Exp-7,8\Mall_Customers.csv")
print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# a) Agglomerative clustering
hc = AgglomerativeClustering(n_clusters=2, linkage='ward')
df['Cluster'] = hc.fit_predict(X_scaled)
print("\\nCluster assignments:")
print(df[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head())

# b) Dendrogram
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Dendrogram for Customer Grouping')
plt.xlabel('Customer Groups')
plt.ylabel('Euclidean Distance')
plt.show()
