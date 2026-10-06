import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

linked = linkage(X, method="ward")

plt.figure(figsize=(10, 6))
dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

model = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

clusters = model.fit_predict(X)

df["Cluster"] = clusters

print(df.head())