import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load dataset
df = pd.read_csv("dataset.csv")

# Select input features
X = df[["Feature1", "Feature2"]]

# Create linkage matrix
linked = linkage(X, method="ward")

# Plot dendrogram
plt.figure(figsize=(8, 5))
dendrogram(linked)
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# Create Hierarchical Clustering model
model = AgglomerativeClustering(n_clusters=3)

# Fit model and get cluster labels
labels = model.fit_predict(X)

# Add cluster labels to dataset
df["Cluster"] = labels

# Display cluster labels
print("Cluster Labels:")
print(df["Cluster"])

# Visualization
plt.scatter(X["Feature1"], X["Feature2"], c=labels)
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("Hierarchical Clustering")
plt.show()

