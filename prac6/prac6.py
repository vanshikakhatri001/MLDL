# ==========================================
# K-Means and Hierarchical Clustering
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# -----------------------------
# 1) Load Dataset
# -----------------------------
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

# -----------------------------
# 2) Select Features
# -----------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Optional Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3) Elbow Method (K-Means)
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# -----------------------------
# 4) Apply K-Means
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y_kmeans)
plt.title("K-Means Clustering")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.show()
# Print cluster labels
print("\nK-Means Cluster Labels:")
print(y_kmeans)

# Print cluster centers
print("\nK-Means Cluster Centers (Scaled):")
print(kmeans.cluster_centers_)

# Print inertia
print("\nK-Means Inertia (WCSS):")
print(kmeans.inertia_)

# Silhouette Score
from sklearn.metrics import silhouette_score
kmeans_sil = silhouette_score(X_scaled, y_kmeans)
print("\nK-Means Silhouette Score:", kmeans_sil)

# -----------------------------
# 5) Hierarchical Clustering - Dendrogram
# -----------------------------
# -----------------------------
# Hierarchical Clustering
# -----------------------------

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Create model
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')

# Fit and predict
y_hc = hc.fit_predict(X_scaled)

# Now you can print
print("\nHierarchical Cluster Labels:")
print(y_hc)

# Silhouette score
hc_sil = silhouette_score(X_scaled, y_hc)
print("\nHierarchical Silhouette Score:", hc_sil)

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(8,5))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()


# -----------------------------
# 6) Apply Hierarchical Clustering
# -----------------------------
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hc = hc.fit_predict(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y_hc)
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.show()
