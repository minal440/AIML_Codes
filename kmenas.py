# Kmeans.py
# -----------------------------------------------
# K-MEANS CLUSTERING WITH ELBOW METHOD & VISUALIZATION
# -----------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Step 1: Generate sample data
X, y = make_blobs(n_samples=500, centers=5, cluster_std=0.7, random_state=42)

# Step 2: Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Determine optimal number of clusters using the Elbow Method
inertia = []  # Sum of squared distances to nearest cluster center
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Step 4: Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

# Step 5: Apply K-Means with optimal k (say 5 from elbow)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 6: Visualize Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', s=40)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label="Cluster Centers")
plt.title("K-Means Clustering Result")
plt.legend()
plt.show()

# Step 7: Display cluster centers and inertia
print("Cluster Centers:\n", centers)
print("\nFinal Inertia (WSS):", kmeans.inertia_)
