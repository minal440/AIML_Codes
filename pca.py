# PCA.py
# -----------------------------------------------
# PRINCIPAL COMPONENT ANALYSIS (PCA) VISUALIZATION & VARIANCE EXPLAINED
# -----------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Step 1: Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA (all components first)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Explained variance ratio
explained_var = pca.explained_variance_ratio_
print("Explained Variance by each component:\n", explained_var)
print("Total Variance Captured:", np.sum(explained_var))

# Step 5: Scree plot (to see which components matter)
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(explained_var)+1), explained_var, marker='o', linestyle='--', color='b')
plt.title("Scree Plot (Explained Variance by Principal Components)")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.show()

# Step 6: Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

# Step 7: Plot PCA 2D projection
plt.figure(figsize=(8, 6))
for i, target in enumerate(np.unique(y)):
    plt.scatter(X_2d[y == target, 0], X_2d[y == target, 1], label=target_names[i])

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Visualization of Iris Dataset")
plt.legend()
plt.show()
