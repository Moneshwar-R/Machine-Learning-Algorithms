# =========================================
# CLUSTERING: K-MEANS + HIERARCHICAL
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# =========================================
# 1. CREATE DATA (Unsupervised → no labels)
# =========================================

X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
print(df.head())

# =========================================
# 2. FEATURE SCALING (IMPORTANT)
# =========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================================
# 3. ELBOW METHOD (Inertia)
# =========================================

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# =========================================
# 4. SILHOUETTE ANALYSIS
# =========================================

sil_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)

plt.figure(figsize=(6,4))
plt.plot(range(2,11), sil_scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.grid(True)
plt.show()

# =========================================
# 5. FINAL K-MEANS MODEL
# =========================================

kmeans = KMeans(n_clusters=4, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# =========================================
# 6. VISUALIZE K-MEANS CLUSTERS
# =========================================

plt.figure(figsize=(6,5))
plt.scatter(
    df["Feature1"],
    df["Feature2"],
    c=df["KMeans_Cluster"],
    cmap="viridis"
)
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("K-Means Clustering")
plt.grid(True)
plt.show()

# =========================================
# 7. PREDICT CLUSTER FOR NEW DATA
# =========================================

new_point = np.array([[1.5, -0.8]])
new_point_scaled = scaler.transform(new_point)
predicted_cluster = kmeans.predict(new_point_scaled)

print("New data point belongs to cluster:", predicted_cluster[0])

# =========================================
# 8. HIERARCHICAL CLUSTERING
# =========================================

linked = linkage(X_scaled, method="ward")

# =========================================
# 9. DENDROGRAM
# =========================================

plt.figure(figsize=(8,5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# =========================================
# 10. ASSIGN HIERARCHICAL CLUSTERS
# =========================================

df["Hierarchical_Cluster"] = fcluster(
    linked,
    t=4,
    criterion="maxclust"
)

print(df.head())
