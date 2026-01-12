# =========================================
# DBSCAN CLUSTERING - COMPLETE EXAMPLE
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# =========================================
# 1. CREATE DATA
# =========================================

X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.1,
    random_state=42
)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])

# =========================================
# 2. FEATURE SCALING (MANDATORY)
# =========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================================
# 3. DBSCAN MODEL
# =========================================

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

# =========================================
# 4. IDENTIFY NOISE
# =========================================

noise_count = (df["DBSCAN_Cluster"] == -1).sum()
print("Noise points detected:", noise_count)

# =========================================
# 5. VISUALIZE DBSCAN RESULT
# =========================================

plt.figure(figsize=(6,5))
plt.scatter(
    df["Feature1"],
    df["Feature2"],
    c=df["DBSCAN_Cluster"],
    cmap="tab10"
)
plt.title("DBSCAN Clustering (-1 = Noise)")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.grid(True)
plt.show()

# =========================================
# 6. SILHOUETTE SCORE (REMOVE NOISE)
# =========================================

filtered_df = df[df["DBSCAN_Cluster"] != -1]

if filtered_df["DBSCAN_Cluster"].nunique() > 1:
    sil_score = silhouette_score(
        filtered_df[["Feature1", "Feature2"]],
        filtered_df["DBSCAN_Cluster"]
    )
    print("Silhouette Score (DBSCAN):", sil_score)
else:
    print("Silhouette score not defined (only one cluster)")

# =========================================
# 7. OPTIONAL: COMPARE WITH K-MEANS
# =========================================

kmeans = KMeans(n_clusters=4, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(
    df["Feature1"],
    df["Feature2"],
    c=df["KMeans_Cluster"],
    cmap="viridis"
)
plt.title("K-Means Clustering")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.grid(True)
plt.show()
