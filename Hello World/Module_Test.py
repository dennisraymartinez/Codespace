import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Create synthetic data using scikit-learn
# make_blobs creates clusters of points that we can analyze
print("Generating synthetic data clusters...")
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# Perform K-means clustering
print("\nPerforming K-means clustering...")
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Calculate cluster centers
centers = kmeans.cluster_centers_

# Create a statistical summary of each cluster
print("\nAnalyzing cluster statistics...")
cluster_stats = df.groupby('Cluster').agg({
    'Feature1': ['mean', 'std', 'count'],
    'Feature2': ['mean', 'std']
}).round(2)

print("\nCluster Statistics:")
print(cluster_stats)

# Create a visualization using seaborn and matplotlib
plt.figure(figsize=(12, 6))

# First subplot: Scatter plot with cluster assignments
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Cluster', palette='deep')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('Cluster Assignments with Centroids')
plt.legend(title='Cluster')

# Second subplot: Distribution of Feature1 by cluster
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Cluster', y='Feature1', palette='deep')
plt.title('Distribution of Feature1 by Cluster')

plt.tight_layout()
print("\nDisplaying visualization...")
plt.show()

# Calculate some additional statistics using NumPy
print("\nOverall Dataset Statistics:")
print(f"Total number of points: {len(df)}")
print(f"Global mean of Feature1: {np.mean(df['Feature1']):.2f}")
print(f"Global standard deviation of Feature1: {np.std(df['Feature1']):.2f}")