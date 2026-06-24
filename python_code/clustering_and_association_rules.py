import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kmedoids
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from mlxtend.frequent_patterns import apriori, association_rules

# ==============================================================================
# 1. Load data and scale features
# ==============================================================================
df = pd.read_csv('../data/processed/customer_baseline.csv')

# Pick features for clustering
feature_cols = ['Frequency', 'TotalQuantity', 'TotalSpent', 'AvgPrice',
                'UniqueProducts', 'SpentPerTransaction']
X = df[feature_cols]

# Scale features (clustering uses distances so all features should have similar range)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 2. K-Means - find best k with Elbow and Silhouette
# ==============================================================================
# Trying different number of clusters and see which works best
k_values = range(2, 11)
inertias = []
silhouettes = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Plot both methods side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(k_values, inertias, 'bo-')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Inertia')

axes[1].plot(k_values, silhouettes, 'ro-')
axes[1].set_title('Silhouette Scores')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Score')

plt.tight_layout()
plt.savefig('../data/processed/kmeans_elbow_silhouette.png')
plt.close()

# Silhouette suggests k=2 but it gives only 2 groups which is not very useful
# Elbow shows that k=4 is a good compromise = more meaningful segmentation
optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

print("Selected k =", optimal_k)
print("\nK-Means cluster profiles:")
print(df.groupby('KMeans_Cluster')[feature_cols].mean().round(2))

# ==============================================================================
# 3. K-Medoids - similar to K-Means but uses real customers as cluster centres
# ==============================================================================

# K-Medoids needs distance matrix between all customers
dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

# Run K-Medoids with same number of clusters as K-Means
result = kmedoids.fasterpam(dist_matrix, medoids=optimal_k, random_state=42)
df['KMedoids_Cluster'] = result.labels

print("\nK-Medoids cluster profiles:")
print(df.groupby('KMedoids_Cluster')[feature_cols].mean().round(2))

# ==============================================================================
# 4. DBSCAN - density-based clustering, picks eps from k-distance plot
# ==============================================================================
# First we need to find a good eps value
# Plot the distance to k-th nearest neighbour for each point, sorted
k_neighbours = 12

nn = NearestNeighbors(n_neighbors=k_neighbours)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)

# get distance to 12th neighbour and sort it
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(10, 5))
plt.plot(k_distances)
plt.axhline(y=0.8, color='r', linestyle='--', label='Selected eps = 0.8')
plt.title('K-Distance Plot for DBSCAN eps Selection')
plt.xlabel('Points (sorted by distance)')
plt.ylabel('Distance to ' + str(k_neighbours) + 'th neighbour')
plt.legend()
plt.tight_layout()
plt.savefig('../data/processed/dbscan_kdistance.png')
plt.close()

# The elbow in the plot is around 0.8, so it's a reason to use this as eps
dbscan_eps = 0.8
dbscan = DBSCAN(eps=dbscan_eps, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)


# Count clusters and noise points (label -1 = noise)
n_noise = (df['DBSCAN_Cluster'] == -1).sum()

# Count unique cluster labels but don't count -1 (noise is not a cluster)
unique_labels = set(df['DBSCAN_Cluster'])
if -1 in unique_labels:
    n_clusters_db = len(unique_labels) - 1
else:
    n_clusters_db = len(unique_labels)

print("\nDBSCAN results:")
print("Clusters found:", n_clusters_db)
print("Noise points:", n_noise, "(", round(n_noise / len(df) * 100, 1), "%)")

if n_clusters_db > 0:
    non_noise = df[df['DBSCAN_Cluster'] != -1]
    print(non_noise.groupby('DBSCAN_Cluster')[feature_cols].mean().round(2))

# ==============================================================================
# 5. Hierarchical clustering with dendrogram
# ==============================================================================
# Dendrogram on a sample because full dataset would be unreadable
np.random.seed(42)
sample_size = min(500, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)

# Build linkage with Ward method and plot dendrogram
Z_sample = linkage(X_scaled[sample_idx], method='ward')

plt.figure(figsize=(14, 6))
dendrogram(Z_sample, truncate_mode='level', p=5, color_threshold=15)
plt.title('Hierarchical Clustering Dendrogram (sample = 500)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('../data/processed/dendrogram.png')
plt.close()

# Apply hierarchical clustering on the full dataset and cut tree to get 4 clusters
Z_full = linkage(X_scaled, method='ward')
df['Hier_Cluster'] = fcluster(Z_full, t=optimal_k, criterion='maxclust')

print("\nHierarchical cluster profiles:")
print(df.groupby('Hier_Cluster')[feature_cols].mean().round(2))

# ==============================================================================
# 6. Compare clustering methods visually
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot K-Means clusters
sc1 = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 2],
                       c=df['KMeans_Cluster'], cmap='viridis', alpha=0.5, s=10)
axes[0].set_title('K-Means')
axes[0].set_xlabel('Frequency (scaled)')
axes[0].set_ylabel('TotalSpent (scaled)')
plt.colorbar(sc1, ax=axes[0])

# Plot DBSCAN clusters
sc2 = axes[1].scatter(X_scaled[:, 0], X_scaled[:, 2],
                       c=df['DBSCAN_Cluster'], cmap='viridis', alpha=0.5, s=10)
axes[1].set_title('DBSCAN')
axes[1].set_xlabel('Frequency (scaled)')
axes[1].set_ylabel('TotalSpent (scaled)')
plt.colorbar(sc2, ax=axes[1])

# Plot Hierarchical clusters
sc3 = axes[2].scatter(X_scaled[:, 0], X_scaled[:, 2],
                       c=df['Hier_Cluster'], cmap='viridis', alpha=0.5, s=10)
axes[2].set_title('Hierarchical')
axes[2].set_xlabel('Frequency (scaled)')
axes[2].set_ylabel('TotalSpent (scaled)')
plt.colorbar(sc3, ax=axes[2])

plt.tight_layout()
plt.savefig('../data/processed/cluster_comparison.png')
plt.close()

# ==============================================================================
# 7. Association Rules with Apriori
# ==============================================================================
# Find which products are often bought together
print("\nAssociation rules analysis:")

txn = pd.read_csv('../data/processed/clean_transactions.csv')

# Use only top 50 most popular products (otherwise too many combinations)
top_products = txn['StockCode'].value_counts().head(50).index
txn_top = txn[txn['StockCode'].isin(top_products)]

# Build basket matrix
# rows => invoices, columns => products, values => 0/1

basket = txn_top.groupby(['Invoice', 'StockCode'])['Quantity'].sum()
basket = basket.unstack(fill_value=0)
basket = (basket > 0).astype(int)

# Apriori finds frequent itemsets (products bought together often)
freq_items = apriori(basket, min_support=0.03, use_colnames=True)

# Generate rules from frequent itemsets
rules = association_rules(freq_items, metric='lift', min_threshold=1.5)
rules = rules.sort_values('lift', ascending=False)

print("Frequent itemsets found:", len(freq_items))
print("Rules generated:", len(rules))
print("\nTop 10 rules by Lift:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string())

# ==============================================================================
# 8. Save results
# ==============================================================================
df.to_csv('../data/processed/customer_clustered.csv', index=False)
print("\n Dataset saved.")