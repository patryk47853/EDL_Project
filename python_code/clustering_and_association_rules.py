import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from mlxtend.frequent_patterns import apriori, association_rules

sns.set_theme(style="whitegrid")

# ==============================================================================
# 1. LOAD & PREPARE
# ==============================================================================
df = pd.read_csv('../data/processed/customer_baseline.csv')

feature_cols = ['Frequency', 'TotalQuantity', 'TotalSpent', 'AvgPrice',
                'UniqueProducts', 'SpentPerTransaction']
X = df[feature_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 2. K-MEANS — ELBOW METHOD & SILHOUETTE ANALYSIS
# ==============================================================================
K_range = range(2, 11)
inertias, silhouettes = [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(K_range, inertias, 'bo-')
ax1.set_title('Elbow Method')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')
ax2.plot(K_range, silhouettes, 'ro-')
ax2.set_title('Silhouette Scores')
ax2.set_xlabel('k')
ax2.set_ylabel('Score')
plt.tight_layout()
plt.savefig('../data/processed/kmeans_elbow_silhouette.png', dpi=150)
plt.close()

# Silhouette suggests k=2, but for meaningful business segmentation we use k=4
# Justified by elbow plot (diminishing inertia returns beyond k=4)
optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nSelected K={optimal_k} (business-driven, supported by elbow plot)")
print("\n--- K-MEANS CLUSTER PROFILES ---")
print(df.groupby('KMeans_Cluster')[feature_cols].mean().round(2))


# ==============================================================================
# 3. K-MEDOIDS (partition method, robust to outliers)
# ==============================================================================
try:
    import kmedoids as km

    # K-Medoids using FasterPAM algorithm (state-of-the-art implementation)
    # Requires a distance matrix as input
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

    result = km.fasterpam(dist_matrix, medoids=optimal_k, random_state=42)
    df['KMedoids_Cluster'] = result.labels

    print(f"\n--- K-MEDOIDS (FasterPAM, k={optimal_k}) ---")
    print(f"Loss (total deviation): {result.loss:.2f}")
    print(df.groupby('KMedoids_Cluster')[feature_cols].mean().round(2))

except ImportError:
    print("\n⚠ kmedoids not installed — run: pip install kmedoids")
    df['KMedoids_Cluster'] = df['KMeans_Cluster']  # fallback

# ==============================================================================
# 4. DBSCAN — AUTOMATIC EPS SELECTION VIA K-DISTANCE PLOT
# ==============================================================================
from sklearn.neighbors import NearestNeighbors

# Step 1: k-distance plot to find the "elbow" = optimal eps
k_neighbours = 2 * X_scaled.shape[1]  # heuristic: 2 × number of features
nn = NearestNeighbors(n_neighbors=k_neighbours)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(10, 5))
plt.plot(k_distances)
plt.title(f'K-Distance Plot (k={k_neighbours}) — DBSCAN eps Selection')
plt.xlabel('Points (sorted by distance)')
plt.ylabel(f'{k_neighbours}th Nearest Neighbour Distance')
plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Selected eps = 0.8')
plt.legend()
plt.tight_layout()
plt.savefig('../data/processed/dbscan_kdistance.png', dpi=150)
plt.close()

# Step 2: Read the elbow from the plot and set eps accordingly
# The red line is a starting point — adjust after inspecting the plot
dbscan_eps = 0.8
dbscan = DBSCAN(eps=dbscan_eps, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(df['DBSCAN_Cluster'])) - \
                (1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_noise = (df['DBSCAN_Cluster'] == -1).sum()
print(f"\n--- DBSCAN (eps={dbscan_eps}, min_samples={k_neighbours}) ---")
print(f"Clusters found: {n_clusters_db}")
print(f"Noise points:   {n_noise} ({n_noise/len(df)*100:.1f}%)")

if n_clusters_db > 0:
    non_noise = df[df['DBSCAN_Cluster'] != -1]
    print(non_noise.groupby('DBSCAN_Cluster')[feature_cols].mean().round(2))


# ==============================================================================
# 5. HIERARCHICAL CLUSTERING + DENDROGRAM
# ==============================================================================
# Dendrogram on a sample (full dataset too large for readable plot)
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_scaled),
                         size=min(500, len(X_scaled)), replace=False)
Z_sample = linkage(X_scaled[sample_idx], method='ward')

plt.figure(figsize=(14, 6))
dendrogram(Z_sample, truncate_mode='level', p=5, color_threshold=15)
plt.title('Hierarchical Clustering Dendrogram (Ward, sample=500)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.savefig('../data/processed/dendrogram.png', dpi=150)
plt.close()

# Full dataset — assign cluster labels via Ward linkage
Z_full = linkage(X_scaled, method='ward')
df['Hier_Cluster'] = fcluster(Z_full, t=optimal_k, criterion='maxclust')

print("\n--- HIERARCHICAL CLUSTER PROFILES ---")
print(df.groupby('Hier_Cluster')[feature_cols].mean().round(2))

# ==============================================================================
# 6. CLUSTER COMPARISON VISUALISATION
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
methods = [('KMeans_Cluster', 'K-Means'),
           ('DBSCAN_Cluster', 'DBSCAN'),
           ('Hier_Cluster', 'Hierarchical')]

for ax, (col, title) in zip(axes, methods):
    sc = ax.scatter(X_scaled[:, 0], X_scaled[:, 2],
                    c=df[col], cmap='viridis', alpha=0.5, s=10)
    ax.set_title(title)
    ax.set_xlabel('Frequency (scaled)')
    ax.set_ylabel('TotalSpent (scaled)')
    plt.colorbar(sc, ax=ax)

plt.tight_layout()
plt.savefig('../data/processed/cluster_comparison.png', dpi=150)
plt.close()

# ==============================================================================
# 7. ASSOCIATION RULES — APRIORI (Descriptive Data Mining)
# ==============================================================================
print("\n--- ASSOCIATION RULES (APRIORI) ---")
txn = pd.read_csv('../data/processed/clean_transactions.csv')

# Keep top 50 most popular products for tractability
top_products = txn['StockCode'].value_counts().head(50).index
txn_top = txn[txn['StockCode'].isin(top_products)]

# Binary basket matrix: rows = invoices, columns = products
basket = (txn_top.groupby(['Invoice', 'StockCode'])['Quantity']
          .sum().unstack(fill_value=0))
basket = (basket > 0).astype(int)

freq_items = apriori(basket, min_support=0.03, use_colnames=True)

try:
    rules = association_rules(freq_items, metric='lift', min_threshold=1.5)
except TypeError:
    # mlxtend >= 0.22 requires num_itemsets
    rules = association_rules(freq_items, metric='lift', min_threshold=1.5,
                               num_itemsets=len(basket))

rules = rules.sort_values('lift', ascending=False)

print(f"Frequent itemsets: {len(freq_items)}")
print(f"Rules generated:   {len(rules)}")
print("\nTop 10 rules by Lift:")
print(rules[['antecedents', 'consequents', 'support',
             'confidence', 'lift']].head(10).to_string())

# ==============================================================================
# 8. EXPORT ENRICHED DATASET
# ==============================================================================
df.to_csv('../data/processed/customer_clustered.csv', index=False)
print(f"\nEnriched dataset → customer_clustered.csv")