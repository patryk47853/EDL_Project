import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ==============================================================================
# 1. DATA LOADING & INITIAL QUALITY ASSESSMENT
# ==============================================================================
df = pd.read_csv('../data/raw/online_retail_II.csv', encoding='utf-8')

print("=" * 60)
print("INITIAL DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nMissing Customer ID: {df['Customer ID'].isnull().sum()} "
      f"({df['Customer ID'].isnull().mean()*100:.1f}%)")

# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================
# Remove rows without Customer ID — customer-centric analysis requires it
clean = df.dropna(subset=['Customer ID']).copy()
clean['Customer ID'] = clean['Customer ID'].astype(int)
clean['InvoiceDate'] = pd.to_datetime(clean['InvoiceDate'])

# Detect and isolate cancellations (Invoice prefix 'C')
mask_cancel = clean['Invoice'].astype(str).str.startswith('C')
print(f"\nCancellations: {mask_cancel.sum()} ({mask_cancel.mean()*100:.1f}%)")

# Keep only valid, positive transactions
sales = clean[
    (~mask_cancel) & (clean['Quantity'] > 0) & (clean['Price'] > 0)
].copy()
sales['TotalLineSum'] = sales['Quantity'] * sales['Price']

print(f"Valid transactions: {len(sales)}")
print(f"Date range: {sales['InvoiceDate'].min().date()} → "
      f"{sales['InvoiceDate'].max().date()}")
print(f"Unique customers: {sales['Customer ID'].nunique()}")

# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS — VISUALISATIONS
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a. Transaction value distribution (log scale — heavy right skew)
sns.histplot(sales['TotalLineSum'], bins=100, kde=True,
             log_scale=True, color='teal', ax=axes[0, 0])
axes[0, 0].set_title('Transaction Value Distribution (Log Scale)')
axes[0, 0].set_xlabel('Total Line Sum (GBP)')

# 3b. Top 10 countries by total revenue
top_countries = sales.groupby('Country')['TotalLineSum'].sum().nlargest(10)
top_countries.plot(kind='barh', ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title('Top 10 Countries by Revenue')

# 3c. Monthly revenue trend (time-series perspective)
monthly = sales.set_index('InvoiceDate').resample('ME')['TotalLineSum'].sum()
monthly.plot(ax=axes[1, 0], color='darkgreen', linewidth=2)
axes[1, 0].set_title('Monthly Revenue Trend')
axes[1, 0].set_ylabel('Revenue (GBP)')

# 3d. Price vs Quantity scatter (sampled to avoid overplotting)
sample = sales.sample(min(5000, len(sales)), random_state=42)
axes[1, 1].scatter(sample['Price'], sample['Quantity'],
                    alpha=0.3, s=10, c='coral')
axes[1, 1].set_title('Price vs Quantity (Sample)')
axes[1, 1].set_xlim(0, 50)
axes[1, 1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('../data/processed/eda_overview.png', dpi=150)
plt.close()

# ==============================================================================
# 4. TEMPORAL SPLIT & FEATURE ENGINEERING
# ==============================================================================
# Instead of a synthetic target we use a REAL temporal split:
#   Period 1 → customer behavioural features
#   Period 2 → actual future spending (regression target)

cutoff = pd.Timestamp('2011-06-01')
p1 = sales[sales['InvoiceDate'] < cutoff]
p2 = sales[sales['InvoiceDate'] >= cutoff]

print(f"\nPeriod 1 (features): "
      f"{p1['InvoiceDate'].min().date()} → {p1['InvoiceDate'].max().date()}")
print(f"Period 2 (target):   "
      f"{p2['InvoiceDate'].min().date()} → {p2['InvoiceDate'].max().date()}")

# Aggregate customer-level features from Period 1
features = p1.groupby('Customer ID').agg(
    Frequency=('Invoice', 'nunique'),
    TotalQuantity=('Quantity', 'sum'),
    TotalSpent=('TotalLineSum', 'sum'),
    AvgPrice=('Price', 'mean'),
    AvgOrderValue=('TotalLineSum', 'mean'),
    UniqueProducts=('StockCode', 'nunique'),
    AvgQuantityPerOrder=('Quantity', 'mean'),
    StdPrice=('Price', 'std'),
).reset_index()

features['StdPrice'] = features['StdPrice'].fillna(0)
features['SpentPerTransaction'] = features['TotalSpent'] / features['Frequency']
features['ProductsPerTransaction'] = (features['UniqueProducts']
                                       / features['Frequency'])

# Real target: actual spending in Period 2
target = p2.groupby('Customer ID')['TotalLineSum'].sum().reset_index()
target.columns = ['Customer ID', 'FutureSpent']

# Inner join — only customers present in BOTH periods
customer_matrix = features.merge(target, on='Customer ID', how='inner')

# Log-transform for regression (heavy right skew in spending)
customer_matrix['LogFutureSpent'] = np.log1p(customer_matrix['FutureSpent'])

# Binary target for classification: above-median = high-value customer
median_spent = customer_matrix['FutureSpent'].median()
customer_matrix['HighValue'] = (
    customer_matrix['FutureSpent'] > median_spent
).astype(int)

print(f"\nCustomers in both periods: {len(customer_matrix)}")
print(f"Median future spending: £{median_spent:.2f}")
print(f"High-value ratio: {customer_matrix['HighValue'].mean()*100:.0f}%")

# ==============================================================================
# 5. EXPORT
# ==============================================================================
customer_matrix.to_csv('../data/processed/customer_baseline.csv', index=False)
sales.to_csv('../data/processed/clean_transactions.csv', index=False)

print("\n--- CUSTOMER MATRIX PREVIEW ---")
print(customer_matrix.head())
print(f"\nExported: customer_baseline.csv & clean_transactions.csv")