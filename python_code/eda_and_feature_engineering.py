import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. Load data
# ==============================================================================
df = pd.read_csv('../data/raw/online_retail_II.csv')

print("Dataset shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Check how many Customer IDs are missing
missing_customer = df['Customer ID'].isnull().sum()
missing_percent = missing_customer / len(df) * 100
print("\nMissing Customer ID:", missing_customer, "(", round(missing_percent, 1), "%)")

# ==============================================================================
# 2. Data cleaning
# ==============================================================================
# Drop rows without Customer ID (for customer-level analysis)
df_clean = df.dropna(subset=['Customer ID'])
df_clean = df_clean.copy()
df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# Invoices starting with 'C' are cancellations - get rid of them
cancelled = df_clean['Invoice'].astype(str).str.startswith('C')
print("\nCancelled orders:", cancelled.sum())

# Keep only valid transactions (no cancellations + positive quantity and price)
sales = df_clean[~cancelled]
sales = sales[sales['Quantity'] > 0]
sales = sales[sales['Price'] > 0]
sales = sales.copy()

# Calculate total value per transaction line
sales['TotalLineSum'] = sales['Quantity'] * sales['Price']

print("Valid transactions:", len(sales))
print("Date range: from", sales['InvoiceDate'].min().date(), "to", sales['InvoiceDate'].max().date())
print("Unique customers:", sales['Customer ID'].nunique())

# ==============================================================================
# 3. EDA - plots to see what the data looks like
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: transaction value distribution (log scale - data is very skewed)
sns.histplot(sales['TotalLineSum'], bins=100, log_scale=True, ax=axes[0, 0])
axes[0, 0].set_title('Transaction Value Distribution (log scale)')
axes[0, 0].set_xlabel('Transaction value (GBP)')

# Plot 2: top 10 countries by revenue
country_revenue = sales.groupby('Country')['TotalLineSum'].sum()
top10 = country_revenue.sort_values(ascending=False).head(10)
top10.plot(kind='barh', ax=axes[0, 1])
axes[0, 1].set_title('Top 10 countries by revenue')

# Plot 3: monthly revenue - checking if there is seasonality
sales_with_date = sales.set_index('InvoiceDate')
monthly = sales_with_date.resample('ME')['TotalLineSum'].sum()
monthly.plot(ax=axes[1, 0])
axes[1, 0].set_title('Monthly revenue')
axes[1, 0].set_ylabel('Revenue (GBP)')

# Plot 4: price vs quantity - sampled because too many points otherwise
sample = sales.sample(5000, random_state=42)
axes[1, 1].scatter(sample['Price'], sample['Quantity'], alpha=0.3, s=10)
axes[1, 1].set_title('Price vs Quantity (sample of 5000)')
axes[1, 1].set_xlabel('Price')
axes[1, 1].set_ylabel('Quantity')
axes[1, 1].set_xlim(0, 50)
axes[1, 1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('../data/processed/eda_overview.png')
plt.close()

# ==============================================================================
# 4. Temporal split and feature engineering
# ==============================================================================
# Split data into two periods:
# - period 1: extract customer features (how often, how much they bought)
# - period 2: this is "the future" - check how much they actually spent later
# Use real target instead of a made-up one

split_date = pd.Timestamp('2011-06-01')
period1 = sales[sales['InvoiceDate'] < split_date]
period2 = sales[sales['InvoiceDate'] >= split_date]

print("\nPeriod 1 (features):", period1['InvoiceDate'].min().date(), "-", period1['InvoiceDate'].max().date())
print("Period 2 (target):", period2['InvoiceDate'].min().date(), "-", period2['InvoiceDate'].max().date())

# Group transactions by customer and calculate stats
features = period1.groupby('Customer ID').agg({
    'Invoice': 'nunique',           # number of orders
    'Quantity': 'sum',              # total items bought
    'TotalLineSum': ['sum', 'mean'], # total and average order value
    'Price': ['mean', 'std'],       # mean and std of price
    'StockCode': 'nunique'          # number of unique products
})

# Pandas creates multi-level column names, so to flatten them:
features.columns = ['Frequency', 'TotalQuantity', 'TotalSpent', 'AvgOrderValue',
                    'AvgPrice', 'StdPrice', 'UniqueProducts']
features = features.reset_index()

# If customer bought only one product there is no std -> fill with 0
features['StdPrice'] = features['StdPrice'].fillna(0)

# Extra features that might be valuable later
features['AvgQuantityPerOrder'] = features['TotalQuantity'] / features['Frequency']
features['SpentPerTransaction'] = features['TotalSpent'] / features['Frequency']
features['ProductsPerTransaction'] = features['UniqueProducts'] / features['Frequency']

# Target: how much customer spent in period 2 (this is the "future")
target = period2.groupby('Customer ID')['TotalLineSum'].sum().reset_index()
target.columns = ['Customer ID', 'FutureSpent']

# Merge features with target to keep only customers present in both periods
customer_matrix = features.merge(target, on='Customer ID')

# Log-transform the target because it is very skewed
customer_matrix['LogFutureSpent'] = np.log1p(customer_matrix['FutureSpent'])

# Make a binary target for classification: is the customer "high value"?
# Threshold = median spending
median_spent = customer_matrix['FutureSpent'].median()
customer_matrix['HighValue'] = (customer_matrix['FutureSpent'] > median_spent).astype(int)

print("\nCustomers in both periods:", len(customer_matrix))
print("Median future spending:", round(median_spent, 2), "GBP")

# ==============================================================================
# 5. Save results
# ==============================================================================
customer_matrix.to_csv('../data/processed/customer_baseline.csv', index=False)
sales.to_csv('../data/processed/clean_transactions.csv', index=False)

print("\nCustomer matrix preview:")
print(customer_matrix.head())