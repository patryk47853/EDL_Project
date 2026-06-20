import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (KFold, StratifiedKFold,
                                      train_test_split, cross_val_score)
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (mean_squared_error, r2_score,
                              roc_curve, auc)
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

# ==============================================================================
# 1. LOAD DATA & DEFINE FEATURE SPACE
# ==============================================================================
df = pd.read_csv('../data/processed/customer_clustered.csv')

feature_cols = ['Frequency', 'TotalQuantity', 'TotalSpent', 'AvgPrice',
                'AvgOrderValue', 'UniqueProducts', 'AvgQuantityPerOrder',
                'StdPrice', 'SpentPerTransaction', 'ProductsPerTransaction',
                'KMeans_Cluster']

X = df[feature_cols]
y_reg = df['LogFutureSpent']   # continuous target (regression)
y_cls = df['HighValue']        # binary target    (classification)

# ==============================================================================
# 2. TRAIN/TEST SPLIT — SCALING *AFTER* SPLIT (prevents data leakage)
# ==============================================================================
(X_train, X_test,
 y_train_reg, y_test_reg,
 y_train_cls, y_test_cls) = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit on train ONLY
X_test_s  = scaler.transform(X_test)        # transform test with train stats

# Cross-validators
kf  = KFold(n_splits=5, shuffle=True, random_state=42)           # regression
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # classification

# ==============================================================================
# 3. K-FOLD CV: LASSO (L1) ALPHA OPTIMISATION
# ==============================================================================
print("=" * 60)
print("K-FOLD CV — LASSO ALPHA SELECTION")
print("=" * 60)

alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
cv_results = {}

for alpha in alphas:
    fold_scores = []
    for train_idx, val_idx in kf.split(X_train_s):
        X_tr, X_val = X_train_s[train_idx], X_train_s[val_idx]
        y_tr, y_val = y_train_reg.iloc[train_idx], y_train_reg.iloc[val_idx]

        model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        model.fit(X_tr, y_tr)
        fold_scores.append(model.score(X_val, y_val))

    mean_r2 = np.mean(fold_scores)
    std_r2  = np.std(fold_scores)
    cv_results[alpha] = mean_r2
    print(f"  α={alpha:<8} → Mean R²={mean_r2:.4f} (±{std_r2:.4f})")

best_alpha = max(cv_results, key=cv_results.get)
print(f"\n✓ Best alpha: {best_alpha}")

# ==============================================================================
# 4. REGRESSION MODELS
# ==============================================================================
# 4a. Lasso (L1 Regularisation — performs feature selection)
lasso = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
lasso.fit(X_train_s, y_train_reg)

print("\n--- LASSO FEATURE SELECTION (L1 PENALTY) ---")
for feat, coef in zip(feature_cols, lasso.coef_):
    status = "✓ RETAINED" if abs(coef) > 1e-6 else "✗ DROPPED"
    print(f"  {feat:<25} Coef={coef:>10.5f}  {status}")
retained = sum(1 for c in lasso.coef_ if abs(c) > 1e-6)
print(f"Retained: {retained}/{len(feature_cols)} features\n")

# 4b. Ridge (L2 Regularisation — shrinks but does not zero out)
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_s, y_train_reg)

# 4c. Random Forest (Bagging — reduces variance via parallel ensemble)
rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                            random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train_reg)

# 4d. Gradient Boosting (Boosting — reduces bias via sequential correction)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                learning_rate=0.1, random_state=42)
gb.fit(X_train_s, y_train_reg)

# --- REGRESSION BENCHMARK ---
print("=" * 60)
print("REGRESSION BENCHMARK")
print("=" * 60)

reg_models = {
    'Lasso (L1)':         lasso,
    'Ridge (L2)':         ridge,
    'Random Forest':      rf,
    'Gradient Boosting':  gb,
}

reg_results = []
for name, mdl in reg_models.items():
    preds = mdl.predict(X_test_s)
    rmse  = np.sqrt(mean_squared_error(y_test_reg, preds))
    r2    = r2_score(y_test_reg, preds)
    cv_r2 = cross_val_score(mdl, X_train_s, y_train_reg,
                             cv=kf, scoring='r2').mean()
    reg_results.append({'Model': name, 'Test RMSE': round(rmse, 4),
                        'Test R²': round(r2, 4),
                        'CV R² (mean)': round(cv_r2, 4)})

print(pd.DataFrame(reg_results).to_string(index=False))

# Feature importance (Random Forest)
plt.figure(figsize=(10, 6))
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
imp.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../data/processed/feature_importance.png', dpi=150)
plt.close()

# ==============================================================================
# 5. CLASSIFICATION MODELS (SVM, KNN, Neural Network)
# ==============================================================================
print("\n" + "=" * 60)
print("CLASSIFICATION BENCHMARK (StratifiedKFold)")
print("=" * 60)

svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_s, y_train_cls)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train_cls)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                     random_state=42, early_stopping=True)
mlp.fit(X_train_s, y_train_cls)

cls_models = {'SVM (RBF)': svm, 'KNN (k=5)': knn, 'MLP Neural Net': mlp}

cls_results = []
for name, mdl in cls_models.items():
    cv_acc = cross_val_score(mdl, X_train_s, y_train_cls,
                              cv=skf, scoring='accuracy')
    test_acc = mdl.score(X_test_s, y_test_cls)
    cls_results.append({'Model': name,
                        'Test Acc': round(test_acc, 4),
                        'CV Acc (mean)': round(cv_acc.mean(), 4),
                        'CV Acc (std)':  round(cv_acc.std(), 4)})

print(pd.DataFrame(cls_results).to_string(index=False))

# ==============================================================================
# 6. ROC CURVES
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

for name, mdl in cls_models.items():
    y_prob = mdl.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_cls, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random baseline')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Classification Models')
ax.legend()
plt.tight_layout()
plt.savefig('../data/processed/roc_curves.png', dpi=150)
plt.close()

# ==============================================================================
# 7. BOOTSTRAP CONFIDENCE INTERVALS
# ==============================================================================
print("\n--- BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 iterations) ---")

np.random.seed(42)
N_BOOT = 1000

for name, mdl in reg_models.items():
    preds = mdl.predict(X_test_s)
    scores = []
    for _ in range(N_BOOT):
        idx = np.random.choice(len(y_test_reg), len(y_test_reg), replace=True)
        scores.append(r2_score(y_test_reg.iloc[idx], preds[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    print(f"  {name:<25} R²  95% CI: [{lo:.4f}, {hi:.4f}]")

for name, mdl in cls_models.items():
    preds = mdl.predict(X_test_s)
    scores = []
    for _ in range(N_BOOT):
        idx = np.random.choice(len(y_test_cls), len(y_test_cls), replace=True)
        scores.append(np.mean(y_test_cls.iloc[idx] == preds[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    print(f"  {name:<25} Acc 95% CI: [{lo:.4f}, {hi:.4f}]")

print("\n✓ All analyses complete. Plots saved to ../data/processed/")