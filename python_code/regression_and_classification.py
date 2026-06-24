import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Load data and pick features
# ==============================================================================
df = pd.read_csv('../data/processed/customer_clustered.csv')

feature_cols = ['Frequency', 'TotalQuantity', 'TotalSpent', 'AvgPrice',
                'AvgOrderValue', 'UniqueProducts', 'AvgQuantityPerOrder',
                'StdPrice', 'SpentPerTransaction', 'ProductsPerTransaction',
                'KMeans_Cluster']

X = df[feature_cols]
y_reg = df['LogFutureSpent']   # target for regression
y_cls = df['HighValue']        # target for classification (0 or 1)

# ==============================================================================
# 2. Train/test split and scaling
# ==============================================================================
# Let's split data first, then scale, cause otherwise test data leaks into training
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Cross-validation objects
kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==============================================================================
# 3. K-Fold CV - finding best alpha for Lasso
# ==============================================================================
# Try different alphas and pick the one with best average R2
print("Lasso alpha selection:")

alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
cv_results = {}

for alpha in alphas:
    fold_scores = []
    for train_idx, val_idx in kf.split(X_train_s):
        X_tr = X_train_s[train_idx]
        X_val = X_train_s[val_idx]
        y_tr = y_train_reg.iloc[train_idx]
        y_val = y_train_reg.iloc[val_idx]

        model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        model.fit(X_tr, y_tr)
        score = model.score(X_val, y_val)
        fold_scores.append(score)

    mean_r2 = np.mean(fold_scores)
    cv_results[alpha] = mean_r2
    print("  alpha =", alpha, "-> Mean R2 =", round(mean_r2, 4))

best_alpha = max(cv_results, key=cv_results.get)
print("Best alpha:", best_alpha)

# ==============================================================================
# 4. Regression models
# ==============================================================================
# Lasso with best alpha from CV; L1 regularisation drops useless features
lasso = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
lasso.fit(X_train_s, y_train_reg)

# Check which features were kept by Lasso
print("\nLasso feature selection:")
retained = 0
for i in range(len(feature_cols)):
    feat = feature_cols[i]
    coef = lasso.coef_[i]
    if abs(coef) > 1e-6:
        status = "RETAINED"
        retained = retained + 1
    else:
        status = "DROPPED"
    print(" ", feat, "coef =", round(coef, 5), "->", status)
print("Retained:", retained, "out of", len(feature_cols), "features\n")

# Ridge - L2 regularisation, shrinks coefficients but does not zero them
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_s, y_train_reg)

# Random Forest - bagging ensemble, reduces variance
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train_reg)

# Gradient Boosting - sequential ensemble, reduces bias
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                learning_rate=0.1, random_state=42)
gb.fit(X_train_s, y_train_reg)

# Compare all regression models on test set
print("Regression results:")

reg_models = {
    'Lasso (L1)': lasso,
    'Ridge (L2)': ridge,
    'Random Forest': rf,
    'Gradient Boosting': gb
}

reg_results = []
for name in reg_models:
    model = reg_models[name]
    preds = model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
    r2 = r2_score(y_test_reg, preds)
    cv_scores = cross_val_score(model, X_train_s, y_train_reg, cv=kf, scoring='r2')
    cv_r2 = cv_scores.mean()
    reg_results.append({
        'Model': name,
        'Test RMSE': round(rmse, 4),
        'Test R2': round(r2, 4),
        'CV R2': round(cv_r2, 4)
    })

print(pd.DataFrame(reg_results))

# Feature importance plot from Random Forest
plt.figure(figsize=(10, 6))
importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values()
importances.plot(kind='barh')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../data/processed/feature_importance.png')
plt.close()

# ==============================================================================
# 5. Classification models
# ==============================================================================
print("\nClassification results:")

# SVM with RBF kernel
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_s, y_train_cls)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train_cls)

# Neural network (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                     random_state=42, early_stopping=True)
mlp.fit(X_train_s, y_train_cls)

cls_models = {
    'SVM (RBF)': svm,
    'KNN (k=5)': knn,
    'MLP Neural Net': mlp
}

cls_results = []
for name in cls_models:
    model = cls_models[name]
    cv_acc = cross_val_score(model, X_train_s, y_train_cls, cv=skf, scoring='accuracy')
    test_acc = model.score(X_test_s, y_test_cls)
    cls_results.append({
        'Model': name,
        'Test Acc': round(test_acc, 4),
        'CV Acc (mean)': round(cv_acc.mean(), 4),
        'CV Acc (std)': round(cv_acc.std(), 4)
    })

print(pd.DataFrame(cls_results))

# ==============================================================================
# 6. ROC curves for classification
# ==============================================================================
plt.figure(figsize=(8, 6))

for name in cls_models:
    model = cls_models[name]
    # Get probability of class 1
    y_prob = model.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_cls, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=name + ' (AUC = ' + str(round(roc_auc, 3)) + ')')

# Diagonal line = random classifier
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig('../data/processed/roc_curves.png')
plt.close()

# ==============================================================================
# 7. Bootstrap - confidence intervals for model performance
# ==============================================================================
print("\nBootstrap 95% confidence intervals (1000 iterations):")

np.random.seed(42)
n_boot = 1000

# For regression models -> R2 confidence interval
for name in reg_models:
    model = reg_models[name]
    preds = model.predict(X_test_s)
    scores = []
    for i in range(n_boot):
        # Resample test set with replacement
        idx = np.random.choice(len(y_test_reg), len(y_test_reg), replace=True)
        score = r2_score(y_test_reg.iloc[idx], preds[idx])
        scores.append(score)
    lo = np.percentile(scores, 2.5)
    hi = np.percentile(scores, 97.5)
    print(" ", name, "R2 CI: [", round(lo, 4), ",", round(hi, 4), "]")

# For classification models -> Accuracy confidence interval
for name in cls_models:
    model = cls_models[name]
    preds = model.predict(X_test_s)
    scores = []
    for i in range(n_boot):
        idx = np.random.choice(len(y_test_cls), len(y_test_cls), replace=True)
        acc = np.mean(y_test_cls.iloc[idx] == preds[idx])
        scores.append(acc)
    lo = np.percentile(scores, 2.5)
    hi = np.percentile(scores, 97.5)
    print(" ", name, "Acc CI: [", round(lo, 4), ",", round(hi, 4), "]")

print("\nPlots saved.")