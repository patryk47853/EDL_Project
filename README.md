# Project: Data Mining on Online Retail II

Data Mining project for the *Estimation, Detection and Learning II* course.
The project applies both descriptive and predictive Data Mining techniques to a
real e-commerce dataset from a UK-based online store.

## Project Goals

The main focus is on two points:

1. **Customer segmentation** - grouping customers based on their past behaviour.
2. **Predicting future customer spending** - using regression with regularization
   and cross-validation.

The project covers four of the six techniques from the Web Mining mind map:
Clustering, Regression, Classification and Association Rules.

## Dataset

Dataset used in this project:
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data

- ~1 000 000 transactions from a UK-based online store
- Period: December 2009 - December 2011

Download the dataset and place the file `online_retail_II.csv` into:
```
data/raw/online_retail_II.csv
```

## Project Structure

```
EDL_Project/
├── python_code/
│   ├── part1_eda_and_feature_engineering.py
│   ├── part2_clustering_and_association_rules.py
│   └── part3_regression_and_classification.py
├── data/
│   ├── raw/                  # place online_retail_II.csv here
│   └── processed/            # generated CSVs and plots
├── requirements.txt
└── README.md
```

## Pipeline

The project is split into three sequential parts. Each part reads the output of
the previous one from `data/processed/`.

### Part 1 - EDA and Feature Engineering
- Loads and cleans the raw data (removes missing Customer IDs and cancelled orders)
- Performs exploratory data analysis (4 plots)
- Uses a **temporal split**: behaviour from Dec 2009 - May 2011 is used as features,
  and spending from Jun 2011 - Dec 2011 is used as the prediction target
- Builds a customer-level feature matrix

### Part 2 - Clustering and Association Rules
- **Clustering:** K-Means, K-Medoids, DBSCAN and Hierarchical clustering
- Uses Elbow method and Silhouette score to choose the number of clusters
- **Association Rules:** Apriori algorithm on the product baskets

### Part 3 - Regression and Classification
- **Regression** (predicting future spending): Lasso, Ridge, Random Forest,
  Gradient Boosting
- **Classification** (high-value customers): SVM, KNN, MLP
- Evaluation with 5-Fold Cross-Validation, ROC curves and bootstrap confidence intervals

## Main Results

| Task | Best model | Score |
|------|-----------|-------|
| Regression | Random Forest | R² = 0.55 |
| Classification | SVM (RBF) | Accuracy = 0.75, AUC = 0.84 |
| Association Rules | Apriori | strongest rule with Lift = 9.11 |

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac/Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to run

Run the three scripts in order:

```bash
python python_code/part1_eda_and_feature_engineering.py
python python_code/part2_clustering_and_association_rules.py
python python_code/part3_regression_and_classification.py
```

All results (CSV files and plots) are saved into `data/processed/`

## Technologies

- Python 3.9
- pandas, numpy
- scikit-learn
- scipy
- mlxtend (Apriori)
- kmedoids
- matplotlib, seaborn
