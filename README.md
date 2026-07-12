# Project: Data Mining on Online Retail II

Data Mining project for the *Estimation, Detection and Learning II* course.

## Dataset

Dataset used in this project:
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data

Download the dataset and place the file `online_retail_II.csv` into:
```
data/raw/online_retail_II.csv
```

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