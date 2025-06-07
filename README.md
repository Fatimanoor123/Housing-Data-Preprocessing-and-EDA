# Data Preprocessing & EDA with Categorical Imputation

This project demonstrates essential steps in preprocessing real-world tabular datasets including handling missing values in categorical features using `SimpleImputer`, exploratory data analysis (EDA) with correlation heatmaps, and a basic train-test split for regression modeling.

## 📁 Files

- `train.csv` – Training dataset (not uploaded due to size/privacy).
- `test.csv` – Testing dataset (not uploaded due to size/privacy).
- `main.py` – Python script containing data processing and visualization logic.

## 🔍 Features

- Imputation of missing **categorical features** with the most frequent value.
- Correlation **heatmaps** using Seaborn for both train and test sets.
- Basic **train-test split** on selected features.
- Setup for simple **linear regression** (placeholders for later model implementation).

## 🧪 Libraries Used

- `pandas`
- `sklearn` (SimpleImputer, train_test_split, LinearRegression, mean_squared_error)
- `matplotlib.pyplot`
- `seaborn`

## 📊 Visualizations

Correlation heatmaps are generated for:
- **Train dataset**
- **Test dataset**

These visualizations help understand relationships between numerical features.

## 🛠 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-preprocessing-eda.git
   cd data-preprocessing-eda

