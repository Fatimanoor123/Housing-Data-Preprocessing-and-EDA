import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load datasets
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

# Select only categorical columns
cat_cols_test = df_test.select_dtypes(include=['object']).columns
cat_cols_train = df_train.select_dtypes(include=['object']).columns

# Initialize the SimpleImputer for categorical data
cat_imputer = SimpleImputer(strategy='most_frequent')

# Fit and transform the categorical columns
df_test[cat_cols_test] = cat_imputer.fit_transform(df_test[cat_cols_test])
df_train[cat_cols_train] = cat_imputer.fit_transform(df_train[cat_cols_train])

# Check for missing categorical values
print("Remaining missing values in test.csv (categorical):", df_test[cat_cols_test].isnull().sum().sum())
print("Remaining missing values in train.csv (categorical):", df_train[cat_cols_train].isnull().sum().sum())

# Function to plot heatmap
def plot_corr_heatmap(df, title):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="YlGnBu", annot=False, cbar=True)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Plot for train data
plot_corr_heatmap(df_train, 'Correlation Heatmap - Train Dataset')

# Plot for test data
plot_corr_heatmap(df_test, 'Correlation Heatmap - Test Dataset')

#test train split
print(df_test.head())

X=df_test.drop(columns=['LotShape', 'MoSold'])

y=df_test['MoSold']

X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=11, test_size=0.2)

print(X_train.shape)

print(X_test.shape)

print(y_test.shape)

print(y_train.shape)