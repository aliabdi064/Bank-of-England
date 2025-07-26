import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import numpy as np

# Define the reports directory
reports_dir = 'reports/'
# Create the directory if it doesn't exist
os.makedirs(reports_dir, exist_ok=True)
print(f"Reports directory: {os.path.abspath(reports_dir)}")

# --- 2.2 Load Efficiently & Select Relevant Columns ---
print("--- Section 2.2: Efficient Data Loading and Column Selection ---")

data_path = 'data/House_Price_Full.csv'

# Define ALL original column names as per your schema, corrected based on CSV inspection
all_column_names = [
    'Transaction unique identifier', 'Price', 'Date of Transfer', 'Postcode',
    'Property Type', 'Old/New', 'Duration',
    'PAON', 'SAON', 'Street', 'Locality', 'Town', 'County', 'District',
    'PPD Category Type', 'Record Status'
]

# Define only the relevant columns you need for analysis
relevant_columns_names = [
    'Price', 'Date of Transfer', 'Property Type', 'Town', 'County',
    'Old/New', 'Duration'
]

# Initialize an empty list to store dataframes from chunks
df_list = []
chunk_size = 500_000 # As requested

try:
    # Read CSV in chunks
    for chunk in pd.read_csv(data_path, header=None, names=all_column_names,
                             chunksize=chunk_size, low_memory=False, encoding='utf-8'):
        # Select only the relevant columns from each chunk
        df_list.append(chunk[relevant_columns_names])
    
    # Concatenate all chunks into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)
    print("Dataset loaded efficiently and relevant columns selected.")
    print(f"Initial shape after loading relevant columns: {df.shape}")

except FileNotFoundError:
    print(f"Error: The file was not found at {data_path}. Please ensure the path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred during efficient CSV loading: {e}")
    exit()

# --- Data Preprocessing and Feature Engineering (New/Enhanced) ---
print("\n--- Data Preprocessing and Feature Engineering (New/Enhanced) ---")

# 1. Remove Duplicates
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
duplicate_rows = initial_rows - df.shape[0]
print(f"Removed {duplicate_rows} duplicate rows.")

# Convert 'Price' to numeric, coercing errors
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# 2. Handle Missing Values (for Town and County if any)
for col in ['Town', 'County']:
    if df[col].isnull().any():
        df[col].fillna('Unknown', inplace=True)
        print(f"Filled missing values in '{col}' with 'Unknown'.")

# 3. Outlier Handling for Price (IQR-based clipping)
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

original_price_outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)].shape[0]
df['Price'] = np.clip(df['Price'], lower_bound, upper_bound)
print(f"Clipped {original_price_outliers} price outliers based on IQR (1.5*IQR).")

# 4. Log Transform Price (Target Variable)
df['Price'] = np.log1p(df['Price'])
print("Price column log-transformed (np.log1p).")

# --- 2.3 Clean and Format (Date and Filtering) ---
print("\n--- Section 2.3: Date Cleaning and Filtering ---")

df['Date of Transfer'] = pd.to_datetime(df['Date of Transfer'], errors='coerce')
df.dropna(subset=['Date of Transfer', 'Price'], inplace=True)

start_date_filter = '2015-01-01'
end_date_filter = '2024-12-31'
df_filtered = df[(df['Date of Transfer'] >= start_date_filter) &
                 (df['Date of Transfer'] <= end_date_filter)].copy()

print(f"Data filtered for years 2015-2024. New shape: {df_filtered.shape}")

# --- Feature Engineering (Continued) ---
print("\n--- Feature Engineering (Continued) ---")

df_filtered['year'] = df_filtered['Date of Transfer'].dt.year
df_filtered['month_of_transfer'] = df_filtered['Date of Transfer'].dt.month
df_filtered['day_of_week_transfer'] = df_filtered['Date of Transfer'].dt.dayofweek
covid_start_date = pd.to_datetime('2020-03-01')
df_filtered['is_post_covid'] = (df_filtered['Date of Transfer'] >= covid_start_date)
print("Temporal features ('year', 'month', 'day of week', 'is_post_covid') created.")

# --- Section 5: Machine Learning Model for Price Prediction (XGBoost) ---
print("\n--- Section 5: Machine Learning Model for Price Prediction (XGBoost) ---")

# 5.1 Data Preparation for ML
print("\n5.1 Data Preparation for ML:")

X = df_filtered[['Property Type', 'Town', 'County', 'Old/New', 'Duration', 'year', 'month_of_transfer', 'day_of_week_transfer', 'is_post_covid']].copy()
y = df_filtered['Price'].copy()

categorical_features = ['Property Type', 'Town', 'County', 'Old/New', 'Duration']
numerical_features = ['year', 'month_of_transfer', 'day_of_week_transfer', 'is_post_covid']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 5.2 Model Training (XGBoost Regressor)
print("\n5.2 Model Training (XGBoost Regressor):\n")
print("Justification for XGBoost:\n")
print("XGBoost (eXtreme Gradient Boosting) is another powerful gradient boosting framework. It is renowned for its performance and is a frequent winner of machine learning competitions. Key advantages include:\n")
print("- **Performance:** Often provides state-of-the-art results on structured/tabular data.\n")
print("- **Regularization:** Includes L1 and L2 regularization to prevent overfitting, which can be more effective than in standard Gradient Boosting.\n")
print("- **Parallel Processing:** Can perform parallel processing during training, making it fast.\n")
print("- **Flexibility:** Highly flexible with numerous parameters for tuning.\n")

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8))
])

print("Training XGBoost model...")
model.fit(X_train, y_train)
print("Model training complete.")

# 5.3 Model Evaluation
print("\n5.3 Model Evaluation:\n")
y_pred = model.predict(X_test)

y_test_original_scale = np.expm1(y_test)
y_pred_original_scale = np.expm1(y_pred)

mae = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
rmse = np.sqrt(mean_squared_error(y_test_original_scale, y_pred_original_scale))
r2 = r2_score(y_test_original_scale, y_pred_original_scale)

print(f"Mean Absolute Error (MAE): £{mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): £{rmse:,.2f}")
print(f"R-squared (R2): {r2:.4f}")

# 5.4 Feature Importance Analysis
print("\n5.4 Feature Importance Analysis:\n")

preprocessor.fit(X_train)
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
scaled_numerical_features = [f'scaled_{col}' for col in numerical_features]
all_feature_names = encoded_feature_names + scaled_numerical_features

feature_importances = model.named_steps['regressor'].feature_importances_

importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Top 15 Feature Importances (XGBoost):")
print(importance_df.head(15))

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importances for House Price Prediction (XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'xgboost_feature_importances.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# 5.5 Prediction and Visualization
print("\n5.5 Prediction and Visualization:\n")

plt.figure(figsize=(10, 10))
plt.scatter(y_test_original_scale, y_pred_original_scale, alpha=0.3)
plt.plot([y_test_original_scale.min(), y_test_original_scale.max()], [y_test_original_scale.min(), y_test_original_scale.max()], '--r', linewidth=2)
plt.title('Actual vs. Predicted House Prices (XGBoost)')
plt.xlabel('Actual Price (£)')
plt.ylabel('Predicted Price (£)')
plt.ticklabel_format(style='plain', axis='x')
plt.ticklabel_format(style='plain', axis='y')
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'xgboost_actual_vs_predicted_prices.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

print("\nXGBoost analysis complete. Review the printed metrics and generated plots for insights.")
