import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np

# --- Configuration ---
DATA_PATH = 'data/House_Price_Full.csv'
REPORTS_DIR = 'reports/'
CHUNK_SIZE = 500_000
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
COVID_START_DATE = '2020-03-01'
RANDOM_STATE = 42

# --- Main Analysis Function ---
def main():
    """
    Main function to run the complete data analysis and modeling pipeline.
    """
    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load and Prepare Data
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    # 2. Feature Engineering
    df = create_features(df)

    # 3. Train and Evaluate Model
    train_and_evaluate(df)

# --- Data Handling Functions ---
def load_data(path):
    """Loads the dataset efficiently in chunks."""
    print("Loading dataset...")
    all_cols = [
        'Transaction unique identifier', 'Price', 'Date of Transfer', 'Postcode',
        'Property Type', 'Old/New', 'Duration', 'PAON', 'SAON', 'Street',
        'Locality', 'Town', 'County', 'District', 'PPD Category Type', 'Record Status'
    ]
    relevant_cols = [
        'Price', 'Date of Transfer', 'Property Type', 'Town', 'County', 'Old/New', 'Duration'
    ]
    df = pd.concat(
        [chunk[relevant_cols] for chunk in pd.read_csv(
            path, header=None, names=all_cols, chunksize=CHUNK_SIZE,
            low_memory=False, encoding='utf-8'
        )],
        ignore_index=True
    )
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Cleans, filters, and transforms the raw data."""
    print("Preprocessing data...")
    df.drop_duplicates(inplace=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Date of Transfer'] = pd.to_datetime(df['Date of Transfer'], errors='coerce')
    df.dropna(subset=['Date of Transfer', 'Price'], inplace=True)

    df = df[(df['Date of Transfer'] >= START_DATE) & (df['Date of Transfer'] <= END_DATE)].copy()

    # Handle outliers and log-transform the target variable
    q1 = df['Price'].quantile(0.25)
    q3 = df['Price'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df['Price'] = np.clip(df['Price'], lower_bound, upper_bound)
    df['Price'] = np.log1p(df['Price'])
    print("Data preprocessing complete.")
    return df

def create_features(df):
    """Creates new features for the model."""
    print("Creating features...")
    df['year'] = df['Date of Transfer'].dt.year
    df['month'] = df['Date of Transfer'].dt.month
    df['dayofweek'] = df['Date of Transfer'].dt.dayofweek
    df['is_post_covid'] = (df['Date of Transfer'] >= pd.to_datetime(COVID_START_DATE)).astype(int)
    print("Feature creation complete.")
    return df

# --- Modeling Function ---
def train_and_evaluate(df):
    """Trains the LightGBM model and evaluates its performance."""
    print("Starting model training and evaluation...")
    
    categorical_features = ['Property Type', 'Town', 'County', 'Old/New', 'Duration']
    numerical_features = ['year', 'month', 'dayofweek', 'is_post_covid']
    
    X = df[categorical_features + numerical_features]
    y = df['Price']

    for col in categorical_features:
        X[col] = X[col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    # Train the model
    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_estimators=1000, learning_rate=0.05, num_leaves=31)
    model.fit(X_train_scaled, y_train, categorical_feature=categorical_features)
    print("Model training complete.")

    # Evaluate the model
    print("\n--- Model Performance ---")
    y_pred = model.predict(X_test_scaled)
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)

    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test_original, y_pred_original)

    print(f"  Mean Absolute Error (MAE): £{mae:,.2f}")
    print(f"  Root Mean Squared Error (RMSE): £{rmse:,.2f}")
    print(f"  R-squared (R2): {r2:.4f}")
    print("--------------------------")

# --- Script Execution ---
if __name__ == '__main__':
    main()