import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
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
    print("\nFirst 5 rows of loaded data:")
    print(df.head())

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
# Fill missing Town/County with 'Unknown' before any encoding
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

# Clip outliers
original_price_outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)].shape[0]
df['Price'] = np.clip(df['Price'], lower_bound, upper_bound)
print(f"Clipped {original_price_outliers} price outliers based on IQR (1.5*IQR). Price range after clipping: £{df['Price'].min():,.2f} - £{df['Price'].max():,.2f}")

# 4. Log Transform Price (Target Variable)
# Apply log1p to handle skewed distribution and make it more Gaussian-like
df['Price'] = np.log1p(df['Price'])
print("Price column log-transformed (np.log1p).")

# --- 2.3 Clean and Format (Date and Filtering) ---
print("\n--- Section 2.3: Date Cleaning and Filtering ---")

# Convert 'Date of Transfer' to datetime objects
# 'errors='coerce'' will turn unparseable dates into NaT (Not a Time)
df['Date of Transfer'] = pd.to_datetime(df['Date of Transfer'], errors='coerce')

# Drop rows where 'Date of Transfer' or 'Price' could not be parsed (NaT values)
df.dropna(subset=['Date of Transfer', 'Price'], inplace=True)
print(f"Shape after dropping rows with invalid dates or prices: {df.shape}")

# Filter records to 2015–2024
start_date_filter = '2015-01-01'
end_date_filter = '2024-12-31'

df_filtered = df[(df['Date of Transfer'] >= start_date_filter) &
                 (df['Date of Transfer'] <= end_date_filter)].copy()

print(f"Data filtered for years 2015-2024. New shape: {df_filtered.shape}")
print("First 5 rows of filtered data (2015-2024):")
print(df_filtered.head())

# --- Feature Engineering (Continued) ---
print("\n--- Feature Engineering (Continued) ---")

# Create new columns: 'year'
df_filtered['year'] = df_filtered['Date of Transfer'].dt.year
print("'year' column created.")

# Create new columns: 'month_of_transfer'
df_filtered['month_of_transfer'] = df_filtered['Date of Transfer'].dt.month
print("'month_of_transfer' column created.")

# Create new columns: 'day_of_week_transfer'
df_filtered['day_of_week_transfer'] = df_filtered['Date of Transfer'].dt.dayofweek
print("'day_of_week_transfer' column created.")

# Create new columns: 'is_post_covid'
# Using March 1, 2020 as the start of the post-COVID period for this analysis
covid_start_date = pd.to_datetime('2020-03-01')
df_filtered['is_post_covid'] = (df_filtered['Date of Transfer'] >= covid_start_date)
print("'is_post_covid' column created.")

print("\nDataFrame after cleaning and formatting:")
print(df_filtered.info())
print(df_filtered.describe())
print(df_filtered.head())


# --- 2. How house prices changed since COVID-19 (2020 onward) ---
print("\n--- Section 2: COVID-19 Impact Analysis (2020 Onward) ---")

# Filter data from 2020 onwards (using the already filtered df_filtered)
# Note: df_filtered already contains data up to 2024, so we just need to filter from 2020
df_covid_analysis = df_filtered[df_filtered['Date of Transfer'] >= pd.to_datetime('2020-01-01')].copy()

print(f"Data from 2020 onwards for COVID analysis: {df_covid_analysis.shape[0]} records.")

# Extract Year and Month for time series analysis
df_covid_analysis['YearMonth'] = df_covid_analysis['Date of Transfer'].dt.to_period('M')

# Calculate average price per month
monthly_avg_price = df_covid_analysis.groupby('YearMonth')['Price'].mean().reset_index()
monthly_avg_price['YearMonth'] = monthly_avg_price['YearMonth'].dt.to_timestamp() # Convert back to timestamp for plotting

print("\nMonthly Average Prices (2020 onwards):")
print(monthly_avg_price.head())

# Visualize monthly average prices
plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_avg_price, x='YearMonth', y=np.expm1(monthly_avg_price['Price'])) # Inverse transform for plotting
plt.title('Average House Price Trend Since COVID-19 (2020 Onward)')
plt.xlabel('Date')
plt.ylabel('Average Price (£)')
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'covid_price_trend.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close() # Close the plot to free memory


# --- 3. Regional price trends (e.g. London vs North East) ---
print("\n--- Section 3: Regional Price Trends ---")

# Calculate average price per County for the entire 2015-2024 period
county_avg_price = df_filtered.groupby('County')['Price'].mean().sort_values(ascending=False).reset_index()

print("\nTop 10 Counties by Average Price (2015-2024):")
print(county_avg_price.head(10))

# Identify some counties for London and North East for comparison
# Updated based on unique county names from previous run
london_counties = [
    'CITY OF LONDON', 'CITY OF WESTMINSTER', 'KENSINGTON AND CHELSEA', 'CAMDEN',
    'HAMMERSMITH AND FULHAM', 'ISLINGTON', 'SOUTHWARK', 'WANDSWORTH',
    'BARNET', 'BEXLEY', 'BRENT', 'BROMLEY', 'CROYDON', 'EALING', 'ENFIELD',
    'GREENWICH', 'HACKNEY', 'HARINGEY', 'HARROW', 'HAVERING', 'HILLINGDON',
    'HOUNSLOW', 'KINGSTON UPON THAMES', 'LAMBETH', 'LEWISHAM', 'MERTON',
    'NEWHAM', 'REDBRIDGE', 'RICHMOND UPON THAMES', 'SUTTON', 'TOWER HAMLETS',
    'WALTHAM FOREST'
]
north_east_counties = ['TYNE AND WEAR', 'DURHAM', 'NORTHUMBERLAND', 'COUNTY DURHAM', 'GATESHEAD', 'NEWCASTLE UPON TYNE', 'NORTH TYNESIDE', 'SOUTH TYNESIDE', 'SUNDERLAND', 'HARTLEPOOL', 'MIDDLESBROUGH', 'REDCAR AND CLEVELAND', 'DARLINGTON']

# Filter data for these specific regions
df_london = df_filtered[df_filtered['County'].isin(london_counties)].copy()
df_north_east = df_filtered[df_filtered['County'].isin(north_east_counties)].copy()

print(f"\nLondon data points: {df_london.shape[0]}")
print(f"North East data points: {df_north_east.shape[0]}")

# Calculate average prices for the selected regions
avg_price_london = np.expm1(df_london['Price'].mean()) # Inverse transform for printing
avg_price_north_east = np.expm1(df_north_east['Price'].mean()) # Inverse transform for printing

print(f"\nAverage Price in London (2015-2024): £{avg_price_london:,.2f}")
print(f"Average Price in North East (2015-2024): £{avg_price_north_east:,.2f}")

# Visualize average prices for top N counties
plt.figure(figsize=(12, 8))
sns.barplot(x=np.expm1(county_avg_price['Price']), y=county_avg_price['County'].head(15)) # Inverse transform for plotting
plt.title('Average House Price by County (Top 15, 2015-2024)')
plt.xlabel('Average Price (£)')
plt.ylabel('County')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'top_county_avg_price.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Visualize price distribution for London vs North East using box plots
plt.figure(figsize=(10, 6))
sns.boxplot(x='County', y=np.expm1(df_filtered[df_filtered['County'].isin(london_counties + north_east_counties)]['Price']),
            data=df_filtered[df_filtered['County'].isin(london_counties + north_east_counties)])
plt.title('House Price Distribution: London vs North East (2015-2024)')
plt.xlabel('Region')
plt.ylabel('Price (£)')
plt.yscale('log') # Use log scale for better visualization if prices vary widely
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'london_vs_ne_boxplot.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Time series comparison for selected regions (e.g., monthly average)
df_regions_time = df_filtered[df_filtered['County'].isin(london_counties + north_east_counties)].copy()
df_regions_time['YearMonth'] = df_regions_time['Date of Transfer'].dt.to_period('M')
monthly_avg_price_regions = df_regions_time.groupby(['YearMonth', 'County'])['Price'].mean().reset_index()
monthly_avg_price_regions['YearMonth'] = monthly_avg_price_regions['YearMonth'].dt.to_timestamp()

plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_avg_price_regions, x='YearMonth', y=np.expm1(monthly_avg_price_regions['Price']), hue='County') # Inverse transform for plotting
plt.title('Monthly Average House Price Trend: London vs North East (2015-2024)')
plt.xlabel('Date')
plt.ylabel('Average Price (£)')
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'london_vs_ne_timeseries.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()


print("\n--- Section 4: Exploratory Data Analysis (EDA) ---")

# --- 4.1 Data Overview ---
print("\n4.1 Data Overview:")
print(f"Shape of the filtered data: {df_filtered.shape}")
print("\nDataFrame Info:")
df_filtered.info()

print("\nMissing Values (Count and Percentage):")
missing_data = df_filtered.isnull().sum()
missing_percentage = (df_filtered.isnull().sum() / len(df_filtered)) * 100
missing_info = pd.DataFrame({'Missing Count': missing_data, 'Missing Percentage (%)': missing_percentage})
print(missing_info[missing_info['Missing Count'] > 0])


# --- 4.2 Descriptive Statistics ---
print("\n4.2 Descriptive Statistics:")
print("\nDescriptive Statistics for Numerical Columns:")
print(df_filtered.describe())

print("\nValue Counts for Categorical Columns:")
for col in ['Property Type', 'Old/New', 'Duration']: # Removed PPD Category Type and Record Status as they are not in relevant_columns_names
    if col in df_filtered.columns:
        print(f"\n--- {col} ---")
        print(df_filtered[col].value_counts())


# --- 4.3 Distribution of Key Variables ---
print("\n4.3 Distribution of Key Variables:")

# Distribution of Price
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.histplot(np.expm1(df_filtered['Price']), bins=50, kde=True) # Inverse transform for plotting
plt.title('Distribution of Price')
plt.xlabel('Price (£)')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='x') # Prevent scientific notation on x-axis

plt.subplot(1, 2, 2)
sns.boxplot(y=np.expm1(df_filtered['Price'])) # Inverse transform for plotting
plt.title('Box Plot of Price')
plt.ylabel('Price (£)')
plt.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'price_distribution.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Distribution of Property Type
plt.figure(figsize=(8, 6))
sns.countplot(data=df_filtered, x='Property Type', order=df_filtered['Property Type'].value_counts().index)
plt.title('Distribution of Property Types')
plt.xlabel('Property Type')
plt.ylabel('Count')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'property_type_distribution.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Distribution of Old/New
plt.figure(figsize=(6, 5))
sns.countplot(data=df_filtered, x='Old/New')
plt.title('Distribution of Old/New Properties')
plt.xlabel('Old/New')
plt.ylabel('Count')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'old_new_distribution.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Distribution of Duration
plt.figure(figsize=(6, 5))
sns.countplot(data=df_filtered, x='Duration')
plt.title('Distribution of Property Duration (Tenure)')
plt.xlabel('Duration')
plt.ylabel('Count')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'duration_distribution.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()


# --- 4.4 Relationships with Price ---
print("\n4.4 Relationships with Price:")

# Price vs. Property Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_filtered, x='Property Type', y=np.expm1(df_filtered['Price']), order=df_filtered['Property Type'].value_counts().index) # Inverse transform for plotting
plt.title('Price Distribution by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Price (£)')
plt.yscale('log') # Log scale often helps visualize skewed price data
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'price_by_property_type.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Price vs. Old/New
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_filtered, x='Old/New', y=np.expm1(df_filtered['Price'])) # Inverse transform for plotting
plt.title('Price Distribution by Old/New Property Status')
plt.xlabel('Old/New')
plt.ylabel('Price (£)')
plt.yscale('log')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'price_by_old_new.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Price vs. Duration
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_filtered, x='Duration', y=np.expm1(df_filtered['Price'])) # Inverse transform for plotting
plt.title('Price Distribution by Duration (Tenure)')
plt.xlabel('Duration')
plt.ylabel('Price (£)')
plt.yscale('log')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'price_by_duration.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Price vs. Year (Average Price Trend)
plt.figure(figsize=(12, 7))
avg_price_per_year = df_filtered.groupby('year')['Price'].mean().reset_index()
sns.lineplot(data=avg_price_per_year, x='year', y=np.expm1(avg_price_per_year['Price']), marker='o') # Inverse transform for plotting
plt.title('Average House Price Trend by Year (2015-2024)')
plt.xlabel('Year')
plt.ylabel('Average Price (£)')
plt.grid(True)
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'price_trend_by_year.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Price vs. is_post_covid
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_filtered, x='is_post_covid', y=np.expm1(df_filtered['Price'])) # Inverse transform for plotting
plt.title('Price Distribution: Pre-COVID vs. Post-COVID (from March 2020)')
plt.xlabel('Is Post-COVID (True if >= Mar 2020)')
plt.ylabel('Price (£)')
plt.yscale('log')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'price_pre_post_covid.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()


# --- 4.5 Geographical Analysis ---
print("\n4.5 Geographical Analysis:")

# Top 15 Towns by Average Price
top_towns = df_filtered.groupby('Town')['Price'].mean().nlargest(15).reset_index()
plt.figure(figsize=(12, 8))
sns.barplot(x=np.expm1(top_towns['Price']), y=top_towns['Town']) # Inverse transform for plotting
plt.title('Top 15 Towns by Average House Price (2015-2024)')
plt.xlabel('Average Price (£)')
plt.ylabel('Town')
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'top_towns_avg_price.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# Top 15 Counties by Average Price (already done in previous section, but good to reiterate for EDA)
top_counties = df_filtered.groupby('County')['Price'].mean().nlargest(15).reset_index()
plt.figure(figsize=(12, 8))
sns.barplot(x=np.expm1(top_counties['Price']), y=top_counties['County']) # Inverse transform for plotting
plt.title('Top 15 Counties by Average House Price (2015-2024)')
plt.xlabel('Average Price (£)')
plt.ylabel('County')
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'top_counties_avg_price_eda.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

print("\nEDA complete. All plots saved to the 'reports/' directory.")


# --- Section 5: Machine Learning Model for Price Prediction ---
print("\n--- Section 5: Machine Learning Model for Price Prediction ---")

# 5.1 Data Preparation for ML
print("\n5.1 Data Preparation for ML:")

# Define features (X) and target (y)
X = df_filtered[['Property Type', 'Town', 'County', 'Old/New', 'Duration', 'year', 'month_of_transfer', 'day_of_week_transfer', 'is_post_covid']].copy()
y = df_filtered['Price'].copy() # Price is already log-transformed

# Identify categorical and numerical features
categorical_features = ['Property Type', 'Town', 'County', 'Old/New', 'Duration']
numerical_features = ['year', 'month_of_transfer', 'day_of_week_transfer', 'is_post_covid']

# Create a column transformer for preprocessing
# One-hot encode categorical features
# Standardize numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 5.2 Model Training (LightGBM Regressor)
print("\n5.2 Model Training (LightGBM Regressor):\n")
print("Justification for LightGBM:\n")
print("LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It's highly efficient, fast, and performs exceptionally well on large, tabular datasets like this one. Key advantages include:\n")
print("- **Speed and Efficiency:** LightGBM uses a novel technique called Gradient-based One-Side Sampling (GOSS) to filter out data instances with small gradients, and Exclusive Feature Bundling (EFB) to bundle mutually exclusive features, significantly speeding up training without losing accuracy.\n")
print("- **Accuracy:** It consistently delivers high accuracy, often outperforming other boosting algorithms like XGBoost on many tasks.\n")
print("- **Handles Categorical Features:** While we are explicitly One-Hot Encoding here for broader compatibility, LightGBM has native support for categorical features, which can be beneficial.\n")
print("- **Scalability:** It's designed to be distributed and can handle large datasets efficiently.\n")
print("- **Interpretability (Feature Importance):** Tree-based models like LightGBM provide feature importances, which are crucial for understanding which factors drive price predictions and for deriving policy recommendations.\n")

# Create the LightGBM model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31))
])

print("Training LightGBM model...")
model.fit(X_train, y_train)
print("Model training complete.")

# 5.3 Model Evaluation
print("\n5.3 Model Evaluation:\n")
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values for evaluation
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

# Get feature names after one-hot encoding and scaling
# This requires fitting the preprocessor first to get the transformed column names
preprocessor.fit(X_train)
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
scaled_numerical_features = [f'scaled_{col}' for col in numerical_features] # Prefix for scaled features
all_feature_names = encoded_feature_names + scaled_numerical_features

# Get feature importances from the trained LightGBM model
feature_importances = model.named_steps['regressor'].feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Top 15 Feature Importances:")
print(importance_df.head(15))

# Visualize feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importances for House Price Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'feature_importances.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

# 5.5 Prediction and Visualization
print("\n5.5 Prediction and Visualization:\n")

# Create a scatter plot of actual vs. predicted prices
plt.figure(figsize=(10, 10))
plt.scatter(y_test_original_scale, y_pred_original_scale, alpha=0.3)
plt.plot([y_test_original_scale.min(), y_test_original_scale.max()], [y_test_original_scale.min(), y_test_original_scale.max()], '--r', linewidth=2) # Perfect prediction line
plt.title('Actual vs. Predicted House Prices')
plt.xlabel('Actual Price (£)')
plt.ylabel('Predicted Price (£)')
plt.ticklabel_format(style='plain', axis='x')
plt.ticklabel_format(style='plain', axis='y')
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(reports_dir, 'actual_vs_predicted_prices.png')
plt.savefig(plot_path)
print(f"Saved plot: {plot_path}")
plt.close()

print("\nMachine Learning analysis complete. Review the printed metrics and generated plots for insights.")
