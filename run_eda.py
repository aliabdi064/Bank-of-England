import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# Convert 'Price' to numeric, coercing errors
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# --- 2.3 Clean and Format ---
print("\n--- Section 2.3: Data Cleaning and Formatting ---")

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

# Create new columns: 'year'
df_filtered['year'] = df_filtered['Date of Transfer'].dt.year
print("\n'year' column created.")

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
sns.lineplot(data=monthly_avg_price, x='YearMonth', y='Price')
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
# This is a simplified approach; a more robust method would involve
# mapping postcodes or using geographical boundaries.
london_counties = ['GREATER LONDON'] # 'GREATER LONDON' is often used for London in this dataset
north_east_counties = ['TYNE AND WEAR', 'DURHAM', 'NORTHUMBERLAND'] # Example counties in North East

# Filter data for these specific regions
df_london = df_filtered[df_filtered['County'].isin(london_counties)].copy()
df_north_east = df_filtered[df_filtered['County'].isin(north_east_counties)].copy()

print(f"\nLondon data points: {df_london.shape[0]}")
print(f"North East data points: {df_north_east.shape[0]}")

# Calculate average prices for the selected regions
avg_price_london = df_london['Price'].mean()
avg_price_north_east = df_north_east['Price'].mean()

print(f"\nAverage Price in London (2015-2024): £{avg_price_london:,.2f}")
print(f"Average Price in North East (2015-2024): £{avg_price_north_east:,.2f}")

# Visualize average prices for top N counties
plt.figure(figsize=(12, 8))
sns.barplot(x='Price', y='County', data=county_avg_price.head(15))
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
sns.boxplot(x='County', y='Price',
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
sns.lineplot(data=monthly_avg_price_regions, x='YearMonth', y='Price', hue='County')
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
sns.histplot(df_filtered['Price'], bins=50, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price (£)')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='x') # Prevent scientific notation on x-axis

plt.subplot(1, 2, 2)
sns.boxplot(y=df_filtered['Price'])
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
sns.boxplot(data=df_filtered, x='Property Type', y='Price', order=df_filtered['Property Type'].value_counts().index)
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
sns.boxplot(data=df_filtered, x='Old/New', y='Price')
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
sns.boxplot(data=df_filtered, x='Duration', y='Price')
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
sns.lineplot(data=avg_price_per_year, x='year', y='Price', marker='o')
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
sns.boxplot(data=df_filtered, x='is_post_covid', y='Price')
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
sns.barplot(x='Price', y='Town', data=top_towns)
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
sns.barplot(x='Price', y='County', data=top_counties)
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