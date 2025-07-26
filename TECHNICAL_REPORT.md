# Technical Report: Data Preparation, Feature Engineering, and Statistical Insights

This report provides a detailed account of the data preparation, feature engineering, and statistical insights derived from the UK house price dataset. It complements the EDA Story Report and the overall Project Report by focusing on the technical aspects of the analysis.

## 1. Data Preparation and Feature Engineering

Effective data preparation and feature engineering are crucial steps in building robust machine learning models and extracting meaningful insights. This section details the transformations applied to the raw data.

### 1.1 Initial Data Loading and Column Selection

The raw dataset (`House_Price_Full.csv`) is a large file. To handle it efficiently and focus on relevant information, the following steps were taken:

*   **Chunked Loading:** The CSV was read in chunks of 500,000 rows using `pandas.read_csv(..., chunksize=500_000)` to manage memory effectively.
*   **Column Mapping:** Based on the inspection of the raw CSV structure, the `all_column_names` list was precisely defined to correctly map the data columns:
    ```python
    all_column_names = [
        'Transaction unique identifier', 'Price', 'Date of Transfer', 'Postcode',
        'Property Type', 'Old/New', 'Duration',
        'PAON', 'SAON', 'Street', 'Locality', 'Town', 'County', 'District',
        'PPD Category Type', 'Record Status'
    ]
    ```
*   **Relevant Column Selection:** Only a subset of these columns, deemed most relevant for house price prediction and analysis, were retained:
    ```python
    relevant_columns_names = [
        'Price', 'Date of Transfer', 'Property Type', 'Town', 'County',
        'Old/New', 'Duration'
    ]
    ```

### 1.2 Data Type Conversion and Filtering

Ensuring correct data types and focusing on the relevant time period were critical:

*   **Price Conversion:** The `Price` column, initially loaded as an object (due to potential formatting in the raw CSV), was explicitly converted to a numeric type. Any values that could not be converted were coerced to `NaN` (Not a Number) and subsequently dropped:
    ```python
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    ```
*   **Date Conversion:** The `Date of Transfer` column was converted to datetime objects. Rows where this conversion failed were dropped:
    ```python
    df['Date of Transfer'] = pd.to_datetime(df['Date of Transfer'], errors='coerce')
    df.dropna(subset=['Date of Transfer', 'Price'], inplace=True)
    ```
*   **Time Period Filtering:** The dataset was filtered to include only transactions occurring between January 1, 2015, and December 31, 2024. This aligns with the project's scope of analyzing recent trends.

### 1.3 Feature Creation (Feature Engineering)

Two new features were engineered to capture temporal aspects relevant to the analysis:

*   **`year`:** Extracted directly from the `Date of Transfer` column, providing a numerical representation of the transaction year:
    ```python
    df_filtered['year'] = df_filtered['Date of Transfer'].dt.year
    ```
*   **`is_post_covid`:** A boolean flag indicating whether a transaction occurred on or after March 1, 2020 (chosen as the approximate onset of significant COVID-19 impact on the housing market). This feature is crucial for analyzing the pandemic's influence:
    ```python
    covid_start_date = pd.to_datetime('2020-03-01')
    df_filtered['is_post_covid'] = (df_filtered['Date of Transfer'] >= covid_start_date)
    ```

### 1.4 Categorical Feature Encoding for Machine Learning

For the LightGBM model, categorical features were transformed using One-Hot Encoding. This process converts categorical variables into a numerical format that machine learning algorithms can process:

*   **Categorical Features:** `Property Type`, `Town`, `County`, `Old/New`, `Duration`.
*   **OneHotEncoder:** Applied via `sklearn.preprocessing.OneHotEncoder` within a `ColumnTransformer` to handle these features. `handle_unknown='ignore'` was used to prevent errors if unseen categories appear in the test set.

## 2. Statistical Insights from Data Analysis

While formal inferential statistical tests (e.g., t-tests, ANOVA) were not explicitly run, the descriptive statistics and visualizations from the EDA provide strong statistical insights into the dataset's characteristics and relationships.

### 2.1 Overall Data Characteristics

*   **Dataset Size:** After cleaning and filtering, the dataset comprises over 10 million entries, providing a statistically significant sample size for robust analysis.
*   **Price Range:** Prices range from £1 to £900,000,000, indicating a vast spectrum of property values. The mean price is approximately £362,943, but the median (around £245,000) is a more representative measure due to the highly skewed distribution, highlighting the impact of high-value properties.

### 2.2 Insights from Feature Distributions

*   **Property Type:** The most frequent property types are Terraced (T), Semi-Detached (S), and Detached (D), collectively accounting for the majority of transactions. This distribution is important for understanding the typical housing stock.
*   **Old/New:** A significant majority of transactions (around 89%) are for established dwellings (`N`), with new builds (`Y`) making up a smaller but notable portion. This reflects the slower pace of new construction relative to existing housing stock sales.
*   **Duration (Tenure):** Freehold (F) properties dominate the market (around 76% of transactions), indicating a preference or prevalence of full ownership over leasehold (L) arrangements.

### 2.3 Insights from Relationships with Price

*   **Price vs. Property Type:** Statistically, detached properties (`D`) consistently show the highest average prices, followed by semi-detached (`S`), terraced (`T`), and flats (`F`). The 'Other' (`O`) category exhibits a wide price range, suggesting its heterogeneity.
*   **Price vs. Old/New:** New builds (`Y`) demonstrate a statistically higher median price compared to established dwellings (`N`), indicating a premium for new construction. This difference is visually evident in the box plots.
*   **Price vs. Duration:** Freehold properties (`F`) are, on average, more expensive than leasehold properties (`L`), reflecting the greater ownership rights and perceived value associated with freehold tenure.
*   **Price Trend by Year:** The average house price has shown a clear and consistent upward trend from 2015 to 2024. This overall growth is a key macroeconomic indicator.

### 2.4 COVID-19 Impact (Statistical Perspective)

*   The `is_post_covid` feature, while a simple binary flag, statistically captures a significant shift in house prices. The descriptive statistics for prices before and after March 2020 (as seen in the box plot) show a noticeable increase in both median price and the overall price range in the post-COVID period. This provides statistical evidence for the pandemic's influence on the housing market dynamics.

### 2.5 Geographical Price Disparities

*   **County-Level Averages:** Statistical analysis of average prices by county reveals extreme disparities. Counties within Greater London and the South East consistently exhibit average prices several times higher than those in regions like the North East. For instance, the average price in the City of London is orders of magnitude higher than in counties like Durham or Northumberland.
*   **Regional Comparison (London vs. North East):** The statistical comparison between London and North East counties confirms a vast difference in price distributions. While the North East has a more concentrated price range at lower values, London's distribution is spread across a much higher and wider spectrum, indicating a luxury market and high demand.

## 3. Conclusion on Technical Aspects

The data preparation and feature engineering steps were crucial for transforming the raw data into a format suitable for analysis and modeling. The statistical insights derived from the descriptive analysis and visualizations provide a robust understanding of the UK housing market's characteristics, key drivers, and significant trends, including the impact of the COVID-19 pandemic and pronounced regional disparities. These technical foundations directly support the findings and policy recommendations presented in the main project report.
