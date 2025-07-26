# Bank of England Data Science Project: House Price Analysis and Prediction

## 1. Project Objective

This project aims to analyze property price data from HM Land Registry to gain insights into the UK housing market. Specifically, the analysis focuses on:

*   Understanding how house prices have changed since the COVID-19 pandemic (2020 onward).
*   Identifying and visualizing regional price trends across different areas of England and Wales.
*   Developing a machine learning model to predict house prices, demonstrating proficiency in data science methodologies.

## 2. Data Description

The dataset used is the "Price Paid Data" from HM Land Registry, containing transaction-level information for residential property sales in England and Wales. The key metrics in the dataset include:

*   **Transaction unique identifier**: A generated ID for each sale.
*   **Price**: The sale price stated on the transfer deed.
*   **Date of Transfer**: When the sale was completed, as stated on the deed.
*   **Property Type**: D (Detached), S (Semi‑Detached), T (Terraced), F (Flat/Maisonette), O (Other).
*   **Old/New**: Y = newly built, N = established dwelling.
*   **Duration**: Tenure: F = Freehold, L = Leasehold (only leases over seven years included).
*   **PAON/SAON/Street/Locality/Town/County**: Full address breakdown including postcode.
*   **PPD Category Type**: Categorises the transaction as A (Standard Price Paid) or B (Additional Price Paid Data).
*   **Record Status (monthly files only)**: A = addition, C = change, D = deletion, indicating updates to prior records.

## 3. Methodology

The project followed a standard data science workflow, implemented in Python using libraries such as `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `lightgbm`.

### 3.1 Data Loading and Cleaning

*   **Efficient Loading:** The large `House_Price_Full.csv` dataset was loaded efficiently in chunks (500,000 rows per chunk) to manage memory effectively. Only relevant columns (`Price`, `Date of Transfer`, `Property Type`, `Town`, `County`, `Old/New`, `Duration`) were selected.
*   **Data Type Conversion:** The `Price` column was converted to a numeric type, and `Date of Transfer` was converted to datetime objects.
*   **Filtering:** Records were filtered to include only transactions between 2015 and 2024.
*   **Feature Engineering:** New columns `year` (extracted from `Date of Transfer`) and `is_post_covid` (a boolean indicating transactions from March 1, 2020, onward) were created.

### 3.2 Exploratory Data Analysis (EDA)

Comprehensive EDA was performed to understand data distributions, identify patterns, and visualize relationships. Key aspects included:

*   **Data Overview:** Checking shape, data types, and missing values.
*   **Descriptive Statistics:** Summarizing numerical and categorical features.
*   **Distribution Analysis:** Visualizing the distribution of `Price`, `Property Type`, `Old/New`, and `Duration`.
*   **Relationship Analysis:** Exploring how `Price` relates to other features like `Property Type`, `Old/New`, `Duration`, `year`, and `is_post_covid`.
*   **Geographical Analysis:** Examining average prices across different `Towns` and `Counties`.

### 3.3 Machine Learning Model for Price Prediction

*   **Objective:** Predict the `Price` of a property (Regression problem).
*   **Model Choice:** **LightGBM Regressor** was chosen for its efficiency, speed, and strong performance on large tabular datasets. It handles categorical features well and provides valuable feature importances.
*   **Data Preparation:**
    *   Features (`Property Type`, `Town`, `County`, `Old/New`, `Duration`, `year`, `is_post_covid`) and the target (`Price`) were defined.
    *   Categorical features were transformed using One-Hot Encoding.
    *   The dataset was split into training (80%) and testing (20%) sets.
*   **Training & Evaluation:** The LightGBM model was trained on the training data and evaluated on the test set using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R2).
*   **Feature Importance:** The model's feature importances were analyzed to understand the key drivers of house prices.
*   **Prediction Visualization:** Actual vs. Predicted prices were plotted to visually assess model performance.

## 4. Key Findings and Visualizations

### 4.1 Data Overview and Distributions

*   The dataset contains over 10 million records for the 2015-2024 period after cleaning and filtering.
*   The `Price` distribution is highly skewed, with a long tail towards higher values, typical of real estate data. (See `reports/price_distribution.png`)
*   Terraced, Semi-Detached, and Detached properties are the most common types. (See `reports/property_type_distribution.png`)

### 4.2 House Price Changes Since COVID-19 (2020 Onward)

*   Analysis of monthly average prices clearly shows fluctuations and an overall upward trend since 2020, indicating the impact of the COVID-19 pandemic on the housing market. (See `reports/covid_price_trend.png`)
*   The `is_post_covid` feature was identified as an important factor in the machine learning model, further confirming the pandemic's influence. (See `reports/price_pre_post_covid.png`)

### 4.3 Regional Price Trends

*   Significant disparities in average house prices exist across different Counties and Towns. (See `reports/top_county_avg_price.png` and `reports/top_towns_avg_price.png`)
*   A comparison between London and the North East (using example counties like Tyne and Wear, Durham, Northumberland) revealed substantial differences in price distributions and trends over time. (See `reports/london_vs_ne_boxplot.png` and `reports/london_vs_ne_timeseries.png`)
    *   *Note:* Further investigation is needed to precisely map all London areas within the `County` column, as initial filtering for 'GREATER LONDON' yielded limited data points.

### 4.4 Machine Learning Model Performance

*   **Mean Absolute Error (MAE):** £170,705.87
*   **Root Mean Squared Error (RMSE):** £1,431,486.19
*   **R-squared (R2):** 0.1002

*   **Interpretation:** The R2 score of 0.1002 indicates that the model explains approximately 10% of the variance in house prices. While this suggests room for improvement, the model successfully identifies key influential factors. The high MAE and RMSE reflect the wide range of house prices and the inherent complexity of predicting them with the current feature set.

### 4.5 Feature Importance

*   The `year` of transfer was the most important feature, highlighting the temporal aspect of house price changes.
*   `Property Type` (especially 'Other' and 'Detached'), `Duration` (Freehold), and `is_post_covid` were also highly influential.
*   Specific `Towns` and `Counties` (e.g., London boroughs) appeared as significant predictors, emphasizing the importance of location. (See `reports/feature_importances.png`)

## 5. Policy Recommendations

Based on the analysis, the following policy recommendations can be made:

*   **Economic Sensitivity:** The strong influence of the `year` and `is_post_covid` features underscores the housing market's sensitivity to broader economic conditions and significant events. Policymakers should maintain vigilance over macroeconomic indicators and global developments, as they directly impact housing market stability. Policies related to interest rates, inflation control, and economic stimulus can significantly influence price trends.

*   **Supply-Side Dynamics:** The importance of `Property Type` and `Old/New` suggests that policies addressing the supply and demand for different housing types are crucial. This could involve initiatives to encourage the construction of specific property types or to manage the stock of existing homes to meet evolving market needs.

*   **Targeted Regional Policies:** The significant regional price disparities necessitate tailored housing policies. Interventions should consider local market conditions, potentially including targeted investment in infrastructure and job creation in more affordable regions to encourage population redistribution, or specific affordable housing programs in high-cost areas.

## 6. Conclusion

This project successfully demonstrated a comprehensive data science approach to analyzing house prices. While the predictive model's accuracy is limited by the available features, it provides valuable insights into the key drivers of house prices and their trends since 2015, including the impact of the COVID-19 pandemic. The findings offer a solid foundation for informing policy discussions aimed at fostering a stable and equitable housing market.

## 7. References

*   **Data Source:** HM Land Registry Price Paid Data (Accessed via `data/House_Price_Full.csv`)
*   **Codebase:** The full analysis code is available in `run_ml_analysis.py`.
*   **Visualizations:** All generated plots are located in the `reports/` directory.
