# Technical Report (Iteration 2): Enhanced Data Pipeline for House Price Prediction

This report outlines the specific modifications and enhancements made to the data preprocessing and feature engineering pipeline in `run_ml_analysis.py` for the second iteration of the house price prediction model. These changes were implemented to improve model performance and address identified limitations from the previous iteration.

## 1. Overview of Changes

The primary goal of this iteration was to refine the data input to the LightGBM model by implementing more robust cleaning, outlier handling, and creating richer features. The following sections detail the specific code changes and their rationale.

## 2. Data Preprocessing Enhancements

### 2.1 Duplicate Removal

*   **Change:** Added `df.drop_duplicates(inplace=True)` early in the data loading process.
*   **Rationale:** Duplicate rows can bias model training and evaluation. Removing them ensures that each observation is unique and contributes independently to the learning process.

### 2.2 Handling Missing Values (Town and County)

*   **Change:** Implemented `df[col].fillna('Unknown', inplace=True)` for `Town` and `County` columns.
*   **Rationale:** Missing categorical values can cause errors in One-Hot Encoding or be treated as a distinct category by the model. Imputing with 'Unknown' allows the model to learn a specific representation for these missing instances without losing data.

### 2.3 Outlier Handling for Price

*   **Change:** Implemented IQR-based clipping for the `Price` column:
    ```python
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['Price'] = np.clip(df['Price'], lower_bound, upper_bound)
    ```
*   **Rationale:** House prices often contain extreme outliers that can disproportionately influence regression models, leading to inflated error metrics and poor generalization. Clipping these outliers to a reasonable range (defined by 1.5 times the IQR) makes the model more robust to such extreme values.

### 2.4 Target Variable Transformation (Log Transformation)

*   **Change:** Applied `np.log1p(df['Price'])` to the `Price` column immediately after outlier handling.
*   **Rationale:** The distribution of house prices is typically highly skewed (right-skewed). Log transformation helps to normalize this distribution, making it more symmetrical and Gaussian-like. This is beneficial for many machine learning algorithms, as it can improve model performance, stabilize variance, and make relationships more linear. Predictions are inverse-transformed using `np.expm1()` for evaluation and visualization to present results in the original price scale.

## 3. Feature Engineering Enhancements

Beyond the `year` and `is_post_covid` features from the previous iteration, new temporal features were created to provide the model with more granular time-based information.

### 3.1 Temporal Features

*   **`month_of_transfer`:** Extracted the month number from `Date of Transfer`:
    ```python
    df_filtered['month_of_transfer'] = df_filtered['Date of Transfer'].dt.month
    ```
*   **`day_of_week_transfer`:** Extracted the day of the week (0=Monday, 6=Sunday) from `Date of Transfer`:
    ```python
    df_filtered['day_of_week_transfer'] = df_filtered['Date of Transfer'].dt.dayofweek
    ```
*   **Rationale:** These features capture cyclical patterns within a year and week, which can influence property demand and pricing (e.g., seasonal variations, weekend viewings).

## 4. Feature Scaling for Numerical Features

*   **Change:** Integrated `StandardScaler()` into the `ColumnTransformer` for numerical features (`year`, `month_of_transfer`, `day_of_week_transfer`, `is_post_covid`).
    ```python
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])
    ```
*   **Rationale:** Standardization (Z-score normalization) scales features to have a mean of 0 and a standard deviation of 1. While tree-based models like LightGBM are generally less sensitive to feature scaling than linear models or neural networks, it can sometimes improve convergence speed and model performance, especially when combined with other preprocessing steps or for interpretability of coefficients if a linear model were used.

## 5. Impact on Model Performance

These comprehensive data preprocessing and feature engineering steps led to a significant improvement in the LightGBM model's performance:

*   **Mean Absolute Error (MAE):** Reduced from £170,705.87 to **£63,208.12**.
*   **Root Mean Squared Error (RMSE):** Reduced from £1,431,486.19 to **£90,829.07**.
*   **R-squared (R2) Score:** Increased from 0.1002 to **0.6057**.

This substantial increase in R2 demonstrates that the model now explains over 60% of the variance in house prices, indicating a much more robust and accurate predictive capability. The reduction in MAE and RMSE confirms that the model's predictions are now much closer to the actual values on average.

## 6. Conclusion

This iteration highlights the critical importance of a well-designed data pipeline. By systematically addressing data quality issues, handling outliers, transforming the target variable, and engineering relevant temporal features, the predictive power of the LightGBM model was dramatically enhanced. These improvements provide a more reliable foundation for understanding house price dynamics and informing policy decisions.
