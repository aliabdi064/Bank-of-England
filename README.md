# Bank of England Data Science Project: UK Housing Market Analysis and Price Prediction

## Project Overview

This project undertakes a comprehensive data science analysis of the UK housing market, leveraging historical property transaction data from HM Land Registry. The primary goal is to uncover key trends, understand the impact of significant events like the COVID-19 pandemic, analyze regional disparities, and build a predictive model for house prices. This work aims to provide actionable insights for policymakers and stakeholders interested in the dynamics of the UK real estate sector.

## Key Objectives

*   **Analyze COVID-19 Impact:** Investigate how house prices have evolved since the onset of the COVID-19 pandemic (2020 onward).
*   **Identify Regional Trends:** Explore and visualize significant price variations across different regions of the UK (e.g., London vs. North East).
*   **Predict House Prices:** Develop a machine learning model to predict property prices based on various features.
*   **Derive Policy Insights:** Translate data-driven findings into clear, non-technical policy recommendations.

## Project Structure

```
.
├── 0_POLICY_BRIEFING.md
├── 1_EXPLORATORY_DATA_ANALYSIS.md
├── 1_run_lightgbm_analysis.py
├── 2_DATA_PIPELINE_TECHNICAL_REPORT.md
├── 2_run_xgboost_experiment.py
├── 3_LIGHTGBM_MODEL_REPORT.md
├── 4_XGBOOST_COMPARISON_REPORT.md
├── README.md
├── data/
├── notebooks/
├── reports/
├── requirements.txt
└── .venv/
```

## How to Run and Reproduce the Analysis

To reproduce this analysis and explore the project:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/aliabdi064/Bank-of-England.git
    cd Bank-of-England
    ```
2.  **Set up Python Environment:**
    *   Create and activate a virtual environment:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   Install required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
3.  **Place Data:** Ensure the `House_Price_Full.csv` dataset is placed in the `data/` directory.
4.  **Run the Main Analysis Script:**
    ```bash
    .venv/bin/python 1_run_lightgbm_analysis.py
    ```
    This script will perform all data loading, cleaning, EDA, and machine learning for the primary LightGBM model, saving all generated plots to the `reports/` directory.

## Reports & Visualizations

All detailed reports and visualizations are available in the repository. The recommended reading order is:

*   **0_POLICY_BRIEFING.md:** A high-level summary for a non-technical audience.
*   **1_EXPLORATORY_DATA_ANALYSIS.md:** A narrative report on the initial data findings.
*   **2_DATA_PIPELINE_TECHNICAL_REPORT.md:** A detailed report on data preparation and feature engineering.
*   **3_LIGHTGBM_MODEL_REPORT.md:** A specific report on the primary machine learning model.
*   **4_XGBOOST_COMPARISON_REPORT.md:** A report on the comparative experiment with the XGBoost model.
*   **All Plots:** Located in the `reports/` directory.

## Contact

For any questions or further discussion, please contact .
