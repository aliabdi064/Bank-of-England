# Bank of England Data Science Project: UK Housing Market Analysis and Price Prediction

## Project Overview

This project undertakes a comprehensive data science analysis of the UK housing market, leveraging historical property transaction data from HM Land Registry. The primary goal is to uncover key trends, understand the impact of significant events like the COVID-19 pandemic, analyze regional disparities, and build a predictive model for house prices. This work aims to provide actionable insights for policymakers and stakeholders interested in the dynamics of the UK real estate sector.

## Key Objectives

*   **Analyze COVID-19 Impact:** Investigate how house prices have evolved since the onset of the COVID-19 pandemic (2020 onward).
*   **Identify Regional Trends:** Explore and visualize significant price variations across different regions of the UK (e.g., London vs. North East).
*   **Predict House Prices:** Develop a machine learning model to predict property prices based on various features.
*   **Derive Policy Insights:** Translate data-driven findings into clear, non-technical policy recommendations.

## Methodology

The project follows a robust data science methodology:

1.  **Efficient Data Loading & Preparation:** Handled large datasets by loading in chunks, selecting relevant features, and ensuring data quality through type conversions and handling missing values.
2.  **Feature Engineering:** Created new features (e.g., `year`, `is_post_covid`) to capture temporal trends and the impact of specific events.
3.  **Exploratory Data Analysis (EDA):** Conducted in-depth statistical analysis and visualization to understand data distributions, relationships between variables, and initial trends.
4.  **Machine Learning Modeling:** Implemented a **LightGBM Regressor** for house price prediction, chosen for its efficiency, scalability, and strong performance on tabular data.
5.  **Model Evaluation & Interpretation:** Assessed model performance using standard metrics (MAE, RMSE, R2) and analyzed feature importances to understand key price drivers.
6.  **Comprehensive Reporting:** Generated multiple reports tailored for different audiences, summarizing findings, technical details, and policy implications.

## Key Findings & Insights

*   **Post-COVID Price Surge:** The analysis clearly shows a significant and sustained increase in average house prices since early 2020, indicating a robust market response to the pandemic era.
*   **Pronounced Regional Disparities:** House prices vary dramatically across the UK, with London and the South East commanding significantly higher values compared to other regions. Location remains a dominant factor.
*   **Influential Factors:** Beyond time and location, property type (detached vs. flat), age (new build vs. established), and tenure (freehold vs. leasehold) are strong determinants of property value.
*   **Predictive Modeling:** Our LightGBM model, while demonstrating the complexity of house price prediction, successfully identified the most influential factors, with the year of sale and location being paramount.

## Project Structure

```
. 
├── data/                       # Contains the raw and processed datasets
│   └── House_Price_Full.csv
├── notebooks/                  # Jupyter notebooks for interactive analysis and development
│   └── data_exploration.ipynb
├── reports/                    # Generated reports (PDFs, Markdown) and visualizations (PNGs)
│   ├── data_scientist_exercise.pdf
│   └── *.png                   # All generated plots from EDA and ML analysis
├── .venv/                      # Python virtual environment for project dependencies
├── .gitignore                  # Specifies files and directories to be ignored by Git
├── requirements.txt            # Lists all Python package dependencies
├── REPORT.md                   # Main project report (comprehensive)
├── EDA_STORY_REPORT.md         # Narrative report on Exploratory Data Analysis
├── TECHNICAL_REPORT.md         # Detailed report on data preparation, feature engineering, and statistics
├── PREDICTION_MODEL_REPORT.md  # Specific report on the machine learning model
├── POLICY_BRIEFING.md          # Concise, non-technical briefing for policymakers
└── run_ml_analysis.py          # Python script to run the full analysis (data processing, EDA, ML)
```

## How to Run and Reproduce the Analysis

To reproduce this analysis and explore the project:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/aliabdi064/Bank-of-England.git
    cd Bank-of-England
    ```
2.  **Set up Python Environment:**
    *   Create a virtual environment:
        ```bash
        python3 -m venv .venv
        ```
    *   Activate the virtual environment:
        ```bash
        source .venv/bin/activate
        ```
    *   Install required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
3.  **Place Data:** Ensure the `House_Price_Full.csv` dataset is placed in the `data/` directory.
4.  **Run the Analysis Script:**
    ```bash
    .venv/bin/python run_ml_analysis.py
    ```
    This script will perform all data loading, cleaning, EDA, and machine learning, saving all generated plots to the `reports/` directory.
5.  **Explore Reports:** Read the Markdown reports (`.md` files) in the root directory and the `reports/` folder for detailed findings and visualizations.

## Reports & Visualizations

All detailed reports and visualizations are available in the repository:

*   **Main Project Report:** [`REPORT.md`](REPORT.md)
*   **EDA Story Report:** [`EDA_STORY_REPORT.md`](EDA_STORY_REPORT.md)
*   **Technical Report:** [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)
*   **Prediction Model Report:** [`PREDICTION_MODEL_REPORT.md`](PREDICTION_MODEL_REPORT.md)
*   **Policy Briefing:** [`POLICY_BRIEFING.md`](POLICY_BRIEFING.md)
*   **All Plots:** Located in the [`reports/`](reports/) directory.

## Contact

For any questions or further discussion, please contact [Your Name/Email/GitHub Profile Link].