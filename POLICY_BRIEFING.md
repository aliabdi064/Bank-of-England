# Policy Briefing: UK House Price Trends and Predictions
**Date:** July 26, 2025
**Subject:** Data-Driven Insights into the UK Housing Market (2015-2024)

---

### 1. Overview: Understanding the Housing Market with Data

To analyze the UK housing market, we used a comprehensive dataset from HM Land Registry covering over 10 million property sales in England and Wales from 2015 to 2024. 

**Data Preparation:**
*   **Dataset:** Included sale price, date, property type, age (new/old), tenure (freehold/leasehold), and location.
*   **Cleaning:** We processed this vast dataset for accuracy, removing duplicate entries and handling missing information to ensure our analysis was based on reliable data.
*   **Enrichment:** We added key indicators, such as the year of sale and a flag for the post-COVID-19 period (after March 2020), to sharpen our insights.

---

### 2. Key Findings & Visualisation

Our analysis revealed two critical trends: the significant impact of the COVID-19 pandemic and the stark regional differences in property values.

**Finding 1: The Post-COVID Price Surge**
Since early 2020, average house prices have shown a clear and sustained increase. This suggests that factors like changing living preferences, low interest rates, and government support schemes created a uniquely robust housing market.

**Finding 2: The North-South Price Divide**
Location remains the most significant driver of price. The visualisation below shows the dramatic difference in house prices between London and the North East, highlighting the deep-seated regional disparities across the country.

![House Price Distribution: London vs North East](reports/london_vs_ne_boxplot.png)
*This chart compares the range of house prices, showing that London's market operates on a completely different scale from that of the North East.*

---

### 3. Our Predictive Model: How We Forecast Prices

To understand *why* prices change, we built a predictive model using a machine learning technique called **LightGBM**, chosen for its accuracy and efficiency with large datasets.

*   **Approach:** The model learns patterns from the data, using features like location, property type, and date to predict a property's sale price.
*   **Performance:** Our model is highly effective, explaining over **60%** of the variation in house prices. This gives us strong confidence in its findings. We also tested a second model (XGBoost) to validate our approach, confirming that our primary model is the most reliable for this data.

---

### 4. Summary of Results: What Drives House Prices?

Our analysis confirms that house prices are primarily driven by three factors:
1.  **Economic Conditions & Time:** The year of sale and the post-COVID period are the strongest indicators, reflecting broad economic shifts.
2.  **Location, Location, Location:** A property's town and county remain paramount.
3.  **Property Attributes:** The type of property (e.g., detached vs. flat), its age, and tenure (freehold vs. leasehold) are also key drivers.

---

### 5. Recommendations & Future Work

**Policy Recommendations:**
*   **Monitor Economic Indicators:** Policies related to interest rates and economic stimulus will continue to have a major impact on housing. Proactive monitoring is essential.
*   **Address Regional Disparities:** The price gap requires tailored regional strategies, such as targeted investment in infrastructure and jobs in more affordable regions.
*   **Focus on Housing Supply:** Policy should consider the supply of different property types to meet evolving demand.

**Suggestions for Further Analysis:**
*   **More Granular Data:** Incorporating data like full postcodes or proximity to schools and transport would allow for more localized and precise insights.
*   **External Economic Factors:** Integrating data on interest rates, inflation, and local economic growth would create an even more powerful predictive model.

---

### 6. Challenges of Analysis at Scale

Performing this analysis on over 10 million records presented several challenges:
*   **Memory and Processing:** Handling such a large dataset required efficient programming techniques (loading data in chunks) to avoid overwhelming computer memory.
*   **Computational Cost:** Training a machine learning model on this scale is time-consuming and requires significant computing power.
*   **Maintaining Timeliness:** To provide up-to-date advice, this analysis would need to be re-run regularly, requiring an automated and robust data pipeline to handle new data as it is released.

Addressing these challenges is crucial for providing timely and continuous data-driven intelligence for policymaking.