Link: https://realestateproject-zddhpspex9vvwcrigkhmta.streamlit.app/

# Real Estate Valuation, Risk Profiling & Investment Recommendation Portal

An end-to-end, reproducible machine learning pipeline and interactive business intelligence dashboard designed to predict fair market property values, assess neighborhood safety risks, project expected CAGR returns, and generate BUY/HOLD/SELL recommendations.

The project features a **six-stage Jupyter notebook pipeline** that processes raw data, performs visual exploratory analysis, engineers advanced location features, trains six regression algorithms, runs SHAP mathematical explanations, and applies vectorized business engines. The results are served through a **multipage Streamlit web application**.

---

## 📂 Project Structure

```directory
real_estate_project/
├── data/
│   ├── raw/                              # Original raw dataset
│   │   └── merged_city_crime_data.csv    # Raw data (250k rows, 24 columns)
│   ├── interim/                          # Intermediate pipeline data
│   │   ├── cleaned_dataset.csv           # Outlier-capped & floor-swapped data
│   │   └── feature_engineered_dataset.csv # Exhaustive 38+ feature space data
│   └── processed/                        # Final modeling & prediction data
│       ├── final_dataset_with_recommendations.csv # Dashboard main database
│       ├── final_test_predictions.csv    # Test set predictions for auditing
│       └── model_comparison_table.csv    # Metrics ledger across all 6 models
├── notebooks/                            # Reproducible Jupyter Notebooks
│   ├── 01_data_cleaning.ipynb            # Phase 1: Data validation & winsorization
│   ├── 02_eda.ipynb                      # Phase 2: Visual & statistical exploration
│   ├── 03_feature_engineering.ipynb      # Phase 3: Exhaustive feature creation (Priority)
│   ├── 04_model_development.ipynb        # Phase 4: Model training, tuning & metrics
│   ├── 05_explainable_ai.ipynb           # Phase 5: Global/local SHAP audits
│   └── 06_recommendation_risk_forecast.ipynb # Phase 6: Financial ROI & recommendation engines
├── outputs/                              # Pipeline artifacts
│   ├── plots/                            # High-resolution (300 dpi) & interactive plots
│   │   ├── cleaning/                     # Pre- and post-capping boxplots
│   │   ├── eda/                          # Distributions, bivariate regressions, heatmaps
│   │   ├── feature_engineering/          # Pearson ranked correlation bar chart
│   │   ├── model/                        # RF importances, R2 comparisons, residuals
│   │   ├── shap/                         # beeswarm summary, Waterfall local cards
│   │   └── recommendation/               # Risk score, CAGR forecast curves, recommendation splits
│   ├── reports/                          # Phase-specific markdown summary reports (6 total)
│   │   ├── 01_cleaning_report.md
│   │   ├── 02_eda_report.md
│   │   ├── 03_feature_engineering_report.md
│   │   ├── 04_model_comparison_report.md
│   │   ├── 05_explainability_report.md
│   │   └── 06_recommendation_risk_forecast_report.md
│   └── models/                           # Serialized model binaries
│       ├── best_model.pkl                # Best performing regressor
│       ├── scaler.pkl                    # Standard Scaler
│       ├── one_hot_encoder.pkl           # Categorical encoder
│       └── target_encoders.pkl           # High-cardinality City/Locality map dicts
├── app/                                  # Multipage Streamlit Application
│   ├── app.py                            # Landing/home page
│   ├── utils.py                          # Shared prediction, feature, & financial helpers
│   └── pages/                            # Interactive dashboards (Streamlit auto-detected)
│       ├── 1_📊_EDA_Dashboard.py          # live market filtering & charts
│       ├── 2_🏠_Price_Prediction.py      # Custom property valuation form
│       ├── 3_💰_Investment_Recommendation.py # BUY/HOLD/SELLGlowing glowing badges
│       ├── 4_⚠️_Risk_Analysis.py          # Interactive Plotly gauge safety breakdowns
│       ├── 5_📈_Future_Price_Forecast.py  # 1/3/5 year CAGR what-if forecast calibration
│       ├── 6_🔍_Explainable_AI.py         # live SHAP waterfall property audits
│       ├── 7_⚖️_Property_Comparison.py   # Side-by-side comparative radars
│       └── 8_🌍_Market_Insights.py       # Ranked leaderboards and regional insights
├── requirements.txt                      # Pinned dependency requirements
└── test_pipeline.py                      # Pipeline validation test script
```

---

## 🚀 Execution Guide

### 1. Installation & Environment Setup
Ensure you have Python installed (v3.9 - v3.13 supported). Run the following commands from the project root directory:

```bash
# Install all required libraries
pip install -r requirements.txt
```

### 2. Running the Pipeline Notebooks
Each notebook is completely **independent**—it reads its input artifact from disk and saves its processed outputs back to disk. To run the pipeline in order, open your Jupyter environment and run the notebooks in `notebooks/`:

- Run `01_data_cleaning.ipynb` → produces `data/interim/cleaned_dataset.csv`
- Run `02_eda.ipynb` → generates exploratory plots in `outputs/plots/eda/`
- Run `03_feature_engineering.ipynb` → produces `data/interim/feature_engineered_dataset.csv`
- Run `04_model_development.ipynb` → produces serialized model binaries in `outputs/models/`
- Run `05_explainable_ai.ipynb` → generates SHAP beeswarm/waterfall plots in `outputs/plots/shap/`
- Run `06_recommendation_risk_forecast.ipynb` → produces final database `data/processed/final_dataset_with_recommendations.csv`

*(Note: You can also run them programmatically via terminal command: `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`)*

### 3. Launching the Streamlit Web Application
Once the pipeline has completed and produced the final database and model files, boot up the interactive dashboard with:

```bash
streamlit run app/app.py
```
This will automatically launch the portal in your default web browser (usually at `http://localhost:8501`).

---

## ⚙️ Core Methodologies & Feature Engineering

### A. Floor Validation & Swapping Strategy
We discovered that **116,304 rows** (46.52% of the raw data) suffered from swapped floor index columns where `Floor_No > Total_Floors`. Rather than discarding this data, we resolved the issue mathematically by setting:
- `Floor_No = min(Floor_No, Total_Floors)`
- `Total_Floors = max(Floor_No, Total_Floors)`
This guarantees `Floor_No <= Total_Floors` across the entire database, preventing data loss.

### B. Proximity & Infrastructure Synthesis (City Mappings)
Since the raw dataset lacks physical distance columns, we mapped the **42 unique cities** to their real-world Metro and Airport grid. We then combined this city-level mapping with the property's `Public_Transport_Accessibility` to synthesize realistic distance columns:
- `metro_distance_km`: Mapped from operational metro cities, ranging from 0.2km (High accessibility) to 7.0km (Low accessibility), or 50.0km for non-metro cities.
- `airport_distance_km`: Mapped from commercial airport metropolitan zones, ranging from 5.0km to 50.0km, or mapped directly to regional highway travel distances (e.g. Noida to Delhi Airport ~35km).
- `connectivity_index`: A weighted combination of transit options: $100 \times (0.6 \times \frac{1}{1 + \text{metro\_dist}} + 0.4 \times \frac{1}{1 + \text{airport\_dist}})$.

### C. Neighborhood Risk Score Engine (0-100)
Every property is evaluated on a 0 to 100 risk scale by combining neighborhood safety, building physical integrity, market volatility, and liquidity:
$$\text{Risk Score} = 100 \times \left( 0.35 \times \text{crime\_index\_normalized} + 0.25 \times \frac{\text{Age}}{35} + 0.20 \times \text{volatility\_normalized} + 0.20 \times (1.0 - \text{liquidity\_normalized}) \right)$$
- *Low Risk*: score < 35 | *Medium Risk*: 35 <= score < 60 | *High Risk*: score >= 60.

### D. ROI Projections & CAGRs
Valuation appreciation is modeled dynamically based on city tier growth baselines (Tier 1: 8.0%, Tier 2: 6.0%, Tier 3: 5.0%), adjusted by the individual property's attractiveness score ($\pm 2\%$). Future prices are projected using standard compound interest:
$$\text{Price}_{T} = \text{Price}_{0} \times (1 + g)^T$$

### E. Decision Matrix (BUY / HOLD / SELL)
- **BUY**: The property is **undervalued by > 5%** (fair prediction > listed price), is located in a **Low Risk neighborhood (<35)**, and has a **high projected return (3-Yr ROI >= 15%)**.
- **SELL**: The property is **overvalued by > 5%** and is located in a **High Risk neighborhood (>=60)**.
- **HOLD**: Stable, moderate-risk assets.
