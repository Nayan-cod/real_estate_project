# Investment Recommendation and Risk Assessment Report (Phase 6)

## 1. Executive Summary
We successfully implemented the financial and risk scoring engines on top of our machine learning model. Across our entire database of 250,000 properties, **6143 properties (2.46%)** have been identified as high-conviction **BUY** opportunities. These properties represent undervalued assets in low-risk neighborhoods with strong social and physical infrastructures.

## 2. Risk Score Engine Weightings
Neighborhood and structural risk was evaluated on a **0 to 100 scale** using the following weighted formula:
- **Crime Index (35%):** Min-max scaled crime rate. Closer to 0 reduces risk.
- **Property Age (25%):** Mapped decay representing structural risks. Brand new properties have 0 age risk.
- **Market Volatility (20%):** Standard deviation of Price per SqFt in that locality. Low volatility indicates a stable market.
- **Demand/Liquidity (20%):** Local listing frequency. High listing volume implies high liquidity, reducing transaction risk.

### Risk Category Breakdown
- **Low Risk (Score < 35):** 34480 properties (13.79%)
- **Medium Risk (35 <= Score < 60):** 163115 properties (65.25%)
- **High Risk (Score >= 60):** 52405 properties (20.96%)

## 3. Dynamic ROI Projection Parameters
Expected annual price appreciation (growth rate $g$) was modeled based on city tier baseline growth rate, adjusted by individual property quality (attractiveness score):
- **Tier 1 Cities (High Growth):** 8.0% baseline appreciation
- **Tier 2 Cities (Stable Growth):** 6.0% baseline appreciation
- **Tier 3 Cities (Emerging Growth):** 5.0% baseline appreciation
- **Attractiveness Modifier:** Mapped linearly between -2% and +2% based on `investment_attractiveness_score`.

## 4. Recommendation Decision Matrix

| Recommendation | Undervalued Ratio | Risk Category | Expected 3-Yr ROI | Volume | Share |
|---|---|---|---|---|---|
| **BUY** | > 1.05 (Fair price > 5% above listed) | Low Risk (<35) | >= 15.0% | 6143 | 2.46% |
| **HOLD** | Stable / Moderate | Medium Risk (35-60) | Any | 234424 | 93.77% |
| **SELL** | < 0.95 (Listed price > 5% above fair) | High Risk (>=60) | Any | 9433 | 3.77% |

## 5. Visual Artifacts
The following plots have been generated and saved to `outputs/plots/recommendation/`:
- `risk_score_distribution.png` - Distribution of properties across risk buckets
- `recommendation_distribution.png` - Distribution of BUY/HOLD/SELL suggestions
- `sample_price_forecasts.png` - 5-year price trajectory curves for sample properties
