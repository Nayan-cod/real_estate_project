# Explainable AI (XAI) Report (Phase 5)

## 1. Executive Summary
To ensure complete transparency and model auditability, we implemented SHAP (SHapley Additive exPlanations) to explain the best performing model. SHAP values mathematically distribute the predicted price among the contributing features, showing exactly why a property is valued at its specific price.

## 2. Global Explainability Insights
- **Primary Price Driver:** The SHAP summary beeswarm plot shows that property size (`Size_in_SqFt`) is the most significant price driver. High values of size pull the SHAP value strongly positive (pushing price UP), while small sizes pull it negative.
- **Locality Premium:** `locality_target_enc` holds the second-highest global importance, proving that geographical micro-markets dictate real estate valuation.
- **Layout Spaciousness:** Interaction features like `area_x_bhk` show a high positive impact, indicating that layout spaciousness adds substantial value.
- **Infrastructure Connectivity:** Mapped variables like `connectivity_index` and `metro_distance_km` show significant global impacts. Properties with high connectivity indices command a premium.

## 3. Local Explanation Cards (Sample Properties)

### Low-Price (Budget) Segment Property
```
Property Price Predicted: 67.83 Lakhs (Actual: 65.33 Lakhs)
Reason:
 - BHK = 1.0 (DOWN (-2.0%)
 - furnished_status_score = 0.0 (DOWN (-1.3%)
 - Nearby_Schools = 1.0 (DOWN (-0.5%)
 - amenity_score_x_infra_growth = 3.6 (UP (+0.1%)
```

### Mid-Price Segment Property
```
Property Price Predicted: 207.97 Lakhs (Actual: 224.72 Lakhs)
Reason:
 - BHK = 5.0 (UP (+0.5%)
 - furnished_status_score = 1.0 (UP (+0.5%)
 - Nearby_Hospitals = 1.0 (DOWN (-0.5%)
 - has_airport = 0.0 (UP (+0.2%)
```

### High-Price (Luxury) Segment Property
```
Property Price Predicted: 509.77 Lakhs (Actual: 517.41 Lakhs)
Reason:
 - furnished_status_score = 0.0 (DOWN (-1.6%)
 - BHK = 4.0 (UP (+0.4%)
 - amenity_score_x_infra_growth = 10.4 (UP (+0.1%)
 - Nearby_Hospitals = 7.0 (UP (+0.1%)
```

## 4. Visual Artifacts Directory
The following high-resolution plots have been generated and saved to `outputs/plots/shap/`:
- `shap_beeswarm_summary.png` - Global feature impact distributions (beeswarm)
- `shap_feature_importance_bar.png` - Global ranked SHAP feature importances
- `shap_waterfall_budget.png` - Local waterfall explanation for the budget property
- `shap_waterfall_mid.png` - Local waterfall explanation for the mid-market property
- `shap_waterfall_luxury.png` - Local waterfall explanation for the luxury property
- `shap_vs_native_importance.png` - Side-by-side comparison of SHAP vs Native importances
