# Model Development and Comparison Report (Phase 4)

## 1. Executive Summary
We trained and evaluated six different regression models to predict real estate prices. A strict, leak-free preprocessing pipeline was implemented, and hyperparameter tuning was conducted via 3-fold cross-validation. The best performing model is **XGBoost** with an **R² of 0.9812** and an **RMSE of 14.49 Lakhs**.

## 2. Preprocessing & Leakage Prevention
- **Target Encoding:** Mapped `City` and `Locality` to their mean price. The mappings were fit **strictly on the training split** and applied to the test split to prevent leakage.
- **Feature Scaling:** A `StandardScaler` was fit on training features only and applied to transform both splits.
- **Target Transformation:** Models were trained on the log-transformed target (`log_price`) to handle skewness, and predictions were exponentiated back to the original Lakhs scale for metrics calculation.

## 3. Model Performance Comparison Table

| Model Family | MAE (Lakhs) | RMSE (Lakhs) | MSE | R² Score |
|---|---|---|---|---|
| Linear Regression | 14.00 | 20.90 | 436.76 | 0.9609 |
| Random Forest | 13.31 | 17.61 | 310.15 | 0.9722 |
| XGBoost | 10.80 | 14.49 | 209.82 | 0.9812 |
| CatBoost | 11.04 | 15.10 | 228.10 | 0.9796 |
| LightGBM | 10.88 | 14.58 | 212.72 | 0.9809 |
| Gradient Boosting | 12.46 | 16.82 | 282.92 | 0.9746 |

## 4. Feature Selection Insights
- Top predictors selected by Tree Importance: `Size_in_SqFt`, `area_x_bhk`, `city_target_enc`, `locality_target_enc`, `connectivity_index`, `composite_risk_score`.
- Shortlisting the top 15 features successfully retained over 98% of model performance compared to using the full feature set, resulting in highly efficient, deployment-ready estimators.

## 5. Visual Artifacts
The following plots have been generated and saved to `outputs/plots/model/`:
- `top_15_feature_importances.png` - Tree-based selector importances
- `model_r2_comparison_bar.png` - Bar comparison of R² scores
- `best_model_actual_vs_predicted.png` - Scatter fit plot for the best model
- `best_model_residuals.png` - Residual scatter plot verifying homoscedasticity
