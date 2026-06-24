import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import shap
from utils import load_models_and_preprocessors, load_dataset, preprocess_custom_input

st.set_page_config(page_title="Explainable AI", page_icon="🔍", layout="wide")

st.markdown("## 🔍 Explainable AI (SHAP) Dashboard")
st.markdown("Understand the inner workings of our machine learning model through SHAP (SHapley Additive exPlanations).")

# Tab structure: Global Explainability vs Local Explainability
tab1, tab2 = st.tabs(["🌎 Global Model Explanations", "🏠 Local Property Auditing"])

with tab1:
    st.markdown("### 🌎 Global Feature Impacts")
    st.write(
        "These visualizations illustrate how different features globally impact the model's price predictions. "
        "The SHAP summary beeswarm plot show the distribution of impacts (positive or negative) across features, "
        "while the bar chart shows the average absolute feature importance."
    )
    
    col1, col2 = st.columns(2)
    
    # Paths to pre-saved global SHAP plots
    beeswarm_path = 'outputs/plots/shap/shap_beeswarm_summary.png'
    importance_path = 'outputs/plots/shap/shap_feature_importance_bar.png'
    
    if not os.path.exists(beeswarm_path):
        beeswarm_path = '../outputs/plots/shap/shap_beeswarm_summary.png'
        importance_path = '../outputs/plots/shap/shap_feature_importance_bar.png'
        
    with col1:
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, caption="SHAP Summary Beeswarm Plot (Global Feature Distribution)", use_container_width=True)
        else:
            st.warning("Global beeswarm summary plot not found. Ensure Notebook 5 has been executed successfully.")
            
    with col2:
        if os.path.exists(importance_path):
            st.image(importance_path, caption="SHAP Feature Importance (Mean |SHAP Value|)", use_container_width=True)
        else:
            st.warning("Global feature importance bar plot not found. Ensure Notebook 5 has been executed successfully.")

with tab2:
    st.markdown("### 🏠 Local Property Pricing Audit")
    st.write(
        "Select a property from our database to run a live SHAP audit. The waterfall chart will illustrate "
        "exactly how the model started from the database-average price baseline and added/subtracted "
        "pricing increments to arrive at the final fair market valuation."
    )
    
    try:
        df = load_dataset()
        model, scaler, ohe, te = load_models_and_preprocessors()
        
        st.markdown("#### 🔍 Select Property to Audit")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            cities = sorted(df['City'].unique())
            sel_city = st.selectbox("Filter City", options=cities, key="te_city")
        with col_s2:
            localities = sorted(df[df['City'] == sel_city]['Locality'].unique())
            sel_locality = st.selectbox("Filter Locality", options=localities, key="te_locality")
        with col_s3:
            subset = df[(df['City'] == sel_city) & (df['Locality'] == sel_locality)]
            if subset.empty:
                st.warning("No properties found.")
                property_id = None
            else:
                property_options = {f"ID: {row['ID']} | {row['BHK']} BHK {row['Property_Type']} (Fair Price: ₹{row['Predicted_Price']:.1f}L)": row['ID'] for _, row in subset.head(50).iterrows()}
                selected_label = st.selectbox("Select Property ID", options=list(property_options.keys()), key="te_id")
                property_id = property_options[selected_label]
                
        if property_id:
            prop = df[df['ID'] == property_id].iloc[0]
            
            # Reconstruct raw inputs to process into scaled features
            raw_inputs = {
                'State': prop['State'],
                'City': prop['City'],
                'Locality': prop['Locality'],
                'Property_Type': prop['Property_Type'],
                'BHK': prop['BHK'],
                'Size_in_SqFt': prop['Size_in_SqFt'],
                'Year_Built': prop['Year_Built'],
                'Furnished_Status': prop['Furnished_Status'],
                'Floor_No': prop['Floor_No'],
                'Total_Floors': prop['Total_Floors'],
                'Nearby_Schools': prop['Nearby_Schools'],
                'Nearby_Hospitals': prop['Nearby_Hospitals'],
                'Public_Transport_Accessibility': prop['Public_Transport_Accessibility'],
                'Parking_Space': prop['Parking_Space'],
                'Security': prop['Security'],
                'Amenities': prop['Amenities'],
                'Facing': prop['Facing'],
                'Owner_Type': prop['Owner_Type'],
                'Availability_Status': prop['Availability_Status'],
                'Crime_Rate_Per_Lakh': prop['Crime_Rate_Per_Lakh']
            }
            
            # Preprocess to scaled features
            df_scaled, eng_dict = preprocess_custom_input(raw_inputs, te, ohe, scaler)
            
            # Display information
            left_col, right_col = st.columns([2, 1])
            
            with left_col:
                st.markdown("#### ⚡ Live SHAP Waterfall Plot")
                
                # Live SHAP computation wrapped in a spinner
                with st.spinner("Calculating SHAP values live for property..."):
                    try:
                        # Load final_test_predictions to act as background reference
                        test_predictions_path = 'data/processed/final_test_predictions.csv'
                        if not os.path.exists(test_predictions_path):
                            test_predictions_path = '../data/processed/final_test_predictions.csv'
                            
                        test_pred_df = pd.read_csv(test_predictions_path)
                        # Prepare background reference of 100 rows to speed up Tree SHAP
                        X_ref = test_pred_df.drop(columns=['Actual_Price', 'Predicted_Price']).sample(100, random_state=42)
                        
                        # Align columns
                        X_ref = X_ref[scaler.feature_names_in_]
                        # Scale background
                        X_ref_scaled = scaler.transform(X_ref)
                        
                        # Initialize explainer and calculate SHAP values
                        explainer = shap.Explainer(model, X_ref_scaled)
                        shap_values_single = explainer(df_scaled.values)
                        
                        # Generate Plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # We override the feature names on the shap object so they show up beautifully
                        shap_values_single.feature_names = df_scaled.columns.tolist()
                        shap.plots.waterfall(shap_values_single[0], show=False)
                        plt.title(f"SHAP Local Waterfall Audit (Property ID: {property_id})", fontsize=14, pad=20)
                        st.pyplot(fig)
                        plt.close(fig)
                        
                    except Exception as shap_err:
                        st.error(f"Could not compute SHAP live: {shap_err}")
                        st.info("Displaying local pricing factors card as fallback:")
                        
                        # Fallback Rule-Based pricing factors card
                        factors = []
                        size = prop['Size_in_SqFt']
                        if size > 2500:
                            factors.append("🟢 **Large Area Premium**: Physical spaciousness is significantly above average.")
                        elif size < 1000:
                            factors.append("🔴 **Size Discount**: Compact footprint reduces baseline valuation.")
                            
                        locality_val = prop['city_target_enc_demo'] if 'city_target_enc_demo' in prop else te['global_mean_price']
                        if locality_val > te['global_mean_price'] * 1.15:
                            factors.append(f"🟢 **Locality Premium**: Located in a premium city micro-market.")
                            
                        age = 2025 - prop['Year_Built']
                        if age <= 2:
                            factors.append("🟢 **New Construction Premium**: Modern structure with minimal depreciation.")
                        elif age > 15:
                            factors.append("🔴 **Age Depreciation**: Building age of 15+ years reduces physical structure value.")
                            
                        st.markdown(
                            f"<div style='background: #fff; padding: 1.5rem; border-radius: 8px; border: 1px solid #cbd5e1;'>"
                            + "".join([f"<p style='margin-bottom: 0.5rem;'>{f}</p>" for f in factors])
                            + "</div>",
                            unsafe_allow_html=True
                        )
                        
            with right_col:
                st.markdown("#### 📋 Valuation Reconciliation")
                st.write(
                    f"Audit ledger for property ID **{property_id}** in **{prop['Locality']}, {prop['City']}**:"
                )
                
                st.metric("Fair Predicted Price", f"₹{prop['Predicted_Price']:.2f} Lakhs")
                st.metric("Original Listed Price", f"₹{prop['Price_in_Lakhs']:.2f} Lakhs")
                
                ratio = prop['Predicted_Price'] / prop['Price_in_Lakhs']
                delta_str = f"{abs((ratio - 1)*100):.1f}% Undervalued" if ratio > 1.05 else f"{abs((1 - ratio)*100):.1f}% Overvalued" if ratio < 0.95 else "Fairly Valued"
                st.metric("Market Price Alignment", delta_str, delta=f"{ratio:.2f}x Ratio", delta_color="normal" if ratio > 1.05 else "inverse")
                
                st.write(
                    "💡 **How to interpret the waterfall chart:** The baseline value $E[f(X)]$ at the top represents the average log-price "
                    "prediction across the database. Each horizontal bar shows how much a specific feature (like size, locality, or age) "
                    "contributed to push the prediction up (red/right) or down (blue/left), summing up to the final predicted value $f(x)$ at the bottom."
                )
                
    except Exception as e:
        st.error(f"Error loading explainability tab: {e}")
