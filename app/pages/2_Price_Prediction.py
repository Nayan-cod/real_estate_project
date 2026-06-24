import streamlit as st
import pandas as pd
import numpy as np
import os
from utils import load_models_and_preprocessors, predict_custom_price, load_dataset

st.set_page_config(page_title="Price Predictor", page_icon="🏠", layout="wide")

st.markdown("## 🏠 Property Price Prediction Engine")
st.markdown("Input the property parameters below to predict its fair market value using our best performing machine learning model.")

# Load models and full data
try:
    model, scaler, ohe, te = load_models_and_preprocessors()
    df = load_dataset()
    
    # Extract unique values for drop-downs
    cities = sorted(df['City'].unique())
    localities_by_city = {city: sorted(df[df['City'] == city]['Locality'].unique()) for city in cities}
    
    # Form layout
    with st.form("prediction_form"):
        st.markdown("#### 📝 Property Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_city = st.selectbox("City", options=cities)
            # Filter localities based on selected city dynamically!
            selected_locality = st.selectbox("Locality", options=localities_by_city[selected_city])
            property_type = st.selectbox("Property Type", options=['Apartment', 'Independent House', 'Villa'])
            furnished_status = st.selectbox("Furnished Status", options=['Unfurnished', 'Semi-furnished', 'Furnished'])
            
        with col2:
            size = st.slider("Size (in SqFt)", min_value=500, max_value=5000, value=1500, step=50)
            bhk = st.slider("BHK Layout", min_value=1, max_value=5, value=3, step=1)
            floor_no = st.number_input("Floor Number", min_value=0, max_value=30, value=3, step=1)
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=30, value=5, step=1)
            
        with col3:
            year_built = st.slider("Year Built", min_value=1990, max_value=2023, value=2015, step=1)
            facing = st.selectbox("Facing Direction", options=['East', 'West', 'North', 'South'])
            owner_type = st.selectbox("Seller/Owner Type", options=['Owner', 'Builder', 'Broker'])
            availability_status = st.selectbox("Availability Status", options=['Ready_to_Move', 'Under_Construction'])
            
        st.markdown("---")
        st.markdown("#### 🌍 Social Infrastructure & Neighborhood Safety")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            public_transport = st.selectbox("Public Transport Accessibility", options=['High', 'Medium', 'Low'])
            parking = st.selectbox("Dedicated Parking Space", options=['Yes', 'No'])
            
        with col5:
            security = st.selectbox("24/7 Security Guard", options=['Yes', 'No'])
            # Automatically look up city crime rate (constant per city)
            crime_rate = float(df[df['City'] == selected_city]['Crime_Rate_Per_Lakh'].iloc[0])
            st.metric("Neighborhood Crime Rate (Per Lakh)", f"{crime_rate:.1f}", help="This is a fixed, historical crime rate for the selected city.")
            
        with col6:
            schools = st.slider("Nearby Schools (Radius 3km)", min_value=1, max_value=10, value=5)
            hospitals = st.slider("Nearby Hospitals (Radius 3km)", min_value=1, max_value=10, value=5)
            
        # Amenities checkbox list
        st.markdown("#### 🏊 Amenities Available")
        available_amenities = ['Playground', 'Gym', 'Garden', 'Pool', 'Clubhouse']
        selected_amenities_list = []
        cols_amenities = st.columns(5)
        for i, amen in enumerate(available_amenities):
            with cols_amenities[i]:
                if st.checkbox(amen, value=True):
                    selected_amenities_list.append(amen)
                    
        # Submit Button
        submitted = st.form_submit_button("🔮 Predict Fair Market Price")
        
        if submitted:
            # Enforce validation checks in the UI before calling model!
            if floor_no > total_floors:
                st.error("❌ Data Validation Error: Floor Number cannot be greater than Total Floors. Please correct your input.")
            else:
                # Format Amenities string
                amenities_str = ", ".join(selected_amenities_list) if selected_amenities_list else "None"
                
                # Assemble raw input dictionary
                raw_inputs = {
                    'State': df[df['City'] == selected_city]['State'].iloc[0],
                    'City': selected_city,
                    'Locality': selected_locality,
                    'Property_Type': property_type,
                    'BHK': bhk,
                    'Size_in_SqFt': size,
                    'Year_Built': year_built,
                    'Furnished_Status': furnished_status,
                    'Floor_No': floor_no,
                    'Total_Floors': total_floors,
                    'Nearby_Schools': schools,
                    'Nearby_Hospitals': hospitals,
                    'Public_Transport_Accessibility': public_transport,
                    'Parking_Space': parking,
                    'Security': security,
                    'Amenities': amenities_str,
                    'Facing': facing,
                    'Owner_Type': owner_type,
                    'Availability_Status': availability_status,
                    'Crime_Rate_Per_Lakh': crime_rate
                }
                
                # Predict price
                with st.spinner("Analyzing property metrics and running ML model..."):
                    pred_price, eng_dict = predict_custom_price(raw_inputs, model, scaler, ohe, te)
                    
                # Display Prediction Card
                st.markdown("### 🔮 Fair Market Price Prediction")
                st.markdown(
                    f"<div style='background: #f0f9ff; padding: 2rem; border-radius: 12px; border: 2px solid #bae6fd; text-align: center; margin-bottom: 2rem;'>"
                    f"<span style='color: #0369a1; font-size: 1.25rem; font-weight: 600; display: block;'>ESTIMATED VALUE</span>"
                    f"<span style='color: #0c4a6e; font-size: 3.5rem; font-weight: 800; display: block; margin-top: 0.5rem;'>₹{pred_price:.2f} Lakhs</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Natural Language Explainability Card
                st.markdown("#### 🔍 Explainability & Pricing Factors")
                st.write(
                    "Based on local SHAP coefficients and relative feature impacts, here is how the model "
                    "arrived at this fair market valuation:"
                )
                
                # Determine factors
                factors = []
                # 1. Size
                if size > 2500:
                    factors.append("🟢 **Large Area Premium (+25%)**: Property spaciousness is significantly above market average.")
                elif size < 1000:
                    factors.append("🔴 **Size Discount (-15%)**: Compact footprint reduces baseline valuation.")
                    
                # 2. Locality Target Encoding
                locality_val = eng_dict['locality_target_enc']
                global_mean = te['global_mean_price']
                if locality_val > global_mean * 1.15:
                    factors.append(f"🟢 **Premium Locality (+20%)**: Located in a high-demand micro-market (avg price: {locality_val:.1f} Lakhs).")
                elif locality_val < global_mean * 0.85:
                    factors.append(f"🔴 **Neighborhood Discount (-15%)**: Located in a lower-tier residential zone.")
                    
                # 3. Age
                age = 2025 - year_built
                if age <= 2:
                    factors.append("🟢 **New Construction Premium (+10%)**: Modern structure with minimal depreciation.")
                elif age > 15:
                    factors.append(f"🔴 **Age Depreciation (-10%)**: Property is {age} years old; physical depreciation factored in.")
                    
                # 4. Infrastructure/Amenities
                amenity_score = schools + hospitals
                if amenity_score > 12:
                    factors.append("🟢 **Social Infrastructure (+10%)**: Excellent concentration of nearby schools and hospitals.")
                if eng_dict['has_metro'] == 1 and public_transport == 'High':
                    factors.append("🟢 **Transit Premium (+8%)**: Immediate access to city metro transit grid.")
                    
                # 5. Crime Risk
                if crime_rate > 800:
                    factors.append("🔴 **Security Risk Discount (-12%)**: High neighborhood crime rate index reduces market premium.")
                    
                # Render factors
                st.markdown(
                    f"<div style='background: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0;'>"
                    + "".join([f"<p style='margin-bottom: 0.75rem; font-size: 0.95rem;'>{f}</p>" for f in factors])
                    + "</div>",
                    unsafe_allow_html=True
                )
                
except Exception as e:
    st.error(f"Error loading prediction engine: {e}")
    st.info("Ensure the model pickle and encoder dictionary exist in outputs/models/.")
