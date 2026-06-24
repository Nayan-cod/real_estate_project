import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Define lists of cities with operational Metro systems and commercial Airports
METRO_CITIES = {
    'Ahmedabad', 'Bangalore', 'Chennai', 'Dwarka', 'Faridabad', 'Gurgaon', 
    'Hyderabad', 'Jaipur', 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai', 'Nagpur', 
    'New Delhi', 'Noida', 'Pune'
}

AIRPORT_CITIES = {
    'Ahmedabad', 'Amritsar', 'Bangalore', 'Bhopal', 'Bhubaneswar', 'Bilaspur', 
    'Chennai', 'Coimbatore', 'Dehradun', 'Durgapur', 'Gaya', 'Guwahati', 
    'Hyderabad', 'Indore', 'Jaipur', 'Kochi', 'Kolkata', 'Lucknow', 'Mangalore', 
    'Mumbai', 'Mysore', 'Nagpur', 'New Delhi', 'Patna', 'Pune', 'Raipur', 
    'Ranchi', 'Silchar', 'Surat', 'Trivandrum', 'Vijayawada', 'Vishakhapatnam'
}

NON_AIRPORT_DISTANCES = {
    'Cuttack': 30.0, 'Dwarka': 20.0, 'Faridabad': 35.0, 'Gurgaon': 15.0, 
    'Haridwar': 40.0, 'Jamshedpur': 120.0, 'Ludhiana': 110.0, 'Noida': 35.0, 'Warangal': 140.0
}

@st.cache_resource
def load_models_and_preprocessors():
    """Load the trained model and all preprocessors once and cache them."""
    model_path = 'outputs/models/best_model.pkl'
    scaler_path = 'outputs/models/scaler.pkl'
    ohe_path = 'outputs/models/one_hot_encoder.pkl'
    te_path = 'outputs/models/target_encoders.pkl'
    
    # Fallback to absolute paths if running from a subfolder
    if not os.path.exists(model_path):
        model_path = '../outputs/models/best_model.pkl'
        scaler_path = '../outputs/models/scaler.pkl'
        ohe_path = '../outputs/models/one_hot_encoder.pkl'
        te_path = '../outputs/models/target_encoders.pkl'
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    ohe = joblib.load(ohe_path)
    te = joblib.load(te_path)
    
    return model, scaler, ohe, te

@st.cache_data
def load_dataset():
    """Load the final enriched dataset with recommendations."""
    paths = [
        'data/processed/final_dataset_with_recommendations.csv.gz',
        '../data/processed/final_dataset_with_recommendations.csv.gz',
        'data/processed/final_dataset_with_recommendations.csv',
        '../data/processed/final_dataset_with_recommendations.csv'
    ]
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("Could not find final_dataset_with_recommendations.csv or its compressed .gz version.")

def preprocess_custom_input(raw_inputs, te, ohe, scaler):
    """
    Takes a dictionary of raw user inputs, engineers all features in the exact
    same order as the training set, applies OHE and target encoding, and scales.
    Returns a DataFrame ready for model prediction.
    """
    df = pd.DataFrame([raw_inputs])
    
    # 1. Age & Condition Features
    df['Age_of_Property'] = 2025 - df['Year_Built']
    df['area_per_bhk'] = df['Size_in_SqFt'] / df['BHK']
    df['is_studio'] = ((df['BHK'] == 1) & (df['Size_in_SqFt'] < 800)).astype(int)
    
    # 2. Floor-related Features
    df['floor_ratio'] = df['Floor_No'] / df['Total_Floors']
    df['is_ground_floor'] = (df['Floor_No'] == 0).astype(int)
    df['is_top_floor'] = (df['Floor_No'] == df['Total_Floors']).astype(int)
    df['floors_remaining'] = df['Total_Floors'] - df['Floor_No']
    
    # 3. Age & Condition
    df['is_new_property'] = (df['Age_of_Property'] <= 2).astype(int)
    df['depreciation_factor'] = np.exp(-0.02 * df['Age_of_Property'])
    df['renovation_likely_flag'] = ((df['Age_of_Property'] > 20) & (df['Property_Type'] != 'Apartment')).astype(int)
    
    # 4. Proximity & Location Features (City Mappings)
    df['has_metro'] = df['City'].apply(lambda x: 1 if x in METRO_CITIES else 0)
    df['has_airport'] = df['City'].apply(lambda x: 1 if x in AIRPORT_CITIES else 0)
    
    # Synthesize realistic distances using Public_Transport_Accessibility
    acc = df['Public_Transport_Accessibility'].iloc[0]
    city = df['City'].iloc[0]
    
    if df['has_metro'].iloc[0] == 1:
        if acc == 'High': df['metro_distance_km'] = 0.7
        elif acc == 'Medium': df['metro_distance_km'] = 2.3
        else: df['metro_distance_km'] = 5.2
    else:
        df['metro_distance_km'] = 50.0
        
    if df['has_airport'].iloc[0] == 1:
        if acc == 'High': df['airport_distance_km'] = 10.0
        elif acc == 'Medium': df['airport_distance_km'] = 22.5
        else: df['airport_distance_km'] = 40.0
    else:
        df['airport_distance_km'] = NON_AIRPORT_DISTANCES.get(city, 80.0)
        
    df['connectivity_index'] = 100 * ((1 / (1 + df['metro_distance_km'])) * 0.6 + (1 / (1 + df['airport_distance_km'])) * 0.4)
    df['amenity_score'] = df['Nearby_Schools'] + df['Nearby_Hospitals']
    
    # We use global training medians for composite scoring at inference time
    # Crime rate median ~390.4, amenity score median ~11
    df['is_prime_location'] = ((df['Crime_Rate_Per_Lakh'] < 390.4) & (df['amenity_score'] > 11) & (df['has_metro'] == 1)).astype(int)
    
    # 5. Economic/Risk Features
    df['num_amenities'] = df['Amenities'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    
    # Crime index normalized (min 83.9, max 3192.4 based on training data)
    df['crime_index_normalized'] = (df['Crime_Rate_Per_Lakh'] - 83.9) / (3192.4 - 83.9)
    
    acc_score = df['Public_Transport_Accessibility'].map({'High': 1.0, 'Medium': 0.5, 'Low': 0.0}).iloc[0]
    sec_score = 1.0 if df['Security'].iloc[0] == 'Yes' else 0.0
    park_score = 1.0 if df['Parking_Space'].iloc[0] == 'Yes' else 0.0
    df['infra_growth_normalized'] = (acc_score * 0.4 + sec_score * 0.3 + park_score * 0.2 + (df['num_amenities'] / 5.0) * 0.1)
    
    df['composite_risk_score'] = 100 * (df['crime_index_normalized'] * 0.4 + (df['Age_of_Property'] / 35.0) * 0.3 + df['floor_ratio'] * 0.3)
    df['investment_attractiveness_score'] = 100 * (df['infra_growth_normalized'] * 0.4 + (df['amenity_score'] / 20.0) * 0.3 + (1.0 - df['crime_index_normalized']) * 0.3)
    
    # 6. Interaction Terms
    df['area_x_bhk'] = df['Size_in_SqFt'] * df['BHK']
    df['age_x_crime_index'] = df['Age_of_Property'] * df['crime_index_normalized']
    df['amenity_score_x_infra_growth'] = df['amenity_score'] * df['infra_growth_normalized']
    df['Size_in_SqFt^2'] = df['Size_in_SqFt'] ** 2
    df['Size_in_SqFt BHK'] = df['Size_in_SqFt'] * df['BHK']
    
    # 7. Encoding-ready Categorical
    df['furnished_status_score'] = df['Furnished_Status'].map({'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2})
    df['locality_frequency'] = 0.002  # Use average frequency for custom inputs
    df['is_ready_to_move'] = (df['Availability_Status'] == 'Ready_to_Move').astype(int)
    
    # 8. Target encoding mapping
    df['city_target_enc'] = df['City'].map(te['city_target_map']).fillna(te['global_mean_price'])
    df['locality_target_enc'] = df['Locality'].map(te['locality_target_map']).fillna(te['global_mean_price'])
    
    # 9. One-Hot Encoding nominals
    ohe_cols = ['Property_Type', 'Facing', 'Owner_Type', 'Availability_Status']
    ohe_arr = ohe.transform(df[ohe_cols])
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    
    # Concatenate OHE and drop original categories
    drop_cat_cols = ['State', 'City', 'Locality', 'Property_Type', 'Facing', 'Owner_Type', 'Availability_Status', 
                     'Amenities', 'Furnished_Status', 'Public_Transport_Accessibility', 'Parking_Space', 
                     'Security', 'floor_category', 'property_age_category', 'population_density_category', 'property_type_grouped', 'city_tier']
    
    df_features = df.drop(columns=[c for c in drop_cat_cols if c in df.columns], errors='ignore')
    df_features = pd.concat([df_features, ohe_df], axis=1)
    
    # Align features to ensure exact same columns as the scaler expects
    feature_list = scaler.feature_names_in_
    for col in feature_list:
        if col not in df_features.columns:
            df_features[col] = 0.0
            
    df_features = df_features[feature_list]
    
    # Scale
    df_scaled = pd.DataFrame(scaler.transform(df_features), columns=feature_list, index=df.index)
    return df_scaled, df.iloc[0].to_dict()

def predict_custom_price(raw_inputs, model, scaler, ohe, te):
    """Predict price in Lakhs for a dictionary of custom inputs."""
    df_scaled, engineered_dict = preprocess_custom_input(raw_inputs, te, ohe, scaler)
    pred_log = model.predict(df_scaled)
    pred_price = np.expm1(pred_log)[0]
    return pred_price, engineered_dict

def estimate_roi_and_forecast(price, attractiveness, city):
    """Calculate dynamic annual growth rate, ROI projections, and 5-year price forecasts."""
    # Mapped city tier (approximate baseline mapping for custom inputs)
    # Tier 1 cities: Delhi, Mumbai, Bangalore, Chennai, Hyderabad, Pune, Ahmedabad
    # Tier 2: Jaipur, Lucknow, Kochi, Coimbatore, Patna, Indore, Surat, Nagpur, Dehradun, Chandigarh, Bhubaneswar, Cuttack
    # Tier 3: all others
    tier1_cities = {'New Delhi', 'Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad'}
    tier2_cities = {'Jaipur', 'Lucknow', 'Kochi', 'Coimbatore', 'Patna', 'Indore', 'Surat', 'Nagpur', 'Dehradun', 'Bhubaneswar', 'Cuttack', 'Noida', 'Gurgaon', 'Faridabad', 'Dwarka'}
    
    if city in tier1_cities:
        baseline_rate = 0.08
        city_tier = 'Tier 1'
    elif city in tier2_cities:
        baseline_rate = 0.06
        city_tier = 'Tier 2'
    else:
        baseline_rate = 0.05
        city_tier = 'Tier 3'
        
    # Annual growth rate
    annual_rate = baseline_rate + ((attractiveness - 50) / 50.0) * 0.02
    
    # ROIs
    roi_1 = annual_rate * 100
    roi_3 = ((1 + annual_rate)**3 - 1) * 100
    roi_5 = ((1 + annual_rate)**5 - 1) * 100
    
    # Forecasts
    fc_1 = price * (1 + annual_rate)
    fc_3 = price * (1 + annual_rate)**3
    fc_5 = price * (1 + annual_rate)**5
    
    return {
        'city_tier': city_tier,
        'annual_growth_rate': annual_rate,
        'roi_1_year': roi_1,
        'roi_3_years': roi_3,
        'roi_5_years': roi_5,
        'forecast_1_year': fc_1,
        'forecast_3_years': fc_3,
        'forecast_5_years': fc_5
    }

def recommend_property(listed_price, predicted_price, risk_score, roi_3):
    """Generate BUY/HOLD/SELL recommendation based on financial indicators."""
    undervalued_ratio = predicted_price / listed_price
    
    # Risk category
    if risk_score < 35:
        risk_cat = 'Low Risk'
    elif risk_score < 60:
        risk_cat = 'Medium Risk'
    else:
        risk_cat = 'High Risk'
        
    if (undervalued_ratio > 1.05) and (risk_cat == 'Low Risk') and (roi_3 >= 15.0):
        rec = 'BUY'
        reasons = [
            f"Asset is **undervalued** by {((undervalued_ratio - 1)*100):.1f}% (Fair value: {predicted_price:.2f} Lakhs vs Listed: {listed_price:.2f} Lakhs).",
            f"Extremely **low risk neighborhood** (Neighborhood Risk Score: {risk_score:.1f} / 100).",
            f"Strong appreciation potential with a projected 3-year return of **{roi_3:.1f}%**."
        ]
    elif (undervalued_ratio < 0.95) and (risk_cat == 'High Risk'):
        rec = 'SELL'
        reasons = [
            f"Asset is **overvalued** by {((1 - undervalued_ratio)*100):.1f}% (Listed price: {listed_price:.2f} Lakhs is above fair value: {predicted_price:.2f} Lakhs).",
            f"High risk profile (Neighborhood Risk Score: {risk_score:.1f} / 100), driven by elevated crime or age.",
            "Weak risk-reward profile; exit or avoid recommended."
        ]
    else:
        rec = 'HOLD'
        reasons = [
            f"Asset is fairly valued (fair value: {predicted_price:.2f} Lakhs vs listed: {listed_price:.2f} Lakhs).",
            f"Balanced risk exposure (Neighborhood Risk Score: {risk_score:.1f} / 100, {risk_cat}).",
            f"Stable return profile with projected 3-year ROI of **{roi_3:.1f}%**."
        ]
        
    return rec, reasons, risk_cat
