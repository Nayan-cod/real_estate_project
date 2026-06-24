import streamlit as st
import pandas as pd
import numpy as np
import os
from utils import load_dataset, estimate_roi_and_forecast, recommend_property

st.set_page_config(page_title="Investment Recommendations", page_icon="💰", layout="wide")

st.markdown("## 💰 Real Estate Investment Decision Engine")
st.markdown("This engine identifies high-conviction undervalued real estate assets by comparing actual listed prices against fair market model predictions.")

# Load database
try:
    df = load_dataset()
    
    # Toggle Input Mode
    mode = st.radio("Select Input Mode", options=["Select Property from Database", "Input Custom Property Parameters"])
    
    if mode == "Select Property from Database":
        st.markdown("#### 🔍 Select Property Lookup")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cities = sorted(df['City'].unique())
            sel_city = st.selectbox("Filter by City", options=cities)
        with col2:
            localities = sorted(df[df['City'] == sel_city]['Locality'].unique())
            sel_locality = st.selectbox("Filter by Locality", options=localities)
        with col3:
            # Filter properties by city and locality
            subset = df[(df['City'] == sel_city) & (df['Locality'] == sel_locality)]
            if subset.empty:
                st.warning("No properties found in this locality.")
                property_id = None
            else:
                property_options = {f"ID: {row['ID']} | {row['BHK']} BHK {row['Property_Type']} (Listed: ₹{row['Price_in_Lakhs']:.1f}L)": row['ID'] for _, row in subset.head(50).iterrows()}
                selected_label = st.selectbox("Select Property ID", options=list(property_options.keys()))
                property_id = property_options[selected_label]
                
        if property_id:
            # Fetch property row from database
            prop = df[df['ID'] == property_id].iloc[0]
            
            listed_p = prop['Price_in_Lakhs']
            predicted_p = prop['Predicted_Price']
            risk_s = prop['risk_score']
            roi_3y = prop['roi_3_years']
            rec = prop['recommendation']
            
            # Recompute reasons
            rec, reasons, risk_cat = recommend_property(listed_p, predicted_p, risk_s, roi_3y)
            
            # Display investment card
            st.markdown("---")
            st.markdown("### 📊 Investment Analysis Card")
            
            # Glowing Badge Style
            if rec == 'BUY':
                badge_style = "background: #dcfce7; border: 3px solid #22c55e; color: #166534; box-shadow: 0 0 15px rgba(34, 197, 94, 0.4);"
                badge_text = "BUY: UNDERVALUED HIGH-CONVICTION ASSET"
            elif rec == 'SELL':
                badge_style = "background: #fee2e2; border: 3px solid #ef4444; color: #991b1b; box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);"
                badge_text = "SELL: OVERVALUED HIGH-RISK ASSET"
            else:
                badge_style = "background: #fef9c3; border: 3px solid #eab308; color: #854d0e; box-shadow: 0 0 15px rgba(234, 179, 8, 0.2);"
                badge_text = "HOLD: FAIR VALUE STABLE ASSET"
                
            st.markdown(
                f"<div style='padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem; {badge_style}'>"
                f"<span style='font-size: 1.1rem; font-weight: 700; display: block; letter-spacing: 1px;'>DECISION ENGINE RECOMMENDATION</span>"
                f"<span style='font-size: 2.25rem; font-weight: 800; display: block; margin-top: 0.5rem;'>{badge_text}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Display financial metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Listed Price", f"₹{listed_p:.2f} L")
            with col_m2:
                st.metric("Fair Market Value", f"₹{predicted_p:.2f} L")
            with col_m3:
                st.metric("Expected 3-Yr ROI", f"{roi_3y:.1f}%")
            with col_m4:
                st.metric("Neighborhood Risk Score", f"{risk_s:.1f} / 100", delta=risk_cat, delta_color="inverse")
                
            st.markdown("#### 🔬 Contributing Rationale Factors")
            st.markdown(
                f"<div style='background: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0;'>"
                + "".join([f"<p style='margin-bottom: 0.75rem; font-size: 1rem;'> - {r}</p>" for r in reasons])
                + "</div>",
                unsafe_allow_html=True
            )
            
    else:
        st.markdown("#### 📝 Custom Property Parameters for Valuation")
        st.info("To compute a recommendation for a custom property, please enter the key financial parameters below:")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            listed_price_input = st.number_input("Listed Price of Property (₹ Lakhs)", min_value=10.0, max_value=1000.0, value=200.0)
            predicted_price_input = st.number_input("Predicted Fair Price from ML model (₹ Lakhs)", min_value=10.0, max_value=1000.0, value=220.0)
        with col_c2:
            custom_risk_score = st.slider("Neighborhood Risk Score (0-100)", min_value=0.0, max_value=100.0, value=28.0)
            custom_roi = st.slider("Projected 3-Year ROI (%)", min_value=0.0, max_value=100.0, value=18.0)
            
        custom_submit = st.button("🔮 Evaluate Investment Recommendation")
        
        if custom_submit:
            rec, reasons, risk_cat = recommend_property(listed_price_input, predicted_price_input, custom_risk_score, custom_roi)
            
            st.markdown("---")
            if rec == 'BUY':
                badge_style = "background: #dcfce7; border: 3px solid #22c55e; color: #166534; box-shadow: 0 0 15px rgba(34, 197, 94, 0.4);"
                badge_text = "BUY: UNDERVALUED HIGH-CONVICTION ASSET"
            elif rec == 'SELL':
                badge_style = "background: #fee2e2; border: 3px solid #ef4444; color: #991b1b; box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);"
                badge_text = "SELL: OVERVALUED HIGH-RISK ASSET"
            else:
                badge_style = "background: #fef9c3; border: 3px solid #eab308; color: #854d0e; box-shadow: 0 0 15px rgba(234, 179, 8, 0.2);"
                badge_text = "HOLD: FAIR VALUE STABLE ASSET"
                
            st.markdown(
                f"<div style='padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem; {badge_style}'>"
                f"<span style='font-size: 1.1rem; font-weight: 700; display: block; letter-spacing: 1px;'>DECISION ENGINE RECOMMENDATION</span>"
                f"<span style='font-size: 2.25rem; font-weight: 800; display: block; margin-top: 0.5rem;'>{badge_text}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Display financial metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Listed Price", f"₹{listed_price_input:.2f} L")
            with col_m2:
                st.metric("Fair Market Value", f"₹{predicted_price_input:.2f} L")
            with col_m3:
                st.metric("Expected 3-Yr ROI", f"{custom_roi:.1f}%")
            with col_m4:
                st.metric("Neighborhood Risk Score", f"{custom_risk_score:.1f} / 100", delta=risk_cat, delta_color="inverse")
                
            st.markdown("#### 🔬 Contributing Rationale Factors")
            st.markdown(
                f"<div style='background: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0;'>"
                + "".join([f"<p style='margin-bottom: 0.75rem; font-size: 1rem;'> - {r}</p>" for r in reasons])
                + "</div>",
                unsafe_allow_html=True
            )
            
except Exception as e:
    st.error(f"Error loading investment engine: {e}")
