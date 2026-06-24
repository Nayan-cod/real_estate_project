import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import load_dataset

st.set_page_config(page_title="Risk Profiler", page_icon="⚠️", layout="wide")

st.markdown("## ⚠️ Neighborhood Risk Profiler")
st.markdown("Assess the risk profile of properties based on crime rate, physical building age, neighborhood price volatility, and demand liquidity.")

# Load database
try:
    df = load_dataset()
    
    st.markdown("#### 🔍 Select Property to Analyze")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cities = sorted(df['City'].unique())
        sel_city = st.selectbox("Filter by City", options=cities)
    with col2:
        localities = sorted(df[df['City'] == sel_city]['Locality'].unique())
        sel_locality = st.selectbox("Filter by Locality", options=localities)
    with col3:
        subset = df[(df['City'] == sel_city) & (df['Locality'] == sel_locality)]
        if subset.empty:
            st.warning("No properties found.")
            property_id = None
        else:
            property_options = {f"ID: {row['ID']} | {row['BHK']} BHK {row['Property_Type']} (Risk Score: {row['risk_score']:.1f})": row['ID'] for _, row in subset.head(50).iterrows()}
            selected_label = st.selectbox("Select Property ID", options=list(property_options.keys()))
            property_id = property_options[selected_label]
            
    if property_id:
        prop = df[df['ID'] == property_id].iloc[0]
        
        risk_score = prop['risk_score']
        risk_category = prop['risk_category']
        crime_val = prop['Crime_Rate_Per_Lakh']
        age_val = 2025 - prop['Year_Built']
        volatility = prop['locality_volatility']
        frequency = prop['locality_frequency']
        
        st.markdown("---")
        
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.markdown("#### 📊 Risk Gauge")
            
            # Interactive Plotly Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Neighborhood Risk Score", 'font': {'size': 20, 'family': 'Outfit'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#1f4068"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 35], 'color': 'lightgreen'},
                        {'range': [35, 60], 'color': 'khaki'},
                        {'range': [60, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), template='plotly_white')
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Display summary card
            st.markdown(
                f"<div style='background: #f8fafc; padding: 1.25rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 1rem;'>"
                f"<h5 style='color: #0f172a; margin-top: 0;'>Risk Summary Card</h5>"
                f"<p style='margin-bottom: 0.5rem;'><b>Risk Segment:</b> {risk_category}</p>"
                f"<p style='margin-bottom: 0.5rem;'><b>Neighborhood Crime Index:</b> {crime_val:.1f} per lakh</p>"
                f"<p style='margin-bottom: 0.5rem;'><b>Property Age:</b> {age_val} years</p>"
                f"<p style='margin-bottom: 0;'><b>Market Volatility:</b> {volatility:.4f} Lakhs/SqFt</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            
        with right_col:
            st.markdown("#### 🔬 Contributing Risk Factors Breakdown")
            
            # Normalize risk components for comparison (0-100 scale)
            # Recreate normalized values for the specific property
            c_norm = (crime_val - 83.9) / (3192.4 - 83.9) * 100
            a_norm = (age_val / 35.0) * 100
            
            # Volatility normalized (localities min/max)
            vol_min, vol_max = df['locality_volatility'].min(), df['locality_volatility'].max()
            v_norm = (volatility - vol_min) / (vol_max - vol_min) * 100
            
            # Frequency normalized (localities min/max)
            freq_min, freq_max = df['locality_frequency'].min(), df['locality_frequency'].max()
            f_norm = (1.0 - (frequency - freq_min) / (freq_max - freq_min)) * 100
            
            risk_factors = {
                'Neighborhood Crime (35% wt)': c_norm,
                'Structural Age (25% wt)': a_norm,
                'Market Volatility (20% wt)': v_norm,
                'Lack of Demand/Liquidity (20% wt)': f_norm
            }
            
            risk_factors_df = pd.DataFrame(list(risk_factors.items()), columns=['Risk Component', 'Normalized Intensity (0-100)'])
            
            fig_bar = px.bar(
                risk_factors_df,
                x='Normalized Intensity (0-100)',
                y='Risk Component',
                orientation='h',
                color='Normalized Intensity (0-100)',
                color_continuous_scale=px.colors.sequential.Sunsetdark,
                title="Normalized Risk Component Intensities",
                labels={'Normalized Intensity (0-100)': 'Intensity Score (0-100)'},
                template='plotly_white'
            )
            fig_bar.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.write(
                "💡 **How to interpret this breakdown:** Each factor is normalized on a scale from 0 (lowest risk in database) "
                "to 100 (highest risk in database). The weighted sum of these four factors yields the final gauge score on the left."
            )
            
except Exception as e:
    st.error(f"Error rendering risk profiler: {e}")
