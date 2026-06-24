import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import load_dataset

st.set_page_config(page_title="Property Comparison", page_icon="⚖️", layout="wide")

st.markdown("## ⚖️ Property Comparison Dashboard")
st.markdown("Compare two properties side-by-side across pricing, layout, neighborhood safety, investment potential, and connectivity metrics.")

# Load database
try:
    df = load_dataset()
    
    # Selection Panels
    col_sel1, col_sel2 = st.columns(2)
    
    # Property A Selection
    with col_sel1:
        st.markdown("### 🏢 Property A")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            cities_a = sorted(df['City'].unique())
            city_a = st.selectbox("Select City A", options=cities_a, key="city_a")
        with col_a2:
            localities_a = sorted(df[df['City'] == city_a]['Locality'].unique())
            locality_a = st.selectbox("Select Locality A", options=localities_a, key="locality_a")
            
        subset_a = df[(df['City'] == city_a) & (df['Locality'] == locality_a)]
        if subset_a.empty:
            st.warning("No properties.")
            id_a = None
        else:
            prop_options_a = {f"ID: {row['ID']} | {row['BHK']} BHK {row['Property_Type']} (Listed: ₹{row['Price_in_Lakhs']:.1f}L)": row['ID'] for _, row in subset_a.head(30).iterrows()}
            label_a = st.selectbox("Select Property A ID", options=list(prop_options_a.keys()), key="label_a")
            id_a = prop_options_a[label_a]
            
    # Property B Selection
    with col_sel2:
        st.markdown("### 🏢 Property B")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            cities_b = sorted(df['City'].unique())
            city_b = st.selectbox("Select City B", options=cities_b, key="city_b")
        with col_b2:
            localities_b = sorted(df[df['City'] == city_b]['Locality'].unique())
            locality_b = st.selectbox("Select Locality B", options=localities_b, key="locality_b")
            
        subset_b = df[(df['City'] == city_b) & (df['Locality'] == locality_b)]
        if subset_b.empty:
            st.warning("No properties.")
            id_b = None
        else:
            prop_options_b = {f"ID: {row['ID']} | {row['BHK']} BHK {row['Property_Type']} (Listed: ₹{row['Price_in_Lakhs']:.1f}L)": row['ID'] for _, row in subset_b.head(30).iterrows()}
            label_b = st.selectbox("Select Property B ID", options=list(prop_options_b.keys()), key="label_b")
            id_b = prop_options_b[label_b]
            
    if id_a and id_b:
        prop_a = df[df['ID'] == id_a].iloc[0]
        prop_b = df[df['ID'] == id_b].iloc[0]
        
        st.markdown("---")
        
        # Display side-by-side comparison table
        st.markdown("### 📊 Metric Comparison Ledger")
        
        comparison_data = {
            'Metric Feature': [
                'Property ID', 'City', 'Locality', 'Property Type', 'BHK Layout', 'Size (SqFt)',
                'Listed Price (₹ Lakhs)', 'Fair Model Price (₹ Lakhs)', 'Furnished Status',
                'Risk Score (0-100)', 'Expected 3-Yr ROI (%)', 'Investment Recommendation'
            ],
            'Property A': [
                int(prop_a['ID']), prop_a['City'], prop_a['Locality'], prop_a['Property_Type'], int(prop_a['BHK']), f"{prop_a['Size_in_SqFt']:,} SqFt",
                f"₹{prop_a['Price_in_Lakhs']:.2f} L", f"₹{prop_a['Predicted_Price']:.2f} L", prop_a['Furnished_Status'],
                f"{prop_a['risk_score']:.1f} ({prop_a['risk_category']})", f"{prop_a['roi_3_years']:.1f}%", prop_a['recommendation']
            ],
            'Property B': [
                int(prop_b['ID']), prop_b['City'], prop_b['Locality'], prop_b['Property_Type'], int(prop_b['BHK']), f"{prop_b['Size_in_SqFt']:,} SqFt",
                f"₹{prop_b['Price_in_Lakhs']:.2f} L", f"₹{prop_b['Predicted_Price']:.2f} L", prop_b['Furnished_Status'],
                f"{prop_b['risk_score']:.1f} ({prop_b['risk_category']})", f"{prop_b['roi_3_years']:.1f}%", prop_b['recommendation']
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Radar/Spider Chart
        st.markdown("### 🕸️ Normalized Radar Comparison Chart")
        st.write(
            "This radar chart plots both properties across six normalized dimensions (0 to 100). "
            "For all dimensions (including safety, which is calculated as `100 - Risk Score`), **larger areas represent superior value**."
        )
        
        # Normalization logic
        # Price: lower is better/cheaper, but let's show affordability (100 - normalized_price)
        p_max = df['Price_in_Lakhs'].max()
        norm_afford_a = (1.0 - (prop_a['Price_in_Lakhs'] / p_max)) * 100
        norm_afford_b = (1.0 - (prop_b['Price_in_Lakhs'] / p_max)) * 100
        
        # Area (Size)
        s_max = df['Size_in_SqFt'].max()
        norm_size_a = (prop_a['Size_in_SqFt'] / s_max) * 100
        norm_size_b = (prop_b['Size_in_SqFt'] / s_max) * 100
        
        # BHK
        b_max = df['BHK'].max()
        norm_bhk_a = (prop_a['BHK'] / b_max) * 100
        norm_bhk_b = (prop_b['BHK'] / b_max) * 100
        
        # ROI
        roi_max = df['roi_3_years'].max()
        norm_roi_a = (prop_a['roi_3_years'] / roi_max) * 100
        norm_roi_b = (prop_b['roi_3_years'] / roi_max) * 100
        
        # Safety (100 - Risk Score)
        norm_safety_a = 100 - prop_a['risk_score']
        norm_safety_b = 100 - prop_b['risk_score']
        
        # Connectivity
        norm_conn_a = prop_a['connectivity_index']
        norm_conn_b = prop_b['connectivity_index']
        
        categories = ['Affordability', 'Physical Size', 'BHK Layout', '3-Yr ROI Potential', 'Neighborhood Safety', 'Transit Connectivity']
        
        fig_radar = go.Figure()
        
        # Add Property A trace
        fig_radar.add_trace(go.Scatterpolar(
            r=[norm_afford_a, norm_size_a, norm_bhk_a, norm_roi_a, norm_safety_a, norm_conn_a],
            theta=categories,
            fill='toself',
            name=f"Property A (ID: {prop_a['ID']})",
            line_color='#1f4068'
        ))
        
        # Add Property B trace
        fig_radar.add_trace(go.Scatterpolar(
            r=[norm_afford_b, norm_size_b, norm_bhk_b, norm_roi_b, norm_safety_b, norm_conn_b],
            theta=categories,
            fill='toself',
            name=f"Property B (ID: {prop_b['ID']})",
            line_color='#e43f5a'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=500,
            margin=dict(l=50, r=50, t=30, b=30),
            template='plotly_white'
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
except Exception as e:
    st.error(f"Error loading comparison engine: {e}")
