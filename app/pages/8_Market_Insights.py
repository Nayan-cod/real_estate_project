import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_dataset

st.set_page_config(page_title="Market Insights", page_icon="🌍", layout="wide")

st.markdown("## 🌍 Regional Market Insights Dashboard")
st.markdown("Identify macro and micro-market leaders across investment appeal, capital growth, and neighborhood safety.")

# Load database
try:
    df = load_dataset()
    
    # City Filter Widget
    st.sidebar.markdown("### 🔍 Filter Insights")
    all_cities = sorted(df['City'].unique())
    selected_city = st.sidebar.selectbox("Select City of Interest", options=["All Cities"] + all_cities)
    
    # Apply city filter
    if selected_city != "All Cities":
        city_df = df[df['City'] == selected_city]
        scope_str = f"in **{selected_city}**"
    else:
        city_df = df
        scope_str = "across **All Cities**"
        
    st.markdown(f"Displaying top investment sectors {scope_str}:")
    
    # Calculations: group by Locality and compute averages
    # We filter out localities with very few properties (e.g. < 5) to avoid statistical anomalies
    loc_counts = city_df['Locality'].value_counts()
    valid_localities = loc_counts[loc_counts >= 5].index.tolist()
    
    loc_summary = city_df[city_df['Locality'].isin(valid_localities)].groupby('Locality').agg({
        'City': 'first',
        'investment_attractiveness_score': 'mean',
        'roi_3_years': 'mean',
        'risk_score': 'mean',
        'Price_in_Lakhs': 'mean'
    }).reset_index()
    
    # Tab layout for different leaders
    tab1, tab2, tab3 = st.tabs(["💎 Top Investment Sectors", "🚀 High Growth Zones", "🛡️ Safest Havens"])
    
    with tab1:
        st.markdown(f"#### 💎 Top 10 Localities by Investment Attractiveness {scope_str}")
        st.write(
            "Ranked by **Investment Attractiveness Score** (a weighted index of infrastructure quality, "
            "amenity density, and low crime rates)."
        )
        
        top_inv = loc_summary.sort_values(by='investment_attractiveness_score', ascending=False).head(10)
        
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            # Display Table
            table_df = top_inv.copy().rename(columns={
                'Locality': 'Locality Name',
                'investment_attractiveness_score': 'Attractiveness (0-100)',
                'Price_in_Lakhs': 'Avg Price (Lakhs)',
                'roi_3_years': 'Avg 3-Yr ROI'
            })
            table_df['Attractiveness (0-100)'] = table_df['Attractiveness (0-100)'].apply(lambda x: f"{x:.1f}")
            table_df['Avg Price (Lakhs)'] = table_df['Avg Price (Lakhs)'].apply(lambda x: f"₹{x:.2f}L")
            table_df['Avg 3-Yr ROI'] = table_df['Avg 3-Yr ROI'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(table_df[['Locality Name', 'City', 'Attractiveness (0-100)', 'Avg Price (Lakhs)', 'Avg 3-Yr ROI']], hide_index=True, width="stretch")
            
        with col_t2:
            # Display Plotly Bar
            fig_inv = px.bar(
                top_inv,
                x='investment_attractiveness_score',
                y='Locality',
                orientation='h',
                color='investment_attractiveness_score',
                color_continuous_scale=px.colors.sequential.Tealgrn,
                title="Investment Attractiveness Leaderboard",
                labels={'investment_attractiveness_score': 'Attractiveness Score', 'Locality': 'Locality'},
                template='plotly_white'
            )
            fig_inv.update_layout(coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_inv, use_container_width=True)
            
    with tab2:
        st.markdown(f"#### 🚀 Top 10 High Growth Localities {scope_str}")
        st.write(
            "Ranked by **Projected 3-Year capital gains ROI %** (CAGR-based dynamic appreciation rates)."
        )
        
        top_growth = loc_summary.sort_values(by='roi_3_years', ascending=False).head(10)
        
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            # Display Table
            table_df = top_growth.copy().rename(columns={
                'Locality': 'Locality Name',
                'roi_3_years': 'Projected 3-Yr ROI (%)',
                'Price_in_Lakhs': 'Avg Price (Lakhs)',
                'investment_attractiveness_score': 'Attractiveness Score'
            })
            table_df['Projected 3-Yr ROI (%)'] = table_df['Projected 3-Yr ROI (%)'].apply(lambda x: f"{x:.1f}%")
            table_df['Avg Price (Lakhs)'] = table_df['Avg Price (Lakhs)'].apply(lambda x: f"₹{x:.2f}L")
            table_df['Attractiveness Score'] = table_df['Attractiveness Score'].apply(lambda x: f"{x:.1f}")
            st.dataframe(table_df[['Locality Name', 'City', 'Projected 3-Yr ROI (%)', 'Avg Price (Lakhs)', 'Attractiveness Score']], hide_index=True, width="stretch")
            
        with col_g2:
            # Display Plotly Bar
            fig_growth = px.bar(
                top_growth,
                x='roi_3_years',
                y='Locality',
                orientation='h',
                color='roi_3_years',
                color_continuous_scale=px.colors.sequential.Electric,
                title="Capital Growth ROI Leaderboard (%)",
                labels={'roi_3_years': 'Projected 3-Year ROI (%)', 'Locality': 'Locality'},
                template='plotly_white'
            )
            fig_growth.update_layout(coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_growth, use_container_width=True)
            
    with tab3:
        st.markdown(f"#### 🛡️ Top 10 Safest Neighborhoods {scope_str}")
        st.write(
            "Ranked by **Lowest Neighborhood Risk Score** (weighted index of safety, crime, and physical stability)."
        )
        
        top_safety = loc_summary.sort_values(by='risk_score', ascending=True).head(10)
        
        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            # Display Table
            table_df = top_safety.copy().rename(columns={
                'Locality': 'Locality Name',
                'risk_score': 'Risk Score (Lower is Safer)',
                'Price_in_Lakhs': 'Avg Price (Lakhs)',
                'roi_3_years': 'Avg 3-Yr ROI'
            })
            table_df['Risk Score (Lower is Safer)'] = table_df['Risk Score (Lower is Safer)'].apply(lambda x: f"{x:.1f}")
            table_df['Avg Price (Lakhs)'] = table_df['Avg Price (Lakhs)'].apply(lambda x: f"₹{x:.2f}L")
            table_df['Avg 3-Yr ROI'] = table_df['Avg 3-Yr ROI'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(table_df[['Locality Name', 'City', 'Risk Score (Lower is Safer)', 'Avg Price (Lakhs)', 'Avg 3-Yr ROI']], hide_index=True, width="stretch")
            
        with col_s2:
            # Display Plotly Bar
            fig_safety = px.bar(
                top_safety,
                x='risk_score',
                y='Locality',
                orientation='h',
                color='risk_score',
                color_continuous_scale=px.colors.sequential.Mint_r,
                title="Neighborhood Safety Leaderboard (Lower is Safer)",
                labels={'risk_score': 'Risk Score (0-100)', 'Locality': 'Locality'},
                template='plotly_white'
            )
            fig_safety.update_layout(coloraxis_showscale=False, yaxis={'categoryorder':'total descending'}) # Descending because lower risk is better
            st.plotly_chart(fig_safety, use_container_width=True)
            
    st.markdown("---")
    st.markdown("#### 🗺️ Geographical Overview Map")
    st.info(
        "ℹ️ **Geographical Map Note:** The raw dataset `merged_city_crime_data.csv` does not contain latitude "
        "and longitude columns. Therefore, the interactive geographical overview map is gracefully omitted "
        "in favor of the comprehensive, high-resolution regional tables and leaderboards displayed above."
    )
    
except Exception as e:
    st.error(f"Error loading market insights: {e}")
