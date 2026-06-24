import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_dataset

st.set_page_config(page_title="EDA Market Dashboard", page_icon="📊", layout="wide")

st.markdown("## 📊 Real Estate Market Exploratory Dashboard")
st.markdown("Use this dashboard to visually and statistically analyze the distribution of properties, regional pricing trends, and feature correlations.")

# Load data
try:
    df = load_dataset()
    
    # Sidebar Filtering Panel
    st.sidebar.markdown("### 🔍 Filter Properties")
    
    # City filter
    all_cities = sorted(df['City'].unique())
    selected_cities = st.sidebar.multiselect("Select Cities", options=all_cities, default=all_cities[:3])
    
    # BHK filter
    all_bhks = sorted(df['BHK'].unique())
    selected_bhks = st.sidebar.multiselect("Select BHK Layouts", options=all_bhks, default=all_bhks)
    
    # Price slider
    min_p, max_p = float(df['Price_in_Lakhs'].min()), float(df['Price_in_Lakhs'].max())
    price_range = st.sidebar.slider("Price Range (₹ Lakhs)", min_value=min_p, max_value=max_p, value=(min_p, max_p))
    
    # Apply filters
    filtered_df = df[
        (df['City'].isin(selected_cities)) &
        (df['BHK'].isin(selected_bhks)) &
        (df['Price_in_Lakhs'] >= price_range[0]) &
        (df['Price_in_Lakhs'] <= price_range[1])
    ]
    
    if filtered_df.empty:
        st.warning("No properties match the selected filter criteria. Please broaden your selection.")
    else:
        st.markdown(f"**Properties showing:** `{len(filtered_df):,}` of `{len(df):,}` total records.")
        
        # Row 1: Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Column Distribution Histogram")
            dist_col = st.selectbox("Select Column to Visualize", options=['Price_in_Lakhs', 'Size_in_SqFt', 'Age_of_Property', 'Crime_Rate_Per_Lakh', 'connectivity_index'])
            
            fig_hist = px.histogram(
                filtered_df.sample(min(len(filtered_df), 5000), random_state=42),
                x=dist_col,
                color='BHK',
                marginal='box',
                nbins=30,
                title=f"Distribution of {dist_col} (Sample of 5,000 max)",
                color_discrete_sequence=px.colors.qualitative.Safe,
                template='plotly_white'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col2:
            st.markdown("#### 🔗 Interactive Correlation Heatmap")
            # Select key numeric features
            corr_cols = ['Price_in_Lakhs', 'Size_in_SqFt', 'BHK', 'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals', 'connectivity_index', 'risk_score', 'investment_attractiveness_score']
            corr_matrix = df[corr_cols].corr()
            
            fig_heat = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap (All Database Records)",
                aspect='auto',
                template='plotly_white'
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        st.markdown("---")
        
        # Row 2: Bivariate & Categorical Plots
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### 📐 Size vs Price Scatter")
            fig_scatter = px.scatter(
                filtered_df.sample(min(len(filtered_df), 1500), random_state=42),
                x='Size_in_SqFt',
                y='Price_in_Lakhs',
                color='BHK',
                size='Nearby_Schools',
                hover_data=['City', 'Locality', 'Property_Type'],
                title="Property Size vs Price (Sample of 1,500 max)",
                labels={'Size_in_SqFt': 'Size (SqFt)', 'Price_in_Lakhs': 'Price (Lakhs)'},
                color_continuous_scale=px.colors.sequential.Plasma,
                template='plotly_white'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col4:
            st.markdown("#### 🏙️ Regional Average Pricing")
            avg_prices = filtered_df.groupby(['City', 'BHK'])['Price_in_Lakhs'].mean().reset_index()
            fig_bar = px.bar(
                avg_prices,
                x='City',
                y='Price_in_Lakhs',
                color='BHK',
                barmode='group',
                title="Average Price per BHK per City",
                labels={'Price_in_Lakhs': 'Average Price (Lakhs)'},
                color_continuous_scale=px.colors.sequential.Viridis,
                template='plotly_white'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
except Exception as e:
    st.error(f"Error rendering dashboard: {e}")
    st.info("Ensure the pipeline has run and data is saved in data/processed/final_dataset_with_recommendations.csv.")
