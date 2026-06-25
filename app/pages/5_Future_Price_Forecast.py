import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_dataset

st.set_page_config(page_title="Future Price Forecasts", page_icon="📈", layout="wide")

st.markdown("## 📈 Real Estate Price Forecasting Dashboard")
st.markdown("Visualize the projected price trajectory of properties over a 1-year, 3-year, and 5-year horizon based on locality growth CAGRs.")

# Load database
try:
    df = load_dataset()
    
    st.markdown("#### 🔍 Select Property to Forecast")
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
            property_options = {f"ID: {row['ID']} | {row['BHK']} BHK {row['Property_Type']} (Current: ₹{row['Price_in_Lakhs']:.1f}L)": row['ID'] for _, row in subset.head(50).iterrows()}
            selected_label = st.selectbox("Select Property ID", options=list(property_options.keys()))
            property_id = property_options[selected_label]
            
    if property_id:
        prop = df[df['ID'] == property_id].iloc[0]
        
        current_price = prop['Price_in_Lakhs']
        baseline_rate = prop['annual_growth_rate']
        city = prop['City']
        locality = prop['Locality']
        
        st.markdown("---")
        
        # Interactive "What-If" Appreciation Slider
        st.sidebar.markdown("### ⚙️ Forecast Calibration")
        custom_rate = st.sidebar.slider(
            "Annual appreciation CAGR (%)", 
            min_value=-5.0, 
            max_value=25.0, 
            value=float(baseline_rate * 100.0),
            step=0.5
        )
        
        # Convert slider % back to decimal
        appreciation_rate = custom_rate / 100.0
        
        # Calculate trajectories
        years = ['Year 0 (Current)', 'Year 1', 'Year 3', 'Year 5']
        prices = [
            current_price,
            current_price * (1 + appreciation_rate),
            current_price * (1 + appreciation_rate)**3,
            current_price * (1 + appreciation_rate)**5
        ]
        
        forecast_data = pd.DataFrame({
            'Timeline': years,
            'Projected Price (₹ Lakhs)': prices
        })
        
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            st.markdown("#### 📈 Price Appreciation Trajectory")
            
            # Interactive Line Chart using Plotly
            fig_line = px.line(
                forecast_data,
                x='Timeline',
                y='Projected Price (₹ Lakhs)',
                text=forecast_data['Projected Price (₹ Lakhs)'].apply(lambda x: f"₹{x:.1f}L"),
                markers=True,
                title=f"5-Year Valuation Forecast (at {custom_rate:.1f}% CAGR)",
                labels={'Projected Price (₹ Lakhs)': 'Valuation (Lakhs)'},
                template='plotly_white'
            )
            fig_line.update_traces(line_color='#e43f5a', line_width=3, marker_size=10, textposition="top center")
            fig_line.update_layout(height=400, yaxis_range=[current_price * 0.8, max(prices) * 1.2])
            st.plotly_chart(fig_line, use_container_width=True)
            
        with right_col:
            st.markdown("#### 📋 Forecast Ledger Table")
            st.write(
                f"Appreciation projections for property ID **{property_id}** in **{locality}, {city}**:"
            )
            
            # Formatted table
            table_df = forecast_data.copy()
            table_df['Projected Price (₹ Lakhs)'] = table_df['Projected Price (₹ Lakhs)'].apply(lambda x: f"₹{x:.2f} Lakhs")
            st.dataframe(table_df, hide_index=True, width="stretch")
            
            # Summary details
            net_gain = prices[-1] - current_price
            net_gain_pct = (prices[-1] / current_price - 1) * 100
            st.markdown(
                f"<div style='background: #f0fdf4; padding: 1.25rem; border-radius: 8px; border: 1px solid #bbf7d0; margin-top: 1.5rem;'>"
                f"<span style='color: #166534; font-size: 0.9rem; font-weight: 600; display: block;'>PROJECTED NET APPRECIATION</span>"
                f"<span style='color: #14532d; font-size: 1.75rem; font-weight: 800; display: block; margin-top: 0.25rem;'>+₹{net_gain:.2f} Lakhs</span>"
                f"<span style='color: #15803d; font-size: 0.9rem; font-weight: 600; display: block; margin-top: 0.25rem;'>Total gain of {net_gain_pct:.1f}% over 5 years.</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            
except Exception as e:
    st.error(f"Error rendering forecasting dashboard: {e}")
