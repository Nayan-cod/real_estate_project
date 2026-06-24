import streamlit as st
import pandas as pd
import os
from utils import load_dataset

# Set page configurations
st.set_page_config(
    page_title="Antigravity Real Estate Analytics Portal",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Shared CSS for professional aesthetics (vibrant color touches, sleek spacing)
st.markdown("""
<style>
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f4068, #162447, #e43f5a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid #f1f5f9;
        margin-bottom: 1.5rem;
    }
    .custom-card-header {
        font-weight: 600;
        font-size: 1.2rem;
        color: #0f172a;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .custom-card-body {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .tech-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Branding
with st.sidebar:
    st.markdown("### 🏠 Antigravity Portal")
    st.markdown("---")
    st.markdown("**Version:** 1.0.0 (Production)")
    st.markdown("**Core Framework:** Streamlit")
    st.markdown("**Pipeline Status:** ✅ All Stages Ready")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This portal is an interactive dashboard sitting on top of a highly optimized, end-to-end "
        "data science pipeline for property valuation, neighborhood risk assessment, and financial forecasting."
    )
    st.markdown("Developed by Google DeepMind Advanced Pair Programmers.")

# Main Header
st.markdown("<div class='main-header'>Real Estate Valuation & Investment Analytics Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A clean, reproducible, end-to-end machine learning pipeline & interactive business intelligence dashboard</div>", unsafe_allow_html=True)

# Load data for summary statistics
try:
    df = load_dataset()
    
    # Dashboard metrics
    total_properties = len(df)
    avg_price = df['Price_in_Lakhs'].mean()
    num_cities = df['City'].nunique()
    num_localities = df['Locality'].nunique()
    buy_percentage = (df['recommendation'] == 'BUY').mean() * 100
    
    st.markdown("### 📈 Full Database Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Properties Catalogued", value=f"{total_properties:,}")
    with col2:
        st.metric(label="Average Listed Price", value=f"₹{avg_price:.2f} Lakhs")
    with col3:
        st.metric(label="Cities Covered", value=f"{num_cities} (Tier 1-3)")
    with col4:
        st.metric(label="Undervalued BUY Options", value=f"{buy_percentage:.1f}%")
        
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.warning("Please ensure the notebook pipeline has run successfully and produced the processed CSV files.")

st.markdown("---")

# Main Content Layout
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("### 🎯 System Architecture & Overview")
    st.write(
        "The system combines a six-stage reproducible Jupyter notebook pipeline with an advanced rule-based "
        "investment recommendation engine. By analyzing structural property features, local neighborhood crime indexes, "
        "historical price-per-square-foot volatility, and social infrastructure density (schools/hospitals), the platform "
        "identifies high-conviction undervalued real estate assets."
    )
    
    # Cards detailing the pipeline
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<div class='custom-card-header'>⚙️ Reproducible Pipeline Stages</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='custom-card-body'>
        <ul>
            <li><b>Stage 1: Data Cleaning:</b> Validates 250k rows, swaps 116k floor violations mathematically without data loss, and caps outliers via winsorization.</li>
            <li><b>Stage 2: Exploratory Data Analysis:</b> 17 visual and statistical analyses (univariate, bivariate, multivariate) mapping baseline prices.</li>
            <li><b>Stage 3: Exhaustive Feature Engineering:</b> Creates 38+ columns including real-world geographical Metro & Airport proximity synthesis, risk indices, and layout density factors.</li>
            <li><b>Stage 4: Model Development:</b> Compares 6 tuned regression algorithms (Linear Regression, RF, XGBoost, CatBoost, LightGBM, GBR) using leak-free cross-validation.</li>
            <li><b>Stage 5: Explainable AI (SHAP):</b> Renders global beeswarm summaries and local waterfall cards to audit predictions.</li>
            <li><b>Stage 6: Recommendation & Forecasting:</b> Financial CAGR projections, risk score profiling (0-100), and BUY/HOLD/SELL vectorized engines.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("### 🛠️ Technology Stack")
    badges = [
        "Python 3.13", "Streamlit 1.40", "Pandas 2.3", "NumPy 2.3", "Scikit-Learn 1.8", 
        "XGBoost", "CatBoost", "LightGBM", "SHAP 0.52", "Plotly 6.8", "Joblib", "Matplotlib"
    ]
    st.markdown("".join([f"<span class='tech-badge'>{b}</span>" for b in badges]), unsafe_allow_html=True)
    
    st.markdown("<div class='custom-card' style='background: #fdf2f8; border-color: #fbcfe8;'>", unsafe_allow_html=True)
    st.markdown("<div class='custom-card-header' style='color: #be185d;'>💡 Quick Start Navigation</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='custom-card-body' style='color: #9d174d;'>
        Use the sidebar navigation to explore:
        <ol>
            <li><b>📊 EDA Dashboard:</b> Live interactive market visualization.</li>
            <li><b>🏠 Price Prediction:</b> Test the ML model on custom inputs.</li>
            <li><b>💰 Investment:</b> Find BUY, HOLD, and SELL property insights.</li>
            <li><b>⚠️ Risk Analysis:</b> Deep-dive neighborhood risk scoring.</li>
            <li><b>📈 Future Forecast:</b> Toggle growth CAGR projections.</li>
            <li><b>🔍 Explainable AI:</b> Inspect the SHAP mathematical audits.</li>
            <li><b>⚖️ Property Comparison:</b> Compare two properties side-by-side.</li>
            <li><b>🌍 Market Insights:</b> Geographical overview and top growth areas.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
