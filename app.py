
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pmdarima as pm
from arch import arch_model 
import scipy.stats as stats
import time

# MODULAR IMPORTS
from content.about_text import ABOUT_CONTENT
from content.qa_text import QA_MASTERCLASS
from content.footer import display_footer

# ═══════════════════════════════════════════════════════════════════════════════
# 1. THEME & SIDEBAR CSS RESTORATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Institutional Risk & Yield Terminal", layout="wide")

CORPORATE_BLUE = "#002147" 
GOLD = "#FFD700"

st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, {CORPORATE_BLUE} 0%, #004b8d 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        margin-bottom: 2rem; border-bottom: 5px solid {GOLD};
    }}
    [data-testid="stSidebar"] {{ background-color: {CORPORATE_BLUE} !important; }}
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {{ color: white !important; font-weight: bold; }}
    .config-info {{
        background-color: #f0f2f6; padding: 10px; border-radius: 5px;
        border-left: 5px solid {GOLD}; margin-bottom: 20px; font-size: 0.9rem; color: {CORPORATE_BLUE};
    }}
    </style>
    <div class="main-header">
        <h1 style="margin-bottom: 0; color: white;">INTEREST RATE FORECASTING DASHBOARD</h1>
        <h2 style="margin-top: 0; font-size: 1.3rem; opacity: 0.9; color: white;">Multi-Model (ARIMA, Vasicek, CIR) Institutional Terminal</h2>
        <p style="margin-top: 10px; font-weight: bold; font-size: 1.1rem; color: {GOLD};">Prof. V. Ravichandran | The Mountain Path</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_label = st.selectbox("Benchmark Maturity", ["US 10Y (^TNX)", "US 30Y (^TYX)", "US 5Y (^FVX)"])
    ticker = ticker_label.split("(")[1].replace(")", "")
    lookback = st.slider("Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 30)
    conf_level = st.select_slider("Confidence Level (α)", options=[0.90, 0.95, 0.99], value=0.95)
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 ARIMA", "🌪️ GARCH", "🎲 Vasicek", "☀️ CIR", "🧪 Backtest", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A"])

with tabs[0]:
    st.header("📖 Institutional Methodology")
    st.write(ABOUT_CONTENT["intro"])
    st.info(ABOUT_CONTENT["workflow"])
    c1, c2, c3 = st.columns(3)
    c1.markdown(ABOUT_CONTENT["arima"])
    c2.markdown(ABOUT_CONTENT["vasicek"])
    c3.markdown(ABOUT_CONTENT["cir"])

if run_btn:
    # DATA FETCHING LOGIC...
    data = yf.Ticker(ticker).history(period=f"{lookback}y")
    if not data.empty:
        yields = data['Close'].resample('B').last().ffill()
        returns = 100 * yields.pct_change().dropna()
        
        # MODEL ENGINES...
        model_arima = pm.auto_arima(yields, seasonal=False)
        arima_fc = model_arima.predict(n_periods=horizon)
        f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
        
        # GARCH & STOCHASTIC CALCS...
        # [Insert existing Vasicek/CIR simulation logic here]
        
        config_summary = f"**Current Run:** {ticker_label} | **Lookback:** {lookback}Y | **Horizon:** {horizon}D"

        # POPULATE MODEL TABS...
        for i in range(1, 8):
            with tabs[i]: st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
            
        # [Insert existing Plotting/Metrics Logic]

        # 📋 EXPORT TAB (Fixed colorful & inclusive)
        with tabs[8]:
            st.subheader("📋 Institutional Quantitative Report")
            export_df = pd.DataFrame({
                "Date": f_dates.strftime('%Y-%m-%d'),
                "ARIMA": arima_fc.values,
                "Vasicek_Median": v_med, # Ensure v_med is calculated above
                "CIR_Median": c_med      # Ensure c_med is calculated above
            })
            st.dataframe(export_df.style.background_gradient(cmap='YlGnBu').format(precision=4), use_container_width=True)
            st.download_button("📥 Download Report (CSV)", export_df.to_csv(index=False).encode('utf-8'), "Analysis.csv")

with tabs[9]:
    for q, a in QA_MASTERCLASS:
        with st.expander(q): st.write(a)

display_footer()
