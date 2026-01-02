
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
# 1. HEADER REDESIGN
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
    .config-info {{
        background-color: #f0f2f6; padding: 10px; border-radius: 5px;
        border-left: 5px solid {GOLD}; margin-bottom: 20px; font-size: 0.9rem;
    }}
    </style>
    <div class="main-header">
        <h1 style="margin-bottom: 0;">INTEREST RATE FORECASTING DASHBOARD</h1>
        <h2 style="margin-top: 0; font-size: 1.3rem; opacity: 0.9;">Multi-Model (ARIMA, Vasicek, CIR) Institutional Terminal</h2>
        <p style="margin-top: 10px; font-weight: bold; font-size: 1.1rem;">Prof. V. Ravichandran</p>
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

# Pre-defined config string for model tabs
config_summary = f"**Current Run:** {ticker_label} | **Lookback:** {lookback}Y | **Horizon:** {horizon}D | **Alpha:** {conf_level}"

if run_btn:
    # ... [Data Fetching Engine with 6-step retry stays the same] ...
    
    # 📈 ARIMA TAB
    with tabs[1]:
        st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
        # ... [Plotting logic] ...

    # 🌪️ GARCH TAB
    with tabs[2]:
        st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
        # ... [Plotting logic] ...

    # 🎲 VASICEK TAB
    with tabs[3]:
        st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
        # ... [Simulation logic] ...

    # ☀️ CIR TAB
    with tabs[4]:
        st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
        # ... [Simulation logic] ...

    # 🧪 BACKTESTING
    with tabs[5]:
        st.markdown(f'<div class="config-info">{config_summary} | Walk-Forward: 30D</div>', unsafe_allow_html=True)
        # ... [Backtest logic] ...

    # ... [Metrics & Diagnostics Logic] ...

    # 📋 EXPORT TAB (Amended for all Models)
    with tabs[8]:
        st.subheader("📋 Institutional Quantitative Report")
        export_df = pd.DataFrame({
            "Date": f_dates.strftime('%Y-%m-%d'),
            "ARIMA": arima_fc.values,
            "Vasicek_Median": v_med,
            "CIR_Median": c_med,
            "Volatility_Risk": cond_vol.tail(len(f_dates)).values
        })
        st.dataframe(export_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        st.download_button("📥 Download Report (CSV)", export_df.to_csv(index=False).encode('utf-8'), f"{ticker}_analysis.csv")

# CALL MODULAR FOOTER
display_footer()
