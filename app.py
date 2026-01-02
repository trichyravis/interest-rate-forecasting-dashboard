
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# INTEGRATION: Custom Institutional Modules
from india_yield_fetcher import IntelligentYieldFetcher, CacheManager
from yield_scheduler import YieldDataScheduler

# ═══════════════════════════════════════════════════════════════════════════════
# 1. INITIALIZATION (The Engine is already working in your logs!)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Institutional Yield Terminal", layout="wide")

@st.cache_resource
def init_institutional_engine():
    fetcher = IntelligentYieldFetcher()
    scheduler = YieldDataScheduler()
    try:
        scheduler.start() 
    except:
        pass
    return fetcher, scheduler

fetcher, scheduler = init_institutional_engine()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. UI BRANDING (CORRECTED TYPO)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
    <div style='background: linear-gradient(135deg, #003366 0%, #0066CC 100%); padding: 25px; border-radius: 15px; color: white; text-align: center;'>
        <h1 style='margin:0;'>INTEREST RATE ANALYTICS TERMINAL</h1>
        <p style='margin:0;'>The Mountain Path - World of Finance | Institutional Data Engine</p>
    </div>
""", unsafe_allow_html=True) # Fixed: changed from unsafe_allow_True

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR (This will now reappear)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📡 Live Data Feeds")
    benchmark = st.selectbox("Benchmark Instrument", ["India 10Y G-Sec", "US 10Y Treasury"])
    
    if benchmark == "India 10Y G-Sec":
        cache_data = CacheManager.load_latest_yield()
        if cache_data:
            val, src = cache_data
            st.success(f"Latest India Yield: {val:.3f}%")
            st.caption(f"Source: {src}")
    
    st.header("⚙️ ARIMA Engine")
    horizon = st.slider("Forecast Horizon", 5, 60, 15)
    run_btn = st.button("🚀 EXECUTE FORECAST")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA FETCHING & ARIMA
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Processing Multi-Source Data..."):
        if benchmark == "India 10Y G-Sec":
            hist_data = yf.download("IN10Y.NS", period="5y", progress=False)['Close']
        else:
            hist_data = yf.download("^TNX", period="5y", progress=False)['Close']

        if not hist_data.empty:
            series = hist_data.dropna().resample('B').last().ffill()
            try:
                model = pm.auto_arima(series, seasonal=False, suppress_warnings=True)
                forecast = model.predict(n_periods=horizon)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index[-250:], y=series[-250:], name="Historical", line=dict(color="#003366")))
                f_dates = pd.date_range(series.index[-1], periods=horizon+1, freq='B')[1:]
                fig.add_trace(go.Scatter(x=f_dates, y=forecast, name="Forecast", line=dict(color="orange", width=3)))
                
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Modeling Error: {e}")
