
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# INTEGRATION: Importing your custom modules
from india_yield_fetcher import IntelligentYieldFetcher, CacheManager
from yield_scheduler import YieldDataScheduler

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION (As per QUICK_START_INDIA_YIELD.md)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Institutional Yield Terminal", layout="wide")

@st.cache_resource
def init_institutional_engine():
    fetcher = IntelligentYieldFetcher()
    scheduler = YieldDataScheduler()
    try:
        scheduler.start() # Starts background automation (Monday 8AM/Daily 6PM)
    except:
        pass
    return fetcher, scheduler

fetcher, scheduler = init_institutional_engine()

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
    <div style='background: linear-gradient(135deg, #003366 0%, #0066CC 100%); padding: 25px; border-radius: 15px; color: white; text-align: center;'>
        <h1 style='margin:0;'>INTEREST RATE ANALYTICS TERMINAL</h1>
        <p style='margin:0;'>Powered by Intelligent Multi-Source Fetcher (Box-Jenkins Framework)</p>
    </div>
""", unsafe_allow_True=True)

with st.sidebar:
    st.header("📡 Live Data Feeds")
    benchmark = st.selectbox("Benchmark Instrument", ["India 10Y G-Sec", "US 10Y Treasury"])
    
    # SYSTEM STATUS: Pulling from your CacheManager
    if benchmark == "India 10Y G-Sec":
        cache_data = CacheManager.load_latest_yield()
        if cache_data:
            val, src = cache_data
            st.success(f"Latest India Yield: {val:.3f}%")
            st.caption(f"Source: {src}")
    
    st.header("⚙️ ARIMA Engine")
    horizon = st.slider("Forecast Horizon", 5, 60, 15)
    model_choice = st.radio("Optimization", ["Auto-ARIMA", "Standard (1,1,1)"])
    
    run_btn = st.button("🚀 EXECUTE FORECAST")

# ═══════════════════════════════════════════════════════════════════════════════
# CORE ANALYTICS TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Forecast View", "🧪 Backtesting", "📊 Yield Metrics", "📚 Educational Hub"])

if run_btn:
    with st.spinner("Fetching Institutional Data..."):
        
        # 1. Fetching Data using your Intelligent Fetcher
        if benchmark == "India 10Y G-Sec":
            # Primary: Fetch history for ARIMA (NSE Proxy)
            hist_data = yf.download("IN10Y.NS", period="5y", progress=False)['Close']
            # Fallback for current point from your cache
            current_val, source = fetcher.fetch_yield(use_cache=True)
        else:
            hist_data = yf.download("^TNX", period="5y", progress=False)['Close']

        # 2. Box-Jenkins Modeling (Data Prep & Selection)
        series = hist_data.dropna().resample('B').last().ffill()
        
        try:
            if model_choice == "Auto-ARIMA":
                model = pm.auto_arima(series, seasonal=False, suppress_warnings=True)
                forecast = model.predict(n_periods=horizon)
                order = model.order
            else:
                fit = ARIMA(series, order=(1,1,1)).fit()
                forecast = fit.forecast(steps=horizon)
                order = (1,1,1)

            # 3. Outputs
            with tabs[0]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index[-300:], y=series[-300:], name="Historical Yield", line=dict(color="#003366")))
                
                f_dates = pd.date_range(series.index[-1], periods=horizon+1, freq='B')[1:]
                fig.add_trace(go.Scatter(x=f_dates, y=forecast, name="Predicted Yield", line=dict(color="orange", width=3)))
                
                fig.update_layout(title=f"ARIMA {order} Forecast for {benchmark}", height=500)
                st.plotly_chart(fig, width="stretch")

            with tabs[2]:
                c1, c2, c3 = st.columns(3)
                curr_y = series.iloc[-1]
                fc_y = forecast.iloc[-1]
                bps_move = (fc_y - curr_y) * 100
                
                c1.metric("Current Rate", f"{curr_y:.3f}%")
                c2.metric("Term Forecast", f"{fc_y:.3f}%")
                c3.metric("Movement (bps)", f"{bps_move:+.1f} bps", delta_color="inverse")

        except Exception as e:
            st.error(f"Modeling Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EDUCATIONAL HUB (Citing your Guide)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📖 ARIMA Methodology Guide")
    st.info("Directly aligned with 'ARIMA Modeling: Comprehensive Q&A Guide' by Prof. V. Ravichandran.")
    
    
    
    st.markdown("""
    ### Institutional Workflow Implemented:
    * **Stage 1: Multi-Source Identification**: Using FRED (Authoritative) and Investing.com (Scraping) to ensure data continuity.
    * **Stage 2: Automatic Differencing**: Calculating 'I' component to achieve stationarity.
    * **Stage 3: Information Criteria**: Selecting model order based on AIC/BIC scores.
    * **Stage 4: Automated Scheduling**: Using `yield_scheduler.py` to refresh terminal data every 6 hours.
    """)
    st.success("Refer to the uploaded PDF (Final Version ARIMA_Modeling.pdf) for deep technical dives into Stage 4: Diagnostic Checking.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional Version 1.2</p>", unsafe_allow_html=True)
