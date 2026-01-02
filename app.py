
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from india_yield_fetcher import IntelligentYieldFetcher
from yield_scheduler import YieldDataScheduler
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE INSTITUTIONAL FETCHERS (Per Quick Start Guide)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_system():
    fetcher = IntelligentYieldFetcher()
    scheduler = YieldDataScheduler()
    # Start scheduler in background if not already running
    try:
        scheduler.start()
    except:
        pass
    return fetcher, scheduler

fetcher, scheduler = init_system()

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD UI
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Institutional Yield Dashboard", layout="wide")

st.markdown("""
    <div style='background: linear-gradient(135deg, #003366 0%, #0066CC 100%); padding: 20px; border-radius: 15px; color: white; text-align: center;'>
        <h1>INTEREST RATE ANALYTICS TERMINAL</h1>
        <p>Powered by Intelligent Multi-Source Fetcher & Box-Jenkins Methodology</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("📡 Data Engine Control")
    benchmark = st.selectbox("Benchmark", ["India 10Y G-Sec", "US 10Y Treasury", "US 30Y Treasury"])
    
    st.header("⚙️ Model Parameters")
    horizon = st.slider("Forecast Horizon", 1, 60, 10)
    mode = st.radio("ARIMA Mode", ["Auto ARIMA", "Manual"])
    
    run_button = st.button("🚀 EXECUTE ANALYSIS")

# ═══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC: INTEGRATING YOUR FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════
if run_button:
    with st.spinner("Synchronizing Multi-Source Feeds..."):
        
        # 1. FETCH DATA
        if benchmark == "India 10Y G-Sec":
            # Use your custom IntelligentYieldFetcher logic
            result = fetcher.fetch_yield(use_cache=True, cache_ttl_hours=6)
            if result:
                val, source = result
                # Note: Fetcher returns current value, for ARIMA we need history
                # We pull history from FRED via the module's helper
                from india_yield_fetcher import FREDDataFetcher
                history = FREDDataFetcher.fetch_historical()
                source_label = f"Source: {source} (Multi-Source Engine)"
            else:
                st.error("All India sources failed. Check logs.")
                st.stop()
        else:
            # Standard US Treasury logic via yfinance
            ticker_map = {"US 10Y Treasury": "^TNX", "US 30Y Treasury": "^TYX"}
            history = yf.download(ticker_map[benchmark], period="5y")['Close']
            source_label = "Source: Yahoo Finance"

        # 2. RUN MODEL (Box-Jenkins)
        # Ensure data is stationary and clean
        series = history.dropna().resample('B').last().ffill()
        
        if mode == "Auto ARIMA":
            model = pm.auto_arima(series, seasonal=False)
            fc = model.predict(n_periods=horizon)
            order = model.order
        else:
            fit = ARIMA(series, order=(1,1,1)).fit()
            fc = fit.forecast(steps=horizon)
            order = (1,1,1)

        # 3. DISPLAY RESULTS
        st.subheader(f"📈 Yield Forecast: {benchmark}")
        st.caption(source_label)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical", line=dict(color="#003366")))
        
        f_dates = pd.date_range(series.index[-1], periods=horizon+1, freq='B')[1:]
        fig.add_trace(go.Scatter(x=f_dates, y=fc, name="ARIMA Forecast", line=dict(color="orange", width=3)))
        
        st.plotly_chart(fig, use_container_width=True)

        # 4. METRICS (BPS CALCULATOR)
        c1, c2, c3 = st.columns(3)
        current_rate = series.iloc[-1]
        fc_rate = fc.iloc[-1]
        bps_move = (fc_rate - current_rate) * 100
        
        c1.metric("Current Yield", f"{current_rate:.3f}%")
        c2.metric("Forecasted Yield", f"{fc_rate:.3f}%")
        c3.metric("BPS Movement", f"{bps_move:+.1f} bps")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>The Mountain Path - World of Finance | Prof. V. Ravichandran</p>", unsafe_allow_html=True)
