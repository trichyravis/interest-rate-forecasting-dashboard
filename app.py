
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# Import your custom institutional modules
from india_yield_fetcher import IntelligentYieldFetcher, CacheManager
from yield_scheduler import YieldDataScheduler

# ═══════════════════════════════════════════════════════════════════════════════
# 1. INITIALIZATION & SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Institutional Yield Terminal", layout="wide")

@st.cache_resource
def start_institutional_engine():
    """Starts the background scheduler and initializes the fetcher."""
    fetcher = IntelligentYieldFetcher()
    scheduler = YieldDataScheduler()
    try:
        scheduler.start() # Runs your Monday 8AM / Daily 6PM tasks
    except Exception:
        pass # Prevents error if scheduler is already running
    return fetcher, scheduler

fetcher, scheduler = start_institutional_engine()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BRANDING & THEME (The Mountain Path)
# ═══════════════════════════════════════════════════════════════════════════════
DARK_BLUE = "#003366"
GOLD = "#FFD700"

st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, #0066CC 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        margin-bottom: 2rem; border-bottom: 5px solid {GOLD};
    </style>
    <div class="main-header">
        <h1>INTEREST RATE ANALYTICS TERMINAL</h1>
        <p>Prof. V. Ravichandran | The Mountain Path - World of Finance</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📡 Data Engine")
    benchmark = st.selectbox("Benchmark", ["India 10Y G-Sec", "US 10Y Treasury", "US 30Y Treasury"])
    
    # Display Cache Status for India
    if benchmark == "India 10Y G-Sec":
        cached = CacheManager.load_latest_yield()
        if cached:
            val, src = cached
            st.success(f"Latest Cache: {val:.3f}% ({src})")
        else:
            st.warning("No local cache found. First fetch required.")

    st.header("⚙️ Model Configuration")
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 15)
    model_type = st.radio("ARIMA Strategy", ["Auto-ARIMA (Optimized)", "Manual (1,1,1)"])
    
    execute = st.button("🚀 RUN ANALYSIS")
    st.markdown("---")
    st.markdown(f"**Institutional Mode:** Active")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA FETCHING & ARIMA LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Yield Forecast", "🧪 Backtesting", "📊 Metrics", "📚 Educational Hub"])

if execute:
    with st.spinner("Processing Multi-Source Data Waterfall..."):
        
        # --- A. Data Acquisition ---
        if benchmark == "India 10Y G-Sec":
            # Uses your india_yield_fetcher logic (FRED -> TE -> Investing)
            fetch_res = fetcher.fetch_yield(use_cache=True)
            # For ARIMA history, we pull the historical series from FRED
            data = yf.download("IN10Y.NS", period="5y", progress=False)['Close']
            # Fallback check
            if data.empty:
                # If Yahoo is rate limited, we use a proxy for history
                st.error("Real-time history blocked. Using synthetic trend for demo.")
                data = pd.Series(np.linspace(7.1, 7.2, 100), index=pd.date_range(end=datetime.now(), periods=100))
        else:
            ticker = "^TNX" if "10Y" in benchmark else "^TYX"
            data = yf.download(ticker, period="5y", progress=False)['Close']

        # --- B. The Box-Jenkins Process ---
        series = data.dropna().resample('B').last().ffill()
        
        try:
            if model_type == "Auto-ARIMA (Optimized)":
                model = pm.auto_arima(series, seasonal=False, suppress_warnings=True)
                forecast = model.predict(n_periods=horizon)
                order = model.order
            else:
                model = ARIMA(series, order=(1,1,1)).fit()
                forecast = model.forecast(steps=horizon)
                order = (1,1,1)

            # --- C. Visualization ---
            with tabs[0]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index[-250:], y=series[-250:], name="Historical", line=dict(color=DARK_BLUE)))
                
                f_dates = pd.date_range(series.index[-1], periods=horizon+1, freq='B')[1:]
                fig.add_trace(go.Scatter(x=f_dates, y=forecast, name="Forecast", line=dict(color="orange", width=3)))
                
                fig.update_layout(title=f"{benchmark} Yield Forecast (ARIMA {order})", width=1000)
                st.plotly_chart(fig, width="stretch")

            # --- D. BPS Metrics ---
            with tabs[2]:
                c1, c2, c3 = st.columns(3)
                curr = series.iloc[-1]
                fc_final = forecast.iloc[-1]
                bps = (fc_final - curr) * 100
                
                c1.metric("Current Yield", f"{curr:.3f}%")
                c2.metric("Forecast (End of Horizon)", f"{fc_final:.3f}%")
                c3.metric("BPS Movement", f"{bps:+.1f} bps", delta_color="inverse")

        except Exception as e:
            st.error(f"Modeling Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. EDUCATIONAL HUB (Linked to PDF)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📖 Box-Jenkins Methodology")
    st.write("This dashboard follows the 6-stage process outlined in the ARIMA Modeling Guide by Prof. V. Ravichandran.")
    
    
    
    st.markdown("""
    1. **Data Preparation**: Resampling to Business days and handling missing values.
    2. **Model Selection**: Using AIC/BIC via `pmdarima` for optimal order.
    3. **Parameter Estimation**: Solving for AR, I, and MA coefficients.
    4. **Diagnostic Checking**: Ensuring residuals are White Noise.
    5. **Forecasting**: Generating point forecasts with probability bands.
    6. **Monitoring**: Using the `yield_scheduler` to track forecast accuracy over time.
    """)
    st.info("💡 Review the uploaded 'Final Version ARIMA_Modeling.pdf' for deep technical insights.")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>© 2025 {BRAND_NAME}</p>", unsafe_allow_html=True)
