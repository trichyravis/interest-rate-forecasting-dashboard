
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & BRANDING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="US Treasury Analytics Terminal", layout="wide")

DARK_BLUE = "#003366"
GOLD = "#FFD700"

# Professional CSS Injection
st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, #0066CC 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        margin-bottom: 2rem; border-bottom: 5px solid {GOLD};
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px; white-space: pre-wrap; background-color: #F0F2F6;
        border-radius: 5px 5px 0px 0px; padding: 10px 20px;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; color: {DARK_BLUE} !important; font-weight: bold; }}
    </style>
    <div class="main-header">
        <h1>US TREASURY ANALYTICS TERMINAL</h1>
        <p>Prof. V. Ravichandran | The Mountain Path - World of Finance</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR RESTORATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🇺🇸 Benchmark Selection")
    ticker_label = st.selectbox("Treasury Maturity", [
        "US 10Y Treasury (^TNX)", 
        "US 30Y Treasury (^TYX)", 
        "US 5Y Treasury (^FVX)"
    ])
    ticker = ticker_label.split("(")[1].replace(")", "")
    
    st.header("⚙️ Model Parameters")
    lookback = st.slider("Historical Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20)
    
    st.header("🔬 Methodology")
    model_type = st.radio("Optimization Strategy", ["Auto-ARIMA (Optimized)", "Manual ARIMA (1,1,1)"])
    
    st.markdown("---")
    run_btn = st.button("🚀 EXECUTE ANALYSIS")
    st.info("Uses Real-time Global Bond Feeds")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TABS RESTORATION & LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Forecast View", "📊 Yield Metrics", "📚 Educational Hub", "⚠️ Model Assumptions"])

if run_btn:
    with st.spinner(f"Analyzing {ticker} via Box-Jenkins Framework..."):
        # Data Acquisition
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback*365)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if not data.empty:
            # Data Cleaning
            series = data['Close'].dropna()
            if isinstance(series, pd.DataFrame): # Handle potential multi-index
                series = series.iloc[:, 0]
            series = series.resample('B').last().ffill()

            try:
                # Execution of ARIMA
                if model_type == "Auto-ARIMA (Optimized)":
                    model = pm.auto_arima(series, seasonal=False, suppress_warnings=True)
                    forecast = model.predict(n_periods=horizon)
                    order = model.order
                else:
                    fit = ARIMA(series, order=(1,1,1)).fit()
                    forecast = fit.forecast(steps=horizon)
                    order = (1,1,1)

                # --- TAB 1: FORECAST ---
                with tabs[0]:
                    fig = go.Figure()
                    # Show last 18 months for context
                    recent = series.tail(380)
                    fig.add_trace(go.Scatter(x=recent.index, y=recent, name="Historical Yield", line=dict(color=DARK_BLUE, width=2)))
                    
                    f_dates = pd.date_range(series.index[-1], periods=horizon+1, freq='B')[1:]
                    fig.add_trace(go.Scatter(x=f_dates, y=forecast, name="Predicted Path", line=dict(color="orange", width=4, dash='dot')))
                    
                    fig.update_layout(title=f"Yield Projection: ARIMA{order}", hovermode="x unified", template="plotly_white")
                    st.plotly_chart(fig, width="stretch")

                # --- TAB 2: METRICS ---
                with tabs[1]:
                    c1, c2, c3 = st.columns(3)
                    curr = series.iloc[-1]
                    pred = forecast.iloc[-1]
                    bps = (pred - curr) * 100
                    
                    c1.metric("Current Spot Rate", f"{curr:.3f}%")
                    c2.metric("Term Forecast", f"{pred:.3f}%")
                    c3.metric("Shift (Basis Points)", f"{bps:+.1f} bps", delta_color="inverse")
                    
                    st.markdown("### Model Diagnostics")
                    st.write(f"**Identified Order:** ARIMA{order}")
                    st.write(f"**Observation Count:** {len(series)} business days")

            except Exception as e:
                st.error(f"Computation Error: {e}")
        else:
            st.error("Market Feed Unavailable. Please try a different maturity.")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EDUCATIONAL HUB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🎓 Box-Jenkins (ARIMA) Methodology")
    
    st.write("""
    The dashboard utilizes the classic Box-Jenkins approach to time-series forecasting. 
    This involves a three-stage iterative process:
    1. **Identification:** Determining if the series is stationary (using differencing).
    2. **Estimation:** Using Maximum Likelihood to find the AR and MA coefficients.
    3. **Diagnostic Checking:** Ensuring residuals are independent (White Noise).
    """)
    st.info("💡 Deep Dive: Refer to 'Final Version ARIMA_Modeling.pdf' for Stage 4 Diagnostic insights.")

with tabs[3]:
    st.header("⚠️ Model Assumptions & Disclaimers")
    st.warning("""
    * **Stationarity:** The model assumes that the statistical properties of the series do not change over time.
    * **Exogenous Shocks:** ARIMA does not account for sudden Federal Reserve policy shifts or geopolitical events.
    * **Data Frequency:** This terminal utilizes daily closing yields.
    """)

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Version</p>", unsafe_allow_html=True)
