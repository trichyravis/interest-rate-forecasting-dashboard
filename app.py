
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & THEME
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Interest Rate Forecasting Dashboard", layout="wide")

DARK_BLUE = "#003366"
GOLD = "#FFD700"

st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, #0066CC 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        margin-bottom: 2rem; border-bottom: 5px solid {GOLD};
    }}
    [data-testid="stSidebar"] {{ background-color: #f0f2f6; border-right: 2px solid {DARK_BLUE}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f0f2f6; border-radius: 5px 5px 0 0; padding: 10px 20px; color: {DARK_BLUE};
    }}
    .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; font-weight: bold; color: {DARK_BLUE} !important; }}
    </style>
    <div class="main-header">
        <h1>INTEREST RATE FORECASTING DASHBOARD</h1>
        <p>Prof. V. Ravichandran | The Mountain Path - World of Finance</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR - PROFILE AT BOTTOM
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_label = st.selectbox("Benchmark Maturity", ["US 10Y (^TNX)", "US 30Y (^TYX)", "US 5Y (^FVX)"])
    ticker = ticker_label.split("(")[1].replace(")", "")
    
    lookback = st.slider("Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20)
    
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS", use_container_width=True)

    # Spacer to push profile to bottom
    for _ in range(15):
        st.write("")
        
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: #FFFFFF; border: 1px solid {DARK_BLUE};">
            <h3 style="color: {DARK_BLUE}; margin: 0;">Prof. V. Ravichandran</h3>
            <p style="color: gray; font-size: 0.85rem; margin: 5px 0;">28+ Years Finance Experience</p>
            <hr style="margin: 10px 0;">
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank">
                <button style="background-color: #0077b5; color: white; border: none; padding: 8px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Forecast View", "🧪 Backtesting", "📊 Yield Metrics", "📚 Educational Hub"])

if run_btn:
    with st.spinner("Accessing Institutional Feeds..."):
        # Stable US Data Fetching
        data = yf.download(ticker, period=f"{lookback}y", interval="1d", progress=False)
        
        if not data.empty:
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()

            try:
                # Execution of ARIMA Model
                model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
                arima_fc = model_arima.predict(n_periods=horizon)
                order = model_arima.order

                # TAB 1: Forecast View
                with tabs[0]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=yields.index[-250:], y=yields.tail(250), name="Historical Yield", line=dict(color=DARK_BLUE)))
                    f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
                    fig.add_trace(go.Scatter(x=f_dates, y=arima_fc, name=f"ARIMA{order} Forecast", line=dict(color="orange", dash='dot', width=3)))
                    fig.update_layout(title=f"{ticker_label} Projection Path", template="plotly_white", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                # TAB 2: Backtesting
                with tabs[1]:
                    st.subheader("30-Day Walk-Forward Validation")
                    train, test = yields.iloc[:-30], yields.iloc[-30:]
                    bt_model = pm.auto_arima(train, seasonal=False)
                    bt_forecast = bt_model.predict(n_periods=30)
                    
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=test.index, y=test, name="Realized Market Data", line=dict(color=DARK_BLUE)))
                    fig_bt.add_trace(go.Scatter(x=test.index, y=bt_forecast, name="Model Prediction", line=dict(color="gray", dash='dash')))
                    fig_bt.update_layout(template="plotly_white")
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    mae = np.mean(np.abs(test.values - bt_forecast.values))
                    st.success(f"**Mean Absolute Error (MAE):** {mae:.4f}")

                # TAB 3: Yield Metrics
                with tabs[2]:
                    c1, c2, c3 = st.columns(3)
                    curr = float(yields.iloc[-1])
                    pred = float(arima_fc.iloc[-1])
                    c1.metric("Current Spot Rate", f"{curr:.3f}%")
                    c2.metric("Term Forecast", f"{pred:.3f}%")
                    c3.metric("BPS Shift", f"{(pred-curr)*100:+.1f} bps", delta_color="inverse")
                    
                    st.markdown("### Model Diagnostics")
                    st.write(f"**Optimal Order identified by Auto-ARIMA:** {order}")

                # TAB 4: Educational Hub
                with tabs[3]:
                    st.header("🎓 Understanding ARIMA & Box-Jenkins")
                    st.markdown("""
                    The **Box-Jenkins methodology** is a mathematical framework for finding the best-fit 
                    time series model for forecasting. It consists of three primary stages:
                    
                    1. **Identification**: Checking for stationarity (constant mean and variance).
                    2. **Estimation**: Finding the optimal $p, d, q$ parameters.
                    3. **Diagnostic Checking**: Ensuring the residuals (errors) are "White Noise."
                    """)
                    
                    st.info("💡 Tip: In interest rate modeling, the 'Integrated' (d) component often accounts for the non-stationary 'trend' in yield movements.")

            except Exception as e:
                st.error(f"Computation Error: {e}")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Edition</p>", unsafe_allow_html=True)
