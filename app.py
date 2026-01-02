
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model 
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & INSTITUTIONAL THEME
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
    [data-testid="stSidebar"] {{ background-color: #0d47a1; color: white; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f0f2f6; border-radius: 5px 5px 0 0; padding: 10px 20px; color: {DARK_BLUE};
    }}
    .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; font-weight: bold; color: {DARK_BLUE} !important; }}
    .summary-box {{
        background-color: #ffffff; padding: 20px; border-radius: 10px; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin-bottom: 25px;
    }}
    </style>
    <div class="main-header">
        <h1>INTEREST RATE FORECASTING DASHBOARD</h1>
        <p>Traditional Time Series Modeling for Global Yields</p>
        <p style="font-size: 0.9rem;">Prof. V. Ravichandran | 28+ Years Finance Experience</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR - CONTROLS AT TOP, PROFILE AT BOTTOM
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ ARIMA Configuration")
    ticker_label = st.selectbox("Select Security", [
        "US 10Y Treasury (^TNX)", 
        "US 30Y Treasury (^TYX)", 
        "US 5Y Treasury (^FVX)"
    ])
    ticker = ticker_label.split("(")[1].replace(")", "")
    
    lookback = st.selectbox("Years of Historical Data", [1, 3, 5, 10], index=2)
    
    st.header("🔮 Forecast Settings")
    horizon = st.slider("Forecast Horizon (Periods)", 5, 60, 10)
    
    run_btn = st.button("🟡 FETCH YIELDS & RUN MODEL", use_container_width=True)
    
    # PUSH PROFILE TO BOTTOM
    st.markdown("<br>" * 10, unsafe_allow_html=True) 
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: left; padding: 10px; border-radius: 10px; background-color: #003366; color: white; border: 1px solid {GOLD};">
            <h4 style="margin: 0;">Prof. V. Ravichandran</h4>
            <p style="font-size: 0.8rem; margin: 5px 0;">28+ Years Finance Experience</p>
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank" style="text-decoration: none;">
                <button style="background-color: #0077b5; color: white; border: none; padding: 8px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">🔗 LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SELECTION SUMMARY SECTION
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Analyzing Market Data..."):
        data = yf.download(ticker, period=f"{lookback}y", interval="1d", progress=False)
        
        if not data.empty:
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()
            returns = 100 * yields.pct_change().dropna()

            try:
                # RUN MODELS
                model_arima = pm.auto_arima(yields, seasonal=False)
                arima_fc = model_arima.predict(n_periods=horizon)
                
                # SELECTION SUMMARY BOX
                st.markdown("### 📊 Selection Summary")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.write("**Yield**"); st.subheader(ticker)
                with c2: st.write("**History**"); st.subheader(f"{lookback}y")
                with c3: st.write("**Mode**"); st.subheader("Auto ARIMA")
                with c4: st.write("**Horizon**"); st.subheader(horizon)
                st.markdown("---")

                # TABS INTERFACE
                tabs = st.tabs(["📈 Forecast", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📚 Educational Hub"])

                with tabs[0]: # Forecast
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=yields.index[-250:], y=yields.tail(250), name="Historical", line=dict(color=DARK_BLUE)))
                    f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
                    fig.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(color="orange", dash='dot')))
                    fig.update_layout(title="Yield Projection", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                with tabs[1]: # Backtesting
                    train, test = yields.iloc[:-30], yields.iloc[-30:]
                    bt_model = pm.auto_arima(train, seasonal=False)
                    bt_fc = bt_model.predict(n_periods=30)
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=test.index, y=test, name="Realized"))
                    fig_bt.add_trace(go.Scatter(x=test.index, y=bt_fc, name="Predicted", line=dict(dash='dash')))
                    st.plotly_chart(fig_bt, use_container_width=True)

                with tabs[3]: # Metrics
                    m1, m2, m3 = st.columns(3)
                    curr, pred = float(yields.iloc[-1]), float(arima_fc.iloc[-1])
                    m1.metric("Current Yield", f"{curr:.3f}%")
                    m2.metric("Forecasted Yield", f"{pred:.3f}%")
                    m3.metric("Basis Point Shift", f"{(pred-curr)*100:+.1f} bps")

                with tabs[4]: # Educational Hub
                    st.header("🎓 Box-Jenkins ARIMA Framework")
                                        st.info("The ARIMA model combines Autoregressive (p), Integrated (d), and Moving Average (q) components to forecast future interest rate paths.")

            except Exception as e:
                st.error(f"Modeling Error: {e}")

st.markdown("<br><p style='text-align: center; color: gray;'>The Mountain Path - World of Finance | Built for Professional Excellence</p>", unsafe_allow_html=True)
