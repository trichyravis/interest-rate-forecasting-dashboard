
import streamlit as st  # THIS MUST BE LINE 1
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
# 1. PAGE CONFIG & BRANDING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Institutional Yield & Volatility Terminal", layout="wide")

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
    </style>
    <div class="main-header">
        <h1>INTEREST RATE & VOLATILITY TERMINAL</h1>
        <p>The Mountain Path - World of Finance | Quantitative Analytics</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR - PROFILE & CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Profile Card
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: #FFFFFF; border: 1px solid {DARK_BLUE};">
            <h3 style="color: {DARK_BLUE}; margin: 0;">Prof. V. Ravichandran</h3>
            <p style="color: gray; font-size: 0.85rem; margin: 5px 0;">The Mountain Path - World of Finance</p>
            <hr style="margin: 10px 0;">
            <a href="https://www.linkedin.com/in/v-ravichandran-finance" target="_blank">
                <button style="background-color: #0077b5; color: white; border: none; padding: 8px; border-radius: 5px; width: 100%; cursor: pointer;">LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🇺🇸 Treasury Benchmarks")
    ticker_label = st.selectbox("Maturity", ["US 10Y (^TNX)", "US 30Y (^TYX)", "US 5Y (^FVX)"])
    ticker = ticker_label.split("(")[1].replace(")", "")
    
    lookback = st.slider("Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast (Days)", 5, 60, 20)
    
    st.markdown("---")
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE (ARIMA + GARCH)
# ═══════════════════════════════════════════════════════════════════════════════


tabs = st.tabs(["📈 Rate Forecast", "🌪️ Volatility (GARCH)", "📊 Risk Metrics", "📚 Educational Hub"])

if run_btn:
    with st.spinner("Processing Market Data..."):
        data = yf.download(ticker, period=f"{lookback}y", progress=False)
        
        if not data.empty:
            # Flatten data for metric calculations
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()
            returns = 100 * yields.pct_change().dropna()

            try:
                # ARIMA
                model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
                arima_fc = model_arima.predict(n_periods=horizon)
                
                # GARCH
                garch = arch_model(returns, p=1, q=1, vol='Garch', dist='Normal')
                res_garch = garch.fit(disp='off')
                garch_fc = res_garch.forecast(horizon=horizon)
                
                # TAB 1: Rate Forecast
                with tabs[0]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=yields.index[-250:], y=yields.tail(250), name="Actual", line=dict(color=DARK_BLUE)))
                    f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
                    fig.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(color="orange", dash='dot')))
                    st.plotly_chart(fig, use_container_width=True)

                # TAB 2: Volatility
                with tabs[1]:
                    ann_vol = np.sqrt(res_garch.conditional_volatility**2 * 252)
                    fig_v = go.Figure()
                    fig_v.add_trace(go.Scatter(x=yields.index[-250:], y=ann_vol.tail(250), name="GARCH Vol", line=dict(color="red")))
                    st.plotly_chart(fig_v, use_container_width=True)

                # TAB 3: Metrics (Fixed Scalar Error)
                with tabs[2]:
                    c1, c2, c3 = st.columns(3)
                    curr = float(yields.iloc[-1])
                    pred = float(arima_fc.iloc[-1])
                    var_val = float(np.sqrt(garch_fc.variance.values[-1, 0]) * 1.645)
                    
                    c1.metric("Current Spot", f"{curr:.3f}%")
                    c2.metric("Predicted Rate", f"{pred:.3f}%")
                    c3.metric("Daily VaR (95%)", f"{var_val:.3f}%")

            except Exception as e:
                st.error(f"Computation Error: {e}")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance</p>", unsafe_allow_html=True)
