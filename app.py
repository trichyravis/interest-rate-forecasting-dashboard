
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
# 1. PAGE CONFIG & INSTITUTIONAL THEME (OXFORD BLUE & GOLD)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Interest Rate Forecasting Dashboard", layout="wide")

CORPORATE_BLUE = "#002147" 
GOLD = "#FFD700"

st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, {CORPORATE_BLUE} 0%, #004b8d 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        margin-bottom: 2rem; border-bottom: 5px solid {GOLD};
    }}
    [data-testid="stSidebar"] {{ background-color: {CORPORATE_BLUE} !important; color: white !important; }}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h2 {{ color: white !important; }}
    
    /* Button Styling - High Visibility */
    div.stButton > button:first-child {{
        background-color: {GOLD} !important;
        color: {CORPORATE_BLUE} !important;
        font-weight: bold !important;
        width: 100%; border-radius: 8px;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{ gap: 12px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f0f2f6; border-radius: 5px 5px 0 0; padding: 10px 15px; color: {CORPORATE_BLUE};
    }}
    .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; font-weight: bold; }}
    </style>
    
    <div class="main-header">
        <h1>INTEREST RATE FORECASTING DASHBOARD</h1>
        <p>The Mountain Path - World of Finance | Institutional Research Terminal</p>
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
    
    st.header("🎨 Display Settings")
    show_step = st.checkbox("Show Step-Wise Curve", value=True)
    
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

    for _ in range(10): st.write("") # Spacer
        
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.15); border: 1px solid {GOLD};">
            <h3 style="color: white !important; margin: 0;">Prof. V. Ravichandran</h3>
            <p style="color: #ffffff !important; font-size: 0.85rem; margin: 5px 0;">28+ Years Finance Experience</p>
            <hr style="margin: 10px 0; border-color: {GOLD};">
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank" style="text-decoration: none;">
                <button style="background-color: #0077b5; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">🔗 LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE & TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About & Assumptions", "📈 Forecast", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Educational Hub"])

with tabs[0]: # About Tab (Always visible)
    st.header("📖 About this Terminal")
    st.write("""
    This institutional-grade tool is designed to forecast interest rate paths using 
    **Autoregressive Integrated Moving Average (ARIMA)** models. It serves as a 
    quantitative decision-support system for Fixed Income analysts.
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🕹️ How to Operate")
        st.markdown("""
        1. **Select Benchmark:** Choose the US Treasury maturity from the sidebar.
        2. **Set Parameters:** Adjust the historical lookback and forecast window.
        3. **Run Analysis:** Click the 'Execute' button to trigger the ARIMA engine.
        4. **Analyze Tabs:** Review forecasts, backtests, and diagnostic residuals.
        """)
    with c2:
        st.subheader("📑 Model Assumptions")
        st.markdown("""
        * **Stationarity:** The model assumes the series can be made stationary via differencing ($d$).
        * **Linearity:** It assumes future values are linear functions of past data and errors.
        * **Ceteris Paribus:** It does not account for sudden "Black Swan" events or Federal Reserve policy shifts.
        """)

if run_btn:
    with st.spinner("Processing Market Data..."):
        data = yf.download(ticker, period=f"{lookback}y", interval="1d", progress=False)
        
        if not data.empty:
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()

            try:
                # Execution
                model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
                arima_fc = model_arima.predict(n_periods=horizon)
                f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
                order = model_arima.order

                with tabs[1]: # Forecast
                    if show_step:
                        fig_step = go.Figure()
                        fig_step.add_trace(go.Scatter(x=f_dates, y=arima_fc, mode='lines+markers', line_shape='hv', 
                                                    line=dict(color='#FF4B4B', width=4), name="Rate Steps"))
                        fig_step.update_layout(template="plotly_dark", title=f"Institutional Step-Curve: ARIMA{order}", paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_step, use_container_width=True)
                    else:
                        fig_main = go.Figure()
                        fig_main.add_trace(go.Scatter(x=yields.index[-250:], y=yields.tail(250), name="Actual", line=dict(color=CORPORATE_BLUE)))
                        fig_main.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="Forecast", line=dict(color="orange", dash='dot')))
                        fig_main.update_layout(template="plotly_white")
                        st.plotly_chart(fig_main, use_container_width=True)

                with tabs[2]: # Backtesting
                    train, test = yields.iloc[:-30], yields.iloc[-30:]
                    bt_model = pm.auto_arima(train, seasonal=False)
                    bt_fc = bt_model.predict(n_periods=30)
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=test.index, y=test, name="Market Data", line=dict(color=CORPORATE_BLUE)))
                    fig_bt.add_trace(go.Scatter(x=test.index, y=bt_fc, name="Model Prediction", line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig_bt, use_container_width=True)
                    st.success(f"MAE (Last 30 Days): {np.mean(np.abs(test.values - bt_fc.values)):.4f}")

                with tabs[3]: # FIXED: Diagnostics
                    st.subheader("🔍 ARIMA Residual Diagnostics")
                    # Check if residuals are white noise
                    residuals = model_arima.resid()
                    fig_resid = go.Figure()
                    fig_resid.add_trace(go.Scatter(y=residuals, mode='lines', name='Residuals', line=dict(color='gray')))
                    fig_resid.update_layout(title="Residual Errors (Standardized)", template="plotly_white")
                    st.plotly_chart(fig_resid, use_container_width=True)
                    
                    st.markdown("""
                    **Goal:** Residuals should fluctuate randomly around zero. If patterns exist, the model parameters ($p, d, q$) may need adjustment.
                    """)
                    

                with tabs[4]: # Metrics
                    c1, c2, c3 = st.columns(3)
                    curr, pred = float(yields.iloc[-1]), float(arima_fc.iloc[-1])
                    c1.metric("Current Spot", f"{curr:.3f}%")
                    c2.metric("Forecasted Rate", f"{pred:.3f}%")
                    c3.metric("BPS Move", f"{(pred-curr)*100:+.1f} bps")

                with tabs[5]: # FIXED: Export
                    st.subheader("📋 Export Forecast Results")
                    export_df = pd.DataFrame({"Forecast Date": f_dates, "Predicted Yield (%)": arima_fc})
                    st.dataframe(export_df, use_container_width=True)
                    csv = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download as CSV", data=csv, file_name=f"yield_forecast_{ticker}.csv", mime='text/csv')

                with tabs[6]: # Educational Hub
                    st.header("🎓 Box-Jenkins ARIMA Framework")
                    
                    st.markdown(f"**Identified Model:** ARIMA{order}")

            except Exception as e:
                st.error(f"Computation Error: {e}")
