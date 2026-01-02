
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model 
import scipy.stats as stats
import warnings
import time

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & INSTITUTIONAL THEME
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
    [data-testid="stSidebar"] {{ background-color: {CORPORATE_BLUE} !important; color: white !important; }}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{ color: white !important; }}
    div.stButton > button:first-child {{
        background-color: {GOLD} !important; color: {CORPORATE_BLUE} !important;
        font-weight: bold !important; width: 100%; border-radius: 8px; border: none;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: {GOLD} !important; font-weight: bold; color: {CORPORATE_BLUE} !important; 
    }}
    .config-box {{
        background-color: #f8f9fa; padding: 15px; border-radius: 10px;
        border-left: 5px solid {CORPORATE_BLUE}; margin-bottom: 20px;
    }}
    </style>
    <div class="main-header">
        <h1 style="margin-bottom: 0;">INTEREST RATE FORECASTING DASHBOARD</h1>
        <h2 style="margin-top: 0; font-size: 1.5rem; opacity: 0.9;">(using ARIMA)</h2>
        <p style="margin-top: 10px; font-weight: bold; font-size: 1.1rem;">
            Prof. V. Ravichandran | The Mountain Path - World of Finance
        </p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR - PROFILE & CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_label = st.selectbox("Benchmark Maturity", ["US 10Y (^TNX)", "US 30Y (^TYX)", "US 5Y (^FVX)"])
    ticker = ticker_label.split("(")[1].replace(")", "")
    lookback = st.slider("Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20)
    
    st.header("🛡️ Risk Parameters")
    conf_level = st.select_slider("Confidence Level (α)", options=[0.90, 0.95, 0.99], value=0.95)
    
    st.header("🎨 UI Settings")
    show_step = st.checkbox("Show Step-Wise Curve", value=True)
    
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

    for _ in range(8): st.write("")
        
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.15); border: 1px solid {GOLD};">
            <h3 style="color: white !important; margin: 0;">Prof. V. Ravichandran</h3>
            <p style="color: white !important; font-size: 0.85rem; margin: 5px 0;">The Mountain Path - World of Finance</p>
            <hr style="margin: 10px 0; border-color: {GOLD};">
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank">
                <button style="background-color: #0077b5; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">🔗 LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE & TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast", "🌪️ GARCH Volatility", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Educational Hub"])

with tabs[0]: 
    st.header("📖 Institutional Research Methodology")
    st.markdown("### About this Platform")
    st.write("""
    This quantitative terminal utilizes a dual-engine approach to model interest rates. 
    It employs the **ARIMA** framework for directional pathing and **GARCH** for risk estimation.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📑 Assumptions")
        st.markdown("- **Stationarity:** Yields are differenced to stabilize mean.\n- **Clustering:** Volatility is regime-dependent.\n- **Mean Reversion:** Rates revert to local trends.")
    with col2:
        st.subheader("⚠️ Limitations")
        st.markdown("- **Black Swans:** Does not predict structural 'jump' events.\n- **Univariate:** Does not include exogenous variables like CPI/GDP.")

if run_btn:
    data = pd.DataFrame()
    wait_times = [0, 5, 10, 20, 30, 60] 
    success = False

    for attempt, delay in enumerate(wait_times):
        if delay > 0:
            st.warning(f"⚠️ Yahoo Finance busy. Retrying in {delay}s...")
            time.sleep(delay)
        with st.spinner(f"Fetching Data {attempt + 1}/6..."):
            try:
                t_obj = yf.Ticker(ticker)
                data = t_obj.history(period=f"{lookback}y")
                if not data.empty:
                    success = True
                    break
            except: continue

    if not success or data.empty:
        st.error("❌ Data retrieval failed. Please try again.")
    else:
        yields = data['Close'].dropna()
        if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
        yields = yields.resample('B').last().ffill()
        returns = 100 * yields.pct_change().dropna()

        try:
            model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
            arima_fc = model_arima.predict(n_periods=horizon)
            f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
            
            garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
            cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)

            with tabs[1]: # 📈 Forecast View
                st.markdown(f"""
                <div class="config-box">
                    <strong>Current Configuration:</strong> {ticker_label} | 
                    <strong>Historical Lookback:</strong> {lookback} Years | 
                    <strong>Forecast Horizon:</strong> {horizon} Days | 
                    <strong>Model:</strong> ARIMA{model_arima.order}
                </div>
                """, unsafe_allow_html=True)

                if show_step:
                    fig_step = go.Figure()
                    fig_step.add_trace(go.Scatter(x=f_dates, y=arima_fc, mode='lines+markers', line_shape='hv', line=dict(color='#FF4B4B', width=4)))
                    fig_step.update_layout(template="plotly_dark", title="Step-Wise Yield Forecast", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_step, width='stretch')
                else:
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Actual"))
                    fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA", line=dict(dash='dot', color='orange')))
                    fig_f.update_layout(template="plotly_white")
                    st.plotly_chart(fig_f, width='stretch')

            with tabs[2]: # 🌪️ GARCH
                st.subheader("🌪️ Conditional Volatility (GARCH 1,1)")
                
                fig_v = go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color='red')))
                fig_v.update_layout(title="Annualized Volatility (%)", template="plotly_white")
                st.plotly_chart(fig_v, width='stretch')

            with tabs[4]: # 🔍 Diagnostics
                st.subheader("🔍 Residual Analysis")
                resid = model_arima.resid()
                fig_resid = go.Figure(go.Scatter(y=resid, mode='lines', line=dict(color='gray')))
                fig_resid.update_layout(title="Standardized Residuals (White Noise Check)", template="plotly_white")
                st.plotly_chart(fig_resid, width='stretch')

            with tabs[6]: # 📋 Export
                export_df = pd.DataFrame({"Date": f_dates, "Forecast": arima_fc})
                st.subheader("📋 Export Terminal")
                st.dataframe(export_df, width='stretch')
                st.download_button("📥 Download Forecast (CSV)", export_df.to_csv().encode('utf-8'), "yield_report.csv")

        except Exception as e:
            st.error(f"Computation Error: {e}")

# --- 📚 Q&A HUB ---
with tabs[7]:
    st.header("🎓 Quantitative Q&A Hub")
    with st.expander("❓ What is the Box-Jenkins Methodology?"):
        st.write("A 3-stage process (Identification, Estimation, Diagnostics) for fitting ARIMA models.")
        
    with st.expander("❓ Explain Nelson-Siegel Model?"):
        st.write("A factor model fitting the yield curve using Level, Slope, and Curvature.")
        

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional Edition</p>", unsafe_allow_html=True)
