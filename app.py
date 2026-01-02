
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

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & INSTITUTIONAL THEME
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
    
    st.header("🛡️ Risk Parameters")
    conf_level = st.select_slider("Confidence Level (α)", options=[0.90, 0.95, 0.99], value=0.95)
    
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

    for _ in range(10): st.write("")
        
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
# 3. DASHBOARD INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast", "🌪️ Volatility", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Education"])

with tabs[0]: 
    st.header("📖 Institutional Methodology")
    st.write("This platform provides a structured quantitative framework for sovereign debt analytics.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🕹️ Operational Guide")
        st.markdown("""
        1. **Configure Parameters**: Set the historical lookback and forecast window in the sidebar.
        2. **Execute**: The engine fits an ARIMA(p,d,q) for direction and GARCH(1,1) for risk.
        3. **Analyze**: Review tabs for directional paths, tail-risk (VaR/ES), and diagnostic residuals.
        """)
    with c2:
        st.subheader("📑 Model Assumptions")
        st.markdown("""
        - **Linearity**: ARIMA assumes future values are linear functions of past data and errors. [cite: 687]
        - **Volatility Clustering**: GARCH assumes variance is time-dependent and clusters. 
        - **Mean Reversion**: Rates gravitate toward a long-term equilibrium over time. [cite: 1182]
        """)

if run_btn:
    with st.spinner("Processing Yield & Volatility Engines..."):
        data = yf.download(ticker, period=f"{lookback}y", progress=False)
        
        if not data.empty:
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()
            returns = 100 * yields.pct_change().dropna()

            try:
                # 1. ENGINES
                model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
                arima_fc = model_arima.predict(n_periods=horizon)
                f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
                
                garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
                latest_vol = garch_fit.conditional_volatility.iloc[-1]
                cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)

                # 2. TAB POPULATION
                with tabs[1]: # Forecast
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=yields.index[-250:], y=yields.tail(250), name="Actual"))
                    fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(dash='dot', color='orange')))
                    fig_f.update_layout(title=f"{ticker} Forecast Path", template="plotly_white")
                    st.plotly_chart(fig_f, width='stretch')

                with tabs[2]: # GARCH
                    st.subheader("🌪️ Volatility Clustering (GARCH)")
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol, name="Ann. Volatility", line=dict(color='red')))
                    st.plotly_chart(fig_vol, width='stretch')

                with tabs[3]: # Backtesting
                    st.subheader("🧪 30-Day Walk-Forward Validation")
                    train_bt, test_bt = yields.iloc[:-30], yields.iloc[-30:]
                    bt_model = pm.auto_arima(train_bt, seasonal=False)
                    bt_fc = bt_model.predict(n_periods=30)
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt, name="Realized"))
                    fig_bt.add_trace(go.Scatter(x=test_bt.index, y=bt_fc, name="Predicted", line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig_bt, width='stretch')

                with tabs[4]: # Diagnostics
                    st.subheader("🔍 ARIMA Residual Diagnostics")
                    resid = model_arima.resid()
                    fig_resid = go.Figure(go.Scatter(y=resid, mode='lines', line=dict(color='gray')))
                    fig_resid.update_layout(title="Standardized Residuals (White Noise Check)", template="plotly_white")
                    st.plotly_chart(fig_resid, width='stretch')
                    st.info("💡 Goal: Residuals should be random with zero mean and constant variance. [cite: 676]")

                with tabs[5]: # Metrics
                    z_score = stats.norm.ppf(conf_level)
                    var_val = latest_vol * z_score
                    es_val = latest_vol * (stats.norm.pdf(z_score) / (1 - conf_level))
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Rate", f"{yields.iloc[-1]:.3f}%")
                    c2.metric("Forecasted", f"{arima_fc.iloc[-1]:.3f}%")
                    c3.metric("VaR (Daily)", f"{var_val:.3f}%")
                    c4.metric("Exp. Shortfall", f"{es_val:.3f}%")

                with tabs[6]: # Export
                    export_df = pd.DataFrame({"Date": f_dates, "Forecast": arima_fc})
                    st.dataframe(export_df, width='stretch')
                    st.download_button("Download CSV", export_df.to_csv().encode('utf-8'), "forecast_report.csv")

                with tabs[7]: # Education
                    st.header("🎓 Quantitative Theory")
                    with st.expander("1. Univariate Models (ARIMA & GARCH)"):
                        st.write("ARIMA captures momentum and trends [cite: 678], while GARCH captures volatility clustering. ")
                    with st.expander("2. Stochastic Models (Vasicek & CIR)"):
                        st.write("Vasicek incorporates mean reversion [cite: 764], while CIR ensures rates never drop below zero. [cite: 806]")
                    with st.expander("3. Factor Models (Nelson-Siegel)"):
                        st.write("Explains the curve using Level, Slope, and Curvature factors. [cite: 846, 854]")
                    
            except Exception as e:
                st.error(f"Computation Error: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Edition</p>", unsafe_allow_html=True)
