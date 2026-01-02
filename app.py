
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
# 1. PAGE CONFIG & THEME
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
        background-color: {GOLD} !important;
        color: {CORPORATE_BLUE} !important;
        font-weight: bold !important;
        width: 100%; border-radius: 8px;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; font-weight: bold; color: {CORPORATE_BLUE} !important; }}
    </style>
    <div class="main-header">
        <h1>INTEREST RATE FORECASTING DASHBOARD</h1>
        <p>The Mountain Path - World of Finance | Institutional Risk & Yield Analytics</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR
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

    for _ in range(8): st.write("")
        
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.15); border: 1px solid {GOLD};">
            <h3 style="color: white !important; margin: 0;">Prof. V. Ravichandran</h3>
            <hr style="margin: 10px 0; border-color: {GOLD};">
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank" style="text-decoration: none;">
                <button style="background-color: #0077b5; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">🔗 LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE & TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast", "🌪️ GARCH Volatility", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Education"])

# POPULATE ABOUT TAB IMMEDIATELY
with tabs[0]:
    st.header("📖 Institutional Methodology")
    st.write("This terminal provides a dual-framework analysis for sovereign debt benchmarks.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🕹️ Operational Guide")
        st.markdown("""
        1. **Configure Parameters**: Use the sidebar to set history and forecast length.
        2. **Execute**: The engine will fetch live Treasury data and fit ARIMA/GARCH.
        3. **Analyze Risk**: Check the Metrics tab for VaR and Expected Shortfall.
        """)
    with col_b:
        st.subheader("📑 Core Assumptions")
        st.markdown("""
        - **Mean Reversion**: ARIMA assumes rates eventually revert to a local trend.
        - **Volatility Clustering**: GARCH assumes variance is time-dependent.
        - **Data Source**: Live feeds provided by Yahoo Finance (Delayed 15m).
        """)

if run_btn:
    with st.spinner("Accessing Institutional Feeds..."):
        # Rate Limit Resilient Download
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=f"{lookback}y")
            if data.empty:
                # Fallback to standard download if history() fails
                data = yf.download(ticker, period=f"{lookback}y", progress=False)
        except Exception:
            st.error("⚠️ Data connection lost. Yahoo Finance is currently rate-limiting this session. Please wait 1-2 minutes and click 'Execute' again.")
            st.stop()
        
        if not data.empty:
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()
            returns = 100 * yields.pct_change().dropna()

            try:
                # 1. ARIMA & GARCH Fitting
                model_arima = pm.auto_arima(yields, seasonal=False)
                arima_fc = model_arima.predict(n_periods=horizon)
                f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]

                garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
                latest_vol = garch_fit.conditional_volatility.iloc[-1]
                cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)
                
                # 2. VaR & Expected Shortfall
                z_score = stats.norm.ppf(conf_level)
                var_val = latest_vol * z_score
                pdf_z = stats.norm.pdf(z_score)
                es_val = latest_vol * (pdf_z / (1 - conf_level))

                with tabs[1]: # Forecast
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Historical"))
                    fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(dash='dot', color='orange')))
                    fig_f.update_layout(title="Yield Rate Projection", template="plotly_white")
                    st.plotly_chart(fig_f, width='stretch')

                with tabs[2]: # GARCH Tab
                    st.subheader("🌪️ Volatility Clustering (GARCH 1,1)")
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol, name="Ann. Volatility", line=dict(color='red')))
                    st.plotly_chart(fig_vol, width='stretch')
                    
                with tabs[5]: # Metrics Tab
                    st.subheader(f"📊 Market Risk Summary (α = {conf_level*100:.0f}%)")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Rate", f"{yields.iloc[-1]:.3f}%")
                    c2.metric("Forecasted Rate", f"{arima_fc.iloc[-1]:.3f}%")
                    c3.metric("Value-at-Risk (VaR)", f"{var_val:.3f}%")
                    c4.metric("Expected Shortfall", f"{es_val:.3f}%")

                    x = np.linspace(-5, 5, 200)
                    y = stats.norm.pdf(x, 0, 1)
                    fig_risk = go.Figure()
                    fig_risk.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='Normal Dist', line=dict(color=CORPORATE_BLUE)))
                    mask_var = x < -z_score
                    fig_risk.add_trace(go.Scatter(x=x[mask_var], y=y[mask_var], fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.4)', name='Tail Risk'))
                    fig_risk.update_layout(title="Risk Zone: Probability of Extreme Moves", template="plotly_white")
                    st.plotly_chart(fig_risk, width='stretch')

                with tabs[6]: # Export
                    export_df = pd.DataFrame({"Date": f_dates, "Forecast": arima_fc})
                    st.dataframe(export_df, width='stretch')
                    st.download_button("Download CSV", export_df.to_csv().encode('utf-8'), "forecast.csv")

                with tabs[7]: # Education
                    st.header("🎓 The Quantitative Edge")
                    
                    st.markdown("""
                    **Expected Shortfall (ES)** is superior to VaR because it tells us the average magnitude of a loss *given that* the loss threshold has been exceeded.
                    """)
                    

            except Exception as e:
                st.error(f"Computation Error: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional Risk Terminal</p>", unsafe_allow_html=True)
