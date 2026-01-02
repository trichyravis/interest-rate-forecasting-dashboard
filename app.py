
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
# 3. ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast", "🌪️ GARCH Volatility", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Education"])

if run_btn:
    with st.spinner("Calculating Risk-Adjusted Yield Paths..."):
        data = yf.download(ticker, period=f"{lookback}y", progress=False)
        
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
                
                # 2. VaR & Expected Shortfall (CVaR) Calculation
                z_score = stats.norm.ppf(conf_level)
                var_val = latest_vol * z_score
                
                # Formula: ES = σ * [pdf(z) / (1-α)]
                pdf_z = stats.norm.pdf(z_score)
                es_val = latest_vol * (pdf_z / (1 - conf_level))

                with tabs[5]: # Metrics Tab
                    st.subheader(f"📊 Market Risk Summary (α = {conf_level*100:.0f}%)")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Forecasted Rate", f"{arima_fc.iloc[-1]:.3f}%")
                    c2.metric("Ann. Volatility", f"{np.sqrt(latest_vol**2 * 252):.1f}%")
                    c3.metric("Value-at-Risk (VaR)", f"{var_val:.3f}%", help="Minimum expected loss in tail")
                    c4.metric("Expected Shortfall", f"{es_val:.3f}%", help="Average loss beyond VaR threshold")

                    # Visual Display
                    x = np.linspace(-5, 5, 200)
                    y = stats.norm.pdf(x, 0, 1)
                    fig_risk = go.Figure()
                    fig_risk.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='Return Dist', line=dict(color=CORPORATE_BLUE)))
                    
                    # Shade VaR vs ES
                    mask_var = x < -z_score
                    fig_risk.add_trace(go.Scatter(x=x[mask_var], y=y[mask_var], fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.5)', name='VaR Area'))
                    
                    fig_risk.add_annotation(x=-z_score, y=0.1, text=f"VaR: {var_val:.2f}%", showarrow=True, arrowhead=1)
                    fig_risk.add_annotation(x=-z_score-0.8, y=0.03, text=f"ES: {es_val:.2f}%", font=dict(color="red"))
                    
                    fig_risk.update_layout(title="Tail Risk Visualization: VaR vs. Expected Shortfall", template="plotly_white")
                    st.plotly_chart(fig_risk, use_container_width=True)

                with tabs[7]: # Education
                    st.header("🎓 Advanced Risk: Expected Shortfall (ES)")
                    
                    st.markdown(f"""
                    **Expected Shortfall (ES)** is the coherent risk measure that answers: *"If we have a bad day and exceed our VaR, what is the expected magnitude of that loss?"*
                    
                    Unlike VaR, ES accounts for the **severity of losses** in the tail of the distribution, making it superior for managing 'fat-tail' risks in the bond market.
                    """)

            except Exception as e:
                st.error(f"Computation Error: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Quantitative Risk Terminal</p>", unsafe_allow_html=True)
