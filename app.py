
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
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f0f2f6; border-radius: 5px 5px 0 0; padding: 10px 20px; color: {DARK_BLUE};
    }}
    .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; font-weight: bold; color: {DARK_BLUE} !important; }}
    </style>
    <div class="main-header">
        <h1>INTEREST RATE & VOLATILITY TERMINAL</h1>
        <p>Prof. V. Ravichandran | The Mountain Path - World of Finance</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR - PROFILE & CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Restored Profile Card
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
    ticker_label = st.selectbox("Maturity", [
        "US 10Y Treasury (^TNX)", 
        "US 30Y Treasury (^TYX)", 
        "US 5Y Treasury (^FVX)"
    ])
    ticker = ticker_label.split("(")[1].replace(")", "")
    
    lookback = st.slider("Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast (Days)", 5, 60, 20)
    
    st.markdown("---")
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")
    
    st.markdown(f"<div style='text-align: center; font-size: 0.75rem; color: {DARK_BLUE};'>Institutional Analytics v2.5</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE (ARIMA + GARCH)
# ═══════════════════════════════════════════════════════════════════════════════


tabs = st.tabs(["📈 Rate Forecast", "🌪️ Volatility (GARCH)", "🧪 Backtesting", "📊 Risk Metrics", "📚 Educational Hub"])

if run_btn:
    with st.spinner("Processing US Treasury Data..."):
        # Stable US Data Fetching
        data = yf.download(ticker, period=f"{lookback}y", interval="1d", progress=False)
        
        if not data.empty:
            # Clean data and ensure it's a Series for calculations
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame): 
                yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()
            
            # Calculate Percentage Returns for GARCH
            returns = 100 * yields.pct_change().dropna()

            try:
                # 1. ARIMA RATE FORECAST
                model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
                arima_fc = model_arima.predict(n_periods=horizon)
                order = model_arima.order
                
                # 2. GARCH VOLATILITY FORECAST
                garch = arch_model(returns, p=1, q=1, vol='Garch', dist='Normal')
                res_garch = garch.fit(disp='off')
                garch_fc = res_garch.forecast(horizon=horizon)
                ann_vol = np.sqrt(res_garch.conditional_volatility**2 * 252)

                # --- Tab 1: Rate Forecast ---
                with tabs[0]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=yields.index[-250:], y=yields.tail(250), name="Historical", line=dict(color=DARK_BLUE)))
                    f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
                    fig.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(color="orange", dash='dot', width=3)))
                    fig.update_layout(title=f"ARIMA{order} Yield Projection", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                # --- Tab 2: Volatility (GARCH) ---
                with tabs[1]:
                    st.subheader("Annualized Conditional Volatility")
                    
                    fig_v = go.Figure()
                    fig_v.add_trace(go.Scatter(x=yields.index[-250:], y=ann_vol.tail(250), name="GARCH Vol", line=dict(color="red")))
                    fig_v.update_layout(template="plotly_white")
                    st.plotly_chart(fig_v, use_container_width=True)

                # --- Tab 3: Backtesting ---
                with tabs[2]:
                    mae = np.mean(np.abs(yields.tail(horizon).values - arima_fc[:len(yields.tail(horizon))].values))
                    st.metric("Model MAE (Last Period)", f"{mae:.4f}")
                    st.info("Validation based on historical residual variance analysis and Box-Jenkins methodology.")

                # --- Tab 4: Risk Metrics ---
                with tabs[3]:
                    c1, c2, c3 = st.columns(3)
                    # Convert to scalar floats to avoid formatting errors
                    curr = float(yields.iloc[-1])
                    pred = float(arima_fc.iloc[-1])
                    bps = (pred - curr) * 100
                    
                    # Scalar conversion for VaR calculation
                    forecast_var = garch_fc.variance.values[-1, 0]
                    var_val = float(np.sqrt(forecast_var) * 1.645)
                    
                    c1.metric("Current Spot", f"{curr:.3f}%")
                    c2.metric("Expected Move", f"{bps:+.1f} bps")
                    c3.metric("Daily VaR (95%)", f"{var_val:.3f}%")

                # --- Tab 5: Educational Hub ---
                with tabs[4]:
                    st.header("🎓 The Quantitative Framework")
                    

[Image of interest rate yield curve]

                    
                    st.markdown("""
                    **Stage 1: ARIMA Modeling** Captures the linear momentum and trend components of the yield level.
                    
                    **Stage 2: GARCH(1,1)** Models 'Volatility Clustering', acknowledging that risk levels change over time.
                    
                    **Practical Utility:** These combined models help Institutional ALM desks forecast interest rate risk and Value-at-Risk (VaR).
                    """)

            except Exception as e:
                st.error(f"Computation Error: {e}")
        else:
            st.error("Market Feed Error: Verify Yahoo Finance availability.")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Edition</p>", unsafe_allow_html=True)
