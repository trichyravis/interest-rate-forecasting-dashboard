
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model  # For GARCH Volatility
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & INSTITUTIONAL BRANDING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Institutional Yield & Volatility Terminal", layout="wide")

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
# 2. SIDEBAR - CONTROL CENTER
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/mountain.png", width=80)
    st.title("Control Center")
    
    st.header("🏦 Benchmark Selection")
    ticker_label = st.selectbox("US Treasury Maturity", [
        "US 10Y Treasury (^TNX)", 
        "US 30Y Treasury (^TYX)", 
        "US 5Y Treasury (^FVX)"
    ])
    ticker = ticker_label.split("(")[1].replace(")", "")
    
    st.header("📅 Configuration")
    lookback = st.slider("Historical Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20)
    
    st.header("🔬 Models")
    arima_mode = st.radio("Rate Path", ["Auto-ARIMA (Optimized)", "ARIMA (1,1,1)"])
    
    st.markdown("---")
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")
    
    st.markdown(f"""
        <div style='text-align: center; color: {DARK_BLUE}; font-size: 0.8rem; padding-top: 20px;'>
            <b>The Mountain Path</b><br>Institutional Analytics v2.0
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE & TABS
# ═══════════════════════════════════════════════════════════════════════════════


tabs = st.tabs(["📈 Rate Forecast", "🌪️ Volatility (GARCH)", "🧪 Backtesting", "📊 Risk Metrics", "📚 Educational Hub"])

if run_btn:
    with st.spinner(f"Processing {ticker} via Box-Jenkins & GARCH Frameworks..."):
        # Data Acquisition (US Specific)
        data = yf.download(ticker, period=f"{lookback}y", interval="1d", progress=False)

        if not data.empty:
            # Flatten Series to avoid Format Errors
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame):
                yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()
            
            # Calculate Returns for GARCH
            returns = 100 * yields.pct_change().dropna()

            try:
                # A. ARIMA RATE FORECAST
                if arima_mode == "Auto-ARIMA (Optimized)":
                    model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
                    arima_fc = model_arima.predict(n_periods=horizon)
                    order = model_arima.order
                else:
                    model_arima = ARIMA(yields, order=(1,1,1)).fit()
                    arima_fc = model_arima.forecast(steps=horizon)
                    order = (1,1,1)

                # B. GARCH VOLATILITY FORECAST
                garch = arch_model(returns, p=1, q=1, vol='Garch', dist='Normal')
                res_garch = garch.fit(disp='off')
                garch_fc = res_garch.forecast(horizon=horizon)
                
                # Annualized Conditional Volatility
                annualized_vol = np.sqrt(res_garch.conditional_volatility**2 * 252)
                forecast_vol = np.sqrt(garch_fc.variance.values[-1] * 252)

                # ══════════════════════════════════════════════════════════════
                # RENDER TABS
                
                # TAB 1: RATE FORECAST
                with tabs[0]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=yields.index[-250:], y=yields.tail(250), name="Historical", line=dict(color=DARK_BLUE)))
                    f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
                    fig.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Projection", line=dict(color="orange", width=4, dash='dot')))
                    fig.update_layout(title=f"{ticker} Yield Projection (ARIMA{order})", template="plotly_white", hovermode="x unified")
                    st.plotly_chart(fig, width="stretch")

                # TAB 2: VOLATILITY (GARCH)
                with tabs[1]:
                    st.subheader("Volatility Clustering Analysis")
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(x=yields.index[-250:], y=annualized_vol.tail(250), name="Cond. Volatility", line=dict(color="red")))
                    fig_vol.update_layout(title="Annualized Conditional Volatility (%)", template="plotly_white")
                    st.plotly_chart(fig_vol, width="stretch")

                # TAB 3: BACKTESTING
                with tabs[2]:
                    st.subheader("Model Validation Metrics")
                    mae = np.mean(np.abs(yields.tail(10).values - arima_fc[:10].values))
                    st.write(f"**Mean Absolute Error (Last 10 Days):** {mae:.4f}")
                    st.info("The Box-Jenkins process requires continuous diagnostic checking of residuals to ensure the model remains parsimonious.")

                # TAB 4: RISK METRICS (Indentation & Scalar Fixed)
                with tabs[3]:
                    c1, c2, c3 = st.columns(3)
                    curr_rate = float(yields.iloc[-1])
                    pred_rate = float(arima_fc.iloc[-1])
                    bps = (pred_rate - curr_rate) * 100
                    
                    # Calculate Scalar VaR
                    forecast_var = garch_fc.variance.values[-1, 0]
                    var_val = float(np.sqrt(forecast_var) * 1.645)
                    
                    c1.metric("Current Spot", f"{curr_rate:.3f}%")
                    c2.metric("Expected Move", f"{bps:+.1f} bps")
                    c3.metric("VaR (95% Daily)", f"{var_val:.3f}%")

                # TAB 5: EDUCATIONAL HUB
                with tabs[4]:
                    st.header("🎓 Quantitative Frameworks")
                    
                    st.markdown("""
                    **Stage 1: ARIMA Modeling** Captures the linear dependency and stationarity of the yield levels. Essential for predicting the 'Rate Path'.
                    
                    **Stage 2: GARCH Volatility** Captures 'Heteroskedasticity' (varying risk levels). Essential for Pricing Options and Risk Management (VaR).
                    
                    **Stage 3: Integration** Combining both allows for a comprehensive look at both the mean (rate) and variance (risk) of the bond market.
                    """)

            except Exception as e:
                st.error(f"Quant Engine Error: {e}")
        else:
            st.error("Historical feed interrupted. Please check Yahoo Finance connectivity.")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Analytics</p>", unsafe_allow_html=True)
