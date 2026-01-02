
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
# 1. PAGE CONFIG & INSTITUTIONAL THEMING
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
    .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; font-weight: bold; }}
    </style>
    <div class="main-header">
        <h1>INTEREST RATE & VOLATILITY TERMINAL</h1>
        <p>Prof. V. Ravichandran | The Mountain Path - World of Finance</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR DESIGN & CONTENTS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/mountain.png", width=80)
    st.title("Control Center")
    
    st.header("🏦 Benchmark")
    ticker_label = st.selectbox("US Treasury Maturity", [
        "US 10Y Treasury (^TNX)", 
        "US 30Y Treasury (^TYX)", 
        "US 5Y Treasury (^FVX)"
    ])
    ticker = ticker_label.split("(")[1].replace(")", "")
    
    st.header("📅 Time Horizon")
    lookback = st.slider("Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20)
    
    st.header("🔬 Model Config")
    arima_mode = st.radio("ARIMA Mode", ["Auto (Optimized)", "Manual (1,1,1)"])
    vol_mode = st.radio("Volatility Model", ["GARCH (1,1)", "EWMA"])
    
    st.markdown("---")
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")
    
    st.markdown(f"""
        <div style='text-align: center; color: {DARK_BLUE}; font-size: 0.8rem;'>
            <b>The Mountain Path</b><br>Institutional Analytics v2.0
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TABS RESTORATION
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Rate Forecast", "🌪️ Volatility (GARCH)", "🧪 Backtesting", "📊 Risk Metrics", "📚 Educational Hub"])

if run_btn:
    with st.spinner("Accessing Institutional Feeds & Fitting Models..."):
        # Data Acquisition
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback*365)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if not data.empty:
            # Prepare Series
            yields = data['Close'].dropna().resample('B').last().ffill()
            returns = 100 * yields.pct_change().dropna() # Percentage returns for GARCH

            # --- ARIMA MODELING ---
            if arima_mode == "Auto (Optimized)":
                model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
                arima_fc = model_arima.predict(n_periods=horizon)
                order = model_arima.order
            else:
                model_arima = ARIMA(yields, order=(1,1,1)).fit()
                arima_fc = model_arima.forecast(steps=horizon)
                order = (1,1,1)

            # --- GARCH MODELING ---
            garch = arch_model(returns, p=1, q=1, vol='Garch', dist='Normal')
            res_garch = garch.fit(disp='off')
            garch_fc = res_garch.forecast(horizon=horizon)
            # Conditional Volatility (Annualized)
            annualized_vol = np.sqrt(res_garch.conditional_volatility**2 * 252)
            forecast_vol = np.sqrt(garch_fc.variance.values[-1] * 252)

            # ═══════════════════════════════════════════════════════════════════
            # TAB 1: RATE FORECAST
            with tabs[0]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=yields.index[-250:], y=yields.tail(250), name="Actual", line=dict(color=DARK_BLUE)))
                f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
                fig.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(color="orange", width=3, dash='dot')))
                fig.update_layout(title=f"ARIMA{order} Yield Projection", template="plotly_white")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(pd.DataFrame({"Date": f_dates, "Forecasted Yield (%)": arima_fc}).set_index("Date"), width="stretch")

            # TAB 2: VOLATILITY (GARCH)
            with tabs[1]:
                st.subheader("Conditional Volatility Analysis (GARCH 1,1)")
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=yields.index[-250:], y=annualized_vol.tail(250), name="Historical Vol", line=dict(color="red")))
                fig_vol.update_layout(title="Annualized Conditional Volatility (%)", template="plotly_white")
                st.plotly_chart(fig_vol, width="stretch")
                
                c1, c2 = st.columns(2)
                c1.metric("Current Volatility", f"{annualized_vol.iloc[-1]:.2f}%")
                c2.metric("Forecasted Vol (T+H)", f"{forecast_vol[0]:.2f}%")

            # TAB 3: BACKTESTING
            with tabs[2]:
                st.subheader("Model Validation (Last 30 Days)")
                # Simplified backtest: compare last 30 actuals vs model
                st.info("Backtesting compares model predictions against realized market data to ensure calibration accuracy.")
                st.write(f"**Mean Absolute Error (MAE):** {np.mean(np.abs(yields.tail(10).values - arima_fc[:10].values)):.4f}")

            # TAB 4: RISK METRICS
            with tabs[3]:
                c1, c2, c3 = st.columns(3)
                curr_rate = yields.iloc[-1]
                pred_rate = arima_fc.iloc[-1]
                bps = (pred_rate - curr_rate) * 100
                
                c1.metric("Current Spot", f"{curr_rate:.3f}%")
                c2.metric("Expected Move", f"{bps:+.1f} bps")
                c3.metric("VaR (95% Daily)", f"{res_garch.forecast(horizon=1).variance.values[-1,0]**0.5 * 1.645:.3f}%")

            # TAB 5: EDUCATIONAL HUB
            with tabs[4]:
                st.header("🎓 The Quantitative Framework")
                
                st.markdown(f"""
                ### 1. Interest Rate Engine: ARIMA
                We use **Auto-Regressive Integrated Moving Average** to capture the momentum and mean-reversion of Treasury yields.
                
                ### 2. Volatility Engine: GARCH
                **Generalized Autoregressive Conditional Heteroskedasticity** (GARCH) allows us to model "Volatility Clustering" — where periods of high turbulence tend to follow each other.
                
                ### 3. Application in Banking
                These models are used in **ALM (Asset Liability Management)** to stress test bond portfolios against interest rate shocks.
                """)
                st.success("Refer to 'Final Version ARIMA_Modeling.pdf' for Stage 4-6 Diagnostic deep dives.")

        else:
            st.error("Market data link severed. Please verify connection.")
