
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
# 3. DASHBOARD INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast", "🌪️ GARCH Volatility", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Educational Hub"])

with tabs[0]:
    st.header("📖 Institutional Methodology")
    st.write("This terminal provides a dual-framework analysis using Prof. Ravichandran’s quantitative standards for yield and risk estimation.")

if run_btn:
    data = pd.DataFrame()
    wait_times = [0, 5, 10, 20, 30, 60] 
    success = False

    for attempt, delay in enumerate(wait_times):
        if delay > 0:
            st.warning(f"⚠️ Attempt {attempt} failed (Rate Limited). Retrying in {delay}s...")
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
        st.error("❌ Yahoo Finance Limit Reached. Please try again in 5 minutes.")
    else:
        yields = data['Close'].dropna()
        if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
        yields = yields.resample('B').last().ffill()
        returns = 100 * yields.pct_change().dropna()

        try:
            # ENGINES
            model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
            arima_fc = model_arima.predict(n_periods=horizon)
            f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
            
            garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
            latest_vol = garch_fit.conditional_volatility.iloc[-1]
            cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)

            # TAB 1: FORECAST
            with tabs[1]:
                if show_step:
                    fig_step = go.Figure()
                    fig_step.add_trace(go.Scatter(x=f_dates, y=arima_fc, mode='lines+markers', line_shape='hv', line=dict(color='#FF4B4B', width=4)))
                    fig_step.update_layout(template="plotly_dark", title="Step-Wise Forecast", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_step, width='stretch')
                else:
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Actual"))
                    fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA", line=dict(dash='dot', color='orange')))
                    fig_f.update_layout(template="plotly_white")
                    st.plotly_chart(fig_f, width='stretch')

            # TAB 2: RESTORED GARCH VOLATILITY
            with tabs[2]:
                st.subheader("🌪️ Conditional Volatility (GARCH 1,1)")
                
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol, name="Annualized Vol", line=dict(color='red')))
                fig_v.update_layout(title="Annualized Conditional Volatility (%)", template="plotly_white")
                st.plotly_chart(fig_v, width='stretch')

            # TAB 3: BACKTESTING
            with tabs[3]:
                st.subheader("🧪 30-Day Walk-Forward Validation")
                train, test = yields.iloc[:-30], yields.iloc[-30:]
                bt_model = pm.auto_arima(train, seasonal=False)
                bt_fc = bt_model.predict(n_periods=30)
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=test.index, y=test, name="Realized"))
                fig_bt.add_trace(go.Scatter(x=test.index, y=bt_fc, name="Predicted", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig_bt, width='stretch')

            # TAB 5: METRICS
            with tabs[5]:
                z_score = stats.norm.ppf(conf_level)
                var_val = latest_vol * z_score
                es_val = latest_vol * (stats.norm.pdf(z_score) / (1 - conf_level))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Rate", f"{yields.iloc[-1]:.3f}%")
                c2.metric("Forecasted", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("Daily VaR", f"{var_val:.3f}%")
                c4.metric("Exp. Shortfall", f"{es_val:.3f}%")

                x_d = np.linspace(-5, 5, 200); y_d = stats.norm.pdf(x_d, 0, 1)
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=x_d, y=y_d, fill='tozeroy', line=dict(color=CORPORATE_BLUE)))
                fig_r.add_trace(go.Scatter(x=x_d[x_d < -z_score], y=y_d[x_d < -z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)'))
                fig_r.update_layout(title="Tail Risk Visualization", template="plotly_white")
                st.plotly_chart(fig_r, width='stretch')

        except Exception as e:
            st.error(f"Computation Error: {e}")

# --- RESTORED COMPREHENSIVE Q&A HUB ---
with tabs[7]:
    st.header("🎓 Quantitative Finance Q&A Hub")
    st.write("Detailed notes covering Interest Rate Forecasting Methodologies.")

    with st.expander("❓ What is the Box-Jenkins Methodology (ARIMA)?"):
        st.write("""
        **Definition:** A systematic 3-stage iterative process (Identification, Estimation, Diagnostics) for fitting time series models.
        **Application:** Essential for interest rates because they are usually non-stationary. The model uses Autoregressive (p), Integrated (d), and Moving Average (q) components.
        """)
        

    with st.expander("❓ How does GARCH model Volatility Clustering?"):
        st.write("""
        **Definition:** GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models time-varying variance.
        **Logic:** In fixed income, large changes tend to be followed by large changes. GARCH captures these regimes, unlike standard deviation which assumes risk is constant.
        """)

    with st.expander("❓ Explain Stochastic Models: Vasicek vs CIR?"):
        st.write("""
        **Vasicek Model:** Assumes interest rates are mean-reverting. If rates are high, they are pulled down; if low, they are pulled up.
        **CIR Model:** Similar to Vasicek but introduces a 'square root' term that prevents interest rates from becoming negative, which is a key mathematical requirement in many economies.
        """)

    with st.expander("❓ What is the Nelson-Siegel Model for Yield Curves?"):
        st.write("""
        **Definition:** A factor-based model used by central banks to fit the entire term structure.
        **Three Factors:** Level (long-term), Slope (short-term), and Curvature (medium-term). It allows analysts to visualize the 'hump' or 'twist' in the yield curve.
        """)
        

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Edition</p>", unsafe_allow_html=True)
