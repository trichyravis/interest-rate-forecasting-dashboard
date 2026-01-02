
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pmdarima as pm
from arch import arch_model 
import scipy.stats as stats
import time
import warnings

# ═══════════════════════════════════════════════════════════════════════════════
# MODULAR IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from content.about_text import ABOUT_CONTENT
    from content.qa_text import QA_MASTERCLASS
    from content.footer import display_footer
except ImportError:
    st.error("Critical Error: 'content' folder or files missing.")
    st.stop()

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. THEME & SIDEBAR DESIGN
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
    [data-testid="stSidebar"] {{ background-color: {CORPORATE_BLUE} !important; }}
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {{ 
        color: white !important; font-weight: bold; 
    }}
    div.stButton > button:first-child {{
        background-color: {GOLD} !important; color: {CORPORATE_BLUE} !important;
        font-weight: bold !important; width: 100%; border-radius: 8px; border: none;
    }}
    .config-info {{
        background-color: #f0f2f6; padding: 10px; border-radius: 5px;
        border-left: 5px solid {GOLD}; margin-bottom: 20px; font-size: 0.9rem; color: {CORPORATE_BLUE};
    }}
    </style>
    <div class="main-header">
        <h1 style="margin-bottom: 0; color: white;">INTEREST RATE FORECASTING DASHBOARD</h1>
        <h2 style="margin-top: 0; font-size: 1.3rem; opacity: 0.9; color: white;">Multi-Model (ARIMA, Vasicek, CIR) Institutional Terminal</h2>
        <p style="margin-top: 10px; font-weight: bold; font-size: 1.1rem; color: {GOLD};">Prof. V. Ravichandran | The Mountain Path</p>
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
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 30)
    conf_level = st.select_slider("Confidence Level (α)", options=[0.90, 0.95, 0.99], value=0.95)
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 ARIMA", "🌪️ GARCH", "🎲 Vasicek", "☀️ CIR", "🧪 Backtest", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A"])

# Display static content in About Tab
with tabs[0]:
    st.header("📖 Institutional Methodology")
    st.write(ABOUT_CONTENT["intro"])
    st.info(ABOUT_CONTENT["workflow"])
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"### ARIMA\n{ABOUT_CONTENT['arima']}")
    with c2: st.markdown(f"### Vasicek\n{ABOUT_CONTENT['vasicek']}")
    with c3: st.markdown(f"### CIR\n{ABOUT_CONTENT['cir']}")

if run_btn:
    try:
        data = yf.Ticker(ticker).history(period=f"{lookback}y")
        if not data.empty:
            yields = data['Close'].resample('B').last().ffill()
            returns = 100 * yields.pct_change().dropna()
            f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
            config_summary = f"**Asset:** {ticker_label} | **Horizon:** {horizon}D"

            # --- MODEL CALCULATIONS ---
            # ARIMA
            model_arima = pm.auto_arima(yields, seasonal=False)
            arima_fc = model_arima.predict(n_periods=horizon)

            # GARCH
            garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
            cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)

            # STOCHASTIC PARAMETERS
            r0, kappa, theta, sigma = yields.iloc[-1]/100, 0.20, 0.045, 0.015
            dt, n_paths = 1/252, 1000
            
            # --- 🎲 VASICEK SIMULATION ---
            v_paths = np.zeros((n_paths, horizon))
            v_paths[:, 0] = r0
            for i in range(1, horizon):
                v_paths[:, i] = v_paths[:, i-1] + kappa*(theta-v_paths[:, i-1])*dt + sigma*np.random.normal(0, np.sqrt(dt), n_paths)
            v_med = np.percentile(v_paths, 50, axis=0)*100

            # --- ☀️ CIR SIMULATION ---
            c_paths = np.zeros((n_paths, horizon))
            c_paths[:, 0] = r0
            for i in range(1, horizon):
                # dr = kappa(theta - r)dt + sigma * sqrt(r) * dW
                c_paths[:, i] = c_paths[:, i-1] + kappa*(theta-c_paths[:, i-1])*dt + sigma*np.sqrt(np.maximum(c_paths[:, i-1], 0))*np.random.normal(0, np.sqrt(dt), n_paths)
            c_med = np.percentile(c_paths, 50, axis=0)*100

            # --- 🧪 BACKTESTING (30-DAY) ---
            train_bt, test_bt = yields.iloc[:-30], yields.iloc[-30:]
            model_bt = pm.auto_arima(train_bt, seasonal=False)
            m_bt_pred = model_bt.predict(n_periods=30)

            # --- RENDERING TABS ---
            with tabs[1]: # ARIMA
                st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Actual"))
                fig_a.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="Forecast", line=dict(dash='dot', color='orange')))
                st.plotly_chart(fig_a, width='stretch')

            with tabs[2]: # GARCH
                st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
                fig_g = go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color='red')))
                st.plotly_chart(fig_g, width='stretch')

            with tabs[3]: # VASICEK (RESTORED)
                st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
                st.subheader("Vasicek Mean-Reversion Simulation")
                fig_v = go.Figure(go.Scatter(x=f_dates, y=v_med, name="Vasicek Median Path", line=dict(color='orange')))
                fig_v.add_hline(y=theta*100, line_dash="dash", line_color="gray", annotation_text="Theta (Equilibrium)")
                fig_v.update_layout(yaxis_title="Yield (%)", template="plotly_white")
                st.plotly_chart(fig_v, width='stretch')
                

            with tabs[4]: # CIR (RESTORED)
                st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
                st.subheader("Cox-Ingersoll-Ross (Non-Negative) Simulation")
                fig_c = go.Figure(go.Scatter(x=f_dates, y=c_med, name="CIR Median Path", line=dict(color='green')))
                fig_c.update_layout(yaxis_title="Yield (%)", template="plotly_white")
                st.plotly_chart(fig_c, width='stretch')

            with tabs[5]: # BACKTEST (RESTORED)
                st.markdown(f'<div class="config-info">{config_summary} | Test Window: 30D</div>', unsafe_allow_html=True)
                st.subheader("Walk-Forward Validation")
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt, name="Realized Yield"))
                fig_bt.add_trace(go.Scatter(x=test_bt.index, y=m_bt_pred, name="Predicted Yield", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig_bt, width='stretch')
                rmse = np.sqrt(np.mean((test_bt - m_bt_pred)**2))
                st.success(f"**Root Mean Square Error (RMSE):** {rmse:.4f}%")

            with tabs[6]: # DIAGNOSTICS
                st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
                st.plotly_chart(go.Figure(go.Scatter(y=model_arima.resid(), line=dict(color='gray'))), width='stretch')
                

            with tabs[7]: # METRICS
                st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
                z = stats.norm.ppf(conf_level)
                vol_now = garch_fit.conditional_volatility.iloc[-1]
                var_calc, es_calc = vol_now * z, vol_now * (stats.norm.pdf(z)/(1-conf_level))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spot", f"{yields.iloc[-1]:.3f}%")
                c2.metric("Forecast", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("Daily VaR", f"{var_calc:.3f}%")
                c4.metric("Exp. Shortfall", f"{es_calc:.3f}%")
                

            with tabs[8]: # EXPORT
                export_df = pd.DataFrame({"Date": f_dates.strftime('%Y-%m-%d'), "ARIMA": arima_fc, "Vasicek": v_med, "CIR": c_med})
                st.dataframe(export_df.style.background_gradient(cmap='YlGnBu'), width='stretch')
                st.download_button("📥 Download Data", export_df.to_csv(index=False).encode('utf-8'), "Report.csv")

    except Exception as e:
        st.error(f"Computation Error: {e}")

with tabs[9]: # Q&A Tab
    for q, a in QA_MASTERCLASS:
        with st.expander(q): st.write(a)

display_footer()
