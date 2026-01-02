
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
# 1. PAGE CONFIG & THEME (OXFORD BLUE & GOLD)
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
        border-left: 5px solid {CORPORATE_BLUE}; margin-bottom: 20px; color: {CORPORATE_BLUE};
    }}
    </style>
    <div class="main-header">
        <h1 style="margin-bottom: 0;">INTEREST RATE FORECASTING DASHBOARD</h1>
        <h2 style="margin-top: 0; font-size: 1.5rem; opacity: 0.9;">(Multi-Model Institutional Terminal)</h2>
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
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 30)
    st.header("🛡️ Risk Parameters")
    conf_level = st.select_slider("Confidence Level (α)", options=[0.90, 0.95, 0.99], value=0.95)
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

    for _ in range(5): st.write("")
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.15); border: 1px solid {GOLD};">
            <h3 style="color: white !important; margin: 0;">Prof. V. Ravichandran</h3>
            <hr style="margin: 10px 0; border-color: {GOLD};">
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank">
                <button style="background-color: #0077b5; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">🔗 LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE & TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About Platform", "📈 ARIMA Forecast", "🌪️ GARCH Risk", "🎲 Vasicek Path", "☀️ CIR Path", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Masterclass"])

with tabs[0]:
    st.header("📖 Institutional Methodology & Framework")
    st.markdown("### 1. About the Dashboard")
    st.write("Designed by Prof. V. Ravichandran to bridge academic theory with institutional practice. This terminal uses ARIMA for technical pathing, GARCH for risk regimes, and Vasicek/CIR for stochastic equilibrium.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📊 ARIMA")
        st.markdown("**Scope:** Short-term momentum.\n**Assumptions:** Weak-form efficiency.\n**Limits:** Fails at structural pivots.")
    with col2:
        st.subheader("🎲 Vasicek")
        st.markdown("**Scope:** Mean-reversion pathing.\n**Assumptions:** Constant volatility.\n**Limits:** Rates can become negative.")
    with col3:
        st.subheader("☀️ CIR")
        st.markdown("**Scope:** Low-rate environments.\n**Assumptions:** Rate-dependent volatility.\n**Limits:** Ignores regime jumps.")

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

    if success:
        yields = data['Close'].dropna()
        if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
        yields = yields.resample('B').last().ffill()
        returns = 100 * yields.pct_change().dropna()

        try:
            # --- MODEL ENGINES ---
            model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
            arima_fc = model_arima.predict(n_periods=horizon)
            f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
            
            garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
            cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)

            r0, kappa, theta, sigma = yields.iloc[-1]/100, 0.20, 0.045, 0.015
            dt, n_paths = 1/252, 1000

            # --- POPULATE TABS ---
            with tabs[1]: # ARIMA
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="History"))
                fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="Forecast", line=dict(dash='dot', color='orange')))
                st.plotly_chart(fig_f, width='stretch')

            with tabs[2]: # GARCH
                fig_v = go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color='red')))
                st.plotly_chart(fig_v, width='stretch')

            with tabs[3]: # VASICEK
                v_paths = np.zeros((n_paths, horizon))
                v_paths[:, 0] = r0
                for i in range(1, horizon):
                    v_paths[:, i] = v_paths[:, i-1] + kappa * (theta - v_paths[:, i-1]) * dt + sigma * np.random.normal(0, np.sqrt(dt), n_paths)
                v_med = np.percentile(v_paths, 50, axis=0)*100
                fig_vas = go.Figure(go.Scatter(x=f_dates, y=v_med, name="Vasicek Median", line=dict(color='orange')))
                st.plotly_chart(fig_vas, width='stretch')

            with tabs[4]: # CIR
                c_paths = np.zeros((n_paths, horizon))
                c_paths[:, 0] = r0
                for i in range(1, horizon):
                    c_paths[:, i] = c_paths[:, i-1] + kappa * (theta - c_paths[:, i-1]) * dt + sigma * np.sqrt(np.maximum(c_paths[:, i-1], 0)) * np.random.normal(0, np.sqrt(dt), n_paths)
                c_med = np.percentile(c_paths, 50, axis=0)*100
                fig_cir = go.Figure(go.Scatter(x=f_dates, y=c_med, name="CIR Median", line=dict(color='green')))
                st.plotly_chart(fig_cir, width='stretch')

            with tabs[5]: # BACKTESTING
                st.subheader("🧪 30-Day Walk-Forward Validation")
                train_bt, test_bt = yields.iloc[:-30], yields.iloc[-30:]
                # Benchmarking all 3 models for the Performance Summary
                m_arima_bt = pm.auto_arima(train_bt, seasonal=False).predict(n_periods=30)
                
                # Simple stochastic medians for backtest
                v_paths_bt = np.zeros((100, 30)); v_paths_bt[:, 0] = train_bt.iloc[-1]/100
                c_paths_bt = np.zeros((100, 30)); c_paths_bt[:, 0] = train_bt.iloc[-1]/100
                for i in range(1, 30):
                    v_paths_bt[:, i] = v_paths_bt[:, i-1] + kappa * (theta - v_paths_bt[:, i-1]) * dt + sigma * np.random.normal(0, np.sqrt(dt), 100)
                    c_paths_bt[:, i] = c_paths_bt[:, i-1] + kappa * (theta - c_paths_bt[:, i-1]) * dt + sigma * np.sqrt(np.maximum(c_paths_bt[:, i-1], 0)) * np.random.normal(0, np.sqrt(dt), 100)
                m_v_bt = np.percentile(v_paths_bt, 50, axis=0)*100
                m_c_bt = np.percentile(c_paths_bt, 50, axis=0)*100

                rmse_a = np.sqrt(np.mean((test_bt.values - m_arima_bt.values)**2))
                rmse_v = np.sqrt(np.mean((test_bt.values - m_v_bt)**2))
                rmse_c = np.sqrt(np.mean((test_bt.values - m_c_bt)**2))

                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt, name="Market Actual"))
                fig_bt.add_trace(go.Scatter(x=test_bt.index, y=m_arima_bt, name="ARIMA Prediction", line=dict(dash='dash')))
                st.plotly_chart(fig_bt, width='stretch')

            with tabs[6]: # DIAGNOSTICS
                resid = model_arima.resid()
                st.plotly_chart(go.Figure(go.Scatter(y=resid, mode='lines', line=dict(color='gray'))), width='stretch')

            with tabs[7]: # METRICS & PERFORMANCE SUMMARY
                st.subheader(f"📊 Quantitative Risk Metrics (α={conf_level})")
                z_score = stats.norm.ppf(conf_level)
                latest_vol_daily = garch_fit.conditional_volatility.iloc[-1]
                var_val = latest_vol_daily * z_score
                es_val = latest_vol_daily * (stats.norm.pdf(z_score)/(1-conf_level))
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Rate", f"{yields.iloc[-1]:.3f}%")
                c2.metric("ARIMA Forecast", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("Daily VaR", f"{var_val:.3f}%")
                c4.metric("Expected Shortfall", f"{es_val:.3f}%")

                st.markdown("### 🏆 Model Performance Summary (RMSE)")
                perf_df = pd.DataFrame({
                    "Model": ["ARIMA (Momentum)", "Vasicek (Stochastic)", "CIR (Non-Negative)"],
                    "RMSE (%)": [rmse_a, rmse_v, rmse_c],
                    "Status": ["✅ Best" if rmse_a == min(rmse_a, rmse_v, rmse_c) else "Secondary",
                              "✅ Best" if rmse_v == min(rmse_a, rmse_v, rmse_c) else "Secondary",
                              "✅ Best" if rmse_c == min(rmse_a, rmse_v, rmse_c) else "Secondary"]
                })
                st.table(perf_df)

                x_d = np.linspace(-4, 4, 200); y_d = stats.norm.pdf(x_d, 0, 1)
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=x_d, y=y_d, fill='tozeroy', name='Normal', line=dict(color=CORPORATE_BLUE)))
                fig_r.add_trace(go.Scatter(x=x_d[x_d < -z_score], y=y_d[x_d < -z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', name='Tail Risk'))
                st.plotly_chart(fig_r, width='stretch')

            with tabs[8]: # COLORFUL EXPORT
                st.subheader("📋 Colorful Data Export Terminal")
                export_df = pd.DataFrame({
                    "Date": f_dates.strftime('%Y-%m-%d'), 
                    "ARIMA": arima_fc.values.round(4), 
                    "Vasicek": v_med.round(4), 
                    "CIR": c_med.round(4)
                })
                
                def color_models(val):
                    color = 'gold' if val > yields.iloc[-1] else 'lightblue'
                    return f'background-color: {color}'

                st.dataframe(export_df.style.applymap(color_models, subset=['ARIMA', 'Vasicek', 'CIR']), width='stretch')
                st.download_button("📥 Download Colorful Report (CSV)", export_df.to_csv(index=False).encode('utf-8'), "multi_model_report.csv")

        except Exception as e: st.error(f"Error: {e}")

with tabs[9]:
    st.header("🎓 Q&A Masterclass")
    with st.expander("❓ What is RMSE?"): st.write("Root Mean Square Error (RMSE) measures the average magnitude of error between predicted and actual values.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance</p>", unsafe_allow_html=True)
