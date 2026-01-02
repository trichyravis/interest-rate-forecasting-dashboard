
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import pmdarima as pm
from arch import arch_model 
import scipy.stats as stats
import warnings
import time
import os

# --- MODULAR IMPORT LOGIC ---
# This ensures that if the content files are missing, the app doesn't crash
try:
    from content.about_text import ABOUT_CONTENT
    from content.qa_text import QA_MASTERCLASS
except ImportError:
    ABOUT_CONTENT = {"intro": "Methodology Content Missing", "arima": "", "vasicek": "", "cir": ""}
    QA_MASTERCLASS = [("Error", "Content files not found in /content/ folder.")]

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
# 3. DASHBOARD TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 ARIMA", "🌪️ GARCH", "🎲 Vasicek", "☀️ CIR", "🧪 Backtest", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Masterclass"])

with tabs[0]:
    st.header("📖 Institutional Methodology")
    st.write(ABOUT_CONTENT["intro"])
    c1, c2, c3 = st.columns(3)
    with c1: st.info(f"**ARIMA (Momentum)**\n\n{ABOUT_CONTENT['arima']}")
    with c2: st.warning(f"**Vasicek (Equilibrium)**\n\n{ABOUT_CONTENT['vasicek']}")
    with c3: st.success(f"**CIR (Non-Negative)**\n\n{ABOUT_CONTENT['cir']}")

if run_btn:
    data = pd.DataFrame()
    wait_times = [0, 5, 10, 20, 30, 60] 
    success = False

    for attempt, delay in enumerate(wait_times):
        if delay > 0:
            st.warning(f"⚠️ Rate limited. Retrying in {delay}s...")
            time.sleep(delay)
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
            # --- ENGINES ---
            model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
            arima_fc = model_arima.predict(n_periods=horizon)
            f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
            
            garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
            cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)

            r0, kappa, theta, sigma = yields.iloc[-1]/100, 0.20, 0.045, 0.015
            dt, n_paths = 1/252, 1000

            # 📈 ARIMA
            with tabs[1]:
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="History"))
                fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="Forecast", line=dict(dash='dot', color='orange')))
                st.plotly_chart(fig_f, use_container_width=True)

            # 🌪️ GARCH
            with tabs[2]:
                st.plotly_chart(go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color='red'))), use_container_width=True)

            # 🎲 VASICEK
            with tabs[3]:
                v_paths = np.zeros((n_paths, horizon))
                v_paths[:, 0] = r0
                for i in range(1, horizon):
                    v_paths[:, i] = v_paths[:, i-1] + kappa*(theta-v_paths[:, i-1])*dt + sigma*np.random.normal(0, np.sqrt(dt), n_paths)
                v_med = np.percentile(v_paths, 50, axis=0)*100
                st.plotly_chart(go.Figure(go.Scatter(x=f_dates, y=v_med, name="Vasicek Median")), use_container_width=True)

            # ☀️ CIR
            with tabs[4]:
                c_paths = np.zeros((n_paths, horizon))
                c_paths[:, 0] = r0
                for i in range(1, horizon):
                    c_paths[:, i] = c_paths[:, i-1] + kappa*(theta-c_paths[:, i-1])*dt + sigma*np.sqrt(np.maximum(c_paths[:, i-1],0))*np.random.normal(0, np.sqrt(dt), n_paths)
                c_med = np.percentile(c_paths, 50, axis=0)*100
                st.plotly_chart(go.Figure(go.Scatter(x=f_dates, y=c_med, name="CIR Median", line=dict(color='green'))), use_container_width=True)

            # 🧪 BACKTESTING
            with tabs[5]:
                train_bt, test_bt = yields.iloc[:-30], yields.iloc[-30:]
                m_bt = pm.auto_arima(train_bt, seasonal=False).predict(n_periods=30)
                rmse_bt = np.sqrt(np.mean((test_bt.values - m_bt.values)**2))
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt, name="Market"))
                fig_bt.add_trace(go.Scatter(x=test_bt.index, y=m_bt, name="ARIMA Forecast", line=dict(dash='dash')))
                st.plotly_chart(fig_bt, use_container_width=True)
                st.success(f"**Walk-Forward RMSE:** {rmse_bt:.4f}%")

            # 🔍 DIAGNOSTICS
            with tabs[6]:
                resid = model_arima.resid()
                st.plotly_chart(go.Figure(go.Scatter(y=resid, mode='lines', line=dict(color='gray'))), use_container_width=True)
                st.info("💡 Residuals should appear as random white noise around zero.")

            # 📊 METRICS & TAIL RISK
            with tabs[7]:
                z_score = stats.norm.ppf(conf_level)
                latest_vol_daily = garch_fit.conditional_volatility.iloc[-1]
                var_val = latest_vol_daily * z_score
                es_val = latest_vol_daily * (stats.norm.pdf(z_score)/(1-conf_level))
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spot Rate", f"{yields.iloc[-1]:.3f}%")
                c2.metric("ARIMA Forecast", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("Daily VaR", f"{var_val:.3f}%")
                c4.metric("Exp. Shortfall (ES)", f"{es_val:.3f}%")

                x_d = np.linspace(-4, 4, 200); y_d = stats.norm.pdf(x_d, 0, 1)
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=x_d, y=y_d, fill='tozeroy', name='Normal', line=dict(color=CORPORATE_BLUE)))
                fig_r.add_trace(go.Scatter(x=x_d[x_d < -z_score], y=y_d[x_d < -z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', name='Tail Risk Zone'))
                st.plotly_chart(fig_r, use_container_width=True)

            # 📋 AMENDED EXPORT TAB
            with tabs[8]:
                st.subheader("📋 Comprehensive Institutional Quantitative Report")
                export_df = pd.DataFrame({
                    "Date": f_dates.strftime('%Y-%m-%d'),
                    "ARIMA Forecast (%)": arima_fc.values.round(4),
                    "Vasicek Median (%)": v_med.round(4),
                    "CIR Median (%)": c_med.round(4),
                    "Expected Volatility (%)": (cond_vol.tail(len(f_dates)).values).round(4)
                })

                st.dataframe(export_df.style.background_gradient(cmap='YlGnBu').format(precision=4), use_container_width=True)
                st.download_button("📥 Download Quantitative Report (CSV)", export_df.to_csv(index=False).encode('utf-8'), f"{ticker}_report.csv")

        except Exception as e: st.error(f"Execution Error: {e}")

# 📚 MASTERCLASS Q&A
with tabs[9]:
    st.header("🎓 Masterclass Q&A")
    for q, a in QA_MASTERCLASS:
        with st.expander(q): st.write(a)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Edition</p>", unsafe_allow_html=True)
