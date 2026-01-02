
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
    st.write("Designed by Prof. V. Ravichandran, this terminal provides a multi-model approach to interest rate forecasting, bridging academic theory and market reality.")
    
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

            with tabs[5]: # BACKTESTING & RMSE CALCULATIONS
                train_bt, test_bt = yields.iloc[:-30], yields.iloc[-30:]
                # Models
                m_arima_bt = pm.auto_arima(train_bt, seasonal=False).predict(n_periods=30)
                # Stochastic benchmarks (small scale for speed)
                v_bt = np.zeros((50, 30)); v_bt[:, 0] = train_bt.iloc[-1]/100
                c_bt = np.zeros((50, 30)); c_bt[:, 0] = train_bt.iloc[-1]/100
                for i in range(1, 30):
                    v_bt[:, i] = v_bt[:, i-1] + kappa*(theta-v_bt[:, i-1])*dt + sigma*np.random.normal(0, np.sqrt(dt), 50)
                    c_bt[:, i] = c_bt[:, i-1] + kappa*(theta-c_bt[:, i-1])*dt + sigma*np.sqrt(np.maximum(c_bt[:, i-1],0))*np.random.normal(0, np.sqrt(dt), 50)
                m_v_bt = np.percentile(v_bt, 50, axis=0)*100
                m_c_bt = np.percentile(c_bt, 50, axis=0)*100

                rmse_a = np.sqrt(np.mean((test_bt.values - m_arima_bt.values)**2))
                rmse_v = np.sqrt(np.mean((test_bt.values - m_v_bt)**2))
                rmse_c = np.sqrt(np.mean((test_bt.values - m_c_bt)**2))

                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt, name="Market Actual"))
                fig_bt.add_trace(go.Scatter(x=test_bt.index, y=m_arima_bt, name="ARIMA Prediction", line=dict(dash='dash')))
                st.plotly_chart(fig_bt, width='stretch')

            with tabs[7]: # METRICS & VAR/ES
                z_score = stats.norm.ppf(conf_level)
                latest_vol_daily = garch_fit.conditional_volatility.iloc[-1]
                var_val = latest_vol_daily * z_score
                es_val = latest_vol_daily * (stats.norm.pdf(z_score)/(1-conf_level))
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spot Rate", f"{yields.iloc[-1]:.3f}%")
                c2.metric("ARIMA Forecast", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("Daily VaR", f"{var_val:.3f}%")
                c4.metric("Exp. Shortfall (ES)", f"{es_val:.3f}%")

                st.markdown("### 🏆 Model Performance Summary (RMSE)")
                perf_df = pd.DataFrame({
                    "Model": ["ARIMA (Momentum)", "Vasicek (Stochastic)", "CIR (Non-Negative)"],
                    "RMSE (%)": [f"{rmse_a:.4f}", f"{rmse_v:.4f}", f"{rmse_c:.4f}"],
                    "Status": ["✅ Best" if rmse_a == min(rmse_a, rmse_v, rmse_c) else "Secondary",
                              "✅ Best" if rmse_v == min(rmse_a, rmse_v, rmse_c) else "Secondary",
                              "✅ Best" if rmse_c == min(rmse_a, rmse_v, rmse_c) else "Secondary"]
                })
                st.table(perf_df)

                x_d = np.linspace(-4, 4, 200); y_d = stats.norm.pdf(x_d, 0, 1)
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=x_d, y=y_d, fill='tozeroy', name='Standard Normal', line=dict(color=CORPORATE_BLUE)))
                fig_r.add_trace(go.Scatter(x=x_d[x_d < -z_score], y=y_d[x_d < -z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', name='Tail Risk Zone'))
                fig_r.update_layout(title="Tail Risk Visualization (VaR Zone)", template="plotly_white")
                st.plotly_chart(fig_r, width='stretch')

            with tabs[8]: # COLORFUL EXPORT
                st.subheader("📋 Colorful Data Export Terminal")
                export_df = pd.DataFrame({
                    "Date": f_dates.strftime('%Y-%m-%d'), 
                    "ARIMA (%)": arima_fc.values.round(4), 
                    "Vasicek (%)": v_med.round(4), 
                    "CIR (%)": c_med.round(4)
                })
                
                def color_logic(val):
                    color = '#FFD700' if val > yields.iloc[-1] else '#ADD8E6'
                    return f'background-color: {color}; color: black; font-weight: bold'

                st.dataframe(export_df.style.applymap(color_logic, subset=['ARIMA (%)', 'Vasicek (%)', 'CIR (%)']), width='stretch')
                st.download_button("📥 Download Full Report (CSV)", export_df.to_csv(index=False).encode('utf-8'), "multi_model_report.csv")

        except Exception as e: st.error(f"Execution Error: {e}")

# --- TAB 9: COMPREHENSIVE Q&A HUB ---
with tabs[9]:
    st.header("🎓 Quantitative Knowledge Base: 15 Core Questions")
    
    qa = [
        ("1. What is the fundamental objective of interest rate risk modeling?", "The primary objective is to quantify the potential impact of fluctuating interest rates on the value of a financial instrument, portfolio, or balance sheet, enabling informed hedging and capital allocation."),
        ("2. How does the ARIMA model utilize the Box-Jenkins methodology?", "It follows a three-stage iterative process: Identification (checking for stationarity), Estimation (selecting p, d, q parameters), and Diagnostic Checking (ensuring residuals are white noise)."),
        ("3. Why is 'Mean Reversion' a core assumption in Stochastic models?", "Economic theory suggests that interest rates cannot drift to infinity or stay at zero forever; they are pulled back to a long-term equilibrium by central bank policy and economic fundamentals."),
        ("4. What is the primary 'Negative Rate' flaw in the Vasicek Model?", "The Vasicek model assumes constant volatility. If the current rate is very low, a large negative random shock can mathematically push the simulated rate below zero."),
        ("5. How does the CIR model solve the Vasicek limitation?", "The CIR model makes volatility proportional to the square root of the rate ($\sigma\sqrt{r_t}$). As the rate approaches zero, volatility also approaches zero, preventing negative outcomes."),
        ("6. What is the difference between Value-at-Risk (VaR) and Expected Shortfall (ES)?", "VaR is a threshold measure (the 'minimum' loss in the worst 5% of cases). ES is a coherent risk measure that calculates the 'average' loss within that worst 5% zone."),
        ("7. What is 'Volatility Clustering' in GARCH models?", "It is the empirical observation that 'large changes tend to be followed by large changes, and small by small.' GARCH captures these high and low volatility regimes."),
        ("8. How is RMSE used to evaluate model performance?", "Root Mean Square Error (RMSE) calculates the square root of the average squared difference between predicted and actual values. A lower RMSE indicates higher predictive accuracy."),
        ("9. What is the role of 'Kappa' ($\kappa$) in a Vasicek simulation?", "Kappa represents the 'speed of mean reversion.' It determines how quickly the interest rate is pulled back to its long-term equilibrium ($\theta$) after a shock."),
        ("10. Why do we difference yield data ($d=1$) in ARIMA?", "Most interest rate series are non-stationary (they have a trend). Differencing stabilizes the mean, making the data suitable for AR and MA modeling."),
        ("11. What is a 'Monte Carlo' simulation in this context?", "It involves generating thousands of random paths based on a specific stochastic differential equation (like CIR) to observe the probability distribution of future rates."),
        ("12. How does the confidence level ($\alpha$) impact VaR?", "A higher confidence level (e.g., 99% vs 95%) results in a larger VaR, as you are looking deeper into the extreme 'tail' of the probability distribution."),
        ("13. What are 'Exogenous' variables in interest rate modeling?", "These are factors outside the model, such as inflation (CPI), employment data (NFP), or geopolitical events that significantly influence rate movements but aren't captured by univariate models."),
        ("14. What is the significance of the 'Long-term Equilibrium' ($\theta$)?", "Theta represents the theoretical level where interest rates should settle in a steady-state economy, often influenced by the central bank's inflation target and neutral rate."),
        ("15. Why is the 'White Noise' check important in diagnostics?", "If residuals are not white noise, it means the model has failed to capture some pattern or 'signal' in the data, implying the model parameters need adjustment.")
    ]

    for q, a in qa:
        with st.expander(q):
            st.write(a)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Edition</p>", unsafe_allow_html=True)
