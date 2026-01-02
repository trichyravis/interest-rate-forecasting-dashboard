
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
# 3. ANALYTICS ENGINE & TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast", "🌪️ GARCH Volatility", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Educational Hub"])

# --- TAB 0: DETAILED INSTITUTIONAL METHODOLOGY ---
with tabs[0]:
    st.header("📖 Institutional Research Methodology")
    st.markdown("### About this Platform")
    st.write("""
    This terminal is a quantitative decision-support system designed to bridge the gap between academic theory and institutional practice. 
    It utilizes a **dual-framework approach** to analyze sovereign debt benchmarks, providing a structured thinking process for interest rate scenarios.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Scope & Objectives")
        st.markdown("""
        - **Directional Pathing:** Employs **ARIMA(p,d,q)** to identify momentum and immediate trends.
        - **Risk Estimation:** Utilizes **GARCH(1,1)** to capture volatility clustering—where high-risk periods persist.
        - **Tail-Risk Quantification:** Provides conditional **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**.
        """)
        
    with col2:
        st.subheader("📑 Fundamental Assumptions & Limitations")
        st.markdown("""
        - **Mean Reversion:** Assumes rates gravitate toward long-term equilibrium.
        - **Stationarity:** Yields stabilized through first-order differencing ($d=1$).
        - **Limitations:** Univariate models cannot predict 'Black Swan' events or sudden structural Fed pivots.
        """)

if run_btn:
    data = pd.DataFrame()
    # 🕒 ROBUST 6-STEP RETRY LOGIC (Total potential wait: 125 seconds)
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
        st.error("❌ High Traffic: Yahoo Finance has restricted access. Please wait and try again.")
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

            # TAB 1: FORECAST WITH CONFIG SUMMARY
            with tabs[1]:
                st.markdown(f"""
                <div class="config-box">
                    <strong>Current Configuration:</strong> {ticker_label} | 
                    <strong>Historical Lookback:</strong> {lookback} Years | 
                    <strong>Forecast Horizon:</strong> {horizon} Days | 
                    <strong>Model:</strong> ARIMA{model_arima.order}
                </div>
                """, unsafe_allow_html=True)

                if show_step:
                    fig_step = go.Figure()
                    fig_step.add_trace(go.Scatter(x=f_dates, y=arima_fc, mode='lines+markers', line_shape='hv', 
                                                line=dict(color='#FF4B4B', width=4)))
                    fig_step.update_layout(template="plotly_dark", title="Step-Wise Yield Forecast", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_step, width='stretch')
                else:
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Actual Yield"))
                    fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(dash='dot', color='orange')))
                    fig_f.update_layout(title="Yield Rate Path", template="plotly_white")
                    st.plotly_chart(fig_f, width='stretch')

            # TAB 2: GARCH VOLATILITY
            with tabs[2]:
                st.subheader("🌪️ Conditional Volatility (GARCH 1,1)")
                fig_v = go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color='red')))
                fig_v.update_layout(title="Annualized Conditional Volatility (%)", template="plotly_white")
                st.plotly_chart(fig_v, width='stretch')
                st.info("💡 High peaks represent 'Volatility Shocks'. GARCH helps in pricing risk and calculating Value-at-Risk (VaR).")

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
                st.success(f"**Mean Absolute Error (MAE):** {np.mean(np.abs(test.values - bt_fc.values)):.4f}")

            # TAB 4: DIAGNOSTICS
            with tabs[4]:
                st.subheader("🔍 ARIMA Residual Analysis")
                resid = model_arima.resid()
                fig_resid = go.Figure(go.Scatter(y=resid, mode='lines', line=dict(color='gray')))
                fig_resid.update_layout(title="Standardized Residuals (White Noise Check)", template="plotly_white")
                st.plotly_chart(fig_resid, width='stretch')

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
                fig_r.add_trace(go.Scatter(x=x_d, y=y_d, fill='tozeroy', name='Normal Dist', line=dict(color=CORPORATE_BLUE)))
                fig_r.add_trace(go.Scatter(x=x_d[x_d < -z_score], y=y_d[x_d < -z_score], fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', name='Tail Risk Zone'))
                fig_r.update_layout(title="Tail Risk Visualization", template="plotly_white")
                st.plotly_chart(fig_r, width='stretch')

            # TAB 6: EXPORT
            with tabs[6]:
                export_df = pd.DataFrame({"Date": f_dates, "Forecast": arima_fc})
                st.subheader("📋 Export Terminal")
                st.dataframe(export_df, width='stretch')
                st.download_button("📥 Download Forecast (CSV)", export_df.to_csv().encode('utf-8'), "yield_report.csv")

        except Exception as e:
            st.error(f"Computation Error: {e}")

# --- TAB 7: EXPANDED Q&A HUB ---
with tabs[7]:
    st.header("🎓 Quantitative Finance Q&A Hub")
    st.write("Professional insights bridging academic theory with market practice.")

    with st.expander("❓ What is the Box-Jenkins Methodology (ARIMA)?"):
        st.write("""
        **Definition:** A systematic 3-stage iterative process—Identification, Estimation, and Diagnostics—for fitting models to non-stationary data. 
        It is essential for interest rates because they often exhibit trends and momentum.
        """)
        
    with st.expander("❓ How does GARCH capture 'Volatility Clustering'?"):
        st.write("""
        **Definition:** GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models time-varying variance. 
        It recognizes the empirical reality that high-volatility periods follow high-volatility shocks, allowing for market-sensitive risk metrics.
        """)
        
    with st.expander("❓ Explain Stochastic Models: Vasicek vs CIR?"):
        st.write("""
        **Vasicek:** A continuous-time model assuming interest rates are mean-reverting toward a long-term equilibrium. 
        **CIR:** A successor model that ensures rates stay non-negative by making volatility proportional to the square root of the rate level.
        """)

    with st.expander("❓ What is the significance of the Nelson-Siegel Model?"):
        st.write("""
        **Definition:** A factor-based model that explains the entire yield curve shape using Level, Slope, and Curvature factors. 
        It is the industry workhorse used by central banks for term structure estimation.
        """)
        
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional Edition</p>", unsafe_allow_html=True)
