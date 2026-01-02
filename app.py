
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
    [data-testid="stSidebar"] {{
        background-color: {CORPORATE_BLUE} !important;
        color: white !important;
    }}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{ color: white !important; }}
    div.stButton > button:first-child {{
        background-color: {GOLD} !important;
        color: {CORPORATE_BLUE} !important;
        font-weight: bold !important;
        width: 100%; border-radius: 8px;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: {GOLD} !important; 
        font-weight: bold; 
        color: {CORPORATE_BLUE} !important; 
    }}
    </style>
    <div class="main-header">
        <h1>INTEREST RATE FORECASTING DASHBOARD</h1>
        <p>Prof. V. Ravichandran | The Mountain Path - World of Finance</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR - PROFILE & CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_label = st.selectbox("Benchmark Maturity", ["US 10Y (^TNX)", "US 30Y (^TYX)", "US 5Y (^FVX)"])
    ticker = ticker_label.split("(")[1].replace(")", "")
    lookback = st.slider("Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20)
    
    st.header("🛡️ Risk Parameters")
    conf_level = st.select_slider("VaR Confidence Level (α)", options=[0.90, 0.95, 0.99], value=0.95)
    
    st.header("🎨 UI Settings")
    show_step = st.checkbox("Show Step-Wise Curve", value=True)
    
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

    for _ in range(8): st.write("")
        
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.15); border: 1px solid {GOLD};">
            <h3 style="color: white !important; margin: 0;">Prof. V. Ravichandran</h3>
            <p style="color: white !important; font-size: 0.85rem; margin: 5px 0;">The Mountain Path - World of Finance</p>
            <hr style="margin: 10px 0; border-color: {GOLD};">
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank" style="text-decoration: none;">
                <button style="background-color: #0077b5; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">🔗 LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TABBED INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast", "🌪️ GARCH Volatility", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Educational Hub"])

# --- TAB 0: DETAILED ABOUT ---
with tabs[0]:
    st.header("📖 Institutional Research Methodology")
    st.markdown("""
    This terminal is a quantitative decision-support system designed to bridge the gap between 
    academic theory and fixed-income market practice. It utilizes a **dual-framework approach** to analyze sovereign debt benchmarks.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Scope & Objectives")
        st.markdown("""
        - **Directional Pathing:** ARIMA methodology to identify momentum and mean-reversion.
        - **Risk Estimation:** GARCH(1,1) to model regime-dependent volatility clustering.
        - **Tail-Risk:** Calculation of Value-at-Risk (VaR) and Expected Shortfall (ES).
        """)
    with col2:
        st.subheader("📑 Fundamental Assumptions")
        st.markdown("""
        - **Mean Reversion:** Rates gravitate toward a local trend.
        - **Stationarity:** Yields are stabilized through first-order differencing ($d=1$).
        - **Limitations:** Does not account for exogenous 'Black Swan' events or sudden Fed pivots.
        """)

# --- EXECUTION LOGIC ---
if run_btn:
    data = None
    # 🕒 PROGRESSIVE RETRY LOGIC (5s, 10s, 20s)
    wait_times = [0, 5, 10, 20]
    for attempt, delay in enumerate(wait_times):
        if delay > 0:
            st.warning(f"⚠️ Yahoo Finance busy. Retrying in {delay} seconds (Attempt {attempt})...")
            time.sleep(delay)
        with st.spinner("Fetching Institutional Data..."):
            try:
                data = yf.download(ticker, period=f"{lookback}y", progress=False)
                if not data.empty: break
            except: continue

    if data is None or data.empty:
        st.error("❌ Data retrieval failed. Please try again in a few minutes.")
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
                    fig_step.add_trace(go.Scatter(x=f_dates, y=arima_fc, mode='lines+markers', line_shape='hv', 
                                                line=dict(color='#FF4B4B', width=4), name="Step Curve"))
                    fig_step.update_layout(template="plotly_dark", title="Institutional Step-Wise Forecast", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_step, width='stretch')
                else:
                    fig_main = go.Figure()
                    fig_main.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Actual"))
                    fig_main.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA", line=dict(dash='dot', color='orange')))
                    fig_main.update_layout(template="plotly_white")
                    st.plotly_chart(fig_main, width='stretch')

            # TAB 2: VOLATILITY
            with tabs[2]:
                st.subheader("🌪️ Volatility Clustering (GARCH 1,1)")
                fig_vol = go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, line=dict(color='red')))
                fig_vol.update_layout(template="plotly_white", title="Annualized Conditional Volatility (%)")
                st.plotly_chart(fig_vol, width='stretch')

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
                st.subheader("🔍 ARIMA Residual Diagnostics")
                resid = model_arima.resid()
                fig_resid = go.Figure(go.Scatter(y=resid, mode='lines', line=dict(color='gray')))
                fig_resid.update_layout(template="plotly_white", title="Standardized Residual Errors")
                st.plotly_chart(fig_resid, width='stretch')
                st.info("💡 Residuals should resemble 'White Noise'—random fluctuations around zero.")

            # TAB 5: METRICS (VaR & ES)
            with tabs[5]:
                z_score = stats.norm.ppf(conf_level)
                var_val = latest_vol * z_score
                es_val = latest_vol * (stats.norm.pdf(z_score) / (1 - conf_level))
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Rate", f"{yields.iloc[-1]:.3f}%")
                c2.metric("Forecasted", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("Daily VaR", f"{var_val:.3f}%")
                c4.metric("Exp. Shortfall", f"{es_val:.3f}%")

                x_dist = np.linspace(-5, 5, 200)
                y_dist = stats.norm.pdf(x_dist, 0, 1)
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Scatter(x=x_dist, y=y_dist, fill='tozeroy', name='Normal Dist', line=dict(color=CORPORATE_BLUE)))
                mask = x_dist < -z_score
                fig_risk.add_trace(go.Scatter(x=x_dist[mask], y=y_dist[mask], fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.4)', name='Tail Risk'))
                fig_risk.update_layout(title="Tail Risk Visualization: VaR vs Expected Shortfall Zone", template="plotly_white")
                st.plotly_chart(fig_risk, width='stretch')

            # TAB 6: EXPORT
            with tabs[6]:
                st.subheader("📋 Data Export Terminal")
                export_df = pd.DataFrame({"Date": f_dates, "Forecast": arima_fc})
                st.dataframe(export_df, width='stretch')
                st.download_button("📥 Download Report (CSV)", export_df.to_csv().encode('utf-8'), f"{ticker}_report.csv")

        except Exception as e:
            st.error(f"Computation Error: {e}")

# --- TAB 7: DETAILED Q&A EDUCATIONAL HUB ---
with tabs[7]:
    st.header("🎓 Quantitative Finance Q&A Hub")
    st.write("Excerpts from Prof. V. Ravichandran’s Institutional Series.")

    with st.expander("❓ What is the Box-Jenkins Methodology?"):
        st.write("It is an iterative 3-stage process: Identification, Estimation, and Diagnostics used to fit ARIMA models to non-stationary interest rate data.")
        
    
    with st.expander("❓ Why use GARCH instead of standard Volatility?"):
        st.write("Standard volatility assumes 'Homoscedasticity' (constant risk). GARCH accounts for 'Volatility Clustering,' recognizing that high-risk periods persist.")
        
    
    with st.expander("❓ What is the difference between VaR and Expected Shortfall?"):
        st.write("VaR is a threshold loss. Expected Shortfall (ES) measures the average loss *beyond* that threshold, capturing severe tail-risk.")
        
    
    with st.expander("❓ What are Stochastic Models like Vasicek and CIR?"):
        st.write("These models treat rates as a 'random walk' with mean-reversion. CIR specifically ensures rates stay non-negative.")

    with st.expander("❓ How does the Nelson-Siegel Model fit the Yield Curve?"):
        st.write("It decomposes the curve into Level, Slope, and Curvature factors ($\beta_0, \beta_1, \beta_2$).")
        

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Edition</p>", unsafe_allow_html=True)
