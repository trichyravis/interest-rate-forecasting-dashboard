
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

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & THEME
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Interest Rate Forecasting Dashboard", layout="wide")

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
        background-color: {GOLD} !important;
        color: {CORPORATE_BLUE} !important;
        font-weight: bold !important;
        width: 100%; border-radius: 8px;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; font-weight: bold; color: {CORPORATE_BLUE} !important; }}
    </style>
    <div class="main-header">
        <h1>INTEREST RATE FORECASTING DASHBOARD</h1>
        <p>The Mountain Path - World of Finance | Institutional Research Terminal</p>
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
# 3. ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 Forecast", "🌪️ GARCH Volatility", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A Educational Hub"])

with tabs[0]:
    st.header("📖 Institutional Methodology")
    st.write("This terminal provides a dual-framework analysis for sovereign debt benchmarks using Prof. Ravichandran’s quantitative standards.")

if run_btn:
    with st.spinner("Processing Market Risk Engines..."):
        data = yf.download(ticker, period=f"{lookback}y", progress=False)
        
        if not data.empty:
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()
            returns = 100 * yields.pct_change().dropna()

            try:
                # 1. Fit ARIMA & GARCH
                model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
                arima_fc = model_arima.predict(n_periods=horizon)
                f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
                
                garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
                latest_vol = garch_fit.conditional_volatility.iloc[-1]
                cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)

                # 2. Populate Tabs
                with tabs[1]: # Forecast
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Actual Yield"))
                    fig_f.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="ARIMA Forecast", line=dict(dash='dot', color='orange')))
                    fig_f.update_layout(title=f"{ticker} Forecast Path", template="plotly_white")
                    st.plotly_chart(fig_f, use_container_width=True)

                with tabs[2]: # GARCH
                    st.subheader("🌪️ Volatility Clustering Analysis")
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol, name="Ann. Volatility", line=dict(color='red')))
                    st.plotly_chart(fig_vol, use_container_width=True)

                with tabs[3]: # Backtesting
                    st.subheader("🧪 30-Day Walk-Forward Validation")
                    train_bt, test_bt = yields.iloc[:-30], yields.iloc[-30:]
                    bt_model = pm.auto_arima(train_bt, seasonal=False)
                    bt_fc = bt_model.predict(n_periods=30)
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt, name="Realized Market Data", line=dict(color=CORPORATE_BLUE)))
                    fig_bt.add_trace(go.Scatter(x=test_bt.index, y=bt_fc, name="Model Prediction", line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig_bt, use_container_width=True)

                with tabs[4]: # Diagnostics
                    st.subheader("🔍 Residual Analysis")
                    resid = model_arima.resid()
                    fig_resid = go.Figure(go.Scatter(y=resid, mode='lines', line=dict(color='gray')))
                    fig_resid.update_layout(title="Standardized Residuals (White Noise Check)", template="plotly_white")
                    st.plotly_chart(fig_resid, use_container_width=True)

                with tabs[5]: # Metrics & VaR
                    z_score = stats.norm.ppf(conf_level)
                    var_val = latest_vol * z_score
                    es_val = latest_vol * (stats.norm.pdf(z_score) / (1 - conf_level))
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Rate", f"{yields.iloc[-1]:.3f}%")
                    c2.metric("Ann. Volatility", f"{cond_vol.iloc[-1]:.1f}%")
                    c3.metric("Daily VaR", f"{var_val:.3f}%")
                    c4.metric("Exp. Shortfall", f"{es_val:.3f}%")

                    # Risk Distribution Plot
                    x = np.linspace(-5, 5, 200)
                    y = stats.norm.pdf(x, 0, 1)
                    fig_risk = go.Figure()
                    fig_risk.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='Normal Dist', line=dict(color=CORPORATE_BLUE)))
                    mask_tail = x < -z_score
                    fig_risk.add_trace(go.Scatter(x=x[mask_tail], y=y[mask_tail], fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.5)', name='Risk Zone'))
                    fig_risk.update_layout(title="Tail Risk: VaR vs Expected Shortfall", template="plotly_white")
                    st.plotly_chart(fig_risk, use_container_width=True)

                with tabs[6]: # Export
                    export_df = pd.DataFrame({"Date": f_dates, "Forecast": arima_fc})
                    st.dataframe(export_df, use_container_width=True)
                    st.download_button("Download CSV", export_df.to_csv().encode('utf-8'), "forecast_report.csv")

            except Exception as e:
                st.error(f"Computation Error: {e}")

# 📚 DETAILED Q&A EDUCATIONAL HUB (Outside the run_btn block to be always accessible)
with tabs[7]:
    st.header("🎓 Quantitative Finance Q&A Hub")
    st.write("Bridging Academic Theory with Institutional Practice.")

    with st.expander("❓ What is the Box-Jenkins Methodology and why is it used here?"):
        st.write("""
        The Box-Jenkins methodology refers to the systematic process of identifying, estimating, and checking **ARIMA** models.
        It is used here because interest rates often exhibit 'momentum' (Autoregression) and 'trends' (Integration).
        By following this 3-stage process (Identification, Estimation, Diagnostics), we ensure the model is 'parsimonious'—meaning it uses the fewest parameters possible to achieve the highest accuracy.
        """)
        

    with st.expander("❓ How does GARCH differ from standard Volatility measures?"):
        st.write("""
        Standard volatility (like Standard Deviation) assumes that risk is constant over time (**Homoskedasticity**). 
        However, financial markets exhibit **Volatility Clustering**—where high-volatility days are likely to be followed by more high-volatility days.
        **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** models this time-varying risk, allowing for more accurate pricing of derivatives and risk management.
        """)
        

    with st.expander("❓ What is Value-at-Risk (VaR) vs. Expected Shortfall (ES)?"):
        st.write("""
        **VaR** answers: "What is the minimum loss I can expect with 95% confidence?" It is a threshold.
        **Expected Shortfall (ES)** answers: "If I exceed my VaR threshold, what is the average magnitude of that loss?"
        Institutional risk managers prefer ES because it captures 'Tail Risk' more effectively than VaR.
        """)
        

    with st.expander("❓ What are Stochastic Models like Vasicek and CIR?"):
        st.write("""
        Stochastic models treat interest rates as a continuous 'random walk' with mean-reverting properties.
        - **Vasicek Model:** Assumes rates revert to a long-term mean but can technically become negative.
        - **Cox-Ingersoll-Ross (CIR):** Improves on Vasicek by ensuring rates stay positive, as volatility increases with the square root of the rate level.
        """)
        

    with st.expander("❓ How does the Nelson-Siegel Model help in Yield Curve Fitting?"):
        st.write("""
        The Nelson-Siegel model decomposes the yield curve into three interpretable factors:
        1. **Level ($\beta_0$):** The long-term rate.
        2. **Slope ($\beta_1$):** The short-term spread.
        3. **Curvature ($\beta_2$):** The 'hump' in the medium-term rates.
        This is the industry standard used by central banks to estimate the term structure of interest rates.
        """)
        

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Institutional US Edition</p>", unsafe_allow_html=True)
