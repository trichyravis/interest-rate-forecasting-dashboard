
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go
import pmdarima as pm
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & DATA CACHING (To fix YFRateLimitError)
# ═══════════════════════════════════════════════════════════════════════════════
DARK_BLUE = "#003366"
LIGHT_BLUE = "#0066CC"
GOLD_COLOR = "#FFD700"
BRAND_NAME = "The Mountain Path - World of Finance"

# Improved Ticker List for Stability
RATE_TICKERS = {
    "IN10Y.NS": "India 10Y Benchmark (NSE Feed)",
    "^TNX": "US 10Y Treasury Yield",
    "^TYX": "US 30Y Treasury Yield",
    "^FVX": "US 5Y Treasury Yield",
    "INR=X": "USD/INR Exchange Rate (Proxy for Macro Sentiment)"
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yield_data(ticker, days_back):
    """Caches data for 1 hour to prevent Yahoo Finance Rate Limiting."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

st.set_page_config(page_title="Interest Rate Dashboard - The Mountain Path", page_icon="🏦", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
    <style>
    .hero-title {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4); border: 4px solid {DARK_BLUE}; color: white; text-align: center; }}
    [data-testid="stSidebar"] {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; }}
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{ color: white !important; font-weight: 600 !important; }}
    .stButton>button {{ background-color: {GOLD_COLOR} !important; color: {DARK_BLUE} !important; font-weight: bold !important; border-radius: 10px !important; width: 100%; }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='hero-title'><h1>INTEREST RATE FORECASTING DASHBOARD</h1><p>Traditional Time Series Modeling for Global & Indian Yields</p><p>Prof. V. Ravichandran | 28+ Years Finance Experience</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏦 Rate Selection")
    ticker = st.selectbox("Select Benchmark", options=list(RATE_TICKERS.keys()), format_func=lambda x: f"{x} - {RATE_TICKERS[x]}")
    lookback = st.selectbox("Historical Data Range", [1, 2, 3, 5, 10], index=2)
    freq = st.radio("Data Frequency", ["Daily", "Weekly", "Monthly"])
    
    st.markdown("### ⚙️ ARIMA Configuration")
    transformation = st.radio("Yield Transformation", ["Yield Level (%)", "Yield Changes (First Difference)", "Log Yields"], index=0)
    model_mode = st.radio("Model Selection", ["Manual ARIMA", "Auto ARIMA"], index=1)
    
    p, d, q = 1, 1, 1
    if model_mode == "Manual ARIMA":
        c1, c2, c3 = st.columns(3)
        p, d, q = c1.slider("p", 0, 5, 1), c2.slider("d", 0, 2, 1), c3.slider("q", 0, 5, 1)
    
    st.markdown("### 🔮 Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (Periods)", 1, 60, 10)
    refresh_button = st.button("🔄 RUN ANALYSIS")
    
    st.markdown("---")
    st.markdown("### Prof. V. Ravichandran")
    st.markdown(f"<a href='https://www.linkedin.com/in/trichyravis' target='_blank' style='display: block; padding: 0.5rem; background: #0077b5; color: white; text-align: center; text-decoration: none; border-radius: 5px; font-weight: bold;'>🔗 LinkedIn Profile</a>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING & TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Forecast", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "⚠️ Assumptions", "📋 Export", "📚 Educational Hub"])
results = None

if refresh_button:
    with st.spinner("Accessing Financial Terminal..."):
        data = fetch_yield_data(ticker, lookback*365)
        
        if not data.empty:
            raw_yields = data['Close'][ticker] if isinstance(data.columns, pd.MultiIndex) else data['Close']
            raw_yields = raw_yields.dropna()
            res_map = {"Daily": "B", "Weekly": "W", "Monthly": "M"}
            raw_yields = raw_yields.resample(res_map[freq]).last().ffill()
            
            # Basis Point Calculation Info
            ann_factor = 252 if freq == "Daily" else 52 if freq == "Weekly" else 12
            ann_vol = raw_yields.pct_change().dropna().std() * np.sqrt(ann_factor)

            # Transformation logic
            def apply_trans(series, t):
                if t == "Yield Changes (First Difference)": return series.diff().dropna()
                if t == "Log Yields": return np.log(series)
                return series

            train_series = apply_trans(raw_yields, transformation)

            try:
                if model_mode == "Auto ARIMA":
                    model = pm.auto_arima(train_series, seasonal=False)
                    fc, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
                    order, aic, fit_obj = model.order, model.aic(), model
                else:
                    fit = ARIMA(train_series, order=(p, d, q)).fit()
                    fc_res = fit.get_forecast(steps=forecast_horizon)
                    fc, conf_int = fc_res.predicted_mean, fc_res.conf_int(alpha=0.05)
                    order, aic, fit_obj = (p, d, q), fit.aic, fit

                # Reversion (Inversion)
                last_val = raw_yields.iloc[-1]
                if transformation == "Yield Changes (First Difference)":
                    inv_fc = last_val + np.cumsum(fc)
                    inv_low = last_val + np.cumsum(conf_int[:, 0] if model_mode == "Auto ARIMA" else conf_int.iloc[:, 0])
                    inv_high = last_val + np.cumsum(conf_int[:, 1] if model_mode == "Auto ARIMA" else conf_int.iloc[:, 1])
                elif transformation == "Log Yields":
                    inv_fc = np.exp(fc)
                    inv_low = np.exp(conf_int[:, 0] if model_mode == "Auto ARIMA" else conf_int.iloc[:, 0])
                    inv_high = np.exp(conf_int[:, 1] if model_mode == "Auto ARIMA" else conf_int.iloc[:, 1])
                else:
                    inv_fc, inv_low, inv_high = fc, (conf_int[:, 0] if model_mode == "Auto ARIMA" else conf_int.iloc[:, 0]), (conf_int[:, 1] if model_mode == "Auto ARIMA" else conf_int.iloc[:, 1])

                f_dates = pd.date_range(raw_yields.index[-1], periods=forecast_horizon + 1, freq=res_map[freq])[1:]
                fc_df = pd.DataFrame({"Forecasted Yield (%)": np.array(inv_fc).flatten(), "Lower CI": np.array(inv_low).flatten(), "Upper CI": np.array(inv_high).flatten()}, index=f_dates)
                bps_move = (fc_df["Forecasted Yield (%)"].iloc[-1] - raw_yields.iloc[-1]) * 100

                results = {"raw": raw_yields, "fc_df": fc_df, "order": order, "aic": aic, "vol": ann_vol, "bps": bps_move, "resid": fit_obj.resid() if model_mode == "Auto ARIMA" else fit_obj.resid}
            except Exception as e: st.error(f"Computation Error: {e}")
        else:
            st.error("India RBI Ticker currently rate-limited by Yahoo. Please try a US Benchmark or refresh in 10 minutes.")

if results:
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Actual Yield", line=dict(color=DARK_BLUE)))
        fig.add_trace(go.Scatter(x=results["fc_df"].index.tolist()+results["fc_df"].index.tolist()[::-1], 
                                 y=results["fc_df"]["Upper CI"].tolist()+results["fc_df"]["Lower CI"].tolist()[::-1],
                                 fill='toself', fillcolor='rgba(255,165,0,0.1)', line=dict(color='rgba(255,255,255,0)'), name="95% CI Band"))
        fig.add_trace(go.Scatter(x=results["fc_df"].index, y=results["fc_df"]["Forecasted Yield (%)"], name="Forecast", line=dict(color='orange', width=3)))
        st.plotly_chart(fig, width="stretch") # Updated Syntax

    with tabs[3]:
        st.subheader("📊 Yield Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Optimal Order", str(results["order"]))
        c2.metric("Annual Volatility", f"{results['vol']*100:.2f}%")
        c3.metric("Forecasted Move (bps)", f"{results['bps']:+.1f} bps")

    with tabs[5]:
        st.dataframe(results["fc_df"].style.format("{:.4f}"), width="stretch") # Updated Syntax

with tabs[6]:
    st.header("📚 Educational Hub")
    st.info("💡 **Basis Points (bps):** 100 bps = 1%. Interest rate moves are measured in bps because they impact the 'Cost of Capital' for trillions in debt.")
    
    st.markdown("""
    ### Benefits of Rate Forecasting:
    * **Cost of Debt Management:** CFOs use forecasts to time bond issuances.
    * **Valuation (DCF):** Rates are the denominator in every valuation model.
    * **Asset Allocation:** Shifts between Equities and Bonds are driven by yield expectations.
    """)
