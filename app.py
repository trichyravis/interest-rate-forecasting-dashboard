
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
from fredapi import Fred
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
DARK_BLUE = "#003366"
LIGHT_BLUE = "#0066CC"
GOLD_COLOR = "#FFD700"
BRAND_NAME = "The Mountain Path - World of Finance"

# Ticker Mapping
YAHOO_TICKERS = {
    "IN10Y.NS": "India 10Y Benchmark (NSE)",
    "^TNX": "US 10Y Treasury Yield",
    "^TYX": "US 30Y Treasury Yield",
    "^FVX": "US 5Y Treasury Yield"
}

FRED_TICKERS = {
    "DGS10": "US 10-Year Treasury Constant Maturity Rate",
    "DGS30": "US 30-Year Treasury Constant Maturity Rate",
    "DGS5": "US 5-Year Treasury Constant Maturity Rate",
    "DBAA": "Moody's Seasoned Baa Corporate Bond Yield",
    "CPIAUCSL": "Consumer Price Index (CPI Inflation)"
}

st.set_page_config(page_title="Interest Rate Dashboard - The Mountain Path", page_icon="🏦", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA ENGINES (WITH RETRY & CACHING)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data_unified(source, ticker, days_back, api_key=None):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    if source == "Yahoo Finance":
        wait_times = [5, 10, 20]
        attempt = 0
        while attempt <= len(wait_times):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty and attempt < len(wait_times): raise Exception("Empty")
                # Normalize column name for ARIMA logic
                df = df[['Close']] if isinstance(df.columns, pd.Index) else df.xs('Close', axis=1, level=0)
                return df
            except Exception:
                if attempt < len(wait_times):
                    time.sleep(wait_times[attempt])
                    attempt += 1
                else: return pd.DataFrame()
                
    elif source == "FRED (Federal Reserve)":
        try:
            fred = Fred(api_key=api_key)
            series = fred.get_series(ticker, observation_start=start_date)
            df = pd.DataFrame(series, columns=['Close'])
            return df
        except Exception as e:
            st.error(f"FRED API Error: {e}")
            return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR & UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<style>.hero-title {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); padding: 2rem; border-radius: 20px; color: white; text-align: center; }} [data-testid='stSidebar'] {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; }} [data-testid='stSidebar'] p, [data-testid='stSidebar'] label {{ color: white !important; }} .stButton>button {{ background-color: {GOLD_COLOR} !important; color: {DARK_BLUE} !important; font-weight: bold !important; }}</style>", unsafe_allow_html=True)

st.markdown(f"<div class='hero-title'><h1>INTEREST RATE FORECASTING DASHBOARD</h1><p>Dual-Engine Modeling: Yahoo Finance & FRED</p><p>Prof. V. Ravichandran | 28+ Years Finance Experience</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔌 Data Engine")
    source = st.selectbox("Source", ["Yahoo Finance", "FRED (Federal Reserve)"])
    fred_key = ""
    if source == "FRED (Federal Reserve)":
        fred_key = st.text_input("FRED API Key", type="password", help="Get a free key at fred.stlouisfed.org")
    
    ticker_list = YAHOO_TICKERS if source == "Yahoo Finance" else FRED_TICKERS
    ticker = st.selectbox("Benchmark", options=list(ticker_list.keys()), format_func=lambda x: f"{x} - {ticker_list[x]}")
    
    st.markdown("### ⚙️ Model Settings")
    lookback = st.selectbox("History (Years)", [1, 2, 3, 5, 10], index=2)
    transformation = st.radio("Transformation", ["Yield Level (%)", "Yield Changes", "Log Yields"])
    model_mode = st.radio("Mode", ["Auto ARIMA", "Manual ARIMA"], index=0)
    
    forecast_horizon = st.slider("Horizon", 1, 60, 10)
    refresh_button = st.button("🔄 RUN ANALYSIS")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Forecast", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "⚠️ Assumptions", "📋 Export", "📚 Educational Hub"])

if refresh_button:
    if source == "FRED (Federal Reserve)" and not fred_key:
        st.warning("Please enter your FRED API Key in the sidebar.")
    else:
        with st.spinner(f"Fetching data from {source}..."):
            raw_data = fetch_data_unified(source, ticker, lookback*365, fred_key)
            
            if not raw_data.empty:
                series = raw_data['Close'].dropna().resample('B').last().ffill()
                
                # ARIMA Pipeline
                def trans(s, t):
                    if t == "Yield Changes": return s.diff().dropna()
                    if t == "Log Yields": return np.log(s)
                    return s
                
                train_series = trans(series, transformation)
                
                try:
                    if model_mode == "Auto ARIMA":
                        model = pm.auto_arima(train_series, seasonal=False)
                        fc, conf = model.predict(n_periods=forecast_horizon, return_conf_int=True)
                        order = model.order
                    else:
                        model = ARIMA(train_series, order=(1,1,1)).fit()
                        res = model.get_forecast(steps=forecast_horizon)
                        fc, conf = res.predicted_mean, res.conf_int()
                    
                    # Inversion
                    last = series.iloc[-1]
                    if transformation == "Yield Changes":
                        inv_fc = last + np.cumsum(fc)
                    elif transformation == "Log Yields":
                        inv_fc = np.exp(fc)
                    else:
                        inv_fc = fc
                    
                    f_dates = pd.date_range(series.index[-1], periods=forecast_horizon + 1, freq='B')[1:]
                    fc_df = pd.DataFrame({"Forecast": np.array(inv_fc).flatten()}, index=f_dates)
                    bps = (inv_fc[-1] - last) * 100

                    with tabs[0]:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical", line=dict(color=DARK_BLUE)))
                        fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["Forecast"], name="Forecast", line=dict(color="orange", width=3)))
                        st.plotly_chart(fig, width="stretch")
                    
                    with tabs[3]:
                        st.subheader("📊 Performance Metrics")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Optimal Order", str(order))
                        c2.metric("Forecasted Move", f"{bps:+.1f} bps")
                        c3.metric("Data Source", source)

                except Exception as e: st.error(f"Model Error: {e}")
            else:
                st.error("No data found. If using Yahoo, the server might be rate-limited. If using FRED, check your API Key and Ticker.")

# ═══════════════════════════════════════════════════════════════════════════════
# EDUCATIONAL HUB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("📚 Educational Hub")
    st.markdown("""
    ### 🔌 Understanding Data Sources
    * **Yahoo Finance**: Great for real-time market data but sensitive to high-frequency requests (Rate Limits).
    * **FRED**: The gold standard for institutional economic data. It provides clean, authoritative series directly from the Federal Reserve.
    
    ### 🏦 Benefits of Forecasting
    
    * **Cost of Capital Management**: Predicting shifts in the yield curve allows for better timing in debt issuance.
    * **Risk Mitigation**: 95% Confidence Intervals provide a 'Fan Chart' view of potential rate volatility.
    """)
