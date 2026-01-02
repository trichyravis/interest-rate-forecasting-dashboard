
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & BRANDING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="US Treasury Analytics Terminal", layout="wide")

DARK_BLUE = "#003366"
GOLD = "#FFD700"

# Professional CSS Injection
st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, #0066CC 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        margin-bottom: 2rem; border-bottom: 5px solid {GOLD};
    }}
    [data-testid="stSidebar"] {{ background-color: #f0f2f6; }}
    </style>
    <div class="main-header">
        <h1>US TREASURY ANALYTICS TERMINAL</h1>
        <p>Prof. V. Ravichandran | The Mountain Path - World of Finance</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR - US TICKER SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🇺🇸 Treasury Benchmarks")
    ticker_label = st.selectbox("Select Maturity", [
        "US 10Y Treasury (^TNX)", 
        "US 30Y Treasury (^TYX)", 
        "US 5Y Treasury (^FVX)"
    ])
    ticker = ticker_label.split("(")[1].replace(")", "")
    
    st.header("⚙️ Forecast Settings")
    lookback_years = st.slider("Historical Lookback (Years)", 1, 10, 5)
    forecast_horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20)
    
    model_type = st.radio("Model Selection", ["Auto-ARIMA (Optimized)", "ARIMA (1,1,1)"])
    
    run_btn = st.button("🚀 EXECUTE ANALYSIS")
    st.markdown("---")
    st.info("Institutional Grade Data provided via Yahoo Finance API.")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CORE ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Forecast View", "📊 Yield Metrics", "📚 Educational Hub"])

if run_btn:
    with st.spinner(f"Fetching data for {ticker}..."):
        # Fetch Data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years*365)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if not data.empty:
            # Handle multi-index if yfinance returns it
            if isinstance(data.columns, pd.MultiIndex):
                series = data['Close'][ticker].dropna()
            else:
                series = data['Close'].dropna()
            
            # Resample to Business Days
            series = series.resample('B').last().ffill()

            # --- Box-Jenkins ARIMA Process ---
            try:
                if model_type == "Auto-ARIMA (Optimized)":
                    model = pm.auto_arima(series, seasonal=False, suppress_warnings=True)
                    forecast = model.predict(n_periods=forecast_horizon)
                    order = model.order
                else:
                    fit = ARIMA(series, order=(1,1,1)).fit()
                    forecast = fit.forecast(steps=forecast_horizon)
                    order = (1,1,1)

                # --- Tab 1: Visualization ---
                with tabs[0]:
                    fig = go.Figure()
                    # Plot recent history (last 1 year for clarity)
                    recent_hist = series.tail(252)
                    fig.add_trace(go.Scatter(x=recent_hist.index, y=recent_hist, name="Historical Yield", line=dict(color=DARK_BLUE, width=2)))
                    
                    # Forecast line
                    f_dates = pd.date_range(series.index[-1], periods=forecast_horizon+1, freq='B')[1:]
                    fig.add_trace(go.Scatter(x=f_dates, y=forecast, name="Predicted Trend", line=dict(color="orange", width=4, dash='dot')))
                    
                    fig.update_layout(
                        title=f"{ticker_label} - ARIMA {order} Model",
                        xaxis_title="Date", yaxis_title="Yield (%)",
                        hovermode="x unified", height=500
                    )
                    st.plotly_chart(fig, width="stretch")

                # --- Tab 2: Yield Metrics & BPS ---
                with tabs[1]:
                    c1, c2, c3 = st.columns(3)
                    curr_val = series.iloc[-1]
                    pred_val = forecast.iloc[-1]
                    bps_change = (pred_val - curr_val) * 100
                    
                    c1.metric("Current Yield", f"{curr_val:.3f}%")
                    c2.metric("Forecasted Yield", f"{pred_val:.3f}%")
                    c3.metric("BPS Shift", f"{bps_change:+.1f} bps", delta_color="inverse")
                    
                    st.markdown("### Model Diagnostics")
                    st.write(f"**Optimal Order:** {order}")
                    st.write(f"**Data Points Analyzed:** {len(series)}")

            except Exception as e:
                st.error(f"Modeling Error: {e}")
        else:
            st.error("No data found for this ticker. Please try again.")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EDUCATIONAL HUB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🎓 Learning: US Treasury Forecasting")
    st.write("""
    Forecasting the US Treasury yield curve is the cornerstone of global macro strategy. 
    By using the **Box-Jenkins (ARIMA) methodology**, we identify:
    * **Autoregression (p):** How past yields influence the present.
    * **Integration (d):** The trend stability of the yield levels.
    * **Moving Average (q):** The impact of recent market 'shocks'.
    """)
    st.info("💡 Refer to the **ARIMA Modeling Guide PDF** in your project folder for deep-dive mathematical explanations.")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>© 2026 The Mountain Path - World of Finance | Prof. V. Ravichandran</p>", unsafe_allow_html=True)
