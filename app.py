
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & INSTITUTIONAL THEME (STRICT SIDEBAR CSS)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Interest Rate Forecasting Dashboard", layout="wide")

DARK_BLUE = "#003366"  # Institutional Navy
GOLD = "#FFD700"

st.markdown(f"""
    <style>
    /* Header Styling */
    .main-header {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, #0066CC 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        margin-bottom: 2rem; border-bottom: 5px solid {GOLD};
    }}
    
    /* SIDEBAR DARK BLUE BACKGROUND & WHITE TEXT */
    [data-testid="stSidebar"] {{
        background-color: {DARK_BLUE} !important;
        color: white !important;
    }}
    
    /* Sidebar Headers and Labels to White */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
        color: white !important;
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{ gap: 12px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f0f2f6; border-radius: 5px 5px 0 0; padding: 10px 15px; color: {DARK_BLUE};
    }}
    .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; font-weight: bold; }}
    </style>
    
    <div class="main-header">
        <h1>INTEREST RATE FORECASTING DASHBOARD</h1>
        <p>The Mountain Path - World of Finance | Quantitative Analytics</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR - PROFILE AT BOTTOM
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_label = st.selectbox("Benchmark Maturity", ["US 10Y (^TNX)", "US 30Y (^TYX)", "US 5Y (^FVX)"])
    ticker = ticker_label.split("(")[1].replace(")", "")
    lookback = st.slider("Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20)
    
    st.header("🎨 Display Settings")
    show_step = st.checkbox("Show Step-Wise Curve", value=True)
    
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS", use_container_width=True)

    # Spacer to push profile card down
    for _ in range(12): st.write("")
        
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.1); border: 1px solid {GOLD};">
            <h3 style="color: white !important; margin: 0;">Prof. V. Ravichandran</h3>
            <p style="color: #ddd !important; font-size: 0.85rem; margin: 5px 0;">28+ Years Finance Experience</p>
            <hr style="margin: 10px 0; border-color: {GOLD};">
            <a href="https://www.linkedin.com/in/trichyravis" target="_blank" style="text-decoration: none;">
                <button style="background-color: #0077b5; color: white; border: none; padding: 8px; border-radius: 5px; width: 100%; cursor: pointer; font-weight: bold;">🔗 LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICS ENGINE & TABBED UI
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Processing Market Data..."):
        data = yf.download(ticker, period=f"{lookback}y", interval="1d", progress=False)
        
        if not data.empty:
            yields = data['Close'].dropna()
            if isinstance(yields, pd.DataFrame): yields = yields.iloc[:, 0]
            yields = yields.resample('B').last().ffill()

            try:
                # ARIMA logic
                model_arima = pm.auto_arima(yields, seasonal=False, suppress_warnings=True)
                arima_fc = model_arima.predict(n_periods=horizon)
                f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]

                tabs = st.tabs(["📈 Forecast", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Educational Hub"])

                with tabs[0]: # Forecast View
                    if show_step:
                        st.subheader("📡 Institutional Step-Wise Forecast")
                        fig_step = go.Figure()
                        fig_step.add_trace(go.Scatter(x=f_dates, y=arima_fc, mode='lines+markers', line_shape='hv', 
                                                    line=dict(color='#FF4B4B', width=4), name="Step Curve"))
                        fig_step.update_layout(template="plotly_dark", title="Step-Wise Rate Projection", paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_step, use_container_width=True)
                    else:
                        fig_main = go.Figure()
                        fig_main.add_trace(go.Scatter(x=yields.index[-250:], y=yields.tail(250), name="Actual", line=dict(color=DARK_BLUE)))
                        fig_main.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="Forecast", line=dict(color="orange", dash='dot')))
                        fig_main.update_layout(template="plotly_white")
                        st.plotly_chart(fig_main, use_container_width=True)

                with tabs[1]: # Backtesting
                    train, test = yields.iloc[:-30], yields.iloc[-30:]
                    bt_model = pm.auto_arima(train, seasonal=False)
                    bt_fc = bt_model.predict(n_periods=30)
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=test.index, y=test, name="Market"))
                    fig_bt.add_trace(go.Scatter(x=test.index, y=bt_fc, name="Model Prediction", line=dict(dash='dash')))
                    st.plotly_chart(fig_bt, use_container_width=True)
                    st.success(f"MAE: {np.mean(np.abs(test.values - bt_fc.values)):.4f}")

                with tabs[3]: # Metrics
                    c1, c2, c3 = st.columns(3)
                    curr, pred = float(yields.iloc[-1]), float(arima_fc.iloc[-1])
                    c1.metric("Current Spot", f"{curr:.3f}%")
                    c2.metric("Forecasted Rate", f"{pred:.3f}%")
                    c3.metric("BPS Move", f"{(pred-curr)*100:+.1f} bps")

                with tabs[5]: # Educational Hub
                    st.header("🎓 Box-Jenkins ARIMA Framework")
                    
                    st.markdown("""
                    **Stage 1: Identification** (Stationarity check).  
                    **Stage 2: Estimation** (Parameter selection).  
                    **Stage 3: Diagnostics** (Residual check).
                    """)

            except Exception as e:
                st.error(f"Computation Error: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>The Mountain Path - World of Finance | Institutional US Edition</p>", unsafe_allow_html=True)
