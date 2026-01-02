
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
# CONFIGURATION & INTEREST RATE TICKERS
# ═══════════════════════════════════════════════════════════════════════════════
DARK_BLUE = "#003366"
LIGHT_BLUE = "#0066CC"
GOLD_COLOR = "#FFD700"
BRAND_NAME = "The Mountain Path - World of Finance"

RATE_TICKERS = {
    "IN10Y.NS": "India 10Y Benchmark (NSE)",
    "^TNX": "US 10Y Treasury Yield",
    "^TYX": "US 30Y Treasury Yield",
    "^FVX": "US 5Y Treasury Yield",
    "Z0=F": "India 10Y Bond Futures (Alternative)"
}

st.set_page_config(page_title="Interest Rate Dashboard - The Mountain Path", page_icon="🏦", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
    <style>
    .hero-title {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4); border: 4px solid {DARK_BLUE}; color: white; text-align: center; }}
    [data-testid="stSidebar"] {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; }}
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div[role="radiogroup"] p, [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {{ color: white !important; font-weight: 600 !important; }}
    [data-testid="stSidebar"] .st-ae div {{ color: white !important; }}
    div[data-baseweb="select"] > div, input {{ color: {DARK_BLUE} !important; }}
    [data-testid="stSidebar"] .st-at {{ color: white !important; }}
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
    refresh_button = st.button("🔄 FETCH YIELDS & RUN MODEL")
    
    st.markdown("---")
    st.markdown("### Prof. V. Ravichandran")
    st.markdown("*28+ Years Finance Experience*")
    st.markdown(f"<a href='https://www.linkedin.com/in/trichyravis' target='_blank' style='display: block; padding: 0.5rem; background: #0077b5; color: white; text-align: center; text-decoration: none; border-radius: 5px; font-weight: bold;'>🔗 LinkedIn Profile</a>", unsafe_allow_html=True)

# Summary Dashboard
st.markdown("### 📊 Selection Summary")
sm1, sm2, sm3, sm4 = st.columns(4)
sm1.metric("Yield", ticker)
sm2.metric("History", f"{lookback}y")
sm3.metric("Mode", model_mode)
sm4.metric("Horizon", f"{forecast_horizon}")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📈 Forecast", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "⚠️ Assumptions", "📋 Export", "📚 Educational Hub"])
results = None

if refresh_button:
    with st.spinner("Accessing RBI/US Treasury Terminal..."):
        data = yf.download(ticker, start=datetime.now()-timedelta(days=lookback*365))
        if not data.empty:
            raw_yields = data['Close'][ticker] if isinstance(data.columns, pd.MultiIndex) else data['Close']
            raw_yields = raw_yields.dropna()
            res_map = {"Daily": "B", "Weekly": "W", "Monthly": "M"}
            raw_yields = raw_yields.resample(res_map[freq]).last().ffill()
            
            # Volatility (Annualized)
            ann_factor = 252 if freq == "Daily" else 52 if freq == "Weekly" else 12
            ann_vol = raw_yields.pct_change().dropna().std() * np.sqrt(ann_factor)

            # Backtest Split
            bt_size = min(20, len(raw_yields)//5)
            bt_train_raw, bt_actual_raw = raw_yields[:-bt_size], raw_yields[-bt_size:]
            
            def get_series(series, t):
                if t == "Yield Changes (First Difference)": return series.diff().dropna()
                if t == "Log Yields": return np.log(series)
                return series

            train_series = get_series(raw_yields, transformation)
            bt_train_series = get_series(bt_train_raw, transformation)

            try:
                # Forecasting
                if model_mode == "Auto ARIMA":
                    model = pm.auto_arima(train_series, seasonal=False)
                    fc, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
                    order, aic, fit_obj = model.order, model.aic(), model
                    
                    bt_model = pm.auto_arima(bt_train_series, seasonal=False)
                    bt_fc_vals = bt_model.predict(n_periods=bt_size)
                else:
                    fit = ARIMA(train_series, order=(p, d, q)).fit()
                    fc_res = fit.get_forecast(steps=forecast_horizon)
                    fc, conf_int = fc_res.predicted_mean, fc_res.conf_int(alpha=0.05)
                    order, aic, fit_obj = (p, d, q), fit.aic, fit
                    
                    bt_fit = ARIMA(bt_train_series, order=(p, d, q)).fit()
                    bt_fc_vals = bt_fit.forecast(steps=bt_size)

                def invert(fc_vals, last_p, t):
                    if t == "Yield Changes (First Difference)": return last_p + np.cumsum(fc_vals)
                    if t == "Log Yields": return np.exp(fc_vals)
                    return fc_vals

                inv_fc = invert(fc, raw_yields.iloc[-1], transformation)
                inv_low = invert(conf_int[:, 0] if model_mode == "Auto ARIMA" else conf_int.iloc[:, 0], raw_yields.iloc[-1], transformation)
                inv_high = invert(conf_int[:, 1] if model_mode == "Auto ARIMA" else conf_int.iloc[:, 1], raw_yields.iloc[-1], transformation)
                inv_bt = invert(bt_fc_vals, bt_train_raw.iloc[-1], transformation)
                
                f_dates = pd.date_range(raw_yields.index[-1], periods=forecast_horizon + 1, freq=res_map[freq])[1:]
                fc_df = pd.DataFrame({"Forecasted Yield (%)": np.array(inv_fc).flatten(), "Lower CI": np.array(inv_low).flatten(), "Upper CI": np.array(inv_high).flatten()}, index=f_dates)
                
                bt_comp = pd.DataFrame({"Actual": bt_actual_raw.values, "Predicted": np.array(inv_bt).flatten()}, index=bt_actual_raw.index)
                bt_comp["bps Variance"] = (bt_comp["Predicted"] - bt_comp["Actual"]) * 100
                
                fitted = fit_obj.fittedvalues() if hasattr(fit_obj, 'fittedvalues') and callable(fit_obj.fittedvalues) else fit_obj.fittedvalues
                rmse = np.sqrt(np.mean((fitted - train_series)**2))
                
                # Basis Point Move Calculation
                current_yield = raw_yields.iloc[-1]
                forecast_end_yield = fc_df["Forecasted Yield (%)"].iloc[-1]
                bps_change = (forecast_end_yield - current_yield) * 100

                results = {"raw": raw_yields, "fc_df": fc_df, "bt_comp": bt_comp, "order": order, "aic": aic, "fit_obj": fit_obj, "rmse": rmse, "vol": ann_vol, "bps_move": bps_change, "resid": fit_obj.resid() if model_mode == "Auto ARIMA" else fit_obj.resid}
            except Exception as e: st.error(f"Analysis Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════
if results:
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Historical Yield", line=dict(color=DARK_BLUE)))
        fig.add_trace(go.Scatter(x=results["fc_df"].index.tolist()+results["fc_df"].index.tolist()[::-1], 
                                 y=results["fc_df"]["Upper CI"].tolist()+results["fc_df"]["Lower CI"].tolist()[::-1],
                                 fill='toself', fillcolor='rgba(255,165,0,0.1)', line=dict(color='rgba(255,255,255,0)'), name="95% CI Range"))
        fig.add_trace(go.Scatter(x=results["fc_df"].index, y=results["fc_df"]["Forecasted Yield (%)"], name="ARIMA Forecast", line=dict(color='orange', width=3)))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("🧪 Backtesting (Hindcasting)")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Actual", line=dict(color=DARK_BLUE)))
        fig_bt.add_trace(go.Scatter(x=results["bt_comp"].index, y=results["bt_comp"]["Predicted"], name="Prediction", line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_bt, use_container_width=True)
        st.dataframe(results["bt_comp"].style.format("{:.4f}"))

    with tabs[2]:
        fig_diag, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].plot(results["resid"]); axes[0,0].set_title("Standardized Residuals")
        plot_acf(results["resid"], ax=axes[1,0], lags=20); axes[1,0].set_title("Residual ACF")
        stats.probplot(results["resid"], dist="norm", plot=axes[1,1]); axes[1,1].set_title("Normal Q-Q Plot")
        plt.tight_layout(); st.pyplot(fig_diag)

    with tabs[3]:
        st.subheader("📊 Interest Rate Analysis Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Optimal Order", str(results["order"]))
        c1.metric("AIC Score", f"{results['aic']:.2f}")
        
        c2.metric("RMSE", f"{results['rmse']:.4f}")
        c2.metric("Annualized Volatility", f"{results['vol']*100:.2f}%")
        
        # --- THE BASIS POINT (BPS) CALCULATOR CARD ---
        color = "normal" if results["bps_move"] == 0 else "inverse" if results["bps_move"] > 0 else "normal"
        c3.metric("Forecasted Move (bps)", f"{results['bps_move']:+.1f} bps", 
                  help="1 basis point = 0.01%. This shows the change from current yield to the end of the forecast.")
        c3.metric("Ljung-Box p-val", f"{acorr_ljungbox(results['resid'], lags=[10], return_df=True)['lb_pvalue'].values[0]:.3f}")

    with tabs[4]:
        st.header("⚠️ Assumptions & Limitations")
        st.markdown("""
        * **Linearity Assumption:** Forecasts assume a linear relationship with past yield levels.
        * **Shock Sensitivity:** Traditional ARIMA cannot predict 'surprise' Repo rate hikes or FOMC policy pivots.
        * **Mean Reversion:** While yields are mean-reverting, structural economic shifts can break past patterns.
        """)

    with tabs[5]:
        st.dataframe(results["fc_df"].style.background_gradient(cmap='RdYlGn', axis=0).format("{:.4f}"), use_container_width=True)

    with tabs[6]:
        st.header("📚 Educational Hub: Fixed Income Forecasting")
        st.markdown("""
        ### Why do we forecast Basis Points (bps)?
        In bond markets, yields are moved in tiny increments. A move of 1% is considered massive. Traders use **Basis Points** to discuss shifts:
        - **100 bps = 1.00%**
        - **25 bps = 0.25%** (Standard Central Bank move size)
        
        ### Strategic Benefits:
        * **Cost of Capital:** Helps CFOs time debt issuances.
        * **Portfolio Duration:** Helps managers manage interest rate risk.
        """)
        st.success("This dashboard helps bridge the gap between Yield Curve theory and Practical Analytics.")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>{BRAND_NAME} | Built for Professional Excellence</p>", unsafe_allow_html=True)
