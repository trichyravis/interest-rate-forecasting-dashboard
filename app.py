
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pmdarima as pm
from arch import arch_model 
import scipy.stats as stats
import time
import warnings

# ═══════════════════════════════════════════════════════════════════════════════
# MODULAR IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from content.about_text import ABOUT_CONTENT
    from content.qa_text import QA_MASTERCLASS
    from content.footer import display_footer
except ImportError:
    st.error("Critical Error: 'content' folder or files missing.")
    st.stop()

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. THEME, SIDEBAR & BUTTON STYLING
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
    /* Sidebar Background */
    [data-testid="stSidebar"] {{ background-color: {CORPORATE_BLUE} !important; }}
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {{ 
        color: white !important; font-weight: bold; 
    }}
    /* FIXED BUTTON TEXT VISIBILITY */
    div.stButton > button:first-child {{
        background-color: {GOLD} !important;
        color: {CORPORATE_BLUE} !important;
        font-weight: bold !important;
        width: 100%;
        border-radius: 8px;
        border: none;
        opacity: 1 !important;
    }}
    .config-info {{
        background-color: #f0f2f6; padding: 10px; border-radius: 5px;
        border-left: 5px solid {GOLD}; margin-bottom: 20px; font-size: 0.9rem; color: {CORPORATE_BLUE};
    }}
    </style>
    <div class="main-header">
        <h1 style="margin-bottom: 0; color: white;">INTEREST RATE FORECASTING DASHBOARD</h1>
        <h2 style="margin-top: 0; font-size: 1.3rem; opacity: 0.9; color: white;">Multi-Model (ARIMA, Vasicek, CIR) Institutional Terminal</h2>
        <p style="margin-top: 10px; font-weight: bold; font-size: 1.1rem; color: {GOLD};">Prof. V. Ravichandran | The Mountain Path</p>
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
    conf_level = st.select_slider("Confidence Level (α)", options=[0.90, 0.95, 0.99], value=0.95)
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["ℹ️ About", "📈 ARIMA", "🌪️ GARCH", "🎲 Vasicek", "☀️ CIR", "🧪 Backtest", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Q&A"])

with tabs[0]:
    st.header("📖 Institutional Methodology")
    st.write(ABOUT_CONTENT["intro"])
    st.info(ABOUT_CONTENT["workflow"])
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"### ARIMA\n{ABOUT_CONTENT['arima']}")
    with c2: st.markdown(f"### Vasicek\n{ABOUT_CONTENT['vasicek']}")
    with c3: st.markdown(f"### CIR\n{ABOUT_CONTENT['cir']}")

if run_btn:
    try:
        data = yf.Ticker(ticker).history(period=f"{lookback}y")
        if not data.empty:
            yields = data['Close'].resample('B').last().ffill()
            returns = 100 * yields.pct_change().dropna()
            f_dates = pd.date_range(yields.index[-1], periods=horizon+1, freq='B')[1:]
            config_summary = f"**Asset:** {ticker_label} | **Horizon:** {horizon}D"

            # MODEL ENGINES
            model_arima = pm.auto_arima(yields, seasonal=False)
            arima_fc = model_arima.predict(n_periods=horizon)
            garch_fit = arch_model(returns, p=1, q=1, vol='Garch').fit(disp='off')
            cond_vol = np.sqrt(garch_fit.conditional_volatility**2 * 252)

            r0, kappa, theta, sigma = yields.iloc[-1]/100, 0.20, 0.045, 0.015
            dt, n_paths = 1/252, 1000
            v_paths = np.zeros((n_paths, horizon))
            c_paths = np.zeros((n_paths, horizon))
            v_paths[:, 0] = c_paths[:, 0] = r0
            
            for i in range(1, horizon):
                v_paths[:, i] = v_paths[:, i-1] + kappa*(theta-v_paths[:, i-1])*dt + sigma*np.random.normal(0, np.sqrt(dt), n_paths)
                c_paths[:, i] = c_paths[:, i-1] + kappa*(theta-c_paths[:, i-1])*dt + sigma*np.sqrt(np.maximum(c_paths[:, i-1],0))*np.random.normal(0, np.sqrt(dt), n_paths)
                
            v_med = np.percentile(v_paths, 50, axis=0)*100
            c_med = np.percentile(c_paths, 50, axis=0)*100

            # --- UPDATED TAB RENDERING ---
            with tabs[1]:
                st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(x=yields.index[-200:], y=yields.tail(200), name="Actual", line=dict(color=CORPORATE_BLUE)))
                fig_a.add_trace(go.Scatter(x=f_dates, y=arima_fc, name="Forecast", line=dict(dash='dot', color='orange')))
                st.plotly_chart(fig_a, width='stretch')

            with tabs[2]: # GARCH (Adjusted Color Theme)
                st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
                fig_g = go.Figure()
                fig_g.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol, 
                                          line=dict(color='#E3120B', width=1.5), 
                                          fill='tozeroy', fillcolor='rgba(227, 18, 11, 0.1)', 
                                          name="Ann. Volatility"))
                fig_g.update_layout(title="Conditional Volatility (GARCH 1,1) - Institutional Risk Regime", 
                                  yaxis_title="Volatility (%)", template="plotly_white")
                st.plotly_chart(fig_g, width='stretch')

            with tabs[7]: # METRICS & VAR GRAPH
                st.markdown(f'<div class="config-info">{config_summary}</div>', unsafe_allow_html=True)
                z = stats.norm.ppf(conf_level)
                vol_now = garch_fit.conditional_volatility.iloc[-1]
                var_calc, es_calc = vol_now * z, vol_now * (stats.norm.pdf(z)/(1-conf_level))
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spot", f"{yields.iloc[-1]:.3f}%")
                c2.metric("Forecast", f"{arima_fc.iloc[-1]:.3f}%")
                c3.metric("Daily VaR", f"{var_calc:.3f}%")
                c4.metric("Exp. Shortfall", f"{es_calc:.3f}%")
                
                # --- RESTORED VAR GRAPH ---
                st.write("### 🛡️ Tail Risk Distribution (VaR Zone)")
                x_d = np.linspace(-4, 4, 300)
                y_d = stats.norm.pdf(x_d, 0, 1)
                fig_var = go.Figure()
                fig_var.add_trace(go.Scatter(x=x_d, y=y_d, fill='tozeroy', name='Standard Normal', line=dict(color=CORPORATE_BLUE)))
                # Shading the Tail Risk
                tail_x = x_d[x_d < -z]
                tail_y = y_d[x_d < -z]
                fig_var.add_trace(go.Scatter(x=tail_x, y=tail_y, fill='tozeroy', fillcolor='rgba(255,0,0,0.5)', name='Tail Risk Zone'))
                fig_var.add_vline(x=-z, line_dash="dash", line_color="red", annotation_text=f"VaR {conf_level*100}%")
                fig_var.update_layout(template="plotly_white", xaxis_title="Standard Deviations", yaxis_title="Probability Density")
                st.plotly_chart(fig_var, width='stretch')

            # [Other tabs remain with width='stretch' updates...]

            with tabs[8]: # EXPORT
                export_df = pd.DataFrame({
                    "Date": f_dates.strftime('%Y-%m-%d'),
                    "ARIMA (%)": arima_fc.values.round(4),
                    "Vasicek (%)": v_med.round(4),
                    "CIR (%)": c_med.round(4)
                })
                st.dataframe(export_df.style.background_gradient(cmap='YlGnBu'), width='stretch')
                st.download_button("📥 Download Report", export_df.to_csv(index=False).encode('utf-8'), "Analysis.csv")

    except Exception as e:
        st.error(f"Computation Error: {e}")

with tabs[9]:
    for q, a in QA_MASTERCLASS:
        with st.expander(q): st.write(a)

display_footer()
