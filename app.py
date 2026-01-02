"""
Interest Rate Forecasting Dashboard
Box-Jenkins (ARIMA) Time Series Modeling for Global & Indian Yields

Developed by: Prof. V. Ravichandran
28+ Years Corporate Finance & Banking Experience
10+ Years Academic Excellence
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_acfm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Interest Rate Forecasting Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional styling
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary-blue: #003366;
        --secondary-blue: #004d80;
        --gold: #FFD700;
        --light-gray: #F0F2F6;
        --dark-gray: #36454F;
    }
    
    /* Streamlit customization */
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #003366;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #FFD700;
    }
    
    .section-header {
        color: #003366;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #FFD700;
        padding-bottom: 0.5rem;
    }
    
    .highlight-box {
        background-color: #e6f2ff;
        border-left: 4px solid #003366;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("### 🏔️ The Mountain Path - World of Finance")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📊 Navigation",
    options=[
        "Dashboard",
        "ARIMA Modeling",
        "Forecasting & Risk",
        "Backtesting",
        "Educational Hub"
    ],
    index=0
)

st.sidebar.markdown("---")

# Benchmark selection
benchmark = st.sidebar.selectbox(
    "📈 Select Benchmark",
    options=["India 10Y G-Sec", "US 10Y Treasury"],
    help="Choose the yield benchmark for analysis"
)

# Data parameters
data_years = st.sidebar.slider(
    "📅 Historical Data (Years)",
    min_value=1,
    max_value=20,
    value=10,
    help="Length of historical data to retrieve"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Prof. V. Ravichandran**")
st.sidebar.markdown("*28+ Years Corporate Finance & Banking*")
st.sidebar.markdown("*10+ Years Academic Excellence*")

# ============================================================================
# DATA RETRIEVAL & PROCESSING
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_yield_data(benchmark, years):
    """Fetch historical yield data from Yahoo Finance"""
    
    tickers = {
        "India 10Y G-Sec": "^INFY",  # Placeholder - can be adjusted
        "US 10Y Treasury": "^TNX"
    }
    
    ticker = tickers[benchmark]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            interval='1d'
        )
        
        # Use closing price as yield
        if isinstance(data, pd.DataFrame):
            yields = data['Close']
        else:
            yields = data
            
        return yields
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def prepare_data(yields):
    """Prepare and clean yield data"""
    
    # Remove NaN values
    yields = yields.dropna()
    
    # Remove duplicates
    yields = yields[~yields.index.duplicated(keep='first')]
    
    # Sort by date
    yields = yields.sort_index()
    
    return yields

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def adf_test(series):
    """Augmented Dickey-Fuller Test"""
    result = adfuller(series, autolag='AIC')
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }

def kpss_test(series):
    """KPSS Test"""
    result = kpss(series, regression='c', nlags='auto')
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[3],
        'is_stationary': result[1] > 0.05
    }

# ============================================================================
# ARIMA MODELING
# ============================================================================

@st.cache_resource
def fit_auto_arima(series, seasonal=False):
    """Automatically identify optimal ARIMA parameters"""
    
    model = auto_arima(
        series,
        start_p=0,
        max_p=5,
        start_q=0,
        max_q=5,
        seasonal=seasonal,
        stepwise=True,
        suppress_warnings=True,
        information_criterion='aic',
        trace=False,
        error_action='ignore'
    )
    
    return model

def fit_arima_model(series, order):
    """Fit ARIMA model with specified parameters"""
    
    try:
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {str(e)}")
        return None

# ============================================================================
# FORECASTING & CALCULATIONS
# ============================================================================

def forecast_yields(model, steps, confidence=0.95):
    """Generate forecasts with confidence intervals"""
    
    forecast_result = model.get_forecast(steps=steps)
    forecast_df = forecast_result.conf_int(alpha=1-confidence)
    forecast_df['forecast'] = forecast_result.predicted_mean
    forecast_df.columns = ['lower_ci', 'upper_ci', 'forecast']
    
    return forecast_df

def calculate_basis_points(current_yield, forecast_yield):
    """Convert yield changes to basis points"""
    
    change_bps = (forecast_yield - current_yield) * 100
    return change_bps

def calculate_volatility(series, periods=252):
    """Calculate annualized volatility"""
    
    returns = series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(periods)
    
    return volatility

def backtest_model(series, model, test_split=0.2):
    """Perform out-of-sample backtesting"""
    
    split_point = int(len(series) * (1 - test_split))
    train_data = series[:split_point]
    test_data = series[split_point:]
    
    # Refit model on training data
    fitted_model = model
    
    # Generate forecasts for test period
    forecasts = []
    for i in range(len(test_data)):
        forecast = fitted_model.forecast(steps=1)[0]
        forecasts.append(forecast)
        
        # Update model with actual value
        fitted_model = ARIMA(
            series[:split_point+i+1],
            order=fitted_model.model_orders['order']
        ).fit()
    
    forecasts = np.array(forecasts)
    
    # Calculate metrics
    mae = mean_absolute_error(test_data, forecasts)
    rmse = np.sqrt(mean_squared_error(test_data, forecasts))
    mape = np.mean(np.abs((test_data - forecasts) / test_data)) * 100
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'forecasts': forecasts,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_yield_series(yields, title):
    """Plot historical yield series"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yields.index,
        y=yields.values,
        mode='lines',
        name='Historical Yield',
        line=dict(color='#003366', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Yield:</b> %{y:.3f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Yield (%)',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_forecast_with_ci(yields, forecast_df, title):
    """Plot forecasts with confidence intervals"""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=yields.index,
        y=yields.values,
        mode='lines',
        name='Historical Yield',
        line=dict(color='#003366', width=2)
    ))
    
    # Point forecast
    forecast_dates = pd.date_range(
        start=yields.index[-1],
        periods=len(forecast_df)+1,
        freq='D'
    )[1:]
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='#FFD700', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_df['upper_ci'],
        fill=None,
        mode='lines',
        name='Upper 95% CI',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_df['lower_ci'],
        fill='tonexty',
        mode='lines',
        name='95% Confidence Interval',
        line=dict(width=0),
        fillcolor='rgba(0, 51, 102, 0.2)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Yield (%)',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_acf_pacf(series):
    """Plot ACF and PACF"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    plot_acf(series, lags=40, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    
    plot_acf(series, lags=40, ax=axes[1], method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')
    
    plt.tight_layout()
    return fig

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "Dashboard":
    st.markdown("<h1 style='color: #003366;'>🏦 Interest Rate Forecasting Dashboard</h1>", unsafe_allow_html=True)
    
    # Fetch data
    yields = fetch_yield_data(benchmark, data_years)
    
    if yields is not None:
        yields = prepare_data(yields)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Yield",
                f"{yields.iloc[-1]:.3f}%",
                f"{(yields.iloc[-1] - yields.iloc[-5]):.3f}% (5-day)"
            )
        
        with col2:
            st.metric(
                "YTD Change",
                f"{(yields.iloc[-1] - yields.iloc[0]):.3f}%",
                f"{((yields.iloc[-1] - yields.iloc[0])/yields.iloc[0]*100):.2f}%"
            )
        
        with col3:
            st.metric(
                "Volatility (Annual)",
                f"{calculate_volatility(yields)*100:.2f}%"
            )
        
        with col4:
            st.metric(
                "30-Day Average",
                f"{yields.tail(30).mean():.3f}%"
            )
        
        st.markdown("---")
        
        # Historical yield plot
        st.markdown("<div class='section-header'>📈 Historical Yield Trends</div>", unsafe_allow_html=True)
        fig_yields = plot_yield_series(yields, f"{benchmark} - Historical Trends")
        st.plotly_chart(fig_yields, use_container_width=True)
        
        # Yield statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("**Yield Statistics**")
            st.write(f"Mean: {yields.mean():.3f}%")
            st.write(f"Std Dev: {yields.std():.3f}%")
            st.write(f"Min: {yields.min():.3f}%")
            st.write(f"Max: {yields.max():.3f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("**Recent Performance**")
            st.write(f"Last Close: {yields.iloc[-1]:.3f}%")
            st.write(f"Previous Close: {yields.iloc[-2]:.3f}%")
            st.write(f"52-Week High: {yields.tail(252).max():.3f}%")
            st.write(f"52-Week Low: {yields.tail(252).min():.3f}%")
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# PAGE: ARIMA MODELING
# ============================================================================

elif page == "ARIMA Modeling":
    st.markdown("<h1 style='color: #003366;'>📊 ARIMA Model Identification</h1>", unsafe_allow_html=True)
    
    yields = fetch_yield_data(benchmark, data_years)
    if yields is not None:
        yields = prepare_data(yields)
        
        # Model selection approach
        st.markdown("<div class='section-header'>🔧 Model Selection</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_selection = st.radio(
                "Select Parameters",
                options=["Auto-ARIMA", "Manual Selection"],
                index=0
            )
        
        if auto_selection == "Auto-ARIMA":
            st.info("⏳ Analyzing data for optimal ARIMA parameters...")
            
            with st.spinner("Running Auto-ARIMA analysis..."):
                auto_model = fit_auto_arima(yields, seasonal=False)
                optimal_order = auto_model.order
                
                st.success(f"✅ Optimal Parameters: ARIMA{optimal_order}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("p (AR)", optimal_order[0])
                with col2:
                    st.metric("d (I)", optimal_order[1])
                with col3:
                    st.metric("q (MA)", optimal_order[2])
            
            fitted_model = auto_model
            selected_order = optimal_order
        
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                p = st.slider("p (AR Order)", 0, 5, 1)
            with col2:
                d = st.slider("d (Differencing)", 0, 2, 1)
            with col3:
                q = st.slider("q (MA Order)", 0, 5, 1)
            
            selected_order = (p, d, q)
            fitted_model = fit_arima_model(yields, selected_order)
        
        if fitted_model is not None:
            st.markdown("---")
            st.markdown("<div class='section-header'>📋 Stationarity Tests</div>", unsafe_allow_html=True)
            
            # ADF Test
            adf_result = adf_test(yields)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Augmented Dickey-Fuller (ADF) Test**")
                st.write(f"Test Statistic: {adf_result['test_statistic']:.6f}")
                st.write(f"P-Value: {adf_result['p_value']:.6f}")
                
                if adf_result['is_stationary']:
                    st.success("✅ Series is STATIONARY (p < 0.05)")
                else:
                    st.warning("⚠️ Series is NON-STATIONARY (p ≥ 0.05)")
            
            with col2:
                st.markdown("**Critical Values**")
                for key, value in adf_result['critical_values'].items():
                    st.write(f"{key}: {value:.4f}")
            
            st.markdown("---")
            st.markdown("<div class='section-header'>📊 ACF & PACF Analysis</div>", unsafe_allow_html=True)
            
            fig_acf_pacf = plot_acf_pacf(yields)
            st.pyplot(fig_acf_pacf)
            
            st.markdown("---")
            st.markdown("<div class='section-header'>📈 Model Summary</div>", unsafe_allow_html=True)
            
            st.text(fitted_model.summary())
            
            st.markdown("---")
            st.markdown("<div class='section-header'>🔍 Residual Diagnostics</div>", unsafe_allow_html=True)
            
            fig_diagnostics = fitted_model.plot_diagnostics(figsize=(12, 8))
            st.pyplot(fig_diagnostics)

# ============================================================================
# PAGE: FORECASTING & RISK
# ============================================================================

elif page == "Forecasting & Risk":
    st.markdown("<h1 style='color: #003366;'>🎯 Forecasting & Risk Analytics</h1>", unsafe_allow_html=True)
    
    yields = fetch_yield_data(benchmark, data_years)
    if yields is not None:
        yields = prepare_data(yields)
        
        # Fit model
        with st.spinner("Fitting ARIMA model..."):
            auto_model = fit_auto_arima(yields, seasonal=False)
        
        st.markdown("<div class='section-header'>⚙️ Forecast Configuration</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_days = st.slider(
                "Forecast Horizon (Trading Days)",
                min_value=1,
                max_value=252,
                value=60,
                step=1
            )
        
        with col2:
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
        
        # Generate forecast
        forecast_df = forecast_yields(auto_model, forecast_days, confidence_level)
        
        st.markdown("---")
        st.markdown("<div class='section-header'>📈 Forecast with Confidence Intervals</div>", unsafe_allow_html=True)
        
        fig_forecast = plot_forecast_with_ci(yields, forecast_df, f"{benchmark} - {forecast_days}D Forecast")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.markdown("---")
        st.markdown("<div class='section-header'>💹 Basis Points Analysis</div>", unsafe_allow_html=True)
        
        current_yield = yields.iloc[-1]
        forecast_yield = forecast_df['forecast'].iloc[-1]
        bps_change = calculate_basis_points(current_yield, forecast_yield)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Yield",
                f"{current_yield:.3f}%"
            )
        
        with col2:
            st.metric(
                "Forecast Yield (End Period)",
                f"{forecast_yield:.3f}%"
            )
        
        with col3:
            color_indicator = "🔴" if bps_change < 0 else "🟢"
            st.metric(
                f"{color_indicator} Change (bps)",
                f"{bps_change:.1f}"
            )
        
        st.markdown("---")
        st.markdown("<div class='section-header'>⚠️ Risk Metrics</div>", unsafe_allow_html=True)
        
        volatility = calculate_volatility(yields)
        ci_width = forecast_df['upper_ci'].iloc[-1] - forecast_df['lower_ci'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Annualized Volatility",
                f"{volatility*100:.2f}%"
            )
        
        with col2:
            st.metric(
                "95% CI Width (End)",
                f"{ci_width:.4f}%"
            )
        
        with col3:
            st.metric(
                "Lower Bound",
                f"{forecast_df['lower_ci'].iloc[-1]:.3f}%"
            )
        
        # Forecast table
        st.markdown("---")
        st.markdown("<div class='section-header'>📊 Forecast Values</div>", unsafe_allow_html=True)
        
        forecast_dates = pd.date_range(start=yields.index[-1], periods=forecast_days+1, freq='D')[1:]
        forecast_display = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_df['forecast'].values,
            'Lower CI': forecast_df['lower_ci'].values,
            'Upper CI': forecast_df['upper_ci'].values
        })
        
        st.dataframe(forecast_display.head(20), use_container_width=True)

# ============================================================================
# PAGE: BACKTESTING
# ============================================================================

elif page == "Backtesting":
    st.markdown("<h1 style='color: #003366;'>🔄 Out-of-Sample Backtesting</h1>", unsafe_allow_html=True)
    
    yields = fetch_yield_data(benchmark, data_years)
    if yields is not None:
        yields = prepare_data(yields)
        
        st.markdown("<div class='section-header'>⚙️ Backtest Configuration</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_split = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            ) / 100
        
        with col2:
            st.info(f"Train Size: {int((1-test_split)*len(yields))} | Test Size: {int(test_split*len(yields))}")
        
        # Fit model
        with st.spinner("Fitting ARIMA model and running backtest..."):
            auto_model = fit_auto_arima(yields, seasonal=False)
            backtest_results = backtest_model(yields, auto_model, test_split=test_split)
        
        st.markdown("---")
        st.markdown("<div class='section-header'>📊 Backtesting Results</div>", unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mean Absolute Error (MAE)",
                f"{backtest_results['mae']:.4f}%"
            )
        
        with col2:
            st.metric(
                "Root Mean Squared Error (RMSE)",
                f"{backtest_results['rmse']:.4f}%"
            )
        
        with col3:
            st.metric(
                "Mean Absolute Percentage Error (MAPE)",
                f"{backtest_results['mape']:.2f}%"
            )
        
        # Visualization
        fig = go.Figure()
        
        train_idx = backtest_results['train_data'].index
        test_idx = backtest_results['test_data'].index
        
        fig.add_trace(go.Scatter(
            x=train_idx,
            y=backtest_results['train_data'],
            mode='lines',
            name='Training Data',
            line=dict(color='#003366', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_idx,
            y=backtest_results['test_data'],
            mode='lines',
            name='Actual Test Data',
            line=dict(color='#004d80', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_idx,
            y=backtest_results['forecasts'],
            mode='lines',
            name='Predicted Values',
            line=dict(color='#FFD700', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Backtest: Actual vs. Predicted",
            xaxis_title="Date",
            yaxis_title="Yield (%)",
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals analysis
        st.markdown("---")
        st.markdown("<div class='section-header'>📈 Forecast Errors</div>", unsafe_allow_html=True)
        
        residuals = backtest_results['test_data'].values - backtest_results['forecasts']
        
        fig_residuals = go.Figure()
        
        fig_residuals.add_trace(go.Scatter(
            x=test_idx,
            y=residuals,
            mode='markers',
            marker=dict(color='#003366', size=8),
            name='Forecast Errors'
        ))
        
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig_residuals.update_layout(
            title="Forecast Errors Over Test Period",
            xaxis_title="Date",
            yaxis_title="Error (%)",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_residuals, use_container_width=True)

# ============================================================================
# PAGE: EDUCATIONAL HUB
# ============================================================================

elif page == "Educational Hub":
    st.markdown("<h1 style='color: #003366;'>📚 Educational Hub</h1>", unsafe_allow_html=True)
    
    # Tabs for different topics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Box-Jenkins Method",
        "Stationarity",
        "ARIMA Components",
        "Risk Management",
        "Real-World Applications"
    ])
    
    with tab1:
        st.markdown("## 🔄 The Box-Jenkins Methodology")
        
        st.markdown("""
        The **Box-Jenkins (ARIMA) method** is a systematic approach to time series forecasting that was developed 
        by George Box and Gwilym Jenkins in 1970. It remains one of the most powerful tools in financial econometrics.
        
        ### The Four-Step Process:
        
        #### 1️⃣ **Identification**
        - Determine if the series is stationary (mean, variance, autocorrelation are constant over time)
        - Use stationarity tests: Augmented Dickey-Fuller (ADF), KPSS
        - Examine ACF and PACF plots to identify AR(p) and MA(q) orders
        - Determine differencing order (d) if series is non-stationary
        
        #### 2️⃣ **Estimation**
        - Estimate coefficients using Maximum Likelihood Estimation (MLE)
        - Calculate standard errors and confidence intervals
        - Compare models using information criteria (AIC, BIC)
        - Auto-ARIMA automates this process
        
        #### 3️⃣ **Diagnostic Checking**
        - Verify residuals follow white noise (zero mean, constant variance, no autocorrelation)
        - Ljung-Box test for residual autocorrelation
        - Visual inspection of residuals (Q-Q plot, histograms)
        - ACF/PACF of residuals should show no significant patterns
        
        #### 4️⃣ **Forecasting**
        - Generate point forecasts and confidence intervals
        - Monitor forecast accuracy as new data arrives
        - Update model parameters periodically
        """)
        
        # Add visual
        st.info("💡 **Pro Tip**: A good ARIMA model has uncorrelated residuals that behave like white noise, " +
                "with no patterns remaining to be explained.")
    
    with tab2:
        st.markdown("## 📊 Stationarity & Differencing")
        
        st.markdown("""
        ### What is Stationarity?
        
        A **stationary time series** has:
        - Constant mean over time
        - Constant variance over time
        - Autocorrelation that depends only on lag, not on time
        
        Most financial time series are **non-stationary** (yields, prices, rates drift over time), 
        but their **differences** (returns, yield changes) are stationary.
        
        ### Testing for Stationarity:
        
        **Augmented Dickey-Fuller (ADF) Test:**
        - Null Hypothesis (H₀): Unit root exists (non-stationary)
        - If p-value < 0.05: Reject H₀ → Series is stationary
        - If p-value ≥ 0.05: Fail to reject H₀ → Series is non-stationary
        
        **KPSS Test:**
        - Null Hypothesis (H₀): Series is stationary
        - If p-value > 0.05: Fail to reject H₀ → Series is stationary
        - If p-value ≤ 0.05: Reject H₀ → Series is non-stationary
        
        ### Differencing:
        
        If a series is non-stationary, we take the **first difference**:
        
        $$\\Delta Y_t = Y_t - Y_{t-1}$$
        
        For I(2) processes (two unit roots), we may need **second differencing**:
        
        $$\\Delta^2 Y_t = (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2})$$
        
        The parameter **d** in ARIMA(p,d,q) represents the order of differencing.
        """)
    
    with tab3:
        st.markdown("## 🔧 ARIMA(p,d,q) Components")
        
        st.markdown("""
        ### ARIMA Structure:
        
        **ARIMA(p, d, q)** consists of three components:
        
        #### **AR (AutoRegressive) - Order p**
        Regression of the variable on its own past values:
        
        $$Y_t = \\phi_1 Y_{t-1} + \\phi_2 Y_{t-2} + ... + \\phi_p Y_{t-p} + \\epsilon_t$$
        
        - Used when past values have strong influence on current value
        - Identified by significant spikes in **PACF** plot
        - Common for mean-reverting series
        
        #### **I (Integrated) - Order d**
        Number of times series must be differenced to achieve stationarity:
        
        - **d = 0**: Series is already stationary
        - **d = 1**: First difference is stationary (most common)
        - **d = 2**: Second difference is stationary (rare)
        
        #### **MA (Moving Average) - Order q**
        Regression of the variable on past forecast errors:
        
        $$Y_t = \\mu + \\epsilon_t + \\theta_1 \\epsilon_{t-1} + \\theta_2 \\epsilon_{t-2} + ... + \\theta_q \\epsilon_{t-q}$$
        
        - Used when recent shocks influence current value
        - Identified by significant spikes in **ACF** plot
        - Smooths out temporary fluctuations
        
        ### Practical Guidelines:
        
        | ACF Pattern | PACF Pattern | Model | Characteristics |
        |---|---|---|---|
        | Decays | Cuts off at lag p | AR(p) | Past values matter |
        | Cuts off at lag q | Decays | MA(q) | Recent shocks matter |
        | Decays | Decays | ARMA(p,q) | Mixed process |
        | None significant | None significant | White Noise | Random |
        """)
    
    with tab4:
        st.markdown("## ⚠️ Risk Management in Fixed Income")
        
        st.markdown("""
        ### Key Risk Metrics:
        
        #### **Duration & Convexity**
        - **Duration**: Measures interest rate sensitivity (in years)
        - **Modified Duration**: Dollar price change per basis point move
        - **Convexity**: Accounts for non-linear price changes
        
        Formula: **% Price Change ≈ -Duration × Yield Change + 0.5 × Convexity × (Yield Change)²**
        
        #### **Basis Point (bps) Analysis**
        - 1 basis point = 0.01% = 1/100 of 1%
        - Used universally in fixed income markets
        - Bond prices move inversely to yields
        
        Example: If a 10Y bond has 6 years duration and yields rise by 50 bps:
        - Expected price decline ≈ -6 × 0.50% = -3.0%
        
        #### **Value at Risk (VaR)**
        - Probability of maximum loss within a confidence interval
        - 95% VaR: 5% chance of loss exceeding this amount
        - Common in portfolio management and regulatory capital
        
        #### **Scenario Analysis**
        - Parallel shift: All yields move by same amount
        - Non-parallel shift: Steepening/Flattening of curve
        - Risk reversal: Testing extreme but plausible scenarios
        
        ### Interest Rate Forecasting for Risk:
        
        Yield forecasts help:
        1. **Portfolio positioning**: Duration optimization
        2. **Hedging decisions**: Futures, options, swaps
        3. **Asset-liability management**: Matching cash flows
        4. **Return forecasting**: Expected returns on bonds
        5. **Risk capital allocation**: Managing concentration risk
        """)
    
    with tab5:
        st.markdown("## 🌍 Real-World Applications")
        
        st.markdown("""
        ### Central Banks & Monetary Policy
        - **RBI (Reserve Bank of India)** uses yield curve forecasting for policy decisions
        - **Federal Reserve** monitors 10Y Treasury for inflation expectations
        - Helps design repo operations and liquidity management
        
        ### Bond Portfolio Management
        - **Active traders**: Use ARIMA for short-term tactical positioning
        - **Fixed income funds**: Asset allocation decisions
        - **Bond dealers**: Inventory risk management
        
        ### Corporate Finance
        - **Debt issuance timing**: Issue bonds when yields are expected to rise
        - **Refinancing decisions**: Refinance maturing debt proactively
        - **M&A financing**: Lock in rates for expected deals
        
        ### Mortgage & Real Estate
        - **Mortgage origination**: Volume forecasting based on rate expectations
        - **Home buying decisions**: Timing purchases around rate expectations
        - **Mortgage-backed securities**: Prepayment risk analysis
        
        ### Trading & Speculation
        - **Yield curve trades**: Bet on steepening/flattening
        - **Relative value**: Comparing bonds across maturities
        - **Arbitrage**: Finding mispriced securities
        
        ### Market Intelligence
        - Communicating expectations to investors
        - Setting market reference points
        - Justifying investment theses
        
        ### Case Study: March 2020 (COVID-19 Pandemic)
        
        **The Scenario:**
        - Sudden spike in yields as panic selling hit markets
        - ARIMA models showed extreme deviation from historical patterns
        - Mean reversion analysis predicted yield compression
        
        **The Outcome:**
        - Federal Reserve intervened with massive QE
        - Yields fell sharply, rewarding mean-reversion strategies
        - Those who forecasted correctly made substantial returns
        """)
    
    # Additional resources
    st.markdown("---")
    st.markdown("### 📖 Recommended Readings")
    
    resources = {
        "Box-Jenkins Forecasting": "Box, G. E., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control",
        "ARIMA in Finance": "Tsay, R. S. (2010). Analysis of Financial Time Series (3rd ed.)",
        "Fixed Income Analytics": "Fabozzi, F. J. (2018). Bond Markets, Analysis and Strategies",
        "Risk Management": "Hull, J. C. (2018). Risk Management and Financial Institutions"
    }
    
    for title, citation in resources.items():
        st.write(f"📚 **{title}**  \n{citation}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #003366;'>
    <p><strong>The Mountain Path - World of Finance</strong></p>
    <p>Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence</p>
    <p style='font-size: 0.85rem; color: #666;'>
    This dashboard is provided for educational purposes. Past performance is not indicative of future results. 
    Use forecasting models with appropriate professional judgment.
    </p>
</div>
""", unsafe_allow_html=True)
