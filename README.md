# 🏦 Interest Rate Forecasting Dashboard

### Real-Time Box-Jenkins Time Series Modeling for Global & Indian Yields

**Developed by: Prof. V. Ravichandran**  
*28+ Years Corporate Finance & Banking Experience | 10+ Years Academic Excellence*

---

## 🏔️ Project Overview

This dashboard utilizes the **Box-Jenkins (ARIMA) Methodology** to forecast interest rate yields for benchmarks including the **India 10Y Government Security (RBI)** and **US 10Y Treasury**. The application bridges classical financial econometrics with modern data analytics, designed for educators, students, and financial professionals studying fixed income securities and yield forecasting.

The dashboard integrates historical data retrieval, advanced statistical modeling, backtesting capabilities, and risk analytics into an intuitive web-based interface accessible for educational and analytical purposes.

---

## 🚀 Key Features

### 1. **Univariate ARIMA Modeling**
   - Automated ARIMA(p,d,q) parameter identification using Auto-ARIMA
   - Full Box-Jenkins iterative process implementation
   - Stationarity testing (ADF, KPSS)
   - Automatic differencing and seasonal decomposition

### 2. **Basis Point (bps) Calculator**
   - Direct forecasting of yield shifts in bond-market terminology
   - Translates statistical forecasts into basis points (0.01% increments)
   - Sensitivity analysis for varying forecast horizons
   - Real-time bps impact visualization

### 3. **Hindcasting (Backtesting)**
   - Train/Test split validation (80/20 default)
   - Out-of-sample performance evaluation
   - Variance analysis and statistical accuracy metrics
   - Visual comparison of actual vs. forecasted values

### 4. **Risk Analytics**
   - 95% Confidence Intervals with statistical fan charts
   - Annualized volatility indexing
   - Forecast uncertainty quantification
   - Residual diagnostics for white noise validation

### 5. **Multi-Benchmark Support**
   - India 10Y G-Sec (RBI) - INR yields
   - US 10Y Treasury - USD yields
   - Easy extensibility to additional benchmarks

### 6. **Educational Hub**
   - Integrated learning modules on ARIMA lifecycle
   - Theoretical background on Box-Jenkins methodology
   - Risk management concepts in fixed income
   - Interactive explanations of model diagnostics

---

## 🛠️ Installation & Local Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/interest-rate-forecasting-dashboard.git
cd interest-rate-forecasting-dashboard
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

---

## 📊 Dashboard Sections

### **Main Dashboard**
- Real-time data visualization for India 10Y G-Sec and US 10Y Treasury
- Historical price trends and yield curves
- Current market statistics and volatility metrics

### **ARIMA Modeling**
- Parameter selection interface (p, d, q values)
- Stationarity test results
- Model diagnostics and ACF/PACF plots
- Residual analysis with Ljung-Box test

### **Forecasting & Risk**
- Forecast horizon selection (1-252 trading days)
- Point forecasts with confidence intervals
- Basis point calculations
- Statistical fan chart visualization

### **Backtesting Module**
- Historical performance evaluation
- Out-of-sample accuracy metrics (MAE, RMSE, MAPE)
- Train/Test split visualization
- Model reliability assessment

### **Educational Content**
- Box-Jenkins methodology overview
- Stationarity and differencing explanation
- Risk management applications
- Real-world case studies in fixed income

---

## 📚 Methodology

The application follows the **iterative Box-Jenkins process** for time series forecasting:

#### 1. **Identification Phase**
   - Test for stationarity using Augmented Dickey-Fuller (ADF) test
   - Determine differencing order (d) if non-stationary
   - Analyze ACF and PACF plots for AR and MA components
   - Initial parameter estimation (p, q)

#### 2. **Estimation Phase**
   - Maximum Likelihood Estimation (MLE) for coefficient optimization
   - Auto-ARIMA for automated parameter selection
   - Model validation on alternative specifications
   - Information criteria comparison (AIC, BIC)

#### 3. **Diagnostic Checking Phase**
   - Ljung-Box test for residual white noise
   - Visual inspection of residuals (normality, heteroskedasticity)
   - ACF/PACF of residuals for independence
   - Quantile-Quantile (Q-Q) plots for normality assessment

#### 4. **Forecasting Phase**
   - Generate point forecasts with confidence intervals
   - Calculate forecast error bounds
   - Basis point translation for bond market interpretation
   - Risk quantification using volatility estimates

---

## 📈 Data Sources

- **India 10Y G-Sec**: Yahoo Finance (^INFY ticker equivalent or direct RBI data)
- **US 10Y Treasury**: Yahoo Finance (^TNX ticker)
- **Real-time Updates**: Daily automated data refresh via yfinance API

---

## 🔧 Configuration & Customization

### Modifying Color Scheme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FFD700"           # Gold accent
backgroundColor = "#FFFFFF"        # White background
secondaryBackgroundColor = "#F0F2F6"  # Light gray
textColor = "#003366"             # Dark blue text
```

### Adjusting Data Parameters
Modify the `app.py` configuration section:
```python
# Historical data window (in years)
HISTORY_YEARS = 10

# Forecast horizons
MAX_FORECAST_DAYS = 252

# Confidence level for intervals
CONFIDENCE_LEVEL = 0.95
```

---

## 🚀 Deployment to Streamlit Cloud

### Step 1: Push to GitHub
1. Create a repository named `interest-rate-forecasting-dashboard`
2. Push all files (including `app.py`, `requirements.txt`, `config.toml`)
3. Ensure `.gitignore` is properly configured

### Step 2: Connect to Streamlit Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Choose `app.py` as the main file
5. Click "Deploy"

Your dashboard will be live within minutes at a custom Streamlit URL.

---

## 📊 Performance & Limitations

### Strengths
✓ Fast execution for real-time analysis  
✓ Educational clarity in model interpretation  
✓ Flexible parameter tuning  
✓ Robust statistical foundations  

### Limitations
⚠ Univariate modeling (no exogenous variables)  
⚠ Assumes linear relationships  
⚠ Sensitive to structural breaks in data  
⚠ Requires stationary series after differencing  

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report issues via GitHub Issues
- Suggest improvements
- Submit pull requests with enhancements
- Share alternative implementations

For major changes, please open an issue first to discuss proposed modifications.

---

## 📧 Contact & Support

**Prof. V. Ravichandran**  
*The Mountain Path - World of Finance*

📱 LinkedIn: [linkedin.com/in/trichyravis](https://www.linkedin.com/in/trichyravis)  
🌐 Educational Platform: The Mountain Path - World of Finance  

For questions about the dashboard, ARIMA methodology, or educational licensing, please reach out directly.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Educational Use**: This software is provided for educational purposes in advanced financial risk management. Users should exercise appropriate judgment when applying forecasting models to real-world financial decisions.

---

## 🎓 Educational Context

This dashboard was developed as part of advanced financial risk management and fixed income securities curriculum. It serves as a practical implementation of theoretical concepts taught in:

- **Financial Risk Management**
- **Fixed Income Securities & Analysis**
- **Financial Derivatives**
- **Investment Banking**
- **Advanced Financial Modeling**

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Status**: Production Ready for Educational Use

# 🏦 Interest Rate Forecasting Dashboard (v1.1)
## Dual-Engine: Yahoo Finance & Federal Reserve (FRED)

**Developed by: Prof. V. Ravichandran**

## 🚀 New in this Version
- **FRED Integration**: Users can now input a FRED API Key to pull authoritative US economic data.
- **Resilient Data Fetching**: Implemented progressive retry logic (5s, 10s, 20s) for Yahoo Finance.
- **Streamlit 2026 Ready**: Updated all chart and table syntax to `width="stretch"`.
- **Basis Point (bps) Engine**: Clearer metrics for bond market movements.
