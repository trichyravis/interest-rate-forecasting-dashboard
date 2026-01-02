
QA_MASTERCLASS = [
    ("1. What is the 'Neutral Rate' in the context of Mean Reversion?", "The Neutral Rate (r*) is the theoretical equilibrium rate where the economy is at full employment with stable inflation. In our Stochastic models, this is represented by Theta (θ)."),
    ("2. Why does the Terminal use GARCH (1,1) instead of simple Standard Deviation?", "Standard deviation assumes volatility is constant. GARCH (1,1) accounts for 'Volatility Clustering,' recognizing that risk is regime-dependent and varies over time."),
    ("3. Explain the Box-Jenkins methodology used in the ARIMA tab.", "It is a 3-stage iterative process: 1. Identification (stationarity/autocorrelation), 2. Estimation (parameter fitting), and 3. Diagnostics (residual white noise check)."),
    ("4. What is the significance of the 'Square Root' term in the CIR Model?", "The term σ√r ensures that as interest rates approach zero, volatility also approaches zero. This mathematically prevents the simulation from producing negative interest rates."),
    ("5. How do we interpret the 'Kappa' (κ) parameter?", "Kappa represents the speed of mean reversion. It quantifies how aggressively the market pulls interest rates back toward the long-term equilibrium (Theta) after a shock."),
    ("6. What is 'Expected Shortfall' (ES) and why is it superior to VaR?", "Value-at-Risk only tells you the threshold of a loss. ES (or Conditional VaR) tells you the average magnitude of loss once that threshold is breached, capturing 'tail risk' more effectively."),
    ("7. What is 'Stationarity' and why is it required for ARIMA?", "A series is stationary if its mean and variance stay constant over time. Since yields trend, we use first-order differencing (d=1) to remove the trend and stabilize the data."),
    ("8. How does 'Duration' interact with these forecast models?", "Forecasted yield shifts are applied to the portfolio's modified duration to estimate the potential impact on the Net Asset Value (NAV) of a bond portfolio."),
    ("9. What are the limitations of the Vasicek Model?", "The primary limitation is that it assumes constant volatility and can theoretically allow interest rates to become negative, which is economically rare."),
    ("10. Define 'White Noise' in the Diagnostics tab.", "White noise indicates that the residuals (errors) of the model are purely random. This proves the model has extracted all available information from the historical data."),
    ("11. What is a 'Walk-Forward' Backtest?", "It is a validation technique where the model is trained on one segment of data and tested on the following segment, simulating real-world predictive performance."),
    ("12. How does 'Convexity' affect yield risk?", "While duration measures linear risk, convexity accounts for the curvature of the price-yield relationship. Forecasted paths help quantify this non-linear risk."),
    ("13. What is 'Drift' in a Stochastic Differential Equation (SDE)?", "Drift is the deterministic component of the model that defines the directional trend toward the equilibrium, as opposed to the random 'Diffusion' or noise component."),
    ("14. Why is RMSE the preferred metric for Backtesting?", "Root Mean Square Error (RMSE) penalizes larger errors more heavily, making it an excellent gauge for the reliability of interest rate predictions."),
    ("15. When should an analyst prioritize ARIMA over CIR?", "ARIMA should be prioritized for short-term technical forecasting (1-10 days), whereas CIR is superior for long-term strategic pathing and asset-liability management.")
]
