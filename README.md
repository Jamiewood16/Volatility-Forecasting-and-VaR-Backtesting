# Volatility Forecasting and VaR Backtesting

This project was part of my ACFI233 Econometrics for Finance II module.  
It focuses on forecasting realised volatility of the S&P 500 ETF (SPY) and backtesting Value-at-Risk models.

## Contents
- `EconII.py` → Python code for analysis
- `Individual_Project_Data.csv` → Input dataset (SPY realised volatility measures, 2000–2023)
- `ECON II.docx` → Full report (methods, results, discussion)

## Methods
- Unbiasedness test of VIX²
- Summary stats + Jarque-Bera normality test
- HAR and SHAR models (OLS with HAC errors)
- Out-of-sample forecasting & model averaging
- VaR forecasting and backtesting (PoF, Traffic Light test)

## Key Findings
- VIX² is biased, log transformation gives best explanatory power.
- HAR-log model provides highest in-sample fit.
- Model averaging balances bias and efficiency in out-of-sample forecasts.
- All VaR models fail backtesting → unsuitable for capital allocation.

## Requirements
- Python 3.x
- pandas, numpy, statsmodels, matplotlib

## How to Run
```bash
python EconII.py
