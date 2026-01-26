# Global_Stock_Market_Forecasting
- 📘 **GitHub Notebook:**  
  [Global Stock Market Forecasting (ML & DL)](Global_Stock_Market_Forecasting_Using_ML_DL.ipynb)

- 🚀 **Run on Google Colab:**  
  [Open in Colab](https://colab.research.google.com/drive/1L_MUPQuQUL-V9RcRWi6rVWyElkI_sVA9)

<br>📌 Project Overview<br>

This project presents a comprehensive end-to-end stock market forecasting system that combines statistical analysis, machine learning, deep learning, 
and time-series modeling to analyze and predict global stock prices.

It covers the entire pipeline:
Data collection from Yahoo Finance
Exploratory data analysis (EDA)
Feature engineering & technical indicators
Stationarity & correlation analysis
Multiple forecasting models
Model comparison & explainability
The project is designed to be educational, modular, and production-ready, making it suitable for research, academic evaluation, and real-world financial analysis.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
<br>🎯 Objectives<br>
<br>Analyze historical stock price behavior of major global companies
Extract technical and seasonal features for time-series forecasting<br>
Build and compare multiple forecasting models:
Linear & Regularized Regression
LSTM-based Deep Learning models
Prophet
ARIMA
XGBoost
Perform multi-day future price forecasting
Explain model predictions using SHAP
Identify short-term investment trends (Uptrend / Downtrend)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br>Stocks Covered<br>

The project analyzes 15 major global stocks:

AAPL, MSFT, GOOGL, AMZN, META,
NVDA, TSLA, JPM, V, WMT,
JNJ, XOM, TSM, KO, MCD

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br>🛠️ Tech Stack<br>

Programming & Tools:-
Python 3.9+
Google Colab / Jupyter Notebook

Libraries Used:-
Data Handling: pandas, numpy
Visualization: matplotlib, seaborn, plotly, quantstats
Financial Data: yfinance, ta
ML Models: scikit-learn, xgboost
DL Models: tensorflow / keras
Time Series: statsmodels, pmdarima, prophet
Optimization: optuna
Explainability: shap
Portfolio Analysis: PyPortfolioOpt

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br>Project Workflow<br>
1️⃣ Data Collection
Stock price data fetched from Yahoo Finance
Time range: Jan 2020 → Present
OHLCV data extracted per ticker

2️⃣ Exploratory Data Analysis (EDA)

Price distribution (Open, Close)
Skewness analysis
Daily returns visualization
Volume trends
Open vs Close scatter analysis
Correlation heatmaps
Risk vs Return plots

3️⃣ Feature Engineering

Custom technical indicators were created:
Price Change
Daily Returns
Price Range
Volume Change
Moving Averages (10, 20, 50 days)
RSI (Relative Strength Index)
MACD & Signal Line
Bollinger Bands

Seasonal Indicators:
Spring
Summer
Autumn
Winter

4️⃣ Statistical Analysis
ADF Test for stationarity
ACF plots for lag analysis
Correlation analysis between stocks

5️⃣ Data Preprocessing
Min-Max normalization
Time-series aware train/test split (70/30)
Reshaping for LSTM input
Handling missing values

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br>Models Implemented<br>

🔹 1. Ridge Regression (Linear Model)
Hyperparameter tuning using Optuna

Metrics:
MAE
MSE
RMSE
R² Score
Used for baseline forecasting

🔹 2. LSTM + Conv1D (Deep Learning)

CNN-LSTM hybrid architecture
Bidirectional LSTM layers
Optimized using Optuna

Loss: Huber
Early stopping enabled
Multi-step future forecasting (30–60 days)

🔹 3. Facebook Prophet

Trend + seasonality modeling
Multiple external regressors
Weekly & yearly seasonality
Confidence intervals for predictions

🔹 4. ARIMA / ARIMAX

Automatic order selection using auto_arima
Exogenous regressors included
Business-day frequency handling

🔹 5. XGBoost Regressor

TimeSeriesSplit cross-validation
Hyperparameter tuning with Optuna
Feature-based forecasting
Robust performance evaluation

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br>Model Explainability<br>

SHAP (SHapley Additive Explanations) used to:
Identify most influential features
Understand feature impact on predictions
Generate SHAP summary & dependence plots
📈 Forecasting & Trend Analysis
Multi-day future price forecasting

<br>Moving-average-based trend detection:<br>

📈 Uptrend → Investment Opportunity
📉 Downtrend → Risk Alert
Interactive Plotly visualizations

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br>▶️ How to Run<br>
Option 1: Google Colab (Recommended)

-Open the notebook
-Run cells sequentially
-No local setup required

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br>Future Enhancements<br>

<br>-Add real-time data streaming<br>
<br>-Deploy as a web dashboard (Streamlit / FastAPI)<br>
<br>-Reinforcement learning for trading strategies<br>
<br>-Multi-asset portfolio optimization<br>
<br>-News & sentiment analysis integration<br>

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br>Author<br>

<br>Onkar Chougule<br>
<br>🎓 ENTC Engineering | AI & ML Enthusiast<br>
<br>📊 Financial Time-Series & Deep Learning<br>

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
