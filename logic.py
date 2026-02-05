import yfinance as yf
import pandas as pd
import numpy as np
from scipy import signal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class StrategyEngine:
    def __init__(self):
        self.liquidity_blacklist = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'GOOGL', 'BIL', 'SHV', 'CASH_USD']
        self.analyzer = SentimentIntensityAnalyzer() # For Text/Sound Analysis

    # --- HELPERS TO PREVENT ERRORS ---
    def to_scalar(self, val):
        """Forces complex data structures into a single float."""
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                if val.empty: return 0.0
                return float(val.iloc[0])
            if isinstance(val, (np.ndarray, np.generic)):
                return float(val.item())
            return float(val)
        except: return 0.0

    def to_scalar_array(self, series):
        """Forces complex data structures into a clean 1D Numpy Array."""
        try:
            # If it's a DataFrame (multi-index), take first column
            if isinstance(series, pd.DataFrame):
                return series.iloc[:, 0].to_numpy()
            return series.to_numpy()
        except:
            return np.array([])

    # --- MODULE 1: PORTFOLIO SANITIZER ---
    def get_fund_holdings(self, fund_ticker):
        print(f"📡 FETCHING {fund_ticker}...")
        try:
            fund = yf.Ticker(fund_ticker)
            holdings_data = fund.funds_data.top_holdings
            clean_holdings = {}
            if holdings_data is not None:
                for index, row in holdings_data.iterrows():
                    clean_holdings[str(index).strip()] = row['Holding Percent']
            return clean_holdings
        except: return {}

    def analyze_holding_health(self, ticker):
        try:
            ticker = ticker.split(" ")[0]
            data = yf.download(ticker, period="3mo", progress=False)
            
            if data.empty: return 0.0, 0.0, 0.0

            # Flatten Multi-Index
