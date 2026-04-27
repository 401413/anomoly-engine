import yfinance as yf
import pandas as pd
import numpy as np
from scipy import signal
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pywt

class StrategyEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer() 

    def to_scalar(self, val):
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                if val.empty: return 0.0
                return float(val.iloc[0])
            if isinstance(val, (np.ndarray, np.generic)):
                return float(val.item())
            return float(val)
        except: return 0.0

    def to_scalar_array(self, series):
        try:
            if isinstance(series, pd.DataFrame):
                return series.iloc[:, 0].to_numpy()
            return series.to_numpy()
        except: return np.array([])

    # --- MODULE 1: PORTFOLIO ---
    def get_fund_holdings(self, fund_ticker):
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
            clean_ticker = ticker.split(" ")[0]
            data = yf.download(clean_ticker, period="6mo", progress=False)
            if data.empty: return 0.0, 0.0, 0.0
            
            if isinstance(data.columns, pd.MultiIndex): 
                data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()]

            start = self.to_scalar(data['Close'].iloc[0])
            end = self.to_scalar(data['Close'].iloc[-1])
            run_up = (end - start) / start if start != 0 else 0.0
            
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs_gain = self.to_scalar(gain.iloc[-1])
            rs_loss = self.to_scalar(loss.iloc[-1])
            rsi = 100 - (100 / (1 + (rs_gain / rs_loss))) if rs_loss != 0 else 50.0
            
            vol = self.to_scalar(data['Volume'].mean())
            price = self.to_scalar(data['Close'].mean())
            dollar_vol = (vol * price) / 1_000_000 
            
            return run_up, rsi, dollar_vol
        except: return 0.0, 0.0, 0.0

    def sanitize_signals(self, fund_ticker):
        raw = self.get_fund_holdings(fund_ticker)
        logs = []
        if not raw: return pd.DataFrame()
        for ticker, weight in raw.items():
            if "USD" in ticker: continue
            run_up, rsi, vol_m = self.analyze_holding_health(ticker)
            is_trap = (run_up > 0.20 and vol_m < 20)
            is_hot = (run_up > 0.30 or rsi > 75)
            status, reason = ('REJECTED', 'Reflexivity Trap') if is_trap else ('REJECTED', 'Overheated') if is_hot else ('APPROVED', 'Clean Signal')
            logs.append({'Ticker': ticker, 'Weight': weight, 'Status': status, 'Reason': reason, 'RSI': round(rsi, 1), 'Return_3M': f"{run_up:.1%}", 'Vol_M': f"${vol_m:.1f}M"})
        return pd.DataFrame(logs)

    # --- MODULE 2: TEXT ---
    def analyze_sound_signal(self, text_input):
        score = self.analyzer.polarity_scores(text_input)
        compound = score['compound']
        return ("POSITIVE" if compound >= 0.05 else "NEGATIVE" if compound <= -0.05 else "NEUTRAL"), compound

    # --- MODULE 3: WAVES ---
    def generate_spectrogram_data(self, ticker):
        try:
            clean_ticker = ticker.split(" ")[0].upper()
            data = yf.download(clean_ticker, period="2y", interval="1d", progress=False)
            
            if data.empty: 
                data = yf.download(clean_ticker, period="max", interval="1d", progress=False)
            
            if data.empty or len(data) < 30: return None, None, None, None, None, None

            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            prices = self.to_scalar_array(data['Close'])
            price_dates = data.index
            
            returns = np.diff(prices)
            returns_dates = price_dates[1:] 
            
            n_per_seg = 60 if len(returns) > 100 else 10
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, window='hann', nperseg=n_per_seg, noverlap=int(n
