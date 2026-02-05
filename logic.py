import yfinance as yf
import pandas as pd
import numpy as np
from scipy import signal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class StrategyEngine:
    def __init__(self):
        self.liquidity_blacklist = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'GOOGL', 'BIL', 'SHV', 'CASH_USD']
        self.analyzer = SentimentIntensityAnalyzer() 

    # --- HELPERS ---
    def to_scalar(self, val):
        """Forces complex data into a single float."""
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                if val.empty: return 0.0
                return float(val.iloc[0])
            if isinstance(val, (np.ndarray, np.generic)):
                return float(val.item())
            return float(val)
        except: return 0.0

    def to_scalar_array(self, series):
        """Forces complex data into a 1D Numpy Array."""
        try:
            if isinstance(series, pd.DataFrame):
                return series.iloc[:, 0].to_numpy()
            return series.to_numpy()
        except:
            return np.array([])

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
            ticker = ticker.split(" ")[0]
            data = yf.download(ticker, period="3mo", progress=False)
            
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
            if "USD" in ticker or "CASH" in ticker: continue
            
            if ticker in self.liquidity_blacklist:
                logs.append({
                    'Ticker': ticker, 'Weight': weight, 'Status': 'REJECTED', 
                    'Reason': 'Beta Proxy', 'RSI': 0, 'Return_3M': 0, 'Vol_M': 0
                })
                continue

            run_up, rsi, vol_m = self.analyze_holding_health(ticker)
            
            is_trap = (run_up > 0.20) and (vol_m < 20) 
            is_hot = (run_up > 0.30) or (rsi > 75)
            
            status = 'APPROVED'
            reason = 'Clean Signal'
            if is_trap: status, reason = 'REJECTED', 'Reflexivity Trap'
            elif is_hot: status, reason = 'REJECTED', 'Overheated'

            logs.append({
                'Ticker': ticker, 'Weight': weight, 'Status': status, 'Reason': reason,
                'RSI': round(rsi, 1), 'Return_3M': f"{run_up:.1%}", 'Vol_M': f"${vol_m:.1f}M"
            })
            
        return pd.DataFrame(logs)

    # --- MODULE 2: TEXT ---
    def analyze_sound_signal(self, text_input):
        score = self.analyzer.polarity_scores(text_input)
        compound = score['compound']
        if compound >= 0.05: return "POSITIVE", compound
        elif compound <= -0.05: return "NEGATIVE", compound
        else: return "NEUTRAL", compound

    # --- MODULE 3: WAVES (UPDATED) ---
    def generate_spectrogram_data(self, ticker):
        """
        Returns: Frequencies (f), Times (t), Intensities (Sxx), AND Prices.
        """
        try:
            # 1. Get History
            data = yf.download(ticker.split(" ")[0], period="2y", interval="1d", progress=False)
            if data.empty: return None, None, None, None

            # 2. Clean & Flatten
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            prices = self.to_scalar_array(data['Close'])
            
            # 3. Detrend
            returns = np.diff(prices)
            
            # 4. Signal Processing
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, window='hann', nperseg=60, noverlap=50)
            
            # Return 4 values to match app.py expectation
            return f, t, Sxx, prices
            
        except Exception as e:
            print(f"Spectrogram Error: {e}")
            return None, None, None, None
