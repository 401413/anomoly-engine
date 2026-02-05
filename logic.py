import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class StrategyEngine:
    def __init__(self):
        self.liquidity_blacklist = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'GOOGL', 'BIL', 'SHV', 'CASH_USD']
        self.analyzer = SentimentIntensityAnalyzer() # For "Sound" Analysis

    def to_scalar(self, val):
        """Forces data to a single float to prevent errors."""
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                if val.empty: return 0.0
                return float(val.iloc[0])
            return float(val)
        except: return 0.0

    def get_fund_holdings(self, fund_ticker):
        # ... (Same as before, keep your existing get_fund_holdings code) ...
        # For brevity, I am assuming you kept the robust version from the previous step.
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
        """Returns Metrics AND Data for display."""
        try:
            ticker = ticker.split(" ")[0]
            data = yf.download(ticker, period="3mo", progress=False)
            
            if data.empty: return 0.0, 0.0, 0.0

            # Flatten Multi-Index
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()]

            # 1. Metrics
            start = self.to_scalar(data['Close'].iloc[0])
            end = self.to_scalar(data['Close'].iloc[-1])
            run_up = (end - start) / start if start != 0 else 0.0
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs_gain = self.to_scalar(gain.iloc[-1])
            rs_loss = self.to_scalar(loss.iloc[-1])
            rsi = 100 - (100 / (1 + (rs_gain / rs_loss))) if rs_loss != 0 else 50.0
            
            # Liquidity ($M)
            vol = self.to_scalar(data['Volume'].mean())
            price = self.to_scalar(data['Close'].mean())
            dollar_vol = (vol * price) / 1_000_000 # Convert to Millions
            
            return run_up, rsi, dollar_vol

        except: return 0.0, 0.0, 0.0

    def sanitize_signals(self, fund_ticker):
        raw = self.get_fund_holdings(fund_ticker)
        logs = []
        
        if not raw: return pd.DataFrame()

        for ticker, weight in raw.items():
            if "USD" in ticker or "CASH" in ticker: continue
            
            # A. BLACKLIST
            if ticker in self.liquidity_blacklist:
                logs.append({
                    'Ticker': ticker, 'Weight': weight, 'Status': 'REJECTED', 
                    'Reason': 'Beta Proxy', 
                    'RSI': 0, 'Return_3M': 0, 'Vol_M': 0 # Placeholders
                })
                continue

            # B. ANALYSIS
            run_up, rsi, vol_m = self.analyze_holding_health(ticker)
            
            # Logic
            is_trap = (run_up > 0.20) and (vol_m < 20) # <$20M daily vol
            is_hot = (run_up > 0.30) or (rsi > 75)
            
            status = 'APPROVED'
            reason = 'Clean Signal'
            
            if is_trap:
                status = 'REJECTED'
                reason = 'Reflexivity Trap'
            elif is_hot:
                status = 'REJECTED'
                reason = 'Overheated'

            logs.append({
                'Ticker': ticker, 
                'Weight': weight, 
                'Status': status, 
                'Reason': reason,
                'RSI': round(rsi, 1),
                'Return_3M': f"{run_up:.1%}",
                'Vol_M': f"${vol_m:.1f}M"
            })
            
        return pd.DataFrame(logs)

    def analyze_sound_signal(self, text_input):
        """
        Analyzes the 'Sound' (Text Sentiment) of a CEO Statement/News.
        Positive Score > 0.05 = Buy Signal.
        """
        score = self.analyzer.polarity_scores(text_input)
        compound = score['compound']
        
        if compound >= 0.05: return "POSITIVE", compound
        elif compound <= -0.05: return "NEGATIVE", compound
        else: return "NEUTRAL", compound
