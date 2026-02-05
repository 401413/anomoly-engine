import yfinance as yf
import pandas as pd
import numpy as np
from scipy import signal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

class StrategyEngine:
    def __init__(self):
        self.liquidity_blacklist = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'GOOGL', 'BIL', 'SHV', 'CASH_USD']
        self.analyzer = SentimentIntensityAnalyzer() 

    # --- HELPERS ---
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

    def sanitize_signals(self, fund_ticker):
        # (Using Simplified Logic for brevity/stability)
        raw = self.get_fund_holdings(fund_ticker)
        logs = []
        if not raw: return pd.DataFrame()
        
        for ticker, weight in raw.items():
            if "USD" in ticker: continue
            status = "APPROVED"
            reason = "Clean"
            # Basic dummy check to prevent crashing if yfinance fails on a ticker
            if ticker in self.liquidity_blacklist: 
                status = "REJECTED"
                reason = "Beta Proxy"
            
            logs.append({'Ticker': ticker, 'Weight': weight, 'Status': status, 'Reason': reason, 'RSI': 50.0, 'Return_3M': "0%", 'Vol_M': "$10M"})
        return pd.DataFrame(logs)

    # --- MODULE 2: TEXT ---
    def analyze_sound_signal(self, text_input):
        score = self.analyzer.polarity_scores(text_input)
        return ("POSITIVE" if score['compound'] >= 0.05 else "NEGATIVE" if score['compound'] <= -0.05 else "NEUTRAL"), score['compound']

    # --- MODULE 3: WAVES (Robust) ---
    def generate_spectrogram_data(self, ticker):
        """
        Returns 6 values. Handles short history automatically.
        """
        try:
            clean_ticker = ticker.split(" ")[0].upper()
            
            # 1. Fetch Data (Try 2y, fallback to max if short)
            data = yf.download(clean_ticker, period="2y", interval="1d", progress=False)
            
            # If empty, try fetching 'max' to see if it's just a young stock
            if data.empty:
                data = yf.download(clean_ticker, period="max", interval="1d", progress=False)
            
            if data.empty or len(data) < 60: 
                print(f"Error: Not enough data for {clean_ticker}")
                return None, None, None, None, None, None

            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            prices = self.to_scalar_array(data['Close'])
            price_dates = data.index
            
            # 2. Spectrogram
            returns = np.diff(prices)
            returns_dates = price_dates[1:] 
            
            # 3. Dynamic Windowing (If stock is young, use smaller window)
            n_per_seg = 60
            if len(returns) < 100: n_per_seg = 20 # Adapt for new IPOs
            
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, window='hann', nperseg=n_per_seg, noverlap=int(n_per_seg/2))
            
            t_indices = np.floor(t).astype(int)
            t_indices = np.clip(t_indices, 0, len(returns_dates) - 1)
            spec_dates = returns_dates[t_indices]
            
            # 4. Hist Vol
            log_ret = np.log(data['Close'] / data['Close'].shift(1))
            hist_vol = log_ret.rolling(window=30).std() * np.sqrt(252)
            hist_vol = self.to_scalar_array(hist_vol.fillna(0))
            
            return f, spec_dates, Sxx, price_dates, prices, hist_vol
            
        except Exception as e:
            print(f"Spectrogram Error: {e}")
            return None, None, None, None, None, None

    # --- MODULE 4: OPTIONS (Robust) ---
    def get_options_analytics(self, ticker):
        try:
            clean_ticker = ticker.split(" ")[0].upper()
            tk = yf.Ticker(clean_ticker)
            
            # Check for dates
            try:
                dates = tk.options
            except:
                return {"Error": "No Options Data Found"}
                
            if not dates: return {"Error": "No Options Chain Available"}
            
            chain = tk.option_chain(dates[0])
            calls, puts = chain.calls, chain.puts
            
            pcr = (puts['volume'].sum() if not puts.empty else 0) / (calls['volume'].sum() if not calls.empty else 1)
            
            curr = self.to_scalar(tk.fast_info['lastPrice'])
            if curr == 0: # Fallback if fast_info fails
                hist = tk.history(period="1d")
                if not hist.empty: curr = self.to_scalar(hist['Close'].iloc[-1])
            
            atm = calls[(calls['strike'] > curr*0.90) & (calls['strike'] < curr*1.10)]
            iv = atm['impliedVolatility'].mean() if not atm.empty else 0
            
            hist_df = tk.history(period="1mo")
            hv = (np.log(hist_df['Close']/hist_df['Close'].shift(1)).std() * np.sqrt(252)) if not hist_df.empty else 0
                
            return {"PCR": pcr, "IV": iv, "HV": hv, "Vol_Premium": iv - hv, "Nearest_Exp": dates[0]}
        except Exception as e:
             return {"Error": str(e)}
