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

    def analyze_holding_health(self, ticker):
        try:
            ticker = ticker.split(" ")[0]
            data = yf.download(ticker, period="3mo", progress=False)
            if data.empty: return 0.0, 0.0, 0.0
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()]

            start, end = self.to_scalar(data['Close'].iloc[0]), self.to_scalar(data['Close'].iloc[-1])
            run_up = (end - start) / start if start != 0 else 0.0
            
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs_gain, rs_loss = self.to_scalar(gain.iloc[-1]), self.to_scalar(loss.iloc[-1])
            rsi = 100 - (100 / (1 + (rs_gain / rs_loss))) if rs_loss != 0 else 50.0
            
            vol, price = self.to_scalar(data['Volume'].mean()), self.to_scalar(data['Close'].mean())
            return run_up, rsi, (vol * price) / 1_000_000 
        except: return 0.0, 0.0, 0.0

    def sanitize_signals(self, fund_ticker):
        raw = self.get_fund_holdings(fund_ticker)
        logs = []
        if not raw: return pd.DataFrame()
        for ticker, weight in raw.items():
            if "USD" in ticker or "CASH" in ticker: continue
            if ticker in self.liquidity_blacklist:
                logs.append({'Ticker': ticker, 'Weight': weight, 'Status': 'REJECTED', 'Reason': 'Beta Proxy', 'RSI': 0, 'Return_3M': 0, 'Vol_M': 0})
                continue
            run_up, rsi, vol_m = self.analyze_holding_health(ticker)
            is_trap, is_hot = (run_up > 0.20 and vol_m < 20), (run_up > 0.30 or rsi > 75)
            status, reason = ('REJECTED', 'Reflexivity Trap') if is_trap else ('REJECTED', 'Overheated') if is_hot else ('APPROVED', 'Clean Signal')
            logs.append({'Ticker': ticker, 'Weight': weight, 'Status': status, 'Reason': reason, 'RSI': round(rsi, 1), 'Return_3M': f"{run_up:.1%}", 'Vol_M': f"${vol_m:.1f}M"})
        return pd.DataFrame(logs)

    # --- MODULE 2: TEXT ---
    def analyze_sound_signal(self, text_input):
        score = self.analyzer.polarity_scores(text_input)
        return ("POSITIVE" if score['compound'] >= 0.05 else "NEGATIVE" if score['compound'] <= -0.05 else "NEUTRAL"), score['compound']

    # --- MODULE 3: WAVES (FIXED ALIGNMENT) ---
    def generate_spectrogram_data(self, ticker):
        """
        Returns aligned arrays for plotting.
        """
        try:
            clean_ticker = ticker.split(" ")[0].upper()
            data = yf.download(clean_ticker, period="2y", interval="1d", progress=False)
            
            # Fallback for young stocks
            if data.empty: data = yf.download(clean_ticker, period="max", interval="1d", progress=False)
            if data.empty or len(data) < 60: return None, None, None, None, None, None

            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            prices = self.to_scalar_array(data['Close'])
            price_dates = data.index
            
            # Detrend
            returns = np.diff(prices)
            returns_dates = price_dates[1:] # Diff loses 1 point
            
            # Spectrogram
            n_per_seg = 60 if len(returns) > 100 else 20
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, window='hann', nperseg=n_per_seg, noverlap=int(n_per_seg/2))
            
            # Map Indices to Dates
            t_indices = np.floor(t).astype(int)
            t_indices = np.clip(t_indices, 0, len(returns_dates) - 1)
            spec_dates = returns_dates[t_indices]
            
            # --- CRITICAL FIX: SLICE PRICE TO MATCH SPECTROGRAM ---
            # The spectrogram starts later (due to windowing). We must find the start date 
            # and slice the 'prices' array so they align on the chart.
            start_date = spec_dates[0]
            
            # Create a mask to filter prices starting from the first spectrogram date
            mask = price_dates >= start_date
            aligned_prices = prices[mask]
            aligned_dates = price_dates[mask]
            
            # Calculate Hist Vol (aligned)
            log_ret = np.log(data['Close'] / data['Close'].shift(1))
            hist_vol = log_ret.rolling(window=30).std() * np.sqrt(252)
            hist_vol = self.to_scalar_array(hist_vol.fillna(0))
            # Slice Hist Vol using the same mask
            aligned_hist_vol = hist_vol[mask]
            
            return f, spec_dates, Sxx, aligned_dates, aligned_prices, aligned_hist_vol
            
        except Exception as e:
            print(f"Spectrogram Error: {e}")
            return None, None, None, None, None, None

    # --- MODULE 4: OPTIONS (Safe Mode) ---
    def get_options_analytics(self, ticker):
        try:
            clean_ticker = ticker.split(" ")[0].upper()
            tk = yf.Ticker(clean_ticker)
            
            try:
                dates = tk.options
            except: dates = None
            
            if not dates: return None # Graceful exit if no options
            
            chain = tk.option_chain(dates[0])
            calls, puts = chain.calls, chain.puts
            
            pcr = (puts['volume'].sum() if not puts.empty else 0) / (calls['volume'].sum() if not calls.empty else 1)
            
            curr = self.to_scalar(tk.fast_info['lastPrice'])
            if curr == 0: 
                hist = tk.history(period="1d")
                if not hist.empty: curr = self.to_scalar(hist['Close'].iloc[-1])
            
            atm = calls[(calls['strike'] > curr*0.90) & (calls['strike'] < curr*1.10)]
            iv = atm['impliedVolatility'].mean() if not atm.empty else 0
            
            hist_df = tk.history(period="1mo")
            hv = (np.log(hist_df['Close']/hist_df['Close'].shift(1)).std() * np.sqrt(252)) if not hist_df.empty else 0
                
            return {"PCR": pcr, "IV": iv, "HV": hv, "Vol_Premium": iv - hv, "Nearest_Exp": dates[0]}
        except:
             return None
