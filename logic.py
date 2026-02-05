import yfinance as yf
import pandas as pd
import numpy as np
from scipy import signal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class StrategyEngine:
    def __init__(self):
        # REMOVED the "Liquidity Blacklist" so you can scan ANYTHING (IREN, WULF, etc.)
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
            clean_ticker = ticker.split(" ")[0]
            # Use 'max' for small caps to ensure we get data even if gaps exist
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
            
            # Simplified Logic: High RSI = Overheated, High Mom/Low Vol = Trap
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
            
            # Less strict data check (allow for shorter history)
            if data.empty or len(data) < 30: return None, None, None, None, None, None

            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            prices = self.to_scalar_array(data['Close'])
            price_dates = data.index
            
            returns = np.diff(prices)
            returns_dates = price_dates[1:] 
            
            n_per_seg = 60 if len(returns) > 100 else 10 # Adaptive window
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, window='hann', nperseg=n_per_seg, noverlap=int(n_per_seg/2))
            
            t_indices = np.floor(t).astype(int)
            t_indices = np.clip(t_indices, 0, len(returns_dates) - 1)
            spec_dates = returns_dates[t_indices]
            
            # ALIGNMENT
            start_date = spec_dates[0]
            mask = price_dates >= start_date
            aligned_prices = prices[mask]
            aligned_dates = price_dates[mask]
            
            # HISTORICAL VOLATILITY (Realized)
            log_ret = np.log(data['Close'] / data['Close'].shift(1))
            hist_vol = log_ret.rolling(window=30).std() * np.sqrt(252)
            hist_vol = self.to_scalar_array(hist_vol.fillna(0))
            aligned_hist_vol = hist_vol[mask]
            
            return f, spec_dates, Sxx, aligned_dates, aligned_prices, aligned_hist_vol
            
        except Exception as e:
            print(f"Spectrogram Error: {e}")
            return None, None, None, None, None, None

    # --- MODULE 4: OPTIONS (RAW / ILLIQUID MODE) ---
    def get_options_analytics(self, ticker):
        try:
            clean_ticker = ticker.split(" ")[0].upper()
            tk = yf.Ticker(clean_ticker)
            
            # 1. Dates Check
            try: dates = tk.options
            except: dates = None
            if not dates: return {"Error": "No Options Chain Found (Ticker might not have options)"}
            
            # 2. Raw Chain Fetch (No Filtering)
            chain = tk.option_chain(dates[0])
            calls = chain.calls
            puts = chain.puts
            
            # 3. Handle Empty Dataframes (Ghost Chains)
            if calls.empty and puts.empty:
                return {"Error": "Options exist but data is empty right now."}

            # 4. Illiquid Pricing Logic (Use Bid/Ask Midpoint if LastPrice is stale)
            # We treat 0 volume as VALID data now.
            
            # Calc Put/Call Ratio (Volume based)
            call_vol = calls['volume'].sum()
            put_vol = puts['volume'].sum()
            pcr = put_vol / call_vol if call_vol > 0 else 0.0
            
            # 5. Implied Volatility (The "Fear" Metric)
            # We average the IV of options Near-The-Money (NTM)
            curr = self.to_scalar(tk.fast_info['lastPrice'])
            if curr == 0: 
                hist = tk.history(period="1d")
                if not hist.empty: curr = self.to_scalar(hist['Close'].iloc[-1])
            
            # Wide filter for illiquid stocks (Strike +/- 20%)
            atm_calls = calls[(calls['strike'] > curr*0.80) & (calls['strike'] < curr*1.20)]
            
            # If ATM calls have 0 IV (common in bad data), try Puts
            if not atm_calls.empty and atm_calls['impliedVolatility'].mean() > 0.01:
                iv = atm_calls['impliedVolatility'].mean()
            else:
                # Fallback to Puts
                atm_puts = puts[(puts['strike'] > curr*0.80) & (puts['strike'] < curr*1.20)]
                iv = atm_puts['impliedVolatility'].mean() if not atm_puts.empty else 0
            
            # 6. Historical Volatility (The "Reality" Metric)
            hist_df = tk.history(period="3mo")
            hv = 0
            if not hist_df.empty:
                # 30-Day Realized Vol
                log_ret = np.log(hist_df['Close']/hist_df['Close'].shift(1))
                hv = log_ret.rolling(window=30).std().iloc[-1] * np.sqrt(252)

            return {
                "PCR": pcr, 
                "IV": iv, 
                "HV": hv, 
                "Vol_Premium": iv - hv, 
                "Nearest_Exp": dates[0],
                "Current_Price": curr
            }
        except Exception as e:
             return {"Error": str(e)}
