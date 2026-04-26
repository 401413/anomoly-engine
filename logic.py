import yfinance as yf
import pandas as pd
import numpy as np
from scipy import signal
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class StrategyEngine:
    def __init__(self):
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

    # ... [Keep your existing MODULE 1, 2, and 3 exactly as they are] ...

    # --- MODULE 4: OPTIONS (DURATION & LIQUIDITY) ---
    def _fetch_macro_vix(self):
        """Fetches the 0DTE VIX and 30DTE VIX for macro baselining."""
        try:
            vix_1d = self.to_scalar(yf.Ticker("^VIX1D").history(period="1d")['Close'].iloc[-1]) / 100.0
            vix_30d = self.to_scalar(yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]) / 100.0
            return vix_1d, vix_30d
        except:
            return 0.0, 0.0

    def _process_chain_window(self, tk, exp_date, curr_price, macro_vix):
        """Processes a specific expiration date to extract IV, Volume, OI, and PCR."""
        try:
            chain = tk.option_chain(exp_date)
            calls, puts = chain.calls, chain.puts
            
            if calls.empty and puts.empty:
                return None

            call_vol = calls['volume'].sum()
            put_vol = puts['volume'].sum()
            total_vol = call_vol + put_vol
            
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            total_oi = call_oi + put_oi
            
            pcr = put_vol / call_vol if call_vol > 0 else 0.0

            # Filter Near-The-Money (NTM) for Implied Volatility calculation
            atm_calls = calls[(calls['strike'] > curr_price * 0.85) & (calls['strike'] < curr_price * 1.15)]
            atm_puts = puts[(puts['strike'] > curr_price * 0.85) & (puts['strike'] < curr_price * 1.15)]
            
            iv_calls = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else 0.0
            iv_puts = atm_puts['impliedVolatility'].mean() if not atm_puts.empty else 0.0
            
            # Blend IV, fallback if one side is illiquid
            if iv_calls > 0.01 and iv_puts > 0.01:
                iv = (iv_calls + iv_puts) / 2
            else:
                iv = max(iv_calls, iv_puts)

            return {
                "Date": exp_date,
                "IV": iv,
                "Macro_VIX": macro_vix,
                "Vol_Premium": iv - macro_vix if macro_vix > 0 else 0,
                "PCR": pcr,
                "Total_Volume": total_vol,
                "Total_OI": total_oi,
                "Calls_DF": calls,
                "Puts_DF": puts
            }
        except:
            return None

    def get_options_analytics(self, ticker):
        try:
            clean_ticker = ticker.split(" ")[0].upper()
            tk = yf.Ticker(clean_ticker)
            
            try: dates = tk.options
            except: dates = None
            if not dates: return {"Error": "No Options Chain Found."}

            curr_price = self.to_scalar(tk.fast_info['lastPrice'])
            if curr_price == 0: 
                hist = tk.history(period="1d")
                if not hist.empty: curr_price = self.to_scalar(hist['Close'].iloc[-1])

            # Historical Realized Volatility (30-day)
            hist_df = tk.history(period="3mo")
            hv = 0
            hist_vol_avg = 0
            if not hist_df.empty:
                log_ret = np.log(hist_df['Close']/hist_df['Close'].shift(1))
                hv = log_ret.rolling(window=30).std().iloc[-1] * np.sqrt(252)
                hist_vol_avg = hist_df['Volume'].mean() # Underlying historic volume

            vix_1d, vix_30d = self._fetch_macro_vix()

            # Date Parsing & Window Selection
            today = datetime.today()
            date_objs = [(d, (datetime.strptime(d, '%Y-%m-%d') - today).days) for d in dates]
            
            # Find 0DTE (0 to 1 days)
            dte_0_matches = [d for d, days in date_objs if 0 <= days <= 1]
            dte_0_date = dte_0_matches[0] if dte_0_matches else None
            
            # Find 30DTE (27 to 37 days)
            dte_30_matches = [d for d, days in date_objs if 27 <= days <= 37]
            if dte_30_matches:
                # Get the one closest to 30 days
                dte_30_date = min(dte_30_matches, key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - today).days - 30))
            else:
                # Fallback: Closest to 30 days if window is empty
                dte_30_date = min(dates, key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - today).days - 30))

            # Process the windows
            data_0dte = self._process_chain_window(tk, dte_0_date, curr_price, vix_1d) if dte_0_date else None
            data_30dte = self._process_chain_window(tk, dte_30_date, curr_price, vix_30d) if dte_30_date else None

            return {
                "Current_Price": curr_price,
                "HV_30D": hv,
                "Underlying_Avg_Vol": hist_vol_avg,
                "0DTE": data_0dte,
                "30DTE": data_30dte
            }
        except Exception as e:
             return {"Error": str(e)}
