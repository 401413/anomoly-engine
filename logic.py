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

    # ... [Keep MODULE 1, 2, and 3 exactly as they are] ...

    # --- MODULE 4: OPTIONS & BETA ---
    def _fetch_macro_vix(self):
        try:
            vix_1d = self.to_scalar(yf.Ticker("^VIX1D").history(period="1d")['Close'].iloc[-1]) / 100.0
            vix_30d = self.to_scalar(yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]) / 100.0
            return vix_1d, vix_30d
        except:
            return 0.0, 0.0

    def _calculate_beta(self, ticker, period="1y"):
        try:
            stock = yf.Ticker(ticker).history(period=period)['Close'].pct_change().dropna()
            spy = yf.Ticker("SPY").history(period=period)['Close'].pct_change().dropna()
            data = pd.concat([stock, spy], axis=1).dropna()
            data.columns = ['Stock', 'SPY']
            cov = np.cov(data['Stock'], data['SPY'])[0][1]
            var = np.var(data['SPY'])
            return cov / var if var > 0 else 1.0
        except:
            return 1.0

    def _process_chain_window(self, tk, exp_date, curr_price, macro_vix, beta):
        try:
            chain = tk.option_chain(exp_date)
            calls, puts = chain.calls, chain.puts
            
            if calls.empty and puts.empty: return None

            call_vol = calls['volume'].sum()
            put_vol = puts['volume'].sum()
            total_vol = call_vol + put_vol
            
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            total_oi = call_oi + put_oi
            
            pcr = put_vol / call_vol if call_vol > 0 else 0.0

            atm_calls = calls[(calls['strike'] > curr_price * 0.85) & (calls['strike'] < curr_price * 1.15)]
            atm_puts = puts[(puts['strike'] > curr_price * 0.85) & (puts['strike'] < curr_price * 1.15)]
            
            iv_calls = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else 0.0
            iv_puts = atm_puts['impliedVolatility'].mean() if not atm_puts.empty else 0.0
            iv = (iv_calls + iv_puts) / 2 if (iv_calls > 0.01 and iv_puts > 0.01) else max(iv_calls, iv_puts)

            beta_adj_vix = macro_vix * beta

            return {
                "Date": exp_date,
                "IV": iv,
                "Macro_VIX": macro_vix,
                "Beta_Adj_VIX": beta_adj_vix,
                "Vol_Premium": iv - beta_adj_vix,
                "PCR": pcr,
                "Total_Volume": total_vol,
                "Call_Vol": call_vol,
                "Put_Vol": put_vol,
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

            hist_df = tk.history(period="3mo")
            hv = 0
            if not hist_df.empty:
                log_ret = np.log(hist_df['Close']/hist_df['Close'].shift(1))
                hv = log_ret.rolling(window=30).std().iloc[-1] * np.sqrt(252)

            vix_1d, vix_30d = self._fetch_macro_vix()
            beta = self._calculate_beta(clean_ticker)

            today = datetime.today()
            date_objs = [(d, (datetime.strptime(d, '%Y-%m-%d') - today).days) for d in dates]
            
            dte_0_matches = [d for d, days in date_objs if 0 <= days <= 1]
            dte_0_date = dte_0_matches[0] if dte_0_matches else None
            
            dte_30_matches = [d for d, days in date_objs if 27 <= days <= 37]
            dte_30_date = min(dte_30_matches, key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - today).days - 30)) if dte_30_matches else min(dates, key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - today).days - 30))

            data_0dte = self._process_chain_window(tk, dte_0_date, curr_price, vix_1d, beta) if dte_0_date else None
            data_30dte = self._process_chain_window(tk, dte_30_date, curr_price, vix_30d, beta) if dte_30_date else None

            return {
                "Current_Price": curr_price,
                "HV_30D": hv,
                "Beta": beta,
                "0DTE": data_0dte,
                "30DTE": data_30dte
            }
        except Exception as e:
             return {"Error": str(e)}

    # --- MODULE 5: WAVELET REGIME ---
    def generate_wavelet_energy(self, ticker):
        try:
            clean_ticker = ticker.split(" ")[0].upper()
            data = yf.download(clean_ticker, period="60d", interval="5m", progress=False)
            if data.empty: return None, None, None
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

            prices = data['Close'].to_numpy()
            
            # Perform Discrete Wavelet Transform (Daubechies 4)
            coeffs = pywt.wavedec(prices, 'db4', level=4)
            
            # Extract Low Frequency (Approximation / Scale 4) and High Frequency (Detail / Scale 1)
            cA4 = coeffs[0] 
            cD1 = coeffs[-1]
            
            # Calculate rolling energy
            energy_low = np.convolve(cA4**2, np.ones(10)/10, mode='same')
            energy_high = np.convolve(cD1**2, np.ones(10)/10, mode='same')
            
            # Prevent division by zero
            energy_high[energy_high == 0] = 1e-10
            ratio = energy_low[:len(prices)] / energy_high[:len(prices)] # Approximation resize
            
            # Normalize array lengths for plotting
            target_len = min(len(prices), len(ratio))
            
            return data.index[-target_len:], prices[-target_len:], ratio[-target_len:]
        except Exception as e:
            print(e)
            return None, None, None
