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
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, window='hann', nperseg=n_per_seg, noverlap=int(n_per_seg/2))
            
            t_indices = np.floor(t).astype(int)
            t_indices = np.clip(t_indices, 0, len(returns_dates) - 1)
            spec_dates = returns_dates[t_indices]
            
            start_date = spec_dates[0]
            mask = price_dates >= start_date
            aligned_prices = prices[mask]
            aligned_dates = price_dates[mask]
            
            log_ret = np.log(data['Close'] / data['Close'].shift(1))
            hist_vol = log_ret.rolling(window=30).std() * np.sqrt(252)
            hist_vol = self.to_scalar_array(hist_vol.fillna(0))
            aligned_hist_vol = hist_vol[mask]
            
            return f, spec_dates, Sxx, aligned_dates, aligned_prices, aligned_hist_vol
            
        except Exception as e:
            print(f"Spectrogram Error: {e}")
            return None, None, None, None, None, None

    # --- MODULE 4: OPTIONS & BETA ---
    def _fetch_macro_vix(self):
        try:
            vix_1d = self.to_scalar(yf.Ticker("^VIX1D").history(period="5d")['Close'].iloc[-1]) / 100.0
            vix_30d = self.to_scalar(yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]) / 100.0
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

            # Bulletproof Price Fetching
            hist_5d = tk.history(period="5d")
            curr_price = self.to_scalar(hist_5d['Close'].iloc[-1]) if not hist_5d.empty else 0.0
            if curr_price == 0.0: return {"Error": "Failed to fetch current spot price."}

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
            if dte_30_matches:
                dte_30_date = min(dte_30_matches, key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - today).days - 30))
            else:
                dte_30_date = min(dates, key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - today).days - 30))

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
            # Hardcoded to 59d to prevent Yahoo Finance silent failures
            data = yf.download(clean_ticker, period="59d", interval="5m", progress=False)
            if data.empty: return None, None, None
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

            prices = data['Close'].to_numpy()
            
            coeffs = pywt.wavedec(prices, 'db4', level=4)
            cA4 = coeffs[0] 
            cD1 = coeffs[-1]
            
            energy_low = np.convolve(cA4**2, np.ones(10)/10, mode='same')
            energy_high = np.convolve(cD1**2, np.ones(10)/10, mode='same')
            
            energy_high[energy_high == 0] = 1e-10
            ratio = energy_low[:len(prices)] / energy_high[:len(prices)] 
            
            target_len = min(len(prices), len(ratio))
            
            return data.index[-target_len:], prices[-target_len:], ratio[-target_len:]
        except Exception as e:
            print(e)
            return None, None, None
