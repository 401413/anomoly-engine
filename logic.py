import numpy as np
import pandas as pd
import scipy.stats as si
from scipy import signal
import pywt
import yfinance as yf
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class StrategyEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Grab credentials from Streamlit Secrets seamlessly
        self.tiingo_key = st.secrets.get("TIINGO_API_KEY", None)
        self.av_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", None)

    def fetch_validated_history(self, ticker: str, period_years: int = 2) -> pd.DataFrame:
        """
        Fetches daily history. Prioritizes Tiingo/AlphaVantage for data accuracy,
        falling back to a defensive yfinance wrapper if keys aren't present.
        """
        ticker_clean = ticker.split(" ")[0].upper()
        
        # 1. Try Tiingo EOD Feed
        if self.tiingo_key:
            try:
                start_date = (pd.Timestamp.now() - pd.DateOffset(years=period_years)).strftime('%Y-%m-%d')
                url = f"https://api.tiingo.com/tiingo/daily/{ticker_clean}/prices?startDate={start_date}&token={self.tiingo_key}"
                res = requests.get(url, timeout=5).json()
                if isinstance(res, list) and len(res) > 0:
                    df = pd.DataFrame(res)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.rename(columns={'adjClose': 'Close', 'adjVolume': 'Volume'}, inplace=True)
                    return df[['Close', 'Volume']]
            except Exception:
                pass # Fallback if API limit reached or network error occurs

        # 2. Try Alpha Vantage
        if self.av_key:
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker_clean}&outputsize=full&apikey={self.av_key}"
                res = requests.get(url, timeout=5).json()
                if "Time Series (Daily)" in res:
                    raw_data = res["Time Series (Daily)"]
                    df = pd.DataFrame.from_dict(raw_data, orient='index', dtype=float)
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)
                    df.rename(columns={'5. adjusted close': 'Close', '6. volume': 'Volume'}, inplace=True)
                    return df[['Close', 'Volume']].last(f"{period_years}Y")
            except Exception:
                pass

        # 3. Defensive yfinance Fallback
        try:
            df = yf.download(ticker_clean, period=f"{period_years}y", progress=False)
            if df.empty:
                return pd.DataFrame()
            # Drop multi-index structural layers if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            if 'Close' not in df.columns and 'Last' in df.columns:
                df['Close'] = df['Last']
            return df[['Close', 'Volume']].dropna()
        except Exception:
            return pd.DataFrame()

    def sanitize_signals(self, fund_ticker: str) -> pd.DataFrame:
        try:
            fund = yf.Ticker(fund_ticker)
            holdings_data = fund.funds_data.top_holdings
            if holdings_data is None or holdings_data.empty:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

        logs = []
        for index, row in holdings_data.iterrows():
            ticker = str(index).strip()
            if "USD" in ticker or not ticker:
                continue
            
            weight = row.get('Holding Percent', 0.0)
            df_hist = self.fetch_validated_history(ticker, period_years=1)
            
            if df_hist.empty or len(df_hist) < 30:
                continue
                
            prices = df_hist['Close'].values
            start, end = float(prices[0]), float(prices[-1])
            run_up = (end - start) / start if start != 0 else 0.0
            
            # RSI calculation
            delta = pd.Series(prices).diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + (gain / loss))) if loss != 0 else 50.0
            
            vol_m = (df_hist['Volume'].mean() * df_hist['Close'].mean()) / 1_000_000
            
            is_trap = (run_up > 0.20 and vol_m < 20)
            is_hot = (run_up > 0.30 or rsi > 75)
            status, reason = ('REJECTED', 'Reflexivity Trap') if is_trap else ('REJECTED', 'Overheated') if is_hot else ('APPROVED', 'Clean Signal')
            
            logs.append({
                'Ticker': ticker, 'Weight': weight, 'Status': status, 'Reason': reason,
                'RSI': round(float(rsi), 1), 'Return_1Y': f"{run_up:.1%}", 'Vol_M': f"${vol_m:.1f}M"
            })
        return pd.DataFrame(logs)

    def analyze_sound_signal(self, text_input: str) -> tuple[str, float]:
        score = self.analyzer.polarity_scores(text_input)
        compound = score['compound']
        return ("POSITIVE" if compound >= 0.05 else "NEGATIVE" if compound <= -0.05 else "NEUTRAL"), compound

    def generate_spectrogram_data(self, ticker: str) -> tuple:
        df = self.fetch_validated_history(ticker, period_years=2)
        if df.empty or len(df) < 60:
            return None, None, None, None, None, None

        prices = df['Close'].astype(float).values
        dates = df.index
        returns = np.diff(prices)
        returns_dates = dates[1:]
        
        nperseg = min(60, len(returns))
        f, t, Sxx = signal.spectrogram(returns, fs=1.0, window='hann', nperseg=nperseg, noverlap=int(nperseg // 2))
        
        t_indices = np.clip(np.floor(t).astype(int), 0, len(returns_dates) - 1)
        spec_dates = returns_dates[t_indices]
        
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        hist_vol = (log_ret.rolling(window=30).std() * np.sqrt(252)).fillna(0).values
        
        return f, spec_dates, Sxx, dates, prices, hist_vol

    def _calc_bs_gamma(self, S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
        if T <= 0 or sigma <= 0.01 or S <= 0 or K <= 0:
            return 0.0
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        except Exception:
            return 0.0

    def get_options_analytics(self, ticker: str) -> dict:
        ticker_clean = ticker.split(" ")[0].upper()
        tk = yf.Ticker(ticker_clean)
        
        try:
            dates = tk.options
            if not dates:
                return {"Error": "No options chains available."}
        except Exception:
            return {"Error": "Options connection failed."}

        df_hist = self.fetch_validated_history(ticker_clean, period_years=1)
        if df_hist.empty:
            return {"Error": "Failed to resolve spot price."}
            
        curr_price = float(df_hist['Close'].iloc[-1])
        log_ret = np.log(df_hist['Close'] / df_hist['Close'].shift(1))
        hv = float(log_ret.rolling(window=30).std().iloc[-1] * np.sqrt(252))

        # Benchmarks macro indices safely
        vix_30d = 0.18
        try:
            vix_df = yf.download("^VIX", period="5d", progress=False)
            if not vix_df.empty:
                vix_30d = float(vix_df['Close'].iloc[-1]) / 100.0
        except Exception:
            pass

        today = pd.Timestamp.today()
        parsed_dates = []
        for d in dates:
            try:
                days = (pd.to_datetime(d) - today).days
                if days >= 0:
                    parsed_dates.append((d, days))
            except Exception:
                continue

        if not parsed_dates:
            return {"Error": "No clear options expiration windows matched."}

        # Match structural 30 DTE window
        dte_30_date = min(parsed_dates, key=lambda x: abs(x[1] - 30))[0]
        
        # Process structural target chain
        try:
            chain = tk.option_chain(dte_30_date)
            calls, puts = chain.calls.copy(), chain.puts.copy()
        except Exception:
            return {"Error": "Target option chain download failed."}

        T = max(abs((pd.to_datetime(dte_30_date) - today).days), 1.0) / 365.0

        for df_side in [calls, puts]:
            if df_side.empty: continue
            df_side['strike'] = df_side['strike'].astype(float)
            df_side['impliedVolatility'] = df_side['impliedVolatility'].astype(float)
            df_side['openInterest'] = df_side['openInterest'].fillna(0).astype(float)
            df_side['ask'] = df_side['ask'].astype(float)
            df_side['bid'] = df_side['bid'].astype(float)
            
            df_side['Gamma'] = df_side.apply(lambda r: self._calc_bs_gamma(curr_price, r['strike'], T, r['impliedVolatility']), axis=1)
            df_side['GEX'] = df_side['openInterest'] * df_side['Gamma'] * 100 * curr_price
            df_side['Spread_%'] = np.where(df_side['ask'] > 0, (df_side['ask'] - df_side['bid']) / df_side['ask'], 0.0)

        # Crisp, targeted True ATM calculation (2.5% allocation band)
        atm_mask_c = (calls['strike'] >= curr_price * 0.975) & (calls['strike'] <= curr_price * 1.025)
        atm_mask_p = (puts['strike'] >= curr_price * 0.975) & (puts['strike'] <= curr_price * 1.025)
        
        iv_c = calls.loc[atm_mask_c, 'impliedVolatility'].mean() if atm_mask_c.any() else 0.15
        iv_p = puts.loc[atm_mask_p, 'impliedVolatility'].mean() if atm_mask_p.any() else 0.15
        iv_clean = float((iv_c + iv_p) / 2.0)

        return {
            "Current_Price": curr_price, "HV_30D": hv, "Date": dte_30_date, "IV": iv_clean,
            "Macro_VIX": vix_30d, "Vol_Premium": iv_clean - vix_30d,
            "Total_OI": float(calls['openInterest'].sum() + puts['openInterest'].sum()),
            "Calls_DF": calls, "Puts_DF": puts
        }

    def generate_wavelet_energy(self, ticker: str) -> tuple:
        """
        Calculates low/high institutional energy metrics over clean arrays
        using Wavelet Signal Reconstruction to fully maintain length dimensions.
        """
        df = self.fetch_validated_history(ticker, period_years=1)
        if df.empty or len(df) < 32:
            return None, None, None

        prices = df['Close'].astype(float).values
        dates = df.index

        # Execute Discrete Wavelet Transform
        wavelet = 'db4'
        level = 4
        coeffs = pywt.wavedec(prices, wavelet, level=level)

        # Reconstruct isolated low-frequency path (Zero out details)
        low_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        low_signal = pywt.waverec(low_coeffs, wavelet)[:len(prices)]

        # Reconstruct isolated high-frequency noise paths
        high_coeffs = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:-1]] + [coeffs[-1]]
        high_signal = pywt.waverec(high_coeffs, wavelet)[:len(prices)]

        # Apply convolution matching original dimensions N cleanly
        window = 20
        energy_low = np.convolve(low_signal**2, np.ones(window)/window, mode='same')
        energy_high = np.convolve(high_signal**2, np.ones(window)/window, mode='same')
        
        energy_high[energy_high == 0] = 1e-10
        ratio = energy_low / energy_high

        return dates, prices, ratio
