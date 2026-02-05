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
                logs.append({'Ticker': ticker, 'Weight': weight, 'Status': 'REJECTED', 'Reason': 'Beta Proxy', 'RSI': 0, 'Return_3M': 0, 'Vol_M': 0})
                continue

            run_up, rsi, vol_m = self.analyze_holding_health(ticker)
            is_trap = (run_up > 0.20) and (vol_m < 20) 
            is_hot = (run_up > 0.30) or (rsi > 75)
            
            status, reason = ('APPROVED', 'Clean Signal')
            if is_trap: status, reason = ('REJECTED', 'Reflexivity Trap')
            elif is_hot: status, reason = ('REJECTED', 'Overheated')

            logs.append({'Ticker': ticker, 'Weight': weight, 'Status': status, 'Reason': reason, 'RSI': round(rsi, 1), 'Return_3M': f"{run_up:.1%}", 'Vol_M': f"${vol_m:.1f}M"})
            
        return pd.DataFrame(logs)

    # --- MODULE 2: TEXT ---
    def analyze_sound_signal(self, text_input):
        score = self.analyzer.polarity_scores(text_input)
        compound = score['compound']
        if compound >= 0.05: return "POSITIVE", compound
        elif compound <= -0.05: return "NEGATIVE", compound
        else: return "NEUTRAL", compound

    # --- MODULE 3: WAVES (UPDATED FOR DATES) ---
    def generate_spectrogram_data(self, ticker):
        """
        Returns: Freqs (f), SpecDates (t_dates), Intensities (Sxx), PriceDates (p_dates), Prices
        """
        try:
            # 1. Fetch Data with Dates
            data = yf.download(ticker.split(" ")[0], period="2y", interval="1d", progress=False)
            if data.empty: return None, None, None, None, None

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Extract Prices and the specific Date Index
            prices = self.to_scalar_array(data['Close'])
            price_dates = data.index # This is the X-axis for the price line
            
            # 2. Detrend (Returns)
            returns = np.diff(prices)
            # Returns array is 1 shorter than prices, so we adjust date reference
            returns_dates = price_dates[1:] 
            
            # 3. Spectrogram Calculation
            # fs=1.0 means 1 sample per day
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, window='hann', nperseg=60, noverlap=50)
            
            # 4. Map Spectrogram 't' (indices) to Actual Dates
            # 't' returns the center index of the window. We round to nearest int to get the date.
            t_indices = np.floor(t).astype(int)
            
            # Safety clip to ensure we don't exceed the date array
            t_indices = np.clip(t_indices, 0, len(returns_dates) - 1)
            
            spec_dates = returns_dates[t_indices] # Map indices to Datetime objects
            
            return f, spec_dates, Sxx, price_dates, prices
            
        except Exception as e:
            print(f"Spectrogram Error: {e}")
            return None, None, None, None, None

    # --- MODULE 4: OPTIONS ---
    def get_options_analytics(self, ticker):
        try:
            tk = yf.Ticker(ticker.split(" ")[0])
            dates = tk.options
            if not dates: return None
            
            chain = tk.option_chain(dates[0])
            calls, puts = chain.calls, chain.puts
            
            total_call_vol = calls['volume'].sum() if not calls.empty else 1
            total_put_vol = puts['volume'].sum() if not puts.empty else 0
            pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 0
            
            current_price = self.to_scalar(tk.fast_info['lastPrice'])
            atm_calls = calls[ (calls['strike'] > current_price * 0.95) & (calls['strike'] < current_price * 1.05) ]
            avg_iv = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else 0
            
            hist_data = tk.history(period="1mo")
            if not hist_data.empty:
                returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
                hist_vol = returns.std() * np.sqrt(252)
            else: hist_vol = 0
                
            return {
                "PCR": pcr, "IV": avg_iv, "HV": hist_vol, 
                "Vol_Premium": avg_iv - hist_vol, "Nearest_Exp": dates[0]
            }
        except: return None
