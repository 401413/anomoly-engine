import yfinance as yf
import pandas as pd
import numpy as np

class StrategyEngine:
    def __init__(self):
        # Assets we NEVER want to own because they are just for liquidity/indexing
        self.liquidity_blacklist = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'GOOGL', 'BIL', 'SHV', 'CASH_USD']

    def get_fund_holdings(self, fund_ticker):
        """
        REAL DATA FETCH:
        Pulls the Top 10 holdings of a UCITS ETF/Fund from Yahoo Finance.
        Returns a dictionary: {'TICKER': Weight}
        """
        print(f"📡 CONNECTING TO LIVE FEED: Fetching holdings for {fund_ticker}...")
        try:
            fund = yf.Ticker(fund_ticker)
            # Fetch holdings (structure varies by ETF provider, using yfinance generic)
            holdings_data = fund.funds_data.top_holdings
            
            clean_holdings = {}
            if holdings_data is not None:
                # yfinance returns a pandas df; we iterate to build our dict
                for index, row in holdings_data.iterrows():
                    ticker_symbol = index 
                    weight = row['Holding Percent']
                    clean_holdings[ticker_symbol] = weight
            
            print(f"   ✅ FOUND {len(clean_holdings)} HOLDINGS.")
            return clean_holdings
            
        except Exception as e:
            print(f"   ❌ ERROR FETCHING HOLDINGS: {e}")
            return {}

    def analyze_holding_health(self, ticker):
        """
        Analyzes Price, RSI, AND LIQUIDITY (Volume).
        """
        try:
            # Get Ticker Info for Market Cap (check if it's a micro-cap trap)
            # Note: Fetching .info can be slow; for a fast demo, we rely on volume.
            
            # Get History for Price/Vol
            data = yf.download(ticker, period="3mo", progress=False)
            if data.empty: return 0, 0, 0

            # 1. Price Momentum (3-month run-up)
            start_price = float(data['Close'].iloc[0])
            end_price = float(data['Close'].iloc[-1])
            run_up_pct = (end_price - start_price) / start_price
            
            # 2. RSI Calculation (14-day)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])
            
            # 3. Liquidity Check (Avg Daily Dollar Volume)
            avg_vol = data['Volume'].mean()
            avg_price = data['Close'].mean()
            dollar_volume = avg_vol * avg_price # e.g., $50,000,000 traded per day
            
            return run_up_pct, current_rsi, dollar_volume

        except:
            return 0, 0, 0

    def sanitize_signals(self, fund_ticker):
        """
        MASTER LOGIC:
        1. Get Real Holdings
        2. Filter out Blacklist (Mega Caps)
        3. Filter out Reflexivity Traps (High Price / Low Vol)
        """
        raw_holdings = self.get_fund_holdings(fund_ticker)
        clean_log = []
        
        if not raw_holdings:
            return pd.DataFrame()

        for ticker, weight in raw_holdings.items():
            
            # A. GENERIC BLACKLIST (Stripping "Beta")
            if ticker in self.liquidity_blacklist:
                clean_log.append({'Ticker': ticker, 'Weight': weight, 'Status': 'REJECTED', 'Reason': 'Index/Beta Proxy'})
                continue
            
            # B. DEEP DIVE (Price + Liquidity)
            run_up, rsi, dollar_vol = self.analyze_holding_health(ticker)
            
            # --- THE "REFLEXIVITY TRAP" FILTER ---
            # TRAP: Stock is up >20%, but trades less than $20M a day.
            # Rationale: The fund's own buying likely inflated the price.
            is_reflexivity_trap = (run_up > 0.20) and (dollar_vol < 20_000_000)
            
            if is_reflexivity_trap:
                 clean_log.append({'Ticker': ticker, 'Weight': weight, 'Status': 'REJECTED', 'Reason': '⚠️ Reflexivity Trap (Low Liquidity Run-up)'})
            
            elif run_up > 0.30 or rsi > 75:
                 clean_log.append({'Ticker': ticker, 'Weight': weight, 'Status': 'REJECTED', 'Reason': 'Overheated (Mean Reversion Risk)'})
                 
            else:
                clean_log.append({'Ticker': ticker, 'Weight': weight, 'Status': 'APPROVED', 'Reason': 'Clean Alpha Signal'})
            
        return pd.DataFrame(clean_log)
