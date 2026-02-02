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
            # This pulls the 'Top Holdings' table
            # Note: yfinance structure can vary, this targets the 'holdings' attribute
            holdings_data = fund.funds_data.top_holdings
            
            # Convert to our clean format dictionary {Ticker: Percent}
            clean_holdings = {}
            if holdings_data is not None:
                # yfinance returns pandas df usually
                for index, row in holdings_data.iterrows():
                    # Check if index is the ticker or a column
                    ticker_symbol = index 
                    weight = row['Holding Percent']
                    clean_holdings[ticker_symbol] = weight
            
            print(f"   ✅ FOUND {len(clean_holdings)} HOLDINGS.")
            return clean_holdings
            
        except Exception as e:
            print(f"   ❌ ERROR FETCHING HOLDINGS: {e}")
            # Fallback for demo stability if API fails on a specific fund
            return {}

    def analyze_holding_health(self, ticker):
        """
        REAL MARKET DATA:
        Downloads price history to calculate RSI and Run-up.
        """
        try:
            # Download last 3 months of daily data
            data = yf.download(ticker, period="3mo", progress=False)
            if data.empty:
                return 0, 0 # No data
                
            # 1. Calculate % Run Up
            start_price = float(data['Close'].iloc[0])
            end_price = float(data['Close'].iloc[-1])
            run_up_pct = (end_price - start_price) / start_price
            
            # 2. Calculate RSI (14-day)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])
            
            return run_up_pct, current_rsi
            
        except:
            return 0, 0

    def sanitize_signals(self, fund_ticker):
        """
        MASTER LOGIC:
        1. Get Real Holdings
        2. Filter out Blacklist
        3. Filter out High RSI/Run-up
        """
        raw_holdings = self.get_fund_holdings(fund_ticker)
        clean_log = []
        
        if not raw_holdings:
            return pd.DataFrame(columns=['Ticker', 'Weight', 'Status', 'Reason'])

        for ticker, weight in raw_holdings.items():
            status = "PENDING"
            reason = "Checking..."
            
            # A. LIQUIDITY FILTER
            if ticker in self.liquidity_blacklist:
                clean_log.append({'Ticker': ticker, 'Weight': weight, 'Status': 'REJECTED', 'Reason': 'Liquidity Proxy'})
                continue
            
            # B. HEALTH CHECK (Price Feed)
            run_up, rsi = self.analyze_holding_health(ticker)
            
            # THE "STRIP" LOGIC
            # If stock rose > 30% in 3 months OR RSI > 70 (Overbought) -> STRIP IT
            if run_up > 0.30 or rsi > 70:
                clean_log.append({'Ticker': ticker, 'Weight': weight, 'Status': 'REJECTED', 'Reason': f'Overheated (RSI {rsi:.0f})'})
            else:
                clean_log.append({'Ticker': ticker, 'Weight': weight, 'Status': 'APPROVED', 'Reason': 'Healthy Signal'})
            
        return pd.DataFrame(clean_log)
