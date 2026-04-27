import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from logic import StrategyEngine
from plotly.subplots import make_subplots

st.set_page_config(page_title="Alpha Engine", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .stDataFrame {border: 1px solid #444;}
    .metric-card {background-color: #0E1117; border: 1px solid #303030; padding: 15px;}
</style>
""", unsafe_allow_html=True)

engine = StrategyEngine()

st.title("⚡ Systematic Alpha Engine")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio", "🎙️ Sentiment", "🌊 Spectral & Momentum", "📉 Options", "📐 Wavelet Regime"])

# --- TAB 1: PORTFOLIO ---
with tab1:
    st.header("Reflexivity Filter")
    target_fund = st.text_input("UCITS Ticker", value="ICLN")
    if st.button("Run Scan"):
        with st.spinner("Scanning..."):
            df = engine.sanitize_signals(target_fund)
            if not df.empty:
                st.dataframe(df)
            else: st.error("No holdings found.")

# --- TAB 2: SENTIMENT ---
with tab2:
    st.header("Sentiment Analysis")
    txt = st.text_area("Text", value="Growth is strong.")
    if st.button("Analyze"):
        sig, score = engine.analyze_sound_signal(txt)
        st.metric("Signal", sig, f"{score:.2f}")

# --- TAB 3: SPECTRAL & HISTORY ---
with tab3:
    st.header("Spectral Density & Momentum Signals")
    spec_ticker = st.text_input("Ticker Symbol", value="NVDA")
    
    if st.button("Generate Wave"):
        with st.spinner("Calculating..."):
            f, spec_dates, Sxx, price_dates, prices, hist_vol = engine.generate_spectrogram_data(spec_ticker)
            
            if Sxx is not None:
                df_tech = pd.DataFrame({'Close': prices}, index=price_dates)
                
                delta = df_tech['Close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rs = gain / loss
                df_tech['RSI'] = 100 - (100 / (1 + rs))

                ema_12 = df_tech['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = df_tech['Close'].ewm(span=26, adjust=False).mean()
                df_tech['MACD'] = ema_12 - ema_26
                df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
                df_tech['MACD_Hist'] = df_tech['MACD'] - df_tech['MACD_Signal']
                
                try:
                    df_vol = yf.download(spec_ticker.split(" ")[0].upper(), period="2y", interval="1d", progress=False)
                    if isinstance(df_vol.columns, pd.MultiIndex): df_vol.columns = df_vol.columns.get_level_values(0)
                    df_tech['Volume'] = df_vol['Volume'].reindex(df_tech.index).fillna(0)
                except:
                    df_tech['Volume'] = 0
                
                df_tech['Prev_Close'] = df_tech['Close'].shift(1)
                colors_vol = ['#2ca02c' if c >= p else '#d62728' for c, p in zip(df_tech['Close'], df_tech['Prev_Close'])]

                fig = make_subplots(
                    rows=5, cols=1, 
                    shared_xaxes=True, 
                    row_heights=[0.35, 0.15, 0.15, 0.15, 0.20], 
                    vertical_spacing=0.04,
                    specs=[
                        [{"secondary_y": True}],
                        [{"secondary_y": False}],
                        [{"secondary_y": False}],
                        [{"secondary_y": False}],
                        [{"secondary_y": False}]
                    ],
                    subplot_titles=("Spectral Density & Price", "30-Day Realized Volatility (%)", "Daily Trading Volume (Shares)", "MACD (12, 26, 9)", "RSI (14)")
                )
                
                fig.add_trace(go.Heatmap(z=10*np.log10(Sxx+1e-10), x=spec_dates, y=f, colorscale='Magma', colorbar=dict(x=1.05, title="Power")), row=1, col=1)
                fig.add_trace(go.Scatter(x=price_dates, y=prices, line=dict(color='cyan', width=2), name='Price'), row=1, col=1, secondary_y=True)
                fig.add_trace(go.Scatter(x=price_dates, y=hist_vol*100, line=dict(color='#ff5e5e'), name='Hist Vol', fill='tozeroy'), row=2, col=1)
                fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['Volume'], marker_color=colors_vol, name='Volume'), row=3, col=1)

                colors_macd = ['#2ca02c' if val >= 0 else '#d62728' for val in df_tech['MACD_Hist']]
                fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_Hist'], marker_color=colors_macd, name='MACD Hist'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], line=dict(color='#1f77b4', width=2), name='MACD'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_Signal'], line=dict(color='#ff7f0e', width=1.5), name='Signal'), row=4, col=1)

                fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['RSI'], line=dict(color='#9467bd', width=2), name='RSI'), row=5, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=5, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=5, col=1)
                
                fig.update_layout(height=1200, showlegend=False, template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Could not generate data for {spec_ticker}. History too short or data error.")

# --- TAB 4: OPTIONS (DURATION & GEX FOCUS) ---
with tab4:
    st.header("Duration & Gamma Arbitrage")
    st.markdown("Isolates Implied Volatility (IV) and exposes Dealer Gamma (GEX) walls to identify 'Pinning' targets.")
    
    opt_ticker = st.text_input("Options Ticker", value="SPY")
    
    if st.button("Analyze Options Structure"):
        with st.spinner("Fetching Term Structure & Calculating GEX..."):
            data = engine.get_options_analytics(opt_ticker)
            
            if data and "Error" not in data:
                curr_price = data.get('Current_Price', 0.0)
                hv = data.get('HV_30D', 0.0)
                beta = data.get('Beta', 1.0)
                
                st.subheader(f"Underlying Profile: {opt_ticker.upper()} @ ${curr_price:.2f}")
                st.caption(f"30-Day Realized Volatility (HV): **{hv:.1%}** | Asset Beta vs SPY: **{beta:.2f}**")
                st.divider()

                def render_duration_block(title, window_data, is_0dte=False):
                    if not window_data:
                        st.warning(f"No active options data found for this timeframe.")
                        return
                        
                    st.markdown(f"### {title} (Expiry: {window_data['Date']})")
                    
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Put/Call Ratio", f"{window_data['PCR']:.2f}")
                    c2.metric(f"Implied Vol (IV)", f"{window_data['IV']:.1%}")
                    
                    vix_label = "Beta-Adj VIX1D" if is_0dte else "Beta-Adj VIX"
                    c3.metric(vix_label, f"{window_data['Beta_Adj_VIX']:.1%}")
                    
                    premium = window_data['Vol_Premium']
                    delta_color = "normal" if premium > 0 else "inverse"
                    c4.metric("Variance Spread", f"{premium:.1%}", delta="Expensive" if premium > 0.05 else "Discounted", delta_color=delta_color)
                    
                    c5.metric("Total Volume", f"{window_data['Total_Volume']:,.0f}", f"OI: {window_data['Total_OI']:,.0f}")

                    # Filter for Strike Range (+/- 15%)
                    calls, puts = window_data['Calls_DF'], window_data['Puts_DF']
                    calls = calls[(calls['strike'] >= curr_price * 0.85) & (calls['strike'] <= curr_price * 1.15)]
                    puts = puts[(puts['strike'] >= curr_price * 0.85) & (puts['strike'] <= curr_price * 1.15)]

                    # GEX Visualization
                    st.markdown("#### Net Dealer Gamma Exposure (GEX)")
                    fig_gex = go.Figure()
                    fig_gex.add_trace(go.Bar(name='Call GEX (Positive Gamma)', x=calls['strike'], y=calls['GEX'], marker_color='rgba(0, 255, 0, 0.7)'))
                    fig_gex.add_trace(go.Bar(name='Put GEX (Negative Gamma)', x=puts['strike'], y=puts['GEX'], marker_color='rgba(255, 0, 0, 0.7)'))
                    fig_gex.add_vline(x=curr_price, line_dash="dash", line_color="white", annotation_text="Spot")
                    fig_gex.update_layout(barmode='relative', height=350, template="plotly_dark", xaxis_title="Strike", yaxis_title="Net GEX (Shares)")
                    st.plotly_chart(fig_gex, use_container_width=True)

                    # Pricing and Slippage Analysis
                    st.markdown("#### Execution Feasibility (Bid/Ask Spread Trap)")
                    st.caption("Average slippage metrics for Near-The-Money (NTM) strikes. Wide spreads destroy vertical spread mechanics.")
                    
                    ntm_calls = calls[(calls['strike'] >= curr_price * 0.95) & (calls['strike'] <= curr_price * 1.05)]
                    if not ntm_calls.empty:
                        avg_call_spread = ntm_calls['Spread_%'].mean()
                        avg_put_spread = puts[(puts['strike'] >= curr_price * 0.95) & (puts['strike'] <= curr_price * 1.05)]['Spread_%'].mean()
                        
                        col_a, col_b = st.columns(2)
                        col_a.metric("Avg Call Slippage (NTM)", f"{avg_call_spread:.1%}", delta="Illiquid" if avg_call_spread > 0.10 else "Liquid", delta_color="inverse")
                        col_b.metric("Avg Put Slippage (NTM)", f"{avg_put_spread:.1%}", delta="Illiquid" if avg_put_spread > 0.10 else "Liquid", delta_color="inverse")

                render_duration_block("Structural Window (27-37 DTE)", data.get('30DTE'), is_0dte=False)
                st.divider()
                render_duration_block("Noise Window (0-1 DTE)", data.get('0DTE'), is_0dte=True)

            elif data and "Error" in data:
                st.error(f"Analysis Failed: {data['Error']}")

# --- MODULE 5: WAVELET REGIME ---
    def generate_wavelet_energy(self, ticker):
        try:
            clean_ticker = ticker.split(" ")[0].upper()
            
            # UPGRADE: Shifted from fragile 5m data to highly stable 1h data over 1 year.
            # This perfectly captures institutional TWAP execution without hitting API walls.
            data = yf.download(clean_ticker, period="1y", interval="1h", progress=False)
            
            if data.empty: return None, None, None
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

            prices = data['Close'].to_numpy()
            
            # Perform Discrete Wavelet Transform
            coeffs = pywt.wavedec(prices, 'db4', level=4)
            
            # Extract Low Frequency (Institutional Trend) and High Frequency (Retail/HFT Noise)
            cA4 = coeffs[0] 
            cD1 = coeffs[-1]
            
            # Calculate rolling energy (expanded window for hourly data)
            energy_low = np.convolve(cA4**2, np.ones(20)/20, mode='same')
            energy_high = np.convolve(cD1**2, np.ones(20)/20, mode='same')
            
            energy_high[energy_high == 0] = 1e-10
            ratio = energy_low[:len(prices)] / energy_high[:len(prices)] 
            
            target_len = min(len(prices), len(ratio))
            
            return data.index[-target_len:], prices[-target_len:], ratio[-target_len:]
        except Exception as e:
            print(f"Wavelet Error: {e}")
            return None, None, None
