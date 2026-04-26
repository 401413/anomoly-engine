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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "🎙️ Sentiment", "🌊 Spectral & Momentum", "📉 Options (Illiquid)"])

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
                # --- CALCULATE MACD & RSI ---
                df_tech = pd.DataFrame({'Close': prices}, index=price_dates)
                
                # 1. Calculate RSI (14-period Wilder's Smoothing)
                delta = df_tech['Close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rs = gain / loss
                df_tech['RSI'] = 100 - (100 / (1 + rs))

                # 2. Calculate MACD (12, 26, 9)
                ema_12 = df_tech['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = df_tech['Close'].ewm(span=26, adjust=False).mean()
                df_tech['MACD'] = ema_12 - ema_26
                df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
                df_tech['MACD_Hist'] = df_tech['MACD'] - df_tech['MACD_Signal']
                
                # 3. Fetch Underlying Trading Volume cleanly
                try:
                    df_vol = yf.download(spec_ticker.split(" ")[0].upper(), period="2y", interval="1d", progress=False)
                    if isinstance(df_vol.columns, pd.MultiIndex): df_vol.columns = df_vol.columns.get_level_values(0)
                    df_tech['Volume'] = df_vol['Volume'].reindex(df_tech.index).fillna(0)
                except:
                    df_tech['Volume'] = 0
                
                # Set Volume Colors (Green for up days, Red for down days)
                df_tech['Prev_Close'] = df_tech['Close'].shift(1)
                colors_vol = ['#2ca02c' if c >= p else '#d62728' for c, p in zip(df_tech['Close'], df_tech['Prev_Close'])]

                # --- BUILD THE 5-ROW SUBPLOT ---
                fig = make_subplots(
                    rows=5, cols=1, 
                    shared_xaxes=True, 
                    row_heights=[0.35, 0.15, 0.15, 0.15, 0.20], 
                    vertical_spacing=0.04,
                    specs=[
                        [{"secondary_y": True}],  # Row 1: Spectrogram & Price
                        [{"secondary_y": False}], # Row 2: Volatility
                        [{"secondary_y": False}], # Row 3: Trading Volume (NEW)
                        [{"secondary_y": False}], # Row 4: MACD
                        [{"secondary_y": False}]  # Row 5: RSI
                    ],
                    subplot_titles=("Spectral Density & Price", "30-Day Realized Volatility (%)", "Daily Trading Volume (Shares)", "MACD (12, 26, 9)", "RSI (14)")
                )
                
                # Row 1: Heatmap
                fig.add_trace(go.Heatmap(z=10*np.log10(Sxx+1e-10), x=spec_dates, y=f, colorscale='Magma', colorbar=dict(x=1.05, title="Power")), row=1, col=1)
                
                # Row 1: Price (Secondary Y)
                fig.add_trace(go.Scatter(x=price_dates, y=prices, line=dict(color='cyan', width=2), name='Price'), row=1, col=1, secondary_y=True)
                
                # --- SIGNALS TEMPORARILY COMMENTED OUT FOR RE-CALIBRATION ---
                # fig.add_trace(go.Scatter(x=buy_strong.index, y=buy_strong['Close'], mode='markers', marker=dict(symbol='triangle-up', size=16, color='#00ff00')), row=1, col=1, secondary_y=True)
                # fig.add_trace(go.Scatter(x=sell_strong.index, y=sell_strong['Close'], mode='markers', marker=dict(symbol='triangle-down', size=16, color='#ff0000')), row=1, col=1, secondary_y=True)

                # Row 2: Historical Volatility
                fig.add_trace(go.Scatter(x=price_dates, y=hist_vol*100, line=dict(color='#ff5e5e'), name='Hist Vol', fill='tozeroy'), row=2, col=1)
                
                # Row 3: Daily Trading Volume (NEW)
                fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['Volume'], marker_color=colors_vol, name='Volume'), row=3, col=1)

                # Row 4: MACD
                colors_macd = ['#2ca02c' if val >= 0 else '#d62728' for val in df_tech['MACD_Hist']]
                fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_Hist'], marker_color=colors_macd, name='MACD Hist'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], line=dict(color='#1f77b4', width=2), name='MACD'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_Signal'], line=dict(color='#ff7f0e', width=1.5), name='Signal'), row=4, col=1)

                # Row 5: RSI
                fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['RSI'], line=dict(color='#9467bd', width=2), name='RSI'), row=5, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=5, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=5, col=1)
                
                # Formatting and Layout Update
                fig.update_layout(
                    height=1200, 
                    showlegend=False, 
                    template="plotly_dark", 
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                # Lock Axis Titles and Limits
                fig.update_yaxes(title_text="Frequency", row=1, col=1, secondary_y=False)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text="Vol (%)", row=2, col=1)
                fig.update_yaxes(title_text="Shares", row=3, col=1)
                fig.update_yaxes(title_text="MACD", row=4, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=5, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Could not generate data for {spec_ticker}. History too short or data error.")

# --- TAB 4: OPTIONS (ILLIQUID FOCUS) ---
with tab4:
    st.header("Illiquid Options Analyzer")
    st.markdown("This tool compares **Implied Volatility (IV)** vs. **Realized Volatility (HV)** to find expensive premiums, alongside active volume positioning.")
    
    opt_ticker = st.text_input("Options Ticker", value="IREN")
    
    if st.button("Analyze Chain"):
        with st.spinner("Fetching Raw Chain & Volume..."):
            data = engine.get_options_analytics(opt_ticker)
            
            if data and "Error" not in data:
                # 1. PCR & Volatility Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Put/Call Ratio (PCR)", f"{data['PCR']:.2f}", help="> 1 means more Puts are trading (Bearish). < 1 means more Calls (Bullish).")
                c2.metric("Implied Vol (Fear)", f"{data['IV']:.1%}")
                c3.metric("Realized Vol (Reality)", f"{data['HV']:.1%}")
                
                premium = data['Vol_Premium']
                delta_color = "normal" if premium > 0 else "inverse"
                c4.metric("Writer's Edge (IV - HV)", f"{premium:.1%}", delta="Expensive" if premium > 0.1 else "Cheap", delta_color=delta_color)
                
                st.divider()
                
                # 2. Extract active Put/Call volume straight from the chain
                tk = yf.Ticker(opt_ticker.split(" ")[0].upper())
                dates = tk.options
                if dates:
                    chain = tk.option_chain(dates[0])
                    calls, puts = chain.calls, chain.puts
                    curr_price = data['Current_Price']
                    
                    # Filter for Strikes within +/- 20% of Current Price to keep chart clean
                    calls = calls[(calls['strike'] >= curr_price * 0.8) & (calls['strike'] <= curr_price * 1.2)]
                    puts = puts[(puts['strike'] >= curr_price * 0.8) & (puts['strike'] <= curr_price * 1.2)]

                    # Visualizing Call vs Put Volume by Strike
                    fig_opt_vol = go.Figure()
                    fig_opt_vol.add_trace(go.Bar(name='Call Volume (Bullish)', x=calls['strike'], y=calls['volume'], marker_color='#00ff00'))
                    fig_opt_vol.add_trace(go.Bar(name='Put Volume (Bearish)', x=puts['strike'], y=puts['volume'], marker_color='#ff0000'))
                    
                    # Add current price line
                    fig_opt_vol.add_vline(x=curr_price, line_dash="dash", line_color="white", annotation_text="Current Price")
                    fig_opt_vol.update_layout(barmode='group', title=f"Options Volume by Strike (Nearest Expiry: {dates[0]})", height=400, template="plotly_dark", xaxis_title="Strike Price", yaxis_title="Volume")
                    st.plotly_chart(fig_opt_vol, use_container_width=True)

                st.divider()

                # 3. Visualizing the Volatility Spread
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(name='Implied Vol (Fear)', x=['Volatility'], y=[data['IV']], marker_color='#ff5e5e')) # Red
                fig_vol.add_trace(go.Bar(name='Realized Vol (Reality)', x=['Volatility'], y=[data['HV']], marker_color='cyan')) # Cyan
                fig_vol.update_layout(barmode='group', title="Fear vs. Reality: Are Options Overpriced?", height=300, template="plotly_dark")
                st.plotly_chart(fig_vol, use_container_width=True)
                
                if premium > 0.20:
                    st.success(f"🔥 **SUPER PREMIUM:** Options are trading 20%+ higher than actual volatility. This is a classic 'Illiquid Writer' setup.")
                elif premium > 0.10:
                    st.info(f"✅ **EXPENSIVE:** Good conditions for writing calls/puts.")
                else:
                    st.warning("⚠️ **CHEAP/FAIR:** Premiums are low. Writing here is risky.")
                    
            elif data and "Error" in data:
                st.error(f"Analysis Failed: {data['Error']}")
                st.caption("Note: For micro-caps, if no one has traded an option in days, Yahoo Finance may return an empty chain.")
