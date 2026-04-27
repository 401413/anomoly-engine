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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio", "🎙️ Sentiment", "🌊 Spectral", "📉 Options", "📐 Wavelet Regime"])

# ... [Keep Tab 1, 2, and 3 exactly as they are] ...

# --- TAB 4: OPTIONS (DURATION FOCUS) ---
with tab4:
    st.header("Duration & Variance Arbitrage")
    st.markdown("Isolates Implied Volatility (IV) against Beta-Adjusted macro benchmarks.")
    
    opt_ticker = st.text_input("Options Ticker", value="SPY")
    
    if st.button("Analyze Options Structure"):
        with st.spinner("Fetching Term Structure & Liquidity..."):
            data = engine.get_options_analytics(opt_ticker)
            
            if data and "Error" not in data:
                st.subheader(f"Underlying Profile: {opt_ticker.upper()} @ ${data['Current_Price']:.2f}")
                st.caption(f"30-Day Realized Volatility (HV): **{data['HV_30D']:.1%}** | Asset Beta vs SPY: **{data['Beta']:.2f}**")
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
                    
                    st.caption(f"Volume Breakdown - Calls: {window_data['Call_Vol']:,.0f} | Puts: {window_data['Put_Vol']:,.0f}")

                    curr_price = data['Current_Price']
                    calls, puts = window_data['Calls_DF'], window_data['Puts_DF']
                    calls = calls[(calls['strike'] >= curr_price * 0.9) & (calls['strike'] <= curr_price * 1.1)]
                    puts = puts[(puts['strike'] >= curr_price * 0.9) & (puts['strike'] <= curr_price * 1.1)]

                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Bar(name='Call Volume', x=calls['strike'], y=calls['volume'], marker_color='rgba(0, 255, 0, 0.6)'), secondary_y=False)
                    fig.add_trace(go.Bar(name='Put Volume', x=puts['strike'], y=puts['volume'], marker_color='rgba(255, 0, 0, 0.6)'), secondary_y=False)
                    fig.add_trace(go.Scatter(name='Call OI', x=calls['strike'], y=calls['openInterest'], mode='lines+markers', line=dict(color='#00ff00', width=2)), secondary_y=True)
                    fig.add_trace(go.Scatter(name='Put OI', x=puts['strike'], y=puts['openInterest'], mode='lines+markers', line=dict(color='#ff0000', width=2)), secondary_y=True)

                    fig.add_vline(x=curr_price, line_dash="dash", line_color="white", annotation_text="Spot")
                    fig.update_layout(barmode='group', height=350, template="plotly_dark", xaxis_title="Strike", yaxis_title="Volume")
                    st.plotly_chart(fig, use_container_width=True)

                render_duration_block("Structural Window (27-37 DTE)", data['30DTE'], is_0dte=False)
                st.divider()
                render_duration_block("Noise Window (0-1 DTE)", data['0DTE'], is_0dte=True)

            elif data and "Error" in data:
                st.error(f"Analysis Failed: {data['Error']}")

# --- TAB 5: WAVELET REGIME ---
with tab5:
    st.header("Wavelet-Based Regime Detection")
    st.markdown("Separates low-frequency institutional accumulation from high-frequency retail noise using DWT Energy Ratios.")
    wav_ticker = st.text_input("Asset Ticker", value="BTC-USD")
    
    if st.button("Calculate DWT Energy"):
        with st.spinner("Processing 5-min intervals..."):
            dates, prices, energy_ratio = engine.generate_wavelet_energy(wav_ticker)
            if dates is not None:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.05)
                fig.add_trace(go.Scatter(x=dates, y=prices, name='Price', line=dict(color='cyan')), row=1, col=1)
                fig.add_trace(go.Scatter(x=dates, y=energy_ratio, name='Low/High Energy Ratio', line=dict(color='orange')), row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", title="Institutional Signal (Low-Freq Energy Ratio)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to retrieve 5-min intraday data. Ticker may be invalid or data unavailable.")
