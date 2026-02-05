import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from logic import StrategyEngine

st.set_page_config(page_title="Waystone/Alpha Engine", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .stDataFrame {border: 1px solid #444;}
    .metric-card {background-color: #0E1117; border: 1px solid #303030; padding: 15px;}
</style>
""", unsafe_allow_html=True)

engine = StrategyEngine()

st.title("⚡ Systematic Alpha Engine")
st.markdown("Multi-Factor Anomaly Detection: **Reflexivity**, **Sentiment**, **Spectral Analysis**, & **Derivatives**.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Sanitizer", "🎙️ CEO Sentiment", "🌊 Spectral Wave", "📉 Options & Volatility"])

# --- TAB 1: PORTFOLIO ---
with tab1:
    st.header("Reflexivity & Liquidity Filter")
    col_input, col_btn = st.columns([3, 1])
    target_fund = col_input.text_input("Target UCITS Fund (Ticker)", value="ICLN")
    
    if col_btn.button("Run Portfolio Scan"):
        with st.spinner(f"Auditing underlying assets of {target_fund}..."):
            df = engine.sanitize_signals(target_fund)
            if not df.empty:
                approved = df[df['Status']=='APPROVED']
                st.metric("Alpha Signals Found", len(approved), f"{len(df)-len(approved)} Rejected")
                
                def color_row(row):
                    color = '#d4edda' if row['Status'] == 'APPROVED' else '#f8d7da'
                    return [f'background-color: {color}; color: black'] * len(row)
                    
                display_cols = ['Ticker', 'Status', 'Reason', 'RSI', 'Return_3M', 'Vol_M']
                st.dataframe(df[display_cols].style.apply(color_row, axis=1), use_container_width=True)
            else:
                st.error("No holdings found.")

# --- TAB 2: SENTIMENT ---
with tab2:
    st.header("Executive Sound & Sentiment")
    ceo_text = st.text_area("Transcript / Statement", height=150, value="We are seeing robust demand.")
    if st.button("Analyze Tone"):
        signal, score = engine.analyze_sound_signal(ceo_text)
        st.metric("Algorithmic Signal", signal, f"Confidence: {score:.2f}")

# --- TAB 3: WAVES (UPDATED) ---
with tab3:
    st.header("Spectral Wave Analysis")
    spec_ticker = st.text_input("Analyze Ticker Waveform", value="NVDA")
    if st.button("Generate Spectrogram"):
        with st.spinner("Decomposing Price Waves (FFT)..."):
            # UNPACK 5 VALUES (Freqs, SpecDates, Intensity, PriceDates, Prices)
            f, spec_dates, Sxx, price_dates, prices = engine.generate_spectrogram_data(spec_ticker)
            
            if Sxx is not None:
                fig = go.Figure()
                
                # 1. Heatmap (Spectrogram) - Uses 'spec_dates' for alignment
                fig.add_trace(go.Heatmap(
                    z=10 * np.log10(Sxx + 1e-10), 
                    x=spec_dates, # ACTUAL DATES
                    y=f, 
                    colorscale='Magma', 
                    colorbar=dict(title='Energy (dB)', x=1.1)
                ))
                
                # 2. Price Line - Uses 'price_dates' (Full History)
                fig.add_trace(go.Scatter(
                    x=price_dates, # ACTUAL DATES
                    y=prices, 
                    mode='lines', 
                    line=dict(color='cyan', width=2), 
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title=f"Spectral Density + Price: {spec_ticker}",
                    xaxis_title="Date",
                    yaxis=dict(title="Frequency", side="left"),
                    yaxis2=dict(title="Price ($)", side="right", overlaying="y", showgrid=False),
                    height=600,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not generate data.")

# --- TAB 4: OPTIONS ---
with tab4:
    st.header("Derivatives & Volatility Insights")
    opt_ticker = st.text_input("Analyze Options Chain", value="TSLA")
    
    if st.button("Analyze Volatility"):
        with st.spinner("Fetching Option Chain..."):
            data = engine.get_options_analytics(opt_ticker)
            
            if data:
                c1, c2, c3 = st.columns(3)
                c1.metric("Put/Call Ratio", f"{data['PCR']:.2f}", delta="Bearish" if data['PCR'] > 1 else "Bullish", delta_color="inverse")
                c2.metric("Implied Vol (IV)", f"{data['IV']:.1%}")
                premium = data['Vol_Premium']
                c3.metric("Writer's Spread (IV - HV)", f"{premium:.1%}", delta="Expensive" if premium > 0.1 else "Cheap", delta_color="normal")
                
                if premium > 0.10: st.success("OPPORTUNITY: Options are Expensive. Consider Writing Volatility.")
                elif premium < -0.05: st.warning("OPPORTUNITY: Options are Cheap. Consider Buying Volatility.")
                else: st.info("Market is Efficiently Priced.")
            else:
                st.error("Could not fetch options data.")
