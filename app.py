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
st.markdown("Multi-Factor Anomaly Detection: **Reflexivity**, **Sentiment**, & **Spectral Analysis**.")

tab1, tab2, tab3 = st.tabs(["📊 Portfolio Sanitizer", "🎙️ CEO Sentiment", "🌊 Spectral Wave Analysis"])

# --- TAB 1 ---
with tab1:
    st.header("Reflexivity & Liquidity Filter")
    st.caption("Audits UCITS ETF holdings to reject 'Liquidity Traps' (High Momentum / Low Volume).")
    
    col_input, col_btn = st.columns([3, 1])
    target_fund = col_input.text_input("Target UCITS Fund (Ticker)", value="ICLN")
    
    if col_btn.button("Run Portfolio Scan"):
        with st.spinner(f"Auditing underlying assets of {target_fund}..."):
            df = engine.sanitize_signals(target_fund)
            
            if not df.empty:
                approved = df[df['Status']=='APPROVED']
                st.metric("Alpha Signals Found", len(approved), f"{len(df)-len(approved)} Rejected", delta_color="normal")
                
                def color_row(row):
                    color = '#d4edda' if row['Status'] == 'APPROVED' else '#f8d7da'
                    return [f'background-color: {color}; color: black'] * len(row)

                display_cols = ['Ticker', 'Status', 'Reason', 'RSI', 'Return_3M', 'Vol_M']
                st.dataframe(df[display_cols].style.apply(color_row, axis=1), use_container_width=True)
            else:
                st.error("No holdings found. Try a liquid ETF ticker like ARKK, SMH, or ICLN.")

# --- TAB 2 ---
with tab2:
    st.header("Executive Sound & Sentiment Analysis")
    st.markdown("NLP Engine: Detects hidden confidence or stress signals.")
    
    ceo_text = st.text_area("Transcript / Statement", height=150, 
                            value="We are seeing robust demand and our supply chain is clearing up faster than expected.")
    
    if st.button("Analyze Tone"):
        signal, score = engine.analyze_sound_signal(ceo_text)
        col1, col2 = st.columns(2)
        col1.metric("Algorithmic Signal", signal)
        col2.metric("Confidence Score", f"{score:.2f}")
        
        if signal == "POSITIVE":
            st.success("✅ **BULLISH:** Management language indicates high conviction.")
        elif signal == "NEGATIVE":
            st.error("⚠️ **BEARISH:** Detected hedging/negative language.")

# --- TAB 3 ---
with tab3:
    st.header("Financial Signal Processing (DSP)")
    st.markdown("Visualizing Price History as a **Sound Wave**. We search for 'Resonance'—where hidden frequency energy builds up before a move.")
    
    spec_ticker = st.text_input("Analyze Ticker Waveform", value="NVDA")
    
    if st.button("Generate Spectrogram"):
        with st.spinner("Decomposing Price Waves (FFT)..."):
            f, t, Sxx = engine.generate_spectrogram_data(spec_ticker)
            
            if Sxx is not None:
                fig = go.Figure(data=go.Heatmap(
                    z=10 * np.log10(Sxx + 1e-10),
                    x=t, y=f,
                    colorscale='Magma',
                    colorbar=dict(title='Energy (dB)')
                ))
                
                fig.update_layout(
                    title=f"Spectral Density Map: {spec_ticker}",
                    xaxis_title="Time (Days)",
                    yaxis_title="Frequency (Cycles)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Interpretation:** Look for 'Dark Pockets' at the top (High Freq noise dying down) while the bottom (Low Freq trend) remains bright. This 'Quiet Accumulation' often precedes a breakout.")
            else:
                st.error("Could not generate wave data. Try a liquid stock.")
