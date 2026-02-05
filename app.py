import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from logic import StrategyEngine
from plotly.subplots import make_subplots # Needed for stacked charts

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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Sanitizer", "🎙️ CEO Sentiment", "🌊 Spectral & Vol History", "📉 Options Chain"])

# --- TAB 1: PORTFOLIO ---
with tab1:
    st.header("Reflexivity & Liquidity Filter")
    col_input, col_btn = st.columns([3, 1])
    target_fund = col_input.text_input("Target UCITS Fund (Ticker)", value="ICLN")
    if col_btn.button("Run Portfolio Scan"):
        with st.spinner(f"Auditing {target_fund}..."):
            df = engine.sanitize_signals(target_fund)
            if not df.empty:
                approved = df[df['Status']=='APPROVED']
                st.metric("Alpha Signals Found", len(approved), f"{len(df)-len(approved)} Rejected")
                st.dataframe(df[['Ticker', 'Status', 'Reason', 'RSI', 'Return_3M', 'Vol_M']].style.apply(lambda x: [f'background-color: {"#d4edda" if x["Status"] == "APPROVED" else "#f8d7da"}; color: black']*6, axis=1), use_container_width=True)
            else: st.error("No holdings found.")

# --- TAB 2: SENTIMENT ---
with tab2:
    st.header("Executive Sound & Sentiment")
    ceo_text = st.text_area("Transcript / Statement", height=150, value="We are seeing robust demand.")
    if st.button("Analyze Tone"):
        signal, score = engine.analyze_sound_signal(ceo_text)
        st.metric("Algorithmic Signal", signal, f"Confidence: {score:.2f}")

# --- TAB 3: SPECTRAL & HISTORY (UPDATED) ---
with tab3:
    st.header("Spectral Wave & Historical Volatility")
    st.caption("Top: The 'Sound' of Price (Spectrogram). Bottom: The History of Realized Volatility.")
    
    spec_ticker = st.text_input("Analyze Ticker", value="NVDA")
    
    if st.button("Generate History"):
        with st.spinner("Calculating Spectral Density & Volatility Surface..."):
            # UNPACK 6 VALUES
            f, spec_dates, Sxx, price_dates, prices, hist_vol = engine.generate_spectrogram_data(spec_ticker)
            
            if Sxx is not None:
                # Create Stacked Subplots (Shared X-Axis)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    row_heights=[0.7, 0.3],
                                    subplot_titles=("Spectral Density + Price", "Historical Realized Volatility (30D)"))

                # 1. Heatmap (Top)
                fig.add_trace(go.Heatmap(
                    z=10 * np.log10(Sxx + 1e-10), 
                    x=spec_dates, y=f, 
                    colorscale='Magma', 
                    colorbar=dict(title='Energy (dB)', x=1.02, y=0.8, len=0.7)
                ), row=1, col=1)

                # 2. Price Line Overlay (Top)
                fig.add_trace(go.Scatter(
                    x=price_dates, y=prices, 
                    mode='lines', line=dict(color='cyan', width=2), 
                    name='Price'
                ), row=1, col=1)

                # 3. Volatility History (Bottom)
                fig.add_trace(go.Scatter(
                    x=price_dates, y=hist_vol * 100, # Convert to %
                    mode='lines', 
                    line=dict(color='#ff5e5e', width=2),
                    name='Realized Vol (30D)',
                    fill='tozeroy', # Fill area to look like a mountain
                    fillcolor='rgba(255, 94, 94, 0.2)'
                ), row=2, col=1)

                # Layout Updates
                fig.update_layout(height=800, showlegend=False, xaxis2_title="Date")
                
                # Assign Price to Secondary Y-Axis on Top Plot
                fig.update_traces(yaxis='y2', selector=dict(name='Price'))
                fig.layout.yaxis2 = dict(overlaying='y', side='right', showgrid=False, title='Price ($)')
                fig.layout.yaxis3.title = 'Volatility (%)' # Bottom chart Y-axis

                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Singularity Check:** Look for dates where the **Spectrogram is Black** (Top) AND the **Red Volatility Line is at Lows** (Bottom). This confirms a 'Variance Bomb' setup.")
            else:
                st.error("Could not generate data.")

# --- TAB 4: OPTIONS SNAPSHOT ---
with tab4:
    st.header("Current Options Chain Analysis")
    opt_ticker = st.text_input("Analyze Options", value="TSLA")
    if st.button("Analyze Chain"):
        data = engine.get_options_analytics(opt_ticker)
        if data:
            c1, c2, c3 = st.columns(3)
            c1.metric("Put/Call Ratio", f"{data['PCR']:.2f}")
            c2.metric("Implied Vol (IV)", f"{data['IV']:.1%}")
            c3.metric("Writer's Spread", f"{data['Vol_Premium']:.1%}")
        else: st.error("No options data.")
