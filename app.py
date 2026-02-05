import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from logic import StrategyEngine
from plotly.subplots import make_subplots

st.set_page_config(page_title="Waystone/Alpha Engine", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .stDataFrame {border: 1px solid #444;}
    .metric-card {background-color: #0E1117; border: 1px solid #303030; padding: 15px;}
</style>
""", unsafe_allow_html=True)

engine = StrategyEngine()

st.title("⚡ Systematic Alpha Engine")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "🎙️ Sentiment", "🌊 Spectral & History", "📉 Options"])

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

# --- TAB 3: SPECTRAL ---
with tab3:
    st.header("Spectral Density")
    spec_ticker = st.text_input("Ticker Symbol", value="NVDA")
    if st.button("Generate Wave"):
        with st.spinner("Calculating..."):
            f, spec_dates, Sxx, price_dates, prices, hist_vol = engine.generate_spectrogram_data(spec_ticker)
            
            if Sxx is not None:
                # FIX: Explicit subplot definitions to prevent "Red Wall"
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    row_heights=[0.7, 0.3], 
                    vertical_spacing=0.05,
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]] # Row 1 has 2 Y-axes, Row 2 has 1
                )
                
                # 1. Heatmap (Row 1)
                fig.add_trace(go.Heatmap(z=10*np.log10(Sxx+1e-10), x=spec_dates, y=f, colorscale='Magma', colorbar=dict(x=1.05)), row=1, col=1)
                
                # 2. Price (Row 1, Secondary Y)
                fig.add_trace(go.Scatter(x=price_dates, y=prices, line=dict(color='cyan', width=2), name='Price'), row=1, col=1, secondary_y=True)
                
                # 3. Volatility (Row 2) - STRICTLY SEPARATE
                fig.add_trace(go.Scatter(x=price_dates, y=hist_vol*100, line=dict(color='#ff5e5e'), name='Hist Vol', fill='tozeroy'), row=2, col=1)
                
                # Layout
                fig.update_layout(height=700, showlegend=False)
                fig.update_yaxes(title_text="Frequency", row=1, col=1, secondary_y=False)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text="Vol (%)", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Could not generate data for {spec_ticker}. History too short.")

# --- TAB 4: OPTIONS ---
with tab4:
    st.header("Options Analysis")
    opt_ticker = st.text_input("Options Ticker", value="TSLA")
    if st.button("Analyze Chain"):
        data = engine.get_options_analytics(opt_ticker)
        
        if data:
            c1, c2, c3 = st.columns(3)
            c1.metric("P/C Ratio", f"{data['PCR']:.2f}")
            c2.metric("Implied Vol", f"{data['IV']:.1%}")
            c3.metric("Premium", f"{data['Vol_Premium']:.1%}")
        else:
            st.warning(f"No liquid options found for {opt_ticker}. (This is common for small caps or crypto miners like WULF)")
