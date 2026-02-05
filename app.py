import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "🎙️ Sentiment", "🌊 Spectral & History", "📉 Options (Illiquid)"])

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
                # Explicit subplot definitions
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    row_heights=[0.7, 0.3], 
                    vertical_spacing=0.05,
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
                )
                
                # 1. Heatmap
                fig.add_trace(go.Heatmap(z=10*np.log10(Sxx+1e-10), x=spec_dates, y=f, colorscale='Magma', colorbar=dict(x=1.05)), row=1, col=1)
                
                # 2. Price (Secondary Y)
                fig.add_trace(go.Scatter(x=price_dates, y=prices, line=dict(color='cyan', width=2), name='Price'), row=1, col=1, secondary_y=True)
                
                # 3. Volatility (Row 2 - Separate)
                fig.add_trace(go.Scatter(x=price_dates, y=hist_vol*100, line=dict(color='#ff5e5e'), name='Hist Vol (30D)', fill='tozeroy'), row=2, col=1)
                
                fig.update_layout(height=700, showlegend=False)
                fig.update_yaxes(title_text="Frequency", row=1, col=1, secondary_y=False)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text="Vol (%)", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Could not generate data for {spec_ticker}. History too short or data error.")

# --- TAB 4: OPTIONS (ILLIQUID FOCUS) ---
with tab4:
    st.header("Illiquid Options Analyzer")
    st.markdown("This tool compares **Implied Volatility (IV)** vs. **Realized Volatility (HV)** to find expensive premiums.")
    
    opt_ticker = st.text_input("Options Ticker", value="IREN")
    
    if st.button("Analyze Chain"):
        with st.spinner("Fetching Raw Chain..."):
            data = engine.get_options_analytics(opt_ticker)
            
            if data and "Error" not in data:
                # 1. The "Writer's Spread"
                c1, c2, c3 = st.columns(3)
                c1.metric("Implied Vol (The Price)", f"{data['IV']:.1%}", help="What the market is charging for options right now.")
                c2.metric("Realized Vol (The Reality)", f"{data['HV']:.1%}", help="How much the stock actually moves.")
                
                premium = data['Vol_Premium']
                delta_color = "normal" if premium > 0 else "inverse"
                c3.metric("Writer's Edge (IV - HV)", f"{premium:.1%}", delta="Expensive" if premium > 0.1 else "Cheap", delta_color=delta_color)
                
                st.divider()
                
                # 2. Visualizing the Spread
                # Simple Bar Chart comparing IV vs HV
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(name='Implied Vol (Fear)', x=['Volatility'], y=[data['IV']], marker_color='#ff5e5e')) # Red
                fig_vol.add_trace(go.Bar(name='Realized Vol (Reality)', x=['Volatility'], y=[data['HV']], marker_color='cyan')) # Cyan
                fig_vol.update_layout(barmode='group', title="Fear vs. Reality: Are Options Overpriced?", height=400)
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
