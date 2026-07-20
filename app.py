import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from logic import StrategyEngine

st.set_page_config(page_title="Alpha Engine Lab", layout="wide", page_icon="⚡")

# Direct scannable CSS overrides
st.markdown("""
<style>
    .stDataFrame {border: 1px solid #303030;}
    h1, h2, h3 {font-family: 'Courier New', monospace;}
</style>
""", unsafe_allow_html=True)

engine = StrategyEngine()

st.title("⚡ Systematic Alpha Engine Lab")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Sanity", "🎙️ Sentiment", "🌊 Spectral Momentum", "📐 Wavelet & GEX Structure"])

with tab1:
    st.header("Reflexivity Validation Filter")
    target_fund = st.text_input("Target UCITS/ETF Ticker", value="ICLN")
    if st.button("Execute Component Scan"):
        with st.spinner("Analyzing underlyings..."):
            df = engine.sanitize_signals(target_fund)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.error("No valid component files or historical matches returned.")

with tab2:
    st.header("Sentiment Parser")
    txt = st.text_area("Research Input Text", value="Growth remains resilient while systematic components look attractive.")
    if st.button("Parse Text"):
        sig, score = engine.analyze_sound_signal(txt)
        st.metric("Signal Output", sig, f"Score: {score:.2f}")

with tab3:
    st.header("Spectral Power Density Distributions")
    spec_ticker = st.text_input("Analysis Symbol Ticker", value="NVDA")
    if st.button("Map Wave Dynamics"):
        with st.spinner("Decomposing frequencies..."):
            f, spec_dates, Sxx, price_dates, prices, hist_vol = engine.generate_spectrogram_data(spec_ticker)
            if Sxx is not None:
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                    subplot_titles=("Spectral Power Density Heatmap", "Historical Realized Volatility", "Asset Underlying Close Price")
                )
                fig.add_trace(go.Heatmap(z=10*np.log10(Sxx+1e-10), x=spec_dates, y=f, colorscale='Viridis', showscale=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=price_dates, y=hist_vol*100, name="Realized Vol", line=dict(color='#ff4b4b')), row=2, col=1)
                fig.add_trace(go.Scatter(x=price_dates, y=prices, name="Spot Price", line=dict(color='#00f2fe')), row=3, col=1)
                fig.update_layout(height=800, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Insufficient timeline data to generate full spectral maps.")

with tab4:
    st.header("Structural Quant Frameworks")
    str_ticker = st.text_input("Asset Target", value="SPY")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("Wavelet Regime Shifts")
        if st.button("Run DWT Deconstruction"):
            with st.spinner("Reconstructing arrays..."):
                dates, prices, ratio = engine.generate_wavelet_energy(str_ticker)
                if dates is not None:
                    fig_wav = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.05)
                    fig_wav.add_trace(go.Scatter(x=dates, y=prices, name="Price", line=dict(color='#00f2fe')), row=1, col=1)
                    fig_wav.add_trace(go.Scatter(x=dates, y=ratio, name="Institutional Low/High Ratio", line=dict(color='#ffb300')), row=2, col=1)
                    fig_wav.update_layout(height=500, template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig_wav, use_container_width=True)
                else:
                    st.error("Wavelet pipeline calculation errored out. Verify input history availability.")

    with col_r:
        st.subheader("Dealer Gamma Exposure (GEX) Wings")
        if st.button("Map Volatility Skew"):
            with st.spinner("Extracting chains..."):
                opt = engine.get_options_analytics(str_ticker)
                if opt and "Error" not in opt:
                    st.markdown(f"**Target Chain Expiry:** `{opt['Date']}`")
                    st.markdown(f"**ATM Vol Premium:** `{opt['Vol_Premium']:.2%}` over baseline indices.")
                    
                    calls, puts, spot = opt['Calls_DF'], opt['Puts_DF'], opt['Current_Price']
                    
                    # Window options tightly around active field areas
                    c_f = calls[(calls['strike'] >= spot * 0.9) & (calls['strike'] <= spot * 1.1)]
                    p_f = puts[(puts['strike'] >= spot * 0.9) & (puts['strike'] <= spot * 1.1)]
                    
                    fig_gex = go.Figure()
                    fig_gex.add_trace(go.Bar(name='Call GEX', x=c_f['strike'], y=c_f['GEX'], marker_color='rgba(0, 255, 150, 0.6)'))
                    fig_gex.add_trace(go.Bar(name='Put GEX', x=p_f['strike'], y=p_f['GEX'], marker_color='rgba(255, 50, 50, 0.6)'))
                    fig_gex.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="Spot Line")
                    fig_gex.update_layout(barmode='relative', height=400, template="plotly_dark", xaxis_title="Strikes", yaxis_title="Net Shares Profile")
                    st.plotly_chart(fig_gex, use_container_width=True)
                elif opt and "Error" in opt:
                    st.error(opt["Error"])
