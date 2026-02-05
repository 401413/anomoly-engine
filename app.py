import streamlit as st
import pandas as pd
from logic import StrategyEngine

st.set_page_config(page_title="Waystone/Alpha Engine", layout="wide", page_icon="⚡")

# --- CSS FOR TRANSPARENCY ---
st.markdown("""
<style>
    .stDataFrame {border: 1px solid #444;}
</style>
""", unsafe_allow_html=True)

engine = StrategyEngine()

st.title("⚡ Systematic Alpha Engine")
st.markdown("Multi-Factor Anomaly Detection: **Reflexivity (Price)** & **Sentiment (Sound)**.")

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Portfolio Sanitizer", "🎙️ CEO Sound/Sentiment"])

with tab1:
    st.header("Reflexivity & Liquidity Filter")
    target_fund = st.text_input("Target UCITS Fund (Ticker)", value="ICLN")
    
    if st.button("Run Portfolio Scan"):
        with st.spinner(f"Auditing underlying assets of {target_fund}..."):
            df = engine.sanitize_signals(target_fund)
            
            if not df.empty:
                # METRICS
                approved = df[df['Status']=='APPROVED']
                st.metric("Alpha Signals Found", len(approved), f"{len(df)-len(approved)} Rejected")
                
                # THE TRANSPARENT DATA TABLE
                st.subheader("Asset Audit Log")
                
                def color_row(row):
                    color = '#d4edda' if row['Status'] == 'APPROVED' else '#f8d7da'
                    return [f'background-color: {color}; color: black'] * len(row)

                # Show the DATA behind the decision
                display_cols = ['Ticker', 'Status', 'Reason', 'RSI', 'Return_3M', 'Vol_M']
                st.dataframe(df[display_cols].style.apply(color_row, axis=1), use_container_width=True)
                
            else:
                st.error("No holdings found. Try a liquid ETF ticker like ARKK, SMH, or ICLN.")

with tab2:
    st.header("Executive Sound & Sentiment Analysis")
    st.markdown("Analyzes the *tone* of management commentary to detect hidden confidence or stress signals.")
    
    # Text Input (Simulating an Earnings Call Transcript snippet)
    st.info("Paste a snippet from a CEO interview or Earnings Call below:")
    ceo_text = st.text_area("Transcript / Statement", height=150, 
                            value="We are seeing robust demand and our supply chain is clearing up faster than expected. We are raising guidance for Q4.")
    
    if st.button("Analyze Sentiment Signal"):
        signal, score = engine.analyze_sound_signal(ceo_text)
        
        col1, col2 = st.columns(2)
        col1.metric("Algorithmic Signal", signal, f"Score: {score:.2f}")
        
        if signal == "POSITIVE":
            st.success("✅ **BULLISH SIGNAL:** Management language indicates high conviction. Correlation with future earnings surprise is positive.")
        elif signal == "NEGATIVE":
            st.error("⚠️ **BEARISH SIGNAL:** Detected hedging/negative language. Risk of guidance downgrade.")
        else:
            st.warning("Featureless / Neutral statement.")
