import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from logic import StrategyEngine

# --- PAGE CONFIG ---
st.set_page_config(page_title="Waystone/Systematic Monitor", layout="wide", page_icon="⚡")

# --- CUSTOM CSS (Institutional Dark Mode) ---
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #303030; padding: 15px; border-radius: 5px;}
    .stAlert {background-color: #262730; border: 1px solid #FF4B4B; color: white;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("System Controls")
target_fund = st.sidebar.text_input("Target UCITS Fund (Ticker)", value="ICLN") # Default: Global Clean Energy
risk_tone = st.sidebar.selectbox("Reporting Tone", ["Risk Committee (Governance)", "Sales Desk (Distribution)"])
st.sidebar.markdown("---")
st.sidebar.caption("Execution Bridge: **CONNECTED (IBKR)**")

# --- MAIN HEADER ---
st.title("⚡ Systematic Anomaly & Signal Engine")
st.markdown("Automated Signal Stripping for Thematic UCITS Funds.")

# --- LOGIC INITIALIZATION ---
engine = StrategyEngine()

# --- INPUT SECTION ---
if st.sidebar.button("Run Analysis", type="primary"):
    with st.spinner(f"Connecting to live feed for {target_fund}..."):
        df_sanitized = engine.sanitize_signals(target_fund)
        
        if not df_sanitized.empty:
            # METRICS ROW
            approved_count = len(df_sanitized[df_sanitized['Status'] == 'APPROVED'])
            rejected_count = len(df_sanitized[df_sanitized['Status'] == 'REJECTED'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target Fund", target_fund)
            col2.metric("Alpha Signals Found", approved_count)
            col3.metric("Traps Rejected", rejected_count, delta="Risk Avoided", delta_color="normal")
            
            # DATA TABLE
            st.markdown("---")
            st.subheader("Signal Sanitization Output")
            
            def color_status(val):
                if val == 'APPROVED': color = '#90ee90'
                elif val == 'REJECTED': color = '#ffcccb'
                else: color = 'white'
                return f'background-color: {color}; color: black'

            st.dataframe(df_sanitized.style.applymap(color_status, subset=['Status']), use_container_width=True)
            
            # AUTOMATED REPORT
            st.markdown("---")
            st.subheader("Automated Commentary")
            approved_tickers = df_sanitized[df_sanitized['Status']=='APPROVED']['Ticker'].tolist()
            rejected_tickers = df_sanitized[df_sanitized['Status']=='REJECTED']['Ticker'].tolist()
            
            if "Risk" in risk_tone:
                report = f"**RISK COMMITTEE:** Filter applied to {target_fund}. {len(rejected_tickers)} holdings flagged for liquidity/momentum distortion (Reflexivity Risk). Exposure authorized only for high-conviction assets: {', '.join(approved_tickers)}."
                st.info(report, icon="🛡️")
            else:
                report = f"**SALES DESK:** While the broad fund {target_fund} is crowded, our system has isolated the true alpha drivers: {', '.join(approved_tickers)}. We have stripped out the passive beta to offer pure thematic exposure."
                st.success(report, icon="🚀")
                
        else:
            st.error("Unable to fetch holdings. Ensure Ticker is a valid ETF (e.g., ARKK, ICLN, SMH).")

# --- DEFAULT VIEW (Before Search) ---
else:
    st.info("👈 Enter a UCITS ETF Ticker in the sidebar to begin live analysis.")
    st.markdown("### System Architecture")
    
    st.markdown("""
    * **Input:** Real-time API feed (Yahoo Finance / Morningstar Proxy).
    * **Filter 1:** Beta Stripping (Removing Microsoft/Apple/Cash).
    * **Filter 2:** Reflexivity Detector (Removing Low-Liquidity/High-Momentum Traps).
    * **Output:** Executable Order List sent to IBKR Bridge.
    """)
