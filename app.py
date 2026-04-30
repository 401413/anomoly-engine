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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio", "🎙️ Sentiment", "🌊 Spectral & Momentum", "📉 Options", "📐 Wavelet Regime"])

# --- TAB 1: PORTFOLIO ---
with tab1:
    st.header("Reflexivity Filter")
    st.info("**Exchange Suffix Guide:** US Equities require no suffix (e.g., `AAPL`). For international markets, append the Yahoo Finance exchange suffix (e.g., London: `.L`, Xetra: `.DE`, Hong Kong: `.HK`).")
    
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
    st.info("""
    **Global Markets & Execution Reality:** Use suffixes for international tickers (e.g., `BARC.L`).  
    *Note: Data retrieved outside active local market hours reflects stale EOD prints and may exhibit widened Bid/Ask spreads.*
    """)
    
    spec_ticker = st.text_input("Ticker Symbol", value="NVDA")
    
    if st.button("Generate Wave"):
        with st.spinner("Calculating..."):
            f, spec_dates, Sxx, price_dates, prices, hist_vol = engine.generate_spectrogram_data(spec_ticker)
            
            if Sxx is not None:
                df_tech = pd.DataFrame({'Close': prices}, index=price_dates)
                
                delta = df_tech['Close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rs = gain / loss
                df_tech['RSI'] = 100 - (100 / (1 + rs))

                ema_12 = df_tech['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = df_tech['Close'].ewm(span=26, adjust=False).mean()
                df_tech['MACD'] = ema_12 - ema_26
                df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
                df_tech['MACD_Hist'] = df_tech['MACD'] - df_tech['MACD_Signal']
                
                try:
                    df_vol = yf.download(spec_ticker.split(" ")[0].upper(), period="2y", interval="1d", progress=False)
                    if isinstance(df_vol.columns, pd.MultiIndex): df_vol.columns = df_vol.columns.get_level_values(0)
                    df_tech['Volume'] = df_vol['Volume'].reindex(df_tech.index).fillna(0)
                except:
                    df_tech['Volume'] = 0
                
                df_tech['Prev_Close'] = df_tech['Close'].shift(1)
                colors_vol = ['#2ca02c' if c >= p else '#d62728' for c, p in zip(df_tech['Close'], df_tech['Prev_Close'])]

                fig = make_subplots(
                    rows=5, cols=1, 
                    shared_xaxes=True, 
                    row_heights=[0.35, 0.15, 0.15, 0.15, 0.20], 
                    vertical_spacing=0.04,
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
                    subplot_titles=("Spectral Density & Price", "30-Day Realized Volatility (%)", "Daily Trading Volume (Shares)", "MACD (12, 26, 9)", "RSI (14)")
                )
                
                fig.add_trace(go.Heatmap(z=10*np.log10(Sxx+1e-10), x=spec_dates, y=f, colorscale='Magma', colorbar=dict(x=1.05, title="Power")), row=1, col=1)
                fig.add_trace(go.Scatter(x=price_dates, y=prices, line=dict(color='cyan', width=2), name='Price'), row=1, col=1, secondary_y=True)
                fig.add_trace(go.Scatter(x=price_dates, y=hist_vol*100, line=dict(color='#ff5e5e'), name='Hist Vol', fill='tozeroy'), row=2, col=1)
                fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['Volume'], marker_color=colors_vol, name='Volume'), row=3, col=1)

                colors_macd = ['#2ca02c' if val >= 0 else '#d62728' for val in df_tech['MACD_Hist']]
                fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_Hist'], marker_color=colors_macd, name='MACD Hist'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], line=dict(color='#1f77b4', width=2), name='MACD'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_Signal'], line=dict(color='#ff7f0e', width=1.5), name='Signal'), row=4, col=1)

                fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['RSI'], line=dict(color='#9467bd', width=2), name='RSI'), row=5, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=5, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=5, col=1)
                
                fig.update_layout(height=1200, showlegend=False, template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Could not generate data for {spec_ticker}. History too short or data error.")

# --- TAB 4: OPTIONS (DURATION & GEX FOCUS) ---
with tab4:
    st.header("Duration, Skew & Gamma Arbitrage")
    st.markdown("Isolates Implied Volatility (IV) and exposes Dealer Gamma (GEX) walls to identify 'Pinning' targets.")
    
    st.warning("""
    ⚠️ **Data Integrity Warning:** Standardized options data is structurally reliable **only for US-listed equities** (no suffixes) during active market hours (09:30 - 16:00 EST).  
    *Queries executed outside these hours reflect the overnight cleared book. Market makers pull liquidity at the bell, resulting in artificially wide (or zeroed) Bid/Ask spreads. This will mathematically distort Implied Volatility and Variance Spread calculations.*
    """)
        
    col_t1, col_t2 = st.columns([1, 2])
    opt_ticker = col_t1.text_input("US Options Ticker", value="SPY")
    
    dealer_mode = col_t2.radio(
        "Dealer Positioning Assumption", 
        ["Standard (Dealers Short Calls / Short Puts)", "Covered Call Heavy (Dealers Long Calls / Short Puts)", "Inverted (Dealers Long Gamma)"],
        horizontal=True
    )
    st.caption("*Note: Institutional market structure varies. Consider Convertible Bond arbitrageurs, structured product issuance, and systemic covered-call overwriting which can heavily invert traditional dealer assumptions.*")
    
    if st.button("Analyze Options Structure"):
        with st.spinner("Fetching Term Structure & Calculating GEX..."):
            data = engine.get_options_analytics(opt_ticker)
            
            if data and "Error" not in data:
                curr_price = data.get('Current_Price', 0.0)
                hv = data.get('HV_30D', 0.0)
                beta = data.get('Beta', 1.0)
                
                st.subheader(f"Underlying Profile: {opt_ticker.upper()} @ ${curr_price:.2f}")
                st.caption(f"30-Day Realized Volatility (HV): **{hv:.1%}** | Asset Beta vs SPY: **{beta:.2f}**")
                st.divider()

                def render_duration_block(title, window_data, is_0dte=False):
                    if not window_data:
                        st.warning(f"No active options data found for this timeframe.")
                        return
                        
                    st.markdown(f"### {title} (Expiry: {window_data['Date']})")
                    
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Put/Call Ratio", f"{window_data.get('PCR', 0):.2f}")
                    c2.metric(f"Implied Vol (IV)", f"{window_data.get('IV', 0):.1%}")
                    
                    vix_label = "Beta-Adj VIX1D" if is_0dte else "Beta-Adj VIX"
                    c3.metric(vix_label, f"{window_data.get('Beta_Adj_VIX', 0):.1%}")
                    
                    premium = window_data.get('Vol_Premium', 0)
                    delta_color = "normal" if premium > 0 else "inverse"
                    c4.metric("Variance Spread", f"{premium:.1%}", delta="Expensive" if premium > 0.05 else "Discounted", delta_color=delta_color)
                    
                    c5.metric("Total Volume", f"{window_data.get('Total_Volume', 0):,.0f}", f"OI: {window_data.get('Total_OI', 0):,.0f}")

                    total_oi = window_data.get('Total_OI', 0)
                    max_capacity = int(total_oi * 0.10)
                    if total_oi > 0:
                        st.info(f"⚖️ **Strategy Capacity Constraint:** Based on institutional execution limits (10% of Open Interest), the maximum recommended trade size for this expiry window is **~{max_capacity:,} contracts**. Exceeding this will likely trigger market maker defense pricing and execution slippage.")

                    calls = window_data.get('Calls_DF', pd.DataFrame())
                    puts = window_data.get('Puts_DF', pd.DataFrame())
                    
                    if calls.empty or puts.empty:
                        st.warning("Incomplete chain data retrieved from API.")
                        return

                    calls = calls[(calls['strike'] >= curr_price * 0.85) & (calls['strike'] <= curr_price * 1.15)].copy()
                    puts = puts[(puts['strike'] >= curr_price * 0.85) & (puts['strike'] <= curr_price * 1.15)].copy()

                    if "Standard" in dealer_mode:
                        c_mult, p_mult = 1.0, -1.0
                    elif "Covered Call" in dealer_mode:
                        c_mult, p_mult = -1.0, -1.0
                    else:
                        c_mult, p_mult = -1.0, 1.0

                    calls['Adj_GEX'] = calls.get('GEX', pd.Series(0, index=calls.index)) * c_mult
                    puts['Adj_GEX'] = puts.get('GEX', pd.Series(0, index=puts.index)) * p_mult

                    st.markdown("#### Net Dealer Gamma Exposure (GEX) & Volatility Skew")
                    
                    fig_gex = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_gex.add_trace(go.Bar(name='Call GEX', x=calls['strike'], y=calls['Adj_GEX'], marker_color='rgba(0, 255, 0, 0.7)'), secondary_y=False)
                    fig_gex.add_trace(go.Bar(name='Put GEX', x=puts['strike'], y=puts['Adj_GEX'], marker_color='rgba(255, 0, 0, 0.7)'), secondary_y=False)
                    
                    iv_col = 'impliedVolatility' if 'impliedVolatility' in puts.columns else None
                    if iv_col:
                        fig_gex.add_trace(go.Scatter(name='Put IV (Convexity Skew)', x=puts['strike'], y=puts[iv_col], mode='lines', line=dict(color='#ffaa00', width=2)), secondary_y=True)

                    fig_gex.add_vline(x=curr_price, line_dash="dash", line_color="white", annotation_text="Spot")
                    fig_gex.update_layout(barmode='relative', height=400, template="plotly_dark", xaxis_title="Strike", yaxis_title="Net GEX (Shares)")
                    fig_gex.update_yaxes(title_text="Implied Volatility", secondary_y=True)
                    st.plotly_chart(fig_gex, use_container_width=True)

                    st.markdown("#### Execution Feasibility (Bid/Ask Spread Trap)")
                    
                    ntm_calls = calls[(calls['strike'] >= curr_price * 0.95) & (calls['strike'] <= curr_price * 1.05)]
                    ntm_puts = puts[(puts['strike'] >= curr_price * 0.95) & (puts['strike'] <= curr_price * 1.05)]
                    
                    avg_call_spread = ntm_calls['Spread_%'].mean() if 'Spread_%' in ntm_calls.columns and not ntm_calls.empty else 0.0
                    avg_put_spread = ntm_puts['Spread_%'].mean() if 'Spread_%' in ntm_puts.columns and not ntm_puts.empty else 0.0
                    
                    avg_call_spread = 0.0 if pd.isna(avg_call_spread) else avg_call_spread
                    avg_put_spread = 0.0 if pd.isna(avg_put_spread) else avg_put_spread
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Avg Call Slippage (NTM)", f"{avg_call_spread:.1%}", delta="Illiquid" if avg_call_spread > 0.10 else "Liquid", delta_color="inverse")
                    col_b.metric("Avg Put Slippage (NTM)", f"{avg_put_spread:.1%}", delta="Illiquid" if avg_put_spread > 0.10 else "Liquid", delta_color="inverse")

                render_duration_block("Structural Window (27-37 DTE)", data.get('30DTE'), is_0dte=False)
                st.divider()
                render_duration_block("Noise Window (0-1 DTE)", data.get('0DTE'), is_0dte=True)

            elif data and "Error" in data:
                st.error(f"Analysis Failed: {data['Error']}")

# --- TAB 5: WAVELET REGIME ---
with tab5:
    st.header("Wavelet-Based Regime Detection")
    st.info("**Global Ticker Format:** Append exchange suffixes for international queries (e.g., `LSEG.L`). DWT Energy ratio captures hourly TWAP accumulation patterns regardless of local market timezone.")
    
    st.markdown("Separates low-frequency institutional accumulation from high-frequency retail noise using DWT Energy Ratios.")
wav_ticker = st.text_input("Asset Ticker", value="SPY")
    
    if st.button("Calculate DWT Energy"):
        with st.spinner("Processing 1-hour intervals..."):
            dates, prices, energy_ratio = engine.generate_wavelet_energy(wav_ticker)
            if dates is not None:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.05)
                fig.add_trace(go.Scatter(x=dates, y=prices, name='Price', line=dict(color='cyan')), row=1, col=1)
                fig.add_trace(go.Scatter(x=dates, y=energy_ratio, name='Low/High Energy Ratio', line=dict(color='orange')), row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", title="Institutional Signal (Low-Freq Energy Ratio)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to retrieve intraday data. Ticker may be invalid or data unavailable.")
