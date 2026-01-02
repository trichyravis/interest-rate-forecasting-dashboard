
# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - RESTORED INSTITUTIONAL PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # 1. Profile Section
    st.markdown(f"""
        <div style="text-align: center; padding: 10px; border-radius: 10px; background-color: #FFFFFF; border: 1px solid {DARK_BLUE};">
            <h3 style="color: {DARK_BLUE}; margin-bottom: 0;">Prof. V. Ravichandran</h3>
            <p style="color: gray; font-size: 0.85rem;">The Mountain Path - World of Finance</p>
            <hr style="margin: 10px 0; border-top: 1px solid #eee;">
            <a href="https://www.linkedin.com/in/v-ravichandran-finance" target="_blank" style="text-decoration: none;">
                <button style="background-color: #0077b5; color: white; border: none; padding: 5px 15px; border-radius: 5px; cursor: pointer; width: 100%;">LinkedIn Profile</button>
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Controls
    st.header("🇺🇸 Benchmark Selection")
    ticker_label = st.selectbox("Treasury Maturity", [
        "US 10Y Treasury (^TNX)", 
        "US 30Y Treasury (^TYX)", 
        "US 5Y Treasury (^FVX)"
    ])
    ticker = ticker_label.split("(")[1].replace(")", "")
    
    st.header("⚙️ Model Parameters")
    lookback = st.slider("Historical Lookback (Years)", 1, 10, 5)
    horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20)
    
    st.markdown("---")
    run_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")
    
    # 3. Branding footer
    st.markdown(f"""
        <div style='text-align: center; color: {DARK_BLUE}; font-size: 0.75rem; padding-top: 20px;'>
            <b>The Mountain Path</b><br>
            <i>Institutional Analytics v2.1</i>
        </div>
    """, unsafe_allow_html=True)
