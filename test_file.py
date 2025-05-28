import streamlit as st

# Custom CSS to mimic the dark green theme
st.markdown("""
    <style>
    .main {
        background-color: #1A3C34;
        color: #E0E0E0;
    }
    .stButton>button {
        background-color: #00CC00;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
    }
    .stTextInput>div>input, .stSelectbox>div>select, .stTextArea>div>textarea {
        background-color: #2A5A4A;
        color: #E0E0E0;
        border: none;
    }
    .stNumberInput>div>input {
        background-color: #2A5A4A;
        color: #E0E0E0;
        border: none;
    }
    h1, h2, h3 {
        color: #E0E0E0;
    }
    .strategy-box {
        background-color: #2A5A4A;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header and Navigation
st.markdown('<h1 style="color: #00CC00;">AlgoTrader</h1>', unsafe_allow_html=True)
st.markdown('<div style="display: flex; justify-content: flex-end; margin-bottom: 20px;"><a href="#" style="color: #00CC00; margin-right: 20px;">Dashboard</a><a href="#" style="color: #00CC00; margin-right: 20px;">Strategies</a><a href="#" style="color: #00CC00; margin-right: 20px;">Trades</a><a href="#" style="color: #00CC00;">Settings</a></div>', unsafe_allow_html=True)

# Title and Subtitle
st.markdown("<h2>Trading Strategies</h2>", unsafe_allow_html=True)
st.markdown("Create or select a trading strategy with customizable parameters.", unsafe_allow_html=True)

# Available Strategies Section
st.markdown("<h3>Available Strategies</h3>", unsafe_allow_html=True)
cols = st.columns(3)

with cols[0]:
    st.markdown('<div class="strategy-box">', unsafe_allow_html=True)
    st.markdown("<h4>Momentum Breakout</h4>", unsafe_allow_html=True)
    st.write("Identifies price breaking out of a recent range with increasing volume.")
    st.markdown('<div style="height: 150px; background-color: #1A3C34; display: flex; justify-content: center; align-items: center;">[Chart Placeholder]</div>', unsafe_allow_html=True)
    if st.button("Select Strategy", key="momentum"):
        st.write("Momentum Breakout selected!")
    st.markdown('</div>', unsafe_allow_html=True)

with cols[1]:
    st.markdown('<div class="strategy-box">', unsafe_allow_html=True)
    st.markdown("<h4>Mean Reversion</h4>", unsafe_allow_html=True)
    st.write("Trades stocks that have deviated significantly from their historical average price.")
    st.markdown('<div style="height: 150px; background-color: #1A3C34; display: flex; justify-content: center; align-items: center;">[Chart Placeholder]</div>', unsafe_allow_html=True)
    if st.button("Select Strategy", key="mean_reversion"):
        st.write("Mean Reversion selected!")
    st.markdown('</div>', unsafe_allow_html=True)

with cols[2]:
    st.markdown('<div class="strategy-box">', unsafe_allow_html=True)
    st.markdown("<h4>Pairs Trading</h4>", unsafe_allow_html=True)
    st.write("Exploits price discrepancies between two historically correlated stocks.")
    st.markdown('<div style="height: 150px; background-color: #1A3C34; display: flex; justify-content: center; align-items: center;">[Chart Placeholder]</div>', unsafe_allow_html=True)
    if st.button("Select Strategy", key="pairs"):
        st.write("Pairs Trading selected!")
    st.markdown('</div>', unsafe_allow_html=True)

# Create New Strategy Section
st.markdown("<h3>Create New Strategy</h3>", unsafe_allow_html=True)
with st.form(key="new_strategy_form"):
    strategy_name = st.text_input("Strategy Name", placeholder="Enter strategy name")
    strategy_type = st.selectbox("Strategy Type", ["Select strategy type", "Momentum Breakout", "Mean Reversion", "Pairs Trading"])
    strategy_description = st.text_area("Strategy Description", placeholder="Enter strategy description")

    # Parameters Section
    st.markdown("<h4>Parameters</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        breakout_period = st.number_input("Breakout Period (days)", min_value=1, value=20, step=1)
        target_profit = st.number_input("Target Profit (%)", min_value=0.0, value=5.0, step=0.1)
    with col2:
        stop_loss = st.number_input("Stop Loss (%)", min_value=0.0, value=2.5, step=0.1)
        max_allocation = st.number_input("Maximum Allocation (%)", min_value=0.0, value=10.0, step=0.1)

    submit_button = st.form_submit_button("Create Strategy")

    if submit_button:
        if strategy_name and strategy_type != "Select strategy type" and strategy_description:
            st.success(f"Strategy '{strategy_name}' created successfully!")
        else:
            st.error("Please fill in all required fields.")