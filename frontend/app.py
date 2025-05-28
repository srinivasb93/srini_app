import streamlit as st
import aiohttp
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from streamlit_autorefresh import st_autorefresh
import altair as alt
import json
import logging
import os, sys
import streamlit_echarts as echarts
from ta.momentum import RSIIndicator
import uuid
# Add the project_root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

from common_utils.upstox_utils import fetch_instruments, get_market_quote
from common_utils.read_write_sql_data import load_sql_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Stock Trading Dashboard", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-top: -5em; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stTextInput>input { border-radius: 5px; }
    .sidebar .sidebar-content { background-color: #e0e0e0; padding: 10px; border-radius: 10px; }
    .metric-box { background-color: #336699; color: white; padding: 10px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .stForm { border: 1px solid #d3d3d3; padding: 15px; border-radius: 10px; }
    .login-container { max-width: 500px; margin: auto; padding: 20px; }
    @media (max-width: 768px) {
        .main { padding: 10px; margin-top: -2em; }
        .metric-box { font-size: 0.9em; padding: 8px; }
        .stButton>button { width: 100%; }
    }
    </style>
""", unsafe_allow_html=True)

BACKEND_URL = "http://localhost:8000"

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@st.cache_data(ttl=3600)
def fetch_index_instruments(index_name, broker):
    try:
        response = run_async(make_api_request("GET", f"{BACKEND_URL}/instruments/{broker}/?exchange=NSE",
                                             token=st.session_state.get('access_token')))
        instruments_df = pd.DataFrame(response)
        index_symbols = instruments_df[instruments_df["name"].str.contains(index_name, na=False)]
        return {row["trading_symbol"]: row["instrument_token"] for _, row in index_symbols.iterrows()}
    except Exception as e:
        logger.error(f"Failed to fetch instruments for {index_name}: {str(e)}")
        return {}

async def make_api_request(method, url, token=None, **kwargs):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if "auth/login" in url:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.request(method, url, headers=headers, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            st.error(f"API request failed: {str(e)}")
            if e.status == 401:
                st.session_state.pop("access_token", None)
                st.error("Session expired. Please log in again.")
                st.rerun()
            raise
        except Exception as e:
            st.error(f"API request failed: {str(e)}")
            raise

if 'instruments_data' not in st.session_state:
    try:
        st.session_state['instruments_data'] = fetch_instruments()
        logger.info("Instruments data loaded")
    except Exception as e:
        logger.error(f"Failed to load instruments: {str(e)}")
        st.error(f"Failed to load instruments: {str(e)}")

instruments = st.session_state.get('instruments_data', {})

# Cached wrapper for API calls
@st.cache_data(ttl=300)
def cached_api_request(method, url, token=None, **kwargs):
    return run_async(make_api_request(method, url, token, **kwargs))

@st.cache_data(ttl=60)
def cached_market_data(method, url, token=None, **kwargs):
    return run_async(make_api_request(method, url, token, **kwargs))

@st.cache_data(ttl=3600)
def cached_historical_data(method, url, token=None, **kwargs):
    return run_async(make_api_request(method, url, token, **kwargs))

@st.cache_data(ttl=86400)
def cached_instruments(method, url, token=None, **kwargs):
    return run_async(make_api_request(method, url, token, **kwargs))

def render_login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("Stock Trading Dashboard")
    tabs = st.tabs(["Login", "Register"])
    with tabs[0]:
        with st.form("login_form"):
            username = st.text_input("Email", help="Enter your registered email address")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                try:
                    response = run_async(make_api_request("POST", f"{BACKEND_URL}/auth/login", data={
                        "username": username,
                        "password": password,
                        "grant_type": "password"
                    }))
                    st.session_state.access_token = response["access_token"]
                    st.success("Login successful!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
                    st.toast("Login failed", icon="❌")
    with tabs[1]:
        with st.form("register_form"):
            new_username = st.text_input("Email", key="register_username")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            upstox_api_key = st.text_input("Upstox API Key")
            upstox_api_secret = st.text_input("Upstox API Secret")
            zerodha_api_key = st.text_input("Zerodha API Key")
            zerodha_api_secret = st.text_input("Zerodha API Secret")
            if st.form_submit_button("Register"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                    st.toast("Registration failed", icon="❌")
                else:
                    try:
                        response = run_async(make_api_request("POST", f"{BACKEND_URL}/auth/register", json={
                            "email": new_username,
                            "password": new_password,
                            "upstox_api_key": upstox_api_key,
                            "upstox_api_secret": upstox_api_secret,
                            "zerodha_api_key": zerodha_api_key,
                            "zerodha_api_secret": zerodha_api_secret
                        }))
                        st.success("Registration successful! Please log in.")
                        st.toast("Registration successful", icon="✅")
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
                        st.toast("Registration failed", icon="❌")
    st.markdown('</div>', unsafe_allow_html=True)

def render_top_nav():
    user_email = "Unknown"
    try:
        profile = run_async(make_api_request("GET", f"{BACKEND_URL}/profile/Upstox/", token=st.session_state.access_token))
        user_email = profile["email"]
    except:
        pass
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown(f"**Welcome, {user_email}**")
    with cols[1]:
        if st.button("Logout", key="top_logout"):
            st.session_state.pop("access_token", None)
            st.rerun()

def render_sidebar():
    with st.sidebar:
        st.title("Trading Dashboard")
        theme = st.selectbox("Theme", ["Light", "Dark"])
        if theme == "Dark":
            st.markdown("""
                <style>
                [data-testid="stAppViewContainer"] { background-color: #1e1e1e; color: #ffffff; }
                [data-testid="stSidebar"] { background-color: #2c2c2c; }
                .metric-box { background-color: #1a3c6e; }
                </style>
            """, unsafe_allow_html=True)
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Get Token", "Order Management", "Order Book", "Positions", "Trade Dashboard",
                     "Portfolio", "Mutual Funds", "Analytics", "Algo Trading", "Strategy Backtesting"],
            icons=["house", "key", "cart", "book", "bar-chart", "clock-history", "briefcase", "fund", "graph-up",
                   "robot", "test-tube"],
            default_index=0,
        )
        broker = st.selectbox("Select Broker", ["Upstox", "Zerodha"], key="select_broker")
        try:
            response = run_async(make_api_request("GET", f"{BACKEND_URL}/profile/{broker}/", token=st.session_state.access_token))
            st.success(f"{broker} Connected: {response['name']}")
        except:
            st.error(f"{broker} Not Connected")
    return selected, broker

async def fetch_dashboard_data(broker, token):
    async with aiohttp.ClientSession() as session:
        funds_task = make_api_request("GET", f"{BACKEND_URL}/funds/{broker}/", token=token)
        portfolio_task = make_api_request("GET", f"{BACKEND_URL}/portfolio/{broker}/", token=token)
        positions_task = make_api_request("GET", f"{BACKEND_URL}/positions/{broker}/", token=token)
        funds_data, portfolio, positions = await asyncio.gather(funds_task, portfolio_task, positions_task, return_exceptions=True)
        return funds_data, portfolio, positions


def render_dashboard(broker):
    st_autorefresh(interval=60000, key="dashboard_refresh")
    st.subheader(f"{broker} Trading Dashboard")
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        try:
            funds_data, portfolio, positions = run_async(fetch_dashboard_data(broker, st.session_state.access_token))
            with col1:
                if isinstance(funds_data, Exception):
                    st.markdown(f'<div class="metric-box">Funds: N/A ({str(funds_data)})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-box">Available Funds: ₹{funds_data["equity"]["available"]:.2f}</div>',
                                unsafe_allow_html=True)
            with col2:
                if isinstance(portfolio, Exception):
                    st.markdown(f'<div class="metric-box">Portfolio: N/A ({str(portfolio)})</div>', unsafe_allow_html=True)
                else:
                    total_value = sum(holding["Quantity"] * holding["LastPrice"] for holding in portfolio)
                    st.markdown(f'<div class="metric-box">Portfolio Value: ₹{total_value:.2f}</div>',
                                unsafe_allow_html=True)
            with col3:
                if isinstance(positions, Exception):
                    st.markdown(f'<div class="metric-box">Positions: N/A ({str(positions)})</div>', unsafe_allow_html=True)
                else:
                    open_positions = len([pos for pos in positions if pos["Quantity"] != 0])
                    st.markdown(f'<div class="metric-box">Open Positions: {open_positions}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to load dashboard data: {str(e)}")

    st.write("### Market Watch")
    with st.container():
        index_name = st.selectbox("Select Index", ["NIFTY 50", "NIFTY NEXT 50"], key="dashboard_index")
        instruments = fetch_index_instruments(index_name, broker)
        instrument = st.selectbox("Select Instrument", list(instruments.keys()), key="dashboard_instrument")
        try:
            ltp_data = cached_market_data("GET", f"{BACKEND_URL}/ltp/{broker}/?instruments={instruments[instrument]}",
                                          token=st.session_state.access_token)
            st.markdown(f'<div class="metric-box">{instrument} LTP: ₹{ltp_data[0]["last_price"]:.2f}</div>',
                        unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to fetch LTP: {str(e)}. Please check your broker connection.")

    if not isinstance(portfolio, Exception) and portfolio:
        st.write("### Portfolio Allocation")
        portfolio_df = pd.DataFrame(portfolio)
        if not portfolio_df.empty:
            chart_data = {
                "tooltip": {"trigger": "item"},
                "legend": {"top": "5%", "left": "center"},
                "series": [{
                    "name": "Portfolio Allocation",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": False,
                    "label": {"show": False, "position": "center"},
                    "emphasis": {"label": {"show": True, "fontSize": "20", "fontWeight": "bold"}},
                    "labelLine": {"show": False},
                    "data": [
                        {"value": holding["Quantity"] * holding["LastPrice"], "name": holding["Symbol"]}
                        for holding in portfolio
                    ],
                    "itemStyle": {
                        "color": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
                    }
                }]
            }
            echarts.st_echarts(options=chart_data, height="400px")
        else:
            st.info("No portfolio data available.")

def render_get_token(broker):
    st.subheader(f"Fetch {broker} Access Token")
    if broker == "Upstox":
        api_key = st.text_input("API Key", value="", help="Enter your Upstox API key")
        redirect_uri = st.text_input("Redirect URI", "https://api.upstox.com/v2/login")
        auth_url = f"https://api-v2.upstox.com/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}"
        st.markdown(f"[Click here to authorize]({auth_url})")
        auth_code = st.text_input("Authorization Code", help="Paste the code from the redirect URL")
        if st.button("Fetch Access Token") and auth_code:
            try:
                response = run_async(make_api_request("POST", f"{BACKEND_URL}/auth/upstox/?auth_code={auth_code}",
                                                     token=st.session_state.access_token))
                st.success("Successfully fetched Upstox Access Token")
                st.toast("Token fetched", icon="✅")
            except Exception as e:
                st.error(f"Failed to fetch Upstox token: {str(e)}")
                st.toast("Token fetch failed", icon="❌")
    elif broker == "Zerodha":
        request_token = st.text_input("Request Token", help="Obtain from Zerodha's redirect URL after login")
        if st.button("Fetch Access Token") and request_token:
            try:
                response = run_async(make_api_request("POST", f"{BACKEND_URL}/auth/zerodha/?request_token={request_token}",
                                                     token=st.session_state.access_token))
                st.success("Successfully fetched Zerodha Access Token")
                st.toast("Token fetched", icon="✅")
            except Exception as e:
                st.error(f"Failed to fetch Zerodha token: {str(e)}")
                st.toast("Token fetch failed", icon="❌")

def render_order_management(broker):
    st.subheader(f"Order Management - {broker}")
    tabs = st.tabs(["Regular Orders", "Scheduled Orders", "GTT Orders", "Auto Orders"])
    with tabs[0]:
        with st.form("regular_order_form"):
            st.write("#### Symbol Selection")
            index_name = st.selectbox("Select Index", ["NIFTY 50", "NIFTY NEXT 50"], key="order_index")
            stock_symbol = st.selectbox("Symbol", options=instruments.keys(), key="order_symbol")
            instrument_token = instruments.get(stock_symbol)
            schedule_order = st.checkbox("Schedule Order", help="Check to schedule the order for a future time")

            st.write("#### Order Details")
            cols = st.columns(4)
            quantity = cols[0].number_input("Quantity", min_value=1, value=1, help="Number of shares to trade")
            order_type = cols[1].selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"], help="Type of order")
            transaction_type = cols[2].radio("Transaction", ["BUY", "SELL"], horizontal=True)
            product_type = cols[3].radio("Product", ['I', 'D'] if broker == "Upstox" else ['MIS', 'CNC'])

            st.write("#### Pricing")
            price = st.number_input("Price", min_value=0.0, value=0.0, disabled=order_type != "LIMIT",
                                    help="Limit price for LIMIT orders")
            trigger_price = st.number_input("Trigger Price", min_value=0.0, value=0.0,
                                            disabled=order_type not in ["SL", "SL-M"],
                                            help="Trigger price for SL/SL-M orders")

            st.write("#### Risk Management")
            stop_loss = st.number_input("Stop-Loss", min_value=0.0, value=0.0, help="Price to limit losses")
            target = st.number_input("Target", min_value=0.0, value=0.0, help="Price to take profits")

            st.write("#### Additional Options")
            amo_order = st.checkbox("AMO Order", help="After Market Order")
            schedule_datetime = None
            if schedule_order:
                schedule_cols = st.columns(2)
                schedule_date = schedule_cols[0].date_input("Schedule Date", value=datetime.now(),
                                                            min_value=datetime.now())
                schedule_time = schedule_cols[1].time_input("Schedule Time", value=datetime.now().time(), step=60)
                schedule_datetime = datetime.combine(schedule_date, schedule_time)

            st.write("#### Order Preview")
            order_data = {
                "trading_symbol": stock_symbol,
                "instrument_token": instrument_token,
                "quantity": quantity,
                "order_type": order_type,
                "transaction_type": transaction_type,
                "product_type": product_type,
                "is_amo": amo_order,
                "price": price,
                "trigger_price": trigger_price,
                "stop_loss": stop_loss,
                "target": target,
                "validity": "DAY",
                "broker": broker,
                "schedule_datetime": schedule_datetime.isoformat() if schedule_datetime else None
            }
            st.json(order_data)

            if st.form_submit_button("Place Order"):
                try:
                    with st.spinner("Placing order..."):
                        response = run_async(make_api_request("POST",
                                                              f"{BACKEND_URL}/{'scheduled-orders' if schedule_order else 'orders'}/",
                                                              json=order_data, token=st.session_state.access_token))
                    st.success(
                        f"Order {'scheduled' if schedule_order else 'placed'}: {response.get('order_id', response.get('message'))}")
                    st.toast("Order processed successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Failed to {'schedule' if schedule_order else 'place'} order: {str(e)}")
                    st.toast("Order failed", icon="❌")

    with tabs[1]:
        try:
            response = cached_api_request("GET", f"{BACKEND_URL}/orders/{broker}/", token=st.session_state.access_token)
            orders_df = pd.DataFrame(response)
            scheduled_df = orders_df[orders_df["status"] == "PENDING"]
            if not scheduled_df.empty:
                st.dataframe(
                    scheduled_df[["order_id", "trading_symbol", "quantity", "order_type", "schedule_datetime"]])
                order_id = st.selectbox("Select Order to Cancel", scheduled_df["order_id"], key="cancel_schedule")
                if st.button("Cancel Scheduled Order"):
                    try:
                        response = run_async(make_api_request("DELETE", f"{BACKEND_URL}/orders/{order_id}",
                                                              token=st.session_state.access_token))
                        st.success(f"Order {order_id} cancelled")
                        st.toast("Order cancelled", icon="✅")
                    except Exception as e:
                        st.error(f"Failed to cancel order: {str(e)}")
                        st.toast("Cancellation failed", icon="❌")
            else:
                st.info("No scheduled orders")
        except Exception as e:
            st.error(f"Failed to fetch scheduled orders: {str(e)}")

    with tabs[2]:
        if broker == "Zerodha":
            try:
                response = cached_api_request("GET", f"{BACKEND_URL}/gtt-orders/{broker}/",
                                              token=st.session_state.access_token)
                gtt_df = pd.DataFrame(response)
                if not gtt_df.empty:
                    st.dataframe(gtt_df[["gtt_order_id", "trading_symbol", "quantity", "trigger_price", "status"]])
                    gtt_id = st.selectbox("Select GTT Order", gtt_df["gtt_order_id"], key="delete_gtt")
                    if st.button("Delete GTT Order"):
                        try:
                            response = run_async(make_api_request("DELETE", f"{BACKEND_URL}/gtt-orders/{broker}/{gtt_id}",
                                                                  token=st.session_state.access_token))
                            st.success(f"GTT Order {gtt_id} deleted")
                            st.toast("GTT Order deleted", icon="✅")
                        except Exception as e:
                            st.error(f"Failed to delete GTT order: {str(e)}")
                            st.toast("Deletion failed", icon="❌")
                else:
                    st.info("No GTT orders")
            except Exception as e:
                st.error(f"Failed to fetch GTT orders: {str(e)}")
        else:
            st.info("GTT orders are only supported for Zerodha")

    with tabs[3]:
        st.write("##### Add New Auto Order")
        auto_cols = st.columns(6)
        auto_stock_symbol = auto_cols[0].selectbox("Select Symbol", options=instruments.keys(), key="auto_symbol")
        auto_instrument_token = instruments.get(auto_stock_symbol)
        auto_transaction_type = auto_cols[1].radio("Transaction Type", ["BUY", "SELL"], horizontal=True,
                                                   key="auto_trans_type")
        auto_order_type = auto_cols[2].selectbox("Order Type", ["MARKET", "LIMIT"], key="auto_order_type")
        auto_product_type = auto_cols[3].selectbox("Product Type", ["I", "D"] if broker == "Upstox" else ["MIS", "CNC"],
                                                   key="auto_product_type")
        if auto_order_type == "LIMIT":
            live_data = get_market_quote(None, auto_instrument_token)
            current_ltp = live_data.get("ltp", 0) if live_data else 0
            auto_limit_price = auto_cols[5].number_input("Limit Price", min_value=0.05, value=current_ltp, step=0.05,
                                                         key="auto_limit_price")
        else:
            auto_limit_price = 0.0
        auto_risk_per_trade = auto_cols[4].number_input("Risk per Trade (%)", min_value=0.1, value=1.0, step=0.1,
                                                        key="auto_risk")
        auto_stop_loss_type = auto_cols[0].selectbox("Stop Loss Type",
                                                     ["Fixed Amount", "Percentage of Entry", "ATR Based"],
                                                     key="auto_sl_type")
        if auto_stop_loss_type == "Fixed Amount":
            auto_stop_loss_value = auto_cols[1].number_input("Stop Loss Value (Rs.)", min_value=1.0, value=100.0,
                                                             step=1.0, key="auto_sl_fixed")
            auto_target_value = auto_cols[2].number_input("Target Value (Rs.)", min_value=1.0, value=250.0, step=1.0,
                                                          key="auto_target_fixed")
        elif auto_stop_loss_type == "Percentage of Entry":
            auto_stop_loss_percent = auto_cols[1].number_input("Stop Loss (%)", min_value=0.1, value=1.0, step=0.1,
                                                               key="auto_sl_pct")
            auto_target_percent = auto_cols[2].number_input("Target (%)", min_value=0.1, value=2.5, step=0.1,
                                                            key="auto_target_pct")
        else:
            auto_atr_period = auto_cols[1].number_input("ATR Period", min_value=5, value=14, step=1,
                                                        key="auto_atr_period")
            auto_stop_loss_atr_mult = auto_cols[2].number_input("Stop Loss ATR Multiplier", min_value=0.5, value=2.0,
                                                                step=0.5, key="auto_sl_atr")
            auto_target_atr_mult = auto_cols[3].number_input("Target ATR Multiplier", min_value=0.5, value=5.0,
                                                             step=0.5, key="auto_target_atr")
        if st.button("Add Auto Order"):
            auto_order = {
                "auto_order_id": str(uuid.uuid4()),
                "instrument_token": auto_instrument_token,
                "trading_symbol": auto_stock_symbol,
                "transaction_type": auto_transaction_type,
                "risk_per_trade": auto_risk_per_trade,
                "stop_loss_type": auto_stop_loss_type,
                "stop_loss_value": auto_stop_loss_value if auto_stop_loss_type == "Fixed Amount" else (
                    auto_stop_loss_percent if auto_stop_loss_type == "Percentage of Entry" else auto_stop_loss_atr_mult),
                "target_value": auto_target_value if auto_stop_loss_type == "Fixed Amount" else (
                    auto_target_percent if auto_stop_loss_type == "Percentage of Entry" else auto_target_atr_mult),
                "atr_period": auto_atr_period if auto_stop_loss_type == "ATR Based" else 14,
                "product_type": auto_product_type,
                "order_type": auto_order_type,
                "limit_price": auto_limit_price,
                "user_id": "default_user",
                "broker": broker
            }
            load_sql_data(pd.DataFrame([auto_order]), "auto_orders", load_type="append", index_required=False)
            st.success(f"Auto order added for {auto_stock_symbol}")
            st.toast("Auto order added", icon="✅")

def render_order_book(broker):
    st.subheader("Order Book")
    tabs = st.tabs(["Orders", "Scheduled Orders", "Auto Orders", "GTT Orders"])
    with tabs[0]:
        try:
            response = cached_api_request("GET", f"{BACKEND_URL}/order-book/{broker}",
                                          token=st.session_state.access_token)
            orders_df = pd.DataFrame(response)
            if not orders_df.empty:
                orders_df['OrderTime'] = pd.to_datetime(orders_df['OrderTime'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    status_filter = st.multiselect("Filter by Status",
                                                   options=["PENDING", "COMPLETED", "REJECTED", "CANCELLED",
                                                            "MODIFIED"], default=[])
                with col2:
                    transaction_filter = st.multiselect("Filter by Transaction Type", options=["BUY", "SELL"],
                                                        default=[])
                with col3:
                    broker_filter = st.multiselect("Filter by Broker", options=["Zerodha", "Upstox"], default=[broker])
                filtered_df = orders_df
                if status_filter:
                    filtered_df = filtered_df[filtered_df["Status"].isin(status_filter)]
                if transaction_filter:
                    filtered_df = filtered_df[filtered_df["TransType"].isin(transaction_filter)]
                if broker_filter:
                    filtered_df = filtered_df[filtered_df["Broker"].isin(broker_filter)]
                st.dataframe(filtered_df.sort_values(by="OrderTime", ascending=False))
                filter_cols = ['complete', 'rejected', 'cancelled', 'cancelled after market order', 'CANCELLED',
                               'CANCELLED AMO', 'REJECTED', 'COMPLETE']
                col1, col2, _, col3 = st.columns([1, 0.75, 0.75, 0.36])
                selected_order_id = col1.selectbox("Select Order to Modify/Cancel", options=filtered_df["OrderID"][
                    ~filtered_df["Status"].isin(filter_cols)].tolist())
                if selected_order_id:
                    order_data = filtered_df[filtered_df["OrderID"] == selected_order_id].iloc[0]
                    with col2:
                        if order_data["Status"] not in filter_cols:
                            if st.button("Cancel Order"):
                                try:
                                    response = run_async(make_api_request("DELETE", f"{BACKEND_URL}/orders/{selected_order_id}",
                                                                          token=st.session_state.access_token))
                                    st.success(f"Order cancelled successfully: {response['message']}")
                                    st.toast("Order cancelled", icon="✅")
                                except Exception as e:
                                    st.error(f"Failed to cancel order: {str(e)}")
                                    st.toast("Cancellation failed", icon="❌")
                    with col3:
                        if st.button("Cancel All Orders", type='primary'):
                            try:
                                response = run_async(make_api_request("DELETE", f"{BACKEND_URL}/orders/all",
                                                                      token=st.session_state.access_token))
                                st.success("All orders cancelled")
                                st.toast("Orders cancelled", icon="✅")
                            except Exception as e:
                                st.error(f"Failed to cancel all orders: {str(e)}")
                                st.toast("Cancellation failed", icon="❌")
                    modify_cols = st.columns(2)
                    with modify_cols[1]:
                        with st.expander("Order Details"):
                            row_cols = st.columns(2)
                            with row_cols[0]:
                                st.write(f"Broker: {order_data['Broker']}")
                                st.write(f"Symbol: {order_data['Symbol']}")
                                st.write(f"Exchange: {order_data['Exchange']}")
                                st.write(f"Transaction Type: {order_data['TransType']}")
                                st.write(f"Order Type: {order_data['OrderType']}")
                                st.write(f"Product: {order_data['Product']}")
                            with row_cols[1]:
                                st.write(f"Quantity: {order_data['Quantity']}")
                                st.write(f"Status: {order_data['Status']}")
                                st.write(f"Price: {order_data['Price']}")
                                st.write(f"Trigger Price: {order_data['TriggerPrice']}")
                    with modify_cols[0]:
                        if order_data["Status"] not in filter_cols:
                            st.subheader("Modify Order")
                            modify_order_cols = st.columns(3)
                            new_quantity = modify_order_cols[0].number_input("New Quantity", min_value=1,
                                                                             value=int(order_data["Quantity"]))
                            new_order_type = modify_order_cols[2].selectbox("Order Type",
                                                                            ['LIMIT', 'MARKET', 'SL', 'SL-M'],
                                                                            index=['LIMIT', 'MARKET', 'SL',
                                                                                   'SL-M'].index(
                                                                                order_data["OrderType"]))
                            if new_order_type in ["LIMIT", "SL"]:
                                new_price = st.number_input("New Price", min_value=0.05, step=0.05, format="%.2f",
                                                            value=float(order_data["Price"]))
                            else:
                                new_price = 0.0
                            if new_order_type in ["SL", "SL-M"]:
                                new_trigger_price = st.number_input("New Trigger Price", min_value=0.05, step=0.05,
                                                                    format="%.2f",
                                                                    value=float(order_data["TriggerPrice"]))
                            else:
                                new_trigger_price = 0.0
                            if st.button("Modify Order"):
                                try:
                                    response = run_async(make_api_request("PUT",
                                                                          f"{BACKEND_URL}/orders/{selected_order_id}/modify",
                                                                          json={
                                                                              "quantity": new_quantity,
                                                                              "order_type": new_order_type,
                                                                              "price": new_price,
                                                                              "trigger_price": new_trigger_price,
                                                                              "validity": "DAY"
                                                                          }, token=st.session_state.access_token))
                                    st.success(f"Order {selected_order_id} modified")
                                    st.toast("Order modified", icon="✅")
                                except Exception as e:
                                    st.error(f"Failed to modify order: {str(e)}")
                                    st.toast("Modification failed", icon="❌")
            else:
                st.info("No orders found")
        except Exception as e:
            st.error(f"Failed to fetch orders: {str(e)}")

    with tabs[1]:
        st.write("##### Scheduled Orders")
        try:
            response = cached_api_request("GET", f"{BACKEND_URL}/orders/{broker}/", token=st.session_state.access_token)
            orders_df = pd.DataFrame(response)
            scheduled_df = orders_df[orders_df["status"] == "PENDING"]
            if not scheduled_df.empty:
                st.dataframe(
                    scheduled_df[["order_id", "trading_symbol", "quantity", "order_type", "schedule_datetime"]])
                selected_order_id = st.selectbox("Select Scheduled Order to Modify/Cancel",
                                                 options=scheduled_df["order_id"])
                if selected_order_id:
                    order = scheduled_df[scheduled_df["order_id"] == selected_order_id].iloc[0]
                    new_quantity = st.number_input("New Quantity", min_value=1, value=int(order["quantity"]))
                    new_price = st.number_input("New Price", min_value=0.0, value=float(order["price"]))
                    new_trigger_price = st.number_input("New Trigger Price", min_value=0.0,
                                                        value=float(order["trigger_price"]))
                    new_schedule_time = st.date_input("New Schedule Time",
                                                      value=pd.to_datetime(order["schedule_datetime"]),
                                                      min_value=datetime.now())
                    new_stop_loss = st.number_input("New Stop-Loss", min_value=0.0, value=float(order["stop_loss"]))
                    new_target = st.number_input("New Target", min_value=0.0, value=float(order["target"]))
                    col_mod, col_can = st.columns(2)
                    with col_mod:
                        if st.button("Modify Scheduled Order"):
                            try:
                                response = run_async(make_api_request("PUT", f"{BACKEND_URL}/orders/{selected_order_id}/modify",
                                                                      json={
                                                                          "quantity": new_quantity,
                                                                          "order_type": order["order_type"],
                                                                          "price": new_price,
                                                                          "trigger_price": new_trigger_price,
                                                                          "validity": "DAY"
                                                                      }, token=st.session_state.access_token))
                                st.success("Scheduled order updated")
                                st.toast("Order updated", icon="✅")
                            except Exception as e:
                                st.error(f"Failed to modify scheduled order: {str(e)}")
                                st.toast("Modification failed", icon="❌")
                    with col_can:
                        if st.button("Cancel Scheduled Order"):
                            try:
                                response = run_async(make_api_request("DELETE", f"{BACKEND_URL}/orders/{selected_order_id}",
                                                                      token=st.session_state.access_token))
                                st.success("Scheduled order cancelled")
                                st.toast("Order cancelled", icon="✅")
                            except Exception as e:
                                st.error(f"Failed to cancel scheduled order: {str(e)}")
                                st.toast("Cancellation failed", icon="❌")
            else:
                st.info("No scheduled orders")
        except Exception as e:
            st.error(f"Failed to fetch scheduled orders: {str(e)}")

    with tabs[2]:
        st.write("##### Auto Orders")
        try:
            response = cached_api_request("GET", f"{BACKEND_URL}/orders/{broker}/", token=st.session_state.access_token)
            auto_orders_df = pd.DataFrame(response)
            if not auto_orders_df.empty:
                auto_orders_df["Symbol"] = auto_orders_df["instrument_token"].map(
                    lambda token: next((k for k, v in instruments.items() if v == token), "Unknown"))
                st.dataframe(
                    auto_orders_df[["auto_order_id", "Symbol", "transaction_type", "risk_per_trade", "stop_loss_type"]])
                selected_auto_order_id = st.selectbox("Select Auto Order to Modify/Delete",
                                                      options=auto_orders_df["auto_order_id"])
                if selected_auto_order_id:
                    order = auto_orders_df[auto_orders_df["auto_order_id"] == selected_auto_order_id].iloc[0]
                    mod_transaction_type = st.radio("Transaction Type", ["BUY", "SELL"],
                                                    index=0 if order["transaction_type"] == "BUY" else 1)
                    mod_order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"],
                                                  index=0 if order["order_type"] == "MARKET" else 1)
                    mod_product_type = st.selectbox("Product Type",
                                                    ["I", "D"] if broker == "Upstox" else ["MIS", "CNC"],
                                                    index=0 if order["product_type"] == "I" else 1)
                    if mod_order_type == "LIMIT":
                        mod_limit_price = st.number_input("Limit Price", min_value=0.05,
                                                          value=float(order["limit_price"]) or get_market_quote(None,
                                                                                                                order[
                                                                                                                    "instrument_token"]).get(
                                                              "ltp", 0), step=0.05)
                    else:
                        mod_limit_price = 0.0
                    mod_risk_per_trade = st.number_input("Risk per Trade (%)", min_value=0.1,
                                                         value=float(order["risk_per_trade"]), step=0.1)
                    mod_stop_loss_type = st.selectbox("Stop Loss Type",
                                                      ["Fixed Amount", "Percentage of Entry", "ATR Based"],
                                                      index=["Fixed Amount", "Percentage of Entry", "ATR Based"].index(
                                                          order["stop_loss_type"]))
                    if mod_stop_loss_type == "Fixed Amount":
                        mod_stop_loss_value = st.number_input("Stop Loss Value (Rs.)", min_value=1.0,
                                                              value=float(order["stop_loss_value"]), step=1.0)
                        mod_target_value = st.number_input("Target Value (Rs.)", min_value=1.0,
                                                           value=float(order["target_value"]), step=1.0)
                    elif mod_stop_loss_type == "Percentage of Entry":
                        mod_stop_loss_percent = st.number_input("Stop Loss (%)", min_value=0.1,
                                                                value=float(order["stop_loss_value"]), step=0.1)
                        mod_target_percent = st.number_input("Target (%)", min_value=0.1,
                                                             value=float(order["target_value"]), step=0.1)
                    else:
                        mod_atr_period = st.number_input("ATR Period", min_value=5, value=int(order["atr_period"]),
                                                         step=1)
                        mod_stop_loss_atr_mult = st.number_input("Stop Loss ATR Multiplier", min_value=0.5,
                                                                 value=float(order["stop_loss_value"]), step=0.5)
                        mod_target_atr_mult = st.number_input("Target ATR Multiplier", min_value=0.5,
                                                              value=float(order["target_value"]), step=0.5)
                    col_mod, col_del = st.columns(2)
                    with col_mod:
                        if st.button("Modify Auto Order"):
                            try:
                                update_data = {
                                    "transaction_type": mod_transaction_type,
                                    "order_type": mod_order_type,
                                    "product_type": mod_product_type,
                                    "limit_price": mod_limit_price,
                                    "risk_per_trade": mod_risk_per_trade,
                                    "stop_loss_type": mod_stop_loss_type,
                                    "stop_loss_value": mod_stop_loss_value if mod_stop_loss_type == "Fixed Amount" else (
                                        mod_stop_loss_percent if mod_stop_loss_type == "Percentage of Entry" else mod_stop_loss_atr_mult),
                                    "target_value": mod_target_value if mod_stop_loss_type == "Fixed Amount" else (
                                        mod_target_percent if mod_stop_loss_type == "Percentage of Entry" else mod_target_atr_mult),
                                    "atr_period": mod_atr_period if mod_stop_loss_type == "ATR Based" else 14
                                }
                                response = run_async(make_api_request("PUT",
                                                                      f"{BACKEND_URL}/auto-orders/{selected_auto_order_id}",
                                                                      json=update_data, token=st.session_state.access_token))
                                st.success(f"Auto order {selected_auto_order_id} modified")
                                st.toast("Auto order modified", icon="✅")
                            except Exception as e:
                                st.error(f"Failed to modify auto order: {str(e)}")
                                st.toast("Modification failed", icon="❌")
                    with col_del:
                        if st.button("Delete Auto Order"):
                            try:
                                response = run_async(make_api_request("DELETE",
                                                                      f"{BACKEND_URL}/auto-orders/{selected_auto_order_id}",
                                                                      token=st.session_state.access_token))
                                st.success(f"Auto order {selected_auto_order_id} deleted")
                                st.toast("Auto order deleted", icon="✅")
                            except Exception as e:
                                st.error(f"Failed to delete auto order: {str(e)}")
                                st.toast("Deletion failed", icon="❌")
            else:
                st.info("No auto orders found")
        except Exception as e:
            st.error(f"Failed to fetch auto orders: {str(e)}")

    with tabs[3]:
        if broker == "Zerodha":
            try:
                response = cached_api_request("GET", f"{BACKEND_URL}/gtt-orders/{broker}/",
                                              token=st.session_state.access_token)
                gtt_df = pd.DataFrame(response)
                if not gtt_df.empty:
                    st.dataframe(gtt_df[["gtt_order_id", "trading_symbol", "quantity", "trigger_price", "status"]])
                    selected_gtt_id = st.selectbox("Select GTT Order to Modify/Delete", options=gtt_df["gtt_order_id"])
                    if selected_gtt_id:
                        gtt_order = gtt_df[gtt_df["gtt_order_id"] == selected_gtt_id].iloc[0]
                        new_gtt_quantity = st.number_input("New Quantity", min_value=1,
                                                           value=int(gtt_order["quantity"]))
                        new_gtt_trigger = st.number_input("New Trigger Price", min_value=0.05,
                                                          value=float(gtt_order["trigger_price"]), step=0.05)
                        new_gtt_limit = st.number_input("New Limit Price", min_value=0.05,
                                                        value=float(gtt_order["limit_price"]), step=0.05)
                        col_mod, col_del = st.columns(2)
                        with col_mod:
                            if st.button("Modify GTT Order"):
                                try:
                                    response = run_async(make_api_request("PUT",
                                                                          f"{BACKEND_URL}/gtt-orders/{broker}/{selected_gtt_id}",
                                                                          json={
                                                                              "instrument_token": gtt_order["instrument_token"],
                                                                              "trading_symbol": gtt_order["trading_symbol"],
                                                                              "transaction_type": gtt_order["transaction_type"],
                                                                              "quantity": new_gtt_quantity,
                                                                              "trigger_type": gtt_order["trigger_type"],
                                                                              "trigger_price": new_gtt_trigger,
                                                                              "limit_price": new_gtt_limit,
                                                                              "last_price": gtt_order["trigger_price"],
                                                                              "broker": broker
                                                                          }, token=st.session_state.access_token))
                                    st.success(f"GTT order {selected_gtt_id} modified")
                                    st.toast("GTT order modified", icon="✅")
                                except Exception as e:
                                    st.error(f"Failed to modify GTT order: {str(e)}")
                                    st.toast("Modification failed", icon="❌")
                        with col_del:
                            if st.button("Delete GTT Order"):
                                try:
                                    response = run_async(make_api_request("DELETE",
                                                                          f"{BACKEND_URL}/gtt-orders/{broker}/{selected_gtt_id}",
                                                                          token=st.session_state.access_token))
                                    st.success(f"GTT order {selected_gtt_id} deleted")
                                    st.toast("GTT order deleted", icon="✅")
                                except Exception as e:
                                    st.error(f"Failed to delete GTT order: {str(e)}")
                                    st.toast("Deletion failed", icon="❌")
                else:
                    st.info("No GTT orders found")
            except Exception as e:
                st.error(f"Failed to fetch GTT orders: {str(e)}")
        else:
            st.info("GTT orders are only supported for Zerodha")

def render_positions(broker):
    st.subheader("Current Positions")
    try:
        response = cached_api_request("GET", f"{BACKEND_URL}/positions/{broker}", token=st.session_state.access_token)
        positions_df = pd.DataFrame(response)
        if not positions_df.empty:
            total_investment = positions_df['AvgPrice'] * positions_df['Quantity']
            total_pnl = positions_df['PnL'].sum()
            total_value = positions_df['LastPrice'] * positions_df['Quantity']
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Investment", f"₹{total_investment.sum():.2f}")
            col2.metric("Total P&L", f"₹{total_pnl:.2f}",
                        f"{(total_pnl / total_investment.sum() * 100 if total_investment.sum() else 0):.2f}%")
            col3.metric("Current Value", f"₹{total_value.sum():.2f}")
            st.dataframe(positions_df, use_container_width=True)
            st.subheader("Position Actions")
            for idx, row in positions_df.iterrows():
                col1, col2 = st.columns([3, 1])
                col1.write(
                    f"{row['Broker']} - {row['Symbol']} - {row['Quantity']} @ {row['AvgPrice']} ({row['Product']})")
                if col2.button("Square Off", key=f"squareoff_{idx}"):
                    try:
                        order_data = {
                            "trading_symbol": row['Symbol'],
                            "instrument_token": row['InstrumentToken'],
                            "quantity": abs(row['Quantity']),
                            "order_type": "MARKET",
                            "transaction_type": "SELL" if row['Quantity'] > 0 else "BUY",
                            "product_type": row['Product'],
                            "is_amo": False,
                            "price": 0.0,
                            "trigger_price": 0.0,
                            "validity": "DAY",
                            "broker": broker
                        }
                        response = run_async(make_api_request("POST", f"{BACKEND_URL}/orders/", json=order_data,
                                                              token=st.session_state.access_token))
                        st.success(f"Square off order placed: {response['order_id']}")
                        st.toast("Square off order placed", icon="✅")
                    except Exception as e:
                        st.error(f"Failed to place square off order: {str(e)}")
                        st.toast("Square off failed", icon="❌")
        else:
            st.info("No positions found")
    except Exception as e:
        st.error(f"Failed to fetch positions: {str(e)}")

async def fetch_trade_dashboard_data(broker, token):
    async with aiohttp.ClientSession() as session:
        positions_task = make_api_request("GET", f"{BACKEND_URL}/positions/{broker}", token=token)
        trade_history_task = make_api_request("GET", f"{BACKEND_URL}/trade-history/{broker}", token=token)
        funds_task = make_api_request("GET", f"{BACKEND_URL}/funds/{broker}", token=token)
        positions, recent_trades, funds_data = await asyncio.gather(positions_task, trade_history_task, funds_task, return_exceptions=True)
        return positions, recent_trades, funds_data

def render_trade_dashboard(broker):
    st.subheader("Trade Dashboard")
    st.write("##### Open Positions")
    try:
        positions, recent_trades, funds_data = run_async(fetch_trade_dashboard_data(broker, st.session_state.access_token))
        if not isinstance(positions, Exception):
            positions_df = pd.DataFrame(positions)
            if not positions_df.empty:
                positions_df["UnrealizedPnL"] = positions_df.apply(
                    lambda row: (get_market_quote(None, row["InstrumentToken"])["ltp"] - row["AvgPrice"]) * row["Quantity"]
                    if row["Quantity"] != 0 else 0, axis=1)
                st.dataframe(positions_df[["InstrumentToken", "Quantity", "AvgPrice", "UnrealizedPnL"]])
                total_unrealized_pnl = positions_df["UnrealizedPnL"].sum()
                st.metric("Total Unrealized P&L", f"₹{total_unrealized_pnl:.2f}")
            else:
                st.info("No open positions")
        else:
            st.error(f"Failed to fetch positions: {str(positions)}")
    except Exception as e:
        st.error(f"Failed to fetch positions: {str(e)}")

    st.write("##### Recent Trades")
    try:
        if not isinstance(recent_trades, Exception):
            recent_trades_df = pd.DataFrame(recent_trades).head(10)
            if not recent_trades_df.empty:
                st.dataframe(recent_trades_df)
                total_realized_pnl = recent_trades_df["pnl"].sum()
                win_rate = len(recent_trades_df[recent_trades_df["pnl"] > 0]) / len(recent_trades_df) * 100
                avg_trade_duration = (pd.to_datetime(recent_trades_df["exit_time"]) - pd.to_datetime(
                    recent_trades_df["entry_time"])).mean().total_seconds() / 60
                st.metric("Total Realized P&L", f"₹{total_realized_pnl:.2f}")
                st.metric("Win Rate", f"{win_rate:.2f}%")
                st.metric("Avg Trade Duration (min)", f"{avg_trade_duration:.2f}")
            else:
                st.info("No recent trades")
        else:
            st.error(f"Failed to fetch trade history: {str(recent_trades)}")
    except Exception as e:
        st.error(f"Failed to fetch trade history: {str(e)}")

    st.write("##### P&L Trend")
    try:
        if not isinstance(recent_trades, Exception):
            all_trades = pd.DataFrame(recent_trades)
            if not all_trades.empty:
                chart = alt.Chart(all_trades).mark_line().encode(
                    x="exit_time:T",
                    y="pnl:Q",
                    tooltip=["exit_time", "pnl"]
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
        else:
            st.error(f"Failed to fetch P&L trend: {str(recent_trades)}")
    except Exception as e:
        st.error(f"Failed to fetch P&L trend: {str(e)}")

    st.write("##### Portfolio Risk Distribution")
    try:
        if not isinstance(funds_data, Exception) and not isinstance(positions, Exception):
            positions_df = pd.DataFrame(positions)
            available_margin = funds_data["equity"]["available"] if funds_data else 1
            if not positions_df.empty:
                positions_df["Risk"] = positions_df["UnrealizedPnL"].abs() / available_margin * 100
                chart = alt.Chart(positions_df).mark_bar().encode(
                    x="InstrumentToken:N",
                    y="Risk:Q",
                    tooltip=["InstrumentToken", "Risk"]
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No risk data to display")
        else:
            st.error(f"Failed to fetch risk distribution: {str(funds_data if isinstance(funds_data, Exception) else positions)}")
    except Exception as e:
        st.error(f"Failed to fetch risk distribution: {str(e)}")

def render_portfolio(broker):
    st.subheader("Portfolio Overview")
    try:
        response = cached_api_request("GET", f"{BACKEND_URL}/portfolio/{broker}", token=st.session_state.access_token)
        portfolio_df = pd.DataFrame(response)
        if not portfolio_df.empty:
            total_value = (portfolio_df["LastPrice"] * portfolio_df["Quantity"]).sum()
            total_buy_value = (portfolio_df["AvgPrice"] * portfolio_df["Quantity"]).sum()
            total_pnl = (portfolio_df["LastPrice"] - portfolio_df["AvgPrice"]) * portfolio_df["Quantity"]
            today_pnl = portfolio_df["DayChange"].sum()
            portfolio_df['BuyValue'] = portfolio_df['AvgPrice'] * portfolio_df['Quantity']
            portfolio_df['MarketValue'] = portfolio_df['LastPrice'] * portfolio_df['Quantity']
            st.dataframe(
                portfolio_df[["Broker", "Symbol", "Exchange", "Quantity", "LastPrice", "AvgPrice", "BuyValue",
                              "MarketValue", "PnL", 'DayChange', 'DayChangePct']]
                .style
                .highlight_max(subset=["BuyValue", "MarketValue", "PnL"], color='#3ee27a')
                .highlight_min(subset=["BuyValue", "MarketValue", "PnL"], color="#ea3c34")
                .format(precision=2, thousands=",")
                .background_gradient(cmap='RdYlGn', subset=['DayChange', 'DayChangePct']),
                hide_index=True,
                use_container_width=True,
            )
            columns = st.columns(4)
            columns[0].metric('Total Buy Value', f'₹{total_buy_value:.2f}')
            columns[1].metric('Total Portfolio Value', f'₹{total_value:.2f}')
            columns[2].metric('Total P&L', f'₹{total_pnl.sum():.2f}',
                              f'{(total_pnl.sum() / total_buy_value * 100):.2f}%')
            columns[3].metric('Day Change', f'₹{today_pnl:.2f}', f'{(today_pnl / total_value * 100):.2f}%')
        else:
            st.info("No holdings found")
    except Exception as e:
        st.error(f"Failed to fetch portfolio: {str(e)}")

    if broker == "Zerodha":
        st.subheader("Mutual Funds Portfolio - Zerodha")
        try:
            response = cached_api_request("GET", f"{BACKEND_URL}/mutual-funds/holdings",
                                          token=st.session_state.access_token)
            mf_portfolio = pd.DataFrame(response)
            if not mf_portfolio.empty:
                mf_portfolio['current_value'] = mf_portfolio['quantity'] * mf_portfolio['last_price']
                mf_portfolio["pnl"] = (mf_portfolio["last_price"] - mf_portfolio["average_price"]) * mf_portfolio[
                    "quantity"]
                mf_portfolio["pnl_pct"] = (mf_portfolio["pnl"] / (
                            mf_portfolio["average_price"] * mf_portfolio["quantity"])) * 100
                st.dataframe(
                    mf_portfolio[["fund", "quantity", "average_price", "last_price", "current_value", "pnl", "pnl_pct"]]
                    .style
                    .highlight_max(subset=["current_value", "pnl"], color='#3ee27a')
                    .highlight_min(subset=["current_value", "pnl"], color="#ea3c34")
                    .format(precision=2, thousands=",")
                    .background_gradient(cmap='RdYlGn', subset=["pnl_pct"]),
                    hide_index=True,
                    use_container_width=True,
                )
                total_mf_buy_value = (mf_portfolio["average_price"] * mf_portfolio["quantity"]).sum()
                total_mf_value = (mf_portfolio["current_value"]).sum()
                total_mf_pnl = (mf_portfolio["pnl"]).sum()
                mf_cols = st.columns(3)
                mf_cols[0].metric("Total Mutual Funds Buy Value", f"₹{total_mf_buy_value:.2f}")
                mf_cols[1].metric("Total Mutual Funds Value", f"₹{total_mf_value:.2f}")
                mf_cols[2].metric("Total Mutual Funds P&L", f"₹{total_mf_pnl:.2f}",
                                  f"{(total_mf_pnl / total_mf_buy_value * 100):.2f}%")
            else:
                st.info("No mutual funds found")
        except Exception as e:
            st.error(f"Failed to fetch mutual fund holdings: {str(e)}")

def render_mutual_funds(broker):
    if broker != "Zerodha":
        st.info("Mutual Funds are only supported for Zerodha")
        return
    st.subheader("Mutual Funds - Zerodha")
    tabs = st.tabs(["Instruments", "Orders", "Holdings", "SIPs"])
    with tabs[0]:
        try:
            response = cached_api_request("GET", f"{BACKEND_URL}/mutual-funds/instruments",
                                          token=st.session_state.access_token)
            instruments_df = pd.DataFrame(response)
            st.dataframe(instruments_df[["tradingsymbol", "name", "minimum_investment"]])
        except Exception as e:
            st.error(f"Error fetching instruments: {str(e)}")
    with tabs[1]:
        with st.form("mf_order_form"):
            scheme_code = st.text_input("Scheme Code", help="Enter the mutual fund scheme code")
            amount = st.number_input("Amount", min_value=100.0, step=100.0, help="Investment amount")
            transaction_type = st.selectbox("Transaction Type", ["BUY", "SELL"])
            if st.form_submit_button("Place MF Order"):
                try:
                    response = run_async(make_api_request("POST", f"{BACKEND_URL}/mutual-funds/orders", json={
                        "scheme_code": scheme_code,
                        "amount": amount,
                        "transaction_type": transaction_type
                    }, token=st.session_state.access_token))
                    st.success(f"Mutual Fund Order placed: {response['order_id']}")
                    st.toast("Order placed", icon="✅")
                except Exception as e:
                    st.error(f"Error placing order: {str(e)}")
                    st.toast("Order failed", icon="❌")
    with tabs[2]:
        try:
            response = cached_api_request("GET", f"{BACKEND_URL}/mutual-funds/holdings",
                                          token=st.session_state.access_token)
            holdings_df = pd.DataFrame(response)
            if not holdings_df.empty:
                st.dataframe(holdings_df)
            else:
                st.info("No mutual fund holdings")
        except Exception as e:
            st.error(f"Error fetching holdings: {str(e)}")
    with tabs[3]:
        try:
            response = cached_api_request("GET", f"{BACKEND_URL}/mutual-funds/sips", token=st.session_state.access_token)
            sips_df = pd.DataFrame(response)
            if not sips_df.empty:
                st.dataframe(sips_df)
                sip_id = st.selectbox("Select SIP to Cancel", sips_df["sip_id"], key="cancel_sip")
                if st.button("Cancel SIP"):
                    try:
                        response = run_async(make_api_request("DELETE", f"{BACKEND_URL}/mutual-funds/sips/{sip_id}",
                                                              token=st.session_state.access_token))
                        st.success(f"SIP {sip_id} cancelled")
                        st.toast("SIP cancelled", icon="✅")
                    except Exception as e:
                        st.error(f"Failed to cancel SIP: {str(e)}")
                        st.toast("Cancellation failed", icon="❌")
            else:
                st.info("No SIPs found")
        except Exception as e:
            st.error(f"Error fetching SIPs: {str(e)}")
        with st.form("sip_form"):
            scheme_code = st.text_input("Scheme Code")
            amount = st.number_input("Amount", min_value=100.0, step=100.0)
            frequency = st.selectbox("Frequency", ["daily", "weekly", "monthly"])
            start_date = st.date_input("Start Date", value=datetime.now())
            if st.form_submit_button("Create SIP"):
                try:
                    response = run_async(make_api_request("POST", f"{BACKEND_URL}/mutual-funds/sips", json={
                        "scheme_code": scheme_code,
                        "amount": amount,
                        "frequency": frequency,
                        "start_date": start_date.isoformat()
                    }, token=st.session_state.access_token))
                    st.success(f"SIP created: {response['sip_id']}")
                    st.toast("SIP created", icon="✅")
                except Exception as e:
                    st.error(f"Error creating SIP: {str(e)}")
                    st.toast("SIP creation failed", icon="❌")

def render_analytics(broker):
    st.subheader("Trade Analytics & Live Feed")
    analytics_cols = st.columns(5)
    stock_symbol = analytics_cols[0].selectbox("Select Symbol", options=instruments.keys(), key='symbol_analysis')
    instrument_token = instruments.get(stock_symbol)
    timeframe = analytics_cols[1].selectbox("Timeframe", ["1minute", "day", "week", "month", "30minute"], index=1)
    ema_period = analytics_cols[2].number_input("EMA Period", min_value=5, value=20, max_value=200)
    lr_period = analytics_cols[3].number_input("LR Period", min_value=5, value=20, max_value=200)
    rsi_period = analytics_cols[4].number_input("RSI Period", min_value=5, value=14, max_value=50)
    show_columns = st.columns(5)
    show_sr = show_columns[0].checkbox("Show Support & Resistance")
    show_trend = show_columns[1].checkbox("Show Trend Lines")
    show_ema = show_columns[2].checkbox("Show EMA")
    show_lr = show_columns[3].checkbox("Show Linear Regression")
    show_rsi = show_columns[4].checkbox("Show RSI")
    try:
        response = cached_historical_data("GET",
                                          f"{BACKEND_URL}/analytics/{instrument_token}?timeframe={timeframe}&ema_period={ema_period}&rsi_period={rsi_period}&lr_period={lr_period}&stochastic_k=14&stochastic_d=3",
                                          token=st.session_state.access_token)
        if "error" in response:
            st.error(response["error"])
            return
        df = pd.DataFrame(response["candles"])
        df["color"] = np.where(df["open"] > df["close"], "rgba(239,83,80,0.9)", "rgba(38,166,154,0.9)")
        chart_options = [{
            "width": "100%",
            "height": 600,
            "layout": {
                "background": {"type": "solid", "color": "white"},
                "textColor": "black"
            },
            "grid": {
                "vertLines": {"color": "rgba(197, 203, 206, 0.5)"},
                "horzLines": {"color": "rgba(197, 203, 206, 0.5)"}
            },
            "crosshair": {"mode": 1},
            "priceScale": {
                "borderColor": "rgba(197, 203, 206, 0.8)",
                "mode": 1
            },
            "timeScale": {
                "borderColor": "rgba(197, 203, 206, 0.8)",
                "barSpacing": 15,
                "timeVisible": True
            },
            "rightPriceScale": {
                "visible": True,
                "borderColor": "rgba(197, 203, 206, 0.8)"
            }
        }]
        series = [{
            "type": "Candlestick",
            "data": json.loads(df[["timestamp", "open", "high", "low", "close", "color"]].to_json(orient="records")),
            "options": {
                "upColor": "rgba(38,166,154,0.9)",
                "downColor": "rgba(239,83,80,0.9)",
                "borderVisible": False,
                "wickUpColor": "rgba(38,166,154,0.9)",
                "wickDownColor": "rgba(239,83,80,0.9)"
            }
        }]
        if show_ema and "ema" in response["indicators"]:
            series.append({
                "type": "Line",
                "data": [{"time": row["timestamp"], "value": val} for row, val in
                         zip(df.to_dict(orient="records"), response["indicators"]["ema"])],
                "options": {"color": "blue", "lineWidth": 2}
            })
        if show_lr and "lr" in response["indicators"]:
            series.append({
                "type": "Line",
                "data": [{"time": row["timestamp"], "value": val} for row, val in
                         zip(df.to_dict(orient="records"), response["indicators"]["lr"])],
                "options": {"color": "purple", "lineWidth": 2}
            })
        if show_rsi and "rsi" in response["indicators"]:
            series.append({
                "type": "Area",
                "data": [{"time": row["timestamp"], "value": val} for row, val in
                         zip(df.to_dict(orient="records"), response["indicators"]["rsi"])],
                "options": {
                    "color": "purple",
                    "lineColor": "purple",
                    "topColor": "rgba(128,0,128,0.2)",
                    "bottomColor": "rgba(128,0,128,0)",
                    "priceScaleId": "rsi",
                    "priceFormat": {"type": "custom", "minMove": 0.01, "precision": 2},
                    "scaleMargins": {"top": 0.7, "bottom": 0.1}
                }
            })
            chart_options[0]["priceScales"] = [
                {"id": "rsi", "mode": 0, "autoScale": True, "minimum": 0, "maximum": 100,
                 "borderColor": "rgba(197, 203, 206, 0.8)"}
            ]
        if show_sr:
            series.append({
                "type": "Line",
                "data": [{"time": row["timestamp"], "value": response["support"]} for row in
                         df.to_dict(orient="records")],
                "options": {"color": "green", "lineStyle": 2, "lineWidth": 1}
            })
            series.append({
                "type": "Line",
                "data": [{"time": row["timestamp"], "value": response["resistance"]} for row in
                         df.to_dict(orient="records")],
                "options": {"color": "red", "lineStyle": 2, "lineWidth": 1}
            })
        st.write("### Candlestick Chart")
        st.markdown("```chartjs\n" + json.dumps({"options": chart_options, "series": series}) + "\n```")
        st.write("### OHLC Data")
        st.dataframe(df[["timestamp", "open", "high", "low", "close", "volume"]])
    except Exception as e:
        st.error(f"Error fetching analytics data: {str(e)}")

def render_algo_trading(broker):
    st.subheader("Algorithmic Trading")
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox("Select Strategy", ["MACD Crossover", "Bollinger Bands", "RSI Oversold/Overbought",
                                                    "Stochastic Oscillator", "Support/Resistance Breakout"])
        stock_symbol = st.selectbox("Select Symbol", options=instruments.keys(), key='symbol_algo')
        instrument_token = instruments.get(stock_symbol)
        quantity = st.number_input("Quantity", min_value=1, value=1)
        if strategy == "MACD Crossover":
            fast_period = st.number_input("Fast EMA Period", min_value=3, value=12)
            slow_period = st.number_input("Slow EMA Period", min_value=5, value=26)
            signal_period = st.number_input("Signal Period", min_value=3, value=9)
            params = {"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period}
        elif strategy == "Bollinger Bands":
            bb_period = st.number_input("Bollinger Band Period", min_value=5, value=20)
            num_std = st.number_input("Number of Standard Deviations", min_value=1.0, value=2.0)
            params = {"period": bb_period, "num_std": num_std}
        elif strategy == "RSI Oversold/Overbought":
            rsi_period = st.number_input("RSI Period", min_value=5, value=14)
            overbought = st.number_input("Overbought Level", min_value=50, max_value=100, value=70)
            oversold = st.number_input("Oversold Level", min_value=0, max_value=50, value=30)
            params = {"period": rsi_period, "overbought": overbought, "oversold": oversold}
        elif strategy == "Stochastic Oscillator":
            k_period = st.number_input("K Period", min_value=5, value=14)
            d_period = st.number_input("D Period", min_value=3, value=3)
            params = {"k_period": k_period, "d_period": d_period}
        elif strategy == "Support/Resistance Breakout":
            lookback = st.number_input("Lookback Period", min_value=5, value=20)
            params = {"lookback": lookback}
    with col2:
        st.subheader("Risk Management")
        stop_loss = st.number_input("Stop Loss (%)", min_value=0.1, value=1.0, max_value=10.0)
        take_profit = st.number_input("Take Profit (%)", min_value=0.1, value=2.0, max_value=20.0)
        st.subheader("Execution Settings")
        execution_type = st.radio("Execution Type", ["Manual", "Automatic"])
        if execution_type == "Automatic":
            interval = st.number_input("Check Interval (minutes)", min_value=1, value=5)
            start_hour = st.number_input("Market Start Hour", min_value=0, max_value=23, value=9)
            start_min = st.number_input("Market Start Minute", min_value=0, max_value=59, value=15)
            end_hour = st.number_input("Market End Hour", min_value=0, max_value=23, value=15)
            end_min = st.number_input("Market End Minute", min_value=0, max_value=59, value=30)
    st.subheader("Strategy Description")
    if strategy == "MACD Crossover":
        st.markdown(
            "**MACD Crossover Strategy**: Generates buy signals when the MACD line crosses above the signal line and sell signals when it crosses below.")
    elif strategy == "Bollinger Bands":
        st.markdown(
            "**Bollinger Bands Strategy**: Generates buy signals when the price touches the lower band and sell signals when it touches the upper band.")
    elif strategy == "RSI Oversold/Overbought":
        st.markdown(
            "**RSI Strategy**: Generates buy signals when RSI is below the oversold level and sell signals when above the overbought level.")
    elif strategy == "Stochastic Oscillator":
        st.markdown(
            "**Stochastic Oscillator Strategy**: Generates buy signals when %K crosses above %D in oversold regions and sell signals when it crosses below in overbought regions.")
    elif strategy == "Support/Resistance Breakout":
        st.markdown(
            "**Support/Resistance Breakout Strategy**: Generates buy signals on resistance breakout and sell signals on support breakdown.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check Current Signal"):
            with st.spinner("Analyzing market data..."):
                try:
                    response = run_async(make_api_request("POST", f"{BACKEND_URL}/algo-trading/execute", json={
                        "strategy": strategy,
                        "instrument_token": instrument_token,
                        "quantity": quantity,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "broker": broker
                    }, token=st.session_state.access_token))
                    if "signal" in response["message"]:
                        st.success(f"Current Signal: {response['message']}")
                    else:
                        st.info("No signal detected")
                except Exception as e:
                    st.error(f"Failed to check signal: {str(e)}")
                    st.toast("Signal check failed", icon="❌")
    with col2:
        if execution_type == "Manual":
            if st.button("Execute Strategy Now"):
                with st.spinner("Executing strategy..."):
                    try:
                        response = run_async(make_api_request("POST", f"{BACKEND_URL}/algo-trading/execute", json={
                            "strategy": strategy,
                            "instrument_token": instrument_token,
                            "quantity": quantity,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "broker": broker
                        }, token=st.session_state.access_token))
                        st.success(response["message"])
                        st.toast("Strategy executed", icon="✅")
                    except Exception as e:
                        st.error(f"Failed to execute strategy: {str(e)}")
                        st.toast("Execution failed", icon="❌")
        else:
            if st.button("Start Automated Trading"):
                try:
                    response = run_async(make_api_request("POST", f"{BACKEND_URL}/algo-trading/schedule", json={
                        "strategy": strategy,
                        "instrument_token": instrument_token,
                        "quantity": quantity,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "broker": broker,
                        "interval_minutes": interval
                    }, token=st.session_state.access_token))
                    st.success(response["message"])
                    st.toast("Automated trading started", icon="✅")
                except Exception as e:
                    st.error(f"Failed to start automated trading: {str(e)}")
                    st.toast("Automated trading failed", icon="❌")

def render_strategy_backtesting(broker):
    st.subheader("Strategy Backtesting")
    with st.form("backtest_form"):
        # Symbol and timeframe selection
        col1, col2 = st.columns(2)
        with col1:
            stock_symbol = st.selectbox("Select Symbol", options=instruments.keys(), key='symbol_backtest')
        with col2:
            timeframe = st.selectbox("Timeframe", ["day", "week", "1minute", "5minute", "30minute"], index=0)
        instrument_token = instruments.get(stock_symbol)

        # Strategy selection
        strategy = st.selectbox("Select Strategy to Backtest", ["Short Sell Optimization", "MACD Crossover", "Bollinger Bands", "RSI Oversold/Overbought"])

        # Strategy-specific parameters
        params = {}
        if strategy == "Short Sell Optimization":
            stocks = ['GOLDBEES', 'JUNIORBEES', 'ICICIB22', 'CPSEETF', 'ITBEES', 'MID150BEES', 'MON100', 'MAFANG', 'HDFCSML250']
            selected_stocks = st.multiselect("Select Stocks to Backtest", stocks, default=stocks[:2])
            initial_investment = st.number_input("Initial Investment (Rs.)", min_value=1000, value=50000, step=1000)
            stop_loss_atr_mult_range = st.slider("Stop Loss ATR Multiplier", 1.0, 4.0, (1.5, 2.5), step=0.5)
            target_atr_mult_range = st.slider("Target ATR Multiplier", 1.0, 7.0, (4.0, 6.0), step=0.5)
            params = {
                "initial_investment": initial_investment,
                "stop_loss_atr_mult_range": [stop_loss_atr_mult_range[0], stop_loss_atr_mult_range[1]],
                "target_atr_mult_range": [target_atr_mult_range[0], target_atr_mult_range[1]]
            }
        elif strategy == "MACD Crossover":
            col1, col2, col3 = st.columns(3)
            with col1:
                fast_period = st.number_input("Fast EMA Period", min_value=3, value=12)
            with col2:
                slow_period = st.number_input("Slow EMA Period", min_value=5, value=26)
            with col3:
                signal_period = st.number_input("Signal Period", min_value=3, value=9)
            params = {"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period}
        elif strategy == "Bollinger Bands":
            col1, col2 = st.columns(2)
            with col1:
                bb_period = st.number_input("Bollinger Band Period", min_value=5, value=20)
            with col2:
                num_std = st.number_input("Number of Standard Deviations", min_value=1.0, value=2.0, step=0.5)
            params = {"period": bb_period, "num_std": num_std}
        elif strategy == "RSI Oversold/Overbought":
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_period = st.number_input("RSI Period", min_value=5, value=14)
            with col2:
                overbought = st.number_input("Overbought Level", min_value=50, max_value=100, value=70)
            with col3:
                oversold = st.number_input("Oversold Level", min_value=0, max_value=50, value=30)
            params = {"period": rsi_period, "overbought": overbought, "oversold": oversold}

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))

        # Submit button
        if st.form_submit_button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    if strategy == "Short Sell Optimization":
                        results = []
                        for stock in selected_stocks:
                            response = run_async(make_api_request("POST", f"{BACKEND_URL}/algo-trading/backtest", json={
                                "instrument_token": instruments.get(stock, instrument_token),
                                "timeframe": timeframe,
                                "strategy": strategy,
                                "params": params,
                                "start_date": start_date.isoformat(),
                                "end_date": end_date.isoformat()
                            }, token=st.session_state.access_token))
                            if "error" in response:
                                st.error(f"Backtest failed for {stock}: {response['error']}")
                                continue
                            results.append((stock, response))
                        for stock, response in results:
                            st.write(f"##### Optimized Results for {stock}")
                            st.write(f"**Initial Investment:** ₹{response['InitialInvestment']:.2f}")
                            st.write(f"**Final Portfolio Value:** ₹{response['FinalPortfolioValue']:.2f}")
                            st.write(f"**Total Profit:** ₹{response['TotalProfit']:.2f}")
                            st.write(f"**Win Rate:** {response['WinRate']:.2f}%")
                            st.write(f"**Total Trades:** {response['TotalTrades']}")
                            st.write(f"**Stop Loss ATR Multiplier:** {response['StopLossATRMult']:.1f}x")
                            st.write(f"**Target ATR Multiplier:** {response['TargetATRMult']:.1f}x")
                            tradebook_df = pd.DataFrame(response["Tradebook"])
                            if not tradebook_df.empty:
                                st.write("##### Tradebook")
                                st.dataframe(tradebook_df)
                                # Plot portfolio value
                                chart = alt.Chart(tradebook_df).mark_line().encode(
                                    x="Date:T",
                                    y="PortfolioValue:Q",
                                    tooltip=["Date", "PortfolioValue", "Profit"]
                                ).interactive()
                                st.altair_chart(chart, use_container_width=True)
                                # Download tradebook
                                csv = tradebook_df.to_csv(index=False)
                                st.download_button(
                                    label=f"Download Tradebook for {stock}",
                                    data=csv,
                                    file_name=f"tradebook_{stock}.csv",
                                    mime="text/csv"
                                )
                    else:
                        response = run_async(make_api_request("POST", f"{BACKEND_URL}/algo-trading/backtest", json={
                            "instrument_token": instrument_token,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "params": params,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat()
                        }, token=st.session_state.access_token))
                        if "error" in response:
                            st.error(f"Backtest failed: {response['error']}")
                        else:
                            st.write(f"##### Backtest Results for {stock_symbol} ({strategy})")
                            st.write(f"**Total Trades:** {response['total_trades']}")
                            st.write(f"**Win Rate:** {response['win_rate']:.2f}%")
                            st.write(f"**Total P&L:** ₹{response['total_pnl']:.2f}")
                            backtest_df = pd.DataFrame(response["data"])
                            if not backtest_df.empty:
                                st.write("##### Trade Data")
                                st.dataframe(backtest_df[["timestamp", "open", "close", "signal", "pnl", "cumulative_pnl"]])
                                # Plot cumulative P&L
                                chart = alt.Chart(backtest_df).mark_line().encode(
                                    x="timestamp:T",
                                    y="cumulative_pnl:Q",
                                    tooltip=["timestamp", "cumulative_pnl", "signal"]
                                ).interactive()
                                st.altair_chart(chart, use_container_width=True)
                                # Download results
                                csv = backtest_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Backtest Results",
                                    data=csv,
                                    file_name=f"backtest_{stock_symbol}_{strategy}.csv",
                                    mime="text/csv"
                                )
                    st.toast("Backtest completed successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
                    st.toast("Backtest failed", icon="❌")

# Main application logic
def main():
    if 'access_token' not in st.session_state:
        render_login_page()
    else:
        render_top_nav()
        selected_page, broker = render_sidebar()
        page_functions = {
            "Dashboard": render_dashboard,
            "Get Token": render_get_token,
            "Order Management": render_order_management,
            "Order Book": render_order_book,
            "Positions": render_positions,
            "Trade Dashboard": render_trade_dashboard,
            "Portfolio": render_portfolio,
            "Mutual Funds": render_mutual_funds,
            "Analytics": render_analytics,
            "Algo Trading": render_algo_trading,
            "Strategy Backtesting": render_strategy_backtesting
        }
        if selected_page in page_functions:
            page_functions[selected_page](broker)
        else:
            st.error("Invalid page selected")

if __name__ == "__main__":
    main()