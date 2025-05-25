import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from tenacity import retry, stop_after_attempt, wait_exponential
from nsepython import nse_get_index_quote
from streamlit_lightweight_charts import renderLightweightCharts
import json
import numpy as np
from ta.momentum import RSIIndicator
import logging
import sys
# Add the project_root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from common_utils.upstox_utils import fetch_instruments
from backend.app.database import get_db

# Configure logging for frontend
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration and styling
st.set_page_config(page_title="Stock Trading Dashboard", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-top: -5em;}
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

# Static index constituents (fallback)
INDEX_CONSTITUENTS = {
    "NIFTY 50": [
        "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "HINDUNILVR",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "M&M", "ULTRACEMCO", "NESTLEIND"
    ],  # Sample top 20
    "NIFTY NEXT 50": [
        "ADANIENT", "ADANIGREEN", "AMBUJACEM", "BAJAJHLDNG", "BERGEPAINT", "BOSCHLTD", "CANBK", "CHOLAFIN", "DABUR",
        "DLF",
        "GODREJCP", "HAVELLS", "HDFCAMC", "ICICIGI", "INDIGO", "IOC", "JINDALSTEL", "JUBLFOOD", "MCDOWELL-N", "NAUKRI"
    ]  # Sample top 20
}


# Cache instrument fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_index_instruments(index_name):
    try:
        symbols = INDEX_CONSTITUENTS.get(index_name, [])
        instruments = {symbol: f"NSE:{symbol}" for symbol in symbols}
        return instruments
    except Exception as e:
        st.error(f"Failed to fetch instruments for {index_name}: {str(e)}")
        return {}


# Retry decorator for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def make_api_request(method, url, token=None, **kwargs):
    if "auth/login" in url:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
    else:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.request(method, url, headers=headers, **kwargs)
    if response.status_code == 401:
        st.session_state.pop("access_token", None)
        st.error("Session expired. Please log in again.")
        st.rerun()
    response.raise_for_status()
    return response.json()


# Initialize instruments
if 'instruments_data' not in st.session_state:
    try:
        st.session_state['instruments_data'] = fetch_instruments()
        logger.info("Instruments data loaded")
    except Exception as e:
        logger.error(f"Failed to load instruments: {str(e)}")
        st.error(f"Failed to load instruments: {str(e)}")


instruments = st.session_state.get('instruments_data', {})
my_stocks = st.session_state.get("my_stocks", {})

# Login page
def render_login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("Stock Trading Dashboard")
    tabs = st.tabs(["Login", "Register"])

    with tabs[0]:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                try:
                    response = make_api_request("POST", f"{BACKEND_URL}/auth/login", data={
                        "username": username,
                        "password": password,
                        "grant_type": "password"
                    })
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
                        response = make_api_request("POST", f"{BACKEND_URL}/auth/register", json={
                            "email": new_username,
                            "password": new_password,
                            "upstox_api_key": upstox_api_key,
                            "upstox_api_secret": upstox_api_secret,
                            "zerodha_api_key": zerodha_api_key,
                            "zerodha_api_secret": zerodha_api_secret
                        })
                        st.success("Registration successful! Please log in.")
                        st.toast("Registration successful", icon="✅")
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
                        st.toast("Registration failed", icon="❌")
    st.markdown('</div>', unsafe_allow_html=True)


# Sidebar rendering
def render_sidebar():
    with st.sidebar:
        st.title("Trading Dashboard")
        theme = st.selectbox("Theme", ["Light", "Dark"])
        if theme == "Dark":
            st.markdown("""
                        <style>
                        .main { background-color: #1e1e1e; color: #ffffff; }
                        .sidebar .sidebar-content { background-color: #2c2c2c; }
                        .metric-box { background-color: #1a3c6e; }
                        </style>
                    """, unsafe_allow_html=True)
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Get Token", "Order Management", "Order Book", "Positions", "Trade Dashboard",
                     "Portfolio", "Mutual Funds", "Market Data", "Order Modification"],
            icons=["house", "key", "cart", "book", "bar-chart", "clock-history", "briefcase", "fund", "graph-up",
                   "pencil"],
            default_index=0,
        )
        broker = st.selectbox("Select Broker", ["Upstox", "Zerodha"], key="select_broker")
        try:
            response = make_api_request("GET", f"{BACKEND_URL}/profile/{broker}/", token=st.session_state.access_token)
            st.success(f"{broker} Connected: {response['name']}")
        except:
            st.error(f"{broker} Not Connected")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
    return selected, broker


# Dashboard page
def render_dashboard(broker):
    st.subheader(f"{broker} Trading Dashboard")
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            funds_data = make_api_request("GET", f"{BACKEND_URL}/funds/{broker}/", token=st.session_state.access_token)
            st.markdown(f'<div class="metric-box">Available Funds: ₹{funds_data["equity"]["available"]}</div>',
                        unsafe_allow_html=True)
        except:
            st.markdown('<div class="metric-box">Funds: N/A</div>', unsafe_allow_html=True)

    with col2:
        try:
            portfolio = make_api_request("GET", f"{BACKEND_URL}/portfolio/{broker}/",
                                         token=st.session_state.access_token)
            total_value = sum(holding["Quantity"] * holding["Last Price"] for holding in portfolio)
            st.markdown(f'<div class="metric-box">Portfolio Value: ₹{total_value:.2f}</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="metric-box">Portfolio: N/A</div>', unsafe_allow_html=True)

    with col3:
        try:
            positions = make_api_request("GET", f"{BACKEND_URL}/positions/{broker}/",
                                         token=st.session_state.access_token)
            open_positions = len([pos for pos in positions if pos["Quantity"] != 0])
            st.markdown(f'<div class="metric-box">Open Positions: {open_positions}</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="metric-box">Positions: N/A</div>', unsafe_allow_html=True)

    st.write("### Market Watch")
    index_name = st.selectbox("Select Index", ["NIFTY 50", "NIFTY NEXT 50"], key="dashboard_index")
    instruments = fetch_index_instruments(index_name)
    instrument = st.selectbox("Select Instrument", list(instruments.keys()), key="dashboard_instrument")
    try:
        ltp_data = make_api_request("GET", f"{BACKEND_URL}/ltp/{broker}/?instruments={instruments[instrument]}",
                                    token=st.session_state.access_token)
        st.markdown(f'<div class="metric-box">{instrument} LTP: ₹{ltp_data[0]["last_price"]}</div>',
                    unsafe_allow_html=True)
    except:
        st.error("Failed to fetch LTP")


# Order Management page
def render_order_management(broker):
    st.subheader(f"Order Management - {broker}")
    tabs = st.tabs(["Regular Orders", "Scheduled Orders", "GTT Orders"])

    with tabs[0]:
        main_cols = st.columns(2)
        index_name = main_cols[0].selectbox("Select Index", ["NIFTY 50", "NIFTY NEXT 50"], key="order_index")
        schedule_order = main_cols[1].checkbox("Schedule Order")

        with st.form("regular_order_form"):

            stock_symbol = st.selectbox("Symbol", options=instruments.keys(), key="order_symbol")
            instrument_token = instruments.get(stock_symbol)
            cols = st.columns(4)
            quantity = cols[0].number_input("Quantity", min_value=1, value=1)
            order_type = cols[1].selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"])
            transaction_type = cols[2].radio("Transaction", ["BUY", "SELL"], horizontal=True)
            product_type = cols[3].radio("Product", ['I', 'D'] if broker == "Upstox" else ['MIS', 'CNC'])
            price = st.number_input("Price", min_value=0.0, value=0.0, disabled=order_type != "LIMIT")
            trigger_price = st.number_input("Trigger Price", min_value=0.0, value=0.0,
                                            disabled=order_type not in ["SL", "SL-M"])
            stop_loss = st.number_input("Stop-Loss", min_value=0.0, value=0.0)
            target = st.number_input("Target", min_value=0.0, value=0.0)
            amo_order = st.checkbox("AMO Order")
            schedule_datetime = None
            if schedule_order:
                schedule_cols = st.columns(2)
                schedule_date = schedule_cols[0].date_input("Schedule Date", value=datetime.now())
                schedule_time = schedule_cols[1].time_input("Schedule Time", value=datetime.now().time(), step=60)
                schedule_datetime = datetime.combine(schedule_date, schedule_time)

            if st.form_submit_button("Place Order"):
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
                try:
                    with st.spinner("Placing order..."):
                        response = make_api_request("POST",
                                                    f"{BACKEND_URL}/{'scheduled-orders' if schedule_order else 'orders'}/",
                                                    json=order_data, token=st.session_state.access_token)
                    st.success(f"Order {'scheduled' if schedule_order else 'placed'}: {response['order_id'] if broker == 'Upstox' else response}")
                    st.toast("Order processed successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Failed to {'schedule' if schedule_order else 'place'} order: {str(e)}")
                    st.toast("Order failed", icon="❌")

    with tabs[1]:
        try:
            response = make_api_request("GET", f"{BACKEND_URL}/orders/{broker}/", token=st.session_state.access_token)
            orders_df = pd.DataFrame(response)
            scheduled_df = orders_df[orders_df["status"] == "PENDING"]
            if not scheduled_df.empty:
                st.dataframe(
                    scheduled_df[["order_id", "trading_symbol", "quantity", "order_type", "schedule_datetime"]])
                order_id = st.selectbox("Select Order to Cancel", scheduled_df["order_id"], key="cancel_schedule")
                if st.button("Cancel Scheduled Order"):
                    try:
                        response = make_api_request("DELETE", f"{BACKEND_URL}/orders/{order_id}",
                                                    token=st.session_state.access_token)
                        st.success(f"Order {order_id} cancelled")
                        st.toast("Order cancelled", icon="✅")
                    except:
                        st.error("Failed to cancel order")
                        st.toast("Cancellation failed", icon="❌")
            else:
                st.info("No scheduled orders")
        except:
            st.error("Failed to fetch scheduled orders")

    with tabs[2]:
        if broker == "Zerodha":
            try:
                response = make_api_request("GET", f"{BACKEND_URL}/gtt-orders/{broker}/",
                                            token=st.session_state.access_token)
                gtt_df = pd.DataFrame(response)
                if not gtt_df.empty:
                    st.dataframe(gtt_df[["gtt_order_id", "trading_symbol", "quantity", "trigger_price", "status"]])
                    gtt_id = st.selectbox("Select GTT Order", gtt_df["gtt_order_id"], key="delete_gtt")
                    if st.button("Delete GTT Order"):
                        try:
                            response = make_api_request("DELETE", f"{BACKEND_URL}/gtt-orders/{broker}/{gtt_id}",
                                                        token=st.session_state.access_token)
                            st.success(f"GTT Order {gtt_id} deleted")
                            st.toast("GTT Order deleted", icon="✅")
                        except:
                            st.error("Failed to delete GTT order")
                            st.toast("Deletion failed", icon="❌")
                else:
                    st.info("No GTT orders")
            except:
                st.error("Failed to fetch GTT orders")
        else:
            st.info("GTT orders are only supported for Zerodha")


# Market Data page
def render_market_data(broker):
    st.subheader("Market Data")
    index_name = st.selectbox("Select Index", ["NIFTY 50", "NIFTY NEXT 50"], key="market_index")
    instruments = fetch_index_instruments(index_name)
    instrument = st.selectbox("Select Instrument", list(instruments.keys()), key="market_instrument")
    interval = st.selectbox("Interval", ["minute", "5minute", "day"])
    date_range = st.date_input("Date Range", [datetime.now() - timedelta(days=7), datetime.now()])

    # Chart customization controls
    st.write("### Chart Settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_sma = st.checkbox("Show SMA", value=True)
        sma_period = st.number_input("SMA Period", min_value=5, max_value=50, value=20, step=1)
    with col2:
        show_rsi = st.checkbox("Show RSI", value=True)
        rsi_period = st.number_input("RSI Period", min_value=5, max_value=50, value=14, step=1)
    with col3:
        show_volume = st.checkbox("Show Volume", value=True)
    with col4:
        show_crosshair = st.checkbox("Show Crosshair", value=True)

    try:
        response = make_api_request(
            "GET",
            f"{BACKEND_URL}/historical-data/{broker}/?instrument={instruments[instrument]}&from_date={date_range[0].isoformat()}&to_date={date_range[1].isoformat()}&interval={interval}",
            token=st.session_state.access_token
        )
        data = response["data"]
        if data:
            # Prepare candlestick data
            df = pd.DataFrame([
                {
                    "time": d["timestamp"].split("T")[0],  # Format as YYYY-MM-DD
                    "open": d["open"],
                    "high": d["high"],
                    "low": d["low"],
                    "close": d["close"],
                    "volume": d["volume"]
                } for d in data
            ])
            df["color"] = np.where(df["open"] > df["close"], "rgba(239,83,80,0.9)",
                                   "rgba(38,166,154,0.9)")  # Bear: red, Bull: green

            # Calculate SMA
            if show_sma:
                df["sma"] = df["close"].rolling(window=sma_period).mean()
                sma_data = [
                    {"time": row["time"], "value": row["sma"]} for _, row in df.iterrows() if not pd.isna(row["sma"])
                ]

            # Calculate RSI
            if show_rsi:
                rsi_indicator = RSIIndicator(df["close"], window=rsi_period)
                df["rsi"] = rsi_indicator.rsi()
                rsi_data = [
                    {"time": row["time"], "value": row["rsi"]} for _, row in df.iterrows() if not pd.isna(row["rsi"])
                ]

            # Prepare volume data
            if show_volume:
                volume_data = [
                    {
                        "time": row["time"],
                        "value": row["volume"],
                        "color": "rgba(0,150,136,0.8)" if row["open"] <= row["close"] else "rgba(255,82,82,0.8)"
                    } for _, row in df.iterrows()
                ]

            # Chart options
            chart_options = [{
                "width": 800,
                "height": 600,  # Increased height to accommodate RSI pane
                "layout": {
                    "background": {"type": "solid", "color": "white"},
                    "textColor": "black"
                },
                "grid": {
                    "vertLines": {"color": "rgba(197, 203, 206, 0.5)"},
                    "horzLines": {"color": "rgba(197, 203, 206, 0.5)"}
                },
                "crosshair": {
                    "mode": show_crosshair  # 0: none, 1: normal
                },
                "priceScale": {
                    "borderColor": "rgba(197, 203, 206, 0.8)",
                    "mode": 1  # Normal scale
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

            # Series data
            series = [
                {
                    "type": "Candlestick",
                    "data": json.loads(df[["time", "open", "high", "low", "close", "color"]].to_json(orient="records")),
                    "options": {
                        "upColor": "rgba(38,166,154,0.9)",
                        "downColor": "rgba(239,83,80,0.9)",
                        "borderVisible": False,
                        "wickUpColor": "rgba(38,166,154,0.9)",
                        "wickDownColor": "rgba(239,83,80,0.9)"
                    }
                }
            ]

            # Add SMA series
            if show_sma:
                series.append({
                    "type": "Line",
                    "data": sma_data,
                    "options": {
                        "color": "blue",
                        "lineWidth": 2,
                        "priceLineVisible": True,
                        "priceLineWidth": 1,
                        "priceLineColor": "blue",
                        "priceLineStyle": 2  # Dashed
                    }
                })

            # Add Volume series
            if show_volume:
                series.append({
                    "type": "Histogram",
                    "data": volume_data,
                    "options": {
                        "priceFormat": {"type": "volume"},
                        "priceScaleId": "",  # Separate scale for volume
                        "scaleMargins": {"top": 0.8, "bottom": 0}  # Position at bottom
                    }
                })

            # Add RSI series
            if show_rsi:
                series.append({
                    "type": "Area",
                    "data": rsi_data,
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
                # Add custom price scale for RSI
                chart_options[0]["priceScales"] = [
                    {"id": "rsi", "mode": 0, "autoScale": True, "minimum": 0, "maximum": 100,
                     "borderColor": "rgba(197, 203, 206, 0.8)"}
                ]

            st.write("### Candlestick Chart")
            renderLightweightCharts(chart_options, series)

            st.write("### OHLC Data")
            st.dataframe(df[["time", "open", "high", "low", "close", "volume"] + (["sma"] if show_sma else []) + (
                ["rsi"] if show_rsi else [])])
        else:
            st.info("No historical data available")
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")


# Mutual Funds page
def render_mutual_funds(broker):
    if broker != "Zerodha":
        st.info("Mutual Funds are only supported for Zerodha")
        return
    st.subheader("Mutual Funds - Zerodha")
    tabs = st.tabs(["Instruments", "Orders", "Holdings", "SIPs"])

    with tabs[0]:
        try:
            response = make_api_request("GET", f"{BACKEND_URL}/mutual-funds/instruments",
                                        token=st.session_state.access_token)
            instruments_df = pd.DataFrame(response)
            st.dataframe(instruments_df[["tradingsymbol", "name", "minimum_investment"]])
        except:
            st.error("Error fetching instruments")

    with tabs[1]:
        with st.form("mf_order_form"):
            scheme_code = st.text_input("Scheme Code")
            amount = st.number_input("Amount", min_value=100.0, step=100.0)
            transaction_type = st.selectbox("Transaction Type", ["BUY", "SELL"])
            if st.form_submit_button("Place MF Order"):
                try:
                    response = make_api_request("POST", f"{BACKEND_URL}/mutual-funds/orders", json={
                        "scheme_code": scheme_code,
                        "amount": amount,
                        "transaction_type": transaction_type
                    }, token=st.session_state.access_token)
                    st.success(f"Mutual Fund Order placed: {response['order_id']}")
                    st.toast("Order placed", icon="✅")
                except:
                    st.error("Error placing order")
                    st.toast("Order failed", icon="❌")

    with tabs[2]:
        try:
            response = make_api_request("GET", f"{BACKEND_URL}/mutual-funds/holdings",
                                        token=st.session_state.access_token)
            holdings_df = pd.DataFrame(response)
            if not holdings_df.empty:
                st.dataframe(holdings_df)
            else:
                st.info("No mutual fund holdings")
        except:
            st.error("Error fetching holdings")

    with tabs[3]:
        st.info("SIP management is not yet implemented in the backend")


# Order Modification page
def render_order_modification(broker):
    st.subheader(f"Modify Order - {broker}")
    try:
        response = make_api_request("GET", f"{BACKEND_URL}/orders/{broker}/", token=st.session_state.access_token)
        orders_df = pd.DataFrame(response)
        if not orders_df.empty:
            order_id = st.selectbox("Select Order", orders_df["order_id"], key="modify_order")
            with st.form("modify_order_form"):
                quantity = st.number_input("New Quantity", min_value=1,
                                           value=int(orders_df[orders_df["order_id"] == order_id]["quantity"].iloc[0]))
                order_type = st.selectbox("New Order Type", ["MARKET", "LIMIT", "SL", "SL-M"],
                                          index=["MARKET", "LIMIT", "SL", "SL-M"].index(
                                              orders_df[orders_df["order_id"] == order_id]["order_type"].iloc[0]))
                price = st.number_input("New Price", min_value=0.0,
                                        value=float(orders_df[orders_df["order_id"] == order_id]["price"].iloc[0]))
                trigger_price = st.number_input("New Trigger Price", min_value=0.0, value=float(
                    orders_df[orders_df["order_id"] == order_id]["trigger_price"].iloc[0]))
                if st.form_submit_button("Modify Order"):
                    try:
                        response = make_api_request("PUT", f"{BACKEND_URL}/orders/{order_id}/modify", json={
                            "quantity": quantity,
                            "order_type": order_type,
                            "price": price,
                            "trigger_price": trigger_price,
                            "validity": "DAY"
                        }, token=st.session_state.access_token)
                        st.success(f"Order {order_id} modified")
                        st.toast("Order modified", icon="✅")
                    except:
                        st.error("Failed to modify order")
                        st.toast("Modification failed", icon="❌")
        else:
            st.info("No orders to modify")
    except:
        st.error("Error fetching orders")


# Main application
def main():
    if "access_token" not in st.session_state:
        render_login_page()
    else:
        selected, broker = render_sidebar()

        if selected == "Dashboard":
            render_dashboard(broker)
        elif selected == "Get Token":
            st.subheader(f"Fetch {broker} Access Token")
            if broker == "Upstox":
                api_key = st.text_input("API Key", value=os.getenv("UPSTOX_API_KEY", ""))
                redirect_uri = st.text_input("Redirect URI", "https://api.upstox.com/v2/login")
                auth_url = f"https://api-v2.upstox.com/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}"
                st.markdown(f"[Click here to authorize]({auth_url})")
                auth_code = st.text_input("Authorization Code")
                if st.button("Fetch Access Token") and auth_code:
                    try:
                        response = make_api_request("POST", f"{BACKEND_URL}/auth/upstox/?auth_code={auth_code}",
                                                    token=st.session_state.access_token)
                        st.success(f"Successfully fetched Upstox Access Token")
                        st.toast("Token fetched", icon="✅")
                    except:
                        st.error("Failed to fetch Upstox token")
                        st.toast("Token fetch failed", icon="❌")
            elif broker == "Zerodha":
                request_token = st.text_input("Request Token")
                if st.button("Fetch Access Token") and request_token:
                    try:
                        response = make_api_request("POST",
                                                    f"{BACKEND_URL}/auth/zerodha/?request_token={request_token}",
                                                    token=st.session_state.access_token)
                        st.success(f"Successfully fetched Zerodha Access Token")
                        st.toast("Token fetched", icon="✅")
                    except:
                        st.error("Zerodha authentication failed")
                        st.toast("Token fetch failed", icon="❌")
        elif selected == "Order Management":
            render_order_management(broker)
        elif selected == "Order Book":
            st.subheader("Order Book")
            try:
                response = make_api_request("GET", f"{BACKEND_URL}/order-book/{broker}",
                                            token=st.session_state.access_token)
                orders_df = pd.DataFrame(response)
                if not orders_df.empty:
                    st.dataframe(orders_df)
                else:
                    st.info("No orders found")
            except:
                st.error("Failed to fetch order book")
        elif selected == "Positions":
            st.subheader("Current Positions")
            try:
                response = make_api_request("GET", f"{BACKEND_URL}/positions/{broker}",
                                            token=st.session_state.access_token)
                positions_df = pd.DataFrame(response)
                if not positions_df.empty:
                    st.dataframe(positions_df)
                else:
                    st.info("No positions found")
            except:
                st.error("Failed to fetch positions")
        elif selected == "Trade Dashboard":
            st.subheader("Trade Dashboard")
            try:
                response = make_api_request("GET", f"{BACKEND_URL}/trade-history/{broker}",
                                            token=st.session_state.access_token)
                trades_df = pd.DataFrame(response)
                if not trades_df.empty:
                    st.dataframe(trades_df)
                else:
                    st.info("No recent trades")
            except:
                st.error("Failed to fetch trade history")
        elif selected == "Portfolio":
            st.subheader("Portfolio Overview")
            try:
                response = make_api_request("GET", f"{BACKEND_URL}/portfolio/{broker}",
                                            token=st.session_state.access_token)
                portfolio_df = pd.DataFrame(response)
                if not portfolio_df.empty:
                    st.dataframe(portfolio_df)
                else:
                    st.info("No holdings found")
            except:
                st.error("Failed to fetch portfolio")
        elif selected == "Mutual Funds":
            render_mutual_funds(broker)
        elif selected == "Market Data":
            render_market_data(broker)
        elif selected == "Order Modification":
            render_order_modification(broker)


if __name__ == "__main__":
    main()