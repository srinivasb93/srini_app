import logging
from nicegui import ui, Client
import aiohttp
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import uuid
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
TOKEN = None
USER_ID = "default_user"
BROKER = "Zerodha"
THEME = "Dark"
INSTRUMENTS = {}


# API Helper
async def fetch_api(endpoint, method="GET", data=None, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request_data = {}
    if "auth/login" in endpoint:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        request_data = {"data": data}
    else:
        request_data = {"json": data}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method, f"{BASE_URL}{endpoint}", headers=headers, **request_data) as response:
                logger.debug(f"API call: {method} {endpoint}, Status: {response.status}")
                if response.status >= 400:
                    error = await response.json()
                    ui.notify(f"Error: {error.get('detail', 'Unknown error')}", type="negative")
                    return None
                return await response.json()
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        ui.notify(f"API request failed: {str(e)}", type="negative")
        return None


# WebSocket Client
async def connect_websocket():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"{BASE_URL}/ws/orders/{USER_ID}") as ws:
                logger.debug("WebSocket connected")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        logger.debug(f"WebSocket message: {data}")
                        ui.notify(f"Order update: {data[0].get('status')}", type="info")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        ui.notify("WebSocket connection failed", type="negative")


# Theme Styling
def apply_theme():
    if THEME == "Dark":
        ui.add_head_html("""
        <style>
            body { background-color: #1e1e1e; color: #ffffff; }
            .q-header, .q-footer { background-color: #2c2c2c; }
            .q-drawer { background-color: #2c2c2c; }
            .metric-box { background-color: #1a3c6e; padding: 10px; border-radius: 5px; }
            .q-card { background-color: #2c2c2c; color: #ffffff; border-radius: 10px; }
            .q-btn { background-color: #4CAF50; color: white; border-radius: 5px; }
        </style>
        """)
    else:
        ui.add_head_html("""
        <style>
            body { background-color: #f5f5f5; color: #000000; }
            .q-header, .q-footer { background-color: #e0e0e0; }
            .q-drawer { background-color: #e0e0e0; }
            .metric-box { background-color: #336699; color: white; padding: 10px; border-radius: 5px; }
            .q-card { background-color: #ffffff; color: #000000; border-radius: 10px; }
            .q-btn { background-color: #4CAF50; color: white; border-radius: 5px; }
        </style>
        """)


# Navigation Bar
def render_navigation():
    with ui.header().classes("bg-gray-800 text-white"):
        ui.label("Algo Trader").classes("text-h6")
        with ui.row().classes("gap-4"):
            for page in ["Dashboard", "Strategies", "Backtesting", "Paper Trading", "Live Trading"]:
                ui.button(page, on_click=lambda p=page: ui.navigate.to(f"/{p.lower().replace(' ', '-')}")).props(
                    "flat color=white")
        ui.button("Logout", on_click=lambda: ui.navigate.to("/")).props("flat color=white")


# Sidebar for Settings
async def render_sidebar():
    global BROKER, THEME
    with ui.left_drawer().classes("bg-gray-800 text-white") as drawer:
        ui.label("Settings").classes("text-h6")
        theme_select = ui.select(["Dark", "Light"], value=THEME).props("outlined dense").classes("w-full")
        theme_select.on("update:model-value", lambda e: globals().update(THEME=e["args"]) or apply_theme())
        broker_select = ui.select(["Upstox", "Zerodha"], value=BROKER).props("outlined dense").classes("w-full")
        broker_select.on("update:model-value", lambda e: globals().update(BROKER=e["args"]))

        async def check_broker_status():
            response = await fetch_api(f"/profile/{BROKER}", token=TOKEN)
            if response:
                ui.notify(f"{BROKER} Connected: {response['name']}", type="positive")
            else:
                ui.notify(f"{BROKER} Not Connected", type="negative")

        ui.button("Check Broker Status", on_click=check_broker_status).props("color=primary")


# Login Page
async def login(client: Client):
    global TOKEN
    TOKEN = None  # Reset token on login page load

    async def handle_login():
        data = {"username": email.value, "password": password.value, "grant_type": "password"}
        response = await fetch_api("/auth/login", method="POST", data=data)
        if response and "access_token" in response:
            global TOKEN, USER_ID
            TOKEN = response["access_token"]
            USER_ID = "default_user"  # Replace with JWT decoding
            ui.notify("Login successful", type="positive")

            # Validate Zerodha and Upstox tokens
            all_tokens_valid = True
            for broker in ["Zerodha", "Upstox"]:
                profile_response = await fetch_api(f"/profile/{broker}", token=TOKEN)
                if not profile_response:
                    all_tokens_valid = False
                    with ui.card().classes("w-96 mx-auto mt-4") as reconnect_card:
                        ui.label(f"Reconnect {broker}").classes("text-h6")
                        if broker == "Upstox":
                            auth_code = ui.input("Authorization Code").props("outlined")

                            async def reconnect_upstox():
                                if auth_code.value:
                                    reconnect_response = await fetch_api(f"/auth/upstox/?auth_code={auth_code.value}",
                                                                         method="POST", token=TOKEN)
                                    if reconnect_response:
                                        ui.notify("Upstox reconnected successfully", type="positive")
                                        # Recheck token validity
                                        profile_response = await fetch_api(f"/profile/{broker}", token=TOKEN)
                                        if profile_response:
                                            reconnect_card.clear()  # Clear the reconnection prompt
                                            # Check if all brokers are now valid
                                            all_valid = True
                                            for b in ["Zerodha", "Upstox"]:
                                                if not await fetch_api(f"/profile/{b}", token=TOKEN):
                                                    all_valid = False
                                                    break
                                            if all_valid:
                                                ui.navigate.to("/dashboard")
                                                asyncio.create_task(connect_websocket())
                                    else:
                                        ui.notify("Failed to reconnect Upstox", type="negative")
                                else:
                                    ui.notify("Please enter an authorization code", type="warning")

                            ui.button("Reconnect Upstox", on_click=reconnect_upstox).props("color=primary")
                        else:
                            request_token = ui.input("Request Token").props("outlined")

                            async def reconnect_zerodha():
                                if request_token.value:
                                    reconnect_response = await fetch_api(
                                        f"/auth/zerodha/?request_token={request_token.value}", method="POST",
                                        token=TOKEN)
                                    if reconnect_response:
                                        ui.notify("Zerodha reconnected successfully", type="positive")
                                        # Recheck token validity
                                        profile_response = await fetch_api(f"/profile/{broker}", token=TOKEN)
                                        if profile_response:
                                            reconnect_card.clear()  # Clear the reconnection prompt
                                            # Check if all brokers are now valid
                                            all_valid = True
                                            for b in ["Zerodha", "Upstox"]:
                                                if not await fetch_api(f"/profile/{b}", token=TOKEN):
                                                    all_valid = False
                                                    break
                                            if all_valid:
                                                ui.navigate.to("/dashboard")
                                                asyncio.create_task(connect_websocket())
                                    else:
                                        ui.notify("Failed to reconnect Zerodha", type="negative")
                                else:
                                    ui.notify("Please enter a request token", type="warning")

                            ui.button("Reconnect Zerodha", on_click=reconnect_zerodha).props("color=primary")

            # Only navigate to dashboard if all tokens are valid
            if all_tokens_valid:
                ui.navigate.to("/dashboard")
                asyncio.create_task(connect_websocket())
            else:
                ui.notify("Please reconnect invalid broker tokens to proceed", type="warning")
        else:
            ui.notify("Login failed", type="negative")

    async def handle_register():
        if new_password.value != confirm_password.value:
            ui.notify("Passwords do not match", type="negative")
            return
        data = {
            "email": new_username.value,
            "password": new_password.value,
            "upstox_api_key": upstox_api_key.value,
            "upstox_api_secret": upstox_api_secret.value,
            "zerodha_api_key": zerodha_api_key.value,
            "zerodha_api_secret": zerodha_api_secret.value
        }
        response = await fetch_api("/auth/register", method="POST", data=data)
        if response:
            ui.notify("Registration successful! Please log in.", type="positive")
        else:
            ui.notify("Registration failed", type="negative")

    apply_theme()
    with ui.card().classes("w-96 mx-auto mt-20"):
        ui.label("Algo Trader").classes("text-h6")
        with ui.tabs().classes("w-full") as tabs:
            login_tab = ui.tab("Login")
            signup_tab = ui.tab("Sign Up")

        with ui.tab_panels(tabs, value=login_tab).classes("w-full"):
            with ui.tab_panel(login_tab):
                email = ui.input("Email").props("outlined")
                password = ui.input("Password").props("outlined type=password")
                ui.button("Login", on_click=handle_login).props("color=primary")

            with ui.tab_panel(signup_tab):
                new_username = ui.input("Email").props("outlined")
                new_password = ui.input("Password").props("outlined type=password")
                confirm_password = ui.input("Confirm Password").props("outlined type=password")
                upstox_api_key = ui.input("Upstox API Key").props("outlined")
                upstox_api_secret = ui.input("Upstox API Secret").props("outlined")
                zerodha_api_key = ui.input("Zerodha API Key").props("outlined")
                zerodha_api_secret = ui.input("Zerodha API Secret").props("outlined")
                ui.button("Sign Up", on_click=handle_register).props("color=primary")


# Dashboard Page
async def dashboard():
    render_navigation()
    await render_sidebar()

    async def fetch_dashboard_data():
        funds_data = await fetch_api(f"/funds/{BROKER}", token=TOKEN)
        portfolio = await fetch_api(f"/portfolio/{BROKER}", token=TOKEN) or []
        positions = await fetch_api(f"/positions/{BROKER}", token=TOKEN) or []

        # Log funds_data to debug its structure
        logger.debug(f"funds_data: {funds_data}")

        # Ensure funds_data is a dictionary and has the expected structure
        if isinstance(funds_data, dict):
            equity = funds_data.get('equity', {})
            available = equity.get('available', 0.0)
            available_funds.text = f"₹{float(available):.2f}"
        else:
            available_funds.text = "₹0.00"
            ui.notify("Failed to fetch available funds", type="negative")

        total_value = sum(
            holding["Quantity"] * holding["LastPrice"] for holding in portfolio if isinstance(holding, dict))
        portfolio_value.text = f"₹{total_value:.2f}"
        open_positions.text = str(
            len([pos for pos in positions if isinstance(pos, dict) and pos.get("Quantity", 0) != 0]))

        if portfolio:
            fig = go.Figure(data=[
                go.Pie(labels=[p["Symbol"] for p in portfolio],
                       values=[p["Quantity"] * p["LastPrice"] for p in portfolio])
            ])
            fig.update()

        instruments = await fetch_api(f"/instruments/{BROKER}/?exchange=NSE", token=TOKEN) or []
        index_symbols = [inst for inst in instruments if "NIFTY 50" in inst["name"]]
        instrument_options = {inst["trading_symbol"]: inst["instrument_token"] for inst in index_symbols}
        instrument_select.options = list(instrument_options.keys())

        if instrument_select.value:
            ltp_data = await fetch_api(f"/ltp/{BROKER}?instruments={instrument_options[instrument_select.value]}",
                                       token=TOKEN)
            ltp_label.text = f"{instrument_select.value} LTP: ₹{ltp_data[0]['last_price']:.2f}" if ltp_data else "N/A"

    with ui.element("div").classes("p-4"):
        ui.label(f"{BROKER} Trading Dashboard").classes("text-h6")
        with ui.grid(columns=3).classes("w-full"):
            with ui.card():
                ui.label("Available Funds").classes("text-subtitle1")
                available_funds = ui.label().classes("text-h5 metric-box")
            with ui.card():
                ui.label("Portfolio Value").classes("text-subtitle1")
                portfolio_value = ui.label().classes("text-h5 metric-box")
            with ui.card():
                ui.label("Open Positions").classes("text-subtitle1")
                open_positions = ui.label().classes("text-h5 metric-box")

        ui.label("Market Watch").classes("text-h6")
        with ui.grid(columns=2).classes("w-full"):
            index_select = ui.select(["NIFTY 50", "NIFTY NEXT 50"], value="NIFTY 50").props("outlined dense")
            instrument_select = ui.select([]).props("outlined dense")
            ltp_label = ui.label().classes("text-h5 metric-box")

        chart = ui.plotly({}).classes("w-full h-96")
        await fetch_dashboard_data()
        ui.timer(60, fetch_dashboard_data)


# Order Management Page
async def order_management():
    render_navigation()
    await render_sidebar()

    async def place_order():
        order_data = {
            "trading_symbol": symbol.value,
            "instrument_token": INSTRUMENTS[symbol.value],
            "quantity": int(quantity.value),
            "order_type": order_type.value,
            "transaction_type": transaction_type.value,
            "product_type": product_type.value,
            "is_amo": amo_order.value,
            "price": float(price.value),
            "trigger_price": float(trigger_price.value),
            "stop_loss": float(stop_loss.value),
            "target": float(target.value),
            "validity": "DAY",
            "broker": BROKER,
            "schedule_datetime": schedule_datetime.value.isoformat() if schedule_order.value and schedule_datetime.value else None
        }
        endpoint = "/scheduled-orders" if schedule_order.value else "/orders"
        response = await fetch_api(f"{endpoint}/", method="POST", data=order_data, token=TOKEN)
        if response:
            ui.notify(
                f"Order {'scheduled' if schedule_order.value else 'placed'}: {response.get('order_id', response.get('message'))}",
                type="positive")
        else:
            ui.notify("Failed to place order", type="negative")

    async def cancel_order(order_id):
        response = await fetch_api(f"/orders/{order_id}", method="DELETE", token=TOKEN)
        if response:
            ui.notify(f"Order {order_id} cancelled", type="positive")
            await refresh_orders()
        else:
            ui.notify("Failed to cancel order", type="negative")

    async def refresh_orders():
        orders = await fetch_api(f"/orders/{BROKER}", token=TOKEN) or []
        scheduled_orders = [order for order in orders if order["status"] == "PENDING"]
        scheduled_grid.clear()
        with scheduled_grid:
            for order in scheduled_orders:
                with ui.card():
                    ui.label(f"Order ID: {order['order_id']}")
                    ui.label(f"Symbol: {order['trading_symbol']}")
                    ui.label(f"Schedule: {order['schedule_datetime']}")
                    ui.button("Cancel", on_click=lambda o=order["order_id"]: cancel_order(o)).props("color=negative")

    instruments = await fetch_api(f"/instruments/{BROKER}/?exchange=NSE", token=TOKEN) or []
    global INSTRUMENTS
    INSTRUMENTS = {inst["trading_symbol"]: inst["instrument_token"] for inst in instruments}
    instrument_options = {inst["trading_symbol"]: inst["instrument_token"] for inst in instruments if
                          "NIFTY 50" in inst["name"]}

    with ui.element("div").classes("p-4"):
        ui.label(f"Order Management - {BROKER}").classes("text-h6")
        with ui.tab_panels(ui.tabs(["Regular Orders", "Scheduled Orders"]).props("dense"),
                           value="Regular Orders").classes("w-full"):
            with ui.tab_panel("Regular Orders"):
                with ui.card().classes("w-96"):
                    ui.label("Symbol Selection").classes("text-h6")
                    index_select = ui.select(["NIFTY 50", "NIFTY NEXT 50"], value="NIFTY 50").props("outlined dense")
                    symbol = ui.select(list(instrument_options.keys())).props("outlined dense")

                    ui.label("Order Details").classes("text-h6 mt-4")
                    with ui.grid(columns=4):
                        quantity = ui.number("Quantity", value=1, min=1).props("outlined dense")
                        order_type = ui.select(["MARKET", "LIMIT", "SL", "SL-M"]).props("outlined dense")
                        transaction_type = ui.radio(["BUY", "SELL"], value="BUY").props("inline")
                        product_type = ui.select(["CNC", "MIS"] if BROKER == "Zerodha" else ["I", "D"]).props(
                            "outlined dense")

                    ui.label("Pricing").classes("text-h6 mt-4")
                    price = ui.number("Price", value=0.0).props("outlined dense")
                    trigger_price = ui.number("Trigger Price", value=0.0).props("outlined dense")

                    ui.label("Risk Management").classes("text-h6 mt-4")
                    stop_loss = ui.number("Stop-Loss", value=0.0).props("outlined dense")
                    target = ui.number("Target", value=0.0).props("outlined dense")

                    ui.label("Additional Options").classes("text-h6 mt-4")
                    amo_order = ui.checkbox("AMO Order")
                    schedule_order = ui.checkbox("Schedule Order")
                    schedule_datetime = ui.input("Schedule DateTime").props("outlined type=datetime-local")
                    schedule_datetime.visible = schedule_order.value

                    ui.button("Place Order", on_click=place_order).props("color=primary mt-4")

            with ui.tab_panel("Scheduled Orders"):
                scheduled_grid = ui.grid(columns=3).classes("w-full")
                await refresh_orders()


# Order Book Page
async def order_book():
    render_navigation()
    await render_sidebar()

    async def cancel_order(order_id):
        response = await fetch_api(f"/orders/{order_id}", method="DELETE", token=TOKEN)
        if response:
            ui.notify(f"Order {order_id} cancelled", type="positive")
            await refresh_orders()
        else:
            ui.notify("Failed to cancel order", type="negative")

    async def refresh_orders():
        response = await fetch_api(f"/order-book/{BROKER}", token=TOKEN) or []
        orders_df = pd.DataFrame(response)
        orders_grid.clear()
        with orders_grid:
            for _, row in orders_df.iterrows():
                with ui.card():
                    ui.label(f"Order ID: {row['OrderID']}")
                    ui.label(f"Symbol: {row['Symbol']}")
                    ui.label(f"Status: {row['Status']}")
                    if row["Status"] not in ["complete", "rejected", "cancelled"]:
                        ui.button("Cancel", on_click=lambda o=row["OrderID"]: cancel_order(o)).props("color=negative")

    with ui.element("div").classes("p-4"):
        ui.label("Order Book").classes("text-h6")
        orders_grid = ui.grid(columns=3).classes("w-full")
        await refresh_orders()
        ui.timer(60, refresh_orders)


# Positions Page
async def positions():
    render_navigation()
    await render_sidebar()

    async def square_off(position):
        order_data = {
            "trading_symbol": position["Symbol"],
            "instrument_token": position["InstrumentToken"],
            "quantity": abs(position["Quantity"]),
            "order_type": "MARKET",
            "transaction_type": "SELL" if position["Quantity"] > 0 else "BUY",
            "product_type": position["Product"],
            "is_amo": False,
            "price": 0.0,
            "trigger_price": 0.0,
            "validity": "DAY",
            "broker": BROKER
        }
        response = await fetch_api("/orders/", method="POST", data=order_data, token=TOKEN)
        if response:
            ui.notify(f"Square off order placed: {response['order_id']}", type="positive")
        else:
            ui.notify("Failed to square off", type="negative")

    with ui.element("div").classes("p-4"):
        ui.label("Current Positions").classes("text-h6")
        positions = await fetch_api(f"/positions/{BROKER}", token=TOKEN) or []
        positions_df = pd.DataFrame(positions)
        if not positions_df.empty:
            total_investment = (positions_df['AvgPrice'] * positions_df['Quantity']).sum()
            total_pnl = positions_df['PnL'].sum()
            total_value = (positions_df['LastPrice'] * positions_df['Quantity']).sum()

            with ui.grid(columns=3).classes("w-full"):
                ui.label(f"Total Investment: ₹{total_investment:.2f}").classes("metric-box")
                ui.label(f"Total P&L: ₹{total_pnl:.2f}").classes("metric-box")
                ui.label(f"Current Value: ₹{total_value:.2f}").classes("metric-box")

            for _, row in positions_df.iterrows():
                with ui.card():
                    ui.label(
                        f"{row['Broker']} - {row['Symbol']} - {row['Quantity']} @ {row['AvgPrice']} ({row['Product']})")
                    ui.button("Square Off", on_click=lambda r=row: square_off(r)).props("color=primary")
        else:
            ui.notify("No positions found", type="info")


# Trade Dashboard Page
async def trade_dashboard():
    render_navigation()
    await render_sidebar()

    async def fetch_trade_data():
        positions = await fetch_api(f"/positions/{BROKER}", token=TOKEN) or []
        trades = await fetch_api(f"/trade-history/{BROKER}", token=TOKEN) or []
        funds_data = await fetch_api(f"/funds/{BROKER}", token=TOKEN) or {}

        positions_df = pd.DataFrame(positions)
        trades_df = pd.DataFrame(trades)

        positions_grid.clear()
        with positions_grid:
            if not positions_df.empty:
                for _, row in positions_df.iterrows():
                    ui.label(f"Symbol: {row['InstrumentToken']}")
                    ui.label(f"Quantity: {row['Quantity']}")
                    ui.label(f"PnL: ₹{row['PnL']:.2f}")

        trades_grid.clear()
        with trades_grid:
            for _, row in trades_df.head(10).iterrows():
                ui.label(f"Symbol: {row['instrument_token']}")
                ui.label(f"PnL: ₹{row['pnl']:.2f}")
                ui.label(f"Time: {row['entry_time']}")

    with ui.element("div").classes("p-4"):
        ui.label("Trade Dashboard").classes("text-h6")
        ui.label("Open Positions").classes("text-subtitle1")
        positions_grid = ui.grid(columns=3).classes("w-full")
        ui.label("Recent Trades").classes("text-subtitle1 mt-4")
        trades_grid = ui.grid(columns=3).classes("w-full")
        await fetch_trade_data()
        ui.timer(60, fetch_trade_data)


# Portfolio Page
async def portfolio():
    render_navigation()
    await render_sidebar()
    with ui.element("div").classes("p-4"):
        ui.label("Portfolio Overview").classes("text-h6")
        portfolio = await fetch_api(f"/portfolio/{BROKER}", token=TOKEN) or []
        portfolio_df = pd.DataFrame(portfolio)
        if not portfolio_df.empty:
            total_value = (portfolio_df["LastPrice"] * portfolio_df["Quantity"]).sum()
            total_buy_value = (portfolio_df["AvgPrice"] * portfolio_df["Quantity"]).sum()
            total_pnl = ((portfolio_df["LastPrice"] - portfolio_df["AvgPrice"]) * portfolio_df["Quantity"]).sum()
            with ui.grid(columns=3).classes("w-full"):
                ui.label(f"Total Buy Value: ₹{total_buy_value:.2f}").classes("metric-box")
                ui.label(f"Total Portfolio Value: ₹{total_value:.2f}").classes("metric-box")
                ui.label(f"Total P&L: ₹{total_pnl:.2f}").classes("metric-box")
            for _, row in portfolio_df.iterrows():
                with ui.card():
                    ui.label(f"Symbol: {row['Symbol']}")
                    ui.label(f"Quantity: {row['Quantity']}")
                    ui.label(f"PnL: ₹{row['PnL']:.2f}")


# Mutual Funds Page
async def mutual_funds():
    render_navigation()
    await render_sidebar()
    with ui.element("div").classes("p-4"):
        ui.label("Mutual Funds - Zerodha").classes("text-h6")
        if BROKER != "Zerodha":
            ui.notify("Mutual Funds are only supported for Zerodha", type="info")
            return
        with ui.tab_panels(ui.tabs(["Holdings", "SIPs"]).props("dense"), value="Holdings").classes("w-full"):
            with ui.tab_panel("Holdings"):
                holdings = await fetch_api("/mutual-funds/holdings", token=TOKEN) or []
                holdings_df = pd.DataFrame(holdings)
                if not holdings_df.empty:
                    for _, row in holdings_df.iterrows():
                        with ui.card():
                            ui.label(f"Fund: {row['fund']}")
                            ui.label(f"Quantity: {row['quantity']}")
                            ui.label(f"Value: ₹{row['last_price'] * row['quantity']:.2f}")
            with ui.tab_panel("SIPs"):
                sips = await fetch_api("/mutual-funds/sips", token=TOKEN) or []
                sips_df = pd.DataFrame(sips)
                if not sips_df.empty:
                    for _, row in sips_df.iterrows():
                        with ui.card():
                            ui.label(f"SIP ID: {row['sip_id']}")
                            ui.label(f"Scheme: {row['scheme_code']}")
                            ui.label(f"Amount: ₹{row['amount']:.2f}")


# Analytics Page
async def analytics():
    render_navigation()
    await render_sidebar()
    with ui.element("div").classes("p-4"):
        ui.label("Trade Analytics & Live Feed").classes("text-h6")
        with ui.grid(columns=3).classes("w-full"):
            symbol = ui.select(list(INSTRUMENTS.keys())).props("outlined dense")
            timeframe = ui.select(["1minute", "day", "week"]).props("outlined dense")
            ema_period = ui.number("EMA Period", value=20, min=5).props("outlined dense")
        response = await fetch_api(
            f"/analytics/{INSTRUMENTS[symbol.value]}?timeframe={timeframe.value}&ema_period={ema_period.value}&rsi_period=14&lr_period=20",
            token=TOKEN)
        if response:
            df = pd.DataFrame(response.get("candles", []))
            fig = go.Figure(data=[
                go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])
            ])
            ui.plotly(fig).classes("w-full h-96")


# Strategies Page (Strategy Builder)
async def strategies():
    render_navigation()
    await render_sidebar()

    async def execute_strategy():
        params = {}
        if strategy.value == "MACD Crossover":
            params = {"fast_period": fast_period.value, "slow_period": slow_period.value,
                      "signal_period": signal_period.value}
        data = {
            "strategy": strategy.value,
            "instrument_token": INSTRUMENTS[symbol.value],
            "quantity": int(quantity.value),
            "stop_loss": float(stop_loss.value),
            "take_profit": float(take_profit.value),
            "broker": BROKER
        }
        response = await fetch_api("/algo-trading/execute", method="POST", data=data, token=TOKEN)
        if response:
            ui.notify(response["message"], type="positive")

    with ui.element("div").classes("p-4"):
        ui.label("Strategy Builder").classes("text-h6")
        with ui.grid(columns=2).classes("w-full"):
            with ui.card():
                strategy = ui.select([
                    "MACD Crossover", "Bollinger Bands", "RSI Oversold/Overbought",
                    "Stochastic Oscillator", "Support/Resistance Breakout"
                ]).props("outlined dense")
                symbol = ui.select(list(INSTRUMENTS.keys())).props("outlined dense")
                quantity = ui.number("Quantity", value=1, min=1).props("outlined dense")

                if strategy.value == "MACD Crossover":
                    fast_period = ui.number("Fast EMA Period", value=12, min=3).props("outlined dense")
                    slow_period = ui.number("Slow EMA Period", value=26, min=5).props("outlined dense")
                    signal_period = ui.number("Signal Period", value=9, min=3).props("outlined dense")

            with ui.card():
                ui.label("Risk Management").classes("text-subtitle1")
                stop_loss = ui.number("Stop Loss (%)", value=1.0, min=0.1).props("outlined dense")
                take_profit = ui.number("Take Profit (%)", value=2.0, min=0.1).props("outlined dense")
                ui.button("Execute Strategy", on_click=execute_strategy).props("color=primary")


# Backtesting Page
async def backtesting():
    render_navigation()
    await render_sidebar()

    async def run_backtest():
        params = {"initial_investment": 50000, "stop_loss_atr_mult_range": [1.5, 2.5],
                  "target_atr_mult_range": [4.0, 6.0]}
        data = {
            "instrument_token": INSTRUMENTS[symbol.value],
            "timeframe": timeframe.value,
            "strategy": "Short Sell Optimization",
            "params": params,
            "start_date": start_date.value,
            "end_date": end_date.value
        }
        response = await fetch_api("/algo-trading/backtest", method="POST", data=data, token=TOKEN)
        if response:
            result_grid.clear()
            with result_grid:
                ui.label(f"Total Profit: ₹{response['TotalProfit']:.2f}")
                ui.label(f"Win Rate: {response['WinRate']:.2f}%")
                tradebook_df = pd.DataFrame(response["Tradebook"])
                if not tradebook_df.empty:
                    fig = go.Figure(data=[
                        go.Scatter(x=tradebook_df["Date"], y=tradebook_df["PortfolioValue"], mode="lines",
                                   name="Portfolio Value")
                    ])
                    chart.update(fig)

    with ui.element("div").classes("p-4"):
        ui.label("Backtesting").classes("text-h6")
        with ui.card().classes("w-96"):
            symbol = ui.select(list(INSTRUMENTS.keys())).props("outlined dense")
            timeframe = ui.select(["minute", "day", "week"]).props("outlined dense")
            start_date = ui.input("Start Date (YYYY-MM-DD)").props("outlined")
            end_date = ui.input("End Date (YYYY-MM-DD)").props("outlined")
            ui.button("Run Backtest", on_click=run_backtest).props("color=primary")

        result_grid = ui.grid(columns=2).classes("w-full mt-4")
        chart = ui.plotly({}).classes("w-full h-96")


# Paper Trading Page
async def paper_trading():
    render_navigation()
    await render_sidebar()
    with ui.element("div").classes("p-4"):
        ui.label("Paper Trading").classes("text-h6")
        ui.label("Simulate trades without real funds.").classes("text-subtitle1")
        ui.notify("Paper Trading feature coming soon!", type="info")


# Live Trading Page
async def live_trading():
    render_navigation()
    await render_sidebar()

    async def fetch_live_data():
        trades = await fetch_api(f"/trade-history/{BROKER}", token=TOKEN) or []
        trades_grid.clear()
        with trades_grid:
            for trade in trades:
                with ui.card():
                    ui.label(f"Symbol: {trade['instrument_token']}")
                    ui.label(f"PnL: ₹{trade['pnl']:.2f}")
                    ui.label(f"Time: {trade['entry_time']}")

    with ui.element("div").classes("p-4"):
        ui.label("Live Trading").classes("text-h6")
        trades_grid = ui.grid(columns=2).classes("w-full")
        await fetch_live_data()
        ui.timer(10, fetch_live_data)


# Main App Setup
ui.page("/")(login)
ui.page("/dashboard")(dashboard)
ui.page("/order-management")(order_management)
ui.page("/order-book")(order_book)
ui.page("/positions")(positions)
ui.page("/trade-dashboard")(trade_dashboard)
ui.page("/portfolio")(portfolio)
ui.page("/mutual-funds")(mutual_funds)
ui.page("/analytics")(analytics)
ui.page("/strategies")(strategies)
ui.page("/backtesting")(backtesting)
ui.page("/paper-trading")(paper_trading)
ui.page("/live-trading")(live_trading)

ui.run(title="Algo Trader", port=8080, dark=True)