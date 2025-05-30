#!/usr/bin/env python3
import logging
from nicegui import ui, Client, app
import aiohttp
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta # Added timedelta
import json
import uuid
import numpy as np
import jwt # PyJWT library for decoding tokens

# Import page rendering functions from modules
from order_management import render_order_management
from strategies import render_strategies_page
from backtesting import render_backtesting_page
from analytics import render_analytics_page # Added analytics import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

# --- State Management --- (Using app.storage for simplicity)
# app.storage.user stores user-specific data like token, email, broker settings
# app.storage.general stores application-wide data like instruments

def get_user_storage():
    return app.storage.user

def get_general_storage():
    return app.storage.general

# --- API Helper --- (Enhanced Error Handling)
async def fetch_api(endpoint, method="GET", data=None, token=None):
    user_storage = get_user_storage()
    if token is None:
        token = user_storage.get("token")

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request_data = {}
    if "auth/login" in endpoint:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        request_data = {"data": data}
    # Handle GET requests with params passed via data dict
    elif method == "GET" and data is not None and isinstance(data, dict):
        request_data = {"params": data}
    elif data is not None:
        headers["Content-Type"] = "application/json"
        request_data = {"json": data}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method, f"{BASE_URL}{endpoint}", headers=headers, **request_data) as response:
                logger.info(f'API call: {method} {endpoint}, Params/Data: {request_data.get("params", request_data.get("json", "N/A"))}, Status: {response.status}')
                if response.status == 200:
                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        return {"success": True, "message": "Operation successful"}
                elif response.status == 204:
                    return {"success": True, "message": "Operation successful (No Content)"}
                elif response.status == 401:
                    ui.notify("Session expired or invalid. Please log in again.", type="negative")
                    user_storage["token"] = None
                    user_storage["email"] = None
                    ui.navigate.to("/")
                    return None
                else:
                    try:
                        error_details = await response.json()
                        error_message = error_details.get("detail", "Unknown API error")
                        if isinstance(error_message, list):
                            error_message = "; ".join([f'{e.get("loc",[""])[-1]}: {e.get("msg","Invalid input")}' for e in error_message])
                        logger.error(f"API Error ({response.status}) for {method} {endpoint}: {error_message}")
                        ui.notify(f"API Error: {error_message}", type="negative")
                    except Exception as json_error:
                        error_text = await response.text()
                        logger.error(f"API Error ({response.status}) for {method} {endpoint}. Could not parse JSON response: {json_error}. Response text: {error_text}")
                        ui.notify(f"API Error ({response.status}): {error_text[:100]}", type="negative")
                    return None
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Connection Error for {method} {endpoint}: {e}")
        ui.notify(f"Connection Error: Could not connect to the backend at {BASE_URL}. Please ensure it's running", type="negative")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during API request to {endpoint}: {e}")
        ui.notify(f"An unexpected error occurred: {e}", type="negative")
        return None

# --- WebSocket Client --- (Improved connection handling)
async def connect_websocket():
    user_storage = get_user_storage()
    token = user_storage.get("token")
    if not token:
        logger.warning("No token found, cannot connect WebSocket.")
        return

    try:
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        user_id = decoded_token.get("sub", "default_user")
        user_storage["user_id"] = user_id
        logger.info(f"User ID from token: {user_id}")
    except jwt.DecodeError:
        logger.error("Failed to decode JWT token to get user ID.")
        user_id = "default_user"
        user_storage["user_id"] = user_id

    ws_url = f"{BASE_URL}/ws/orders/{user_id}?token={token}"
    logger.info(f"Attempting WebSocket connection to: {ws_url}")

    while user_storage.get("token"):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    logger.info("WebSocket connected successfully.")
                    user_storage["websocket_connected"] = True
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                logger.debug(f"WebSocket message received: {data}")
                                if isinstance(data, list) and len(data) > 0 and "status" in data[0]:
                                    ui.notify(f'Order Update: {data[0].get("trading_symbol","N/A")} - {data[0]["status"]}', type="info")
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode WebSocket JSON message: {msg.data}")
                            except Exception as e:
                                logger.exception(f"Error processing WebSocket message: {e}")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket connection closed with exception {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.warning("WebSocket connection closed.")
                            break
        except aiohttp.ClientConnectorError as e:
            logger.error(f"WebSocket connection failed: {e}. Retrying in 10 seconds...")
        except Exception as e:
            logger.exception(f"Unexpected WebSocket error: {e}. Retrying in 10 seconds...")

        user_storage["websocket_connected"] = False
        if not user_storage.get("token"):
            break
        await asyncio.sleep(10)
    logger.info("WebSocket connection loop terminated.")

# --- Theme Styling --- (Simplified)
def apply_theme():
    user_storage = get_user_storage()
    theme_mode = user_storage.get("theme", "Dark")
    if theme_mode == "Dark":
        ui.query("body").classes(add="bg-dark text-white", remove="bg-grey-2 text-black")
    else:
        ui.query("body").classes(add="bg-grey-2 text-black", remove="bg-dark text-white")

# --- UI Components --- #

# Reusable Header
def render_header():
    user_storage = get_user_storage()
    with ui.header(elevated=True).classes("items-center justify-between"):
        with ui.row(wrap=False).classes("items-center"):
            ui.button(icon="menu", on_click=lambda: left_drawer.toggle()).props("flat color=white dense").classes("lt-md")
            ui.label("Algo Trader").classes("text-h6 font-bold")

        with ui.row(wrap=False).classes("items-center gap-2 gt-sm"):
            pages = ["Dashboard", "Order Management", "Order Book", "Positions", "Portfolio",
                     "Analytics", "Strategies", "Backtesting", "Mutual Funds"]
            for page_name in pages:
                page_path = f'/{page_name.lower().replace(" ", "-")}'
                ui.button(page_name, on_click=lambda path=page_path: ui.navigate.to(path)).props("flat color=white")

        with ui.row(wrap=False).classes("items-center gap-2"):
            if user_storage.get("email"):
                ui.label(f'Welcome, {user_storage.get("email")}').classes("text-sm gt-xs")
                ui.button("Logout", on_click=logout, icon="logout").props("flat color=white dense")

# Reusable Left Drawer (Sidebar)
async def render_left_drawer():
    user_storage = get_user_storage()
    with ui.left_drawer(fixed=False).classes("bg-grey-8 text-white") as drawer:
        ui.label("Menu").classes("text-h6 q-pa-md")
        ui.separator()

        with ui.list().classes("lt-md"):
            pages = ["Dashboard", "Order Management", "Order Book", "Positions", "Portfolio",
                     "Analytics", "Strategies", "Backtesting", "Mutual Funds"]
            for page_name in pages:
                page_path = f'/{page_name.lower().replace(" ", "-")}'
                ui.item(page_name, on_click=lambda path=page_path: (ui.navigate.to(path), drawer.hide()))
            ui.separator().classes("lt-md")

        ui.label("Settings").classes("text-subtitle1 q-pa-md")
        theme_select = ui.select(["Dark", "Light"], label="Theme", value=user_storage.get("theme", "Dark"),
                                 on_change=lambda e: update_setting("theme", e.value)).classes("q-ma-md")
        broker_select = ui.select(["Upstox", "Zerodha"], label="Broker", value=user_storage.get("broker", "Zerodha"),
                                  on_change=lambda e: update_setting("broker", e.value)).classes("q-ma-md")

        ui.separator()
        ui.label("Broker Status").classes("text-subtitle1 q-pa-md")
        status_container = ui.column().classes("q-pa-md")

        async def check_and_display_broker_status():
            status_container.clear()
            with status_container:
                for broker in ["Zerodha", "Upstox"]:
                    loading_spinner = ui.spinner(size="sm").props("color=white")
                    status_label = ui.label(f"Checking {broker}...")
                    profile = await fetch_api(f"/profile/{broker}")
                    loading_spinner.delete()
                    if profile and profile.get("name"):
                        status_label.text = f'{broker}: Connected ({profile["name"]})'
                        status_label.classes(add="text-positive")
                    else:
                        status_label.text = f"{broker}: Not Connected"
                        status_label.classes(add="text-negative")

        ui.button("Refresh Status", on_click=check_and_display_broker_status, icon="refresh").classes("q-ma-md")
        await check_and_display_broker_status()

    return drawer

# Helper to update user settings
def update_setting(key, value):
    user_storage = get_user_storage()
    user_storage[key] = value
    if key == "theme":
        apply_theme()
    logger.info(f"Setting updated: {key} = {value}")
    if key == "broker":
        ui.notify(f"Broker changed to {value}. Refreshing data...")

# Logout Function
def logout():
    user_storage = get_user_storage()
    user_storage["token"] = None
    user_storage["email"] = None
    user_storage["user_id"] = None
    user_storage["websocket_connected"] = False
    ui.notify("Logged out successfully.", type="positive")
    ui.navigate.to("/")

# --- Pages --- #

@ui.page("/")
async def login_page(client: Client):
    user_storage = get_user_storage()
    if user_storage.get("token"):
        profile = await fetch_api("/profile/Zerodha")
        if profile:
            ui.navigate.to("/dashboard")
            return
        else:
            user_storage["token"] = None
            user_storage["email"] = None

    async def handle_login():
        login_button.props("loading=true")
        error_label.set_text("")
        data = {"username": email.value, "password": password.value, "grant_type": "password"}
        response = await fetch_api("/auth/login", method="POST", data=data, token=None)
        login_button.props("loading=false")

        if response and "access_token" in response:
            user_storage["token"] = response["access_token"]
            try:
                decoded_token = jwt.decode(response["access_token"], options={"verify_signature": False})
                user_storage["email"] = decoded_token.get("sub")
            except jwt.DecodeError:
                user_storage["email"] = email.value

            logger.info(f'Login successful for {user_storage["email"]}')
            ui.notify("Login successful!", type="positive")
            # asyncio.create_task(connect_websocket())
            ui.navigate.to("/dashboard")
        else:
            error_label.set_text("Login failed. Please check your credentials.")
            ui.notify("Login failed. Check credentials.", type="warning")

    async def handle_register():
        register_button.props("loading=true")
        reg_error_label.set_text("")
        if new_password.value != confirm_password.value:
            reg_error_label.set_text("Passwords do not match.")
            ui.notify("Passwords do not match", type="warning")
            register_button.props("loading=false")
            return
        if not all([new_username.value, new_password.value, confirm_password.value]):
             reg_error_label.set_text("Please fill in all required fields.")
             ui.notify("Please fill in all required fields.", type="warning")
             register_button.props("loading=false")
             return

        data = {
            "email": new_username.value,
            "password": new_password.value,
            "upstox_api_key": upstox_api_key.value or None,
            "upstox_api_secret": upstox_api_secret.value or None,
            "zerodha_api_key": zerodha_api_key.value or None,
            "zerodha_api_secret": zerodha_api_secret.value or None
        }
        response = await fetch_api("/auth/register", method="POST", data=data, token=None)
        register_button.props("loading=false")

        if response and response.get("email") == new_username.value:
            ui.notify("Registration successful! Please log in.", type="positive")
            tabs.set_value(login_tab)
        else:
            reg_error_label.set_text("Registration failed. Please try again.")
            ui.notify("Registration failed.", type="warning")

    apply_theme()
    with ui.column().classes("absolute-center items-center"):
        ui.label("Algo Trader").classes("text-h4 font-weight-bold q-mb-md")
        with ui.card().classes("w-96"):
            with ui.tabs().classes("w-full").props("dense") as tabs:
                login_tab = ui.tab("Login")
                signup_tab = ui.tab("Sign Up")

            with ui.tab_panels(tabs, value=login_tab).classes("w-full q-pa-md"):
                with ui.tab_panel(login_tab):
                    with ui.column().classes("gap-4"):
                        email = ui.input("Email", placeholder="Enter your email",
                                         validation={"Invalid email": lambda v: "@" in v if v else True})
                        password = ui.input("Password", placeholder="Enter your password", password=True, password_toggle_button=True)
                        error_label = ui.label().classes("text-negative text-sm")
                        login_button = ui.button("Login", on_click=handle_login).props("color=primary").classes("w-full")

                with ui.tab_panel(signup_tab):
                    with ui.column().classes("gap-3"):
                        new_username = ui.input("Email *", placeholder="Enter your email",
                                                validation={"Invalid email": lambda v: "@" in v if v else False})
                        new_password = ui.input("Password *", placeholder="Create a password", password=True, password_toggle_button=True,
                                                validation={"Required": bool})
                        confirm_password = ui.input("Confirm Password *", placeholder="Confirm your password", password=True, password_toggle_button=True,
                                                    validation={"Passwords must match": lambda v: v == new_password.value})
                        ui.separator()
                        ui.label("Broker API Keys (Optional)").classes("text-caption")
                        upstox_api_key = ui.input("Upstox API Key", placeholder="Optional")
                        upstox_api_secret = ui.input("Upstox API Secret", placeholder="Optional", password=True)
                        zerodha_api_key = ui.input("Zerodha API Key", placeholder="Optional")
                        zerodha_api_secret = ui.input("Zerodha API Secret", placeholder="Optional", password=True)
                        reg_error_label = ui.label().classes("text-negative text-sm")
                        register_button = ui.button("Sign Up", on_click=handle_register).props("color=primary").classes("w-full")

# --- Protected Page Decorator --- #
def protected_page(func):
    async def wrapper(*args, **kwargs):
        user_storage = get_user_storage()
        token = user_storage.get("token")
        if not token:
            logger.warning(f"Access denied to {func.__name__}. No token.")
            ui.notify("Please log in to access this page.", type="warning")
            ui.navigate.to("/")
            return

        apply_theme()
        render_header()
        global left_drawer
        left_drawer = await render_left_drawer()

        await func(*args, **kwargs)
    return wrapper

# --- Main Application Pages --- #

@ui.page("/dashboard")
@protected_page
async def dashboard_page():
    user_storage = get_user_storage()
    general_storage = get_general_storage()
    broker = user_storage.get("broker", "Zerodha")

    async def fetch_dashboard_data():
        results = await asyncio.gather(
            fetch_api(f"/funds/{broker}"),
            fetch_api(f"/portfolio/{broker}"),
            fetch_api(f"/positions/{broker}"),
            fetch_api(f"/instruments/{broker}/?exchange=NSE"),
            return_exceptions=True
        )
        funds_data, portfolio_data, positions_data, instruments_data = results

        if isinstance(funds_data, dict) and funds_data.get("equity"):
            available = funds_data["equity"].get("available", 0.0)
            available_funds_label.text = f"₹{float(available):,.2f}"
        elif isinstance(funds_data, Exception):
            available_funds_label.text = "Error"
            logger.error(f"Error fetching funds: {funds_data}")
        else:
            available_funds_label.text = "N/A"

        if isinstance(portfolio_data, list) and portfolio_data:
            df = pd.DataFrame(portfolio_data)
            df["Value"] = df["Quantity"] * df["LastPrice"]
            total_value = df["Value"].sum()
            portfolio_value_label.text = f"₹{total_value:,.2f}"
            fig = px.pie(df, names="Symbol", values="Value", title="Portfolio Allocation",
                         hole=.3, color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white" if user_storage.get("theme", "Dark") == "Dark" else "black")
            portfolio_chart.update(fig)
        elif isinstance(portfolio_data, Exception):
            portfolio_value_label.text = "Error"
            portfolio_chart.update(go.Figure())
            logger.error(f"Error fetching portfolio: {portfolio_data}")
        else:
            portfolio_value_label.text = "₹0.00"
            portfolio_chart.update(go.Figure().update_layout(title_text="No portfolio data"))

        if isinstance(positions_data, list):
            open_positions = len([p for p in positions_data if p.get("Quantity", 0) != 0])
            open_positions_label.text = str(open_positions)
        elif isinstance(positions_data, Exception):
            open_positions_label.text = "Error"
            logger.error(f"Error fetching positions: {positions_data}")
        else:
            open_positions_label.text = "0"

        # Update Instruments in general storage and select component
        if isinstance(instruments_data, list):
            equity_instruments = [i for i in instruments_data if i.get("segment") == "NSE" and i.get("instrument_type") == "EQ"]
            general_storage["instruments"] = {i["trading_symbol"]: i["instrument_token"] for i in equity_instruments}
            current_value = instrument_select.value
            instrument_select.options = sorted(list(general_storage["instruments"].keys()))
            if current_value in general_storage["instruments"]:
                instrument_select.set_value(current_value)
            elif instrument_select.options:
                 instrument_select.set_value(instrument_select.options[0])
        elif isinstance(instruments_data, Exception):
             logger.error(f"Error fetching instruments: {instruments_data}")
        else:
             general_storage["instruments"] = {}
             instrument_select.options = []

        await update_ltp()

    async def update_ltp():
        selected_symbol = instrument_select.value
        instruments = general_storage.get("instruments", {})
        if selected_symbol and selected_symbol in instruments:
            instrument_token = instruments[selected_symbol]
            ltp_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
            if ltp_data and isinstance(ltp_data, list) and len(ltp_data) > 0:
                price = ltp_data[0].get("last_price", "N/A")
                ltp_label.text = f"{selected_symbol}: ₹{price:,.2f}" if isinstance(price, (int, float)) else "N/A"
            else:
                ltp_label.text = f"{selected_symbol}: N/A"
                logger.warning(f"Could not fetch LTP for {selected_symbol}")
        else:
            ltp_label.text = "Select Instrument"

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label(f"{broker} Trading Dashboard").classes("text-h5")
        with ui.row().classes("w-full gap-4 justify-around"):
            with ui.card().classes("flex-grow text-center"):
                ui.label("Available Funds").classes("text-caption")
                available_funds_label = ui.label("Loading...").classes("text-h6 font-weight-bold")
            with ui.card().classes("flex-grow text-center"):
                ui.label("Portfolio Value").classes("text-caption")
                portfolio_value_label = ui.label("Loading...").classes("text-h6 font-weight-bold")
            with ui.card().classes("flex-grow text-center"):
                ui.label("Open Positions").classes("text-caption")
                open_positions_label = ui.label("Loading...").classes("text-h6 font-weight-bold")

        with ui.row().classes("w-full gap-4 items-stretch"):
            with ui.card().classes("w-1/3"):
                ui.label("Market Watch").classes("text-subtitle1")
                instrument_select = ui.select(options=[], label="Select Instrument", with_input=True,
                                              on_change=update_ltp).classes("w-full")
                ltp_label = ui.label("Select Instrument").classes("text-h6 q-mt-md")

            with ui.card().classes("flex-grow"):
                ui.label("Portfolio Allocation").classes("text-subtitle1")
                portfolio_chart = ui.plotly().classes("w-full h-64")

    await fetch_dashboard_data()
    ui.timer(60.0, fetch_dashboard_data)

# --- Integrated Pages --- #

@ui.page("/order-management")
@protected_page
async def order_management_page():
    user_storage = get_user_storage()
    general_storage = get_general_storage()
    instruments = general_storage.get("instruments", {})
    if not instruments:
        broker = user_storage.get("broker", "Zerodha")
        logger.info("Instruments not found in storage, fetching...")
        instruments_data = await fetch_api(f"/instruments/{broker}/?exchange=NSE")
        if isinstance(instruments_data, list):
            equity_instruments = [i for i in instruments_data if i.get("segment") == "NSE" and i.get("instrument_type") == "EQ"]
            instruments = {i["trading_symbol"]: i["instrument_token"] for i in equity_instruments}
            general_storage["instruments"] = instruments
        else:
            ui.notify("Failed to load instruments for Order Management.", type="negative")
            instruments = {}
    await render_order_management(fetch_api, user_storage, instruments)

@ui.page("/strategies")
@protected_page
async def strategies_page():
    user_storage = get_user_storage()
    await render_strategies_page(fetch_api, user_storage)

@ui.page("/backtesting")
@protected_page
async def backtesting_page():
    user_storage = get_user_storage()
    await render_backtesting_page(fetch_api, user_storage)

@ui.page("/analytics")
@protected_page
async def analytics_page():
    user_storage = get_user_storage()
    general_storage = get_general_storage()
    instruments = general_storage.get("instruments", {})
    # Ensure instruments are loaded if empty
    if not instruments:
        broker = user_storage.get("broker", "Zerodha")
        logger.info("Instruments not found in storage, fetching for Analytics...")
        instruments_data = await fetch_api(f"/instruments/{broker}/?exchange=NSE")
        if isinstance(instruments_data, list):
            equity_instruments = [i for i in instruments_data if i.get("segment") == "NSE" and i.get("instrument_type") == "EQ"]
            instruments = {i["trading_symbol"]: i["instrument_token"] for i in equity_instruments}
            general_storage["instruments"] = instruments
        else:
            ui.notify("Failed to load instruments for Analytics.", type="negative")
            instruments = {}
    await render_analytics_page(fetch_api, user_storage, instruments)

# --- Other Placeholder Pages --- #

@ui.page("/order-book")
@protected_page
async def order_book_page():
    user_storage = get_user_storage()
    broker = user_storage.get("broker", "Zerodha")
    ui.label("Order Book").classes("text-h5 q-pa-md")
    column_defs = [
        {"headerName": "Order ID", "field": "OrderID"},
        {"headerName": "Time", "field": "OrderTime", "valueFormatter": "value ? new Date(value).toLocaleString() : """},
        {"headerName": "Broker", "field": "Broker"},
        {"headerName": "Symbol", "field": "Symbol"},
        {"headerName": "Type", "field": "TransType"},
        {"headerName": "Qty", "field": "Quantity"},
        {"headerName": "Price", "field": "Price", "valueFormatter": "value ? value.toFixed(2) : """},
        {"headerName": "Status", "field": "Status"},
    ]
    grid = ui.aggrid({
        "columnDefs": column_defs,
        "rowData": [],
        "rowSelection": "single",
        "pagination": True,
        "paginationPageSize": 15,
        "domLayout": "autoHeight",
    }).classes("max-h-96")

    async def fetch_orders():
        orders = await fetch_api(f"/order-book/{broker}")
        if isinstance(orders, list):
            for order in orders:
                if "OrderTime" in order and isinstance(order["OrderTime"], str):
                     try:
                         order["OrderTime"] = datetime.fromisoformat(order["OrderTime"].replace("Z", "+00:00")).isoformat()
                     except ValueError:
                         pass
            await grid.update_grid_options({"rowData": orders})
        else:
            await grid.update_grid_options({"rowData": []})
            ui.notify("Failed to fetch order book.", type="warning")

    await fetch_orders()
    ui.timer(30.0, fetch_orders)

@ui.page("/positions")
@protected_page
async def positions_page():
    user_storage = get_user_storage()
    broker = user_storage.get("broker", "Zerodha")
    ui.label("Positions").classes("text-h5 q-pa-md")
    column_defs = [
        {"headerName": "Symbol", "field": "InstrumentToken"}, # TODO: Map token to symbol
        {"headerName": "Qty", "field": "Quantity"},
        {"headerName": "Avg Price", "field": "AvgPrice", "valueFormatter": "value ? value.toFixed(2) : """},
        {"headerName": "LTP", "field": "LastPrice", "valueFormatter": "value ? value.toFixed(2) : """},
        {"headerName": "P&L", "field": "PnL", "valueFormatter": "value ? value.toFixed(2) : """,
         "cellStyle": lambda params: {"color": "green"} if params.value > 0 else {"color": "red"} if params.value < 0 else {}},
        {"headerName": "Day P&L", "field": "DayPnL", "valueFormatter": "value ? value.toFixed(2) : """,
         "cellStyle": lambda params: {"color": "green"} if params.value > 0 else {"color": "red"} if params.value < 0 else {}},
    ]
    grid = ui.aggrid({
        "columnDefs": column_defs,
        "rowData": [],
        "domLayout": "autoHeight",
    }).classes("max-h-96")

    async def fetch_positions():
        positions = await fetch_api(f"/positions/{broker}")
        if isinstance(positions, list):
            # TODO: Map InstrumentToken to TradingSymbol
            await grid.update_grid_options({"rowData": positions})
        else:
            await grid.update_grid_options({"rowData": []})
            ui.notify("Failed to fetch positions.", type="warning")

    await fetch_positions()
    ui.timer(15.0, fetch_positions)

@ui.page("/portfolio")
@protected_page
async def portfolio_page():
    user_storage = get_user_storage()
    broker = user_storage.get("broker", "Zerodha")
    ui.label("Portfolio Holdings").classes("text-h5 q-pa-md")
    column_defs = [
        {"headerName": "Symbol", "field": "Symbol"},
        {"headerName": "Qty", "field": "Quantity"},
        {"headerName": "Avg Price", "field": "AvgPrice", "valueFormatter": "value ? value.toFixed(2) : """},
        {"headerName": "LTP", "field": "LastPrice", "valueFormatter": "value ? value.toFixed(2) : """},
        {"headerName": "Value", "field": "Value", "valueFormatter": "value ? value.toFixed(2) : """},
        {"headerName": "P&L", "field": "PnL", "valueFormatter": "value ? value.toFixed(2) : """,
         "cellStyle": lambda params: {"color": "green"} if params.value > 0 else {"color": "red"} if params.value < 0 else {}},
        {"headerName": "Day Change", "field": "DayChange", "valueFormatter": "value ? value.toFixed(2) : """},
        {"headerName": "Day Change %", "field": "DayChangePercentage", "valueFormatter": "value ? value.toFixed(2)+"%" : """},
    ]
    grid = ui.aggrid({
        "columnDefs": column_defs,
        "rowData": [],
        "domLayout": "autoHeight",
    }).classes("max-h-96")

    with ui.row().classes("w-full gap-4 justify-around q-my-md"):
        with ui.card().classes("flex-grow text-center"):
            ui.label("Total Investment").classes("text-caption")
            investment_label = ui.label("Loading...").classes("text-h6")
        with ui.card().classes("flex-grow text-center"):
            ui.label("Current Value").classes("text-caption")
            current_value_label = ui.label("Loading...").classes("text-h6")
        with ui.card().classes("flex-grow text-center"):
            ui.label("Overall P&L").classes("text-caption")
            pnl_label = ui.label("Loading...").classes("text-h6")

    async def fetch_portfolio():
        portfolio = await fetch_api(f"/portfolio/{broker}")
        if isinstance(portfolio, list):
            df = pd.DataFrame(portfolio)
            if not df.empty:
                df["Value"] = df["Quantity"] * df["LastPrice"]
                df["Investment"] = df["Quantity"] * df["AvgPrice"]
                df["PnL"] = df["Value"] - df["Investment"]
                await grid.update_grid_options({"rowData": df.to_dict("records")})
                total_investment = df["Investment"].sum()
                total_value = df["Value"].sum()
                total_pnl = df["PnL"].sum()
                pnl_percent = (total_pnl / total_investment * 100) if total_investment else 0
                investment_label.text = f"₹{total_investment:,.2f}"
                current_value_label.text = f"₹{total_value:,.2f}"
                pnl_label.text = f"₹{total_pnl:,.2f} ({pnl_percent:.2f}%)"
                pnl_label.classes(remove="text-positive text-negative")
                if total_pnl > 0: pnl_label.classes(add="text-positive")
                elif total_pnl < 0: pnl_label.classes(add="text-negative")
            else:
                 await grid.update_grid_options({"rowData": []})
                 investment_label.text = "₹0.00"
                 current_value_label.text = "₹0.00"
                 pnl_label.text = "₹0.00 (0.00%)"
        else:
            await grid.update_grid_options({"rowData": []})
            ui.notify("Failed to fetch portfolio.", type="warning")

    await fetch_portfolio()
    ui.timer(60.0, fetch_portfolio)

@ui.page("/mutual-funds")
@protected_page
async def mutual_funds_page():
    user_storage = get_user_storage()
    broker = user_storage.get("broker", "Zerodha")
    ui.label("Mutual Funds").classes("text-h5 q-pa-md")
    if broker != "Zerodha":
        ui.label("Mutual Funds are currently only supported for Zerodha accounts.").classes("q-pa-md text-warning")
        return
    ui.label("Mutual Funds section coming soon...").classes("q-pa-md")

# --- App Initialization --- #

@app.on_startup
async def startup():
    app.storage.user.update({
        "token": None,
        "email": None,
        "user_id": None,
        "broker": "Zerodha",
        "theme": "Dark",
        "websocket_connected": False,
    })
    app.storage.general.update({
        "instruments": {},
    })
    logger.info("User and general storage initialized.")

left_drawer: ui.left_drawer = None

ui.run(title="Algo Trader", port=8080, storage_secret="YOUR_SECRET_KEY_HERE", dark=True)
# IMPORTANT: Replace "YOUR_SECRET_KEY_HERE" with a real secret key.

