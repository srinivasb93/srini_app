import logging
from nicegui import ui, Client, app
import aiohttp
import asyncio
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import json
import time
import secrets
from collections import deque

# Import module functions
from order_management import render_order_management
from strategies import render_strategies_page
from backtesting import render_backtesting_page
from analytics import render_analytics_page
from mutual_funds import render_mutual_funds_page
from orderbook import render_order_book_page
from portfolio import render_portfolio_page
from positions import render_positions_page
from livetrading import render_live_trading_page
from watchlist import render_watchlist_page
from settings import render_settings_page

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

# Storage constants
STORAGE_TOKEN_KEY = 'auth_token'
STORAGE_USER_ID_KEY = 'user_id'
STORAGE_BROKER_KEY = 'default_broker'
STORAGE_THEME_KEY = 'app_theme'
STORAGE_WATCHLIST_KEY = 'user_watchlist'
STORAGE_INSTRUMENTS_CACHE_KEY_PREFIX = 'instruments_cache_'

# WebSocket message queue
websocket_messages = deque()


async def fetch_api(endpoint, method="GET", data=None, params=None, retries=3, backoff=1):
    token = app.storage.user.get(STORAGE_TOKEN_KEY) if hasattr(app, 'storage') and hasattr(app.storage, 'user') else None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request_kwargs = {"headers": headers}

    if method == "POST" and "auth/login" in endpoint:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        request_kwargs["data"] = data
    elif data is not None:
        request_kwargs["json"] = data
    if params:
        request_kwargs["params"] = params

    url = f"{BASE_URL}{endpoint}"
    logger.info(f"Calling API: {method} {url}")

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, **request_kwargs) as response:
                    logger.debug(f"API call: {method} {url}, Status: {response.status}")
                    if response.status == 401:
                        if hasattr(app, 'storage') and hasattr(app.storage, 'user'):
                            app.storage.user.clear()
                        ui.navigate.to('/')
                        ui.notify("Session expired. Please log in again.", type="negative")
                        return {"error": {"code": "UNAUTHORIZED", "message": "Session expired"}, "status": 401}
                    if response.status == 404 and endpoint.startswith("/profile/"):
                        return {"error": {"code": "NOT_FOUND", "message": "Profile not found"}, "status": 404}
                    if response.status >= 400:
                        try:
                            error_data = await response.json()
                            detail = error_data.get('detail', 'Unknown API error')
                            logger.error(f"API Error: {detail}")
                            return {"error": {"code": "API_ERROR", "message": detail}, "status": response.status}
                        except Exception as e:
                            logger.error(f"Error parsing API error: {e}")
                            return {"error": {"code": "UNKNOWN", "message": "Unknown API error"},
                                    "status": response.status}
                    if response.content_type == 'application/json':
                        return await response.json()
                    return await response.text()
        except aiohttp.ClientConnectorError as e:
            logger.error(f"API connection failed: {str(e)}")
            if attempt < retries - 1:
                await asyncio.sleep(backoff * (2 ** attempt))
                continue
            return {"error": {"code": "CONNECTION_FAILED", "message": "Connection failed"}, "status": 503}
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            return {"error": {"code": "UNKNOWN", "message": str(e)}, "status": 500}
    return {"error": {"code": "RETRIES_EXCEEDED", "message": "Max retries exceeded"}, "status": 429}


async def connect_websocket(max_retries=5, initial_backoff=2):
    user_id = app.storage.user.get(STORAGE_USER_ID_KEY, "default_user_id")
    token = app.storage.user.get(STORAGE_TOKEN_KEY)
    if not token:
        logger.warning("No token found for WebSocket connection.")
        return

    ws_url = f"{BASE_URL.replace('http', 'ws')}/ws/orders/{user_id}"
    retry_count = 0

    while retry_count < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    logger.info(f"WebSocket connected to {ws_url}")
                    websocket_messages.append("Real-time order updates connected.")
                    ui.notify("WebSocket connected", type="positive")
                    retry_count = 0  # Reset on successful connection
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                message = f"Order Update: {data.get('trading_symbol', 'Unknown Symbol')} - {data.get('status', 'Status N/A')}"
                                websocket_messages.append(message)
                                ui.notify(message, type="info")
                            except json.JSONDecodeError:
                                logger.error(f"WebSocket non-JSON message: {msg.data}")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info("WebSocket connection closed.")
                            break
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                backoff = initial_backoff * (2 ** retry_count)
                logger.info(f"Retrying WebSocket connection in {backoff} seconds...")
                ui.notify(f"WebSocket disconnected, retrying in {backoff} seconds...", type="warning")
                await asyncio.sleep(backoff)
            else:
                websocket_messages.append("WebSocket connection failed permanently.")
                ui.notify("WebSocket connection failed permanently.", type="negative")
                break


def apply_theme_from_storage():
    try:
        if hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None:
            theme = app.storage.user.get(STORAGE_THEME_KEY, "Dark")
            if theme == "Dark":
                ui.dark_mode().enable()
            else:
                ui.dark_mode().disable()
        else:
            logger.warning("app.storage.user not available for theme application. Defaulting to Dark.")
            ui.dark_mode().enable()
    except Exception as e:
        logger.error(f"Unexpected error applying theme: {e}")
        ui.dark_mode().enable()


async def get_cached_instruments(broker, exchange_filter="NSE", force_refresh=False, cache_ttl=86400):
    cache_key = f"{STORAGE_INSTRUMENTS_CACHE_KEY_PREFIX}{broker}_{exchange_filter}"
    try:
        if not force_refresh:
            cached = app.storage.user.get(cache_key, {})
            if cached.get("data") and cached.get("timestamp", 0) + cache_ttl > time.time():
                return cached["data"]

        logger.info(f"Fetching instruments for {broker}, Exchange: {exchange_filter}...")
        response = await fetch_api(f"/instruments/{broker}/", params={"exchange": exchange_filter})
        if response and isinstance(response, list):
            instruments_map = {
                inst["trading_symbol"]: inst["instrument_token"]
                for inst in response
                if "trading_symbol" in inst and "instrument_token" in inst
            }
            app.storage.user[cache_key] = {"data": instruments_map, "timestamp": time.time()}
            logger.info(f"Fetched {len(instruments_map)} instruments for {broker} ({exchange_filter}).")
            return instruments_map
        logger.error(f"Failed to fetch instruments for {broker} ({exchange_filter}).")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in get_cached_instruments: {e}")
        return {}

def toggle_theme():
    current_theme = app.storage.user.get(STORAGE_THEME_KEY, "Dark")
    new_theme = "Light" if current_theme == "Dark" else "Dark"
    app.storage.user[STORAGE_THEME_KEY] = new_theme
    apply_theme_from_storage()

def render_header():
    with ui.header(elevated=True).classes('justify-between text-white items-center q-pa-sm'):
        ui.label("Xpress Trader").classes("text-2xl font-semibold")
        with ui.row().classes("items-center"):
            pages = ["Dashboard", "Order Management", "Order Book", "Positions",
                     "Portfolio", "Mutual Funds", "Analytics", "Strategies",
                     "Backtesting", "Live Trading", "Watchlist", "Settings"]
            for page_name in pages:
                route = f"/{page_name.lower().replace(' ', '-')}"
                ui.button(page_name, on_click=lambda r=route: ui.navigate.to(r)).props('flat color=white dense')
            ui.button(icon="brightness_6", on_click=toggle_theme).props("flat color=white round dense")

        async def handle_logout():
            try:
                if hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None:
                    app.storage.user.clear()
            except Exception as e:
                logger.error(f"Error during logout: {e}")
            ui.navigate.to('/')
            ui.notify("Logged out successfully.", type="positive")

        ui.button("Logout", on_click=handle_logout).props("flat color=white dense")


@ui.page('/')
async def login_page(client: Client):
    await client.connected()
    try:
        if app.storage.user.get(STORAGE_TOKEN_KEY):
            ui.navigate.to('/dashboard')
            return
    except Exception as e:
        logger.error(f"Error accessing app.storage.user in login_page: {e}")

    apply_theme_from_storage()

    async def handle_login():
        if not email.value or not password.value:
            ui.notify("Email and password are required.", type="warning")
            return
        data = {"username": email.value, "password": password.value, "grant_type": "password"}
        response = await fetch_api("/auth/login", method="POST", data=data)
        if response and "access_token" in response:
            try:
                app.storage.user[STORAGE_TOKEN_KEY] = response["access_token"]
                user_profile = await fetch_api("/users/me")
                if user_profile and "id" in user_profile:
                    app.storage.user[STORAGE_USER_ID_KEY] = user_profile["id"]
                else:
                    app.storage.user[STORAGE_USER_ID_KEY] = user_profile.get("email",
                                                                             "default_user_id") if user_profile else "default_user_id"

                ui.notify("Login successful!", type="positive")
                primary_broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
                profile_response = await fetch_api(f"/profile/{primary_broker}")
                if not profile_response or not profile_response.get("name"):
                    ui.notify(f"{primary_broker} not connected or profile incomplete. Please check Settings.",
                              type="warning", multi_line=True)
                else:
                    ui.notify(f"{primary_broker} connected: {profile_response.get('name', '')}", type="info")

                ui.navigate.to("/dashboard")
                asyncio.create_task(connect_websocket())
            except Exception as e:
                logger.error(f"Unexpected error during login: {e}")
                ui.notify("An unexpected error occurred after login.", type="negative")
        else:
            ui.notify("Login failed. Please check your credentials.", type="negative")

    async def handle_register():
        if new_password.value != confirm_password.value:
            ui.notify("Passwords do not match!", type="negative")
            return
        if not all([new_username.value, new_password.value]):
            ui.notify("Email and Password are required for registration.", type="warning")
            return
        data = {
            "email": new_username.value, "password": new_password.value,
            "upstox_api_key": upstox_api_key.value or None,
            "upstox_api_secret": upstox_api_secret.value or None,
            "zerodha_api_key": zerodha_api_key.value or None,
            "zerodha_api_secret": zerodha_api_secret.value or None
        }
        response = await fetch_api("/auth/register", method="POST", data=data)
        if response:
            ui.notify("Registration successful! Please log in.", type="positive")
            tabs.set_value("Login")

    with ui.column().classes('absolute-center items-center gap-4'):
        ui.label("Algo Trader").classes('text-3xl font-bold text-center mb-4')
        with ui.card().classes('w-96 shadow-xl p-8 rounded-lg'):
            with ui.tabs().props("dense").classes("w-full") as tabs:
                ui.tab(name="Login", label="Login")
                ui.tab(name="Sign Up", label="Sign Up")

            with ui.tab_panels(tabs, value="Login").classes("w-full pt-4") as tabs_panels:
                with ui.tab_panel("Login"):
                    email = ui.input("Email").props("outlined dense clearable").classes("w-full")
                    password = ui.input("Password").props("outlined dense type=password clearable").classes("w-full")
                    ui.button("Login", on_click=handle_login).props("color=primary").classes("w-full mt-4 py-2")
                with ui.tab_panel("Sign Up"):
                    new_username = ui.input("Email").props("outlined dense clearable").classes("w-full")
                    new_password = ui.input("Password").props("outlined dense type=password clearable").classes(
                        "w-full")
                    confirm_password = ui.input("Confirm Password").props(
                        "outlined dense type=password clearable").classes("w-full")
                    ui.label("Broker API Keys (Optional, add in Settings later)").classes("text-xs text-gray-500 mt-2")
                    upstox_api_key = ui.input("Upstox API Key").props("outlined dense clearable").classes("w-full")
                    upstox_api_secret = ui.input("Upstox API Secret").props(
                        "outlined dense type=password clearable").classes("w-full")
                    zerodha_api_key = ui.input("Zerodha API Key").props("outlined dense clearable").classes("w-full")
                    zerodha_api_secret = ui.input("Zerodha API Secret").props(
                        "outlined dense type=password clearable").classes("w-full")
                    ui.button("Sign Up", on_click=handle_register).props("color=primary").classes("w-full mt-4 py-2")


@ui.page('/dashboard')
async def dashboard_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"{broker} Trading Dashboard").classes("text-2xl font-semibold p-4")

    # Display WebSocket messages
    with ui.column().classes("w-full p-4"):
        for message in list(websocket_messages):
            ui.notify(message, type="info" if "connected" in message.lower() else "warning")
            websocket_messages.remove(message)

    metrics_container = ui.grid(columns='repeat(auto-fit, minmax(250px, 1fr))').classes("w-full p-4 gap-4")
    market_watch_container = ui.column().classes("w-full p-4 gap-2")
    chart_container = ui.column().classes("w-full p-4")

    async def fetch_and_update_dashboard():
        funds_data = await fetch_api(f"/funds/{broker}")
        portfolio_data = await fetch_api(f"/portfolio/{broker}") or []
        positions_data = await fetch_api(f"/positions/{broker}") or []
        all_instruments_map = await get_cached_instruments(broker)

        metrics_container.clear()
        with metrics_container:
            with ui.card().classes("p-4 items-center text-center shadow-md rounded-lg"):
                ui.label("Available Funds").classes("text-lg font-medium")
                funds_val = "N/A"
                if funds_data and isinstance(funds_data, dict):
                    equity = funds_data.get('equity', {})
                    available = equity.get('available', 0.0)
                    funds_val = f"₹{float(available):,.2f}"
                ui.label(funds_val).classes("text-2xl font-bold mt-1")
            with ui.card().classes("p-4 items-center text-center shadow-md rounded-lg"):
                ui.label("Portfolio Value").classes("text-lg font-medium")
                portfolio_val_num = sum(
                    h.get("Quantity", 0) * h.get("LastPrice", 0) for h in portfolio_data if isinstance(h, dict))
                ui.label(f"₹{portfolio_val_num:,.2f}").classes("text-2xl font-bold mt-1")
            with ui.card().classes("p-4 items-center text-center shadow-md rounded-lg"):
                ui.label("Open Positions").classes("text-lg font-medium")
                open_pos_count = len([p for p in positions_data if isinstance(p, dict) and p.get("Quantity", 0) != 0])
                ui.label(str(open_pos_count)).classes("text-2xl font-bold mt-1")

        market_watch_container.clear()
        with market_watch_container:
            ui.label("Market Watch").classes("text-xl font-semibold mb-2")

            async def update_ltp(instrument_symbol_input, ltp_label_display):
                if instrument_symbol_input.value and instrument_symbol_input.value in all_instruments_map:
                    instrument_token = all_instruments_map[instrument_symbol_input.value]
                    ltp_response = await fetch_api(f"/ltp/{broker}", params={"instruments": instrument_token})
                    if ltp_response and isinstance(ltp_response, list) and ltp_response:
                        ltp_item = ltp_response[0]
                        ltp_label_display.text = f"LTP: ₹{ltp_item.get('last_price', 'N/A'):,.2f}"
                    else:
                        ltp_label_display.text = "LTP: N/A"
                else:
                    ltp_label_display.text = "LTP: N/A"

            with ui.row().classes("items-center gap-2 w-full"):
                instrument_input = ui.select(options=list(all_instruments_map.keys()),
                                             clearable=True, with_input=True, label="Select Instrument for LTP") \
                    .props("outlined dense").classes("flex-grow")
                ltp_label = ui.label("LTP: N/A").classes("text-lg p-2 bg-gray-100 dark:bg-gray-700 rounded-md shadow")
            instrument_input.on('update:model-value',
                                lambda: asyncio.create_task(update_ltp(instrument_input, ltp_label)))

        chart_container.clear()
        with chart_container:
            if portfolio_data:
                try:
                    labels = [p["Symbol"] for p in portfolio_data if
                              isinstance(p, dict) and "Symbol" in p and p.get("Quantity", 0) * p.get("LastPrice",
                                                                                                     0) > 0]
                    values = [p["Quantity"] * p["LastPrice"] for p in portfolio_data if
                              isinstance(p, dict) and "Quantity" in p and "LastPrice" in p and p.get("Quantity",
                                                                                                     0) * p.get(
                                  "LastPrice", 0) > 0]
                    if labels and values:
                        fig_pie = go.Figure(data=[
                            go.Pie(labels=labels, values=values, hole=.3, textinfo='percent+label',
                                   pull=[0.05] * len(labels))])
                        fig_pie.update_layout(title_text='Portfolio Allocation', showlegend=True,
                                              legend_orientation="h", legend_yanchor="bottom", legend_y=1.02)
                        ui.plotly(fig_pie).classes("w-full h-96 rounded-lg shadow-md")
                    else:
                        ui.label("Not enough data for portfolio allocation chart.").classes("text-gray-500")
                except Exception as e:
                    logger.error(f"Error creating pie chart: {e}")
                    ui.label("Could not load portfolio allocation chart.").classes("text-red-500")
            else:
                ui.label("No portfolio data available for chart.").classes("text-gray-500")

    await fetch_and_update_dashboard()
    ui.timer(30, fetch_and_update_dashboard)


@ui.page('/order-management')
async def order_management_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_order_management(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/analytics')
async def analytics_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_analytics_page(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/strategies')
async def strategies_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_strategies_page(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/backtesting')
async def backtesting_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_backtesting_page(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/mutual-funds')
async def mutual_funds_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_mutual_funds_page(fetch_api, broker)


@ui.page('/order-book')
async def order_book_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_order_book_page(fetch_api, app.storage.user, broker)


@ui.page('/portfolio')
async def portfolio_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_portfolio_page(fetch_api, app.storage.user, broker)


@ui.page('/positions')
async def positions_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_positions_page(fetch_api, app.storage.user, broker)


@ui.page('/live-trading')
async def live_trading_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_live_trading_page(fetch_api, app.storage.user, broker)


@ui.page('/watchlist')
async def watchlist_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_watchlist_page(fetch_api, app.storage.user, get_cached_instruments, broker)


@ui.page('/settings')
async def settings_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_settings_page(fetch_api, app.storage.user, apply_theme_from_storage)

@ui.page('/strategy-performance')
async def strategy_performance_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label("Strategy Performance Dashboard").classes("text-h5 q-pa-md")
    performance_data = await fetch_api(f"/strategies/{broker}/performance")
    if performance_data:
        with ui.grid(columns=3).classes("w-full gap-4"):
            for strategy_id, metrics in performance_data.items():
                with ui.card().classes("p-4"):
                    ui.label(strategy_id).classes("text-subtitle1")
                    ui.label(f"Total PnL: ₹{metrics['pnl']['sum']:.2f}")
                    ui.label(f"Trade Count: {metrics['pnl']['count']}")
    else:
        ui.label("No performance data available.").classes("text-warning")


@app.on_connect
async def on_client_connect(client: Client):
    try:
        apply_theme_from_storage()
    except Exception as e:
        logger.error(f"Error applying theme on client connect: {e}")


if __name__ in {"__main__", "__mp_main__"}:
    storage_secret_key = "my_super_secret_key_for_testing_123_please_change_for_prod"
    ui.run(title="Algo Trader",
           port=8080,
           reload=True,
           uvicorn_reload_dirs='.',
           uvicorn_reload_includes='*.py',
           storage_secret=storage_secret_key)