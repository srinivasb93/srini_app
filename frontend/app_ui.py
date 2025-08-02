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
from theme_manager import theme_manager, apply_page_theme, PageTheme, ThemeMode
from order_management import render_order_management
from strategies import render_strategies_page
from backtesting import render_backtesting_page
from analytics import render_analytics_page
from integration_example import integrate_with_existing_app
from orderbook import render_order_book_page
from portfolio import render_portfolio_page
from positions import render_positions_page
from livetrading import render_live_trading_page
from sip_strategy import render_sip_strategy_page
from watchlist import render_watchlist_page
from settings import render_settings_page
from dashboard import render_dashboard_page

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


# def apply_theme_from_storage():
#     try:
#         if hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None:
#             current_theme = app.storage.user.get(STORAGE_THEME_KEY, "Dark")
#
#             if current_theme == "Dark":
#                 # Enhanced dark theme
#                 ui.add_head_html("""
#                                     <style>
#                                     body {
#                                         background: linear-gradient(135deg, #0a0f23 0%, #1a1f3a 100%) !important;
#                                         color: #ffffff !important;
#                                     }
#
#                                     /* Fix all cards globally */
#                                     .q-card {
#                                         background: rgba(255, 255, 255, 0.08) !important;
#                                         color: #ffffff !important;
#                                         backdrop-filter: blur(20px);
#                                         border: 1px solid rgba(255, 255, 255, 0.1);
#                                     }
#
#                                     /* Fix tab panels */
#                                     .q-tab-panel {
#                                         background: transparent !important;
#                                         color: #ffffff !important;
#                                     }
#
#                                     /* Enhanced dashboard specific */
#                                     .enhanced-dashboard,
#                                     .enhanced-app {
#                                         background: linear-gradient(135deg, #0a0f23 0%, #1a1f3a 100%) !important;
#                                         color: #ffffff !important;
#                                     }
#                                     </style>
#                                     """)
#             else:
#                 # Enhanced light theme
#                 ui.add_head_html("""
#                     <style>
#                     body {
#                         background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
#                         color: #1a202c;
#                     }
#                     .q-header {
#                         background: rgba(255, 255, 255, 0.9) !important;
#                         backdrop-filter: blur(20px);
#                         border-bottom: 1px solid rgba(0, 0, 0, 0.1);
#                     }
#                     .enhanced-dashboard {
#                         background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
#                         color: #1a202c;
#                     }
#                     .dashboard-card {
#                         background: rgba(255, 255, 255, 0.9) !important;
#                         color: #1a202c !important;
#                     }
#                     </style>
#                     """)
#         else:
#             logger.warning("app.storage.user not available for theme application. Defaulting to Dark.")
#             ui.dark_mode().enable()
#     except Exception as e:
#         logger.error(f"Unexpected error applying theme: {e}")
#         ui.dark_mode().enable()

def apply_theme_from_storage():
    """Apply theme using the centralized theme manager"""
    theme_mode = app.storage.user.get(STORAGE_THEME_KEY, "Dark").lower()
    theme_manager.current_theme = ThemeMode(theme_mode)

# Replace the existing toggle_theme() function
def toggle_theme():
    """Toggle theme using theme manager"""
    current = theme_manager.current_theme
    new_theme = ThemeMode.LIGHT if current == ThemeMode.DARK else ThemeMode.DARK
    theme_manager.switch_theme(new_theme, app.storage.user)
    app.storage.user[STORAGE_THEME_KEY] = new_theme.value


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

# def toggle_theme():
#     current_theme = app.storage.user.get(STORAGE_THEME_KEY, "Dark")
#     new_theme = "Light" if current_theme == "Dark" else "Dark"
#     app.storage.user[STORAGE_THEME_KEY] = new_theme
#     apply_theme_from_storage()

def render_header():
    """Enhanced header with compact navigation and profile dropdown"""
    with ui.header(elevated=True).classes('justify-between items-center'):
        # Apply glassmorphism to header
        ui.add_head_html('''
        <style>
            .q-header {
                background: rgba(0, 0, 0, 0.4) !important;
                backdrop-filter: blur(20px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                padding: 0.75rem 1.5rem !important;
            }
        </style>
        ''')

        # Left side - Logo and Title
        with ui.row().classes("items-center gap-3"):
            ui.icon("candlestick_chart", size="1.75rem").classes("text-cyan-400")
            ui.label("AlgoTrade Pro").classes("text-xl font-bold text-white")
            ui.chip("LIVE", color="green").classes("text-xs animate-pulse")

        # Center - Compact Navigation
        with ui.row().classes("nav-tabs-container flex-1 mx-4"):
            # Group related pages
            nav_items = [
                ("Dashboard", "/dashboard", "dashboard"),
                ("Trading", [
                    ("Orders", "/order-management", "shopping_cart"),
                    ("Order Book", "/order-book", "menu_book"),
                    ("Positions", "/positions", "account_balance"),
                    ("Live Trading", "/live-trading", "speed")
                ]),
                ("Portfolio", [
                    ("Holdings", "/portfolio", "pie_chart"),
                    ("Watchlist", "/watchlist", "visibility"),
                    ("Analytics", "/analytics", "analytics")
                ]),
                ("Strategies", [
                    ("All Strategies", "/strategies", "psychology"),
                    ("SIP Strategy", "/sip-strategy", "savings"),
                    ("Backtesting", "/backtesting", "assessment")
                ])
            ]

            for item in nav_items:
                if isinstance(item[1], str):  # Single item
                    name, route, icon = item
                    ui.button(name, icon=icon, on_click=lambda r=route: ui.navigate.to(r)).props(
                        'flat dense no-caps').classes("nav-tab-btn text-sm")
                else:  # Dropdown group
                    group_name, sub_items = item[0], item[1]
                    with ui.button(group_name, icon="arrow_drop_down").props(
                            'flat dense no-caps').classes("nav-tab-btn text-sm"):
                        with ui.menu().classes("bg-gray-900/95 backdrop-blur-lg border border-white/10"):
                            for sub_name, sub_route, sub_icon in sub_items:
                                ui.menu_item(sub_name,
                                             on_click=lambda r=sub_route: ui.navigate.to(r)).props(
                                    f'icon={sub_icon}').classes("text-white hover:bg-white/10")

        # Right side - Status and Profile
        with ui.row().classes("items-center gap-4"):
            # Market Status
            market_open = 9 <= datetime.now().hour < 16
            status_color = "green" if market_open else "red"
            status_text = "Market Open" if market_open else "Market Closed"

            with ui.row().classes(
                    f"status-indicator bg-{status_color}-900/20 border border-{status_color}-500/30 px-3 py-1 rounded-full"):
                ui.icon("circle", size="0.5rem").classes(f"text-{status_color}-500")
                ui.label(status_text).classes("text-sm text-white")

            # Connection Status
            with ui.row().classes("status-indicator bg-cyan-900/20 border border-cyan-500/30 px-3 py-1 rounded-full"):
                ui.icon("wifi", size="1rem").classes("text-cyan-400")
                ui.label(f"Connected: {app.storage.user.get(STORAGE_BROKER_KEY)}").classes("text-sm text-white") if app.storage.user.get(
                    STORAGE_BROKER_KEY) else ui.label("Not Connected").classes("text-sm text-red-400")

            # Current Time
            time_label = ui.label().classes("text-sm text-gray-300 font-mono")

            def update_time():
                time_label.text = datetime.now().strftime("%H:%M:%S IST")

            ui.timer(1, update_time)

            # Theme toggle
            ui.button(icon="brightness_6", on_click=toggle_theme).props(
                "flat round dense").classes("text-white hover:bg-white/10")

            # Profile Dropdown
            with ui.element('div').classes('profile-dropdown'):
                with ui.button(icon="account_circle").props("flat round dense").classes(
                        "text-white hover:bg-white/10") as profile_btn:

                    # Create dropdown menu
                    menu = ui.menu().classes(
                        "profile-dropdown-menu bg-gray-900/95 backdrop-blur-lg border border-white/10"
                    )

                    with menu:
                        # User info
                        user_email = app.storage.user.get('email', 'user@example.com')
                        broker = app.storage.user.get(STORAGE_BROKER_KEY, 'Zerodha')

                        with ui.column().classes("p-3 border-b border-white/10"):
                            ui.label(user_email).classes("text-sm font-semibold text-white")
                            ui.label(f"Broker: {broker}").classes("text-xs text-gray-400")

                        # Menu items
                        with ui.column().classes("py-2"):
                            # View Profile
                            with ui.item(on_click=lambda: ui.navigate.to('/profile')).classes(
                                    "profile-dropdown-item hover:bg-white/10"):
                                ui.icon("person", size="1.2rem").classes("text-gray-400")
                                ui.label("View Profile").classes("text-sm")

                            # Settings
                            with ui.item(on_click=lambda: ui.navigate.to('/settings')).classes(
                                    "profile-dropdown-item hover:bg-white/10"):
                                ui.icon("settings", size="1.2rem").classes("text-gray-400")
                                ui.label("Settings").classes("text-sm")

                            # Divider
                            ui.separator().classes("my-2 bg-white/10")

                            # Logout
                            async def handle_logout():
                                try:
                                    if hasattr(app, 'storage') and hasattr(app.storage, 'user'):
                                        app.storage.user.clear()
                                except Exception as e:
                                    logger.error(f"Error during logout: {e}")
                                ui.navigate.to('/')
                                ui.notify("Logged out successfully", type="positive")

                            with ui.item(on_click=handle_logout).classes(
                                    "profile-dropdown-item hover:bg-white/10"):
                                ui.icon("logout", size="1.2rem").classes("text-red-400")
                                ui.label("Logout").classes("text-sm text-red-400")


@ui.page('/')
async def login_page(client: Client):
    await client.connected()
    try:
        if app.storage.user.get(STORAGE_TOKEN_KEY):
            ui.navigate.to('/dashboard')
            return
    except Exception as e:
        logger.error(f"Error accessing app.storage.user in login_page: {e}")

    apply_page_theme(PageTheme.LOGIN, app.storage.user)

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
        with ui.card().classes('w-96 shadow-2xl p-8 rounded-lg'):
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

    # Authentication check
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return

    # Apply enhanced theme and render header
    apply_page_theme(PageTheme.DASHBOARD, app.storage.user)
    render_header()

    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")

    # Display WebSocket messages
    with ui.column().classes("w-full p-4"):
        try:
            for message in list(websocket_messages):
                ui.notify(message, type="info" if "connected" in message.lower() else "warning")
                websocket_messages.remove(message)
        except Exception as e:
            logger.error(f"Error handling WebSocket messages: {e}")

        # Render the enhanced dashboard using the new module
        try:
            await render_dashboard_page(fetch_api, app.storage.user, get_cached_instruments)
        except Exception as e:
            logger.error(f"Error rendering enhanced dashboard: {e}")

            # Fallback to basic dashboard if enhanced version fails
            broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
            ui.label(f"Dashboard Error - Fallback Mode").classes("text-xl text-red-500 p-4")
            ui.label(f"Error: {str(e)}").classes("text-gray-500 p-4")

            # Basic metrics as fallback
            with ui.row().classes("w-full gap-4 p-4"):
                with ui.card().classes("p-4"):
                    ui.label("Available Funds").classes("text-lg font-medium")
                    ui.label("Loading...").classes("text-2xl font-bold mt-1")

                with ui.card().classes("p-4"):
                    ui.label("Portfolio Value").classes("text-lg font-medium")
                    ui.label("Loading...").classes("text-2xl font-bold mt-1")

                with ui.card().classes("p-4"):
                    ui.label("Open Positions").classes("text-lg font-medium")
                    ui.label("Loading...").classes("text-2xl font-bold mt-1")


@ui.page('/order-management')
async def order_management_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.TRADING, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_order_management(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/analytics')
async def analytics_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.ANALYTICS, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await integrate_with_existing_app(fetch_api, app.storage.user, get_cached_instruments, broker)
    await render_analytics_page(fetch_api, app.storage.user, await get_cached_instruments(broker), broker)


@ui.page('/strategies')
async def strategies_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.DEFAULT, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_strategies_page(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/sip-strategy')
async def sip_strategies_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.SIP_STRATEGY, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_sip_strategy_page(fetch_api, app.storage.user)


@ui.page('/backtesting')
async def backtesting_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.BACKTESTING, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_backtesting_page(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/order-book')
async def order_book_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.ORDERBOOK, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_order_book_page(fetch_api, app.storage.user, broker)


@ui.page('/portfolio')
async def portfolio_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.PORTFOLIO, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_portfolio_page(fetch_api, app.storage.user, broker)


@ui.page('/positions')
async def positions_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.POSITIONS, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_positions_page(fetch_api, app.storage.user, broker)


@ui.page('/live-trading')
async def live_trading_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.LIVE_TRADING, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_live_trading_page(fetch_api, app.storage.user, broker)


@ui.page('/watchlist')
async def watchlist_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.WATCHLIST, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_watchlist_page(fetch_api, app.storage.user, get_cached_instruments, broker)


@ui.page('/settings')
async def settings_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.SETTINGS, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_settings_page(fetch_api, app.storage.user, apply_theme_from_storage)

@ui.page('/strategy-performance')
async def strategy_performance_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_page_theme(PageTheme.DEFAULT, app.storage.user)
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
    # Add static files support for CSS
    # ui.add_css('static/styles.css')

    ui.run(title="AlgoTrade Pro - Advanced Trading Platform",
           port=8080,
           reload=True,
           uvicorn_reload_dirs='.',
           uvicorn_reload_includes='*.py',
           storage_secret=storage_secret_key,
           favicon="🚀")