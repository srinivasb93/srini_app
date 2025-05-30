import logging
from nicegui import ui, Client, app
import aiohttp
import asyncio
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import json
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

# --- Constants for ui.storage ---
STORAGE_TOKEN_KEY = 'auth_token'
STORAGE_USER_ID_KEY = 'user_id'
STORAGE_BROKER_KEY = 'default_broker'  # Will remain in app.storage.user (persistent)
STORAGE_THEME_KEY = 'app_theme'  # Will remain in app.storage.user (persistent)
STORAGE_WATCHLIST_KEY = 'user_watchlist'  # Will remain in app.storage.user (persistent)
STORAGE_INSTRUMENTS_CACHE_KEY_PREFIX = 'instruments_cache_'  # Can use app.storage.session if available, or fallback


# --- API Helper ---
async def fetch_api(endpoint, method="GET", data=None, params=None):
    token = None
    try:
        # PRIORITIZE app.storage.user for token as it's server-side session based
        if hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None:
            token = app.storage.user.get(STORAGE_TOKEN_KEY)
        else:
            logger.warning("app.storage.user not available when trying to fetch API token.")
    except Exception as e:
        logger.error(f"Unexpected error fetching token from app.storage.user: {e}")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request_kwargs = {"headers": headers}

    if method == "POST" and "auth/login" in endpoint:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        request_kwargs["data"] = data
    elif data is not None:
        request_kwargs["json"] = data

    if params is not None:
        request_kwargs["params"] = params

    url = f"{BASE_URL}{endpoint}"
    logger.debug(f"Calling API: {method} {url} with data: {data} and params: {params}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **request_kwargs) as response:
                logger.debug(f"API call: {method} {url}, Status: {response.status}")
                if response.status == 401:
                    ui.notify("Authentication failed or token expired. Please log in.", type="negative")
                    try:
                        if hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None:
                            app.storage.user.clear()  # Clear server-side session data related to user
                    except Exception as e_clear:
                        logger.error(f"Error clearing app.storage.user on 401: {e_clear}")
                    ui.navigate.to("/")
                    return None
                if response.status == 404 and endpoint.startswith("/profile/"):
                    ui.notify(f"Broker configuration not found for {endpoint.split('/')[-1]}.", type="warning")
                    return None

                if response.status >= 400:
                    try:
                        error_data = await response.json()
                        detail = error_data.get('detail', 'Unknown API error')
                        if isinstance(detail, list) and detail:
                            detail_msg = detail[0]
                            loc_info = detail_msg.get('loc', ['body', 'Error'])
                            field_name = loc_info[-1] if len(loc_info) > 1 else loc_info[0]
                            detail = f"{field_name}: {detail_msg.get('msg', 'Invalid input')}"
                        ui.notify(f"API Error: {detail}", type="negative", timeout=7000, multi_line=True)
                    except Exception as e_parse:
                        logger.error(f"Error parsing API error response JSON: {e_parse}")
                        ui.notify(f"API Error (Status {response.status}): {response.reason}", type="negative",
                                  multi_line=True)
                    return None

                if response.content_type == 'application/json':
                    return await response.json()
                return await response.text()
    except aiohttp.ClientConnectorError as e_connect:
        logger.error(f"API connection failed: {str(e_connect)}")
        ui.notify("Cannot connect to the API server. Please ensure it's running and accessible.", type="negative",
                  multi_line=True, timeout=10000)
        return None
    except Exception as e_req:
        logger.error(f"API request failed unexpectedly: {str(e_req)}")
        ui.notify(f"An unexpected error occurred: {str(e_req)}", type="negative", multi_line=True)
        return None


# --- WebSocket Client ---
async def connect_websocket():
    user_id = "default_user"
    token = None
    try:
        if hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None:
            user_id = app.storage.user.get(STORAGE_USER_ID_KEY, "default_user")
            token = app.storage.user.get(STORAGE_TOKEN_KEY)
        if not token:
            logger.warning("No token found for WebSocket connection (from app.storage.user).")
            return
    except Exception as e:
        logger.error(f"Unexpected error getting user_id/token for WebSocket from app.storage.user: {e}")
        return

    ws_url = f"{BASE_URL.replace('http', 'ws')}/ws/orders/{user_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                logger.info(f"WebSocket connected to {ws_url}")
                ui.notify("Real-time order updates connected.", type="positive")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            logger.debug(f"WebSocket message: {data}")
                            if isinstance(data, list) and data:
                                order_info = data[0]
                                ui.notify(
                                    f"Order Update: {order_info.get('trading_symbol', 'Unknown Symbol')} - {order_info.get('status', 'Status N/A')}",
                                    type="info")
                            elif isinstance(data, dict):
                                ui.notify(
                                    f"Order Update: {data.get('trading_symbol', 'Unknown Symbol')} - {data.get('status', 'Status N/A')}",
                                    type="info")
                        except json.JSONDecodeError:
                            logger.error(f"WebSocket non-JSON message: {msg.data}")
                        except Exception as e_ws_msg:
                            logger.error(f"Error processing WebSocket message: {e_ws_msg}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket connection error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.info("WebSocket connection closed.")
                        break
    except Exception as e_ws_conn:
        logger.error(f"WebSocket connection failed: {str(e_ws_conn)}")
        ui.notify("WebSocket connection failed. Real-time updates may be affected.", type="warning")


# --- Theme Styling ---
def apply_theme_from_storage():
    try:
        if hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None:
            theme = app.storage.user.get(STORAGE_THEME_KEY, "Dark")
            if theme == "Dark":
                ui.dark_mode().enable()
            else:
                ui.dark_mode().disable()
        else:  # Fallback if app.storage.user is not available (e.g. very early in lifecycle)
            logger.warning("app.storage.user not available for theme application. Defaulting to Dark.")
            ui.dark_mode().enable()
    except Exception as e_theme_other:
        logger.error(f"Unexpected error applying theme: {e_theme_other}")
        ui.dark_mode().enable()

    # --- Instrument Cache ---


async def get_cached_instruments(broker_override=None, exchange_filter="NSE", force_refresh=False):
    instruments_map = {}
    try:
        # Broker preference from app.storage.user (persistent)
        broker = broker_override or app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
        cache_key = f"{STORAGE_INSTRUMENTS_CACHE_KEY_PREFIX}{broker}_{exchange_filter}"

        # Instrument cache in app.storage.session (browser sessionStorage)
        can_access_session_storage = hasattr(app, 'storage') and hasattr(app.storage,
                                                                         'session') and app.storage.session is not None

        if can_access_session_storage:
            instruments_map = app.storage.session.get(cache_key, {})
        else:
            logger.warning("app.storage.session not available for instrument cache GET. Will force refresh if needed.")
            force_refresh = True

        if not instruments_map or force_refresh:
            logger.info(f"Fetching instruments for {broker}, Exchange: {exchange_filter} from API...")
            response = await fetch_api(f"/instruments/{broker}/", params={"exchange": exchange_filter})

            if response and isinstance(response, list):
                instruments_map = {
                    inst["trading_symbol"]: inst["instrument_token"]
                    for inst in response
                    if "trading_symbol" in inst and "instrument_token" in inst
                }
                if can_access_session_storage:
                    app.storage.session[cache_key] = instruments_map
                logger.info(
                    f"Fetched {len(instruments_map)} instruments for {broker} ({exchange_filter}). Cached if session storage was available.")
            else:
                logger.error(f"Failed to fetch instruments for {broker} ({exchange_filter}). Response: {response}")
        return instruments_map
    except AttributeError as e_storage:  # Catch if app.storage.user itself is the issue
        logger.error(f"app.storage.user not available in get_cached_instruments: {e_storage}")
        return {}
    except Exception as e_instr:
        logger.error(f"Unexpected error in get_cached_instruments: {e_instr}")
        return {}


# --- Common UI Components ---
def render_header():
    with ui.header(elevated=True).classes('justify-between text-white items-center q-pa-sm'):
        ui.label("Algo Trader").classes("text-2xl font-semibold")
        with ui.row().classes("items-center"):
            pages = ["Dashboard", "Order Management", "Order Book", "Positions",
                     "Portfolio", "Mutual Funds", "Analytics", "Strategies",
                     "Backtesting", "Live Trading", "Watchlist", "Settings"]
            for page_name in pages:
                route = f"/{page_name.lower().replace(' ', '-')}"
                ui.button(page_name, on_click=lambda r=route: ui.navigate.to(r)).props('flat color=white dense')

        async def handle_logout():
            try:  # Logout clears server-side user session
                if hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None:
                    app.storage.user.clear()
            except Exception as e:  # Use broader exception for storage access
                logger.error(f"Error clearing app.storage.user during logout: {e}")

            # Also attempt to clear client-side session storage if available, for good measure
            try:
                if hasattr(app, 'storage') and hasattr(app.storage, 'session') and app.storage.session is not None:
                    app.storage.session.clear()
            except Exception as e_sess_clear:
                logger.warning(f"Could not clear app.storage.session on logout (might be okay): {e_sess_clear}")

            ui.navigate.to('/')
            ui.notify("Logged out successfully.", type="positive")

        ui.button("Logout", on_click=handle_logout).props("flat color=white dense")


# --- Page Definitions ---
@ui.page('/')
async def login_page(client: Client):
    await client.connected()

    try:
        # Check token from app.storage.user (server-side session)
        if hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None:
            if app.storage.user.get(STORAGE_TOKEN_KEY):
                ui.navigate.to('/dashboard')
                return
        else:  # This case should ideally not happen if storage_secret is set
            logger.critical(
                "app.storage.user is not available on login page. SessionMiddleware might not be active despite storage_secret.")

    except Exception as e:
        logger.error(f"Error accessing app.storage.user in login_page: {e}")

    try:
        apply_theme_from_storage()
    except Exception as e:
        logger.error(f"Error applying theme in login_page: {e}")

    async def handle_login():
        if not email.value or not password.value:
            ui.notify("Email and password are required.", type="warning");
            return
        data = {"username": email.value, "password": password.value, "grant_type": "password"}
        response = await fetch_api("/auth/login", method="POST", data=data)
        if response and "access_token" in response:
            try:
                if not (hasattr(app, 'storage') and hasattr(app.storage, 'user') and app.storage.user is not None):
                    logger.critical(
                        "app.storage.user is STILL not available after successful API login. Token cannot be stored.")
                    ui.notify("Critical error: Server session storage unavailable. Cannot proceed.", type="negative",
                              timeout=10000)
                    return

                app.storage.user[STORAGE_TOKEN_KEY] = response["access_token"]
                user_profile = await fetch_api(
                    "/users/me")  # This API call will now use the token from app.storage.user
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

            except Exception as e_login_success:
                logger.error(f"Unexpected error during login success handling: {e_login_success}")
                ui.notify("An unexpected error occurred after login.", type="negative")

    async def handle_register():
        if new_password.value != confirm_password.value:
            ui.notify("Passwords do not match!", type="negative");
            return
        if not all([new_username.value, new_password.value]):
            ui.notify("Email and Password are required for registration.", type="warning");
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
            tabs.set_value("Login")  # Switch to login tab by its name

    with ui.column().classes('absolute-center items-center gap-4'):
        ui.label("Algo Trader").classes('text-3xl font-bold text-center mb-4')
        with ui.card().classes('w-96 shadow-xl p-8 rounded-lg'):
            with ui.tabs().classes("w-full").props("dense").on("update:model-value",
                                                               lambda e: tabs_panels.set_value(e.value)) as tabs:
                ui.tab(name="Login", label="Login")
                ui.tab(name="Sign Up", label="Sign Up")

            with ui.tab_panels(tabs, value="Login").on("update:model-value", lambda e: tabs.set_value(e.value)).classes(
                    "w-full pt-4") as tabs_panels:
                with ui.tab_panel(name="Login"):
                    email = ui.input("Email").props("outlined dense clearable").classes("w-full")
                    password = ui.input("Password").props("outlined dense type=password clearable").classes("w-full")
                    ui.button("Login", on_click=handle_login).props("color=primary").classes("w-full mt-4 py-2")
                with ui.tab_panel(name="Sign Up"):
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
            if hasattr(tabs, 'set_value'):
                tabs.set_value("Login")

            # --- Dashboard Page (and other pages) ---


# These pages will now rely on app.storage.user for token and user_id
@ui.page('/dashboard')
async def dashboard_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return  # Check app.storage.user
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"{broker} Trading Dashboard").classes("text-2xl font-semibold p-4")
    metrics_container = ui.grid(columns='repeat(auto-fit, minmax(250px, 1fr))').classes("w-full p-4 gap-4")
    market_watch_container = ui.column().classes("w-full p-4 gap-2")
    chart_container = ui.column().classes("w-full p-4")

    async def fetch_and_update_dashboard():
        current_broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
        funds_data = await fetch_api(f"/funds/{current_broker}")
        portfolio_data = await fetch_api(f"/portfolio/{current_broker}") or []
        positions_data = await fetch_api(f"/positions/{current_broker}") or []
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
            all_instruments_map = await get_cached_instruments(current_broker)

            async def update_ltp(instrument_symbol_input, ltp_label_display):
                if instrument_symbol_input.value and instrument_symbol_input.value in all_instruments_map:
                    instrument_token = all_instruments_map[instrument_symbol_input.value]
                    ltp_response = await fetch_api(f"/ltp/{current_broker}", params={"instruments": instrument_token})
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


# --- Settings Page ---
@ui.page('/settings')
async def settings_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return
    apply_theme_from_storage()
    render_header()
    ui.label("Application Settings").classes("text-2xl font-semibold p-4")
    with ui.card().classes("m-4 p-6 shadow-lg rounded-xl"):
        ui.label("Preferences").classes("text-xl font-semibold mb-4 border-b pb-2")
        current_theme = app.storage.user.get(STORAGE_THEME_KEY, "Dark")
        ui.select(["Dark", "Light"], label="Select Theme", value=current_theme,
                  on_change=lambda e: (app.storage.user.update({STORAGE_THEME_KEY: e.value}),
                                       apply_theme_from_storage())
                  ).props("outlined dense emit-value map-options").classes("w-full md:w-1/2")
        current_broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
        ui.select(["Upstox", "Zerodha"], label="Default Trading Broker", value=current_broker,
                  on_change=lambda e: app.storage.user.update({STORAGE_BROKER_KEY: e.value})
                  ).props("outlined dense emit-value map-options").classes("w-full md:w-1/2 mt-4")
    with ui.card().classes("m-4 p-6 shadow-lg rounded-xl"):
        ui.label("Broker Connections").classes("text-xl font-semibold mb-4 border-b pb-2")
        status_area = ui.column().classes("gap-4")

        async def check_and_display_status(broker_name, display_container):
            profile = await fetch_api(f"/profile/{broker_name}")
            display_container.clear()
            with display_container:
                if profile and profile.get("name"):
                    ui.html(
                        f"<p class='text-lg'>{broker_name} Status: <span class='text-green-500 font-semibold'>Connected as {profile['name']}</span></p>")
                else:
                    ui.html(
                        f"<p class='text-lg'>{broker_name} Status: <span class='text-red-500 font-semibold'>Not Connected</span></p>")
                    if broker_name == "Upstox":
                        auth_code_input = ui.input("Upstox Auth Code").props("outlined dense clearable").classes(
                            "w-full mt-2")

                        async def reconnect_upstox_action():
                            if auth_code_input.value:
                                resp = await fetch_api(f"/auth/upstox/?auth_code={auth_code_input.value}",
                                                       method="POST")
                                if resp: ui.notify(f"Upstox reconnected: {resp.get('message', 'Success')}",
                                                   type="positive"); await check_and_display_status("Upstox",
                                                                                                    display_container)
                            else:
                                ui.notify("Auth code required.", type="warning")

                        ui.button("Reconnect Upstox", on_click=reconnect_upstox_action).props(
                            "color=primary dense").classes("mt-2")
                    elif broker_name == "Zerodha":
                        req_token_input = ui.input("Zerodha Request Token").props("outlined dense clearable").classes(
                            "w-full mt-2")

                        async def reconnect_zerodha_action():
                            if req_token_input.value:
                                resp = await fetch_api(f"/auth/zerodha/?request_token={req_token_input.value}",
                                                       method="POST")
                                if resp: ui.notify(f"Zerodha reconnected: {resp.get('message', 'Success')}",
                                                   type="positive"); await check_and_display_status("Zerodha",
                                                                                                    display_container)
                            else:
                                ui.notify("Request token required.", type="warning")

                        ui.button("Reconnect Zerodha", on_click=reconnect_zerodha_action).props(
                            "color=primary dense").classes("mt-2")

        zerodha_status_container = ui.column()
        upstox_status_container = ui.column()

        with status_area:
            with ui.expansion("Zerodha Connection", icon="link").classes("w-full"):
                await check_and_display_status("Zerodha", zerodha_status_container)
            with ui.expansion("Upstox Connection", icon="link").classes("w-full"):
                await check_and_display_status("Upstox", upstox_status_container)

        async def refresh_all_statuses():
            await asyncio.gather(
                check_and_display_status("Zerodha", zerodha_status_container),
                check_and_display_status("Upstox", upstox_status_container)
            )
            ui.notify("Connection statuses refreshed.", type="info")

        ui.button("Refresh All Connection Statuses", on_click=refresh_all_statuses).props(
            "color=secondary outline").classes("mt-6")


# --- Watchlist Page ---
@ui.page('/watchlist')
async def watchlist_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return  # Check app.storage.user
    apply_theme_from_storage()
    render_header()
    ui.label("My Watchlist").classes("text-2xl font-semibold p-4")
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    all_instruments_map = await get_cached_instruments(broker)
    watchlist_symbols = app.storage.user.get(STORAGE_WATCHLIST_KEY, [])
    watchlist_display_area = ui.column().classes("w-full p-4 gap-2")

    async def refresh_watchlist_ltps():
        watchlist_display_area.clear()
        if not watchlist_symbols:
            with watchlist_display_area:
                ui.label("Your watchlist is empty. Add instruments below.").classes("text-gray-500 p-4")
            return
        with watchlist_display_area:
            with ui.row().classes("w-full font-bold border-b pb-2 mb-2"):
                ui.label("Symbol").classes("w-1/3")
                ui.label("LTP").classes("w-1/3 text-right")
                ui.label("Actions").classes("w-1/3 text-right")
            for symbol_name in list(watchlist_symbols):
                instrument_token = all_instruments_map.get(symbol_name)
                with ui.row().classes("w-full items-center py-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"):
                    ui.label(symbol_name).classes("w-1/3 truncate")
                    ltp_label = ui.label("Fetching...").classes("w-1/3 text-right")
                    action_container = ui.row().classes("w-1/3 justify-end")
                    with action_container:
                        def remove_action_factory(sym_to_remove):
                            async def remove_from_watchlist():
                                if sym_to_remove in watchlist_symbols:
                                    watchlist_symbols.remove(sym_to_remove)
                                    app.storage.user[STORAGE_WATCHLIST_KEY] = watchlist_symbols
                                    ui.notify(f"{sym_to_remove} removed from watchlist.", type="info")
                                    await refresh_watchlist_ltps()

                            return remove_from_watchlist

                        ui.button(icon="delete", on_click=remove_action_factory(symbol_name), color="negative").props(
                            "flat dense round text-xs").tooltip(f"Remove {symbol_name}")

                    if instrument_token:
                        ltp_response = await fetch_api(f"/ltp/{broker}", params={"instruments": instrument_token})
                        if ltp_response and isinstance(ltp_response, list) and ltp_response:
                            ltp_item = ltp_response[0]
                            ltp_label.text = f"₹{ltp_item.get('last_price', 'N/A'):,.2f}"
                        else:
                            ltp_label.text = "N/A"
                    else:
                        ltp_label.text = "Token N/A";
                        ltp_label.classes("text-red-500")

    with ui.card().classes("m-4 p-6 shadow-lg rounded-xl"):
        ui.label("Add to Watchlist").classes("text-xl font-semibold mb-4 border-b pb-2")
        instrument_select = ui.select(options=list(all_instruments_map.keys()),
                                      label="Search and Select Instrument",
                                      with_input=True, clearable=True) \
            .props("outlined dense behavior=menu").classes("w-full md:w-2/3")

        async def add_to_watchlist_action():
            selected_symbol = instrument_select.value
            if selected_symbol and selected_symbol not in watchlist_symbols:
                watchlist_symbols.append(selected_symbol)
                app.storage.user[STORAGE_WATCHLIST_KEY] = watchlist_symbols
                ui.notify(f"{selected_symbol} added to watchlist.", type="positive")
                instrument_select.set_value(None)
                await refresh_watchlist_ltps()
            elif selected_symbol in watchlist_symbols:
                ui.notify(f"{selected_symbol} is already in your watchlist.", type="warning")
            elif not selected_symbol:
                ui.notify("Please select an instrument.", type="warning")

        ui.button("Add Instrument", on_click=add_to_watchlist_action).props("color=primary").classes("mt-3")

    await refresh_watchlist_ltps()
    ui.timer(15, refresh_watchlist_ltps)


# --- Order Management Page ---
@ui.page('/order-management')
async def order_management_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"Order Management - {broker}").classes("text-2xl font-semibold p-4")
    all_instruments_map = await get_cached_instruments(broker)

    with ui.tabs().props("dense").classes("w-full px-4").on("update:model-value", lambda e: order_tab_panels.set_value(
            e.value)) as order_tabs:
        ui.tab(name="Regular", label="Regular")
        ui.tab(name="GTT", label="GTT (Zerodha)")

    with ui.tab_panels(order_tabs, value="Regular").on("update:model-value",
                                                       lambda e: order_tabs.set_value(e.value)).classes(
            "w-full p-4") as order_tab_panels:
        with ui.tab_panel(name="Regular"):
            ui.label("Place Regular Order").classes("text-xl font-medium mb-3")
            with ui.card().classes("p-6 shadow-md rounded-lg w-full lg:w-2/3 xl:w-1/2"):
                with ui.grid(columns=1, md_columns=2).classes("gap-4"):
                    symbol_select = ui.select(options=list(all_instruments_map.keys()), label="Symbol", with_input=True,
                                              clearable=True).props("outlined dense behavior=menu")
                    transaction_type_radio = ui.radio(["BUY", "SELL"], value="BUY", label="Transaction").props(
                        "inline dense")
                    quantity_input = ui.number(label="Quantity", value=1, min=1, step=1).props("outlined dense")
                    product_options = ["CNC", "MIS", "NRML"] if broker == "Zerodha" else ["DELIVERY", "INTRADAY",
                                                                                          "NORMAL"]
                    product_type_select = ui.select(product_options, label="Product Type",
                                                    value=product_options[0]).props("outlined dense")
                    order_type_select = ui.select(["MARKET", "LIMIT", "SL", "SL-M"], label="Order Type",
                                                  value="MARKET").props("outlined dense")
                    price_input = ui.number(label="Price (for LIMIT/SL)", value=0.00, format="%.2f", step=0.05).props(
                        "outlined dense")
                    trigger_price_input = ui.number(label="Trigger Price (for SL/SL-M)", value=0.00, format="%.2f",
                                                    step=0.05).props("outlined dense")
                    price_input.bind_visibility_from(order_type_select, 'value', lambda val: val in ["LIMIT", "SL"])
                    trigger_price_input.bind_visibility_from(order_type_select, 'value',
                                                             lambda val: val in ["SL", "SL-M"])
                    is_amo_checkbox = ui.checkbox("Place as AMO (After Market Order)")

                async def place_regular_order():
                    if not symbol_select.value or not all_instruments_map.get(symbol_select.value):
                        ui.notify("Please select a valid symbol.", type="warning");
                        return
                    if quantity_input.value <= 0:
                        ui.notify("Quantity must be positive.", type="warning");
                        return
                    order_payload = {
                        "broker": broker, "trading_symbol": symbol_select.value,
                        "instrument_token": all_instruments_map[symbol_select.value],
                        "quantity": int(quantity_input.value), "product_type": product_type_select.value,
                        "order_type": order_type_select.value, "transaction_type": transaction_type_radio.value,
                        "price": float(price_input.value) if order_type_select.value in ["LIMIT", "SL"] else 0.0,
                        "trigger_price": float(trigger_price_input.value) if order_type_select.value in ["SL",
                                                                                                         "SL-M"] else 0.0,
                        "is_amo": is_amo_checkbox.value, "validity": "DAY",
                    }
                    response = await fetch_api("/orders/", method="POST", data=order_payload)
                    if response and (
                            response.get("order_id") or response.get("status") == "success" or response.get("message")):
                        msg = response.get("message", f"Order placed/requested. ID: {response.get('order_id', 'N/A')}")
                        ui.notify(msg, type="positive", multi_line=True)

                ui.button("Place Order", on_click=place_regular_order).props("color=primary").classes("mt-6 py-2 px-4")

        with ui.tab_panel(name="GTT"):
            if broker != "Zerodha":
                ui.label("GTT orders are currently supported for Zerodha only via this UI.").classes(
                    "text-orange-600 p-4")
            else:
                ui.label("Create GTT Order (Zerodha)").classes("text-xl font-medium mb-3")
                with ui.card().classes("p-6 shadow-md rounded-lg w-full lg:w-2/3 xl:w-1/2"):
                    gtt_symbol_select = ui.select(options=list(all_instruments_map.keys()), label="Symbol",
                                                  with_input=True, clearable=True).props("outlined dense behavior=menu")
                    gtt_transaction_type = ui.radio(["BUY", "SELL"], value="BUY", label="Transaction").props(
                        "inline dense")
                    gtt_quantity = ui.number(label="Quantity", value=1, min=1).props("outlined dense")
                    ui.label("Trigger Conditions (Single Leg - Price based)").classes("text-sm mt-2")
                    gtt_trigger_price = ui.number(label="Trigger Price", format="%.2f").props("outlined dense")
                    ui.label("Order Details (Once Triggered)").classes("text-sm mt-2")
                    gtt_limit_price = ui.number(label="Limit Price for Order", format="%.2f").props("outlined dense")

                    async def place_gtt_order():
                        if not gtt_symbol_select.value or not all_instruments_map.get(gtt_symbol_select.value):
                            ui.notify("Please select a valid symbol for GTT.", type="warning");
                            return
                        gtt_payload = {
                            "broker": "Zerodha", "trading_symbol": gtt_symbol_select.value,
                            "instrument_token": all_instruments_map[gtt_symbol_select.value],
                            "transaction_type": gtt_transaction_type.value, "quantity": int(gtt_quantity.value),
                            "trigger_type": "single",
                            "trigger_values": [float(gtt_trigger_price.value)],
                            "last_price": float(gtt_trigger_price.value),
                            "orders": [{"order_type": "LIMIT", "product_type": "CNC",
                                        "price": float(gtt_limit_price.value), "quantity": int(gtt_quantity.value)}]
                        }
                        response = await fetch_api("/gtt-orders/", method="POST", data=gtt_payload)
                        if response and response.get("gtt_id"):
                            ui.notify(f"GTT order created. ID: {response['gtt_id']}", type="positive")
                            await load_gtt_orders()

                    ui.button("Create GTT Order", on_click=place_gtt_order).props("color=primary").classes(
                        "mt-6 py-2 px-4")

                gtt_orders_container = ui.column().classes("w-full mt-6")

                async def load_gtt_orders():
                    gtt_orders_container.clear()
                    gtt_list = await fetch_api(f"/gtt-orders/{broker}")
                    if gtt_list and isinstance(gtt_list, list):
                        if not gtt_list:
                            with gtt_orders_container: ui.label("No active GTT orders found.").classes(
                                "text-gray-500"); return
                        with gtt_orders_container:
                            ui.label("Active GTT Orders").classes("text-lg font-medium mb-2")
                            for gtt in gtt_list:
                                with ui.card().classes("w-full p-3 mb-2 shadow"):
                                    condition = gtt.get('condition', {})
                                    ui.label(
                                        f"ID: {gtt.get('id', gtt.get('gtt_id', 'N/A'))} - {condition.get('tradingsymbol', 'Unknown Symbol')}")
                                    ui.label(f"Status: {gtt.get('status', 'N/A')} | Type: {gtt.get('type', 'N/A')}")
                                    gtt_id_for_cancel = gtt.get('id', gtt.get('gtt_id'))
                                    if gtt_id_for_cancel and gtt.get('status', '').lower() in ['active', 'triggered']:
                                        async def cancel_gtt_action_factory(gtt_id_to_cancel):
                                            async def do_gtt_cancel():
                                                del_resp = await fetch_api(f"/gtt-orders/{broker}/{gtt_id_to_cancel}",
                                                                           method="DELETE")
                                                if del_resp:
                                                    ui.notify(f"GTT {gtt_id_to_cancel} cancellation requested.",
                                                              type="info")
                                                    await load_gtt_orders()

                                            return do_gtt_cancel

                                        ui.button("Cancel GTT",
                                                  on_click=cancel_gtt_action_factory(gtt_id_for_cancel)).props(
                                            "color=negative dense text-xs flat")
                    else:
                        with gtt_orders_container:
                            ui.label("Could not fetch GTT orders.").classes("text-red-500")

                await load_gtt_orders()
    if hasattr(order_tabs, 'set_value'):
        order_tabs.set_value("Regular")


# --- Analytics Page ---
@ui.page('/analytics')
async def analytics_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return  # Check app.storage.user
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"Trade Analytics - {broker}").classes("text-2xl font-semibold p-4")
    all_instruments_map = await get_cached_instruments(broker)
    chart_container = ui.column().classes("w-full p-4 min-h-[500px]")
    controls_container = ui.card().classes("m-4 p-6 shadow-lg rounded-xl")
    with controls_container:
        ui.label("Chart Controls").classes("text-xl font-semibold mb-3")
        with ui.row().classes("gap-4 items-end w-full"):
            analytics_symbol_select = ui.select(options=list(all_instruments_map.keys()), label="Select Symbol",
                                                with_input=True, clearable=True).props(
                "outlined dense behavior=menu").classes("flex-grow")
            timeframe_options = {
                "1 Minute": "minute", "3 Minute": "3minute", "5 Minute": "5minute",
                "15 Minute": "15minute", "30 Minute": "30minute", "1 Hour": "60minute",
                "1 Day": "day", "1 Week": "week", "1 Month": "month"
            }
            timeframe_select = ui.select(options=timeframe_options, label="Timeframe", value="days").props(
                "outlined dense emit-value map-options")
            default_end_date = date.today()
            default_start_date_daily = default_end_date - timedelta(days=90)
            start_date_input = ui.date(value=default_start_date_daily.isoformat(), label="Start Date").props(
                "outlined dense")
            end_date_input = ui.date(value=default_end_date.isoformat(), label="End Date").props("outlined dense")

            with ui.column():
                show_ema_checkbox = ui.checkbox("Show EMA")
                ema_period_input = ui.number(label="EMA Period", value=20, min=2, max=200).props("outlined dense")
                ema_period_input.bind_visibility_from(show_ema_checkbox, 'value')

            load_chart_button = ui.button("Load Chart", icon="show_chart").props("color=primary").classes("self-center")

    async def load_chart_data():
        chart_container.clear()
        if not analytics_symbol_select.value or not all_instruments_map.get(analytics_symbol_select.value):
            with chart_container: ui.label("Please select a valid symbol.").classes("text-orange-600 p-4"); return

        instrument_token = all_instruments_map[analytics_symbol_select.value]
        selected_timeframe = timeframe_select.value
        from_date_str = start_date_input.value
        to_date_str = end_date_input.value

        with chart_container:
            spinner = ui.spinner(size='lg', color='primary').classes('absolute-center')

        historical_data_response = await fetch_api(f"/historical-data/{broker}/{instrument_token}",
                                                   params={"timeframe": selected_timeframe,
                                                           "from_date": from_date_str,
                                                           "to_date": to_date_str})
        chart_container.clear()

        if historical_data_response and isinstance(historical_data_response, list) and historical_data_response:
            df = pd.DataFrame(historical_data_response)
            date_col = 'date' if 'date' in df.columns else 'timestamp'
            if date_col not in df.columns or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                with chart_container: ui.label(
                    "Required OHLC or Date/Timestamp column missing in data from API.").classes(
                    "text-red-500 p-4"); return
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                logger.error(f"Error converting date column to datetime: {e}")
                with chart_container:
                    ui.label(f"Error in date format from API: {e}").classes("text-red-500 p-4"); return

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df[date_col],
                                         open=df['open'], high=df['high'],
                                         low=df['low'], close=df['close'],
                                         name=analytics_symbol_select.value))

            if show_ema_checkbox.value and 'close' in df.columns:
                try:
                    ema_values = df['close'].ewm(span=int(ema_period_input.value), adjust=False).mean()
                    fig.add_trace(go.Scatter(x=df[date_col], y=ema_values, mode='lines',
                                             name=f'EMA({ema_period_input.value})',
                                             line=dict(color='orange', width=1)))
                except Exception as e:
                    logger.error(f"Error calculating EMA: {e}")
                    ui.notify("Could not calculate or plot EMA.", type="warning")

            fig.update_layout(
                title=f"{analytics_symbol_select.value} - {timeframe_select.label} Chart",
                xaxis_title="Date", yaxis_title="Price",
                xaxis_rangeslider_visible=False, height=600
            )
            with chart_container:
                ui.plotly(fig).classes("w-full rounded-lg shadow-md")
        else:
            with chart_container:
                msg = "No historical data found for the selected criteria."
                # Check token from app.storage.user now
                if historical_data_response is None and hasattr(app.storage, 'user') and app.storage.user.get(
                        STORAGE_TOKEN_KEY):
                    msg = "Failed to fetch historical data. Check API logs or backend."
                ui.label(msg).classes("text-orange-600 p-4")

    load_chart_button.on_click(load_chart_data)


# --- Fully Implemented Pages (Order Book, Positions, Portfolio, etc. from previous response) ---
@ui.page('/order-book')
async def order_book_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return  # Check app.storage.user
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"Order Book - {broker}").classes("text-2xl font-semibold p-4")

    orders_table_container = ui.column().classes("w-full p-4")

    async def refresh_order_book():
        orders_table_container.clear()
        orders_data = await fetch_api(f"/order-book/{broker}")

        if orders_data and isinstance(orders_data, list):
            if not orders_data:
                with orders_table_container: ui.label("Order book is empty.").classes("text-gray-500"); return

            columns = [
                {'name': 'order_id', 'label': 'Order ID', 'field': 'order_id', 'sortable': True, 'align': 'left'},
                {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True},
                {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type'},
                {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'align': 'right'},
                {'name': 'price', 'label': 'Price', 'field': 'price',
                 'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A', 'align': 'right'},
                {'name': 'trigger_price', 'label': 'Trig. Price', 'field': 'trigger_price',
                 'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A', 'align': 'right'},
                {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True},
                {'name': 'order_timestamp', 'label': 'Timestamp', 'field': 'order_timestamp', 'sortable': True,
                 'format': lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")).strftime(
                     '%Y-%m-%d %H:%M:%S') if isinstance(v, str) else 'N/A'},
                {'name': 'product_type', 'label': 'Product', 'field': 'product_type'},
                {'name': 'actions', 'label': 'Actions', 'field': 'order_id'}
            ]

            with orders_table_container:
                table = ui.table(columns=columns, rows=orders_data, row_key='order_id').classes(
                    'w-full bordered dense-table shadow-md rounded-lg')
                table.add_slot('body-cell-actions', '''
                    <q-td :props="props">
                        <q-btn v-if="props.row.status && props.row.status.toUpperCase() !== 'COMPLETE' && props.row.status.toUpperCase() !== 'CANCELLED' && props.row.status.toUpperCase() !== 'REJECTED'"
                               dense flat round color="negative" icon="cancel" 
                               @click="() => $parent.$emit('cancel_order', props.row.order_id)">
                            <q-tooltip>Cancel Order</q-tooltip>
                        </q-btn>
                    </q-td>
                ''')

                async def handle_cancel_order(order_id_to_cancel):
                    response = await fetch_api(f"/orders/{broker}/{order_id_to_cancel}", method="DELETE")
                    if response:
                        ui.notify(
                            f"Order {order_id_to_cancel} cancellation requested: {response.get('message', 'Success')}",
                            type="info")
                        await refresh_order_book()

                table.on('cancel_order', lambda e: asyncio.create_task(handle_cancel_order(e.args)))
        else:
            with orders_table_container:
                ui.label("Could not fetch order book or it's empty.").classes("text-orange-600")

    await refresh_order_book()
    ui.timer(20, refresh_order_book)


@ui.page('/positions')
async def positions_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"Current Positions - {broker}").classes("text-2xl font-semibold p-4")
    positions_container = ui.column().classes("w-full p-4")

    async def refresh_positions():
        positions_container.clear()
        positions_data = await fetch_api(f"/positions/{broker}")
        if positions_data and isinstance(positions_data, list):
            if not positions_data:
                with positions_container: ui.label("No open positions.").classes("text-gray-500"); return
            total_pnl = sum(pos.get('pnl', 0.0) for pos in positions_data)
            with positions_container:
                with ui.card().classes("p-4 mb-4 shadow-md rounded-lg"):
                    ui.label(f"Overall P&L: ₹{total_pnl:,.2f}").classes(
                        f"text-xl font-semibold {'text-green-500' if total_pnl >= 0 else 'text-red-500'}")
                columns = [
                    {'name': 'tradingsymbol', 'label': 'Symbol', 'field': 'tradingsymbol', 'sortable': True,
                     'align': 'left'},
                    {'name': 'product', 'label': 'Product', 'field': 'product'},
                    {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'align': 'right'},
                    {'name': 'average_price', 'label': 'Avg. Price', 'field': 'average_price',
                     'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A', 'align': 'right'},
                    {'name': 'last_price', 'label': 'LTP', 'field': 'last_price',
                     'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A', 'align': 'right'},
                    {'name': 'pnl', 'label': 'P&L', 'field': 'pnl',
                     'format': lambda v: f'{float(v):,.2f}' if v is not None else 'N/A', 'align': 'right',
                     'classes': lambda row: 'text-green-500' if row.get('pnl', 0) >= 0 else 'text-red-500'},
                    {'name': 'actions', 'label': 'Actions', 'field': 'tradingsymbol'}
                ]
                table = ui.table(columns=columns, rows=positions_data, row_key='tradingsymbol').classes(
                    'w-full bordered dense-table shadow-md rounded-lg')
                table.add_slot('body-cell-actions', '''
                    <q-td :props="props">
                        <q-btn dense flat round color="primary" icon="exit_to_app" 
                               @click="() => $parent.$emit('square_off', props.row)">
                            <q-tooltip>Square Off</q-tooltip>
                        </q-btn>
                    </q-td>
                ''')

                async def handle_square_off(position_row):
                    if not position_row or not position_row.get('tradingsymbol') or not position_row.get(
                            'instrument_token'):
                        ui.notify("Invalid position data for square-off.", type="error");
                        return
                    qty_to_square = abs(position_row.get('quantity', 0))
                    if qty_to_square == 0:
                        ui.notify("Position quantity is zero, cannot square off.", type="warning");
                        return
                    square_off_payload = {
                        "broker": broker, "trading_symbol": position_row['tradingsymbol'],
                        "instrument_token": position_row['instrument_token'], "quantity": qty_to_square,
                        "product_type": position_row['product'], "order_type": "MARKET",
                        "transaction_type": "SELL" if position_row.get('quantity', 0) > 0 else "BUY",
                        "validity": "DAY", "price": 0, "trigger_price": 0
                    }
                    response = await fetch_api("/orders/", method="POST", data=square_off_payload)
                    if response:
                        ui.notify(
                            f"Square off order for {position_row['tradingsymbol']} placed: {response.get('message', 'Success')}",
                            type="info")
                        await refresh_positions()

                table.on('square_off', lambda e: asyncio.create_task(handle_square_off(e.args)))
        else:
            with positions_container:
                ui.label("Could not fetch positions or none are open.").classes("text-orange-600")

    await refresh_positions()
    ui.timer(15, refresh_positions)


@ui.page('/portfolio')
async def portfolio_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"Portfolio Overview - {broker}").classes("text-2xl font-semibold p-4")
    portfolio_container = ui.column().classes("w-full p-4")

    async def refresh_portfolio():
        portfolio_container.clear()
        holdings_data = await fetch_api(f"/portfolio/{broker}")
        if holdings_data and isinstance(holdings_data, list):
            if not holdings_data:
                with portfolio_container: ui.label("No holdings in portfolio.").classes("text-gray-500"); return
            total_invested_value = sum(h.get('average_price', 0) * h.get('quantity', 0) for h in holdings_data)
            total_current_value = sum(h.get('last_price', 0) * h.get('quantity', 0) for h in holdings_data)
            total_overall_pnl = total_current_value - total_invested_value
            with portfolio_container:
                with ui.card().classes("p-4 mb-4 shadow-md rounded-lg"):
                    with ui.grid(columns=3).classes("text-center"):
                        with ui.column(): ui.label("Total Investment").classes("text-sm text-gray-500"); ui.label(
                            f"₹{total_invested_value:,.2f}").classes("text-lg font-semibold")
                        with ui.column(): ui.label("Current Value").classes("text-sm text-gray-500"); ui.label(
                            f"₹{total_current_value:,.2f}").classes("text-lg font-semibold")
                        with ui.column(): ui.label("Overall P&L").classes("text-sm text-gray-500"); ui.label(
                            f"₹{total_overall_pnl:,.2f}").classes(
                            f"text-lg font-semibold {'text-green-500' if total_overall_pnl >= 0 else 'text-red-500'}")
                columns = [
                    {'name': 'tradingsymbol', 'label': 'Symbol', 'field': 'tradingsymbol', 'sortable': True,
                     'align': 'left'},
                    {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'align': 'right'},
                    {'name': 'average_price', 'label': 'Avg. Buy Price', 'field': 'average_price',
                     'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A', 'align': 'right'},
                    {'name': 'last_price', 'label': 'LTP', 'field': 'last_price',
                     'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A', 'align': 'right'},
                    {'name': 'invested_value', 'label': 'Invested', 'field': 'invested_value',
                     'format': lambda v: f'{v:,.2f}', 'align': 'right'},
                    {'name': 'current_value', 'label': 'Current Val.', 'field': 'current_value',
                     'format': lambda v: f'{v:,.2f}', 'align': 'right'},
                    {'name': 'pnl', 'label': 'P&L', 'field': 'pnl', 'format': lambda v: f'{v:,.2f}', 'align': 'right',
                     'classes': lambda r: 'text-green-500' if r.get('pnl', 0) >= 0 else 'text-red-500'},
                    {'name': 'pnl_percentage', 'label': 'P&L %', 'field': 'pnl_percentage',
                     'format': lambda v: f'{v:.2f}%', 'align': 'right',
                     'classes': lambda r: 'text-green-500' if r.get('pnl_percentage', 0) >= 0 else 'text-red-500'},
                ]
                rows_prepared = []
                for h in holdings_data:
                    invested = h.get('average_price', 0) * h.get('quantity', 0)
                    current_val = h.get('last_price', 0) * h.get('quantity', 0)
                    pnl_val = current_val - invested
                    pnl_pct_val = (pnl_val / invested * 100) if invested != 0 else 0
                    rows_prepared.append({**h, 'invested_value': invested, 'current_value': current_val, 'pnl': pnl_val,
                                          'pnl_percentage': pnl_pct_val})
                ui.table(columns=columns, rows=rows_prepared, row_key='tradingsymbol').classes(
                    'w-full bordered dense-table shadow-md rounded-lg')
        else:
            with portfolio_container:
                ui.label("Could not fetch portfolio holdings.").classes("text-orange-600")

    await refresh_portfolio()
    ui.timer(60, refresh_portfolio)


@ui.page('/mutual-funds')
async def mutual_funds_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"Mutual Funds - {broker}").classes("text-2xl font-semibold p-4")
    if broker != "Zerodha":
        ui.label("Mutual Fund operations are currently supported for Zerodha only via this UI.").classes(
            "text-orange-600 p-4");
        return
    all_mf_instruments_map = {}

    async def fetch_mf_instruments():
        nonlocal all_mf_instruments_map
        response = await fetch_api("/mutual-funds/instruments")
        if response and isinstance(response, list):
            all_mf_instruments_map = {mf['scheme_name']: mf['scheme_code'] for mf in response if
                                      'scheme_name' in mf and 'scheme_code' in mf}
            return list(all_mf_instruments_map.keys())
        return []

    mf_instrument_names = await fetch_mf_instruments()
    with ui.tabs().props("dense").classes("w-full px-4").on("update:model-value",
                                                            lambda e: mf_tab_panels.set_value(e.value)) as mf_tabs:
        ui.tab(name="PlaceOrder", label="Place Order")
        ui.tab(name="Holdings", label="Holdings")
        ui.tab(name="SIPs", label="SIPs")
    with ui.tab_panels(mf_tabs, value="PlaceOrder").on("update:model-value",
                                                       lambda e: mf_tabs.set_value(e.value)).classes(
            "w-full p-4") as mf_tab_panels:
        with ui.tab_panel(name="PlaceOrder"):
            ui.label("Place Mutual Fund Order").classes("text-xl font-medium mb-3")
            with ui.card().classes("p-6 shadow-md rounded-lg w-full lg:w-2/3 xl:w-1/2"):
                mf_symbol_select = ui.select(options=mf_instrument_names, label="Scheme Name", with_input=True,
                                             clearable=True).props("outlined dense behavior=menu")
                mf_transaction_type = ui.radio(["BUY", "SELL"], value="BUY", label="Transaction").props("inline dense")
                mf_quantity_input = ui.number(label="Quantity (units, for SELL)", value=0, min=0).props(
                    "outlined dense")
                mf_amount_input = ui.number(label="Amount (₹, for BUY)", value=1000, min=100).props("outlined dense")
                mf_tag_input = ui.input(label="Tag (Optional)").props("outlined dense")
                mf_quantity_input.bind_visibility_from(mf_transaction_type, 'value', lambda val: val == "SELL")
                mf_amount_input.bind_visibility_from(mf_transaction_type, 'value', lambda val: val == "BUY")

                async def place_mf_order_action():
                    if not mf_symbol_select.value or not all_mf_instruments_map.get(mf_symbol_select.value):
                        ui.notify("Please select a valid MF scheme.", type="warning");
                        return
                    payload = {"broker": "Zerodha", "scheme_code": all_mf_instruments_map[mf_symbol_select.value],
                               "transaction_type": mf_transaction_type.value, "tag": mf_tag_input.value or None}
                    if mf_transaction_type.value == "BUY":
                        if mf_amount_input.value <= 0: ui.notify("Amount must be positive for BUY.",
                                                                 type="warning"); return
                        payload["amount"] = int(mf_amount_input.value)
                    else:
                        if mf_quantity_input.value <= 0: ui.notify("Quantity must be positive for SELL.",
                                                                   type="warning"); return
                        payload["quantity"] = float(mf_quantity_input.value)
                    response = await fetch_api("/mutual-funds/orders", method="POST", data=payload)
                    if response and response.get("order_id"): ui.notify(f"MF Order placed. ID: {response['order_id']}",
                                                                        type="positive")

                ui.button("Place MF Order", on_click=place_mf_order_action).props("color=primary").classes("mt-6")
        with ui.tab_panel(name="Holdings"):
            ui.label("Mutual Fund Holdings").classes("text-xl font-medium mb-3")
            mf_holdings_container = ui.column().classes("w-full")

            async def load_mf_holdings():
                mf_holdings_container.clear()
                holdings = await fetch_api("/mutual-funds/holdings")
                if holdings and isinstance(holdings, list):
                    if not holdings:
                        with mf_holdings_container: ui.label("No MF holdings found.").classes("text-gray-500"); return
                    columns = [
                        {'name': 'scheme_name', 'label': 'Scheme Name', 'field': 'scheme_name', 'sortable': True,
                         'align': 'left'},
                        {'name': 'quantity', 'label': 'Units', 'field': 'quantity', 'align': 'right',
                         'format': lambda v: f'{float(v):.4f}' if v is not None else 'N/A'},
                        {'name': 'average_price', 'label': 'Avg. NAV', 'field': 'average_price',
                         'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A', 'align': 'right'},
                        {'name': 'last_price', 'label': 'Current NAV', 'field': 'last_price',
                         'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A', 'align': 'right'},
                        {'name': 'invested_value', 'label': 'Invested', 'field': 'invested_value',
                         'format': lambda v: f'{v:,.2f}', 'align': 'right'},
                        {'name': 'current_value', 'label': 'Current Val.', 'field': 'current_value',
                         'format': lambda v: f'{v:,.2f}', 'align': 'right'},
                        {'name': 'pnl', 'label': 'P&L', 'field': 'pnl', 'format': lambda v: f'{v:,.2f}',
                         'align': 'right',
                         'classes': lambda r: 'text-green-500' if r.get('pnl', 0) >= 0 else 'text-red-500'},
                    ]
                    rows_prepared = []
                    for h in holdings:
                        invested = h.get('average_price', 0) * h.get('quantity', 0)
                        current_val = h.get('last_price', 0) * h.get('quantity', 0)
                        pnl_val = h.get('pnl', current_val - invested)
                        rows_prepared.append(
                            {**h, 'invested_value': invested, 'current_value': current_val, 'pnl': pnl_val})
                    with mf_holdings_container:
                        ui.table(columns=columns, rows=rows_prepared, row_key='scheme_name').classes(
                            'w-full bordered dense-table shadow-md rounded-lg')
                else:
                    with mf_holdings_container:
                        ui.label("Could not fetch MF holdings.").classes("text-orange-600")

            await load_mf_holdings()
            ui.button("Refresh Holdings", on_click=load_mf_holdings).props("outline dense").classes("mt-4")
        with ui.tab_panel(name="SIPs"):
            ui.label("Manage SIPs").classes("text-xl font-medium mb-3")
            mf_sips_list_container = ui.column().classes("w-full")
            with ui.card().classes("p-6 shadow-md rounded-lg w-full lg:w-2/3 xl:w-1/2 mb-6"):
                ui.label("Create New SIP").classes("text-lg font-medium mb-2")
                sip_symbol_select = ui.select(options=mf_instrument_names, label="Scheme Name", with_input=True,
                                              clearable=True).props("outlined dense behavior=menu")
                sip_amount_input = ui.number(label="Installment Amount (₹)", value=1000, min=100).props(
                    "outlined dense")
                sip_frequency_select = ui.select(["monthly"], value="monthly", label="Frequency").props(
                    "outlined dense")
                sip_day_input = ui.number(label="Installment Day (1-28)", value=1, min=1, max=28).props(
                    "outlined dense")
                sip_installments_input = ui.number(label="No. of Installments (-1 for perpetual)", value=-1,
                                                   min=-1).props("outlined dense")
                sip_tag_input = ui.input(label="Tag (Optional)").props("outlined dense")

                async def create_sip_action():
                    if not sip_symbol_select.value or not all_mf_instruments_map.get(sip_symbol_select.value):
                        ui.notify("Please select a valid MF scheme for SIP.", type="warning");
                        return
                    sip_payload = {
                        "broker": "Zerodha", "scheme_code": all_mf_instruments_map[sip_symbol_select.value],
                        "instalment_amount": int(sip_amount_input.value),
                        "frequency": sip_frequency_select.value.upper(),
                        "instalment_day": int(sip_day_input.value), "installments": int(sip_installments_input.value),
                        "tag": sip_tag_input.value or None
                    }
                    response = await fetch_api("/mutual-funds/sips", method="POST", data=sip_payload)
                    if response and response.get("sip_id"):
                        ui.notify(f"SIP created successfully. ID: {response['sip_id']}", type="positive")
                        await load_active_sips()

                ui.button("Create SIP", on_click=create_sip_action).props("color=primary").classes("mt-4")

            async def load_active_sips():
                mf_sips_list_container.clear()
                sips = await fetch_api("/mutual-funds/sips")
                if sips and isinstance(sips, list):
                    if not sips:
                        with mf_sips_list_container: ui.label("No active SIPs found.").classes("text-gray-500"); return
                    with mf_sips_list_container:
                        ui.label("Active SIPs").classes("text-lg font-medium mb-2")
                        for sip in sips:
                            with ui.card().classes("w-full p-3 mb-2 shadow"):
                                ui.label(
                                    f"ID: {sip.get('sip_id')} - {sip.get('tradingsymbol', sip.get('scheme_name', 'Unknown Scheme'))}")
                                ui.label(
                                    f"Amount: ₹{sip.get('instalment_amount', sip.get('amount', 0)):,.0f} | Day: {sip.get('instalment_day', 'N/A')} | Status: {sip.get('status', 'N/A')}")

                                async def cancel_sip_action_factory(sip_id_to_cancel):
                                    async def do_cancel():
                                        del_resp = await fetch_api(f"/mutual-funds/sips/{sip_id_to_cancel}",
                                                                   method="DELETE")
                                        if del_resp: ui.notify(f"SIP {sip_id_to_cancel} cancellation requested.",
                                                               type="info"); await load_active_sips()

                                    return do_cancel

                                if sip.get('status', '').lower() == 'active':
                                    ui.button("Cancel SIP",
                                              on_click=cancel_sip_action_factory(sip.get('sip_id'))).props(
                                        "color=negative dense text-xs flat").classes("mt-1")
                else:
                    with mf_sips_list_container:
                        ui.label("Could not fetch SIPs.").classes("text-orange-600")

            await load_active_sips()
            ui.button("Refresh SIPs", on_click=load_active_sips).props("outline dense").classes("mt-4")
    if hasattr(mf_tabs, 'set_value'):
        mf_tabs.set_value("PlaceOrder")


@ui.page('/strategies')
async def strategies_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"Algo Trading Strategies - {broker}").classes("text-2xl font-semibold p-4")
    all_instruments_map = await get_cached_instruments(broker)
    strategy_options = {
        "Simple SMA Crossover": {"params": ["sma_short_period", "sma_long_period"]},
        "RSI Overbought/Oversold": {"params": ["rsi_period", "rsi_oversold", "rsi_overbought"]},
        "Bollinger Bands Breakout": {"params": ["bb_period", "bb_std_dev"]},
    }
    with ui.card().classes("m-4 p-6 shadow-lg rounded-xl w-full lg:w-2/3"):
        ui.label("Configure and Execute Strategy").classes("text-xl font-semibold mb-4")
        selected_strategy_name = ui.select(options=list(strategy_options.keys()), label="Select Strategy",
                                           clearable=True).props("outlined dense behavior=menu").classes(
            "w-full md:w-1/2")
        strategy_params_container = ui.column().classes("w-full mt-3 mb-3 p-3 border rounded-md")
        with ui.grid(columns=1, md_columns=2).classes("gap-4 w-full"):
            strat_symbol_select = ui.select(options=list(all_instruments_map.keys()), label="Symbol", with_input=True,
                                            clearable=True).props("outlined dense behavior=menu")
            strat_quantity_input = ui.number(label="Quantity", value=1, min=1).props("outlined dense")
            strat_product_type_options = ["CNC", "MIS", "NRML"] if broker == "Zerodha" else ["DELIVERY", "INTRADAY",
                                                                                             "NORMAL"]
            strat_product_type_select = ui.select(strat_product_type_options, label="Product Type",
                                                  value=strat_product_type_options[0]).props("outlined dense")
            strat_stop_loss_input = ui.number(label="Stop Loss (%)", value=0.0, min=0.0, step=0.1, format="%.1f").props(
                "outlined dense suffix=%")
            strat_take_profit_input = ui.number(label="Take Profit (%)", value=0.0, min=0.0, step=0.1,
                                                format="%.1f").props("outlined dense suffix=%")
        strategy_param_inputs = {}

        def update_strategy_params_ui():
            strategy_params_container.clear();
            strategy_param_inputs.clear()
            if selected_strategy_name.value and selected_strategy_name.value in strategy_options:
                params_for_strategy = strategy_options[selected_strategy_name.value]["params"]
                with strategy_params_container:
                    ui.label(f"Parameters for {selected_strategy_name.value}:").classes("text-md font-medium mb-1")
                    for param_name in params_for_strategy:
                        default_val = 20 if "period" in param_name else (
                            30 if "oversold" in param_name else (70 if "overbought" in param_name else 2.0))
                        strategy_param_inputs[param_name] = ui.number(label=param_name.replace("_", " ").title(),
                                                                      value=default_val).props("outlined dense")

        selected_strategy_name.on_value_change(update_strategy_params_ui)
        update_strategy_params_ui()
        execution_type_radio = ui.radio(["Manual Signal Check", "Execute Immediately", "Schedule (Automated)"],
                                        value="Manual Signal Check", label="Execution Type").props("inline dense")

        async def handle_strategy_action():
            if not selected_strategy_name.value or not strat_symbol_select.value or not all_instruments_map.get(
                    strat_symbol_select.value):
                ui.notify("Please select a strategy and a valid symbol.", type="warning");
                return
            strategy_specific_params = {name: input_field.value for name, input_field in strategy_param_inputs.items()}
            payload = {
                "broker": broker, "strategy_name": selected_strategy_name.value,
                "trading_symbol": strat_symbol_select.value,
                "instrument_token": all_instruments_map[strat_symbol_select.value],
                "quantity": int(strat_quantity_input.value), "product_type": strat_product_type_select.value,
                "params": strategy_specific_params,
                "stop_loss_percentage": float(strat_stop_loss_input.value) if strat_stop_loss_input.value > 0 else None,
                "take_profit_percentage": float(
                    strat_take_profit_input.value) if strat_take_profit_input.value > 0 else None,
            }
            endpoint = "/algo-trading/execute"
            if execution_type_radio.value == "Manual Signal Check":
                payload["action"] = "signal_check"
            elif execution_type_radio.value == "Schedule (Automated)":
                endpoint = "/algo-trading/schedule"
            response = await fetch_api(endpoint, method="POST", data=payload)
            if response: ui.notify(response.get("message", "Strategy action processed."), type="info", multi_line=True)

        ui.button("Process Strategy", on_click=handle_strategy_action).props("color=primary").classes("mt-6 py-2 px-4")


@ui.page('/backtesting')
async def backtesting_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"Strategy Backtesting").classes("text-2xl font-semibold p-4")
    all_instruments_map = await get_cached_instruments(broker)
    strategy_options_backtest = {
        "Simple SMA Crossover": {"params": ["sma_short_period", "sma_long_period"]},
        "RSI Overbought/Oversold": {"params": ["rsi_period", "rsi_oversold", "rsi_overbought"]},
    }
    backtest_results_container = ui.column().classes("w-full p-4")
    with ui.card().classes("m-4 p-6 shadow-lg rounded-xl w-full lg:w-2/3"):
        ui.label("Backtest Configuration").classes("text-xl font-semibold mb-4")
        with ui.grid(columns=1, md_columns=2).classes("gap-4 w-full"):
            bt_symbol_select = ui.select(options=list(all_instruments_map.keys()), label="Symbol", with_input=True,
                                         clearable=True).props("outlined dense behavior=menu")
            bt_timeframe_options = {"1 Minute": "minute", "5 Minute": "5minute", "15 Minute": "15minute",
                                    "1 Hour": "hour", "1 Day": "day"}
            bt_timeframe_select = ui.select(options=bt_timeframe_options, label="Timeframe", value="day").props(
                "outlined dense emit-value map-options")
            bt_start_date_input = ui.date(value=(date.today() - timedelta(days=365)).isoformat(),
                                          label="Start Date").props("outlined dense")
            bt_end_date_input = ui.date(value=date.today().isoformat(), label="End Date").props("outlined dense")
            bt_initial_capital_input = ui.number(label="Initial Capital (₹)", value=100000, min=1000, step=1000).props(
                "outlined dense prefix=₹")
        bt_selected_strategy_name = ui.select(options=list(strategy_options_backtest.keys()), label="Select Strategy",
                                              clearable=True).props("outlined dense behavior=menu mt-3").classes(
            "w-full md:w-1/2")
        bt_strategy_params_container = ui.column().classes("w-full mt-3 mb-3 p-3 border rounded-md")
        bt_strategy_param_inputs = {}

        def update_bt_strategy_params_ui():
            bt_strategy_params_container.clear();
            bt_strategy_param_inputs.clear()
            if bt_selected_strategy_name.value and bt_selected_strategy_name.value in strategy_options_backtest:
                params_for_strategy = strategy_options_backtest[bt_selected_strategy_name.value]["params"]
                with bt_strategy_params_container:
                    ui.label(f"Parameters for {bt_selected_strategy_name.value}:").classes("text-md font-medium mb-1")
                    for param_name in params_for_strategy:
                        default_val = 10 if "short" in param_name else (
                            20 if "long" in param_name or "period" in param_name else 0)
                        bt_strategy_param_inputs[param_name] = ui.number(label=param_name.replace("_", " ").title(),
                                                                         value=default_val).props("outlined dense")

        bt_selected_strategy_name.on_value_change(update_bt_strategy_params_ui)
        update_bt_strategy_params_ui()

        async def run_backtest_action():
            backtest_results_container.clear()
            if not bt_selected_strategy_name.value or not bt_symbol_select.value or not all_instruments_map.get(
                    bt_symbol_select.value):
                ui.notify("Please select a strategy and a valid symbol for backtesting.", type="warning");
                return
            strategy_specific_params = {name: input_field.value for name, input_field in
                                        bt_strategy_param_inputs.items()}
            payload = {
                "instrument_token": all_instruments_map.get(bt_symbol_select.value),
                "trading_symbol": bt_symbol_select.value,
                "timeframe": bt_timeframe_select.value, "start_date": bt_start_date_input.value,
                "end_date": bt_end_date_input.value,
                "initial_capital": float(bt_initial_capital_input.value),
                "strategy_name": bt_selected_strategy_name.value,
                "params": strategy_specific_params,
            }
            with backtest_results_container:
                ui.label("Running backtest...").classes("text-lg font-semibold text-blue-500")
                with ui.spinner(size='xl', color='primary').classes('absolute-center'): pass
            response = await fetch_api("/algo-trading/backtest", method="POST", data=payload)
            backtest_results_container.clear()
            if response and isinstance(response, dict):
                with backtest_results_container:
                    ui.label("Backtest Results").classes("text-xl font-bold mb-3")
                    summary = response.get("summary_metrics", {})
                    if summary:
                        with ui.card().classes("p-4 mb-4 shadow-md rounded-lg"):
                            ui.label("Summary Metrics").classes("text-lg font-semibold mb-2")
                            with ui.grid(columns=2, md_columns=3).classes("gap-x-6 gap-y-2"):
                                for key, value in summary.items():
                                    val_str = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
                                    ui.label(f"{key.replace('_', ' ').title()}:").classes("font-medium");
                                    ui.label(val_str)
                    equity_curve_data = response.get("equity_curve", [])
                    if equity_curve_data:
                        df_equity = pd.DataFrame(equity_curve_data)
                        if 'date' in df_equity.columns and 'equity' in df_equity.columns:
                            df_equity['date'] = pd.to_datetime(df_equity['date'])
                            fig_equity = go.Figure(go.Scatter(x=df_equity['date'], y=df_equity['equity'], mode='lines',
                                                              name='Equity Curve'))
                            fig_equity.update_layout(title="Equity Curve", xaxis_title="Date",
                                                     yaxis_title="Portfolio Value (₹)", height=400)
                            ui.plotly(fig_equity).classes("w-full rounded-lg shadow-md mb-4")
                    trade_log = response.get("trade_log", [])
                    if trade_log:
                        ui.label("Trade Log").classes("text-lg font-semibold mt-4 mb-2")
                        trade_columns = [
                            {'name': 'entry_datetime', 'label': 'Entry Time', 'field': 'entry_datetime',
                             'sortable': True},
                            {'name': 'exit_datetime', 'label': 'Exit Time', 'field': 'exit_datetime', 'sortable': True},
                            {'name': 'trade_type', 'label': 'Type', 'field': 'trade_type'},
                            {'name': 'entry_price', 'label': 'Entry Price', 'field': 'entry_price',
                             'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A'},
                            {'name': 'exit_price', 'label': 'Exit Price', 'field': 'exit_price',
                             'format': lambda v: f'{float(v):.2f}' if v is not None else 'N/A'},
                            {'name': 'quantity', 'label': 'Qty', 'field': 'quantity'},
                            {'name': 'pnl', 'label': 'P&L', 'field': 'pnl',
                             'format': lambda v: f'{float(v):,.2f}' if v is not None else 'N/A',
                             'classes': lambda r: 'text-green-500' if r.get('pnl', 0) >= 0 else 'text-red-500'},
                        ]
                        ui.table(columns=trade_columns, rows=trade_log, row_key='entry_datetime').classes(
                            "w-full bordered dense-table shadow-md rounded-lg")

                        def download_trades_csv():
                            if trade_log:
                                df_trades = pd.DataFrame(trade_log);
                                csv_string = df_trades.to_csv(index=False)
                                ui.download(src=csv_string.encode(),
                                            filename=f"backtest_trades_{bt_symbol_select.value}_{date.today()}.csv",
                                            media_type="text/csv")

                        ui.button("Download Trade Log (CSV)", on_click=download_trades_csv).props(
                            "icon=download outline").classes("mt-3")
            else:
                with backtest_results_container:
                    ui.label("Failed to run backtest or no results returned.").classes("text-red-500 p-4")

        ui.button("Run Backtest", on_click=run_backtest_action).props("color=primary").classes("mt-6 py-2 px-4")


@ui.page('/live-trading')
async def live_trading_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY): ui.navigate.to('/'); return  # Check app.storage.user
    apply_theme_from_storage()
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    ui.label(f"Live Trading Activity Monitor - {broker}").classes("text-2xl font-semibold p-4")
    live_activity_container = ui.column().classes("w-full p-4 gap-4")

    async def refresh_live_activity():
        live_activity_container.clear()
        with live_activity_container:
            ui.label("Recent Trades").classes("text-xl font-semibold mb-2")
            recent_trades_area = ui.column().classes("w-full")
            trades = await fetch_api(f"/trade-history/{broker}", params={"limit": 10})
            if trades and isinstance(trades, list):
                if not trades:
                    with recent_trades_area:
                        ui.label("No recent trades.").classes("text-gray-500");
                else:
                    with recent_trades_area:
                        for trade in trades:
                            trade_time_str = trade.get('timestamp', datetime.now().isoformat())
                            try:
                                trade_time = datetime.fromisoformat(trade_time_str.replace("Z", "+00:00")).strftime(
                                    '%H:%M:%S')  # Handle Z for UTC
                            except ValueError:
                                trade_time = trade_time_str
                            pnl_val = trade.get('pnl', 0);
                            pnl_color = 'text-green-500' if pnl_val >= 0 else 'text-red-500'
                            symbol_display = trade.get('tradingsymbol', trade.get('symbol', 'Unknown'))
                            with ui.card().classes("p-3 w-full shadow-sm"):
                                ui.label(
                                    f"[{trade_time}] {trade.get('type', '')} {symbol_display} @ {trade.get('price', 'N/A'):.2f} | Qty: {trade.get('qty', 'N/A')} | P&L: <span class='{pnl_color} font-semibold'>{pnl_val:,.2f}</span>").classes(
                                    "text-sm").props("html")
            else:
                with recent_trades_area:
                    ui.label("Could not fetch recent trades.").classes("text-orange-600")
            ui.label("Running Automated Strategies").classes("text-xl font-semibold mt-6 mb-2")
            running_strats_area = ui.column().classes("w-full")
            with running_strats_area:
                ui.label("Automated strategy status monitoring coming soon.").classes("text-gray-500")

    await refresh_live_activity()
    ui.timer(10, refresh_live_activity)


# --- App Initialization ---
@app.on_connect
async def on_client_connect(client: Client):
    try:
        # This function is called when a new client connects.
        # We apply the theme based on app.storage.user, which should be
        # initialized by SessionMiddleware if storage_secret is set.
        apply_theme_from_storage()
    except Exception as e:
        logger.error(f"Error applying theme on client connect: {e}")


if __name__ in {"__main__", "__mp_main__"}:
    storage_secret_key = "my_super_secret_key_for_testing_123_please_change_for_prod"
    # storage_secret_key = secrets.token_hex(32) # Use this for better security once basic session works

    ui.run(title="Algo Trader",
           port=8080,
           reload=True,
           uvicorn_reload_dirs='.',
           uvicorn_reload_includes='*.py',
           storage_secret=storage_secret_key)

