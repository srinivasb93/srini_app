import logging
from nicegui import ui, Client, app
import aiohttp
import asyncio
import pandas as pd
import plotly.graph_objects as go
import json
import time
import secrets
from collections import deque

# Import module functions
from unified_theme_manager import apply_unified_theme, reset_theme_styles, PageTheme, ThemeMode
from ui_context_manager import safe_notify, create_safe_task, with_safe_ui_context
from cache_manager import frontend_cache, cached_fetch_api, TradingDataCache, FrontendCacheConfig
from cache_invalidation import (
    cache_invalidation_manager,
    invalidate_on_order_placed,
    invalidate_on_strategy_action,
    invalidate_on_position_change,
    invalidate_on_settings_change,
    emergency_cache_clear
)
from cache_admin_ui import render_cache_admin_panel
from order_management import render_order_management
from strategies import render_strategies_page
from analytics import render_analytics_page
from tradingview_charts import render_tradingview_page
from orderbook import render_order_book_page
from ws_events import emit_order_event
from portfolio import render_portfolio_page
from positions import render_positions_page
from livetrading import render_live_trading_page
from sip_strategy import render_sip_strategy_page
from watchlist import render_watchlist_page
from settings import render_settings_page
from dashboard import render_dashboard_page
from backtesting import render_backtesting_page
from enhanced_professional_scanner import create_enhanced_professional_scanner_page
from profile import render_profile_page
from market_utils import MarketStatusManager

# Configure logging with file handler
import os
from pathlib import Path

# Create logs directory relative to project root
project_root = Path(__file__).parent.parent
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

# Serve static assets (e.g., JS libraries) from frontend/static
static_dir = Path(__file__).parent / 'static'
static_dir.mkdir(exist_ok=True)
try:
    app.add_static_files('/static', str(static_dir))
except Exception:
    # In some contexts this may be called multiple times; ignore
    pass


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "frontend.log", mode='a'),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("Frontend logging configured successfully")

BASE_URL = "http://localhost:8000"

# Storage constants
STORAGE_TOKEN_KEY = 'auth_token'
STORAGE_REFRESH_TOKEN_KEY = 'refresh_token'
STORAGE_TOKEN_EXPIRY_KEY = 'token_expiry'
STORAGE_USER_ID_KEY = 'user_id'
STORAGE_BROKER_KEY = 'default_broker'
STORAGE_THEME_KEY = 'app_theme'
STORAGE_WATCHLIST_KEY = 'user_watchlist'
STORAGE_INSTRUMENTS_CACHE_KEY_PREFIX = 'instruments_cache_'

# WebSocket message queue
websocket_messages = deque()

# Global broker change tracking
current_broker = None
broker_change_handlers = []

# Global aiohttp session with proper timeout configuration
# This prevents creating new sessions for every request and avoids timeout issues
_global_session = None

async def get_http_session():
    """Get or create a global aiohttp session with proper timeout configuration"""
    global _global_session
    if _global_session is None or _global_session.closed:
        # Configure timeout for all HTTP requests
        timeout = aiohttp.ClientTimeout(
            total=60,  # Total timeout for entire request
            connect=10,  # Connection timeout
            sock_read=30,  # Socket read timeout
            sock_connect=10  # Socket connection timeout
        )
        # Create TCPConnector with keepalive
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool limit
            ttl_dns_cache=300,  # DNS cache TTL
            keepalive_timeout=30,  # Keep connections alive for 30 seconds
            force_close=False,  # Reuse connections
            enable_cleanup_closed=True
        )
        _global_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
    return _global_session

async def cleanup_http_session():
    """Cleanup global HTTP session"""
    global _global_session
    if _global_session and not _global_session.closed:
        try:
            # Close session gracefully
            await _global_session.close()
            # Wait for connector to finish closing
            await asyncio.sleep(0.25)
        except Exception as e:
            logger.warning(f"Error closing session gracefully: {e}")
            try:
                # Force close if graceful close fails
                if hasattr(_global_session, '_connector') and _global_session._connector:
                    _global_session._connector._close()
            except Exception as e2:
                logger.warning(f"Error force closing connector: {e2}")
        finally:
            _global_session = None

def register_broker_change_handler(handler):
    """Register a handler to be called when broker changes"""
    broker_change_handlers.append(handler)

def trigger_broker_change(new_broker):
    """Trigger all registered broker change handlers and update global state"""
    global current_broker
    current_broker = new_broker
    
    # Clear relevant caches when broker changes
    try:
        from cache_invalidation import invalidate_on_broker_change
        invalidate_on_broker_change(new_broker)
    except Exception as e:
        logger.error(f"Error invalidating caches on broker change: {e}")
    
    # Trigger registered handlers
    for handler in broker_change_handlers:
        try:
            handler(new_broker)
        except Exception as e:
            logger.error(f"Error in broker change handler: {e}")
    
    # Note: UI notifications are handled by individual pages to avoid context issues

def get_current_broker():
    """Get the current broker from storage"""
    return app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")

def create_broker_monitor(refresh_function, check_interval=2):
    """Create a broker monitor that calls refresh_function when broker changes"""
    last_broker = get_current_broker()
    
    def check_broker_change():
        nonlocal last_broker
        current = get_current_broker()
        if current != last_broker:
            logger.info(f"Broker changed from {last_broker} to {current}")
            last_broker = current
            try:
                # Check if the function is async and handle accordingly
                import asyncio
                import inspect
                if inspect.iscoroutinefunction(refresh_function):
                    asyncio.create_task(refresh_function())
                else:
                    refresh_function()
            except Exception as e:
                logger.error(f"Error in broker change refresh: {e}")
    
    # Start monitoring
    ui.timer(check_interval, check_broker_change)
    return check_broker_change

# Admin users configuration - in production, this should come from database
ADMIN_USERS = {
    "srini@gmail.com"
    # Add more admin emails here
}

# For development only - set to True to allow all users to access admin features
# WARNING: Set to False in production!
DEVELOPMENT_ADMIN_MODE = False

def is_admin_user() -> bool:
    """Check if current user has admin privileges"""
    try:
        user_email = app.storage.user.get('email', '')
        # In development, you can also check for specific user IDs or other criteria
        user_id = app.storage.user.get('user_id', '')
        
        # Check if user email is in admin list
        if user_email.lower() in {email.lower() for email in ADMIN_USERS}:
            return True
        
        # Development mode - allow all users to access admin features
        if DEVELOPMENT_ADMIN_MODE:
            logger.warning("DEVELOPMENT_ADMIN_MODE is enabled - all users have admin access!")
            return True
        
        # Additional admin checks can be added here:
        # - Check database for user role
        # - Check specific user IDs
        # - Check domain-based permissions
        
        return False
    except Exception as e:
        logger.error(f"Error checking admin privileges: {e}")
        return False


async def refresh_access_token():
    """
    Refresh the access token using the refresh token.
    Returns True if successful, False otherwise.
    """
    try:
        refresh_token = app.storage.user.get(STORAGE_REFRESH_TOKEN_KEY)
        if not refresh_token:
            logger.warning("No refresh token available")
            return False
        
        # Call the refresh endpoint without authorization header
        url = f"{BASE_URL}/auth/refresh"
        session = await get_http_session()
        
        async with session.post(url, json={"refresh_token": refresh_token}) as response:
            if response.status == 200:
                data = await response.json()
                # Update tokens in storage
                app.storage.user[STORAGE_TOKEN_KEY] = data["access_token"]
                
                # Calculate and store expiry time (5 minutes before actual expiry for safety margin)
                import time
                expires_in = data.get("expires_in", 480 * 60)  # Default 8 hours
                expiry_time = time.time() + expires_in - 300  # 5 minutes safety margin
                app.storage.user[STORAGE_TOKEN_EXPIRY_KEY] = expiry_time
                
                logger.info("Access token refreshed successfully")
                return True
            else:
                logger.error(f"Token refresh failed with status {response.status}")
                return False
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        return False

async def ensure_valid_token():
    """
    Check if token is expired or about to expire and refresh if needed.
    Returns True if token is valid, False if refresh failed.
    """
    try:
        import time
        token = app.storage.user.get(STORAGE_TOKEN_KEY)
        expiry_time = app.storage.user.get(STORAGE_TOKEN_EXPIRY_KEY)
        
        if not token:
            return False
        
        # If no expiry time is set, assume token is still valid
        # (for backward compatibility with existing sessions)
        if not expiry_time:
            return True
        
        current_time = time.time()
        
        # If token is expired or about to expire (within 1 minute)
        if current_time >= expiry_time:
            logger.info("Token expired or about to expire, refreshing...")
            return await refresh_access_token()
        
        return True
    except Exception as e:
        logger.error(f"Error checking token validity: {e}")
        return True  # Assume valid to avoid breaking existing functionality

async def fetch_api(endpoint, method="GET", data=None, params=None, retries=3, backoff=1):
    # Ensure token is valid before making the API call (skip for login/register endpoints)
    if not any(auth_endpoint in endpoint for auth_endpoint in ['/auth/login', '/auth/register', '/auth/refresh']):
        token_valid = await ensure_valid_token()
        if not token_valid and hasattr(app, 'storage') and hasattr(app.storage, 'user'):
            # Token refresh failed, clear storage and redirect to login
            app.storage.user.clear()
            ui.navigate.to('/')
            safe_notify("Session expired. Please log in again.", "negative")
            return {"error": {"code": "UNAUTHORIZED", "message": "Session expired"}, "status": 401}
    
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
            # Use global session instead of creating new one every time
            session = await get_http_session()
            async with session.request(method, url, **request_kwargs) as response:
                    logger.debug(f"API call: {method} {url}, Status: {response.status}")
                    if response.status == 401:
                        if hasattr(app, 'storage') and hasattr(app.storage, 'user'):
                            app.storage.user.clear()
                        # Use safe UI operations to avoid context errors
                        try:
                            ui.navigate.to('/')
                            safe_notify("Session expired. Please log in again.", "negative")
                        except Exception as e:
                            logger.warning(f"Could not show session expired notification: {e}")
                        return {"error": {"code": "UNAUTHORIZED", "message": "Session expired"}, "status": 401}
                    if response.status == 404 and endpoint.startswith("/profile/"):
                        return {"error": {"code": "NOT_FOUND", "message": "Profile not found"}, "status": 404}
                    if response.status >= 400:
                        try:
                            # First try to parse as JSON
                            error_data = await response.json()
                            # Backend puts error message in 'error' key, fallback to 'detail' for compatibility
                            detail = error_data.get('error') or error_data.get('detail') or 'Unknown API error'
                            logger.error(f"API Error: {detail}")
                            return {"error": {"code": "API_ERROR", "message": detail}, "status": response.status}
                        except Exception as e:
                            # If JSON parsing fails, try to get the raw text
                            try:
                                error_text = await response.text()
                                logger.error(f"API Error (text): {error_text}")
                                return {"error": {"code": "API_ERROR", "message": error_text}, "status": response.status}
                            except Exception as text_error:
                                logger.error(f"Error parsing API error: {e}, text error: {text_error}")
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

# Cached version of fetch_api for improved performance
@cached_fetch_api()
async def cached_fetch_api(endpoint, method="GET", data=None, params=None, retries=3, backoff=1):
    """Cached wrapper for fetch_api - automatically handles caching for GET requests"""
    return await fetch_api(endpoint, method, data, params, retries, backoff)

# Cache-aware API functions that handle invalidation
async def place_order_with_invalidation(endpoint, data, broker, user_id):
    """Place order and invalidate related caches"""
    response = await fetch_api(endpoint, method="POST", data=data)
    if response and not response.get('error'):
        invalidate_on_order_placed(broker, user_id, data)
    return response

async def strategy_action_with_invalidation(endpoint, action, broker, strategy_id=None, method="POST", data=None):
    """Execute strategy action and invalidate related caches"""
    response = await fetch_api(endpoint, method=method, data=data)
    if response and not response.get('error'):
        invalidate_on_strategy_action(action, broker, strategy_id)
    return response

async def position_change_with_invalidation(endpoint, broker, user_id, symbol=None, method="PUT", data=None):
    """Execute position-related action and invalidate related caches"""
    response = await fetch_api(endpoint, method=method, data=data)
    if response and not response.get('error'):
        invalidate_on_position_change(broker, user_id, symbol)
    return response

async def settings_change_with_invalidation(endpoint, user_id, setting_type=None, data=None):
    """Update settings and invalidate related caches"""
    response = await fetch_api(endpoint, method="POST", data=data)
    if response and not response.get('error'):
        invalidate_on_settings_change(user_id, setting_type)
    return response

async def fetch_with_cache(endpoint, method="GET", data=None, params=None, ttl=None):
    """
    Fetch API data with intelligent cache control using FrontendCacheConfig
    """
    if method.upper() != "GET":
        return await fetch_api(endpoint, method, data, params)
    
    # Auto-determine TTL based on endpoint if not provided
    if ttl is None:
        ttl = get_ttl_for_endpoint(endpoint)
    
    cache_key = frontend_cache.generate_cache_key("fetch_api", endpoint, method, str(params))
    cached_result = frontend_cache.get(cache_key)
    
    if cached_result is not None:
        logger.debug(f"Cache hit for {endpoint}")
        return cached_result
    
    result = await fetch_api(endpoint, method, data, params)
    
    # Only cache successful results (no error key or error is None)
    if result and not result.get('error'):
        frontend_cache.set(cache_key, result, ttl)
    
    return result

def get_ttl_for_endpoint(endpoint: str) -> int:
    """Auto-determine TTL based on endpoint pattern"""
    if '/quotes' in endpoint or '/ltp' in endpoint:
        return FrontendCacheConfig.LIVE_QUOTES
    elif '/positions' in endpoint:
        return FrontendCacheConfig.POSITION_DATA
    elif '/orders' in endpoint:
        return FrontendCacheConfig.ORDER_STATUS
    elif '/strategies' in endpoint:
        return FrontendCacheConfig.STRATEGY_LIST
    elif '/instruments' in endpoint:
        return FrontendCacheConfig.INSTRUMENTS
    elif '/portfolio' in endpoint:
        return FrontendCacheConfig.PORTFOLIO_DATA
    elif '/watchlist' in endpoint:
        return FrontendCacheConfig.WATCHLIST
    elif '/preferences' in endpoint or '/settings' in endpoint:
        return FrontendCacheConfig.USER_PREFERENCES
    elif '/analytics' in endpoint or '/metrics' in endpoint:
        return FrontendCacheConfig.ANALYTICS_DATA
    else:
        return FrontendCacheConfig.MARKET_DATA  # Default TTL


async def connect_websocket(max_retries=5, initial_backoff=2):
    user_id = app.storage.user.get(STORAGE_USER_ID_KEY, "default_user_id")
    token = app.storage.user.get(STORAGE_TOKEN_KEY)
    if not token:
        logger.warning("No token found for WebSocket connection.")
        return

    ws_url = f"{BASE_URL.replace('http', 'ws')}/ws/orders/{user_id}"
    retry_count = 0
    websocket = None

    async def cleanup_websocket():
        """Cleanup websocket connection"""
        nonlocal websocket
        if websocket and not websocket.closed:
            await websocket.close()

    # Register cleanup with client if available
    try:
        from nicegui import context
        if hasattr(context, 'client') and context.client:
            context.client.on_disconnect(cleanup_websocket)
    except Exception:
        pass  # Context not available, continue without cleanup registration

    while retry_count < max_retries:
        try:
            # Use global session with proper keepalive settings
            session = await get_http_session()
            # Configure WebSocket with timeout and heartbeat
            async with session.ws_connect(
                ws_url,
                timeout=30.0,  # Connection timeout
                heartbeat=10.0,  # Send ping every 10 seconds to keep connection alive
                autoping=True  # Automatically respond to pings
            ) as ws:
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
                                # Emit to registered callbacks
                                try:
                                    await emit_order_event(data)
                                except Exception:
                                    pass
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
    """Apply theme using the unified theme manager"""
    theme_mode = app.storage.user.get(STORAGE_THEME_KEY, "dark").lower()
    # Theme will be applied when pages load

# Replace the existing toggle_theme() function
def toggle_theme():
    """Toggle theme using unified theme manager"""
    from unified_theme_manager import switch_unified_theme, unified_theme_manager, ThemeMode
    current = unified_theme_manager.current_theme
    new_theme = ThemeMode.LIGHT if current == ThemeMode.DARK else ThemeMode.DARK
    switch_unified_theme(new_theme, app.storage.user)
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


def render_header():
    """Enhanced header with compact navigation and profile dropdown"""
    with ui.header(elevated=True).classes('justify-between items-center'):

        # Left side - Logo and Title
        with ui.row().classes("items-center gap-3"):
            ui.icon("candlestick_chart", size="1.75rem").classes("text-cyan-400")
            ui.label("AlgoTrade Pro").classes("text-xl font-bold theme-text-primary")
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
                    ("Analytics", "/analytics", "analytics"),
                    ("TV Charts", "/tv-charts", "stacked_line_chart"),
                    ("Scanner", "/scanner", "search")
                ]),
                ("Strategies", [
                    ("All Strategies", "/strategies", "psychology"),
                    ("SIP Strategy", "/sip-strategy", "savings"),
                    ("Backtesting", "/backtesting", "assessment")
                ]),
                ("Admin", [
                    ("Cache Admin", "/cache-admin", "memory")
                ]) if is_admin_user() else None
            ]

            # Filter out None items (non-admin users)
            nav_items = [item for item in nav_items if item is not None]
            
            for item in nav_items:
                if isinstance(item[1], str):  # Single item
                    name, route, icon = item
                    ui.button(name, icon=icon, on_click=lambda r=route: ui.navigate.to(r)).props(
                        'flat dense no-caps').classes("nav-tab-btn text-sm")
                else:  # Dropdown group
                    group_name, sub_items = item[0], item[1]
                    with ui.button(group_name, icon="arrow_drop_down").props(
                            'flat dense no-caps').classes("nav-tab-btn text-sm"):
                        with ui.menu().classes("q-menu"):
                            for sub_name, sub_route, sub_icon in sub_items:
                                ui.menu_item(sub_name,
                                             on_click=lambda r=sub_route: ui.navigate.to(r)).props(
                                    f'icon={sub_icon}').classes("q-item")

        # Right side - Status and Profile
        with ui.row().classes("items-center gap-4"):
            # Market Status - Dynamic (Event-driven, resource-efficient)
            # Uses MarketStatusManager from market_utils.py for clean separation of concerns
            with ui.row(align_items="center").classes("status-indicator market-status"):
                market_status_icon = ui.icon("circle", size="0.5rem").classes("text-gray-500")
                market_status_label = ui.label("Checking...").classes("text-sm theme-text-primary")
                
                # Initialize market status manager
                status_manager = MarketStatusManager(fetch_api)
                
                async def update_market_status():
                    """Fetch and update market status from backend"""
                    try:
                        # Fetch status using manager
                        response = await status_manager.fetch_market_status()
                        parsed = status_manager.parse_market_status_response(response)
                        
                        # Update UI elements
                        market_status_icon.classes(replace=parsed["css_class"])
                        market_status_label.text = parsed["status"]
                        market_status_label.tooltip(parsed["tooltip"])
                        
                        # Schedule next update
                        schedule_next_update(parsed["is_trading_day"])
                        
                    except Exception as e:
                        logger.error(f"Error updating market status: {e}")
                        market_status_label.text = "Status Unknown"
                        market_status_icon.classes(replace="text-gray-500")
                        schedule_next_update(False, error_retry=True)
                
                def schedule_next_update(is_trading_day, error_retry=False):
                    """Schedule next update at optimal time"""
                    # Cancel existing timer
                    status_manager.cancel_current_timer()
                    
                    # Calculate next update time
                    next_update_seconds, _ = status_manager.calculate_next_update_time(
                        is_trading_day, error_retry
                    )
                    
                    # Create new one-time timer (ui.timer handles async callbacks)
                    status_manager.current_timer = ui.timer(
                        next_update_seconds,
                        update_market_status,
                        once=True
                    )
                
                # Initial update (ui.run_javascript is sync, so we schedule immediately)
                ui.timer(0.1, update_market_status, once=True)

            # Connection Status
            with ui.row().classes("status-indicator connection-status"):
                ui.icon("wifi", size="1rem").classes("text-cyan-400")
                connection_status_label = ui.label().classes("text-sm theme-text-primary")
                
                def update_connection_status():
                    broker = app.storage.user.get(STORAGE_BROKER_KEY)
                    if broker:
                        connection_status_label.text = f"Connected: {broker}"
                        connection_status_label.classes("text-sm theme-text-primary")
                    else:
                        connection_status_label.text = "Not Connected"
                        connection_status_label.classes("text-sm text-red-400")
                
                # Initial status
                update_connection_status()
                
                # Make it reactive by checking periodically
                ui.timer(5, update_connection_status)

            # Broker Switcher
            broker_switcher_container = ui.row().classes("items-center gap-2")
            
            async def load_broker_config():
                """Load user's broker configuration"""
                try:
                    api_keys_response = await fetch_api("/users/me/api-keys")
                    if api_keys_response and api_keys_response.get("configured_brokers"):
                        return api_keys_response
                except Exception as e:
                    logger.error(f"Error loading broker config: {e}")
                return None

            async def render_broker_switcher():
                """Render broker switcher based on user configuration"""
                broker_config = await load_broker_config()
                broker_switcher_container.clear()
                
                with broker_switcher_container:
                    if broker_config and broker_config.get("has_multiple_brokers"):
                        # Multiple brokers - show dropdown
                        configured_brokers = broker_config.get("configured_brokers", [])
                        current_broker = app.storage.user.get(STORAGE_BROKER_KEY, configured_brokers[0] if configured_brokers else "Zerodha")
                        
                        ui.select(
                            options=configured_brokers,
                            value=current_broker,
                            on_change=lambda e: (
                                app.storage.user.update({STORAGE_BROKER_KEY: e.value}),
                                ui.notify(f"Switched to {e.value} - Portfolio/Positions will auto-refresh, Dashboard may need manual refresh", type="info"),
                                # Trigger connection status update
                                update_connection_status(),
                                # Trigger broker change event
                                trigger_broker_change(e.value)
                            )
                        ).classes("text-sm").props("dense outlined")
                    elif broker_config and broker_config.get("configured_brokers"):
                        # Single broker - show static label
                        broker = broker_config.get("configured_brokers")[0]
                        ui.chip(broker, color="blue").classes("text-xs")
                    else:
                        # No brokers configured
                        ui.chip("No Broker", color="red").classes("text-xs")
            
            # Render broker switcher asynchronously
            import asyncio
            asyncio.create_task(render_broker_switcher())

            # Settings Icon
            ui.button(icon="settings", on_click=lambda: ui.navigate.to('/settings')).props(
                "flat round dense").classes("theme-text-primary hover:bg-white/10").tooltip("Settings & Preferences")

            # Theme toggle (with meaningful icon)
            ui.button(icon="dark_mode", on_click=toggle_theme).props(
                "flat round dense").classes("theme-text-primary hover:bg-white/10").tooltip("Toggle Dark/Light Mode")

            # Profile Dropdown
            with ui.element('div').classes('profile-dropdown'):
                with ui.button(icon="account_circle").props("flat round dense").classes(
                        "theme-text-primary hover:bg-white/10") as profile_btn:

                    # Create dropdown menu
                    menu = ui.menu().classes(
                        "profile-dropdown-menu bg-gray-900/95 backdrop-blur-lg border border-white/10"
                    )

                    with menu:
                        # User info (fetched from backend profile)
                        user_email_label = ui.label("Loading...").classes("text-sm font-semibold text-white")
                        broker = app.storage.user.get(STORAGE_BROKER_KEY, 'Zerodha')
                        broker_label = ui.label(f"Broker: {broker}").classes("text-xs text-gray-400")
                        
                        async def fetch_user_profile():
                            """Fetch user profile from backend"""
                            try:
                                # Try to get profile from the current broker
                                profile = await fetch_api(f"/profile/{broker}")
                                if profile and profile.get("email"):
                                    user_email_label.text = profile["email"]
                                    # Store in local storage for future use
                                    app.storage.user['email'] = profile["email"]
                                    if profile.get("name"):
                                        user_email_label.text = profile["name"]
                                else:
                                    # Fallback to stored email
                                    user_email = app.storage.user.get('email', 'user@example.com')
                                    user_email_label.text = user_email
                            except Exception as e:
                                logger.error(f"Error fetching user profile: {e}")
                                # Fallback to stored email
                                user_email = app.storage.user.get('email', 'user@example.com')
                                user_email_label.text = user_email
                        
                        # Fetch profile data
                        ui.timer(0.1, fetch_user_profile, once=True)

                        with ui.column().classes("p-3 border-b border-white/10"):
                            pass  # Labels already added above

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

    apply_unified_theme(PageTheme.LOGIN, app.storage.user)

    async def handle_login():
        if not email.value or not password.value:
            ui.notify("Email and password are required.", type="warning")
            return
        data = {"username": email.value, "password": password.value, "grant_type": "password"}
        response = await fetch_api("/auth/login", method="POST", data=data)
        if response and "access_token" in response:
            try:
                import time
                
                # Store access token
                app.storage.user[STORAGE_TOKEN_KEY] = response["access_token"]
                
                # Store refresh token if provided
                if "refresh_token" in response:
                    app.storage.user[STORAGE_REFRESH_TOKEN_KEY] = response["refresh_token"]
                
                # Calculate and store token expiry time (5 minutes before actual expiry for safety)
                expires_in = response.get("expires_in", 480 * 60)  # Default 8 hours in seconds
                expiry_time = time.time() + expires_in - 300  # Subtract 5 minutes for safety margin
                app.storage.user[STORAGE_TOKEN_EXPIRY_KEY] = expiry_time
                
                # Store user ID from response or fetch profile
                if "user_id" in response:
                    app.storage.user[STORAGE_USER_ID_KEY] = response["user_id"]
                else:
                    user_profile = await fetch_api("/users/me")
                    if user_profile and "id" in user_profile:
                        app.storage.user[STORAGE_USER_ID_KEY] = user_profile["id"]
                    else:
                        app.storage.user[STORAGE_USER_ID_KEY] = user_profile.get("email",
                                                                                 "default_user_id") if user_profile else "default_user_id"

                ui.notify("Login successful!", type="positive")
                
                # Check which brokers are connected and set the first connected one as default
                connected_broker = None
                for broker in ["Zerodha", "Upstox"]:
                    try:
                        profile_response = await fetch_api(f"/profile/{broker}")
                        if profile_response and profile_response.get("name"):
                            connected_broker = broker
                            app.storage.user[STORAGE_BROKER_KEY] = broker
                            ui.notify(f"{broker} connected: {profile_response.get('name', '')}", type="info")
                            break
                    except Exception as e:
                        logger.error(f"Error checking {broker} connection: {e}")
                        continue
                
                if not connected_broker:
                    ui.notify("No brokers connected. Please check Settings to connect your broker.",
                              type="warning", multi_line=True)

                ui.navigate.to("/dashboard")
                create_safe_task(connect_websocket())
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
            # Set default preferences for new user
            try:
                default_preferences = {
                    'default_order_type': 'MARKET',
                    'default_product_type': 'CNC',
                    'refresh_interval': 5,
                    'order_alerts': True,
                    'pnl_alerts': True,
                    'strategy_alerts': True,
                    'daily_loss_limit': 10000,
                    'position_size_limit': 50000,
                    'auto_stop_trading': True,
                    'max_open_positions': 10,
                    'risk_per_trade': 2.0,
                    'max_portfolio_risk': 20.0,
                    'max_orders_per_minute': 10,
                    'request_timeout': 30,
                    'enable_rate_limiting': True,
                    'auto_retry_requests': True
                }
                await fetch_api("/user/preferences", method="POST", data=default_preferences)
                logger.info("Default preferences set for new user")
            except Exception as e:
                logger.error(f"Error setting default preferences: {e}")
                # Don't fail registration if preferences setting fails
            
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

    # Add cleanup handler for client disconnect
    cleanup_tasks = []

    async def page_cleanup():
        """Cleanup function for page resources"""
        for task in cleanup_tasks:
            try:
                if not task.done():
                    task.cancel()
            except Exception as e:
                logger.error(f"Error canceling task: {e}")

    client.on_disconnect(page_cleanup)

    # Apply enhanced theme and render header
    apply_unified_theme(PageTheme.DASHBOARD, app.storage.user)
    render_header()

    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")

    # Display WebSocket messages
    with ui.column().classes("w-full p-4"):
        try:
            # if not client.is_deleted:
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
    apply_unified_theme(PageTheme.ORDER_MANAGEMENT, app.storage.user)  # Changed from TRADING to ORDER_MANAGEMENT
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_order_management(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/analytics')
async def analytics_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.ANALYTICS, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_analytics_page(fetch_api, app.storage.user, await get_cached_instruments(broker))

@ui.page('/tv-charts')
async def tv_charts_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.ANALYTICS, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_tradingview_page(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/strategies')
async def strategies_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.STRATEGIES, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_strategies_page(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/sip-strategy')
async def sip_strategies_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.SIP_STRATEGY, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_sip_strategy_page(fetch_api, app.storage.user)


@ui.page('/backtesting')
async def backtesting_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.BACKTESTING, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_backtesting_page(fetch_api, app.storage.user, await get_cached_instruments(broker))


@ui.page('/order-book')
async def order_book_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.ORDERBOOK, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_order_book_page(fetch_api, app.storage.user, broker)


@ui.page('/portfolio')
async def portfolio_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.PORTFOLIO, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_portfolio_page(fetch_api, app.storage.user, broker)


@ui.page('/positions')
async def positions_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.POSITIONS, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_positions_page(fetch_api, app.storage.user, broker)


@ui.page('/live-trading')
async def live_trading_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.LIVE_TRADING, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_live_trading_page(fetch_api, app.storage.user, broker)


@ui.page('/watchlist')
async def watchlist_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.WATCHLIST, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_watchlist_page(fetch_api, app.storage.user, get_cached_instruments, broker)


@ui.page('/scanner')
async def scanner_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.ENHANCED_SCANNER, app.storage.user)  # Use enhanced scanner theme
    render_header()
    auth_token = app.storage.user.get(STORAGE_TOKEN_KEY)
    create_enhanced_professional_scanner_page(auth_token)


@ui.page('/settings')
async def settings_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.SETTINGS, app.storage.user)
    render_header()
    broker = app.storage.user.get(STORAGE_BROKER_KEY, "Zerodha")
    await render_settings_page(fetch_api, app.storage.user, apply_theme_from_storage)


@ui.page('/profile')
async def profile_page(client: Client):
    """User profile page with complete user data and preferences"""
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.SETTINGS, app.storage.user)
    render_header()
    
    broker = app.storage.user.get(STORAGE_BROKER_KEY, 'Zerodha')
    
    # Render profile page using dedicated module
    await render_profile_page(fetch_api, app.storage.user, broker)


@ui.page('/cache-admin')
async def cache_admin_page(client: Client):
    await client.connected()
    
    # Check authentication
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    
    # Check admin privileges
    if not is_admin_user():
        apply_unified_theme(PageTheme.DEFAULT, app.storage.user)
        render_header()
        
        with ui.column().classes("w-full p-8 items-center"):
            ui.icon("security", size="4rem").classes("text-red-400 mb-4")
            ui.label("Access Denied").classes("text-h4 text-white mb-2")
            ui.label("Cache Administration is restricted to admin users only.").classes("text-gray-400 mb-4")
            ui.label("This page contains sensitive system information and controls.").classes("text-gray-500 text-sm mb-6")
            
            ui.button("Go to Dashboard", icon="dashboard", 
                     on_click=lambda: ui.navigate.to('/dashboard')).classes("q-mt-md")
        return
    
    apply_unified_theme(PageTheme.SETTINGS, app.storage.user)
    render_header()
    
    with ui.column().classes("w-full p-4"):
        # Admin indicator
        with ui.row().classes("items-center mb-4"):
            ui.icon("admin_panel_settings", size="1.5rem").classes("text-yellow-400")
            ui.label("Cache Administration").classes("text-h4 q-pa-md text-white")
            ui.chip("ADMIN ONLY", color="red").classes("text-xs ml-4")
        
        await render_cache_admin_panel()

@ui.page('/strategy-performance')
async def strategy_performance_page(client: Client):
    await client.connected()
    if not app.storage.user.get(STORAGE_TOKEN_KEY):
        ui.navigate.to('/')
        return
    apply_unified_theme(PageTheme.DEFAULT, app.storage.user)
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
    """Handle client connection - apply theme and setup session"""
    try:
        apply_theme_from_storage()
        logger.info(f"Client connected: {client.id}")
    except Exception as e:
        logger.error(f"Error applying theme on client connect: {e}")

@app.on_disconnect
async def on_client_disconnect(client: Client):
    """Handle client disconnection - cleanup resources"""
    try:
        logger.info(f"Client disconnected: {client.id}")
        # Note: Don't clear storage here as it will cause logouts
        # Only cleanup temporary resources
    except Exception as e:
        logger.error(f"Error handling client disconnect: {e}")

@app.on_shutdown
async def on_app_shutdown():
    """Cleanup resources on application shutdown - never raises exceptions"""
    global _global_session  # Declare global at function level
    
    def force_cleanup():
        """Synchronous force cleanup of HTTP session"""
        global _global_session
        try:
            if _global_session and not _global_session.closed:
                _global_session._connector._close()
        except Exception:
            pass
        _global_session = None
    
    try:
        logger.info("Application shutting down - cleaning up resources")
        
        # Cancel any pending background tasks (non-blocking)
        try:
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                logger.info(f"Cancelling {len(tasks)} pending tasks...")
                for task in tasks:
                    try:
                        task.cancel()
                    except Exception:
                        pass
                logger.info("Background tasks cancellation initiated")
        except Exception as e:
            logger.warning(f"Error cancelling tasks: {e}")
        
        # Cleanup HTTP session - try async first, force sync if fails
        cleaned = False
        try:
            # Don't use shield - it can still be cancelled
            # Use a simple timeout instead
            try:
                await asyncio.wait_for(cleanup_http_session(), timeout=1.0)
                logger.info("HTTP session cleaned up")
                cleaned = True
            except asyncio.TimeoutError:
                logger.warning("HTTP cleanup timed out - forcing closure")
            except asyncio.CancelledError:
                logger.info("HTTP cleanup cancelled - forcing closure")
        except Exception as e:
            logger.warning(f"Error during HTTP cleanup: {e}")
        
        # Always force cleanup if async didn't work
        if not cleaned:
            force_cleanup()
            logger.info("HTTP session force-closed")
        
        logger.info("Application shutdown complete")
        
    except asyncio.CancelledError:
        # Catch CancelledError and don't re-raise
        logger.info("Shutdown interrupted - forcing final cleanup")
        force_cleanup()
    except Exception as e:
        logger.error(f"Unexpected error during shutdown: {e}")
        force_cleanup()
    
    # Ensure we never raise an exception from this handler
    # This prevents CancelledError from propagating to uvicorn


# Cache utility functions for application-wide use
def invalidate_user_cache(user_id: str):
    """Invalidate all cached data for a user"""
    TradingDataCache.invalidate_user_data(user_id)
    ui.notify("User cache invalidated", type="info")

def invalidate_market_cache():
    """Invalidate market data caches"""
    TradingDataCache.invalidate_market_data()

def invalidate_strategy_cache():
    """Invalidate strategy and execution related caches"""
    frontend_cache.delete_pattern("fetch_api:*/strategies*")
    frontend_cache.delete_pattern("fetch_api:*/executions*")
    frontend_cache.delete_pattern("safe_api:/strategies*")
    frontend_cache.delete_pattern("safe_api:/executions*")

def invalidate_position_cache(broker: str = None):
    """Invalidate position and order related caches"""
    if broker:
        frontend_cache.delete_pattern(f"fetch_api:*/positions/{broker}*")
        frontend_cache.delete_pattern(f"fetch_api:*/orders/{broker}*")
    else:
        frontend_cache.delete_pattern("fetch_api:*/positions*")
        frontend_cache.delete_pattern("fetch_api:*/orders*")

def get_cache_stats():
    """Get cache statistics for monitoring"""
    return frontend_cache.get_stats()

def clear_all_caches():
    """Clear all frontend caches (emergency function)"""
    frontend_cache.clear()


if __name__ in {"__main__", "__mp_main__"}:
    # Generate a strong storage secret key - in production, use environment variable
    storage_secret_key = secrets.token_urlsafe(32)  # Generate strong random key
    # Add static files support for CSS
    # ui.add_css('static/styles.css')

    ui.run(
        title="AlgoTrade Pro - Advanced Trading Platform",
        port=8084,
        reload=True,  # Disable reload for clean shutdown - use external reloader if needed
        uvicorn_reload_dirs='.',
        uvicorn_reload_includes='*.py',
        storage_secret=storage_secret_key,
        favicon="🚀",
        reconnect_timeout=5.0,  # Reduced timeout to allow faster shutdown
        uvicorn_logging_level="info",
        show=True,  # Don't auto-open browser
    )
