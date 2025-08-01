# Enhanced Watchlist Module - watchlist.py
# Applying beautiful dashboard.py styling consistently

from nicegui import ui
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


def apply_enhanced_dashboard_styles():
    """Apply enhanced CSS styles matching dashboard.py"""
    ui.add_css('static/styles.css')


async def render_watchlist_page(fetch_api, user_storage, get_cached_instruments, broker):
    """Enhanced watchlist page with beautiful dashboard styling"""

    apply_enhanced_dashboard_styles()

    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):

        # Enhanced title section (matching dashboard.py)
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("visibility", size="2rem").classes("text-cyan-400")
                    ui.label(f"My Watchlist - {broker}").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Real-time market data for your tracked instruments").classes(
                    "text-gray-400 dashboard-subtitle")

            # Right side - Action buttons
            with ui.row().classes("items-center gap-4"):
                ui.button("Import List", icon="upload").classes("text-cyan-400")
                ui.button("Export List", icon="download").classes("text-gray-400")

        # Main content in grid layout
        with ui.row().classes("w-full gap-4 p-4"):

            # Add instrument card (enhanced styling)
            with ui.card().classes("dashboard-card w-1/3"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("add_circle", size="1.5rem").classes("text-cyan-400")
                    ui.label("Add Instrument").classes("card-title")

                ui.separator().classes("card-separator")

                # Add instrument form with enhanced styling
                with ui.column().classes("p-4 gap-4"):
                    try:
                        all_instruments_map = await get_cached_instruments(broker)

                        # Convert to list of dictionaries for the select component
                        if isinstance(all_instruments_map, dict):
                            # If it's a dict mapping symbol -> token
                            instrument_options = [
                                f"{symbol} ({exchange})"
                                for symbol, token in list(all_instruments_map.items())[:100]  # Limit for performance
                                for exchange in ["NSE", "BSE"]  # Default exchanges
                            ]
                        else:
                            # If it's a list of instrument objects
                            instrument_options = [
                                f"{inst.trading_symbol} ({inst.exchange})"
                                for inst in all_instruments_map[:100]  # Limit for performance
                            ]

                        instrument_select = ui.select(
                            options=instrument_options,
                            label="Search & Select Instrument",
                            with_input=True,
                            new_value_mode="add-unique"
                        ).classes("w-full").props("use-input input-debounce=300 hide-selected")

                    except Exception as e:
                        logger.error(f"Error loading instruments: {e}")
                        # Fallback options
                        instrument_options = [
                            "RELIANCE (NSE)",
                            "TCS (NSE)",
                            "HDFCBANK (NSE)",
                            "INFY (NSE)",
                            "ITC (NSE)"
                        ]

                        instrument_select = ui.select(
                            options=instrument_options,
                            label="Select Instrument",
                            with_input=True
                        ).classes("w-full")

                    # Enhanced add button
                    add_button = ui.button(
                        "Add to Watchlist",
                        icon="add",
                        on_click=lambda: add_to_watchlist(instrument_select.value, user_storage)
                    ).props("color=primary size=md").classes("w-full")

            # Watchlist display (enhanced styling)
            with ui.card().classes("dashboard-card flex-1"):
                with ui.row().classes("card-header w-full justify-between items-center p-4"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("format_list_bulleted", size="1.5rem").classes("text-green-400")
                        ui.label("Your Watchlist").classes("card-title")

                        # Live update indicator
                        with ui.row().classes("items-center gap-1 ml-2"):
                            ui.element('div').classes("w-2 h-2 bg-green-400 rounded-full animate-pulse")
                            ui.label("Live").classes("text-xs text-green-400")

                    with ui.row().classes("items-center gap-2"):
                        ui.button(icon="refresh", on_click=lambda: refresh_watchlist()).props("flat round").classes(
                            "text-gray-400")
                        ui.button(icon="delete_sweep", on_click=lambda: clear_watchlist(user_storage)).props(
                            "flat round").classes("text-red-400")

                ui.separator().classes("card-separator")

                # Watchlist content
                watchlist_container = ui.column().classes("w-full p-4")

                # Render watchlist items
                await render_enhanced_watchlist_items(fetch_api, user_storage, get_cached_instruments, broker,
                                                      watchlist_container)


async def render_enhanced_watchlist_items(fetch_api, user_storage, get_cached_instruments, broker, container):
    """Render enhanced watchlist items with real-time data"""

    try:
        # Get user's watchlist
        watchlist_symbols = user_storage.get("user_watchlist", [])

        if not watchlist_symbols:
            # Enhanced empty state
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("visibility_off", size="4rem").classes("text-gray-500 mb-4")
                    ui.label("Your watchlist is empty").classes("text-xl text-gray-400 mb-2")
                    ui.label("Add some instruments to start tracking them").classes("text-sm text-gray-500")
            return

        # Get instruments mapping
        all_instruments_map = await get_cached_instruments(broker)

        with container:
            # Table header
            with ui.row().classes(
                    "watchlist-header w-full p-3 text-sm font-semibold text-gray-400 border-b border-gray-700 mb-2"):
                ui.label("Symbol").classes("flex-1")
                ui.label("LTP").classes("w-24 text-right")
                ui.label("Change").classes("w-24 text-right")
                ui.label("Change %").classes("w-24 text-right")
                ui.label("Volume").classes("w-24 text-right")
                ui.label("Actions").classes("w-24 text-center")

            # Render each watchlist item
            for symbol in watchlist_symbols:
                await render_enhanced_watchlist_item(symbol, all_instruments_map, fetch_api, broker, user_storage)

    except Exception as e:
        logger.error(f"Error rendering watchlist items: {e}")
        with container:
            with ui.column().classes("w-full text-center p-8"):
                ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                ui.label("Error loading watchlist").classes("text-xl text-red-400 mb-2")
                ui.label(str(e)).classes("text-sm text-gray-500")


async def render_enhanced_watchlist_item(symbol, instruments_map, fetch_api, broker, user_storage):
    """Render individual enhanced watchlist item with live data"""

    try:
        # Get instrument token for the symbol
        instrument_token = None
        if isinstance(instruments_map, dict):
            instrument_token = instruments_map.get(symbol)
        else:
            # If it's a list of objects
            for inst in instruments_map:
                if hasattr(inst, 'trading_symbol') and inst.trading_symbol == symbol:
                    instrument_token = inst.instrument_token
                    break

        # Initialize default values
        ltp = 0.0
        change = 0.0
        change_pct = 0.0
        volume = 0

        # Fetch live data if instrument token is available
        if instrument_token:
            try:
                ltp_response = await fetch_api(f"/ltp/{broker}", params={"instruments": instrument_token})
                if ltp_response and isinstance(ltp_response, list) and ltp_response:
                    ltp_data = ltp_response[0]
                    ltp = ltp_data.get('last_price', 0.0)
                    change = ltp_data.get('net_change', 0.0)
                    change_pct = ltp_data.get('percentage_change', 0.0)
                    volume = ltp_data.get('volume', 0)
            except Exception as e:
                logger.warning(f"Error fetching LTP for {symbol}: {e}")

        # Determine styling based on change
        if change > 0:
            change_color = "text-green-400"
            border_color = "border-green-500/20"
            bg_color = "hover:bg-green-900/10"
            trend_icon = "trending_up"
        elif change < 0:
            change_color = "text-red-400"
            border_color = "border-red-500/20"
            bg_color = "hover:bg-red-900/10"
            trend_icon = "trending_down"
        else:
            change_color = "text-gray-400"
            border_color = "border-gray-500/20"
            bg_color = "hover:bg-gray-800/50"
            trend_icon = "trending_flat"

        # Render watchlist item row
        with ui.row().classes(
                f"watchlist-item w-full p-3 {bg_color} transition-all duration-200 border-l-2 {border_color} mb-1 rounded-r-lg"):
            # Symbol column
            with ui.column().classes("flex-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon(trend_icon, size="1rem").classes(change_color)
                    ui.label(symbol).classes("text-white font-semibold")
                ui.label("NSE").classes("text-xs text-gray-500")

            # LTP
            ui.label(f"₹{ltp:,.2f}").classes("w-24 text-right text-white font-mono")

            # Change
            ui.label(f"₹{change:+,.2f}").classes(f"w-24 text-right {change_color} font-mono")

            # Change %
            ui.label(f"{change_pct:+.2f}%").classes(f"w-24 text-right {change_color} font-mono")

            # Volume
            ui.label(f"{volume:,}").classes("w-24 text-right text-gray-300 font-mono text-sm")

            # Actions
            with ui.row().classes("w-24 justify-center gap-1"):
                ui.button(
                    icon="shopping_cart",
                    on_click=lambda s=symbol: handle_trade_action(s)
                ).props("flat round size=sm").classes("text-cyan-400")
                ui.button(
                    icon="delete",
                    on_click=lambda s=symbol: remove_from_watchlist(s, user_storage)
                ).props("flat round size=sm").classes("text-red-400")

    except Exception as e:
        logger.error(f"Error rendering watchlist item {symbol}: {e}")
        # Fallback row with error state
        with ui.row().classes("watchlist-item w-full p-3 hover:bg-gray-800/50 border-l-2 border-red-500/20"):
            ui.label(symbol).classes("flex-1 text-white")
            ui.label("Error").classes("text-red-400 text-sm")


def add_to_watchlist(symbol, user_storage):
    """Add symbol to user's watchlist"""
    if not symbol:
        ui.notify("Please select an instrument", type="warning")
        return

    watchlist = user_storage.get("user_watchlist", [])

    if symbol in watchlist:
        ui.notify(f"{symbol} is already in your watchlist", type="warning")
        return

    watchlist.append(symbol)
    user_storage["user_watchlist"] = watchlist

    ui.notify(f"Added {symbol} to watchlist", type="positive")

    # Refresh the page to show updated watchlist
    ui.navigate.to('/watchlist')


def remove_from_watchlist(symbol, user_storage):
    """Remove symbol from user's watchlist"""
    watchlist = user_storage.get("user_watchlist", [])

    if symbol in watchlist:
        watchlist.remove(symbol)
        user_storage["user_watchlist"] = watchlist
        ui.notify(f"Removed {symbol} from watchlist", type="positive")

        # Refresh the page to show updated watchlist
        ui.navigate.to('/watchlist')
    else:
        ui.notify(f"{symbol} not found in watchlist", type="warning")


def clear_watchlist(user_storage):
    """Clear entire watchlist"""
    user_storage["user_watchlist"] = []
    ui.notify("Watchlist cleared", type="positive")
    ui.navigate.to('/watchlist')


def refresh_watchlist():
    """Refresh watchlist data"""
    ui.notify("Refreshing watchlist...", type="info")
    ui.navigate.to('/watchlist')


def handle_trade_action(symbol):
    """Handle trade action for a symbol"""
    ui.notify(f"Opening trade for {symbol}", type="info")
    # Navigate to order management with pre-filled symbol
    ui.navigate.to(f'/order-management?symbol={symbol}')