# Enhanced Watchlist Module - watchlist.py
# Updated to use database-based user-specific storage

from nicegui import ui
import logging
import asyncio
from datetime import datetime
from cache_manager import frontend_cache, FrontendCacheConfig

# Import market utilities
from market_utils import (
    fetch_batch_ltp_data,
    fetch_indices_sectors,
    fetch_stocks_by_index
)

logger = logging.getLogger(__name__)


class DatabaseWatchlistManager:
    """Database-based watchlist manager with user-specific storage"""

    def __init__(self, fetch_api):
        self.fetch_api = fetch_api
        self.current_watchlist = "Default"
        self.items_per_page = 20
        self.current_page = 0
        self.watchlists_cache = {}

    async def load_user_watchlists(self):
        """Load user's watchlists from database"""
        try:
            response = await self.fetch_api("/api/watchlist/")
            if response and "watchlists" in response:
                self.watchlists_cache = {wl["name"]: wl for wl in response["watchlists"]}
                return list(self.watchlists_cache.keys())
            return ["Default"]
        except Exception as e:
            logger.error(f"Error loading user watchlists: {e}")
            return ["Default"]

    async def get_watchlist_symbols(self, watchlist_name=None, page=0):
        """Get symbols for a watchlist with pagination"""
        wl_name = watchlist_name or self.current_watchlist
        try:
            response = await self.fetch_api(f"/api/watchlist/{wl_name}/symbols/", params={
                "page": page,
                "page_size": self.items_per_page
            })
            if response:
                symbols = [item["symbol"] for item in response.get("symbols", [])]
                pagination = response.get("pagination", {})
                return symbols, pagination.get("total_count", 0)
            return [], 0
        except Exception as e:
            logger.error(f"Error fetching watchlist symbols: {e}")
            return [], 0

    async def add_symbols(self, symbols, watchlist_name=None):
        """Add multiple symbols to watchlist"""
        wl_name = watchlist_name or self.current_watchlist
        try:
            if not isinstance(symbols, list):
                symbols = [symbols]

            response = await self.fetch_api(f"/api/watchlist/{wl_name}/symbols/", data={
                "symbols": symbols
            }, method="POST")

            return response.get("added_count", 0), response.get("skipped_count", 0)
        except Exception as e:
            logger.error(f"Error adding symbols to watchlist: {e}")
            return 0, 0

    async def remove_symbol(self, symbol, watchlist_name=None):
        """Remove symbol from watchlist"""
        wl_name = watchlist_name or self.current_watchlist
        try:
            await self.fetch_api(f"/api/watchlist/{wl_name}/symbols/{symbol}/", method="DELETE")
            return True
        except Exception as e:
            logger.error(f"Error removing symbol from watchlist: {e}")
            return False

    async def create_watchlist(self, name):
        """Create new watchlist"""
        try:
            await self.fetch_api("/api/watchlist/create/", data={"name": name}, method="POST")
            return True
        except Exception as e:
            logger.error(f"Error creating watchlist: {e}")
            return False

    async def delete_watchlist(self, name):
        """Delete watchlist"""
        try:
            await self.fetch_api(f"/api/watchlist/{name}/", method="DELETE")
            return True
        except Exception as e:
            logger.error(f"Error deleting watchlist: {e}")
            return False

    async def clear_watchlist(self, watchlist_name=None):
        """Clear all symbols from watchlist"""
        wl_name = watchlist_name or self.current_watchlist
        try:
            await self.fetch_api(f"/api/watchlist/{wl_name}/symbols/", method="DELETE")
            return True
        except Exception as e:
            logger.error(f"Error clearing watchlist: {e}")
            return False


async def render_watchlist_page(fetch_api, user_storage, get_cached_instruments, broker):
    """Enhanced watchlist page with database-based user-specific storage"""

    # Initialize database-based watchlist manager
    watchlist_manager = DatabaseWatchlistManager(fetch_api)

    # Define helper functions first
    watchlist_container = None
    pagination_container = None
    watchlist_title = None
    watchlist_select = None

    async def switch_watchlist(name):
        watchlist_manager.current_watchlist = name
        watchlist_manager.current_page = 0
        if watchlist_title:
            watchlist_title.text = f"{name} Watchlist"
        await render_watchlist_content()

    async def refresh_watchlist_display():
        ui.notify("Refreshing watchlist...", type="info")
        await render_watchlist_content()

    async def clear_current_watchlist():
        success = await watchlist_manager.clear_watchlist()
        if success:
            ui.notify(f"Cleared {watchlist_manager.current_watchlist} watchlist", type="positive")
            await render_watchlist_content()
        else:
            ui.notify("Error clearing watchlist", type="negative")

    async def show_create_watchlist_dialog():
        with ui.dialog() as dialog, ui.card():
            ui.label("Create New Watchlist").classes("text-lg font-bold mb-4")
            name_input = ui.input("Watchlist Name", placeholder="Enter watchlist name")
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                async def create_and_close():
                    if name_input.value:
                        success = await watchlist_manager.create_watchlist(name_input.value)
                        if success:
                            # Reload watchlist options
                            watchlist_names = await watchlist_manager.load_user_watchlists()
                            if watchlist_select:
                                watchlist_select.options = watchlist_names
                                watchlist_select.value = name_input.value
                            watchlist_manager.current_watchlist = name_input.value
                            ui.notify(f"Created watchlist: {name_input.value}", type="positive")
                            dialog.close()
                            # FIXED: Only call switch_watchlist, which internally calls render_watchlist_content
                            # await switch_watchlist(name_input.value)
                        else:
                            ui.notify("Error creating watchlist or name already exists", type="warning")
                    else:
                        ui.notify("Please enter a watchlist name", type="warning")

                ui.button("Create", on_click=create_and_close).props("color=primary")
        dialog.open()

    async def show_delete_watchlist_dialog():
        if watchlist_manager.current_watchlist == "Default":
            ui.notify("Cannot delete Default watchlist", type="warning")
            return

        with ui.dialog() as dialog, ui.card():
            ui.label(f"Delete '{watchlist_manager.current_watchlist}' watchlist?").classes("text-lg font-bold mb-4")
            ui.label("This action cannot be undone.").classes("text-gray-500 mb-4")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                async def delete_and_close():
                    name = watchlist_manager.current_watchlist
                    success = await watchlist_manager.delete_watchlist(name)
                    if success:
                        ui.notify(f"Deleted watchlist: {name}", type="positive")
                        dialog.close()
                        # Switch to Default and reload
                        watchlist_manager.current_watchlist = "Default"
                        watchlist_names = await watchlist_manager.load_user_watchlists()
                        if watchlist_select:
                            watchlist_select.options = watchlist_names
                            watchlist_select.value = "Default"
                        # await switch_watchlist("Default")
                    else:
                        ui.notify("Error deleting watchlist", type="negative")

                ui.button("Delete", on_click=delete_and_close).props("color=negative")
        dialog.open()

    async def render_watchlist_content():
        if not watchlist_container or not pagination_container:
            return

        watchlist_container.clear()
        pagination_container.clear()

        symbols, total_count = await watchlist_manager.get_watchlist_symbols(
            watchlist_manager.current_watchlist,
            watchlist_manager.current_page
        )
        total_pages = (total_count + watchlist_manager.items_per_page - 1) // watchlist_manager.items_per_page

        if not symbols:
            with watchlist_container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("visibility_off", size="4rem").classes("text-gray-500 mb-4")
                    ui.label(f"{watchlist_manager.current_watchlist} watchlist is empty").classes("text-xl text-gray-400 mb-2")
                    ui.label("Add some instruments to start tracking").classes("text-sm text-gray-500")
            return

        # Get live data for current page symbols
        ltp_data_map = await fetch_batch_ltp_data(symbols, get_cached_instruments, broker, fetch_api)

        with watchlist_container:
            # Table header
            with ui.row().classes("watchlist-header w-full p-3 text-sm font-semibold text-gray-400 border-b border-gray-700 mb-2"):
                ui.label("Symbol").classes("flex-1")
                ui.label("LTP").classes("w-24 text-right")
                ui.label("Change").classes("w-24 text-right")
                ui.label("Change %").classes("w-24 text-right")
                ui.label("Volume").classes("w-24 text-right")
                ui.label("Actions").classes("w-24 text-center")

            # Render symbols for current page
            for symbol in symbols:
                await render_watchlist_item(symbol, ltp_data_map, watchlist_manager, render_watchlist_content)

            # Pagination controls
            # ui.label(symbol).classes("font-semibold theme-text-primary text-sm symbol-text")
            with pagination_container:
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"₹{ltp_data_map[symbol].get('last_price', 0):,.2f}").classes("theme-text-secondary text-xs price-text")
                    ui.label(f"({total_count} items)").classes("text-gray-500 text-sm")

                with ui.row().classes("items-center gap-2"):
                    async def go_to_page(page):
                        if 0 <= page < total_pages:
                            watchlist_manager.current_page = page
                            await render_watchlist_content()

                    ui.button(icon="first_page", on_click=lambda: go_to_page(0)).props("flat round").classes("text-gray-400").set_enabled(watchlist_manager.current_page > 0)
                    ui.button(icon="chevron_left", on_click=lambda: go_to_page(watchlist_manager.current_page - 1)).props("flat round").classes("text-gray-400").set_enabled(watchlist_manager.current_page > 0)
                    ui.button(icon="chevron_right", on_click=lambda: go_to_page(watchlist_manager.current_page + 1)).props("flat round").classes("text-gray-400").set_enabled(watchlist_manager.current_page < total_pages - 1)
                    ui.button(icon="last_page", on_click=lambda: go_to_page(total_pages - 1)).props("flat round").classes("text-gray-400").set_enabled(watchlist_manager.current_page < total_pages - 1)


    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):

        # Enhanced title section
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("visibility", size="2rem").classes("text-cyan-400")
                    ui.label(f"Watchlists - {broker}").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Manage and track your favorite instruments").classes("text-gray-400 dashboard-subtitle")

            # Right side - Watchlist management
            with ui.row().classes("items-center gap-4"):
                # Load user watchlists
                watchlist_names = await watchlist_manager.load_user_watchlists()

                # Watchlist selector
                watchlist_select = ui.select(
                    options=watchlist_names,
                    value=watchlist_manager.current_watchlist,
                    label="Select Watchlist",
                    on_change=lambda e: switch_watchlist(e.value),
                ).classes("w-48")

                # Create new watchlist button
                ui.button("New Watchlist", icon="add", on_click=lambda: show_create_watchlist_dialog()).classes("text-cyan-400")

                # Delete watchlist button
                ui.button("Delete Watchlist", icon="delete", on_click=lambda: show_delete_watchlist_dialog()).classes("text-red-400")

        # Main content layout (same structure, updated add function)
        with ui.row().classes("w-full gap-4 p-4"):

            # Left side - Enhanced Add Instrument Card
            with ui.card().classes("dashboard-card w-1/3"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("add_circle", size="1.5rem").classes("text-cyan-400")
                    ui.label("Add Instrument").classes("card-title")

                ui.separator().classes("card-separator")

                # Add instrument form with enhanced filtering
                with ui.column().classes("p-4 gap-4"):

                    # FIXED: Radio button for selection type
                    with ui.column().classes("gap-2"):
                        ui.label("Selection Type").classes("text-sm font-medium text-gray-300")
                        selection_type = ui.radio(
                            options=["Index/Sector", "Direct Search"],
                            value="Index/Sector"
                        ).classes("text-gray-300")

                    # Container for dynamic content
                    selection_container = ui.column().classes("gap-4 mt-2")

                    # State variables
                    current_indices = []
                    index_select = None
                    stock_select = None
                    direct_search_select = None

                    async def update_selection_ui():
                        """Update UI based on selection type"""
                        selection_container.clear()

                        with selection_container:
                            if selection_type.value == "Index/Sector":
                                # Index/Sector selection mode
                                ui.label("Select Index/Sector").classes("text-sm font-medium text-gray-300")

                                # Fetch and populate indices
                                nonlocal current_indices, index_select, stock_select
                                current_indices = await fetch_indices_sectors(fetch_api)

                                index_select = ui.select(
                                    options=current_indices,
                                    label="Index/Sector"
                                ).classes("w-full")

                                ui.label("Select Stock Symbol").classes("text-sm font-medium text-gray-300 mt-2")
                                stock_select = ui.select(
                                    options=[],
                                    label="Stock Symbol",
                                    with_input=True,
                                    multiple=True  # Enable multiple selection
                                ).classes("w-full").props("use-input input-debounce=300")

                                # FIXED: Set up the event handler after the dropdown is created
                                index_select.on_value_change(handle_index_change)

                            else:
                                # Direct search mode (fallback to original)
                                ui.label("Search & Select Instrument").classes("text-sm font-medium text-gray-300")
                                try:
                                    all_instruments_map = await get_cached_instruments(broker)

                                    if isinstance(all_instruments_map, dict):
                                        instrument_options = [symbol for symbol in list(all_instruments_map.keys())[:500]]
                                    else:
                                        instrument_options = [inst.trading_symbol for inst in all_instruments_map[:500]]

                                    nonlocal direct_search_select
                                    direct_search_select = ui.select(
                                        options=instrument_options,
                                        label="Search Instrument",
                                        with_input=True,
                                        new_value_mode="add-unique",
                                    ).classes("w-full").props("use-input input-debounce=300 hide-selected")

                                except Exception as e:
                                    logger.error(f"Error loading instruments: {e}")
                                    direct_search_select = ui.select(
                                        options=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"],
                                        label="Select Instrument",
                                        with_input=True
                                    ).classes("w-full")

                    async def update_stocks_dropdown():
                        """Update stocks dropdown based on selected index"""
                        if index_select and index_select.value:
                            try:
                                stocks = await fetch_stocks_by_index(fetch_api, index_select.value)
                                logger.info(f"Fetched {len(stocks)} stocks for {index_select.value}: {stocks[:5]}...")
                                if stock_select:
                                    stock_select.options = stocks
                                    stock_select.value = None
                                    stock_select.update()
                                    logger.info(f"Updated stock_select options to {len(stocks)} items")
                            except Exception as e:
                                logger.error(f"Error in update_stocks_dropdown: {e}")

                    # FIXED: Properly handle async function call for index selection with UI context
                    def handle_index_change():
                        """Wrapper to properly handle async update_stocks_dropdown with UI context"""
                        if index_select and index_select.value:
                            # Use asyncio to run the coroutine properly
                            import asyncio
                            asyncio.create_task(update_stocks_dropdown())

                    # Initialize with default selection
                    await update_selection_ui()

                    # REMOVED: Redundant event handler setup - now done inside update_selection_ui()

                    # Update UI when selection type changes
                    selection_type.on_value_change(lambda: update_selection_ui())

                    # Add to watchlist button
                    async def add_to_current_watchlist():
                        symbols = None

                        if selection_type.value == "Index/Sector":
                            symbols = stock_select.value if stock_select else None
                        else:
                            symbols = direct_search_select.value if direct_search_select else None

                        if not symbols:
                            ui.notify("Please select at least one instrument", type="warning")
                            return

                        # Handle both single selection and multiple selection
                        if not isinstance(symbols, list):
                            symbols = [symbols]

                        # Extract symbol names
                        symbol_names = []
                        for symbol in symbols:
                            symbol_name = symbol.split(" (")[0] if " (" in symbol else symbol
                            symbol_names.append(symbol_name)

                        # Add to database
                        added_count, skipped_count = await watchlist_manager.add_symbols(symbol_names)

                        # Show appropriate notification
                        if added_count > 0:
                            ui.notify(f"Added {added_count} symbol(s) to {watchlist_manager.current_watchlist}", type="positive")
                        if skipped_count > 0:
                            ui.notify(f"{skipped_count} symbol(s) already existed in watchlist", type="warning")

                        # Clear selections
                        if stock_select:
                            stock_select.value = None
                        if direct_search_select:
                            direct_search_select.value = None

                        await refresh_watchlist_display()

                    ui.button(
                        "Add to Watchlist",
                        icon="add",
                        on_click=add_to_current_watchlist,
                    ).props("color=primary size=md").classes("w-full mt-4")

            # Right side - Watchlist Display (FIXED: removed delete button from here)
            with ui.card().classes("dashboard-card flex-1"):
                with ui.row().classes("card-header w-full justify-between items-center p-4"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("format_list_bulleted", size="1.5rem").classes("text-green-400")
                        watchlist_title = ui.label(f"{watchlist_manager.current_watchlist} Watchlist").classes("card-title")

                        # Live indicator
                        with ui.row().classes("items-center gap-1 ml-2"):
                            ui.element('div').classes("w-2 h-2 bg-green-400 rounded-full animate-pulse")
                            ui.label("Live").classes("text-xs text-green-400")

                    with ui.row().classes("items-center gap-2"):
                        # FIXED: Only refresh and clear buttons (delete moved to header)
                        ui.button(icon="refresh", on_click=refresh_watchlist_display).props("flat round").classes("text-gray-400")
                        ui.button(icon="delete_sweep", on_click=clear_current_watchlist).props("flat round").classes("text-red-400")

                ui.separator().classes("card-separator")

                # Watchlist content container
                watchlist_container = ui.column().classes("w-full")

                # Pagination controls
                pagination_container = ui.row().classes("w-full justify-between items-center p-4")

                # Initial render
                await render_watchlist_content()

    # FIXED: Pagination for large watchlists
    async def render_watchlist_content():
        if not watchlist_container or not pagination_container:
            return

        watchlist_container.clear()
        pagination_container.clear()

        symbols, total_count = await watchlist_manager.get_watchlist_symbols(
            watchlist_manager.current_watchlist,
            watchlist_manager.current_page
        )
        total_pages = (total_count + watchlist_manager.items_per_page - 1) // watchlist_manager.items_per_page

        if not symbols:
            with watchlist_container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("visibility_off", size="4rem").classes("text-gray-500 mb-4")
                    ui.label(f"{watchlist_manager.current_watchlist} watchlist is empty").classes("text-xl text-gray-400 mb-2")
                    ui.label("Add some instruments to start tracking").classes("text-sm text-gray-500")
            return

        # Get live data for current page symbols
        ltp_data_map = await fetch_batch_ltp_data(symbols, get_cached_instruments, broker, fetch_api)

        with watchlist_container:
            # Table header
            with ui.row().classes("watchlist-header w-full p-3 text-sm font-semibold text-gray-400 border-b border-gray-700 mb-2"):
                ui.label("Symbol").classes("flex-1")
                ui.label("LTP").classes("w-24 text-right")
                ui.label("Change").classes("w-24 text-right")
                ui.label("Change %").classes("w-24 text-right")
                ui.label("Volume").classes("w-24 text-right")
                ui.label("Actions").classes("w-24 text-center")

            # Render symbols for current page
            for symbol in symbols:
                await render_watchlist_item(symbol, ltp_data_map, watchlist_manager, render_watchlist_content)

        # FIXED: Pagination controls
        if total_pages > 1:
            with pagination_container:
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"Page {watchlist_manager.current_page + 1} of {total_pages}").classes("text-gray-400")
                    ui.label(f"({total_count} items)").classes("text-gray-500 text-sm")

                with ui.row().classes("items-center gap-2"):
                    async def go_to_page(page):
                        if 0 <= page < total_pages:
                            watchlist_manager.current_page = page
                            await render_watchlist_content()

                    ui.button(icon="first_page", on_click=lambda: go_to_page(0)).props("flat round").classes("text-gray-400").set_enabled(watchlist_manager.current_page > 0)
                    ui.button(icon="chevron_left", on_click=lambda: go_to_page(watchlist_manager.current_page - 1)).props("flat round").classes("text-gray-400").set_enabled(watchlist_manager.current_page > 0)
                    ui.button(icon="chevron_right", on_click=lambda: go_to_page(watchlist_manager.current_page + 1)).props("flat round").classes("text-gray-400").set_enabled(watchlist_manager.current_page < total_pages - 1)
                    ui.button(icon="last_page", on_click=lambda: go_to_page(total_pages - 1)).props("flat round").classes("text-gray-400").set_enabled(watchlist_manager.current_page < total_pages - 1)




async def render_watchlist_item(symbol, ltp_data_map, watchlist_manager, refresh_callback):
    """Render individual watchlist item"""
    try:
        # Get live data
        ltp_data = ltp_data_map.get(symbol, {})
        ltp = ltp_data.get("last_price", 0.0)
        volume = ltp_data.get("volume", 0)
        previous_close = ltp_data.get("previous_close", 0.0)

        # Calculate change
        change = ltp - previous_close if previous_close > 0 else 0.0
        change_pct = (change / previous_close) * 100 if previous_close > 0 else 0.0

        # Styling based on change
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

        # Render item row
        with ui.row().classes(f"watchlist-item w-full p-3 {bg_color} transition-all duration-200 border-l-2 {border_color} mb-1 rounded-r-lg"):
            # Symbol column
            with ui.column().classes("flex-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon(trend_icon, size="1rem").classes(change_color)
                    ui.label(symbol).classes("text-white font-semibold")

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
                ui.button(icon="shopping_cart", on_click=lambda s=symbol: handle_trade_action(s)).props("flat round size=sm").classes("text-cyan-400")

                # FIXED: Use the passed refresh callback instead of undefined function
                async def remove_symbol(sym=symbol):
                    success = await watchlist_manager.remove_symbol(sym)
                    if success:
                        ui.notify(f"Removed {sym} from watchlist", type="positive")
                        # Use the passed refresh callback
                        await refresh_callback()
                    else:
                        ui.notify("Error removing symbol", type="negative")

                ui.button(icon="delete", on_click=lambda s=symbol: remove_symbol(s)).props("flat round size=sm").classes("text-red-400")

    except Exception as e:
        logger.error(f"Error rendering watchlist item {symbol}: {e}")

def handle_trade_action(symbol):
    """Handle trade action for a symbol"""
    ui.notify(f"Opening trade for {symbol}", type="info")
    ui.navigate.to(f'/order-management')
