# Enhanced watchlist.py - Using your beautiful dashboard styling

from nicegui import ui
import logging

logger = logging.getLogger(__name__)


def apply_enhanced_page_styles():
    """Apply the enhanced dashboard styling to this page"""
    ui.add_css('static/styles.css')  # Use your beautiful CSS


async def render_watchlist_page(fetch_api, user_storage, get_cached_instruments, broker):
    """Enhanced watchlist page with dashboard styling"""

    # Apply your beautiful dashboard styling
    apply_enhanced_page_styles()

    # Enhanced app container (like your dashboard)
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):

        # Enhanced title section (matching dashboard style)
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("visibility", size="2rem").classes("text-cyan-400")
                    ui.label(f"My Watchlist - {broker}").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Real-time market data for your tracked instruments").classes(
                    "text-gray-400 dashboard-subtitle")

        with ui.row().classes("w-full no-wrap"):

            # Add instrument card (enhanced styling)
            with ui.card().classes("dashboard-card w-1/3"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("add_circle", size="1.5rem").classes("text-cyan-400")
                    ui.label("Add Instrument").classes("card-title")

                ui.separator().classes("card-separator")

                # Add instrument form with enhanced styling
                with ui.column().classes("p-4 gap-4"):
                    all_instruments_map = await get_cached_instruments(broker)

                    instrument_select = ui.select(
                        options=list(all_instruments_map.keys()),
                        label="Search Instrument",
                        with_input=True,
                        clearable=True
                    ).props("outlined dense dark").classes("w-full")

                    # Enhanced add button
                    async def add_to_watchlist_action():
                        selected_symbol = instrument_select.value
                        watchlist_symbols = user_storage.get("STORAGE_WATCHLIST_KEY", [])

                        if selected_symbol and selected_symbol not in watchlist_symbols:
                            watchlist_symbols.append(selected_symbol)
                            user_storage["STORAGE_WATCHLIST_KEY"] = watchlist_symbols
                            ui.notify(f"✅ {selected_symbol} added to watchlist", type="positive")
                            instrument_select.set_value(None)
                            await refresh_watchlist_ltps()
                        elif selected_symbol in watchlist_symbols:
                            ui.notify(f"⚠️ {selected_symbol} already in watchlist", type="warning")
                        else:
                            ui.notify("Please select an instrument", type="warning")

                    ui.button("Add to Watchlist", on_click=add_to_watchlist_action, icon="add").classes(
                        "buy-button w-full")

            # Main watchlist card (enhanced styling)
            with ui.card().classes("dashboard-card watchlist-card w-2/3"):
                # Header with enhanced styling
                with ui.row().classes("card-header w-full justify-between items-center p-4"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("trending_up", size="1.5rem").classes("text-green-400")
                        ui.label("Live Quotes").classes("card-title")

                    with ui.row().classes("items-center gap-2"):
                        ui.button(icon="refresh").props("flat round").classes("text-cyan-400 refresh-button")
                        ui.button(icon="settings").props("flat round").classes("text-gray-400")

                ui.separator().classes("card-separator")

                # Watchlist content with enhanced styling
                watchlist_display_area = ui.column().classes("watchlist-content w-full p-2")



    # Enhanced watchlist refresh function
    async def refresh_watchlist_ltps():
        """Refresh watchlist with enhanced styling"""
        watchlist_display_area.clear()

        watchlist_symbols = user_storage.get("STORAGE_WATCHLIST_KEY", [])
        all_instruments_map = await get_cached_instruments(broker)

        if not watchlist_symbols:
            with watchlist_display_area:
                with ui.column().classes("w-full items-center justify-center p-8"):
                    ui.icon("trending_up", size="3rem").classes("text-gray-500 mb-4")
                    ui.label("Your watchlist is empty").classes("text-gray-500 text-lg")
                    ui.label("Add instruments using the panel on the right").classes("text-gray-600 text-sm")
            return

        with watchlist_display_area:
            # Enhanced header row
            with ui.row().classes("w-full font-bold border-b border-gray-600 pb-3 mb-3"):
                ui.label("Instrument").classes("w-1/3 text-gray-300 text-sm uppercase tracking-wide")
                ui.label("Price").classes("w-1/4 text-right text-gray-300 text-sm uppercase tracking-wide")
                ui.label("Change").classes("w-1/4 text-right text-gray-300 text-sm uppercase tracking-wide")
                ui.label("Actions").classes("w-1/6 text-right text-gray-300 text-sm uppercase tracking-wide")

            # Enhanced watchlist items
            for symbol_name in list(watchlist_symbols):
                instrument_token = all_instruments_map.get(symbol_name)

                # Enhanced watchlist item with glassmorphism effect
                with ui.row().classes("watchlist-item w-full items-center p-3 mb-2"):

                    # Symbol column with enhanced styling - matches header w-1/3
                    with ui.column().classes("w-1/3 gap-1"):
                        ui.label(symbol_name).classes("font-semibold text-white text-sm symbol-text")
                        ui.label("NSE").classes("text-xs text-gray-400")

                    # Price column - matches header w-1/4
                    ltp_container = ui.column().classes("w-1/4 items-end")
                    with ltp_container:
                        ltp_label = ui.label("Fetching...").classes("text-right font-mono text-white price-text")

                    # Change column - matches header w-1/4
                    change_container = ui.column().classes("w-1/4 items-end")
                    with change_container:
                        change_label = ui.label("--").classes("text-right text-sm change-text")
                        change_pct_label = ui.label("--").classes("text-right text-xs change-pct-text")

                    # Actions column - matches header w-1/6
                    with ui.row().classes("w-1/6 justify-end gap-2"):
                        async def remove_from_watchlist(sym=symbol_name):
                            watchlist_symbols.remove(sym)
                            user_storage["STORAGE_WATCHLIST_KEY"] = watchlist_symbols
                            ui.notify(f"🗑️ {sym} removed from watchlist", type="info")
                            await refresh_watchlist_ltps()

                        ui.button(icon="delete", on_click=remove_from_watchlist).props(
                            "flat dense round size=sm"
                        ).classes("text-red-400 hover:bg-red-500/20").tooltip(f"Remove {symbol_name}")

                        ui.button(icon="show_chart").props(
                            "flat dense round size=sm"
                        ).classes("text-cyan-400 hover:bg-cyan-500/20").tooltip(f"View {symbol_name} chart")

                # Fetch and update price data with enhanced styling
                if instrument_token:
                    try:
                        ltp_response = await fetch_api(f"/ltp/{broker}", params={"instruments": instrument_token})
                        if ltp_response and isinstance(ltp_response, list) and ltp_response:
                            ltp_item = ltp_response[0]
                            price = ltp_item.get('last_price', 0)
                            change = ltp_item.get('net_change', 0)
                            change_pct = ltp_item.get('percentage_change', 0)

                            # Update price with enhanced styling
                            ltp_label.text = f"₹{price:,.2f}"
                            ltp_label.classes("text-right font-mono text-white price-text font-semibold")

                            # Update change with color coding
                            change_class = "positive-change" if change > 0 else "negative-change" if change < 0 else "neutral-change"
                            trend_icon = "trending_up" if change > 0 else "trending_down" if change < 0 else "trending_flat"

                            with change_container:
                                change_container.clear()
                                with ui.row().classes("items-center gap-1 justify-end"):
                                    ui.icon(trend_icon, size="0.75rem").classes(change_class)
                                    ui.label(f"{change:+.2f}").classes(
                                        f"text-sm {change_class} change-text font-semibold")
                                ui.label(f"({change_pct:+.2f}%)").classes(
                                    f"text-xs {change_class} change-pct-text text-right")

                        else:
                            ltp_label.text = "N/A"
                            ltp_label.classes("text-right text-gray-500")
                    except Exception as e:
                        logger.error(f"Error fetching LTP for {symbol_name}: {e}")
                        ltp_label.text = "Error"
                        ltp_label.classes("text-right text-red-500")
                else:
                    ltp_label.text = "Token N/A"
                    ltp_label.classes("text-right text-red-500")

    # Initial load and timer setup
    await refresh_watchlist_ltps()
    ui.timer(1500, refresh_watchlist_ltps)  # Refresh every 15 seconds