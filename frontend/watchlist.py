from nicegui import ui
import logging

logger = logging.getLogger(__name__)

async def render_watchlist_page(fetch_api, user_storage, get_cached_instruments, broker):
    ui.label("My Watchlist").classes("text-2xl font-semibold p-4")
    all_instruments_map = await get_cached_instruments(broker)
    watchlist_symbols = user_storage.get("STORAGE_WATCHLIST_KEY", [])
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
                        async def remove_from_watchlist(sym=symbol_name):
                            if sym in watchlist_symbols:
                                watchlist_symbols.remove(sym)
                                user_storage["STORAGE_WATCHLIST_KEY"] = watchlist_symbols
                                ui.notify(f"{sym} removed from watchlist.", type="info")
                                await refresh_watchlist_ltps()
                        ui.button(icon="delete", on_click=remove_from_watchlist, color="negative").props(
                            "flat dense round text-xs").tooltip(f"Remove {symbol_name}")

                    if instrument_token:
                        ltp_response = await fetch_api(f"/ltp/{broker}", params={"instruments": instrument_token})
                        if ltp_response and isinstance(ltp_response, list) and ltp_response:
                            ltp_item = ltp_response[0]
                            ltp_label.text = f"₹{ltp_item.get('last_price', 'N/A'):,.2f}"
                        else:
                            ltp_label.text = "N/A"
                    else:
                        ltp_label.text = "Token N/A"
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
                user_storage["STORAGE_WATCHLIST_KEY"] = watchlist_symbols
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