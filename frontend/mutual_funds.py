from nicegui import ui

async def render_mutual_funds_page(fetch_api, broker):
    ui.label(f"Mutual Funds - {broker}").classes("text-2xl font-semibold p-4")
    with ui.card().classes("w-full p-4"):
        with ui.tabs().props("dense").classes("w-full").on_value_change(
                lambda e: mf_tab_panels.set_value(e.args)) as mf_tabs:
            ui.tab(name="place_order", label="Place Order")
            ui.tab(name="holdings", label="Holdings")
            ui.tab(name="sips", label="SIPs")

        with ui.tab_panels(mf_tabs, value="place_order").classes("w-full pt-4") as mf_tab_panels:
            with ui.tab_panel("place_order"):
                ui.label("Place Mutual Fund Order").classes("text-h6")
                scheme_select = ui.select(options=[], label="Select Scheme", with_input=True).props(
                    "clearable filter").classes("w-full")

                async def fetch_mf_instruments():
                    instruments = await fetch_api("/mutual-funds/instruments")
                    if instruments and isinstance(instruments, list):
                        scheme_select.options = {inst["tradingsymbol"]: inst["tradingsymbol"] for inst in instruments}
                        scheme_select.update()
                    else:
                        ui.notify("Could not fetch mutual fund instruments.", type="warning")

                await fetch_mf_instruments()

                ui.label("Transaction Type").classes("text-subtitle1")
                mf_transaction_type = ui.radio(options=["BUY", "SELL"], value="BUY").props("inline dense")

                mf_amount = ui.number(label="Amount (₹)", value=1000, min=100, step=100, format="%.2f").classes(
                    "w-full")

                async def place_mf_order():
                    if not scheme_select.value or mf_amount.value < 100:
                        ui.notify("Please select a scheme and enter a valid amount.", type="negative")
                        return
                    order_data = {
                        "scheme_code": scheme_select.value,
                        "amount": float(mf_amount.value),
                        "transaction_type": mf_transaction_type.value
                    }
                    response = await fetch_api("/mutual-funds/orders", method="POST", data=order_data)
                    if isinstance(response, dict) and "error" in response:
                        ui.notify(f"Failed to place order: {response['error']}", type="negative")
                    elif response and response.get("order_id"):
                        ui.notify(f"Order placed successfully: {response['order_id']}", type="positive")
                    else:
                        ui.notify("Failed to place order.", type="negative")

                ui.button("Place Order", on_click=place_mf_order).props("color=primary").classes("mt-4")

            with ui.tab_panel("holdings"):
                ui.label("Mutual Fund Holdings").classes("text-h6")
                holdings_grid = ui.aggrid({
                    "columnDefs": [
                        {"headerName": "Scheme", "field": "tradingsymbol"},
                        {"headerName": "Quantity", "field": "quantity"},
                        {"headerName": "Avg. Price", "field": "average_price"},
                        {"headerName": "Current Value", "field": "current_value"}
                    ],
                    "rowData": [],
                    "pagination": True,
                    "paginationPageSize": 10
                }).classes("w-full")

                async def fetch_holdings():
                    holdings = await fetch_api("/mutual-funds/holdings")
                    if holdings and isinstance(holdings, list):
                        holdings_grid.options["rowData"] = holdings
                        holdings_grid.update()
                    else:
                        ui.notify("Could not fetch holdings.", type="warning")

                await fetch_holdings()
                ui.button("Refresh Holdings", on_click=fetch_holdings).props("outline").classes("mt-4")

            with ui.tab_panel("sips"):
                ui.label("Mutual Fund SIPs").classes("text-h6")
                sips_grid = ui.aggrid({
                    "columnDefs": [
                        {"headerName": "SIP ID", "field": "sip_id"},
                        {"headerName": "Scheme", "field": "scheme_code"},
                        {"headerName": "Amount", "field": "amount"},
                        {"headerName": "Frequency", "field": "frequency"},
                        {"headerName": "Start Date", "field": "start_date"}
                    ],
                    "rowData": [],
                    "rowSelection": "single",
                    "pagination": True,
                    "paginationPageSize": 10
                }).classes("w-full")

                async def fetch_sips():
                    sips = await fetch_api("/mutual-funds/sips")
                    if sips and isinstance(sips, list):
                        sips_grid.options["rowData"] = sips
                        sips_grid.update()
                    else:
                        ui.notify("Could not fetch SIPs.", type="warning")

                await fetch_sips()
                ui.button("Refresh SIPs", on_click=fetch_sips).props("outline").classes("mt-4")