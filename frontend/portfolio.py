from datetime import datetime
from nicegui import ui
import logging

logger = logging.getLogger(__name__)

async def render_portfolio_page(fetch_api, user_storage, broker):
    ui.label(f"Portfolio Overview - {broker}").classes("text-h4 q-pa-md")
    portfolio_container = ui.column().classes("w-full p-4 gap-4")
    status_label = ui.label("Loading portfolio...").classes("text-subtitle1 text-gray-500")

    # Define table columns aligned with get_portfolio response
    columns = [
        {'name': 'Symbol', 'label': 'Symbol', 'field': 'Symbol', 'sortable': True, 'align': 'left', 'classes': 'text-weight-bold'},
        {'name': 'Quantity', 'label': 'Qty', 'field': 'Quantity', 'align': 'right'},
        {'name': 'AvgPrice', 'label': 'Avg. Price', 'field': 'AvgPrice', 'align': 'right'},
        {'name': 'LastPrice', 'label': 'LTP', 'field': 'LastPrice', 'align': 'right'},
        {'name': 'invested_value', 'label': 'Invested', 'field': 'invested_value', 'align': 'right'},
        {'name': 'current_value', 'label': 'Current Val.', 'field': 'current_value', 'align': 'right'},
        {'name': 'PnL', 'label': 'P&L', 'field': 'PnL', 'align': 'right'},
        {'name': 'pnl_percentage', 'label': 'P&L %', 'field': 'pnl_percentage', 'align': 'right'},
        {'name': 'DayChange', 'label': 'Day Change', 'field': 'DayChange', 'align': 'right'},
        {'name': 'DayChangePct', 'label': 'Day Change %', 'field': 'DayChangePct', 'align': 'right'},
    ]

    portfolio_table = ui.table(columns=columns, rows=[], row_key='Symbol').classes('w-full bordered dense-table')
    portfolio_table.add_slot('body-cell-PnL', '''
        <q-td :props="props">
            <span :class="props.row.pnl_classes">{{ props.row.PnL }}</span>
        </q-td>
    ''')
    portfolio_table.add_slot('body-cell-pnl_percentage', '''
        <q-td :props="props">
            <span :class="props.row.pnl_percentage_classes">{{ props.row.pnl_percentage }}</span>
        </q-td>
    ''')
    portfolio_table.add_slot('body-cell-DayChange', '''
        <q-td :props="props">
            <span :class="props.row.day_change_classes">{{ props.row.DayChange }}</span>
        </q-td>
    ''')
    portfolio_table.add_slot('body-cell-DayChangePct', '''
        <q-td :props="props">
            <span :class="props.row.day_change_percentage_classes">{{ props.row.DayChangePct }}</span>
        </q-td>
    ''')

    summary_container = ui.row().classes("w-full justify-around gap-4 mb-4")

    async def refresh_portfolio():
        portfolio_container.clear()
        status_label.text = "Loading portfolio..."

        with portfolio_container:
            with summary_container:
                pass  # Populated after data fetch
            portfolio_table.rows.clear()

        try:
            response = await fetch_api(f"/portfolio/{broker}")
            logger.info(f"Portfolio API response for broker {broker}: {response}")

            if isinstance(response, dict) and response.get("error"):
                status_label.text = f"Error fetching portfolio: {response['error']}"
                ui.notify(f"Portfolio API error: {response['error']}", type="negative", position="top-right")
                logger.error(f"Portfolio API error: {response['error']}")
                return

            holdings_data = response if isinstance(response, list) else []
            if not holdings_data:
                status_label.text = "No holdings in portfolio."
                portfolio_table.update()
                return

            total_invested_value = 0.0
            total_current_value = 0.0
            total_day_pnl = 0.0
            rows_prepared = []

            for h in holdings_data:
                if not isinstance(h, dict):
                    logger.warning(f"Skipping invalid holding item: {h}")
                    continue
                try:
                    symbol = str(h.get('Symbol', 'N/A'))
                    quantity = int(float(h.get('Quantity', 0)))
                    avg_price = float(h.get('AvgPrice', 0))
                    last_price = float(h.get('LastPrice', 0))
                    day_change = float(h.get('DayChange', 0))
                    day_change_pct = float(h.get('DayChangePct', 0))

                    # Skip rows with invalid data
                    if not symbol or symbol == 'N/A' or quantity == 0:
                        logger.warning(f"Skipping holding with invalid data: {h}")
                        continue

                    invested = avg_price * quantity
                    current_val = last_price * quantity
                    pnl_val = current_val - invested
                    pnl_pct_val = (pnl_val / invested * 100) if invested != 0 else 0.0

                    total_invested_value += invested
                    total_current_value += current_val
                    total_day_pnl += day_change

                    rows_prepared.append({
                        'Symbol': symbol,
                        'Quantity': quantity,
                        'AvgPrice': f"{avg_price:.2f}",
                        'LastPrice': f"{last_price:.2f}",
                        'invested_value': f"{invested:,.2f}",
                        'current_value': f"{current_val:,.2f}",
                        'PnL': f"{pnl_val:,.2f}",
                        'pnl_percentage': f"{pnl_pct_val:.2f}%",
                        'pnl_classes': 'text-positive' if pnl_val >= 0 else 'text-negative',
                        'pnl_percentage_classes': 'text-positive' if pnl_pct_val >= 0 else 'text-negative',
                        'DayChange': f"{day_change:,.2f}",
                        'DayChangePct': f"{day_change_pct:.2f}%",
                        'day_change_classes': 'text-positive' if day_change >= 0 else 'text-negative',
                        'day_change_percentage_classes': 'text-positive' if day_change_pct >= 0 else 'text-negative',
                    })
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing holding {h.get('Symbol', 'unknown')}: {e}")
                    continue

            portfolio_table.rows = rows_prepared
            portfolio_table.update()

            summary_container.clear()
            with summary_container:
                with ui.card().classes("p-4 items-center text-center shadow-md rounded-lg flex-grow"):
                    ui.label("Total Investment").classes("text-sm text-gray-500")
                    ui.label(f"₹{total_invested_value:,.2f}").classes("text-xl font-semibold")
                with ui.card().classes("p-4 items-center text-center shadow-md rounded-lg flex-grow"):
                    ui.label("Current Value").classes("text-sm text-gray-500")
                    ui.label(f"₹{total_current_value:,.2f}").classes("text-xl font-semibold")
                overall_pnl = total_current_value - total_invested_value
                overall_pnl_color = 'text-positive' if overall_pnl >= 0 else 'text-negative'
                with ui.card().classes("p-4 items-center text-center shadow-md rounded-lg flex-grow"):
                    ui.label("Overall P&L").classes("text-sm text-gray-500")
                    ui.label(f"₹{overall_pnl:,.2f}").classes(f"text-xl font-semibold {overall_pnl_color}")
                day_pnl_color = 'text-positive' if total_day_pnl >= 0 else 'text-negative'
                with ui.card().classes("p-4 items-center text-center shadow-md rounded-lg flex-grow"):
                    ui.label("Today's P&L").classes("text-sm text-gray-500")
                    ui.label(f"₹{total_day_pnl:,.2f}").classes(f"text-xl font-semibold {day_pnl_color}")

            status_label.text = f"Portfolio updated at {datetime.now().strftime('%H:%M:%S')}"
            if not rows_prepared and holdings_data:
                status_label.text = "Portfolio data found, but could not be processed. Check logs."
                ui.notify("Error processing some portfolio items.", type="warning", position="top-right")

        except Exception as e:
            status_label.text = f"An unexpected error occurred: {e}"
            logger.exception("Unexpected error fetching portfolio")
            ui.notify(f"An unexpected error occurred: {e}", type="negative", position="top-right")
        finally:
            portfolio_table.update()

    ui.button("Refresh Portfolio", on_click=refresh_portfolio, icon="refresh").classes("button-primary mt-4 self-start")

    await refresh_portfolio()
    ui.timer(user_storage.get("portfolio_refresh_interval", 60), refresh_portfolio, active=True)