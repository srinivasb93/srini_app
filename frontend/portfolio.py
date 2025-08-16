from datetime import datetime
from nicegui import ui
import logging

logger = logging.getLogger(__name__)

async def render_portfolio_page(fetch_api, user_storage, broker):
    """Enhanced portfolio page with beautiful dashboard styling"""
    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):
        # Enhanced title section (matching dashboard.py and positions.py)
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-2"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("account_balance_wallet", size="2rem").classes("text-purple-400")
                    ui.label(f"Portfolio Overview - {broker}").classes("text-1xl font-bold theme-text-primary dashboard-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Monitor your investment portfolio and track performance").classes(
                    "text-gray-400 dashboard-subtitle")

            # Right side - Action buttons
            with ui.row().classes("items-center gap-4"):
                ui.button("Export Report", icon="download").classes("text-purple-400")
                ui.button("Refresh", icon="refresh").classes("text-gray-400")

        # Portfolio summary cards
        portfolio_container = ui.column().classes("w-full p-2 gap-4")

        # Status label for updates
        status_label = ui.label("Loading portfolio...").classes("text-sm text-gray-400 p-4")

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

        with portfolio_container:
            # Summary metrics cards ABOVE the table
            summary_container = ui.row().classes("w-full gap-4 mb-2")

            # Enhanced portfolio table card BELOW the summary cards
            with ui.card().classes("dashboard-card w-full"):
                with ui.row().classes("card-header w-full justify-between items-center p-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("table_chart", size="1.5rem").classes("text-purple-400")
                        ui.label("Holdings Details").classes("card-title")

                        # Live update indicator
                        with ui.row().classes("items-center gap-1 ml-2"):
                            ui.element('div').classes("w-2 h-2 bg-purple-400 rounded-full animate-pulse")
                            ui.label("Live Prices").classes("text-xs text-purple-400")

                portfolio_table = ui.table(columns=columns, rows=[], row_key='Symbol').classes('w-full p-2')
                # Add custom header styling with gradient background
                portfolio_table.add_slot('header', '''
                                <q-tr class="bg-gradient-to-r from-purple-500/20 to-blue-500/20 border-b-2 border-purple-400/30">
                                    <q-th v-for="col in props.cols" :key="col.name" :props="props"
                                          class="theme-text-primary font-semibold text-sm tracking-wider py-3">
                                        {{ col.label }}
                                    </q-th>
                                </q-tr>
                            ''')
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

    async def refresh_portfolio():
        status_label.text = "Loading portfolio..."

        # Clear existing table data
        portfolio_table.rows.clear()
        # portfolio_table.update()

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
                # Show empty state
                summary_container.clear()
                with summary_container:
                    with ui.column().classes("w-full text-center p-8"):
                        ui.icon("account_balance_wallet", size="4rem").classes("text-gray-500 mb-4")
                        ui.label("No holdings in portfolio").classes("text-xl text-gray-400 mb-2")
                        ui.label("Your portfolio holdings will appear here after you make investments").classes(
                            "text-sm text-gray-500")
                        ui.button("Start Trading", icon="add",
                                  on_click=lambda: ui.navigate.to('/order-management')).classes("mt-4")
                return

            total_invested_value = 0.0
            total_current_value = 0.0
            total_day_pnl = 0.0
            profitable_holdings = 0
            loss_holdings = 0
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

                    # Count profitable vs loss holdings
                    if pnl_val > 0:
                        profitable_holdings += 1
                    elif pnl_val < 0:
                        loss_holdings += 1

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

            # Calculate metrics
            overall_pnl = total_current_value - total_invested_value
            overall_pnl_pct = (overall_pnl / total_invested_value * 100) if total_invested_value > 0 else 0
            total_holdings = len(rows_prepared)

            # Update enhanced summary cards (matching positions.py style)
            summary_container.clear()
            with summary_container:
                # Total Holdings
                with ui.card().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("account_tree", size="2rem").classes("text-blue-400 mb-2")
                        ui.label("Total Holdings").classes("text-sm text-gray-400")
                        ui.label(str(total_holdings)).classes("text-2xl font-bold theme-text-primary")

                # Profitable Holdings
                with ui.card().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("trending_up", size="2rem").classes("text-green-400 mb-2")
                        ui.label("Profitable").classes("text-sm text-gray-400")
                        ui.label(str(profitable_holdings)).classes("text-2xl font-bold text-green-400")

                # Loss Holdings
                with ui.card().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("trending_down", size="2rem").classes("text-red-400 mb-2")
                        ui.label("In Loss").classes("text-sm text-gray-400")
                        ui.label(str(loss_holdings)).classes("text-2xl font-bold text-red-400")

                # Total Investment
                with ui.card().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("currency_rupee", size="2rem").classes("text-purple-400 mb-2")
                        ui.label("Total Investment").classes("text-sm text-gray-400")
                        ui.label(f"₹{total_invested_value:,.2f}").classes("text-2xl font-bold theme-text-primary")

                # Current Value
                with ui.card().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("account_balance", size="2rem").classes("text-cyan-400 mb-2")
                        ui.label("Current Value").classes("text-sm text-gray-400")
                        ui.label(f"₹{total_current_value:,.2f}").classes("text-2xl font-bold theme-text-primary")

                # Overall P&L
                overall_pnl_color = "text-green-400" if overall_pnl >= 0 else "text-red-400"
                pnl_icon = "trending_up" if overall_pnl >= 0 else "trending_down"
                with ui.card().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon(pnl_icon, size="2rem").classes(f"{overall_pnl_color} mb-2")
                        with ui.row().classes("items-center justify-center gap-2"):
                            ui.label("Overall P&L").classes("text-sm text-gray-400")
                            ui.label(f"({overall_pnl_pct:+.2f}%)").classes(f"text-sm {overall_pnl_color}")
                        ui.label(f"₹{overall_pnl:,.2f}").classes(f"text-2xl font-bold {overall_pnl_color}")


                # Today's P&L
                day_pnl_color = "text-green-400" if total_day_pnl >= 0 else "text-red-400"
                day_pnl_icon = "trending_up" if total_day_pnl >= 0 else "trending_down"
                with ui.card().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon(day_pnl_icon, size="2rem").classes(f"{day_pnl_color} mb-2")
                        with ui.row().classes("items-center justify-center gap-2"):
                            ui.label("Today's P&L").classes("text-sm text-gray-400")
                            ui.label(f"({(total_day_pnl / total_invested_value * 100 if total_invested_value > 0 else 0):+.2f}%)").classes(f"text-sm {day_pnl_color}")
                        ui.label(f"₹{total_day_pnl:,.2f}").classes(f"text-2xl font-bold {day_pnl_color}")

            # Update table with new data
            portfolio_table.rows = rows_prepared
            portfolio_table.update()

            status_label.text = f"Portfolio updated at {datetime.now().strftime('%H:%M:%S')}"
            if not rows_prepared and holdings_data:
                status_label.text = "Portfolio data found, but could not be processed. Check logs."
                ui.notify("Error processing some portfolio items.", type="warning", position="top-right")

        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            status_label.text = f"Error loading portfolio: {str(e)}"
            summary_container.clear()
            with summary_container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                    ui.label("Error loading portfolio").classes("text-xl text-red-400 mb-2")
                    ui.label(str(e)).classes("text-sm text-gray-500")

    # Auto-refresh on page load
    await refresh_portfolio()

    # Add refresh button functionality
    ui.button("Refresh Portfolio", icon="refresh", on_click=refresh_portfolio).classes("m-4")
