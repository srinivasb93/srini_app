from nicegui import ui
import logging
import asyncio

logger = logging.getLogger(__name__)

async def render_positions_page(fetch_api, user_storage, broker):
    ui.label(f"Current Positions - {broker}").classes("text-2xl font-semibold p-4")
    positions_container = ui.column().classes("w-full p-4")

    async def refresh_positions():
        positions_container.clear()
        try:
            positions_data = await fetch_api(f"/positions/{broker}")
            logger.info(f"Positions API response for broker {broker}: {positions_data}")

            if not positions_data or not isinstance(positions_data, list):
                with positions_container:
                    ui.label("No open positions or invalid response.").classes("text-orange-600")
                return

            if not positions_data:
                with positions_container:
                    ui.label("No open positions.").classes("text-gray-500")
                return

            total_pnl = 0.0
            formatted_positions = []
            for pos in positions_data:
                try:
                    symbol = str(pos.get('Symbol', 'N/A'))
                    quantity = int(float(pos.get('Quantity', 0)))
                    avg_price = float(pos.get('AvgPrice', 0))
                    last_price = float(pos.get('LastPrice', 0))
                    pnl = float(pos.get('PnL', 0))
                    instrument_token = str(pos.get('InstrumentToken', ''))

                    # Skip rows with invalid data
                    if not symbol or symbol == 'N/A' or quantity == 0 or not instrument_token:
                        logger.warning(f"Skipping position with invalid data: {pos}")
                        continue

                    total_pnl += pnl
                    formatted_positions.append({
                        'Symbol': symbol,
                        'Product': pos.get('Product', 'N/A'),
                        'Quantity': quantity,
                        'AvgPrice': f"{avg_price:.2f}",
                        'LastPrice': f"{last_price:.2f}",
                        'PnL': f"{pnl:.2f}",
                        'pnl_classes': 'text-green-500' if pnl >= 0 else 'text-red-500',
                        'InstrumentToken': instrument_token
                    })
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing position {pos.get('Symbol', 'unknown')}: {e}")
                    continue

            with positions_container:
                with ui.card().classes("p-4 mb-4 shadow-md rounded-lg"):
                    ui.label(f"Overall P&L: ₹{total_pnl:,.2f}").classes(
                        f"text-xl font-semibold {'text-green-500' if total_pnl >= 0 else 'text-red-500'}")
                columns = [
                    {'name': 'Symbol', 'label': 'Symbol', 'field': 'Symbol', 'sortable': True, 'align': 'left'},
                    {'name': 'Product', 'label': 'Product', 'field': 'Product'},
                    {'name': 'Quantity', 'label': 'Qty', 'field': 'Quantity', 'align': 'right'},
                    {'name': 'AvgPrice', 'label': 'Avg. Price', 'field': 'AvgPrice', 'align': 'right'},
                    {'name': 'LastPrice', 'label': 'LTP', 'field': 'LastPrice', 'align': 'right'},
                    {'name': 'PnL', 'label': 'P&L', 'field': 'PnL', 'align': 'right'},
                    {'name': 'actions', 'label': 'Actions', 'field': 'Symbol'}
                ]
                table = ui.table(columns=columns, rows=formatted_positions, row_key='Symbol').classes(
                    'w-full bordered dense-table shadow-md rounded-lg')
                table.add_slot('body-cell-PnL', '''
                    <q-td :props="props">
                        <span :class="props.row.pnl_classes">{{ props.row.PnL }}</span>
                    </q-td>
                ''')
                table.add_slot('body-cell-actions', '''
                    <q-td :props="props">
                        <q-btn dense flat round color="primary" icon="exit_to_app" 
                               @click="() => $parent.$emit('square_off', props.row)">
                            <q-tooltip>Square Off</q-tooltip>
                        </q-btn>
                    </q-td>
                ''')

                async def handle_square_off(position_row):
                    if not position_row.get('Symbol') or not position_row.get('InstrumentToken'):
                        ui.notify("Invalid position data for square-off.", type="error")
                        return
                    qty_to_square = abs(position_row.get('Quantity', 0))
                    if qty_to_square == 0:
                        ui.notify("Position quantity is zero, cannot square off.", type="warning")
                        return
                    square_off_payload = {
                        "broker": broker,
                        "trading_symbol": position_row['Symbol'],
                        "instrument_token": position_row['InstrumentToken'],
                        "quantity": qty_to_square,
                        "product_type": position_row['Product'],
                        "order_type": "MARKET",
                        "transaction_type": "SELL" if position_row.get('Quantity', 0) > 0 else "BUY",
                        "validity": "DAY",
                        "price": 0,
                        "trigger_price": 0
                    }
                    response = await fetch_api("/orders/", method="POST", data=square_off_payload)
                    if response and response.get('order_id'):
                        ui.notify(f"Square off order for {position_row['Symbol']} placed: Order ID {response['order_id']}", type="info")
                        await refresh_positions()
                    else:
                        ui.notify(f"Failed to square off {position_row['Symbol']}.", type="error")

                table.on('square_off', lambda e: asyncio.create_task(handle_square_off(e.args)))

        except Exception as e:
            with positions_container:
                ui.label(f"Error fetching positions: {str(e)}").classes("text-red-500")
            logger.exception("Unexpected error fetching positions")

    await refresh_positions()
    ui.timer(1500, refresh_positions)