"""
Orderbook Module for NiceGUI Algo Trading Application
Displays tables for Placed, Scheduled, GTT, and Auto Orders with modify and cancel options
"""

import logging
from nicegui import ui
import asyncio
from datetime import datetime, timedelta
import json
from ui_context_manager import safe_notify, create_safe_task, with_safe_ui_context

logger = logging.getLogger(__name__)

# Define tables at module scope
placed_orders_table = None
scheduled_orders_table = None
gtt_orders_table = None
auto_orders_table = None

async def refresh_placed_orders(table, message_container, fetch_api, broker):
    """Refresh placed orders with proper UI context handling"""
    # Show loading state
    def show_loading():
        message_container.clear()
        with message_container:
            with ui.element().classes("relative"):
                ui.spinner(size="lg")
                ui.label("Loading placed orders...").classes("text-subtitle1 text-gray-400 ml-2")

    show_loading()

    try:
        orders_data = await fetch_api(f"/orders/{broker}")
        logger.debug(f"Fetched placed orders: {orders_data}")

        # Handle response in UI context
        def update_ui():
            message_container.clear()
            if orders_data and isinstance(orders_data, list):
                if not orders_data:
                    with message_container:
                        ui.label("No placed orders.").classes("text-gray-500 text-subtitle1")
                    table.rows = []
                    table.update()
                    return

                formatted_orders = []
                for order in orders_data:
                    try:
                        price = float(order.get('price', 0)) if order.get('price') is not None else 0
                        trigger_price = float(order.get('trigger_price', 0)) if order.get('trigger_price') is not None else 0
                        order_timestamp = order.get('order_timestamp', '')
                        if order_timestamp:
                            try:
                                order_timestamp = datetime.strptime(order_timestamp, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d %H:%M:%S')
                            except ValueError:
                                try:
                                    order_timestamp = datetime.fromisoformat(order_timestamp.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
                                except Exception as e:
                                    logger.warning(f"Failed to parse order_timestamp for order {order.get('order_id', 'unknown')}: {str(e)}")
                                    order_timestamp = 'N/A'
                        else:
                            order_timestamp = 'N/A'

                        formatted_order = {
                            'order_id': order.get('order_id', 'N/A'),
                            'trading_symbol': order.get('trading_symbol', 'N/A'),
                            'transaction_type': order.get('transaction_type', 'N/A'),
                            'quantity': order.get('quantity', 0),
                            'price': f"{price:.2f}" if price != 0 else 'N/A',
                            'trigger_price': f"{trigger_price:.2f}" if trigger_price != 0 else 'N/A',
                            'status': order.get('status', 'N/A').upper(),
                            'order_timestamp': order_timestamp,
                            'product_type': order.get('product_type', 'N/A'),
                            'actions': ''
                        }
                        formatted_orders.append(formatted_order)
                    except Exception as e:
                        logger.error(f"Error formatting order {order.get('order_id', 'unknown')}: {str(e)}")
                        continue

                table.rows = formatted_orders
                table.update()
            else:
                with message_container:
                    ui.label("Failed to load placed orders.").classes("text-red-500 text-subtitle1")
                safe_notify("Error fetching placed orders.", "negative")
                logger.error(f"Invalid placed orders response: {orders_data}")
                table.rows = []
                table.update()

        update_ui()

    except Exception as e:
        def show_error():
            message_container.clear()
            with message_container:
                ui.label(f"Error: {str(e)}").classes("text-red-500 text-subtitle1")
            safe_notify(f"Error fetching placed orders: {str(e)}", "negative")

        show_error()
        logger.error(f"Exception in refresh_placed_orders: {str(e)}")
        table.rows = []
        table.update()

async def refresh_scheduled_orders(table, message_container, fetch_api, broker):
    """Refresh scheduled orders with proper UI context handling"""
    def show_loading():
        message_container.clear()
        with message_container:
            with ui.element().classes("relative"):
                ui.spinner(size="lg")
                ui.label("Loading scheduled orders...").classes("text-subtitle1 text-gray-400 ml-2")

    show_loading()

    try:
        orders_data = await fetch_api(f"/scheduled-orders/{broker}")
        logger.debug(f"Fetched scheduled orders: {orders_data}")

        def update_ui():
            message_container.clear()
            if orders_data and isinstance(orders_data, list):
                if not orders_data:
                    with message_container:
                        ui.label("No scheduled orders.").classes("text-gray-500 text-subtitle1")
                    table.rows = []
                    table.update()
                    return

                formatted_orders = []
                for order in orders_data:
                    try:
                        price = float(order.get('price', 0)) if order.get('price') is not None else 0
                        trigger_price = float(order.get('trigger_price', 0)) if order.get('trigger_price') is not None else 0
                        schedule_datetime = order.get('schedule_datetime', '')
                        if schedule_datetime:
                            try:
                                schedule_datetime = datetime.fromisoformat(schedule_datetime.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
                            except Exception as e:
                                logger.warning(f"Failed to parse schedule_datetime for order {order.get('scheduled_order_id', 'unknown')}: {str(e)}")
                                schedule_datetime = 'N/A'
                        else:
                            schedule_datetime = 'N/A'

                        formatted_order = {
                            'scheduled_order_id': order.get('scheduled_order_id', 'N/A'),
                            'trading_symbol': order.get('trading_symbol', 'N/A'),
                            'instrument_token': order.get('instrument_token', 'N/A'),
                            'transaction_type': order.get('transaction_type', 'N/A'),
                            'quantity': order.get('quantity', 0),
                            'price': f"{price:.2f}" if price != 0 else 'N/A',
                            'trigger_price': f"{trigger_price:.2f}" if trigger_price != 0 else 0.0,
                            'status': order.get('status', 'N/A').upper(),
                            'schedule_datetime': schedule_datetime,
                            'product_type': order.get('product_type', 'N/A'),
                            'order_type': order.get('order_type', 'N/A'),
                            'actions': ''
                        }
                        formatted_orders.append(formatted_order)
                    except Exception as e:
                        logger.error(f"Error formatting scheduled order {order.get('scheduled_order_id', 'unknown')}: {str(e)}")
                        continue
                table.rows = formatted_orders
                table.update()
            else:
                with message_container:
                    ui.label("No data available for scheduled orders.").classes("text-blue-500 text-subtitle1")
                safe_notify("Error fetching scheduled orders.", "negative")
                logger.error(f"Invalid scheduled orders response: {orders_data}")
                table.rows = []
                table.update()

        update_ui()

    except Exception as e:
        def show_error():
            message_container.clear()
            with message_container:
                ui.label(f"Error: {str(e)}").classes("text-red-500 text-subtitle1")
            safe_notify(f"Error fetching scheduled orders: {str(e)}", "negative")

        show_error()
        logger.error(f"Exception in refresh_scheduled_orders: {str(e)}")
        table.rows = []
        table.update()

async def refresh_gtt_orders(table, message_container, fetch_api, broker):
    """Refresh GTT orders with proper UI context handling"""
    def show_loading():
        message_container.clear()
        with message_container:
            with ui.element().classes("relative"):
                ui.spinner(size="lg")
                ui.label("Loading GTT orders...").classes("text-subtitle1 text-gray-400 ml-2")

    show_loading()

    try:
        orders_data = await fetch_api(f"/gtt-orders/{broker}")
        logger.debug(f"Fetched GTT orders: {orders_data}")

        def update_ui():
            message_container.clear()
            if orders_data and isinstance(orders_data, list):
                if not orders_data:
                    with message_container:
                        ui.label("No GTT orders.").classes("text-gray-500 text-subtitle1")
                    table.rows = []
                    table.update()
                    return

                formatted_orders = []
                for order in orders_data:
                    try:
                        trigger_price = float(order.get('trigger_price', 0)) if order.get('trigger_price') is not None else 0
                        limit_price = float(order.get('limit_price', 0)) if order.get('limit_price') is not None else 0
                        created_at = order.get('created_at', '')
                        if created_at:
                            try:
                                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
                            except Exception as e:
                                logger.warning(f"Failed to parse created_at for GTT order {order.get('gtt_order_id', 'unknown')}: {str(e)}")
                                created_at = 'N/A'
                        else:
                            created_at = 'N/A'

                        formatted_order = {
                            'gtt_order_id': order.get('gtt_order_id', 'N/A'),
                            'trading_symbol': order.get('trading_symbol', 'N/A'),
                            'instrument_token': order.get('instrument_token', 'N/A'),
                            'transaction_type': order.get('transaction_type', 'N/A'),
                            'quantity': order.get('quantity', 0),
                            'trigger_type': order.get('trigger_type', 'N/A').upper(),
                            'trigger_price': f"{trigger_price:.2f}" if trigger_price != 0 else 'N/A',
                            'limit_price': f"{limit_price:.2f}" if limit_price != 0 else 'N/A',
                            'status': order.get('status', 'N/A').upper(),
                            'created_at': created_at,
                            'actions': ''
                        }
                        formatted_orders.append(formatted_order)
                    except Exception as e:
                        logger.error(f"Error formatting GTT order {order.get('gtt_order_id', 'unknown')}: {str(e)}")
                        continue

                table.rows = formatted_orders
                table.update()
            else:
                with message_container:
                    if not orders_data:
                        ui.label("No data available for GTT orders.").classes("text-blue-500 text-subtitle1")
                safe_notify("Error fetching GTT orders.", "negative")
                logger.error(f"Invalid GTT orders response: {orders_data}")
                table.rows = []
                table.update()

        update_ui()

    except Exception as e:
        def show_error():
            message_container.clear()
            with message_container:
                ui.label(f"Error: {str(e)}").classes("text-red-500 text-subtitle1")
            safe_notify(f"Error fetching GTT orders: {str(e)}", "negative")

        show_error()
        logger.error(f"Exception in refresh_gtt_orders: {str(e)}")
        table.rows = []
        table.update()

async def refresh_auto_orders(table, message_container, fetch_api, broker):
    """Refresh auto orders with proper UI context handling"""
    def show_loading():
        message_container.clear()
        with message_container:
            with ui.element().classes("relative"):
                ui.spinner(size="lg")
                ui.label("Loading auto orders...").classes("text-subtitle1 text-gray-400 ml-2")

    show_loading()

    try:
        orders_data = await fetch_api(f"/auto-orders/{broker}")
        logger.debug(f"Fetched auto orders: {orders_data}")

        def update_ui():
            message_container.clear()
            if orders_data and isinstance(orders_data, list):
                if not orders_data:
                    with message_container:
                        ui.label("No auto orders.").classes("text-gray-500 text-subtitle1")
                    table.rows = []
                    table.update()
                    return

                formatted_orders = []
                for order in orders_data:
                    try:
                        stop_loss_value = float(order.get('stop_loss_value', 0)) if order.get('stop_loss_value') is not None else 0
                        target_value = float(order.get('target_value', 0)) if order.get('target_value') is not None else 0
                        formatted_order = {
                            'auto_order_id': order.get('auto_order_id', 'N/A'),
                            'trading_symbol': order.get('trading_symbol', 'N/A'),
                            'transaction_type': order.get('transaction_type', 'N/A'),
                            'risk_per_trade': f"{order.get('risk_per_trade', 0):.1f}",
                            'stop_loss_type': order.get('stop_loss_type', 'N/A'),
                            'stop_loss_value': f"{stop_loss_value:.2f}" if stop_loss_value != 0 else 'N/A',
                            'target_value': f"{target_value:.2f}" if target_value != 0 else 'N/A',
                            'status': order.get('status', 'N/A').upper(),
                            'actions': ''
                        }
                        formatted_orders.append(formatted_order)
                    except Exception as e:
                        logger.error(f"Error formatting auto order {order.get('auto_order_id', 'unknown')}: {str(e)}")
                        continue

                table.rows = formatted_orders
                table.update()
            else:
                with message_container:
                    if not orders_data:
                        ui.label("No data available for auto orders.").classes("text-blue-500 text-subtitle1")
                safe_notify("Error fetching auto orders.", "negative")
                logger.error(f"Invalid auto orders response: {orders_data}")
                table.rows = []
                table.update()

        update_ui()

    except Exception as e:
        def show_error():
            message_container.clear()
            with message_container:
                ui.label(f"Error: {str(e)}").classes("text-red-500 text-subtitle1")
            safe_notify(f"Error fetching auto orders: {str(e)}", "negative")

        show_error()
        logger.error(f"Exception in refresh_auto_orders: {str(e)}")
        table.rows = []
        table.update()

async def render_order_book_page(fetch_api, user_storage, broker):
    # Order Management header with reduced height
    with ui.card().classes("w-full mb-2 p-2"):
        ui.label(f"Order Book - {broker}").classes("text-h5 text-primary font-medium")

    # Create tabs for different order types
    with ui.tabs().classes('w-full') as tabs:
        placed_tab = ui.tab('Placed Orders')
        scheduled_tab = ui.tab('Scheduled Orders')
        gtt_tab = ui.tab('GTT Orders')
        auto_tab = ui.tab('Auto Orders')

    with ui.tab_panels(tabs).classes('w-full'):
        with ui.tab_panel(placed_tab):
            await render_placed_orders(fetch_api, user_storage, broker)
        with ui.tab_panel(scheduled_tab):
            await render_scheduled_orders(fetch_api, user_storage, broker)
        with ui.tab_panel(gtt_tab):
            await render_gtt_orders(fetch_api, user_storage, broker)
        with ui.tab_panel(auto_tab):
            await render_auto_orders(fetch_api, user_storage, broker)

    # Cancel All Open Orders button
    with ui.row().classes('w-full mt-2'):
        cancel_loading_container = ui.column().classes('w-full')

        async def cancel_all_open_orders():
            def show_loading():
                cancel_loading_container.clear()
                with cancel_loading_container:
                    with ui.element().classes("relative"):
                        ui.spinner(size="lg")
                        ui.label("Cancelling all open orders...").classes("text-subtitle1 text-gray-400 ml-2")

            show_loading()
            with cancel_loading_container:
                try:
                    # Cancel all open orders across types
                    for endpoint in [
                        f"/orders/cancel-all/{broker}",
                        f"/scheduled-orders/cancel-all/{broker}",
                        f"/gtt-orders/cancel-all/{broker}",
                        f"/auto-orders/cancel-all/{broker}"
                    ]:
                        response = await fetch_api(endpoint, method="DELETE")
                        if response and response.get("status") != "success":
                            ui.notify(f"Failed to cancel some orders: {response.get('message', 'Unknown error')}", type="negative")

                    ui.notify("All open orders cancelled successfully.", type="positive")

                    # Refresh all tables
                    global placed_orders_table, scheduled_orders_table, gtt_orders_table, auto_orders_table
                    if placed_orders_table:
                        await refresh_placed_orders(placed_orders_table, ui.column().classes('w-full mt-2'), fetch_api, broker)
                    if scheduled_orders_table:
                        await refresh_scheduled_orders(scheduled_orders_table, ui.column().classes('w-full mt-2'), fetch_api, broker)
                    if gtt_orders_table:
                        await refresh_gtt_orders(gtt_orders_table, ui.column().classes('w-full mt-2'), fetch_api, broker)
                    if auto_orders_table:
                        await refresh_auto_orders(auto_orders_table, ui.column().classes('w-full mt-2'), fetch_api, broker)

                except Exception as e:
                    ui.notify(f"Error cancelling orders: {str(e)}", type="negative")
                    logger.error(f"Error in cancel_all_open_orders: {e}")

            cancel_loading_container.clear()

        ui.button('Cancel All Open Orders', on_click=lambda: asyncio.create_task(cancel_all_open_orders()), icon="delete_forever").classes('bg-red-500 text-white px-4 py-2 rounded')

async def render_placed_orders(fetch_api, user_storage, broker):
    global placed_orders_table
    with ui.card().classes("w-full p-3"):
        ui.label("Placed Orders").classes("text-h6 mb-2")

        # Create message container within the card context
        placed_message_container = ui.column().classes('w-full mt-2')

        # Define table columns
        columns = [
            {'name': 'order_id', 'label': 'Order ID', 'field': 'order_id', 'sortable': True, 'align': 'left'},
            {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True, 'align': 'left'},
            {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type', 'sortable': True, 'align': 'left'},
            {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'sortable': True, 'align': 'right'},
            {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True, 'align': 'right'},
            {'name': 'trigger_price', 'label': 'Trig. Price', 'field': 'trigger_price', 'sortable': True, 'align': 'right'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
            {'name': 'order_timestamp', 'label': 'Timestamp', 'field': 'order_timestamp', 'sortable': True, 'align': 'left'},
            {'name': 'product_type', 'label': 'Product', 'field': 'product_type', 'sortable': True, 'align': 'left'},
            {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'}
        ]

        # Initialize the table with full width utilization
        placed_orders_table = ui.table(
            columns=columns,
            rows=[],
            row_key='order_id'
        ).classes('w-full')

        # Add modify and cancel slots
        placed_orders_table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn v-if="props.row.status && props.row.status !== 'COMPLETE' && props.row.status !== 'CANCELLED' && props.row.status !== 'REJECTED'"
                       dense flat round color="primary" icon="edit" size="sm"
                       @click="() => $parent.$emit('modify_order', props.row)">
                    <q-tooltip>Modify Order</q-tooltip>
                </q-btn>
                <q-btn v-if="props.row.status && props.row.status !== 'COMPLETE' && props.row.status !== 'CANCELLED' && props.row.status !== 'REJECTED'"
                       dense flat round color="negative" icon="cancel" size="sm"
                       @click="() => $parent.$emit('cancel_order', props.row.order_id)">
                    <q-tooltip>Cancel Order</q-tooltip>
                </q-btn>
            </q-td>
        ''')

        # Define modify dialog container at module level to ensure proper context
        modify_dialog_container = ui.column().classes('w-full')

        def create_modify_dialog(order):
            with modify_dialog_container:
                modify_dialog_container.clear()
                with ui.dialog() as dialog, ui.card().classes("w-96"):
                    ui.label(f"Modify Order {order['order_id']}").classes("text-h6 mb-3")

                    with ui.column().classes('w-full gap-2'):
                        quantity_input = ui.number(label="Quantity", value=order.get('quantity', 0), min=1, format='%d').classes("w-full")
                        price_input = ui.number(label="Price", value=float(order.get('price', 0)) if order.get('price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                        trigger_price_input = ui.number(label="Trigger Price", value=float(order.get('trigger_price', 0)) if order.get('trigger_price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                        validity_select = ui.select(options=['DAY', 'IOC'], value=order.get('validity', 'DAY'), label="Validity").classes("w-full")

                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white px-4 py-1")

                        async def confirm_modify():
                            with modify_dialog_container:
                                modify_data = {
                                    "quantity": int(quantity_input.value),
                                    "price": float(price_input.value),
                                    "trigger_price": float(trigger_price_input.value),
                                    "validity": validity_select.value
                                }
                                response = await fetch_api(f"/orders/{order['order_id']}/modify", method="PUT", data=modify_data)
                                if response and response.get("status") == "success":
                                    ui.notify(f"Order {order['order_id']} modified successfully.", type="positive")
                                    await refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker)
                                else:
                                    ui.notify(f"Failed to modify order {order['order_id']}.", type="negative")
                                dialog.close()

                        ui.button("Modify", on_click=lambda: asyncio.create_task(confirm_modify())).classes("bg-blue-500 text-white px-4 py-1")
                    dialog.open()

        async def handle_modify_order(order):
            create_modify_dialog(order)

        async def handle_cancel_order(order_id):
            with placed_message_container:
                try:
                    response = await fetch_api(f"/orders/{order_id}", method="DELETE")
                    if response and response.get("status") == "success":
                        ui.notify(f"Order {order_id} cancelled successfully.", type="positive")
                    else:
                        ui.notify(f"Failed to cancel order {order_id}.", type="negative")
                    await refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker)
                except Exception as e:
                    ui.notify(f"Error cancelling order: {str(e)}", type="negative")
                    logger.error(f"Exception in handle_cancel_order: {str(e)}")

        placed_orders_table.on('modify_order', lambda e: asyncio.create_task(handle_modify_order(e.args)))
        placed_orders_table.on('cancel_order', lambda e: asyncio.create_task(handle_cancel_order(e.args)))

        # Refresh button with reduced margin
        ui.button('Refresh Orders', on_click=lambda: asyncio.create_task(refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker)), icon="refresh").classes('bg-blue-500 text-white px-4 py-2 mt-2')

        # Initial fetch
        await refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker)
        # ui.timer(20, lambda: asyncio.create_task(refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker)))

async def render_scheduled_orders(fetch_api, user_storage, broker):
    global scheduled_orders_table
    with ui.card().classes("w-full p-3"):
        ui.label("Scheduled Orders").classes("text-h6 mb-2")

        # Create message container within the card context
        scheduled_message_container = ui.column().classes('w-full mt-2')

        columns = [
            {'name': 'scheduled_order_id', 'label': 'Order ID', 'field': 'scheduled_order_id', 'sortable': True, 'align': 'left'},
            {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True, 'align': 'left'},
            {'name': 'instrument_token', 'label': 'Instrument', 'field': 'instrument_token', 'sortable': True, 'align': 'left'},
            {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type', 'sortable': True, 'align': 'left'},
            {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'sortable': True, 'align': 'right'},
            {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True, 'align': 'right'},
            {'name': 'trigger_price', 'label': 'Trig. Price', 'field': 'trigger_price', 'sortable': True, 'align': 'right'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
            {'name': 'schedule_datetime', 'label': 'Scheduled Time', 'field': 'schedule_datetime', 'sortable': True, 'align': 'left'},
            {'name': 'product_type', 'label': 'Product', 'field': 'product_type', 'sortable': True, 'align': 'left'},
            {'name': 'order_type', 'label': 'Order Type', 'field': 'order_type', 'sortable': True, 'align': 'left'},
            {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'}
        ]

        scheduled_orders_table = ui.table(
            columns=columns,
            rows=[],
            row_key='scheduled_order_id'
        ).classes('w-full')

        scheduled_orders_table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn v-if="props.row.status && props.row.status === 'PENDING'"
                       dense flat round color="primary" icon="edit" size="sm"
                       @click="() => $parent.$emit('modify_order', props.row)">
                    <q-tooltip>Modify Order</q-tooltip>
                </q-btn>
                <q-btn v-if="props.row.status && props.row.status === 'PENDING'"
                       dense flat round color="negative" icon="cancel" size="sm"
                       @click="() => $parent.$emit('cancel_order', props.row.scheduled_order_id)">
                    <q-tooltip>Cancel Order</q-tooltip>
                </q-btn>
            </q-td>
        ''')

        # Define modify dialog container for scheduled orders to ensure proper context
        scheduled_modify_dialog_container = ui.column().classes('w-full')

        def create_scheduled_modify_dialog(order):
            with scheduled_modify_dialog_container:
                scheduled_modify_dialog_container.clear()
                with ui.dialog() as dialog, ui.card().classes("w-96"):
                    ui.label(f"Modify Scheduled Order {order['scheduled_order_id']}").classes("text-h6 mb-3")

                    with ui.column().classes('w-full gap-2'):
                        quantity_input = ui.number(label="Quantity", value=order.get('quantity', 0), min=1, format='%d').classes("w-full")
                        price_input = ui.number(label="Price", value=float(order.get('price', 0)) if order.get('price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                        trigger_price_input = ui.number(label="Trigger Price", value=float(order.get('trigger_price', 0)) if order.get('trigger_price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                        schedule_date = ui.date(
                            value=(datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
                        ).classes('w-full').props('dense')
                        schedule_time = ui.time(
                            value='09:15'
                        ).classes('w-full').props('dense')

                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white px-4 py-1")

                        async def confirm_modify():
                            # Use explicit container context for UI operations within async task
                            with scheduled_modify_dialog_container:
                                try:
                                    schedule_datetime = datetime.combine(
                                        datetime.strptime(schedule_date.value, '%Y-%m-%d').date(),
                                        datetime.strptime(schedule_time.value, '%H:%M').time()
                                    )
                                    if schedule_datetime <= datetime.now():
                                        safe_notify("Schedule time must be in the future.", "negative")
                                        return
                                except Exception as e:
                                    safe_notify(f"Invalid schedule time: {str(e)}", "negative")
                                    return

                                modify_data = {
                                    "quantity": int(quantity_input.value),
                                    "price": float(price_input.value),
                                    "trigger_price": float(trigger_price_input.value),
                                    "schedule_datetime": schedule_datetime.isoformat(),
                                    "broker": broker,
                                    "product_type": order['product_type'],
                                    "instrument_token": order['instrument_token'],
                                    "trading_symbol": order['trading_symbol'],
                                    "transaction_type": order['transaction_type'],
                                    "order_type": order['order_type']
                                }
                                logger.debug(f"Modify Scheduled Order {order['scheduled_order_id']} - {order}")
                                response = await fetch_api(f"/scheduled-orders/{order['scheduled_order_id']}", method="PUT", data=modify_data)

                                # Enhanced logging to debug the response
                                logger.info(f"API Response for scheduled order modification: {response}")

                                # More robust success checking
                                if response:
                                    # Check for various success indicators
                                    is_success = (
                                        response.get("status") == "success" or
                                        response.get("message") == "success" or
                                        "error" not in response or
                                        response.get("error") is None or
                                        (isinstance(response, dict) and "scheduled_order_id" in response) or
                                        response.get("updated") is True
                                    )

                                    if is_success:
                                        safe_notify(f"Scheduled order {order['scheduled_order_id']} modified successfully.", "positive")
                                        await refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container, fetch_api, broker)
                                    else:
                                        error_msg = response.get("message", response.get("detail", "Unknown error"))
                                        safe_notify(f"Failed to modify scheduled order: {error_msg}", "negative")
                                        logger.error(f"API returned error: {response}")
                                else:
                                    safe_notify(f"Failed to modify scheduled order {order['scheduled_order_id']}.", "negative")
                                    logger.error("API returned no response")

                                dialog.close()

                        ui.button("Modify", on_click=lambda: create_safe_task(confirm_modify())).classes("bg-blue-500 text-white px-4 py-1")
                    dialog.open()

        async def handle_modify_order(order):
            create_scheduled_modify_dialog(order)

        async def handle_cancel_order(order_id):
            with scheduled_message_container:
                with ui.dialog() as dialog, ui.card().classes("w-96"):
                    ui.label(f"Are you sure you want to cancel order {order_id}?").classes("text-h6 mb-3")
                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("No", on_click=dialog.close).classes("bg-gray-500 text-white px-4 py-1")

                        async def confirm_cancel():
                            with scheduled_message_container:
                                try:
                                    response = await fetch_api(f"/scheduled-orders/{order_id}?broker={broker}",
                                                               method="DELETE")
                                    if response and response.get("status") == "success":
                                        ui.notify(f"Order {order_id} cancelled successfully.", type="positive")
                                    else:
                                        ui.notify(f"Failed to cancel order {order_id}.", type="negative")
                                    await refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container,
                                                                   fetch_api, broker)
                                except Exception as e:
                                    ui.notify(f"Error cancelling order: {str(e)}", type="negative")
                                    logger.error(f"Exception in handle_cancel_order: {str(e)}")
                                dialog.close()

                        ui.button("Yes, Cancel", on_click=lambda: asyncio.create_task(confirm_cancel())).classes(
                            "bg-red-500 text-white px-4 py-1")
                    dialog.open()

        scheduled_orders_table.on('modify_order', lambda e: asyncio.create_task(handle_modify_order(e.args)))
        scheduled_orders_table.on('cancel_order', lambda e: asyncio.create_task(handle_cancel_order(e.args)))

        ui.button('Refresh Orders', on_click=lambda: asyncio.create_task(refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container, fetch_api, broker)), icon="refresh").classes('bg-blue-500 text-white px-4 py-2 mt-2')

        await refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container, fetch_api, broker)

async def render_gtt_orders(fetch_api, user_storage, broker):
    global gtt_orders_table
    with ui.card().classes("w-full p-3"):
        ui.label("GTT Orders").classes("text-h6 mb-2")

        # Create message container within the card context
        gtt_message_container = ui.column().classes('w-full mt-2')

        columns = [
            {'name': 'gtt_order_id', 'label': 'Order ID', 'field': 'gtt_order_id', 'sortable': True, 'align': 'left'},
            {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True, 'align': 'left'},
            {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type', 'sortable': True, 'align': 'left'},
            {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'sortable': True, 'align': 'right'},
            {'name': 'trigger_type', 'label': 'Trigger Type', 'field': 'trigger_type', 'sortable': True, 'align': 'left'},
            {'name': 'trigger_price', 'label': 'Trig. Price', 'field': 'trigger_price', 'sortable': True, 'align': 'right'},
            {'name': 'limit_price', 'label': 'Limit Price', 'field': 'limit_price', 'sortable': True, 'align': 'right'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
            {'name': 'created_at', 'label': 'Created At', 'field': 'created_at', 'sortable': True, 'align': 'left'},
            {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'}
        ]

        gtt_orders_table = ui.table(
            columns=columns,
            rows=[],
            row_key='gtt_order_id'
        ).classes('w-full')

        gtt_orders_table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn v-if="props.row.status && props.row.status === 'ACTIVE'"
                       dense flat round color="primary" icon="edit" size="sm"
                       @click="() => $parent.$emit('modify_order', props.row)">
                    <q-tooltip>Modify GTT Order</q-tooltip>
                </q-btn>
                <q-btn v-if="props.row.status && props.row.status === 'ACTIVE'"
                       dense flat round color="negative" icon="cancel" size="sm"
                       @click="() => $parent.$emit('cancel_order', props.row.gtt_order_id)">
                    <q-tooltip>Cancel GTT Order</q-tooltip>
                </q-btn>
            </q-td>
        ''')

        async def handle_modify_order(order):
            with gtt_message_container:
                with ui.dialog() as dialog, ui.card().classes("w-96"):
                    ui.label(f"Modify GTT Order {order['gtt_order_id']}").classes("text-h6 mb-3")
                    with ui.column().classes('w-full gap-2'):
                        quantity_input = ui.number(label="Quantity", value=order.get('quantity', 0), min=1,
                                                   format='%d').classes("w-full")
                        trigger_type_select = ui.select(options=['single', 'two-leg'],
                                                        value=order.get('trigger_type', 'single').lower(),
                                                        label="Trigger Type").classes("w-full")
                        trigger_price_input = ui.number(label="Trigger Price",
                                                        value=float(order.get('trigger_price', 0)) if order.get(
                                                            'trigger_price') != 'N/A' else 0, min=0, step=0.05,
                                                        format='%.2f').classes("w-full")
                        limit_price_input = ui.number(label="Limit Price",
                                                      value=float(order.get('limit_price', 0)) if order.get(
                                                          'limit_price') != 'N/A' else 0, min=0, step=0.05,
                                                      format='%.2f').classes("w-full")
                        second_trigger_price_input = ui.number(label="Second Trigger Price", value=float(
                            order.get('second_trigger_price', 0)) if order.get('second_trigger_price') else 0, min=0,
                                                               step=0.05, format='%.2f').classes("w-full")
                        second_limit_price_input = ui.number(label="Second Limit Price", value=float(
                            order.get('second_limit_price', 0)) if order.get('second_limit_price') else 0, min=0,
                                                             step=0.05, format='%.2f').classes("w-full")

                        def update_oco_fields():
                            is_oco = trigger_type_select.value == 'two-leg'
                            second_trigger_price_input.visible = is_oco
                            second_limit_price_input.visible = is_oco

                        trigger_type_select.on_value_change(update_oco_fields)
                        update_oco_fields()

                    with ui.row().classes("w-full justify-end gap-2 mt-4"):
                        ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white px-4 py-1")

                        with gtt_message_container:
                            async def confirm_modify():
                                modify_data = {
                                    "quantity": int(quantity_input.value),
                                    "trigger_type": trigger_type_select.value,
                                    "trigger_price": float(trigger_price_input.value),
                                    "limit_price": float(limit_price_input.value),
                                    "second_trigger_price": float(
                                        second_trigger_price_input.value) if trigger_type_select.value == 'two-leg' else None,
                                    "second_limit_price": float(
                                        second_limit_price_input.value) if trigger_type_select.value == 'two-leg' else None,
                                    "last_price": float(order.get('last_price', 0)),
                                    "instrument_token": order.get("instrument_token"),
                                    "trading_symbol": order.get("trading_symbol"),
                                    "transaction_type": order.get("transaction_type"),
                                    "broker": broker
                                }
                                response = await fetch_api(f"/gtt-orders/{broker}/{order['gtt_order_id']}", method="PUT",
                                                           data=modify_data)
                                if response and response.get("status") == "success":
                                    ui.notify(f"GTT order {order['gtt_order_id']} modified successfully.", type="positive")
                                    await refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker)
                                else:
                                    ui.notify(f"Failed to modify GTT order {order['gtt_order_id']}.", type="negative")
                                dialog.close()

                        ui.button("Modify", on_click=lambda: asyncio.create_task(confirm_modify())).classes(
                            "bg-blue-500 text-white px-4 py-1")
                    dialog.open()

        async def handle_cancel_order(order_id):
            with gtt_message_container:
                try:
                    response = await fetch_api(f"/gtt-orders/{broker}/{order_id}", method="DELETE")
                    if response and response.get("status") == "success":
                        ui.notify(f"GTT order {order_id} cancelled successfully.", type="positive")
                    else:
                        ui.notify(f"Failed to cancel GTT order {order_id}.", type="negative")
                    await refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker)
                except Exception as e:
                    ui.notify(f"Error cancelling GTT order: {str(e)}", type="negative")
                    logger.error(f"Exception in handle_cancel_order: {str(e)}")

        gtt_orders_table.on('modify_order', lambda e: asyncio.create_task(handle_modify_order(e.args)))
        gtt_orders_table.on('cancel_order', lambda e: asyncio.create_task(handle_cancel_order(e.args)))

        ui.button('Refresh Orders', on_click=lambda: asyncio.create_task(refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker)), icon="refresh").classes('bg-blue-500 text-white px-4 py-2 mt-2')

        await refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker)

async def render_auto_orders(fetch_api, user_storage, broker):
    global auto_orders_table
    with ui.card().classes("w-full p-3"):
        ui.label("Auto Orders").classes("text-h6 mb-2")

        # Create message container within the card context
        auto_message_container = ui.column().classes('w-full mt-2')

        columns = [
            {'name': 'auto_order_id', 'label': 'Order ID', 'field': 'auto_order_id', 'sortable': True, 'align': 'left'},
            {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True, 'align': 'left'},
            {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type', 'sortable': True, 'align': 'left'},
            {'name': 'risk_per_trade', 'label': 'Risk (%)', 'field': 'risk_per_trade', 'sortable': True, 'align': 'right'},
            {'name': 'stop_loss_type', 'label': 'Stop Loss Type', 'field': 'stop_loss_type', 'sortable': True, 'align': 'left'},
            {'name': 'stop_loss_value', 'label': 'Stop Loss', 'field': 'stop_loss_value', 'sortable': True, 'align': 'right'},
            {'name': 'target_value', 'label': 'Target', 'field': 'target_value', 'sortable': True, 'align': 'right'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
            {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'}
        ]

        auto_orders_table = ui.table(
            columns=columns,
            rows=[],
            row_key='auto_order_id'
        ).classes('w-full')

        auto_orders_table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn v-if="props.row.status && props.row.status !== 'CANCELLED'"
                       dense flat round color="primary" icon="edit" size="sm"
                       @click="() => $parent.$emit('modify_order', props.row)">
                    <q-tooltip>Modify Auto Order</q-tooltip>
                </q-btn>
                <q-btn v-if="props.row.status && props.row.status !== 'CANCELLED'"
                       dense flat round color="negative" icon="cancel" size="sm"
                       @click="() => $parent.$emit('cancel_order', props.row.auto_order_id)">
                    <q-tooltip>Cancel Auto Order</q-tooltip>
                </q-btn>
            </q-td>
        ''')

        async def handle_modify_order(order):
            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label(f"Modify Auto Order {order['auto_order_id']}").classes("text-h6 mb-3")

                with ui.column().classes('w-full gap-2'):
                    risk_per_trade_input = ui.number(label="Risk per Trade (%)", value=float(order.get('risk_per_trade', 0)), min=0.1, max=10.0, step=0.1, format='%.1f').classes("w-full")
                    stop_loss_type_select = ui.select(options=['Fixed Amount', 'Percentage of Entry', 'ATR Based'], value=order.get('stop_loss_type', 'Fixed Amount'), label="Stop Loss Type").classes("w-full")
                    stop_loss_value_input = ui.number(label="Stop Loss Value", value=float(order.get('stop_loss_value', 0)) if order.get('stop_loss_value') != 'N/A' else 0, min=0.1, step=0.1, format='%.1f').classes("w-full")
                    target_value_input = ui.number(label="Target Value", value=float(order.get('target_value', 0)) if order.get('target_value') != 'N/A' else 0, min=0.1, step=0.1, format='%.1f').classes("w-full")
                    atr_period_input = ui.number(label="ATR Period", value=order.get('atr_period', 14), min=5, max=50, format='%d').classes("w-full")

                    def update_stop_loss_fields():
                        is_atr = stop_loss_type_select.value == 'ATR Based'
                        atr_period_input.visible = is_atr
                        stop_loss_value_input.label = 'Stop Loss (ATR Multiple)' if is_atr else 'Stop Loss Value'
                        target_value_input.label = 'Target (ATR Multiple)' if is_atr else 'Target Value'

                    stop_loss_type_select.on_value_change(update_stop_loss_fields)
                    update_stop_loss_fields()

                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white px-4 py-1")

                    async def confirm_modify():
                        modify_data = {
                            "risk_per_trade": float(risk_per_trade_input.value),
                            "stop_loss_type": stop_loss_type_select.value,
                            "stop_loss_value": float(stop_loss_value_input.value),
                            "target_value": float(target_value_input.value),
                            "atr_period": int(atr_period_input.value) if stop_loss_type_select.value == 'ATR Based' else None
                        }
                        response = await fetch_api(f"/auto-orders/{order['auto_order_id']}", method="PUT", data=modify_data)
                        if response and response.get("status") == "success":
                            ui.notify(f"Auto order {order['auto_order_id']} modified successfully.", type="positive")
                            await refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker)
                        else:
                            ui.notify(f"Failed to modify auto order {order['auto_order_id']}.", type="negative")
                        dialog.close()

                    ui.button("Modify", on_click=lambda: asyncio.create_task(confirm_modify())).classes("bg-blue-500 text-white px-4 py-1")
                dialog.open()

        async def handle_cancel_order(order_id):
            try:
                response = await fetch_api(f"/auto-orders/{order_id}", method="DELETE")
                if response and response.get("status") == "success":
                    ui.notify(f"Auto order {order_id} cancelled successfully.", type="positive")
                else:
                    ui.notify(f"Failed to cancel auto order {order_id}.", type="negative")
                await refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker)
            except Exception as e:
                ui.notify(f"Error cancelling auto order: {str(e)}", type="negative")
                logger.error(f"Exception in handle_cancel_order: {str(e)}")

        auto_orders_table.on('modify_order', lambda e: asyncio.create_task(handle_modify_order(e.args)))
        auto_orders_table.on('cancel_order', lambda e: asyncio.create_task(handle_cancel_order(e.args)))

        ui.button('Refresh Orders', on_click=lambda: asyncio.create_task(refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker)), icon="refresh").classes('bg-blue-500 text-white px-4 py-2 mt-2')

        await refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker)
        # ui.timer(300, lambda: refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker))