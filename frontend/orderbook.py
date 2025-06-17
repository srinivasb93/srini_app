"""
Orderbook Module for NiceGUI Algo Trading Application
Displays tables for Placed, Scheduled, GTT, and Auto Orders with modify and cancel options
"""

from nicegui import ui
import logging
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

# Global containers
cancel_loading_container = ui.column().classes('w-full mt-2')
dialog_container = ui.column().classes('w-full')  # Container for dialogs to ensure slot context

# Define tables and message containers at module scope
placed_orders_table = None
scheduled_orders_table = None
gtt_orders_table = None
auto_orders_table = None

placed_message_container = ui.column().classes('w-full mt-4')
scheduled_message_container = ui.column().classes('w-full mt-4')
gtt_message_container = ui.column().classes('w-full mt-4')
auto_message_container = ui.column().classes('w-full mt-4')

async def refresh_placed_orders(table, message_container, fetch_api, broker):
    with message_container:
        message_container.clear()
        with ui.element().classes("relative"):
            ui.spinner(size="lg")
            ui.label("Loading placed orders...").classes("text-subtitle1 text-gray-400 ml-2")

    try:
        orders_data = await fetch_api(f"/orders/{broker}")
        logger.debug(f"Fetched placed orders: {orders_data}")
        with message_container:
            message_container.clear()
            if orders_data and isinstance(orders_data, list):
                if not orders_data:
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
                            'actions': ''  # Placeholder for action buttons
                        }
                        formatted_orders.append(formatted_order)
                    except Exception as e:
                        logger.error(f"Error formatting order {order.get('order_id', 'unknown')}: {str(e)}")
                        continue

                table.rows = formatted_orders
                table.update()
            else:
                ui.label("Failed to load placed orders.").classes("text-red-500 text-subtitle1")
                ui.notify("Error fetching placed orders.", type="negative")
                logger.error(f"Invalid placed orders response: {orders_data}")
                table.rows = []
                table.update()
    except Exception as e:
        with message_container:
            message_container.clear()
            ui.label(f"Error: {str(e)}").classes("text-red-500 text-subtitle1")
            ui.notify(f"Error fetching placed orders: {str(e)}", type="negative")
        logger.error(f"Exception in refresh_placed_orders: {str(e)}")
        table.rows = []
        table.update()

async def refresh_scheduled_orders(table, message_container, fetch_api, broker):
    with message_container:
        message_container.clear()
        with ui.element().classes("relative"):
            ui.spinner(size="lg")
            ui.label("Loading scheduled orders...").classes("text-subtitle1 text-gray-400 ml-2")

    try:
        orders_data = await fetch_api(f"/scheduled-orders/{broker}")
        logger.debug(f"Fetched scheduled orders: {orders_data}")
        with message_container:
            message_container.clear()
            if orders_data and isinstance(orders_data, list):
                if not orders_data:
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
                            'transaction_type': order.get('transaction_type', 'N/A'),
                            'quantity': order.get('quantity', 0),
                            'price': f"{price:.2f}" if price != 0 else 'N/A',
                            'trigger_price': f"{trigger_price:.2f}" if trigger_price != 0 else 'N/A',
                            'status': order.get('status', 'N/A').upper(),
                            'schedule_datetime': schedule_datetime,
                            'product_type': order.get('product_type', 'N/A'),
                            'actions': ''
                        }
                        formatted_orders.append(formatted_order)
                    except Exception as e:
                        logger.error(f"Error formatting scheduled order {order.get('scheduled_order_id', 'unknown')}: {str(e)}")
                        continue

                table.rows = formatted_orders
                table.update()
            else:
                ui.label("Failed to load scheduled orders.").classes("text-red-500 text-subtitle1")
                ui.notify("Error fetching scheduled orders.", type="negative")
                logger.error(f"Invalid scheduled orders response: {orders_data}")
                table.rows = []
                table.update()
    except Exception as e:
        with message_container:
            message_container.clear()
            ui.label(f"Error: {str(e)}").classes("text-red-500 text-subtitle1")
            ui.notify(f"Error fetching scheduled orders: {str(e)}", type="negative")
        logger.error(f"Exception in refresh_scheduled_orders: {str(e)}")
        table.rows = []
        table.update()

async def refresh_gtt_orders(table, message_container, fetch_api, broker):
    with message_container:
        message_container.clear()
        with ui.element().classes("relative"):
            ui.spinner(size="lg")
            ui.label("Loading GTT orders...").classes("text-subtitle1 text-gray-400 ml-2")

    try:
        orders_data = await fetch_api(f"/gtt-orders/{broker}")
        logger.debug(f"Fetched GTT orders: {orders_data}")
        with message_container:
            message_container.clear()
            if orders_data and isinstance(orders_data, list):
                if not orders_data:
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
                ui.label("Failed to load GTT orders.").classes("text-red-500 text-subtitle1")
                ui.notify("Error fetching GTT orders.", type="negative")
                logger.error(f"Invalid GTT orders response: {orders_data}")
                table.rows = []
                table.update()
    except Exception as e:
        with message_container:
            message_container.clear()
            ui.label(f"Error: {str(e)}").classes("text-red-500 text-subtitle1")
            ui.notify(f"Error fetching GTT orders: {str(e)}", type="negative")
        logger.error(f"Exception in refresh_gtt_orders: {str(e)}")
        table.rows = []
        table.update()

async def refresh_auto_orders(table, message_container, fetch_api, broker):
    with message_container:
        message_container.clear()
        with ui.element().classes("relative"):
            ui.spinner(size="lg")
            ui.label("Loading auto orders...").classes("text-subtitle1 text-gray-400 ml-2")

    try:
        orders_data = await fetch_api(f"/auto-orders/{broker}")
        logger.debug(f"Fetched auto orders: {orders_data}")
        with message_container:
            message_container.clear()
            if orders_data and isinstance(orders_data, list):
                if not orders_data:
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
                ui.label("Failed to load auto orders.").classes("text-red-500 text-subtitle1")
                ui.notify("Error fetching auto orders.", type="negative")
                logger.error(f"Invalid auto orders response: {orders_data}")
                table.rows = []
                table.update()
    except Exception as e:
        with message_container:
            message_container.clear()
            ui.label(f"Error: {str(e)}").classes("text-red-500 text-subtitle1")
            ui.notify(f"Error fetching auto orders: {str(e)}", type="negative")
        logger.error(f"Exception in refresh_auto_orders: {str(e)}")
        table.rows = []
        table.update()

async def render_order_book_page(fetch_api, user_storage, broker):
    ui.label(f"Order Book - {broker}").classes("text-h4")

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
    with ui.row().classes('w-full mt-4'):
        async def cancel_all_open_orders():
            with cancel_loading_container:
                cancel_loading_container.clear()
                with ui.element().classes("relative"):
                    ui.spinner(size="lg")
                    ui.label("Cancelling all open orders...").classes("text-subtitle1 text-gray-400 ml-2")
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
                await refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker)
                await refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container, fetch_api, broker)
                await refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker)
                await refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker)
            except Exception as e:
                ui.notify(f"Error cancelling orders: {str(e)}", type="negative")
                logger.error(f"Error in cancel_all_open_orders: {e}")
            cancel_loading_container.clear()

        ui.button('Cancel All Open Orders', on_click=lambda: asyncio.create_task(cancel_all_open_orders()), icon="delete_forever").classes('button-danger')

async def render_placed_orders(fetch_api, user_storage, broker):
    global placed_orders_table
    with ui.card().classes("card"):
        ui.label("Placed Orders").classes("text-h6 mb-4")

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

        # Initialize the table
        placed_orders_table = ui.table(
            columns=columns,
            rows=[],
            row_key='order_id'
        ).classes('table')

        # Add modify and cancel slots
        placed_orders_table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn v-if="props.row.status && props.row.status !== 'COMPLETE' && props.row.status !== 'CANCELLED' && props.row.status !== 'REJECTED'"
                       dense flat round color="primary" icon="edit" 
                       @click="() => $parent.$emit('modify_order', props.row)">
                    <q-tooltip>Modify Order</q-tooltip>
                </q-btn>
                <q-btn v-if="props.row.status && props.row.status !== 'COMPLETE' && props.row.status !== 'CANCELLED' && props.row.status !== 'REJECTED'"
                       dense flat round color="negative" icon="cancel" 
                       @click="() => $parent.$emit('cancel_order', props.row.order_id)">
                    <q-tooltip>Cancel Order</q-tooltip>
                </q-btn>
            </q-td>
        ''')

        async def handle_modify_order(order):
            with dialog_container:  # Use explicit container for dialog
                with ui.dialog() as dialog, ui.card().classes("card"):
                    ui.label(f"Modify Order {order['order_id']}").classes("text-h6")
                    quantity_input = ui.number(label="Quantity", value=order.get('quantity', 0), min=1, format='%d').classes("w-full")
                    price_input = ui.number(label="Price", value=float(order.get('price', 0)) if order.get('price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                    trigger_price_input = ui.number(label="Trigger Price", value=float(order.get('trigger_price', 0)) if order.get('trigger_price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                    validity_select = ui.select(options=['DAY', 'IOC'], value=order.get('validity', 'DAY'), label="Validity").classes("w-full")
                    with ui.row().classes("space-x-4 mt-4"):
                        ui.button("Cancel", on_click=dialog.close).classes("button-outline")
                        async def confirm_modify():
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
                        ui.button("Modify", on_click=lambda: asyncio.create_task(confirm_modify())).classes("button-primary")
                    dialog.open()

        async def handle_cancel_order(order_id):
            with cancel_loading_container:
                cancel_loading_container.clear()
                with ui.element().classes("relative"):
                    ui.spinner(size="lg")
                    ui.label(f"Cancelling order {order_id}...").classes("text-subtitle1 text-gray-400 ml-2")
            response = await fetch_api(f"/orders/{order_id}", method="DELETE")
            if response and response.get("status") == "success":
                ui.notify(f"Order {order_id} cancelled successfully.", type="positive")
            else:
                ui.notify(f"Failed to cancel order {order_id}.", type="negative")
            await refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker)
            cancel_loading_container.clear()

        placed_orders_table.on('modify_order', lambda e: asyncio.create_task(handle_modify_order(e.args)))
        placed_orders_table.on('cancel_order', lambda e: asyncio.create_task(handle_cancel_order(e.args)))

        # Refresh button
        ui.button('Refresh Orders', on_click=lambda: asyncio.create_task(refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker)), icon="refresh").classes('button-outline mt-4')

        # Initial fetch
        await refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker)
        ui.timer(20, lambda: refresh_placed_orders(placed_orders_table, placed_message_container, fetch_api, broker))

async def render_scheduled_orders(fetch_api, user_storage, broker):
    global scheduled_orders_table
    with ui.card().classes("card"):
        ui.label("Scheduled Orders").classes("text-h6 mb-4")

        columns = [
            {'name': 'scheduled_order_id', 'label': 'Order ID', 'field': 'scheduled_order_id', 'sortable': True, 'align': 'left'},
            {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True, 'align': 'left'},
            {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type', 'sortable': True, 'align': 'left'},
            {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'sortable': True, 'align': 'right'},
            {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True, 'align': 'right'},
            {'name': 'trigger_price', 'label': 'Trig. Price', 'field': 'trigger_price', 'sortable': True, 'align': 'right'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
            {'name': 'schedule_datetime', 'label': 'Scheduled Time', 'field': 'schedule_datetime', 'sortable': True, 'align': 'left'},
            {'name': 'product_type', 'label': 'Product', 'field': 'product_type', 'sortable': True, 'align': 'left'},
            {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'}
        ]

        scheduled_orders_table = ui.table(
            columns=columns,
            rows=[],
            row_key='scheduled_order_id'
        ).classes('table')

        scheduled_orders_table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn v-if="props.row.status && props.row.status === 'PENDING'"
                       dense flat round color="primary" icon="edit" 
                       @click="() => $parent.$emit('modify_order', props.row)">
                    <q-tooltip>Modify Order</q-tooltip>
                </q-btn>
                <q-btn v-if="props.row.status && props.row.status === 'PENDING'"
                       dense flat round color="negative" icon="cancel" 
                       @click="() => $parent.$emit('cancel_order', props.row.scheduled_order_id)">
                    <q-tooltip>Cancel Order</q-tooltip>
                </q-btn>
            </q-td>
        ''')

        async def handle_modify_order(order):
            with dialog_container:  # Use explicit container for dialog
                with ui.dialog() as dialog, ui.card().classes("card"):
                    ui.label(f"Modify Scheduled Order {order['scheduled_order_id']}").classes("text-h6")
                    quantity_input = ui.number(label="Quantity", value=order.get('quantity', 0), min=1, format='%d').classes("w-full")
                    price_input = ui.number(label="Price", value=float(order.get('price', 0)) if order.get('price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                    trigger_price_input = ui.number(label="Trigger Price", value=float(order.get('trigger_price', 0)) if order.get('trigger_price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                    schedule_date = ui.date(
                        label="Schedule Date",
                        value=(datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
                    ).classes('w-full')
                    schedule_time = ui.time(
                        label="Schedule Time",
                        value='09:15'
                    ).classes('w-full')
                    with ui.row().classes("space-x-4 mt-4"):
                        ui.button("Cancel", on_click=dialog.close).classes("button-outline")
                        async def confirm_modify():
                            try:
                                schedule_datetime = datetime.combine(
                                    datetime.strptime(schedule_date.value, '%Y-%m-%d').date(),
                                    datetime.strptime(schedule_time.value, '%H:%M').time()
                                )
                                if schedule_datetime <= datetime.now():
                                    ui.notify("Schedule time must be in the future.", type="negative")
                                    return
                            except Exception as e:
                                ui.notify(f"Invalid schedule time: {str(e)}", type="negative")
                                return

                            modify_data = {
                                "quantity": int(quantity_input.value),
                                "price": float(price_input.value),
                                "trigger_price": float(trigger_price_input.value),
                                "schedule_datetime": schedule_datetime.isoformat()
                            }
                            response = await fetch_api(f"/scheduled-orders/{order['scheduled_order_id']}", method="PUT", data=modify_data)
                            if response and response.get("status") == "success":
                                ui.notify(f"Scheduled order {order['scheduled_order_id']} modified successfully.", type="positive")
                                await refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container, fetch_api, broker)
                            else:
                                ui.notify(f"Failed to modify scheduled order {order['scheduled_order_id']}.", type="negative")
                            dialog.close()
                        ui.button("Modify", on_click=lambda: asyncio.create_task(confirm_modify())).classes("button-primary")
                    dialog.open()

        async def handle_cancel_order(order_id):
            with cancel_loading_container:
                cancel_loading_container.clear()
                with ui.element().classes("relative"):
                    ui.spinner(size="lg")
                    ui.label(f"Cancelling scheduled order {order_id}...").classes("text-subtitle1 text-gray-400 ml-2")
            response = await fetch_api(f"/scheduled-orders/{order_id}", method="DELETE")
            if response and response.get("status") == "success":
                ui.notify(f"Scheduled order {order_id} cancelled successfully.", type="positive")
            else:
                ui.notify(f"Failed to cancel scheduled order {order_id}.", type="negative")
            await refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container, fetch_api, broker)
            cancel_loading_container.clear()

        scheduled_orders_table.on('modify_order', lambda e: asyncio.create_task(handle_modify_order(e.args)))
        scheduled_orders_table.on('cancel_order', lambda e: asyncio.create_task(handle_cancel_order(e.args)))

        ui.button('Refresh Orders', on_click=lambda: asyncio.create_task(refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container, fetch_api, broker)), icon="refresh").classes('button-outline mt-4')

        await refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container, fetch_api, broker)
        ui.timer(20, lambda: refresh_scheduled_orders(scheduled_orders_table, scheduled_message_container, fetch_api, broker))

async def render_gtt_orders(fetch_api, user_storage, broker):
    global gtt_orders_table
    with ui.card().classes("card"):
        ui.label("GTT Orders").classes("text-h6 mb-4")

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
        ).classes('table')

        gtt_orders_table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn v-if="props.row.status && props.row.status === 'ACTIVE'"
                       dense flat round color="primary" icon="edit" 
                       @click="() => $parent.$emit('modify_order', props.row)">
                    <q-tooltip>Modify GTT Order</q-tooltip>
                </q-btn>
                <q-btn v-if="props.row.status && props.row.status === 'ACTIVE'"
                       dense flat round color="negative" icon="cancel" 
                       @click="() => $parent.$emit('cancel_order', props.row.gtt_order_id)">
                    <q-tooltip>Cancel GTT Order</q-tooltip>
                </q-btn>
            </q-td>
        ''')

        async def handle_modify_order(order):
            with dialog_container:  # Use explicit container for dialog
                with ui.dialog() as dialog, ui.card().classes("card"):
                    ui.label(f"Modify GTT Order {order['gtt_order_id']}").classes("text-h6")
                    quantity_input = ui.number(label="Quantity", value=order.get('quantity', 0), min=1, format='%d').classes("w-full")
                    trigger_type_select = ui.select(options=['single', 'two-leg'], value=order.get('trigger_type', 'single').lower(), label="Trigger Type").classes("w-full")
                    trigger_price_input = ui.number(label="Trigger Price", value=float(order.get('trigger_price', 0)) if order.get('trigger_price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                    limit_price_input = ui.number(label="Limit Price", value=float(order.get('limit_price', 0)) if order.get('limit_price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                    second_trigger_price_input = ui.number(label="Second Trigger Price", value=float(order.get('second_trigger_price', 0)) if order.get('second_trigger_price') else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                    second_limit_price_input = ui.number(label="Second Limit Price", value=float(order.get('second_limit_price', 0)) if order.get('second_limit_price') else 0, min=0, step=0.05, format='%.2f').classes("w-full")

                    def update_oco_fields():
                        is_oco = trigger_type_select.value == 'two-leg'
                        second_trigger_price_input.visible = is_oco
                        second_limit_price_input.visible = is_oco

                    trigger_type_select.on_value_change(update_oco_fields)
                    update_oco_fields()

                    with ui.row().classes("space-x-4 mt-4"):
                        ui.button("Cancel", on_click=dialog.close).classes("button-outline")
                        async def confirm_modify():
                            modify_data = {
                                "quantity": int(quantity_input.value),
                                "trigger_type": trigger_type_select.value,
                                "trigger_price": float(trigger_price_input.value),
                                "limit_price": float(limit_price_input.value),
                                "second_trigger_price": float(second_trigger_price_input.value) if trigger_type_select.value == 'two-leg' else None,
                                "second_limit_price": float(second_limit_price_input.value) if trigger_type_select.value == 'two-leg' else None,
                                "last_price": float(order.get('last_price', 0))
                            }
                            response = await fetch_api(f"/gtt-orders/{broker}/{order['gtt_order_id']}", method="PUT", data=modify_data)
                            if response and response.get("status") == "success":
                                ui.notify(f"GTT order {order['gtt_order_id']} modified successfully.", type="positive")
                                await refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker)
                            else:
                                ui.notify(f"Failed to modify GTT order {order['gtt_order_id']}.", type="negative")
                            dialog.close()
                        ui.button("Modify", on_click=lambda: asyncio.create_task(confirm_modify())).classes("button-primary")
                    dialog.open()

        async def handle_cancel_order(order_id):
            with cancel_loading_container:
                cancel_loading_container.clear()
                with ui.element().classes("relative"):
                    ui.spinner(size="lg")
                    ui.label(f"Cancelling GTT order {order_id}...").classes("text-subtitle1 text-gray-400 ml-2")
            response = await fetch_api(f"/gtt-orders/{broker}/{order_id}", method="DELETE")
            if response and response.get("status") == "success":
                ui.notify(f"GTT order {order_id} cancelled successfully.", type="positive")
            else:
                ui.notify(f"Failed to cancel GTT order {order_id}.", type="negative")
            await refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker)
            cancel_loading_container.clear()

        gtt_orders_table.on('modify_order', lambda e: asyncio.create_task(handle_modify_order(e.args)))
        gtt_orders_table.on('cancel_order', lambda e: asyncio.create_task(handle_cancel_order(e.args)))

        ui.button('Refresh Orders', on_click=lambda: asyncio.create_task(refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker)), icon="refresh").classes('button-outline mt-4')

        await refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker)
        ui.timer(20, lambda: refresh_gtt_orders(gtt_orders_table, gtt_message_container, fetch_api, broker))

async def render_auto_orders(fetch_api, user_storage, broker):
    global auto_orders_table
    with ui.card().classes("card"):
        ui.label("Auto Orders").classes("text-h6 mb-4")

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
        ).classes('table')

        auto_orders_table.add_slot('body-cell-actions', '''
            <q-td :props="props">
                <q-btn v-if="props.row.status && props.row.status !== 'CANCELLED'"
                       dense flat round color="primary" icon="edit" 
                       @click="() => $parent.$emit('modify_order', props.row)">
                    <q-tooltip>Modify Auto Order</q-tooltip>
                </q-btn>
                <q-btn v-if="props.row.status && props.row.status !== 'CANCELLED'"
                       dense flat round color="negative" icon="cancel" 
                       @click="() => $parent.$emit('cancel_order', props.row.auto_order_id)">
                    <q-tooltip>Cancel Auto Order</q-tooltip>
                </q-btn>
            </q-td>
        ''')

        async def handle_modify_order(order):
            with dialog_container:  # Use explicit container for dialog
                with ui.dialog() as dialog, ui.card().classes("card"):
                    ui.label(f"Modify Auto Order {order['auto_order_id']}").classes("text-h6")
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

                    with ui.row().classes("space-x-4 mt-4"):
                        ui.button("Cancel", on_click=dialog.close).classes("button-outline")
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
                        ui.button("Modify", on_click=lambda: asyncio.create_task(confirm_modify())).classes("button-primary")
                    dialog.open()

        async def handle_cancel_order(order_id):
            with cancel_loading_container:
                cancel_loading_container.clear()
                with ui.element().classes("relative"):
                    ui.spinner(size="lg")
                    ui.label(f"Cancelling auto order {order_id}...").classes("text-subtitle1 text-gray-400 ml-2")
            response = await fetch_api(f"/auto-orders/{order_id}", method="DELETE")
            if response and response.get("status") == "success":
                ui.notify(f"Auto order {order_id} cancelled successfully.", type="positive")
            else:
                ui.notify(f"Failed to cancel auto order {order_id}.", type="negative")
            await refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker)
            cancel_loading_container.clear()

        auto_orders_table.on('modify_order', lambda e: asyncio.create_task(handle_modify_order(e.args)))
        auto_orders_table.on('cancel_order', lambda e: asyncio.create_task(handle_cancel_order(e.args)))

        ui.button('Refresh Orders', on_click=lambda: asyncio.create_task(refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker)), icon="refresh").classes('button-outline mt-4')

        await refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker)
        ui.timer(20, lambda: refresh_auto_orders(auto_orders_table, auto_message_container, fetch_api, broker))