"""
Enhanced Orderbook Module for NiceGUI Algo Trading Application
Displays tables for Placed, Scheduled, GTT, and Auto Orders with:
- Status-based filtering
- Pagination (10 orders per page) 
- Enhanced table headers with proper theming
- Refresh icon instead of button
- Better visual alignment and styling
"""

import logging
from nicegui import ui
import asyncio
from datetime import datetime
import math
from ui_context_manager import safe_notify, create_safe_task
from ws_events import register_order_ws_callback

logger = logging.getLogger(__name__)

# Define tables and pagination data at module scope
placed_orders_table = None
scheduled_orders_table = None
gtt_orders_table = None
auto_orders_table = None

# Pagination and filter state
placed_orders_data = []
scheduled_orders_data = []
gtt_orders_data = []
auto_orders_data = []

placed_current_page = 1
scheduled_current_page = 1
gtt_current_page = 1
auto_current_page = 1

placed_status_filter = "All"
scheduled_status_filter = "All"
gtt_status_filter = "All"
auto_status_filter = "All"

# Source selection state (database vs broker) - Only for Regular and GTT orders
placed_source = "database"
gtt_source = "database"

ORDERS_PER_PAGE = 10

# UI component references for updates
placed_filter_select = None
scheduled_filter_select = None
gtt_filter_select = None
auto_filter_select = None

placed_pagination_info = None
scheduled_pagination_info = None
gtt_pagination_info = None
auto_pagination_info = None

def filter_orders_by_status(orders, status_filter):
    """Filter orders by status"""
    if not status_filter or status_filter == "All":
        return orders
    return [order for order in orders if order.get('status') and order.get('status', '').upper() == status_filter.upper()]

def paginate_orders(orders, page, per_page=ORDERS_PER_PAGE):
    """Paginate orders list"""
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    return orders[start_idx:end_idx]

def get_total_pages(total_orders, per_page=ORDERS_PER_PAGE):
    """Calculate total pages"""
    return max(1, math.ceil(total_orders / per_page))

def get_unique_statuses(orders):
    """Get unique status values from orders"""
    statuses = set(order.get('status', '').upper() for order in orders if order.get('status'))
    return ["All"] + sorted(list(statuses))

def create_header_with_controls(title, refresh_callback, cancel_all_callback, filter_select, source_selector=None):
    """Create a header with title, filter, source selector, refresh icon, and cancel all icon"""
    with ui.row().classes("w-full justify-between items-center"):
        ui.label(title).classes("text-h6 font-bold theme-text-primary")
        
        with ui.row().classes("items-center gap-2"):
            # Filter by Status (positioned left of refresh icon)
            if filter_select:
                filter_select.classes("w-32")
            
            # Source selector (database vs broker)
            if source_selector:
                source_selector
            
            # Refresh icon  
            ui.button(icon="refresh", on_click=refresh_callback).props("flat round size=sm color=primary").tooltip("Refresh Orders")
            
            # Cancel All Orders icon (right of refresh icon)
            ui.button(icon="delete_sweep", on_click=cancel_all_callback).props("flat round size=sm color=negative").tooltip("Cancel All Open Orders")

def create_enhanced_table(columns, rows, row_key, classes="w-full"):
    """Create table with enhanced styling and proper theming"""
    table = ui.table(
        columns=columns,
        rows=rows,
        row_key=row_key
    ).classes(classes)
    
    # Enhanced Quasar table props for better visual distinction with theme-aware colors
    table.props('flat bordered separator="cell" table-header-style="background-color: var(--q-primary); color: white; font-weight: 600;" table-style="color: var(--q-text-primary);"')
    
    # Add enhanced table styling with theme-aware background
    table.classes('shadow-sm theme-surface')
    
    # Custom inline styling for better header distinction
    table.style('border-radius: 8px; overflow: hidden;')
    
    return table

def create_pagination_controls(current_page, total_pages, on_page_change, info_container):
    """Create pagination controls"""
    with ui.row().classes("w-full justify-between items-center mt-3"):
        # Page info
        start_item = (current_page - 1) * ORDERS_PER_PAGE + 1
        end_item = min(current_page * ORDERS_PER_PAGE, len(placed_orders_data))
        
        with ui.row().classes("items-center gap-2"):
            ui.label(f"Showing {start_item}-{end_item} of {len(placed_orders_data)} orders").classes("text-caption text-grey")
        
        # Pagination buttons
        with ui.row().classes("items-center gap-1"):
            ui.button(icon="first_page", on_click=lambda: on_page_change(1)).props("flat dense size=sm").set_enabled(current_page > 1)
            ui.button(icon="chevron_left", on_click=lambda: on_page_change(current_page - 1)).props("flat dense size=sm").set_enabled(current_page > 1)
            
            ui.label(f"Page {current_page} of {total_pages}").classes("text-caption mx-2")
            
            ui.button(icon="chevron_right", on_click=lambda: on_page_change(current_page + 1)).props("flat dense size=sm").set_enabled(current_page < total_pages)
            ui.button(icon="last_page", on_click=lambda: on_page_change(total_pages)).props("flat dense size=sm").set_enabled(current_page < total_pages)

def create_status_filter(current_value, on_change, label="Filter by Status"):
    """Create status filter dropdown with predefined options.
    Ensures the initial value is valid; falls back to 'All' otherwise.
    """
    # Standard order status options
    options = [
        "All",
        "Open",
        "Complete",
        "Rejected",
        "Cancelled",
        "Pending",
        "Active",
        "AMO REQ RECEIVED",
    ]

    # Guard against invalid initial values (e.g., dynamic GTT values like "ACTIVE | NONE")
    safe_value = current_value if current_value in options else "All"

    return ui.select(
        options=options,
        value=safe_value,
        label=label,
        on_change=on_change
    ).classes("w-32").props("dense outlined color=primary size=sm")

def create_source_selector(current_value, on_change, label="Data Source"):
    """Create source selection radio buttons for database vs broker"""
    with ui.column().classes("gap-1"):
        ui.label(label).classes("text-caption text-grey-600")
        ui.radio(
            options=["database", "broker"],
            value=current_value,
            on_change=on_change
        ).props("dense inline")
    return None

async def refresh_placed_orders_with_pagination(fetch_api, broker):
    """Refresh placed orders with pagination support"""
    global placed_orders_data, placed_current_page, placed_status_filter, placed_orders_table, placed_source
    
    try:
        # Fetch all orders with source parameter
        orders_data = await fetch_api(f"/orders/{broker}?source={placed_source}")
        logger.debug(f"Fetched placed orders from {placed_source}: {orders_data}")
        
        if orders_data and isinstance(orders_data, list):
            # Format orders
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
                        'broker': order.get('broker', 'N/A'),
                        'trading_symbol': order.get('trading_symbol', 'N/A'),
                        'instrument_token': order.get('instrument_token', 'N/A'),
                        'transaction_type': order.get('transaction_type', 'N/A'),
                        'quantity': order.get('quantity', 0),
                        'order_type': order.get('order_type', 'N/A'),
                        'price': f"{price:.2f}" if price != 0 else 'N/A',
                        'trigger_price': f"{trigger_price:.2f}" if trigger_price != 0 else 'N/A',
                        'product_type': order.get('product_type', 'N/A'),
                        'status': order.get('status', 'N/A').upper(),
                        'remarks': order.get('remarks', 'N/A'),
                        'order_timestamp': order_timestamp,
                        'is_trailing_stop_loss': '✓' if order.get('is_trailing_stop_loss') else '✗',
                        'trailing_stop_loss_percent': f"{order.get('trailing_stop_loss_percent', 0):.1f}%" if order.get('trailing_stop_loss_percent') else 'N/A',
                        'stop_loss': f"{order.get('stop_loss', 0):.2f}" if order.get('stop_loss') else 'N/A',
                        'target': f"{order.get('target', 0):.2f}" if order.get('target') else 'N/A',
                        'is_amo': '✓' if order.get('is_amo') else '✗',
                        'actions': ''
                    }
                    formatted_orders.append(formatted_order)
                except Exception as e:
                    logger.error(f"Error formatting order {order.get('order_id', 'unknown')}: {str(e)}")
                    continue
            
            # Store all orders
            placed_orders_data = formatted_orders
            
            # Update filter options if filter exists
            if placed_filter_select:
                statuses = get_unique_statuses(placed_orders_data)
                placed_filter_select.options = statuses
                if placed_status_filter not in statuses:
                    placed_status_filter = "All"
                    placed_filter_select.value = "All"
            
            # Apply filtering and pagination
            update_placed_orders_table()
            
        else:
            placed_orders_data = []
            if placed_orders_table:
                placed_orders_table.rows = []
                placed_orders_table.update()
                
    except Exception as e:
        logger.error(f"Exception in refresh_placed_orders_with_pagination: {str(e)}")
        placed_orders_data = []
        if placed_orders_table:
            placed_orders_table.rows = []
            placed_orders_table.update()

def update_placed_orders_table():
    """Update the placed orders table with current page and filter"""
    global placed_orders_table, placed_current_page, placed_status_filter, placed_pagination_info
    
    if not placed_orders_table:
        return
    
    # Filter orders
    filtered_orders = filter_orders_by_status(placed_orders_data, placed_status_filter)
    
    # Calculate pagination
    total_pages = get_total_pages(len(filtered_orders))
    if placed_current_page > total_pages:
        placed_current_page = max(1, total_pages)
    
    # Get current page orders
    paginated_orders = paginate_orders(filtered_orders, placed_current_page)
    
    # Update table
    placed_orders_table.rows = paginated_orders
    placed_orders_table.update()
    
    # Update pagination info
    if placed_pagination_info:
        start_item = (placed_current_page - 1) * ORDERS_PER_PAGE + 1
        end_item = min(placed_current_page * ORDERS_PER_PAGE, len(filtered_orders))
        placed_pagination_info.set_text(f"Showing {start_item}-{end_item} of {len(filtered_orders)} orders")

def on_placed_status_change(e):
    """Handle status filter change for placed orders"""
    global placed_status_filter, placed_current_page
    placed_status_filter = e.value
    placed_current_page = 1  # Reset to first page
    update_placed_orders_table()

def on_placed_source_change(e):
    """Handle placed orders source change"""
    global placed_source, placed_current_page
    placed_source = e.value
    placed_current_page = 1  # Reset to first page
    # Trigger refresh to fetch from new source
    if hasattr(ui, '_current_fetch_api') and hasattr(ui, '_current_broker'):
        create_safe_task(refresh_placed_orders_with_pagination(ui._current_fetch_api, ui._current_broker))

def on_gtt_source_change(e):
    """Handle GTT orders source change"""
    global gtt_source, gtt_current_page
    gtt_source = e.value
    gtt_current_page = 1  # Reset to first page
    # Trigger refresh to fetch from new source
    if hasattr(ui, '_current_fetch_api') and hasattr(ui, '_current_broker'):
        create_safe_task(refresh_gtt_orders_with_pagination(ui._current_fetch_api, ui._current_broker))


def on_placed_page_change(page):
    """Handle page change for placed orders"""
    global placed_current_page
    placed_current_page = page
    update_placed_orders_table()

async def render_placed_orders(fetch_api, user_storage, broker):
    """Render enhanced placed orders with filters and pagination"""
    global placed_orders_table, placed_filter_select, placed_pagination_info
    
    # Store fetch_api and broker for source change callback
    ui._current_fetch_api = fetch_api
    ui._current_broker = broker
    
    # Clean container with proper spacing
    with ui.column().classes("w-full"):
        # Modern section header with glassmorphism design
        with ui.card().classes("w-full shadow-md").style(
            "background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%); "
            "backdrop-filter: blur(15px); "
            "border: 1px solid rgba(59, 130, 246, 0.2); "
            "border-radius: 12px; "
            "overflow: hidden;"
        ):
            with ui.row().classes("w-full justify-between items-center p-4"):
                # Enhanced section title with icon and gradient
                with ui.row().classes("items-center gap-3"):
                    ui.icon("receipt_long", size="1.5rem").classes("text-blue-400")
                    with ui.column().classes("gap-0"):
                        ui.label("Placed Orders").classes(
                            "text-lg font-bold"
                        ).style(
                            "background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%); "
                            "-webkit-background-clip: text; "
                            "-webkit-text-fill-color: transparent; "
                            "background-clip: text;"
                        )
                        ui.label("Active trading orders").classes("text-xs theme-text-secondary opacity-75")
                
                # Enhanced controls with modern styling
                with ui.row().classes("items-center gap-3"):
                    # Modern filter dropdown
                    placed_filter_select = create_status_filter(
                        placed_status_filter, 
                        on_placed_status_change,
                        "Filter by Status"
                    )
                    placed_filter_select.classes("w-36").style(
                        "background: rgba(255, 255, 255, 0.1); "
                        "border: 1px solid rgba(255, 255, 255, 0.2); "
                        "border-radius: 8px; "
                        "backdrop-filter: blur(10px); "
                        "min-height: 40px; "
                        "height: 40px; "
                        "align-self: center;"
                    )
                    
                    # Source selector (Database vs Broker) - Horizontal layout
                    with ui.column().classes("gap-1"):
                        ui.label("Data Source").classes("text-caption text-grey-600")
                        ui.radio(
                            options=["database", "broker"],
                            value=placed_source,
                            on_change=on_placed_source_change
                        ).props("dense inline")
                    
                    # Modern refresh button
                    ui.button(
                        icon="refresh", 
                        on_click=lambda: create_safe_task(refresh_placed_orders_with_pagination(fetch_api, broker))
                    ).props("flat round size=md").classes("modern-action-btn").style(
                        "background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); "
                        "color: white; "
                        "border-radius: 10px; "
                        "box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);"
                    ).tooltip("Refresh Orders")
                    
                    # Modern cancel all button
                    ui.button(
                        icon="delete_sweep", 
                        on_click=lambda: create_safe_task(cancel_single_section_orders(fetch_api, broker, "placed"))
                    ).props("flat round size=md").classes("modern-danger-btn").style(
                        "background: linear-gradient(135deg, #ef4044 0%, #dc2626 100%); "
                        "color: white; "
                        "border-radius: 10px; "
                        "box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);"
                    ).tooltip("Cancel All Open Orders")
        
        # Define table columns - complete database columns
        columns = [
            {'name': 'order_id', 'label': 'Order ID', 'field': 'order_id', 'sortable': True, 'align': 'left'},
            {'name': 'broker', 'label': 'Broker', 'field': 'broker', 'sortable': True, 'align': 'left'},
            {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True, 'align': 'left'},
            {'name': 'instrument_token', 'label': 'Token', 'field': 'instrument_token', 'sortable': True, 'align': 'left'},
            {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type', 'sortable': True, 'align': 'left'},
            {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'sortable': True, 'align': 'right'},
            {'name': 'order_type', 'label': 'Order Type', 'field': 'order_type', 'sortable': True, 'align': 'left'},
            {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True, 'align': 'right'},
            {'name': 'trigger_price', 'label': 'Trigger Price', 'field': 'trigger_price', 'sortable': True, 'align': 'right'},
            {'name': 'product_type', 'label': 'Product', 'field': 'product_type', 'sortable': True, 'align': 'left'},
            {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
            {'name': 'remarks', 'label': 'Remarks', 'field': 'remarks', 'sortable': True, 'align': 'left'},
            {'name': 'order_timestamp', 'label': 'Timestamp', 'field': 'order_timestamp', 'sortable': True, 'align': 'left'},
            {'name': 'is_trailing_stop_loss', 'label': 'Trailing SL', 'field': 'is_trailing_stop_loss', 'sortable': True, 'align': 'center'},
            {'name': 'trailing_stop_loss_percent', 'label': 'Trail %', 'field': 'trailing_stop_loss_percent', 'sortable': True, 'align': 'right'},
            {'name': 'stop_loss', 'label': 'Stop Loss', 'field': 'stop_loss', 'sortable': True, 'align': 'right'},
            {'name': 'target', 'label': 'Target', 'field': 'target', 'sortable': True, 'align': 'right'},
            {'name': 'is_amo', 'label': 'AMO', 'field': 'is_amo', 'sortable': True, 'align': 'center'},
            {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'}
        ]
        
    # Create enhanced table
    placed_orders_table = create_enhanced_table(columns, [], 'order_id')

    # Add action buttons for rows that doesn't contain 'COMPLETE', 'REJECTED','CANCELLED'
    placed_orders_table.add_slot('body-cell-actions', '''
        <q-td :props="props">
            <q-btn v-if="props.row.status && props.row.status !== 'COMPLETE' && props.row.status !== 'REJECTED' && !props.row.status.includes('CANCELLED')" 
                   dense flat round color="primary" icon="edit" size="sm" @click="() => $parent.$emit('modify_order', props.row)">
                <q-tooltip>Modify Order</q-tooltip>
            </q-btn>
            <q-btn v-if="props.row.status && props.row.status !== 'COMPLETE' && props.row.status !== 'REJECTED' && !props.row.status.includes('CANCELLED')" dense flat round color="negative" icon="cancel" size="sm" @click="() => $parent.$emit('cancel_order', props.row.order_id)">
                <q-tooltip>Cancel Order</q-tooltip>
            </q-btn>
            </q-td>
        ''')

    # Pagination controls with info on the left
    with ui.row().classes("w-full justify-between items-center mt-3"):
        # Pagination info on the left
        placed_pagination_info = ui.label("Loading...").classes("text-caption text-grey-6")

        # Pagination buttons on the right
        with ui.row().classes("items-center gap-1"):
            ui.button(icon="first_page", on_click=lambda: on_placed_page_change(1)).props("flat dense size=sm color=primary")
            ui.button(icon="chevron_left", on_click=lambda: on_placed_page_change(max(1, placed_current_page - 1))).props("flat dense size=sm color=primary")
            ui.label("Page").classes("text-caption mx-2")
            ui.button(icon="chevron_right", on_click=lambda: on_placed_page_change(placed_current_page + 1)).props("flat dense size=sm color=primary")
            ui.button(icon="last_page", on_click=lambda: on_placed_page_change(get_total_pages(len(filter_orders_by_status(placed_orders_data, placed_status_filter))))).props("flat dense size=sm color=primary")

    # Event handlers
    def handle_modify_order(order):
        # Validate order_id before proceeding
        order_id = order.get('order_id')
        if not order_id or order_id == 'N/A':
            ui.notify("Invalid order ID. Cannot modify this order.", type="negative")
            return
            
        def create_dialog():
            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label(f"Modify Order {order_id}").classes("text-h6 mb-1")

                with ui.column().classes('w-full gap-2'):
                    quantity_input = ui.number(label="Quantity", value=order.get('quantity', 0), min=1, format='%d').classes("w-full")
                    price_input = ui.number(label="Price", value=float(order.get('price', 0)) if order.get('price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                    trigger_price_input = ui.number(label="Trigger Price", value=float(order.get('trigger_price', 0)) if order.get('trigger_price') != 'N/A' else 0, min=0, step=0.05, format='%.2f').classes("w-full")
                    validity_select = ui.select(options=['DAY', 'IOC'], value=order.get('validity', 'DAY'), label="Validity").classes("w-full")

                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")

                    async def confirm_modify():
                        try:
                            # Validate inputs
                            quantity = int(quantity_input.value)
                            price = float(price_input.value)
                            trigger_price = float(trigger_price_input.value)
                            order_type = order.get('order_type', 'MARKET')
                            
                            # Business logic validation
                            if quantity <= 0:
                                ui.notify("Quantity must be greater than 0", type="negative")
                                return
                            
                            # For LIMIT orders, price must be > 0
                            if order_type == 'LIMIT' and price <= 0:
                                ui.notify("Price must be greater than 0 for LIMIT orders", type="negative")
                                return
                            
                            # For SL orders, trigger_price must be > 0
                            if order_type in ['SL', 'SL-M'] and trigger_price <= 0:
                                ui.notify("Trigger price must be greater than 0 for SL orders", type="negative")
                                return
                            
                            modify_data = {
                                "quantity": quantity,
                                "order_type": order_type,
                                "price": price,
                                "trigger_price": trigger_price,
                                "validity": validity_select.value
                            }
                            
                            # Debug logging
                            logger.info(f"Modifying order {order_id} with data: {modify_data}")
                            
                            response = await fetch_api(f"/orders/{order_id}/modify", method="PUT", data=modify_data)
                            
                            if response and response.get("status") == "success":
                                ui.notify(f"Order {order_id} modified successfully.", type="positive")
                                await refresh_placed_orders_with_pagination(fetch_api, broker)
                            else:
                                error_msg = response.get("detail", "Unknown error") if response else "No response"
                                ui.notify(f"Failed to modify order {order_id}: {error_msg}", type="negative")
                        except ValueError as e:
                            ui.notify(f"Invalid input values: {str(e)}", type="negative")
                        except Exception as e:
                            logger.error(f"Error modifying order {order_id}: {str(e)}")
                            ui.notify(f"Error modifying order: {str(e)}", type="negative")
                        finally:
                            dialog.close()

                    ui.button("Modify", on_click=confirm_modify).props("color=primary")
                dialog.open()
        
        create_dialog()

    def handle_cancel_order(order_id):
        # Validate order_id before proceeding
        if not order_id or order_id == 'N/A':
            ui.notify("Invalid order ID. Cannot cancel this order.", type="negative")
            return
            
        def create_confirmation_dialog():
            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label("Confirm Order Cancellation").classes("text-h6 mb-4")
                
                with ui.column().classes('w-full gap-2'):
                    ui.label(f"Are you sure you want to cancel order {order_id}?")
                    ui.label("This action cannot be undone.").classes("text-red-600 font-medium")
                
                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    
                    async def confirm_cancel():
                        try:
                            logger.info(f"Cancelling order {order_id}")
                            response = await fetch_api(f"/orders/{order_id}", method="DELETE")
                            
                            if response and response.get("status") == "success":
                                ui.notify(f"Order {order_id} cancelled successfully.", type="positive")
                            else:
                                error_msg = response.get("detail", "Unknown error") if response else "No response"
                                ui.notify(f"Failed to cancel order {order_id}: {error_msg}", type="negative")
                            
                            await refresh_placed_orders_with_pagination(fetch_api, broker)
                        except Exception as e:
                            logger.error(f"Exception in handle_cancel_order: {str(e)}")
                            ui.notify(f"Error cancelling order: {str(e)}", type="negative")
                        finally:
                            dialog.close()
                    
                    ui.button("Confirm Cancel", on_click=confirm_cancel).props("color=negative")
                
                dialog.open()
        
        create_confirmation_dialog()

    placed_orders_table.on('modify_order', lambda e: handle_modify_order(e.args))
    placed_orders_table.on('cancel_order', lambda e: handle_cancel_order(e.args))

    # Register WS callback to auto-refresh on placed order events
    def _on_order_event(data: dict):
        try:
            if data.get('scope') == 'placed' and data.get('broker', '').lower() == broker.lower():
                create_safe_task(refresh_placed_orders_with_pagination(fetch_api, broker))
        except Exception:
            pass
    register_order_ws_callback(_on_order_event)

    # Initial load
    await refresh_placed_orders_with_pagination(fetch_api, broker)

async def cancel_single_section_orders(fetch_api, broker, section):
    """Cancel orders for a specific section"""
    try:
        endpoint_map = {
            "placed": f"/orders/cancel-all/{broker}",
            "scheduled": f"/scheduled-orders/cancel-all/{broker}",
            "gtt": f"/gtt-orders/cancel-all/{broker}",
            "auto": f"/auto-orders/cancel-all/{broker}"
        }
        
        endpoint = endpoint_map.get(section)
        if endpoint:
            response = await fetch_api(endpoint, method="DELETE")
            if response and response.get("status") == "success":
                safe_notify(f"All {section} orders cancelled successfully.", notify_type="positive")
            else:
                safe_notify(f"Failed to cancel {section} orders.", notify_type="negative")
            
            # Refresh the appropriate section
            if section == "placed":
                await refresh_placed_orders_with_pagination(fetch_api, broker)
            elif section == "scheduled":
                await refresh_scheduled_orders_with_pagination(fetch_api, broker)
            elif section == "gtt":
                await refresh_gtt_orders_with_pagination(fetch_api, broker)
            elif section == "auto":
                await refresh_auto_orders_with_pagination(fetch_api, broker)

    except Exception as e:
        safe_notify(f"Error cancelling {section} orders: {str(e)}", notify_type="negative")

# Similar functions for other order types would follow the same pattern...
# For brevity, I'll create placeholders that can be expanded

async def refresh_scheduled_orders_with_pagination(fetch_api, broker):
    """Refresh scheduled orders with pagination support"""
    global scheduled_orders_data, scheduled_current_page, scheduled_status_filter, scheduled_orders_table, scheduled_filter_select, scheduled_pagination_info
    
    try:
        # Scheduled orders are application-specific and only stored in database
        orders_data = await fetch_api(f"/scheduled-orders/{broker}")
        logger.debug(f"Fetched scheduled orders: {orders_data}")
        
        if orders_data and isinstance(orders_data, list):
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

                    # Format additional timestamps
                    executed_at = order.get('executed_at', '')
                    if executed_at:
                        try:
                            executed_at = datetime.fromisoformat(executed_at.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            executed_at = 'N/A'
                    else:
                        executed_at = 'N/A'
                    
                    created_at = order.get('created_at', '')
                    if created_at:
                        try:
                            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            created_at = 'N/A'
                    else:
                        created_at = 'N/A'

                    formatted_order = {
                        'scheduled_order_id': order.get('scheduled_order_id', 'N/A'),
                        'broker': order.get('broker', 'N/A'),
                        'instrument_token': order.get('instrument_token', 'N/A'),
                        'trading_symbol': order.get('trading_symbol', 'N/A'),
                        'transaction_type': order.get('transaction_type', 'N/A'),
                        'quantity': order.get('quantity', 0),
                        'order_type': order.get('order_type', 'N/A'),
                        'price': f"{price:.2f}" if price != 0 else 'N/A',
                        'trigger_price': f"{trigger_price:.2f}" if trigger_price != 0 else 'N/A',
                        'product_type': order.get('product_type', 'N/A'),
                        'schedule_datetime': schedule_datetime,
                        'stop_loss': f"{order.get('stop_loss', 0):.2f}" if order.get('stop_loss') else 'N/A',
                        'target': f"{order.get('target', 0):.2f}" if order.get('target') else 'N/A',
                        'status': order.get('status', 'N/A').upper(),
                        'is_amo': '✓' if order.get('is_amo') else '✗',
                        'executed_at': executed_at,
                        'created_at': created_at,
                        'actions': ''
                    }
                    formatted_orders.append(formatted_order)
                except Exception as e:
                    logger.error(f"Error formatting scheduled order {order.get('scheduled_order_id', 'unknown')}: {str(e)}")
                    continue
            
            scheduled_orders_data = formatted_orders
            
            # Update filter options
            if scheduled_filter_select:
                statuses = get_unique_statuses(scheduled_orders_data)
                scheduled_filter_select.options = statuses
                if scheduled_status_filter not in statuses:
                    scheduled_status_filter = "All"
                    scheduled_filter_select.value = "All"
            
            update_scheduled_orders_table()
        else:
            scheduled_orders_data = []
            if scheduled_orders_table:
                scheduled_orders_table.rows = []
                scheduled_orders_table.update()
                
    except Exception as e:
        logger.error(f"Exception in refresh_scheduled_orders_with_pagination: {str(e)}")
        scheduled_orders_data = []

def update_scheduled_orders_table():
    """Update scheduled orders table with current page and filter"""
    global scheduled_orders_table, scheduled_current_page, scheduled_status_filter, scheduled_pagination_info
    
    if not scheduled_orders_table:
        return
    
    filtered_orders = filter_orders_by_status(scheduled_orders_data, scheduled_status_filter)
    total_pages = get_total_pages(len(filtered_orders))
    if scheduled_current_page > total_pages:
        scheduled_current_page = max(1, total_pages)
    
    paginated_orders = paginate_orders(filtered_orders, scheduled_current_page)
    scheduled_orders_table.rows = paginated_orders
    scheduled_orders_table.update()
    
    if scheduled_pagination_info:
        start_item = (scheduled_current_page - 1) * ORDERS_PER_PAGE + 1
        end_item = min(scheduled_current_page * ORDERS_PER_PAGE, len(filtered_orders))
        scheduled_pagination_info.set_text(f"Showing {start_item}-{end_item} of {len(filtered_orders)} orders")

def on_scheduled_status_change(e):
    global scheduled_status_filter, scheduled_current_page
    scheduled_status_filter = e.value
    scheduled_current_page = 1
    update_scheduled_orders_table()

def on_scheduled_page_change(page):
    global scheduled_current_page
    scheduled_current_page = page
    update_scheduled_orders_table()

async def render_scheduled_orders(fetch_api, user_storage, broker):
    """Enhanced scheduled orders with filters and pagination"""
    global scheduled_orders_table, scheduled_filter_select, scheduled_pagination_info
    
    # Modern section header with glassmorphism design
    with ui.card().classes("w-full shadow-md").style(
        "background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%); "
        "backdrop-filter: blur(15px); "
        "border: 1px solid rgba(245, 158, 11, 0.2); "
        "border-radius: 12px; "
        "overflow: hidden;"
    ):
        with ui.row().classes("w-full justify-between items-center p-4"):
            # Enhanced section title with icon and gradient
            with ui.row().classes("items-center gap-3"):
                ui.icon("schedule", size="1.5rem").classes("text-amber-400")
                with ui.column().classes("gap-0"):
                    ui.label("Scheduled Orders").classes(
                        "text-lg font-bold"
                    ).style(
                        "background: linear-gradient(135deg, #f59e0b 0%, #a855f7 100%); "
                        "-webkit-background-clip: text; "
                        "-webkit-text-fill-color: transparent; "
                        "background-clip: text;"
                    )
                    ui.label("Time-based order execution").classes("text-xs theme-text-secondary opacity-75")
            
            # Enhanced controls with modern styling
            with ui.row().classes("items-center gap-3"):
                # Modern filter dropdown
                scheduled_filter_select = create_status_filter(
                    scheduled_status_filter, 
                    on_scheduled_status_change,
                    "Filter by Status"
                )
                scheduled_filter_select.classes("w-36").style(
                    "background: rgba(255, 255, 255, 0.1); "
                    "border: 1px solid rgba(255, 255, 255, 0.2); "
                    "border-radius: 8px; "
                    "backdrop-filter: blur(10px); "
                    "min-height: 40px; "
                    "height: 40px; "
                    "align-self: center;"
                )
                
                # Modern refresh button
                ui.button(
                    icon="refresh", 
                    on_click=lambda: create_safe_task(refresh_scheduled_orders_with_pagination(fetch_api, broker))
                ).props("flat round size=md").classes("modern-action-btn").style(
                    "background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); "
                    "color: white; "
                    "border-radius: 10px; "
                    "box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);"
                ).tooltip("Refresh Orders")
                
                # Modern cancel all button
                ui.button(
                    icon="delete_sweep", 
                    on_click=lambda: create_safe_task(cancel_single_section_orders(fetch_api, broker, "scheduled"))
                ).props("flat round size=md").classes("modern-danger-btn").style(
                    "background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); "
                    "color: white; "
                    "border-radius: 10px; "
                    "box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);"
                ).tooltip("Cancel All Open Orders")
    
    # Complete scheduled orders columns
    columns = [
        {'name': 'scheduled_order_id', 'label': 'Order ID', 'field': 'scheduled_order_id', 'sortable': True, 'align': 'left'},
        {'name': 'broker', 'label': 'Broker', 'field': 'broker', 'sortable': True, 'align': 'left'},
        {'name': 'instrument_token', 'label': 'Token', 'field': 'instrument_token', 'sortable': True, 'align': 'left'},
        {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True, 'align': 'left'},
        {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type', 'sortable': True, 'align': 'left'},
        {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'sortable': True, 'align': 'right'},
        {'name': 'order_type', 'label': 'Order Type', 'field': 'order_type', 'sortable': True, 'align': 'left'},
        {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True, 'align': 'right'},
        {'name': 'trigger_price', 'label': 'Trigger Price', 'field': 'trigger_price', 'sortable': True, 'align': 'right'},
        {'name': 'product_type', 'label': 'Product', 'field': 'product_type', 'sortable': True, 'align': 'left'},
        {'name': 'schedule_datetime', 'label': 'Scheduled Time', 'field': 'schedule_datetime', 'sortable': True, 'align': 'left'},
        {'name': 'stop_loss', 'label': 'Stop Loss', 'field': 'stop_loss', 'sortable': True, 'align': 'right'},
        {'name': 'target', 'label': 'Target', 'field': 'target', 'sortable': True, 'align': 'right'},
        {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
        {'name': 'is_amo', 'label': 'AMO', 'field': 'is_amo', 'sortable': True, 'align': 'center'},
        {'name': 'executed_at', 'label': 'Executed At', 'field': 'executed_at', 'sortable': True, 'align': 'left'},
        {'name': 'created_at', 'label': 'Created At', 'field': 'created_at', 'sortable': True, 'align': 'left'},
        {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'}
    ]
    
    scheduled_orders_table = create_enhanced_table(columns, [], 'scheduled_order_id')
    
    scheduled_orders_table.add_slot('body-cell-actions', '''
        <q-td :props="props">
            <q-btn v-if="props.row.status && (props.row.status === 'PENDING' || props.row.status === 'SCHEDULED')"
                   dense flat round color="primary" icon="edit" size="sm"
                   @click="() => $parent.$emit('modify_order', props.row)">
                <q-tooltip>Modify Order</q-tooltip>
            </q-btn>
            <q-btn v-if="props.row.status && (props.row.status === 'PENDING' || props.row.status === 'SCHEDULED')"
                   dense flat round color="negative" icon="cancel" size="sm"
                   @click="() => $parent.$emit('cancel_order', props.row.scheduled_order_id)">
                <q-tooltip>Cancel Order</q-tooltip>
            </q-btn>
        </q-td>
    ''')
    
    # Pagination controls with info on the left
    with ui.row().classes("w-full justify-between items-center mt-4"):
        # Pagination info on the left
        scheduled_pagination_info = ui.label("Loading...").classes("text-caption text-grey-6")
        
        # Pagination buttons on the right
        with ui.row().classes("items-center gap-1"):
            ui.button(icon="first_page", on_click=lambda: on_scheduled_page_change(1)).props("flat dense size=sm color=primary")
            ui.button(icon="chevron_left", on_click=lambda: on_scheduled_page_change(max(1, scheduled_current_page - 1))).props("flat dense size=sm color=primary")
            ui.label("Page").classes("text-caption mx-2")
            ui.button(icon="chevron_right", on_click=lambda: on_scheduled_page_change(scheduled_current_page + 1)).props("flat dense size=sm color=primary")
            ui.button(icon="last_page", on_click=lambda: on_scheduled_page_change(get_total_pages(len(filter_orders_by_status(scheduled_orders_data, scheduled_status_filter))))).props("flat dense size=sm color=primary")
    
    def handle_cancel_order(order_id):
        def create_confirmation_dialog():
            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label("Confirm Scheduled Order Cancellation").classes("text-h6 mb-4")
                
                with ui.column().classes('w-full gap-2'):
                    ui.label(f"Are you sure you want to cancel scheduled order {order_id}?")
                    ui.label("This action cannot be undone.").classes("text-red-600 font-medium")
                
                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    
                    async def confirm_cancel():
                        try:
                            response = await fetch_api(f"/scheduled-orders/{order_id}?broker={broker}", method="DELETE")
                            if response and response.get("status") == "success":
                                ui.notify(f"Order {order_id} cancelled successfully.", type="positive")
                            else:
                                ui.notify(f"Failed to cancel order {order_id}.", type="negative")
                            await refresh_scheduled_orders_with_pagination(fetch_api, broker)
                        except Exception as e:
                            ui.notify(f"Error cancelling order: {str(e)}", type="negative")
                        finally:
                            dialog.close()
                    
                    ui.button("Confirm Cancel", on_click=confirm_cancel).props("color=negative")
                
                dialog.open()
        
        create_confirmation_dialog()
    
    scheduled_orders_table.on('cancel_order', lambda e: handle_cancel_order(e.args))

    # Register WS callback to auto-refresh on scheduled order events
    def _on_sched_event(data: dict):
        try:
            if data.get('scope') == 'scheduled' and data.get('broker', '').lower() == broker.lower():
                create_safe_task(refresh_scheduled_orders_with_pagination(fetch_api, broker))
        except Exception:
            pass
    register_order_ws_callback(_on_sched_event)

    await refresh_scheduled_orders_with_pagination(fetch_api, broker)

async def refresh_gtt_orders_with_pagination(fetch_api, broker, attempt: int = 1):
    """Refresh GTT orders with pagination support"""
    global gtt_orders_data, gtt_current_page, gtt_status_filter, gtt_orders_table, gtt_filter_select, gtt_pagination_info, gtt_source
    
    try:
        orders_data = await fetch_api(f"/gtt-orders/{broker}?source={gtt_source}")
        logger.debug(f"Fetched GTT orders from {gtt_source}: {orders_data}")
        
        if orders_data and isinstance(orders_data, list):
            formatted_orders = []
            for order in orders_data:
                try:
                    trigger_price = float(order.get('trigger_price', 0)) if order.get('trigger_price') is not None else 0
                    # Note: limit_price is available in _full_data for details view
                    created_at = order.get('created_at', '')
                    if created_at:
                        try:
                            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            logger.warning(f"Failed to parse created_at for GTT order {order.get('gtt_order_id', 'unknown')}: {str(e)}")
                            created_at = 'N/A'
                    else:
                        created_at = 'N/A'

                    # Note: last_price, second_trigger_price, second_limit_price are available in _full_data for details view

                    # Process rules array for enhanced display
                    rules = order.get('rules', [])
                    gtt_type = order.get('gtt_type', 'SINGLE')
                    
                    # Create rules summary for table display
                    rules_summary = "No rules"
                    if rules:
                        rule_descriptions = []
                        for rule in rules:
                            strategy = rule.get('strategy', 'UNKNOWN')
                            trigger_type = rule.get('trigger_type', 'UNKNOWN')
                            trigger_price = rule.get('trigger_price', 0)
                            trailing_gap = rule.get('trailing_gap')
                            
                            # Create more readable rule description
                            if strategy == 'ENTRY':
                                rule_desc = f"📈 {strategy} @ ₹{trigger_price:.2f} ({trigger_type})"
                            elif strategy == 'TARGET':
                                rule_desc = f"🎯 {strategy} @ ₹{trigger_price:.2f}"
                            elif strategy == 'STOPLOSS':
                                if trailing_gap:
                                    rule_desc = f"🔄 {strategy} @ ₹{trigger_price:.2f} (Trail: {trailing_gap})"
                                else:
                                    rule_desc = f"🛑 {strategy} @ ₹{trigger_price:.2f} ({trigger_type})"
                            else:
                                rule_desc = f"⚙️ {strategy} @ ₹{trigger_price:.2f} ({trigger_type})"
                            
                            rule_descriptions.append(rule_desc)
                        
                        rules_summary = " | ".join(rule_descriptions)
                    
                    formatted_order = {
                        'gtt_order_id': order.get('gtt_order_id', 'N/A'),
                        'trading_symbol': order.get('trading_symbol', 'N/A'),
                        'transaction_type': order.get('transaction_type', 'N/A'),
                        'quantity': order.get('quantity', 0),
                        'gtt_type': gtt_type,
                        'rules_summary': rules_summary,
                        # Keep status pure for filtering; show type in its own column
                        'status': f"{order.get('status', 'N/A').upper()}",
                        'broker': order.get('broker', 'N/A'),
                        'created_at': created_at,
                        'actions': '',
                        # Store full data for details view
                        '_full_data': order
                    }
                    formatted_orders.append(formatted_order)
                except Exception as e:
                    logger.error(f"Error formatting GTT order {order.get('gtt_order_id', 'unknown')}: {str(e)}")
                    continue
            
            gtt_orders_data = formatted_orders
            
            if gtt_filter_select:
                statuses = get_unique_statuses(gtt_orders_data)
                gtt_filter_select.options = statuses
                if gtt_status_filter not in statuses:
                    gtt_status_filter = "All"
                    gtt_filter_select.value = "All"
            
            update_gtt_orders_table()
        else:
            gtt_orders_data = []
                
    except Exception as e:
        logger.error(f"Exception in refresh_gtt_orders_with_pagination: {str(e)}")
        gtt_orders_data = []

    # If very few results on first attempt, retry once after a brief delay
    try:
        if attempt == 1 and isinstance(gtt_orders_data, list) and len(gtt_orders_data) <= 1:
            await asyncio.sleep(0.5)
            await refresh_gtt_orders_with_pagination(fetch_api, broker, attempt=2)
    except Exception:
        # Best-effort retry; ignore any errors here
        pass

def update_gtt_orders_table():
    global gtt_orders_table, gtt_current_page, gtt_status_filter, gtt_pagination_info
    
    if not gtt_orders_table:
        return
    
    filtered_orders = filter_orders_by_status(gtt_orders_data, gtt_status_filter)
    total_pages = get_total_pages(len(filtered_orders))
    if gtt_current_page > total_pages:
        gtt_current_page = max(1, total_pages)
    
    paginated_orders = paginate_orders(filtered_orders, gtt_current_page)
    gtt_orders_table.rows = paginated_orders
    gtt_orders_table.update()
    
    if gtt_pagination_info:
        start_item = (gtt_current_page - 1) * ORDERS_PER_PAGE + 1
        end_item = min(gtt_current_page * ORDERS_PER_PAGE, len(filtered_orders))
        gtt_pagination_info.set_text(f"Showing {start_item}-{end_item} of {len(filtered_orders)} orders")

def on_gtt_status_change(e):
    global gtt_status_filter, gtt_current_page
    gtt_status_filter = e.value
    gtt_current_page = 1
    update_gtt_orders_table()

def on_gtt_page_change(page):
    global gtt_current_page
    gtt_current_page = page
    update_gtt_orders_table()

async def render_gtt_orders(fetch_api, user_storage, broker):
    """Enhanced GTT orders with filters and pagination"""
    global gtt_orders_table, gtt_filter_select, gtt_pagination_info
    
    # Modern section header with glassmorphism design
    with ui.card().classes("w-full shadow-md").style(
        "background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%); "
        "backdrop-filter: blur(15px); "
        "border: 1px solid rgba(16, 185, 129, 0.2); "
        "border-radius: 12px; "
        "overflow: hidden;"
    ):
        with ui.row().classes("w-full justify-between items-center p-4"):
            # Enhanced section title with icon and gradient
            with ui.row().classes("items-center gap-3"):
                ui.icon("gps_fixed", size="1.5rem").classes("text-emerald-400")
                with ui.column().classes("gap-0"):
                    ui.label("GTT Orders").classes(
                        "text-lg font-bold"
                    ).style(
                        "background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); "
                        "-webkit-background-clip: text; "
                        "-webkit-text-fill-color: transparent; "
                        "background-clip: text;"
                    )
                    ui.label("Good Till Triggered orders").classes("text-xs theme-text-secondary opacity-75")
            
            # Enhanced controls with modern styling
            with ui.row().classes("items-center gap-3"):
                # Modern filter dropdown
                gtt_filter_select = create_status_filter(
                    gtt_status_filter, 
                    on_gtt_status_change,
                    "Filter by Status"
                )
                gtt_filter_select.classes("w-36").style(
                    "background: rgba(255, 255, 255, 0.1); "
                    "border: 1px solid rgba(255, 255, 255, 0.2); "
                    "border-radius: 8px; "
                    "backdrop-filter: blur(10px); "
                    "min-height: 40px; "
                    "height: 40px; "
                    "align-self: center;"
                )
                
                # Source selector (Database vs Broker) - Horizontal layout
                with ui.column().classes("gap-1"):
                    ui.label("Data Source").classes("text-caption text-grey-600")
                    ui.radio(
                        options=["database", "broker"],
                        value=gtt_source,
                        on_change=on_gtt_source_change
                    ).props("dense inline")
                
                # Modern refresh button
                ui.button(
                    icon="refresh", 
                    on_click=lambda: create_safe_task(refresh_gtt_orders_with_pagination(fetch_api, broker))
                ).props("flat round size=md").classes("modern-action-btn").style(
                    "background: linear-gradient(135deg, #10b981 0%, #059669 100%); "
                    "color: white; "
                    "border-radius: 10px; "
                    "box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);"
                ).tooltip("Refresh Orders")
                
                # Modern cancel all button
                ui.button(
                    icon="delete_sweep", 
                    on_click=lambda: create_safe_task(cancel_single_section_orders(fetch_api, broker, "gtt"))
                ).props("flat round size=md").classes("modern-danger-btn").style(
                    "background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); "
                    "color: white; "
                    "border-radius: 10px; "
                    "box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);"
                ).tooltip("Cancel All Open Orders")
        
    # Enhanced GTT orders columns with rules support
    columns = [
        {'name': 'gtt_order_id', 'label': 'Order ID', 'field': 'gtt_order_id', 'sortable': True, 'align': 'left'},
        {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True, 'align': 'left'},
        {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type', 'sortable': True, 'align': 'left'},
        {'name': 'quantity', 'label': 'Qty', 'field': 'quantity', 'sortable': True, 'align': 'right'},
        {'name': 'gtt_type', 'label': 'GTT Type', 'field': 'gtt_type', 'sortable': True, 'align': 'left'},
        {'name': 'rules_summary', 'label': 'Rules Summary', 'field': 'rules_summary', 'sortable': False, 'align': 'left'},
        {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'left'},
        {'name': 'broker', 'label': 'Broker', 'field': 'broker', 'sortable': True, 'align': 'left'},
        {'name': 'created_at', 'label': 'Created At', 'field': 'created_at', 'sortable': True, 'align': 'left'},
        {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'}
    ]
    
    gtt_orders_table = create_enhanced_table(columns, [], 'gtt_order_id')
        
    gtt_orders_table.add_slot('body-cell-actions', '''
        <q-td :props="props">
            <q-btn dense flat round color="primary" icon="info" size="sm"
                   @click="() => $parent.$emit('view_details', props.row)">
                <q-tooltip>View Details</q-tooltip>
            </q-btn>
            <q-btn v-if="props.row.status && (props.row.status === 'ACTIVE' || props.row.status === 'SCHEDULED')"
                   dense flat round color="warning" icon="edit" size="sm"
                   @click="() => $parent.$emit('modify_order', props.row)">
                <q-tooltip>Modify GTT Order</q-tooltip>
            </q-btn>
            <q-btn v-if="props.row.status && (props.row.status === 'ACTIVE' || props.row.status === 'SCHEDULED')"
                   dense flat round color="negative" icon="cancel" size="sm"
                   @click="() => $parent.$emit('cancel_order', props.row.gtt_order_id)">
                <q-tooltip>Cancel GTT Order</q-tooltip>
            </q-btn>
        </q-td>
    ''')

    # Pagination controls with info on the left
    with ui.row().classes("w-full justify-between items-center mt-4"):
        # Pagination info on the left
        gtt_pagination_info = ui.label("Loading...").classes("text-caption text-grey-6")

        # Pagination buttons on the right
        with ui.row().classes("items-center gap-1"):
            ui.button(icon="first_page", on_click=lambda: on_gtt_page_change(1)).props("flat dense size=sm color=primary")
            ui.button(icon="chevron_left", on_click=lambda: on_gtt_page_change(max(1, gtt_current_page - 1))).props("flat dense size=sm color=primary")
            ui.label("Page").classes("text-caption mx-2")
            ui.button(icon="chevron_right", on_click=lambda: on_gtt_page_change(gtt_current_page + 1)).props("flat dense size=sm color=primary")
            ui.button(icon="last_page", on_click=lambda: on_gtt_page_change(get_total_pages(len(filter_orders_by_status(gtt_orders_data, gtt_status_filter))))).props("flat dense size=sm color=primary")

    # GTT Order Handler Functions
    async def handle_view_gtt_details(order_data):
        """Display detailed GTT order information including rules"""
        try:
            # Try to fetch enriched details from backend first
            full_data = None
            try:
                gtt_id = order_data.get('gtt_order_id')
                details = await fetch_api(f"/gtt-orders/{broker}/{gtt_id}")
                if details and not details.get('error'):
                    full_data = details
            except Exception:
                full_data = None

            if not full_data:
                full_data = order_data.get('_full_data', {})
            if not full_data:
                safe_notify("No detailed data available for this order", notify_type="warning")
                return
                
            # Create a detailed dialog
            def create_dialog():
                with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
                    ui.label(f"GTT Order Details - {full_data.get('trading_symbol', 'Unknown')}")
                    
                    with ui.card_section():
                        # Basic order information
                        with ui.row().classes("w-full gap-4 mb-4"):
                            with ui.column().classes("flex-1"):
                                ui.label("Order Information").classes("text-lg font-bold mb-2")
                                ui.label(f"Order ID: {full_data.get('gtt_order_id', 'N/A')}")
                                ui.label(f"Symbol: {full_data.get('trading_symbol', 'N/A')}")
                                ui.label(f"Transaction Type: {full_data.get('transaction_type', 'N/A')}")
                                ui.label(f"Quantity: {full_data.get('quantity', 'N/A')}")
                                ui.label(f"Status: {full_data.get('status', 'N/A')}")
                                ui.label(f"Broker: {full_data.get('broker', 'N/A')}")
                            
                            with ui.column().classes("flex-1"):
                                ui.label("GTT Configuration").classes("text-lg font-bold mb-2")
                                ui.label(f"GTT Type: {full_data.get('gtt_type', 'N/A')}")
                                ui.label(f"Trigger Type: {full_data.get('trigger_type', 'N/A')}")
                                ui.label(f"Created At: {full_data.get('created_at', 'N/A')}")
                                if full_data.get('instrument_token'):
                                    ui.label(f"Instrument Token: {full_data.get('instrument_token', 'N/A')}")
                        
                        # Rules section
                        rules = full_data.get('rules', [])
                        if rules:
                            ui.label("Rules Configuration").classes("text-lg font-bold mb-3")

                            # Build columns and rows for the table
                            columns = [
                                {'name': 'rule_index', 'label': 'Rule #', 'field': 'rule_index', 'align': 'center'},
                                {'name': 'strategy', 'label': 'Strategy', 'field': 'strategy', 'align': 'left'},
                                {'name': 'trigger_type', 'label': 'Trigger Type', 'field': 'trigger_type', 'align': 'left'},
                                {'name': 'trigger_price', 'label': 'Trigger Price', 'field': 'trigger_price', 'align': 'right'},
                                {'name': 'trailing_gap', 'label': 'Trailing Gap', 'field': 'trailing_gap', 'align': 'right'},
                                {'name': 'description', 'label': 'Description', 'field': 'description', 'align': 'left'}
                            ]

                            rows = []
                            for rule in rules:
                                strategy = rule.get('strategy', 'UNKNOWN')
                                trigger_type = rule.get('trigger_type', 'UNKNOWN')
                                try:
                                    trigger_price_val = float(rule.get('trigger_price', 0) or 0)
                                except Exception:
                                    trigger_price_val = 0.0
                                trailing_gap = rule.get('trailing_gap')

                                # Create description
                                if strategy == 'ENTRY':
                                    desc = f"Place {full_data.get('transaction_type', 'BUY')} order when price {trigger_type.lower()} {trigger_price_val:.2f}"
                                elif strategy == 'TARGET':
                                    desc = f"Take profit when price reaches {trigger_price_val:.2f}"
                                elif strategy == 'STOPLOSS':
                                    if trailing_gap:
                                        desc = f"Trailing stop loss with {trailing_gap} gap"
                                    else:
                                        desc = f"Stop loss at {trigger_price_val:.2f}"
                                else:
                                    desc = f"{strategy} at {trigger_price_val:.2f}"

                                rows.append({
                                    'rule_index': rule.get('rule_index', 0),
                                    'strategy': strategy,
                                    'trigger_type': trigger_type,
                                    'trigger_price': f"{trigger_price_val:.2f}",
                                    'trailing_gap': f"{float(trailing_gap):.2f}" if trailing_gap not in (None, '') else 'N/A',
                                    'description': desc
                                })

                            ui.table(columns=columns, rows=rows).classes("w-full").props('dense separator="cell"')
                        else:
                            ui.label("No rules configured for this GTT order").classes("text-grey-6 italic")
                        
                        # Legacy fields (if available)
                        legacy_fields = []
                        if full_data.get('trigger_price'):
                            legacy_fields.append(f"Primary Trigger: {full_data.get('trigger_price'):.2f}")
                        if full_data.get('limit_price'):
                            legacy_fields.append(f"Primary Limit: {full_data.get('limit_price'):.2f}")
                        if full_data.get('second_trigger_price'):
                            legacy_fields.append(f"Secondary Trigger: {full_data.get('second_trigger_price'):.2f}")
                        if full_data.get('second_limit_price'):
                            legacy_fields.append(f"Secondary Limit: {full_data.get('second_limit_price'):.2f}")
                        
                        if legacy_fields:
                            ui.label("Legacy Fields").classes("text-lg font-bold mb-2 mt-4")
                            with ui.row().classes("w-full gap-4"):
                                for field in legacy_fields:
                                    ui.label(field).classes("text-sm")
                    
                    # Close button
                    with ui.card_actions().classes("justify-end"):
                        ui.button("Close", on_click=dialog.close).props("flat color=primary")
                    
                    dialog.open()
            
            # Execute dialog creation (runs in UI context when called from event handler)
            create_dialog()
                
        except Exception as e:
            logger.error(f"Error displaying GTT order details: {str(e)}")
            safe_notify(f"Error displaying order details: {str(e)}", notify_type="negative")

    async def handle_modify_gtt_order(order_data):
        """Display modification dialog for GTT orders and submit changes"""
        try:
            # Fetch enriched details for accurate editing
            gtt_id = order_data.get('gtt_order_id')
            details = await fetch_api(f"/gtt-orders/{broker}/{gtt_id}")
            full_data = details if details and not details.get('error') else order_data.get('_full_data', {})
            if not full_data:
                safe_notify("No detailed data available for this order", notify_type="warning")
                return

            # Initial form values
            trigger_price_val = full_data.get('trigger_price') or 0.0
            limit_price_val = full_data.get('limit_price') or trigger_price_val
            last_price_val = full_data.get('last_price') or 0.0
            quantity_val = full_data.get('quantity') or 1
            trigger_type_val = full_data.get('trigger_type') or "single"
            rules_val = full_data.get('rules')

            # Dialog UI
            def create_modify_dialog():
                with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl"):
                    ui.label(f"Modify GTT Order - {full_data.get('trading_symbol', 'Unknown')}")

                    with ui.card_section():
                        with ui.row().classes("gap-3"):
                            trig = ui.number("Trigger Price", value=trigger_price_val).classes("w-40")
                            lim = ui.number("Limit Price", value=limit_price_val).classes("w-40")
                            qty = ui.number("Quantity", value=quantity_val).classes("w-32")
                        with ui.row().classes("gap-3 mt-2"):
                            lst = ui.number("Last Price", value=last_price_val).classes("w-40")
                            ttype = ui.select(["single", "two_leg"], value=trigger_type_val, label="Trigger Type").classes("w-40")

                    async def submit_modify():
                        data = {
                            "instrument_token": full_data.get('instrument_token', ''),
                            "trading_symbol": full_data.get('trading_symbol', ''),
                            "transaction_type": full_data.get('transaction_type', 'BUY'),
                            "quantity": int(qty.value or quantity_val),
                            "trigger_type": ttype.value or trigger_type_val,
                            "trigger_price": float(trig.value or 0.0),
                            "limit_price": float(lim.value or 0.0),
                            "last_price": float(lst.value or 0.0),
                            "second_trigger_price": full_data.get('second_trigger_price'),
                            "second_limit_price": full_data.get('second_limit_price'),
                            "rules": rules_val,
                            "broker": broker,
                        }
                        resp = await fetch_api(f"/gtt-orders/{broker}/{gtt_id}", method="PUT", data=data)
                        if resp and not resp.get('error') and resp.get('status') == 'success':
                            safe_notify("GTT order modified", notify_type="positive")
                            dialog.close()
                            await refresh_gtt_orders_with_pagination(fetch_api, broker)
                        else:
                            msg = resp.get('message') if resp and isinstance(resp, dict) else 'Modify failed'
                            safe_notify(f"Failed to modify: {msg}", notify_type="negative")

                    with ui.card_actions().classes("justify-end"):
                        ui.button("Update", on_click=lambda: create_safe_task(submit_modify())).props("color=primary")
                        ui.button("Close", on_click=dialog.close).props("flat")

                    dialog.open()
            
            # Execute dialog creation
            create_modify_dialog()
        except Exception as e:
            logger.error(f"Error displaying GTT order modification dialog: {str(e)}")
            safe_notify(f"Error displaying modification dialog: {str(e)}", notify_type="negative")

    async def handle_cancel_gtt_order(order_id):
        """Cancel GTT order with confirmation dialog"""
        try:
            # Get order details for confirmation
            order_data = None
            for order in gtt_orders_data:
                if order.get('gtt_order_id') == order_id:
                    order_data = order
                    break
            
            if not order_data:
                safe_notify("Order not found", notify_type="warning")
                return
            
            # Create confirmation dialog
            def create_confirmation_dialog():
                with ui.dialog() as dialog, ui.card().classes("w-96"):
                    ui.label("Confirm GTT Order Cancellation")
                    
                    with ui.card_section():
                        ui.label("Are you sure you want to cancel this GTT order?")
                        ui.label(f"Order ID: {order_id}")
                        ui.label(f"Symbol: {order_data.get('trading_symbol', 'N/A')}")
                        ui.label(f"Quantity: {order_data.get('quantity', 'N/A')}")
                        ui.label(f"Trigger Price: {order_data.get('trigger_price', 'N/A')}")
                        ui.label("").classes("text-red-600 font-medium")  # Empty line for spacing
                        ui.label("This action cannot be undone.").classes("text-red-600 font-medium")
                    
                    with ui.card_actions().classes("justify-end gap-2"):
                        ui.button("Cancel", on_click=dialog.close).props("flat")
                        
                        async def confirm_cancel():
                            try:
                                response = await fetch_api(f"/gtt-orders/{broker}/{order_id}", method="DELETE")
                                if response and response.get("status") == "success":
                                    safe_notify(f"GTT order {order_id} cancelled successfully.", notify_type="positive")
                                    dialog.close()
                                    await refresh_gtt_orders_with_pagination(fetch_api, broker)
                                else:
                                    safe_notify(f"Failed to cancel GTT order {order_id}.", notify_type="negative")
                            except Exception as e:
                                safe_notify(f"Error cancelling GTT order: {str(e)}", notify_type="negative")
                        
                        ui.button("Confirm Cancel", on_click=lambda: create_safe_task(confirm_cancel())).props("color=negative")
                    
                    dialog.open()
            
            # Execute dialog creation
            create_confirmation_dialog()
        except Exception as e:
            safe_notify(f"Error cancelling GTT order: {str(e)}", notify_type="negative")

    # Set up event handlers
    async def _on_gtt_view_details(e):
        await handle_view_gtt_details(e.args)
    async def _on_gtt_modify(e):
        await handle_modify_gtt_order(e.args)
    async def _on_gtt_cancel(e):
        await handle_cancel_gtt_order(e.args)

    gtt_orders_table.on('cancel_order', _on_gtt_cancel)
    gtt_orders_table.on('view_details', _on_gtt_view_details)
    gtt_orders_table.on('modify_order', _on_gtt_modify)

    # Register WS callback to auto-refresh on GTT order events
    def _on_gtt_event(data: dict):
        try:
            if data.get('scope') == 'gtt' and data.get('broker', '').lower() == broker.lower():
                create_safe_task(refresh_gtt_orders_with_pagination(fetch_api, broker))
        except Exception:
            pass
    register_order_ws_callback(_on_gtt_event)

    # Initial load
    await refresh_gtt_orders_with_pagination(fetch_api, broker)

async def refresh_auto_orders_with_pagination(fetch_api, broker):
    """Refresh auto orders with pagination support"""
    global auto_orders_data, auto_current_page, auto_status_filter, auto_orders_table, auto_filter_select, auto_pagination_info
    
    try:
        orders_data = await fetch_api(f"/auto-orders/{broker}")
        logger.debug(f"Fetched auto orders: {orders_data}")
        
        if orders_data and isinstance(orders_data, list):
            formatted_orders = []
            for order in orders_data:
                try:
                    stop_loss_value = float(order.get('stop_loss_value', 0)) if order.get('stop_loss_value') is not None else 0
                    target_value = float(order.get('target_value', 0)) if order.get('target_value') is not None else 0
                    # Format additional fields
                    limit_price = float(order.get('limit_price', 0)) if order.get('limit_price') is not None else 0
                    created_at = order.get('created_at', '')
                    if created_at:
                        try:
                            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            created_at = 'N/A'
                    else:
                        created_at = 'N/A'

                    formatted_order = {
                        'auto_order_id': order.get('auto_order_id', 'N/A'),
                        'instrument_token': order.get('instrument_token', 'N/A'),
                        'trading_symbol': order.get('trading_symbol', 'N/A'),
                        'transaction_type': order.get('transaction_type', 'N/A'),
                        'risk_per_trade': f"{order.get('risk_per_trade', 0):.1f}",
                        'stop_loss_type': order.get('stop_loss_type', 'N/A'),
                        'stop_loss_value': f"{stop_loss_value:.2f}" if stop_loss_value != 0 else 'N/A',
                        'target_value': f"{target_value:.2f}" if target_value != 0 else 'N/A',
                        'atr_period': order.get('atr_period', 'N/A'),
                        'product_type': order.get('product_type', 'N/A'),
                        'order_type': order.get('order_type', 'N/A'),
                        'limit_price': f"{limit_price:.2f}" if limit_price != 0 else 'N/A',
                        'broker': order.get('broker', 'N/A'),
                        'created_at': created_at,
                        'actions': ''
                    }
                    formatted_orders.append(formatted_order)
                except Exception as e:
                    logger.error(f"Error formatting auto order {order.get('auto_order_id', 'unknown')}: {str(e)}")
                    continue
            
            auto_orders_data = formatted_orders
            
            if auto_filter_select:
                statuses = get_unique_statuses(auto_orders_data)
                auto_filter_select.options = statuses
                if auto_status_filter not in statuses:
                    auto_status_filter = "All"
                    auto_filter_select.value = "All"
            
            update_auto_orders_table()
        else:
            auto_orders_data = []
                
    except Exception as e:
        logger.error(f"Exception in refresh_auto_orders_with_pagination: {str(e)}")
        auto_orders_data = []

def update_auto_orders_table():
    global auto_orders_table, auto_current_page, auto_status_filter, auto_pagination_info
    
    if not auto_orders_table:
        return
    
    filtered_orders = filter_orders_by_status(auto_orders_data, auto_status_filter)
    total_pages = get_total_pages(len(filtered_orders))
    if auto_current_page > total_pages:
        auto_current_page = max(1, total_pages)
    
    paginated_orders = paginate_orders(filtered_orders, auto_current_page)
    auto_orders_table.rows = paginated_orders
    auto_orders_table.update()
    
    if auto_pagination_info:
        start_item = (auto_current_page - 1) * ORDERS_PER_PAGE + 1
        end_item = min(auto_current_page * ORDERS_PER_PAGE, len(filtered_orders))
        auto_pagination_info.set_text(f"Showing {start_item}-{end_item} of {len(filtered_orders)} orders")

def on_auto_status_change(e):
    global auto_status_filter, auto_current_page
    auto_status_filter = e.value
    auto_current_page = 1
    update_auto_orders_table()

def on_auto_page_change(page):
    global auto_current_page
    auto_current_page = page
    update_auto_orders_table()

async def render_auto_orders(fetch_api, user_storage, broker):
    """Enhanced auto orders with filters and pagination"""
    global auto_orders_table, auto_filter_select, auto_pagination_info
    
    # Modern section header with glassmorphism design
    with ui.card().classes("w-full shadow-md").style(
        "background: linear-gradient(135deg, rgba(168, 85, 247, 0.1) 0%, rgba(236, 72, 153, 0.05) 100%); "
        "backdrop-filter: blur(15px); "
        "border: 1px solid rgba(168, 85, 247, 0.2); "
        "border-radius: 12px; "
        "overflow: hidden;"
    ):
        with ui.row().classes("w-full justify-between items-center p-4"):
            # Enhanced section title with icon and gradient
            with ui.row().classes("items-center gap-3"):
                ui.icon("smart_toy", size="1.5rem").classes("text-purple-400")
                with ui.column().classes("gap-0"):
                    ui.label("Auto Orders").classes(
                        "text-lg font-bold"
                    ).style(
                        "background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%); "
                        "-webkit-background-clip: text; "
                        "-webkit-text-fill-color: transparent; "
                        "background-clip: text;"
                    )
                    ui.label("Automated trading orders").classes("text-xs theme-text-secondary opacity-75")
            
            # Enhanced controls with modern styling
            with ui.row().classes("items-center gap-3"):
                # Modern filter dropdown
                auto_filter_select = create_status_filter(
                    auto_status_filter, 
                    on_auto_status_change,
                    "Filter by Status"
                )
                auto_filter_select.classes("w-36").style(
                    "background: rgba(255, 255, 255, 0.1); "
                    "border: 1px solid rgba(255, 255, 255, 0.2); "
                    "border-radius: 8px; "
                    "backdrop-filter: blur(10px); "
                    "min-height: 40px; "
                    "height: 40px; "
                    "align-self: center;"
                )
                
                # Modern refresh button
                ui.button(
                    icon="refresh", 
                    on_click=lambda: create_safe_task(refresh_auto_orders_with_pagination(fetch_api, broker))
                ).props("flat round size=md").classes("modern-action-btn").style(
                    "background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); "
                    "color: white; "
                    "border-radius: 10px; "
                    "box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3);"
                ).tooltip("Refresh Orders")
                
                # Modern cancel all button
                ui.button(
                    icon="delete_sweep", 
                    on_click=lambda: create_safe_task(cancel_single_section_orders(fetch_api, broker, "auto"))
                ).props("flat round size=md").classes("modern-danger-btn").style(
                    "background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); "
                    "color: white; "
                    "border-radius: 10px; "
                    "box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);"
                ).tooltip("Cancel All Open Orders")
    
    # Complete auto orders columns
    columns = [
        {'name': 'auto_order_id', 'label': 'Order ID', 'field': 'auto_order_id', 'sortable': True, 'align': 'left'},
        {'name': 'instrument_token', 'label': 'Token', 'field': 'instrument_token', 'sortable': True, 'align': 'left'},
        {'name': 'trading_symbol', 'label': 'Symbol', 'field': 'trading_symbol', 'sortable': True, 'align': 'left'},
        {'name': 'transaction_type', 'label': 'Type', 'field': 'transaction_type', 'sortable': True, 'align': 'left'},
        {'name': 'risk_per_trade', 'label': 'Risk (%)', 'field': 'risk_per_trade', 'sortable': True, 'align': 'right'},
        {'name': 'stop_loss_type', 'label': 'Stop Loss Type', 'field': 'stop_loss_type', 'sortable': True, 'align': 'left'},
        {'name': 'stop_loss_value', 'label': 'Stop Loss', 'field': 'stop_loss_value', 'sortable': True, 'align': 'right'},
        {'name': 'target_value', 'label': 'Target', 'field': 'target_value', 'sortable': True, 'align': 'right'},
        {'name': 'atr_period', 'label': 'ATR Period', 'field': 'atr_period', 'sortable': True, 'align': 'right'},
        {'name': 'product_type', 'label': 'Product', 'field': 'product_type', 'sortable': True, 'align': 'left'},
        {'name': 'order_type', 'label': 'Order Type', 'field': 'order_type', 'sortable': True, 'align': 'left'},
        {'name': 'limit_price', 'label': 'Limit Price', 'field': 'limit_price', 'sortable': True, 'align': 'right'},
        {'name': 'broker', 'label': 'Broker', 'field': 'broker', 'sortable': True, 'align': 'left'},
        {'name': 'created_at', 'label': 'Created At', 'field': 'created_at', 'sortable': True, 'align': 'left'},
        {'name': 'actions', 'label': 'Actions', 'field': 'actions', 'align': 'center'}
    ]
    
    auto_orders_table = create_enhanced_table(columns, [], 'auto_order_id')
        
    auto_orders_table.add_slot('body-cell-actions', '''
        <q-td :props="props">
            <q-btn v-if="props.row.status && props.row.status !== 'CANCELLED'"
                   dense flat round color="negative" icon="cancel" size="sm"
                   @click="() => $parent.$emit('cancel_order', props.row.auto_order_id)">
                <q-tooltip>Cancel Auto Order</q-tooltip>
            </q-btn>
        </q-td>
    ''')

    # Pagination controls with info on the left
    with ui.row().classes("w-full justify-between items-center mt-4"):
        # Pagination info on the left
        auto_pagination_info = ui.label("Loading...").classes("text-caption text-grey-6")

        # Pagination buttons on the right
        with ui.row().classes("items-center gap-1"):
            ui.button(icon="first_page", on_click=lambda: on_auto_page_change(1)).props("flat dense size=sm color=primary")
            ui.button(icon="chevron_left", on_click=lambda: on_auto_page_change(max(1, auto_current_page - 1))).props("flat dense size=sm color=primary")
            ui.label("Page").classes("text-caption mx-2")
            ui.button(icon="chevron_right", on_click=lambda: on_auto_page_change(auto_current_page + 1)).props("flat dense size=sm color=primary")
            ui.button(icon="last_page", on_click=lambda: on_auto_page_change(get_total_pages(len(filter_orders_by_status(auto_orders_data, auto_status_filter))))).props("flat dense size=sm color=primary")

    def handle_cancel_order(order_id):
        def create_confirmation_dialog():
            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label("Confirm Auto Order Cancellation").classes("text-h6 mb-4")
                
                with ui.column().classes('w-full gap-2'):
                    ui.label(f"Are you sure you want to cancel auto order {order_id}?")
                    ui.label("This action cannot be undone.").classes("text-red-600 font-medium")
                
                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    
                    async def confirm_cancel():
                        try:
                            response = await fetch_api(f"/auto-orders/{order_id}", method="DELETE")
                            if response and response.get("status") == "success":
                                ui.notify(f"Auto order {order_id} cancelled successfully.", type="positive")
                            else:
                                ui.notify(f"Failed to cancel auto order {order_id}.", type="negative")
                            await refresh_auto_orders_with_pagination(fetch_api, broker)
                        except Exception as e:
                            ui.notify(f"Error cancelling auto order: {str(e)}", type="negative")
                        finally:
                            dialog.close()
                    
                    ui.button("Confirm Cancel", on_click=confirm_cancel).props("color=negative")
                
                dialog.open()
        
        create_confirmation_dialog()

    auto_orders_table.on('cancel_order', lambda e: handle_cancel_order(e.args))
    await refresh_auto_orders_with_pagination(fetch_api, broker)

async def render_order_book_page(fetch_api, user_storage, broker):
    """Enhanced Order Book Page with improved visual design and proper lazy loading"""

    # Track which tabs have been loaded to avoid duplicate API calls
    loaded_tabs = {'placed': True, 'scheduled': False, 'gtt': False, 'auto': False}
    
    # Initialize tab content containers
    scheduled_content = None
    gtt_content = None
    auto_content = None

    # Modern page header with glassmorphism design
    with ui.card().classes("w-full shadow-lg").style(
        "background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%); "
        "backdrop-filter: blur(20px); "
        "border: 1px solid rgba(255, 255, 255, 0.2); "
        "border-radius: 16px; "
        "overflow: hidden;"
    ):
        with ui.row().classes("w-full justify-between items-center p-4"):
            # Left - Enhanced page title with gradient effect
            with ui.column().classes("gap-1"):
                ui.label("Order Book").classes("theme-header-text text-2xl font-bold")
                ui.label(f"Trading with {broker} • Real-time order management").classes(
                    "text-sm theme-text-secondary opacity-80"
                )
            
            # Right - Modern tabs with enhanced styling
            with ui.tabs().classes(
                "modern-tabs bg-transparent shadow-sm"
            ).props(
                'indicator-color="primary" active-color="primary" dense no-caps align="right" '
                'class="modern-order-tabs"'
            ).style(
                "background: rgba(0, 0, 0, 0.1); "
                "border-radius: 12px; "
                "padding: 4px; "
                "backdrop-filter: blur(10px);"
            ) as tabs:
                placed_tab = ui.tab('placed', label='📋 Placed', icon='receipt_long').classes(
                    'modern-tab-item px-4 py-2 text-weight-medium text-xs'
                ).style(
                    "border-radius: 8px; "
                    "transition: all 0.3s ease; "
                    "min-width: 100px;"
                )
                scheduled_tab = ui.tab('scheduled', label='⏰ Scheduled', icon='schedule').classes(
                    'modern-tab-item px-4 py-2 text-weight-medium text-xs'
                ).style(
                    "border-radius: 8px; "
                    "transition: all 0.3s ease; "
                    "min-width: 100px;"
                )
                gtt_tab = ui.tab('gtt', label='🎯 GTT', icon='gps_fixed').classes(
                    'modern-tab-item px-4 py-2 text-weight-medium text-xs'
                ).style(
                    "border-radius: 8px; "
                    "transition: all 0.3s ease; "
                    "min-width: 100px;"
                )
                auto_tab = ui.tab('auto', label='🤖 Auto', icon='smart_toy').classes(
                    'modern-tab-item px-4 py-2 text-weight-medium text-xs'
                ).style(
                    "border-radius: 8px; "
                    "transition: all 0.3s ease; "
                    "min-width: 100px;"
                )
    
    # Tab panels with proper lazy loading
    with ui.tab_panels(tabs, value='placed').classes('w-full').style('background: transparent; border: none;'):
        with ui.tab_panel(placed_tab):
            # Load placed orders by default when page opens
            await render_placed_orders(fetch_api, user_storage, broker)
        
        with ui.tab_panel(scheduled_tab):
            # Create container for scheduled orders content
            scheduled_content = ui.column().classes('w-full')
            with scheduled_content:
                with ui.row().classes('w-full justify-center items-center p-8'):
                    ui.icon('schedule', size='2rem', color='grey-5').classes('mb-2')
                    ui.label('Scheduled orders will load when this tab is selected').classes('text-center text-grey-6')
        
        with ui.tab_panel(gtt_tab):
            # Create container for GTT orders content
            gtt_content = ui.column().classes('w-full')
            with gtt_content:
                with ui.row().classes('w-full justify-center items-center p-8'):
                    ui.icon('gps_fixed', size='2rem', color='grey-5').classes('mb-2')
                    ui.label('GTT orders will load when this tab is selected').classes('text-center text-grey-6')
        
        with ui.tab_panel(auto_tab):
            # Create container for auto orders content
            auto_content = ui.column().classes('w-full')
            with auto_content:
                with ui.row().classes('w-full justify-center items-center p-8'):
                    ui.icon('smart_toy', size='2rem', color='grey-5').classes('mb-2')
                    ui.label('Auto orders will load when this tab is selected').classes('text-center text-grey-6')
    
    # Handle tab changes for lazy loading with proper UI context
    async def handle_tab_change(e):
        # Normalize event payload to a tab name string
        raw = getattr(e, 'args', None)
        tab_name = None
        if isinstance(raw, str):
            tab_name = raw
        elif isinstance(raw, dict):
            tab_name = raw.get('value') or raw.get('name') or raw.get('args')
        else:
            # Try attribute-based access (e.g., element with .value)
            tab_name = getattr(raw, 'value', None)
            if not isinstance(tab_name, str) and raw is not None:
                # Fallback to string representation parsing
                raw_str = str(raw)
                for candidate in ('placed', 'scheduled', 'gtt', 'auto'):
                    if candidate in raw_str:
                        tab_name = candidate
                        break

        try:
            if tab_name == 'scheduled':
                if not loaded_tabs['scheduled']:
                    loaded_tabs['scheduled'] = True
                    # Show loading state
                    scheduled_content.clear()
                    with scheduled_content:
                        with ui.row().classes('w-full justify-center items-center p-8'):
                            ui.spinner('dots', size='lg', color='primary')
                            ui.label('Loading scheduled orders...').classes('ml-3 text-grey-6')
                    
                    # Small delay to show loading state and let session settle
                    await asyncio.sleep(0.3)
                    
                    # Load content in the correct UI context
                    scheduled_content.clear()
                    with scheduled_content:
                        await render_scheduled_orders(fetch_api, user_storage, broker)
                else:
                    # Refresh existing content
                    create_safe_task(refresh_scheduled_orders_with_pagination(fetch_api, broker))
                    
            elif tab_name == 'gtt':
                if not loaded_tabs['gtt']:
                    loaded_tabs['gtt'] = True
                    # Show loading state
                    gtt_content.clear()
                    with gtt_content:
                        with ui.row().classes('w-full justify-center items-center p-8'):
                            ui.spinner('dots', size='lg', color='primary')
                            ui.label('Loading GTT orders...').classes('ml-3 text-grey-6')
                    
                    # Small delay to show loading state and let session settle
                    await asyncio.sleep(0.3)
                    
                    # Load content in the correct UI context
                    gtt_content.clear()
                    with gtt_content:
                        await render_gtt_orders(fetch_api, user_storage, broker)
                else:
                    # Refresh existing content
                    create_safe_task(refresh_gtt_orders_with_pagination(fetch_api, broker))

            elif tab_name == 'auto':
                if not loaded_tabs['auto']:
                    loaded_tabs['auto'] = True
                    # Show loading state
                    auto_content.clear()
                    with auto_content:
                        with ui.row().classes('w-full justify-center items-center p-8'):
                            ui.spinner('dots', size='lg', color='primary')
                            ui.label('Loading auto orders...').classes('ml-3 text-grey-6')

                    # Small delay to show loading state and let session settle
                    await asyncio.sleep(0.3)

                    # Load content in the correct UI context
                    auto_content.clear()
                    with auto_content:
                        await render_auto_orders(fetch_api, user_storage, broker)
                else:
                    # Refresh existing content
                    create_safe_task(refresh_auto_orders_with_pagination(fetch_api, broker))
        except Exception as e:
            logger.error(f"Error in handle_tab_change: {str(e)}")
            safe_notify(f"Error loading {tab_name} orders: {str(e)}", notify_type="negative")

    # Set up the tab change handler (both kebab-case and camelCase variants)
    tabs.on('update:model-value', handle_tab_change)
    tabs.on('update:modelValue', handle_tab_change)





