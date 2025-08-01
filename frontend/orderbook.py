# Enhanced Order Book Module - orderbook.py
# Applying beautiful dashboard.py styling consistently

from nicegui import ui
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

def apply_enhanced_dashboard_styles():
    """Apply enhanced CSS styles matching dashboard.py"""
    ui.add_css('static/styles.css')

async def render_order_book_page(fetch_api, user_storage, broker):
    """Enhanced order book page with beautiful dashboard styling"""

    apply_enhanced_dashboard_styles()

    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):

        # Enhanced title section (matching dashboard.py)
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("receipt_long", size="2rem").classes("text-cyan-400")
                    ui.label(f"Order Book - {broker}").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Track your orders and execution status in real-time").classes("text-gray-400 dashboard-subtitle")

            # Right side - Action buttons
            with ui.row().classes("items-center gap-4"):
                ui.button("Export Orders", icon="download").classes("text-cyan-400")
                ui.button("Refresh", icon="refresh").classes("text-gray-400")

        # Order summary cards
        await render_enhanced_order_summary(fetch_api, user_storage, broker)

        # Main order book content
        with ui.card().classes("dashboard-card w-full m-4"):
            with ui.row().classes("card-header w-full justify-between items-center p-4"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("list_alt", size="1.5rem").classes("text-orange-400")
                    ui.label("All Orders").classes("card-title")

                with ui.row().classes("items-center gap-2"):
                    # Filter buttons
                    ui.button("All", on_click=lambda: filter_orders("all")).props("flat").classes("text-cyan-400")
                    ui.button("Open", on_click=lambda: filter_orders("open")).props("flat").classes("text-yellow-400")
                    ui.button("Executed", on_click=lambda: filter_orders("executed")).props("flat").classes("text-green-400")
                    ui.button("Cancelled", on_click=lambda: filter_orders("cancelled")).props("flat").classes("text-red-400")

            ui.separator().classes("card-separator")

            # Orders table container
            orders_container = ui.column().classes("w-full p-4")

            # Render orders table
            await render_enhanced_orders_table(fetch_api, user_storage, broker, orders_container)

async def render_enhanced_order_summary(fetch_api, user_storage, broker):
    """Enhanced order summary metrics"""

    with ui.row().classes("w-full gap-4 p-4"):
        try:
            # Fetch order book data
            orders_data = await fetch_api(f"/order-book/{broker}")

            if orders_data and isinstance(orders_data, list):
                # Calculate summary metrics
                total_orders = len(orders_data)
                open_orders = len([o for o in orders_data if o.get('status', '').upper() in ['OPEN', 'PENDING']])
                executed_orders = len([o for o in orders_data if o.get('status', '').upper() in ['COMPLETE', 'EXECUTED']])
                cancelled_orders = len([o for o in orders_data if o.get('status', '').upper() in ['CANCELLED', 'REJECTED']])

                # Calculate total order value
                total_value = sum(float(o.get('price', 0)) * float(o.get('quantity', 0)) for o in orders_data)
            else:
                total_orders = open_orders = executed_orders = cancelled_orders = total_value = 0

            # Total Orders
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("receipt", size="2rem").classes("text-blue-400 mb-2")
                    ui.label("Total Orders").classes("text-sm text-gray-400")
                    ui.label(str(total_orders)).classes("text-2xl font-bold text-white")

            # Open Orders
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("schedule", size="2rem").classes("text-yellow-400 mb-2")
                    ui.label("Open Orders").classes("text-sm text-gray-400")
                    ui.label(str(open_orders)).classes("text-2xl font-bold text-yellow-400")

            # Executed Orders
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("check_circle", size="2rem").classes("text-green-400 mb-2")
                    ui.label("Executed").classes("text-sm text-gray-400")
                    ui.label(str(executed_orders)).classes("text-2xl font-bold text-green-400")

            # Cancelled Orders
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("cancel", size="2rem").classes("text-red-400 mb-2")
                    ui.label("Cancelled").classes("text-sm text-gray-400")
                    ui.label(str(cancelled_orders)).classes("text-2xl font-bold text-red-400")

            # Total Value
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("currency_rupee", size="2rem").classes("text-purple-400 mb-2")
                    ui.label("Total Value").classes("text-sm text-gray-400")
                    ui.label(f"₹{total_value:,.0f}").classes("text-2xl font-bold text-white")

        except Exception as e:
            logger.error(f"Error fetching order summary: {e}")
            with ui.card().classes("dashboard-card w-full"):
                ui.label("Error loading order summary").classes("text-red-500 text-center p-4")

async def render_enhanced_orders_table(fetch_api, user_storage, broker, container):
    """Enhanced orders table with beautiful styling"""

    try:
        # Fetch orders data
        orders_data = await fetch_api(f"/order-book/{broker}")

        if not orders_data:
            # Enhanced empty state
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("inbox", size="4rem").classes("text-gray-500 mb-4")
                    ui.label("No orders found").classes("text-xl text-gray-400 mb-2")
                    ui.label("Place your first order to see it here").classes("text-sm text-gray-500")
                    ui.button("Place Order", icon="add", on_click=lambda: ui.navigate.to('/order-management')).classes("mt-4")
            return

        with container:
            # Enhanced table header
            with ui.row().classes("orders-header w-full p-3 text-sm font-semibold text-gray-400 border-b border-gray-700 mb-2"):
                ui.label("Order ID").classes("w-32")
                ui.label("Symbol").classes("w-32")
                ui.label("Type").classes("w-20")
                ui.label("Side").classes("w-20")
                ui.label("Quantity").classes("w-24 text-right")
                ui.label("Price").classes("w-24 text-right")
                ui.label("Status").classes("w-24 text-center")
                ui.label("Time").classes("w-32")
                ui.label("Actions").classes("w-24 text-center")

            # Render order rows
            for order in orders_data:
                await render_enhanced_order_row(order, broker, fetch_api)

    except Exception as e:
        logger.error(f"Error rendering orders table: {e}")
        with container:
            with ui.column().classes("w-full text-center p-8"):
                ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                ui.label("Error loading orders").classes("text-xl text-red-400 mb-2")
                ui.label(str(e)).classes("text-sm text-gray-500")

async def render_enhanced_order_row(order, broker, fetch_api):
    """Render individual enhanced order row"""

    try:
        # Extract order details
        order_id = order.get('order_id', 'N/A')
        symbol = order.get('trading_symbol', 'N/A')
        order_type = order.get('order_type', 'N/A')
        transaction_type = order.get('transaction_type', 'N/A')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0)
        status = order.get('status', 'UNKNOWN').upper()
        order_time = order.get('order_timestamp', 'N/A')

        # Format time
        if order_time != 'N/A':
            try:
                if isinstance(order_time, str):
                    dt = datetime.fromisoformat(order_time.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%H:%M:%S')
                else:
                    formatted_time = str(order_time)
            except:
                formatted_time = str(order_time)
        else:
            formatted_time = 'N/A'

        # Determine status styling
        if status in ['COMPLETE', 'EXECUTED']:
            status_color = "text-green-400"
            status_bg = "bg-green-900/20"
            border_color = "border-green-500/20"
        elif status in ['OPEN', 'PENDING']:
            status_color = "text-yellow-400"
            status_bg = "bg-yellow-900/20"
            border_color = "border-yellow-500/20"
        elif status in ['CANCELLED', 'REJECTED']:
            status_color = "text-red-400"
            status_bg = "bg-red-900/20"
            border_color = "border-red-500/20"
        else:
            status_color = "text-gray-400"
            status_bg = "bg-gray-900/20"
            border_color = "border-gray-500/20"

        # Determine side styling
        side_color = "text-green-400" if transaction_type.upper() == "BUY" else "text-red-400"

        # Render order row
        with ui.row().classes(f"order-row w-full p-3 hover:bg-gray-800/50 transition-all duration-200 border-l-2 {border_color} mb-1 rounded-r-lg"):
            # Order ID
            ui.label(order_id[:8] + "..." if len(order_id) > 8 else order_id).classes("w-32 text-gray-300 text-mono text-sm")

            # Symbol
            ui.label(symbol).classes("w-32 text-white font-semibold")

            # Order Type
            ui.label(order_type).classes("w-20 text-gray-300 text-sm")

            # Side
            ui.label(transaction_type).classes(f"w-20 {side_color} font-semibold text-sm")

            # Quantity
            ui.label(f"{quantity:,}").classes("w-24 text-right text-white text-mono")

            # Price
            ui.label(f"₹{price:,.2f}").classes("w-24 text-right text-white text-mono")

            # Status
            with ui.row().classes("w-24 justify-center"):
                ui.chip(status, color=None).classes(f"{status_color} {status_bg} text-xs")

            # Time
            ui.label(formatted_time).classes("w-32 text-gray-400 text-sm text-mono")

            # Actions
            with ui.row().classes("w-24 justify-center gap-1"):
                if status in ['OPEN', 'PENDING']:
                    ui.button(
                        icon="cancel",
                        on_click=lambda oid=order_id: cancel_order(oid, broker, fetch_api)
                    ).props("flat round size=sm").classes("text-red-400")

                ui.button(
                    icon="info",
                    on_click=lambda o=order: show_order_details(o)
                ).props("flat round size=sm").classes("text-cyan-400")

    except Exception as e:
        logger.error(f"Error rendering order row: {e}")
        with ui.row().classes("order-row w-full p-3 border-l-2 border-red-500/20"):
            ui.label("Error loading order").classes("text-red-400")

def filter_orders(filter_type):
    """Filter orders by type"""
    ui.notify(f"Filtering orders: {filter_type}", type="info")
    # This would trigger a re-render with filtered data
    # Implementation would depend on your state management

async def cancel_order(order_id, broker, fetch_api):
    """Cancel an order"""
    try:
        response = await fetch_api(f"/orders/{broker}/cancel", method="POST", data={"order_id": order_id})
        if response:
            ui.notify(f"Order {order_id[:8]}... cancelled successfully", type="positive")
            # Refresh the page
            ui.navigate.to('/order-book')
        else:
            ui.notify("Failed to cancel order", type="negative")
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        ui.notify(f"Error cancelling order: {str(e)}", type="negative")

def show_order_details(order):
    """Show detailed order information"""
    with ui.dialog() as dialog, ui.card().classes("dashboard-card"):
        with ui.column().classes("p-4 gap-4"):
            ui.label("Order Details").classes("text-xl font-bold text-white")

            with ui.column().classes("gap-2"):
                ui.label(f"Order ID: {order.get('order_id', 'N/A')}").classes("text-gray-300")
                ui.label(f"Symbol: {order.get('trading_symbol', 'N/A')}").classes("text-white font-semibold")
                ui.label(f"Type: {order.get('order_type', 'N/A')}").classes("text-gray-300")
                ui.label(f"Side: {order.get('transaction_type', 'N/A')}").classes("text-white")
                ui.label(f"Quantity: {order.get('quantity', 0):,}").classes("text-gray-300")
                ui.label(f"Price: ₹{order.get('price', 0):,.2f}").classes("text-white text-mono")
                ui.label(f"Status: {order.get('status', 'UNKNOWN')}").classes("text-white")

                if order.get('filled_quantity', 0) > 0:
                    ui.label(f"Filled Quantity: {order.get('filled_quantity', 0):,}").classes("text-green-400")
                    ui.label(f"Average Price: ₹{order.get('average_price', 0):,.2f}").classes("text-green-400 text-mono")

            ui.button("Close", on_click=dialog.close).classes("self-end")

    dialog.open()