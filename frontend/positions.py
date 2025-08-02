# Enhanced Positions Module - positions.py
# Applying beautiful dashboard.py styling consistently

from nicegui import ui
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


async def render_positions_page(fetch_api, user_storage, broker):
    """Enhanced positions page with beautiful dashboard styling"""
    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):
        # Enhanced title section (matching dashboard.py)
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("donut_small", size="2rem").classes("text-cyan-400")
                    ui.label(f"Open Positions - {broker}").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Monitor your active trading positions and P&L in real-time").classes(
                    "text-gray-400 dashboard-subtitle")

            # Right side - Action buttons
            with ui.row().classes("items-center gap-4"):
                ui.button("Close All", icon="close", color="red").classes("text-white")
                ui.button("Export Report", icon="download").classes("text-cyan-400")
                ui.button("Refresh", icon="refresh").classes("text-gray-400")

        # Positions summary cards
        await render_enhanced_positions_summary(fetch_api, user_storage, broker)

        # Main positions content
        with ui.card().classes("dashboard-card w-full m-4"):
            with ui.row().classes("card-header w-full justify-between items-center p-4"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("trending_up", size="1.5rem").classes("text-green-400")
                    ui.label("Active Positions").classes("card-title")

                    # Live update indicator
                    with ui.row().classes("items-center gap-1 ml-2"):
                        ui.element('div').classes("w-2 h-2 bg-green-400 rounded-full animate-pulse")
                        ui.label("Live P&L").classes("text-xs text-green-400")

                with ui.row().classes("items-center gap-2"):
                    # Filter buttons
                    ui.button("All", on_click=lambda: filter_positions("all")).props("flat").classes("text-cyan-400")
                    ui.button("Profitable", on_click=lambda: filter_positions("profit")).props("flat").classes(
                        "text-green-400")
                    ui.button("Loss", on_click=lambda: filter_positions("loss")).props("flat").classes("text-red-400")

            ui.separator().classes("card-separator")

            # Positions table container
            positions_container = ui.column().classes("w-full p-4")

            # Render positions table
            await render_enhanced_positions_table(fetch_api, user_storage, broker, positions_container)


async def render_enhanced_positions_summary(fetch_api, user_storage, broker):
    """Enhanced positions summary metrics"""

    with ui.row().classes("w-full gap-4 p-4"):
        try:
            # Fetch positions data
            positions_data = await fetch_api(f"/positions/{broker}")

            if positions_data and isinstance(positions_data, list):
                # Calculate summary metrics
                total_positions = len(positions_data)
                profitable_positions = len([p for p in positions_data if float(p.get('PnL', 0)) > 0])
                loss_positions = len([p for p in positions_data if float(p.get('PnL', 0)) < 0])

                # Calculate total P&L
                total_pnl = sum(float(p.get('PnL', 0)) for p in positions_data)
                total_investment = sum(
                    float(p.get('AvgPrice', 0)) * float(p.get('Quantity', 0)) for p in positions_data)

                # Calculate unrealized P&L percentage
                pnl_percentage = (total_pnl / total_investment * 100) if total_investment > 0 else 0
            else:
                total_positions = profitable_positions = loss_positions = 0
                total_pnl = total_investment = pnl_percentage = 0

            # Total Positions
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("account_tree", size="2rem").classes("text-blue-400 mb-2")
                    ui.label("Total Positions").classes("text-sm text-gray-400")
                    ui.label(str(total_positions)).classes("text-2xl font-bold text-white")

            # Profitable Positions
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("trending_up", size="2rem").classes("text-green-400 mb-2")
                    ui.label("Profitable").classes("text-sm text-gray-400")
                    ui.label(str(profitable_positions)).classes("text-2xl font-bold text-green-400")

            # Loss Positions
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("trending_down", size="2rem").classes("text-red-400 mb-2")
                    ui.label("In Loss").classes("text-sm text-gray-400")
                    ui.label(str(loss_positions)).classes("text-2xl font-bold text-red-400")

            # Total P&L
            pnl_color = "text-green-400" if total_pnl >= 0 else "text-red-400"
            pnl_icon = "trending_up" if total_pnl >= 0 else "trending_down"
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon(pnl_icon, size="2rem").classes(f"{pnl_color} mb-2")
                    ui.label("Unrealized P&L").classes("text-sm text-gray-400")
                    ui.label(f"₹{total_pnl:,.2f}").classes(f"text-2xl font-bold {pnl_color}")
                    ui.label(f"({pnl_percentage:+.2f}%)").classes(f"text-sm {pnl_color}")

            # Investment Value
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("currency_rupee", size="2rem").classes("text-purple-400 mb-2")
                    ui.label("Investment").classes("text-sm text-gray-400")
                    ui.label(f"₹{total_investment:,.0f}").classes("text-2xl font-bold text-white")

        except Exception as e:
            logger.error(f"Error fetching positions summary: {e}")
            with ui.card().classes("dashboard-card w-full"):
                ui.label("Error loading positions summary").classes("text-red-500 text-center p-4")


async def render_enhanced_positions_table(fetch_api, user_storage, broker, container):
    """Enhanced positions table with beautiful styling"""

    try:
        # Fetch positions data
        positions_data = await fetch_api(f"/positions/{broker}")

        if not positions_data:
            # Enhanced empty state
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("donut_small", size="4rem").classes("text-gray-500 mb-4")
                    ui.label("No open positions").classes("text-xl text-gray-400 mb-2")
                    ui.label("Your active positions will appear here after you place trades").classes(
                        "text-sm text-gray-500")
                    ui.button("Start Trading", icon="add",
                              on_click=lambda: ui.navigate.to('/order-management')).classes("mt-4")
            return

        with container:
            # Enhanced table header
            with ui.row().classes(
                    "positions-header w-full p-3 text-sm font-semibold text-gray-400 border-b border-gray-700 mb-2"):
                ui.label("Symbol").classes("w-32")
                ui.label("Product").classes("w-24")
                ui.label("Quantity").classes("w-24 text-right")
                ui.label("Avg Price").classes("w-24 text-right")
                ui.label("LTP").classes("w-24 text-right")
                ui.label("P&L").classes("w-32 text-right")
                ui.label("P&L %").classes("w-24 text-right")
                ui.label("Day P&L").classes("w-32 text-right")
                ui.label("Actions").classes("w-24 text-center")

            # Render position rows
            for position in positions_data:
                await render_enhanced_position_row(position, broker, fetch_api)

    except Exception as e:
        logger.error(f"Error rendering positions table: {e}")
        with container:
            with ui.column().classes("w-full text-center p-8"):
                ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                ui.label("Error loading positions").classes("text-xl text-red-400 mb-2")
                ui.label(str(e)).classes("text-sm text-gray-500")


async def render_enhanced_position_row(position, broker, fetch_api):
    """Render individual enhanced position row"""

    try:
        # Extract position details
        symbol = position.get('Symbol', 'N/A')
        product = position.get('Product', 'N/A')
        quantity = float(position.get('Quantity', 0))
        avg_price = float(position.get('AvgPrice', 0))
        ltp = float(position.get('LastPrice', 0))
        pnl = float(position.get('PnL', 0))

        # Calculate P&L percentage
        pnl_percentage = ((ltp - avg_price) / avg_price * 100) if avg_price > 0 else 0

        # Calculate day P&L (this would typically come from API)
        day_pnl = pnl * 0.3  # Placeholder calculation

        # Determine styling based on P&L
        if pnl > 0:
            pnl_color = "text-green-400"
            border_color = "border-green-500/20"
            bg_hover = "hover:bg-green-900/10"
            trend_icon = "trending_up"
        elif pnl < 0:
            pnl_color = "text-red-400"
            border_color = "border-red-500/20"
            bg_hover = "hover:bg-red-900/10"
            trend_icon = "trending_down"
        else:
            pnl_color = "text-gray-400"
            border_color = "border-gray-500/20"
            bg_hover = "hover:bg-gray-800/50"
            trend_icon = "trending_flat"

        # Determine quantity color (long/short)
        qty_color = "text-green-400" if quantity > 0 else "text-red-400"

        # Render position row
        with ui.row().classes(
                f"position-row w-full p-3 {bg_hover} transition-all duration-200 border-l-2 {border_color} mb-1 rounded-r-lg"):
            # Symbol
            with ui.column().classes("w-32"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon(trend_icon, size="1rem").classes(pnl_color)
                    ui.label(symbol).classes("text-white font-semibold")
                ui.label(position.get('Exchange', 'NSE')).classes("text-xs text-gray-500")

            # Product
            ui.label(product).classes("w-24 text-gray-300 text-sm")

            # Quantity
            ui.label(f"{quantity:,.0f}").classes(f"w-24 text-right {qty_color} font-mono font-semibold")

            # Average Price
            ui.label(f"₹{avg_price:,.2f}").classes("w-24 text-right text-gray-300 font-mono")

            # LTP
            ui.label(f"₹{ltp:,.2f}").classes("w-24 text-right text-white font-mono font-semibold")

            # P&L
            with ui.column().classes("w-32 text-right"):
                ui.label(f"₹{pnl:,.2f}").classes(f"{pnl_color} font-semibold font-mono")

            # P&L %
            ui.label(f"{pnl_percentage:+.2f}%").classes(f"w-24 text-right {pnl_color} font-mono text-sm")

            # Day P&L
            day_pnl_color = "text-green-400" if day_pnl >= 0 else "text-red-400"
            with ui.column().classes("w-32 text-right"):
                ui.label(f"₹{day_pnl:,.2f}").classes(f"{day_pnl_color} font-mono text-sm")

            # Actions
            with ui.row().classes("w-24 justify-center gap-1"):
                ui.button(
                    icon="close",
                    on_click=lambda s=symbol, q=quantity: close_position(s, q, broker, fetch_api)
                ).props("flat round size=sm").classes("text-red-400")
                ui.button(
                    icon="info",
                    on_click=lambda p=position: show_position_details(p)
                ).props("flat round size=sm").classes("text-cyan-400")

    except Exception as e:
        logger.error(f"Error rendering position row: {e}")
        with ui.row().classes("position-row w-full p-3 border-l-2 border-red-500/20"):
            ui.label("Error loading position").classes("text-red-400")


def filter_positions(filter_type):
    """Filter positions by type"""
    ui.notify(f"Filtering positions: {filter_type}", type="info")
    # This would trigger a re-render with filtered data
    # Implementation would depend on your state management


async def close_position(symbol, quantity, broker, fetch_api):
    """Close a position by placing opposite order"""
    try:
        # Determine opposite transaction type
        transaction_type = "SELL" if quantity > 0 else "BUY"
        abs_quantity = abs(quantity)

        # Show confirmation dialog
        with ui.dialog() as dialog, ui.card().classes("dashboard-card"):
            with ui.column().classes("p-4 gap-4"):
                ui.label("Close Position").classes("text-xl font-bold text-white")
                ui.label(f"Are you sure you want to close your {symbol} position?").classes("text-gray-300")

                with ui.column().classes("gap-2 p-2 bg-gray-800/50 rounded-lg"):
                    ui.label(f"Symbol: {symbol}").classes("text-white")
                    ui.label(f"Quantity: {abs_quantity:,}").classes("text-white")
                    ui.label(f"Action: {transaction_type}").classes("text-white")

                with ui.row().classes("gap-2 justify-end w-full"):
                    ui.button("Cancel", on_click=dialog.close).classes("text-gray-400")
                    ui.button(
                        "Close Position",
                        color="red",
                        on_click=lambda: execute_close_position(dialog, symbol, transaction_type, abs_quantity, broker,
                                                                fetch_api)
                    ).classes("text-white")

        dialog.open()

    except Exception as e:
        logger.error(f"Error initiating position close: {e}")
        ui.notify(f"Error: {str(e)}", type="negative")


async def execute_close_position(dialog, symbol, transaction_type, quantity, broker, fetch_api):
    """Execute the close position order"""
    try:
        order_data = {
            "trading_symbol": symbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": "MARKET",
            "product_type": "CNC"
        }

        response = await fetch_api(f"/orders/{broker}/place", method="POST", data=order_data)

        if response and response.get('status') == 'success':
            ui.notify(f"Position close order placed successfully", type="positive")
            dialog.close()
            # Refresh positions
            ui.navigate.to('/positions')
        else:
            ui.notify("Failed to place close order", type="negative")
            dialog.close()

    except Exception as e:
        logger.error(f"Error executing close position: {e}")
        ui.notify(f"Error placing order: {str(e)}", type="negative")
        dialog.close()


def show_position_details(position):
    """Show detailed position information"""
    with ui.dialog() as dialog, ui.card().classes("dashboard-card"):
        with ui.column().classes("p-4 gap-4"):
            ui.label("Position Details").classes("text-xl font-bold text-white")

            with ui.column().classes("gap-2"):
                ui.label(f"Symbol: {position.get('Symbol', 'N/A')}").classes("text-white font-semibold")
                ui.label(f"Exchange: {position.get('Exchange', 'N/A')}").classes("text-gray-300")
                ui.label(f"Product: {position.get('Product', 'N/A')}").classes("text-gray-300")
                ui.label(f"Quantity: {position.get('Quantity', 0):,}").classes("text-white")
                ui.label(f"Average Price: ₹{float(position.get('AvgPrice', 0)):,.2f}").classes("text-white text-mono")
                ui.label(f"Last Price: ₹{float(position.get('LastPrice', 0)):,.2f}").classes("text-white text-mono")

                pnl = float(position.get('PnL', 0))
                pnl_color = "text-green-400" if pnl >= 0 else "text-red-400"
                ui.label(f"Unrealized P&L: ₹{pnl:,.2f}").classes(f"{pnl_color} font-semibold text-mono")

                # Calculate additional metrics
                investment_value = float(position.get('AvgPrice', 0)) * float(position.get('Quantity', 0))
                current_value = float(position.get('LastPrice', 0)) * float(position.get('Quantity', 0))

                ui.label(f"Investment Value: ₹{investment_value:,.2f}").classes("text-gray-300 text-mono")
                ui.label(f"Current Value: ₹{current_value:,.2f}").classes("text-white text-mono")

            ui.button("Close", on_click=dialog.close).classes("self-end")

    dialog.open()