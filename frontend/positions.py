# Enhanced Positions Module - positions.py
# Applying beautiful dashboard.py styling consistently

from nicegui import ui
import logging
import pandas as pd
from datetime import datetime
from cache_manager import frontend_cache, FrontendCacheConfig, TradingDataCache
from cache_invalidation import invalidate_on_position_change

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
                    ui.icon("donut_small", size="2rem").classes("theme-text-accent")
                    # Create dynamic title that updates with broker changes
                    title_label = ui.label(f"Open Positions - {broker}").classes("text-3xl font-bold theme-text-primary dashboard-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")
                    
                    # Update title when broker changes
                    def update_title():
                        current_broker = user_storage.get('default_broker', broker)
                        title_label.text = f"Open Positions - {current_broker}"
                        title_label.update()
                    
                    # Monitor broker changes for title update
                    from app_ui import create_broker_monitor
                    create_broker_monitor(update_title)

                ui.label("Monitor your active trading positions and P&L in real-time").classes(
                    "theme-text-secondary dashboard-subtitle")

            # Right side - Action buttons
            with ui.row().classes("items-center gap-4"):
                ui.button("Close All", icon="close", color="red")
                ui.button("Export Report", icon="download").classes("theme-text-accent")
                ui.button("Refresh", icon="refresh").classes("theme-text-secondary")

        # Positions summary cards
        summary_container = ui.column().classes("w-full")
        await render_enhanced_positions_summary(fetch_api, user_storage, broker, summary_container)

        # Main positions content
        with ui.card().classes("dashboard-card w-full"):
            with ui.row().classes("card-header w-full justify-between items-center p-4"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("trending_up", size="1.5rem").classes("theme-text-success")
                    ui.label("Active Positions").classes("card-title")

                    # Live update indicator
                    with ui.row().classes("items-center gap-1 ml-2"):
                        ui.element('div').classes("w-2 h-2 theme-bg-success rounded-full animate-pulse")
                        ui.label("Live P&L").classes("text-xs theme-text-success")

                with ui.row().classes("items-center gap-2"):
                    # Filter buttons
                    ui.button("All", on_click=lambda: filter_positions("all")).props("flat").classes("theme-text-accent")
                    ui.button("Profitable", on_click=lambda: filter_positions("profit")).props("flat").classes(
                        "theme-text-success")
                    ui.button("Loss", on_click=lambda: filter_positions("loss")).props("flat").classes("theme-text-error")

            ui.separator().classes("card-separator")

            # Positions table container
            positions_container = ui.column().classes("w-full p-4")

            # Render positions table
            await render_enhanced_positions_table(fetch_api, user_storage, broker, positions_container)
            
            # Monitor broker changes and refresh positions data
            from app_ui import create_broker_monitor
            async def refresh_positions():
                # Get current broker from storage
                current_broker = user_storage.get('default_broker', broker)
                # Clear both containers to avoid duplicates
                summary_container.clear()
                positions_container.clear()
                # Re-render both summary and table with new broker
                await render_enhanced_positions_summary(fetch_api, user_storage, current_broker, summary_container)
                await render_enhanced_positions_table(fetch_api, user_storage, current_broker, positions_container)
            create_broker_monitor(refresh_positions)


async def render_enhanced_positions_summary(fetch_api, user_storage, broker, container=None):
    """Enhanced positions summary metrics"""
    
    if container:
        with container:
            await _render_positions_summary_content(fetch_api, user_storage, broker)
    else:
        await _render_positions_summary_content(fetch_api, user_storage, broker)

async def _render_positions_summary_content(fetch_api, user_storage, broker):
    """Render the actual positions summary content"""
    with ui.row().classes("w-full gap-4 p-4"):
        try:
            # Get current broker from storage
            current_broker = user_storage.get('default_broker', broker)
            
            # Fetch positions data with caching
            cache_key = frontend_cache.generate_cache_key("positions", current_broker)
            positions_data = frontend_cache.get(cache_key)
            
            if positions_data is None:
                positions_data = await fetch_api(f"/positions/{current_broker}")
                if positions_data:
                    frontend_cache.set(cache_key, positions_data, FrontendCacheConfig.POSITION_DATA)

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
                    ui.icon("account_tree", size="2rem").classes("theme-text-info mb-2")
                    ui.label("Total Positions").classes("text-sm theme-text-secondary")
                    ui.label(str(total_positions)).classes("text-2xl font-bold theme-text-primary")

            # Profitable Positions
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("trending_up", size="2rem").classes("theme-text-success mb-2")
                    ui.label("Profitable").classes("text-sm theme-text-secondary")
                    ui.label(str(profitable_positions)).classes("text-2xl font-bold theme-text-success")

            # Loss Positions
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("trending_down", size="2rem").classes("theme-text-error mb-2")
                    ui.label("In Loss").classes("text-sm theme-text-secondary")
                    ui.label(str(loss_positions)).classes("text-2xl font-bold theme-text-error")

            # Total P&L
            pnl_color = "theme-text-success" if total_pnl >= 0 else "theme-text-error"
            pnl_icon = "trending_up" if total_pnl >= 0 else "trending_down"
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon(pnl_icon, size="2rem").classes(f"{pnl_color} mb-2")
                    ui.label("Unrealized P&L").classes("text-sm theme-text-secondary")
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"₹{total_pnl:,.2f}").classes(f"text-2xl font-bold {pnl_color}")
                        ui.label(f"({pnl_percentage:+.2f}%)").classes(f"text-sm {pnl_color}")

            # Investment Value
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("currency_rupee", size="2rem").classes("theme-text-purple mb-2")
                    ui.label("Investment").classes("text-sm theme-text-secondary")
                    ui.label(f"₹{total_investment:,.0f}").classes("text-2xl font-bold theme-text-primary")

        except Exception as e:
            logger.error(f"Error fetching positions summary: {e}")
            with ui.card().classes("dashboard-card w-full"):
                ui.label("Error loading positions summary").classes("theme-text-error text-center p-4")


async def render_enhanced_positions_table(fetch_api, user_storage, broker, container):
    """Enhanced positions table with beautiful styling"""

    try:
        # Get current broker from storage
        current_broker = user_storage.get('default_broker', broker)
        
        # Fetch positions data with caching (reuse cached data from summary)
        cache_key = frontend_cache.generate_cache_key("positions", current_broker)
        positions_data = frontend_cache.get(cache_key)
        
        if positions_data is None:
            positions_data = await fetch_api(f"/positions/{current_broker}")
            if positions_data:
                frontend_cache.set(cache_key, positions_data, FrontendCacheConfig.POSITION_DATA)

        if not positions_data:
            # Enhanced empty state
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("donut_small", size="4rem").classes("theme-text-secondary mb-4")
                    ui.label("No open positions").classes("text-xl theme-text-secondary mb-2")
                    ui.label("Your active positions will appear here after you place trades").classes(
                        "text-sm theme-text-secondary")
                    ui.button("Start Trading", icon="add",
                              on_click=lambda: ui.navigate.to('/order-management')).classes("mt-4")
            return

        with container:
            # Enhanced table header using unified grid layout for perfect alignment
            with ui.element('div').classes("positions-header-wrapper w-full rounded-t-lg"):
                header_row = ui.element('div').classes(
                    "positions-header-grid text-xs font-semibold uppercase tracking-wide theme-text-secondary"
                )
                with header_row:
                    ui.label("Symbol")
                    ui.label("Product")
                    ui.label("Quantity").classes("text-right")
                    ui.label("Avg Price").classes("text-right")
                    ui.label("LTP").classes("text-right")
                    ui.label("P&L").classes("text-right")
                    ui.label("P&L %").classes("text-right")
                    ui.label("Day P&L").classes("text-right")
                    ui.label("Actions").classes("text-center")

            # Render position rows with the same grid template for consistent alignment
            for position in positions_data:
                await render_enhanced_position_row(position, broker, fetch_api)

    except Exception as e:
        logger.error(f"Error rendering positions table: {e}")
        with container:
            with ui.column().classes("w-full text-center p-8"):
                ui.icon("error", size="3rem").classes("theme-text-error mb-4")
                ui.label("Error loading positions").classes("text-xl theme-text-error mb-2")
                ui.label(str(e)).classes("text-sm theme-text-secondary")


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
        position_broker = position.get('Broker', broker)

        # Calculate P&L percentage
        pnl_percentage = ((ltp - avg_price) / avg_price * 100) if avg_price > 0 else 0

        # Calculate day P&L (this would typically come from API)
        day_pnl = float(position.get('DayPnL', pnl * 0.3))

        # Determine styling based on P&L
        if pnl > 0:
            pnl_color = "theme-text-success"
            trend_icon = "trending_up"
        elif pnl < 0:
            pnl_color = "theme-text-error"
            trend_icon = "trending_down"
        else:
            pnl_color = "theme-text-secondary"
            trend_icon = "trending_flat"

        # Determine quantity color (long/short)
        if quantity > 0:
            qty_color = "theme-text-success"
        elif quantity < 0:
            qty_color = "theme-text-error"
        else:
            qty_color = "theme-text-secondary"

        # Render position row
        position_type_class = "position-long" if quantity >= 0 else "position-short"
        row_wrapper_classes = (
            f"position-row {position_type_class} "
            "transition-all duration-200 rounded-lg w-full"
        )
        conversion_option = get_conversion_option(position_broker, product)

        with ui.element('div').classes(row_wrapper_classes):
            with ui.element('div').classes("positions-row-grid"):
                # Symbol
                with ui.column().classes("gap-1"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(trend_icon, size="1rem").classes(pnl_color)
                        ui.label(symbol).classes("theme-text-primary font-semibold")
                    ui.label(position.get('Exchange', 'NSE')).classes("text-xs theme-text-secondary")

                # Product
                ui.label(product).classes("theme-text-secondary text-sm")

                # Quantity
                ui.label(f"{quantity:,.0f}").classes(f"text-right {qty_color} font-mono font-semibold")

                # Average Price
                ui.label(f"₹{avg_price:,.2f}").classes("text-right theme-text-secondary font-mono")

                # LTP
                ui.label(f"₹{ltp:,.2f}").classes("text-right theme-text-primary font-mono font-semibold")

                # P&L
                ui.label(f"₹{pnl:,.2f}").classes(f"text-right {pnl_color} font-semibold font-mono w-full")

                # P&L %
                ui.label(f"{pnl_percentage:+.2f}%").classes(f"text-right {pnl_color} font-mono text-sm")

                # Day P&L
                if day_pnl > 0:
                    day_pnl_color = "theme-text-success"
                elif day_pnl < 0:
                    day_pnl_color = "theme-text-error"
                else:
                    day_pnl_color = "theme-text-secondary"
                ui.label(f"₹{day_pnl:,.2f}").classes(f"text-right {day_pnl_color} font-mono text-sm w-full")

                # Actions
                with ui.row().classes("justify-center gap-2"):
                    if conversion_option:
                        ui.button(
                            "Convert",
                            icon="swap_horiz",
                            on_click=lambda p=position, b=position_broker, option=conversion_option: show_convert_position_dialog(
                                p, b, option, fetch_api
                            ),
                        ).props("unelevated size=sm").classes("positions-action-btn")
                    ui.button(
                        "Exit Position",
                        icon="logout",
                        on_click=lambda s=symbol, q=quantity, b=position_broker: close_position(s, q, b, fetch_api)
                    ).props("unelevated size=sm").classes("positions-action-btn positions-action-btn--exit")

    except Exception as e:
        logger.error(f"Error rendering position row: {e}")
        with ui.row().classes("position-row w-full p-3"):
            ui.label("Error loading position").classes("theme-text-error")

def get_conversion_option(broker_name, product):
    """Determine available conversion target for a position"""
    broker_key = (broker_name or "").strip().lower()
    product_key = (product or "").strip().upper()

    zerodha_options = {
        "CNC": {
            "target": "MIS",
            "action_label": "Convert to MIS",
            "description": "Move this Zerodha position from delivery (CNC) to intraday (MIS).",
            "success_message": "Position converted to MIS successfully.",
        },
        "MIS": {
            "target": "CNC",
            "action_label": "Convert to CNC",
            "description": "Move this Zerodha position from intraday (MIS) to delivery (CNC).",
            "success_message": "Position converted to CNC successfully.",
        },
    }

    upstox_options = {
        "D": {
            "target": "I",
            "action_label": "Convert to Intraday",
            "description": "Move this Upstox position from delivery (D) to intraday (I).",
            "success_message": "Position converted to Intraday successfully.",
        },
        "I": {
            "target": "D",
            "action_label": "Convert to Delivery",
            "description": "Move this Upstox position from intraday (I) to delivery (D).",
            "success_message": "Position converted to Delivery successfully.",
        },
    }

    if broker_key == "zerodha":
        return zerodha_options.get(product_key)
    if broker_key == "upstox":
        return upstox_options.get(product_key)
    return None


async def show_convert_position_dialog(position, broker, conversion_option, fetch_api):
    """Display confirmation dialog for position conversion."""
    if not conversion_option:
        ui.notify("Conversion not available for this product.", type="warning")
        return

    try:
        quantity = abs(float(position.get('Quantity', 0)))
    except (TypeError, ValueError):
        quantity = 0

    if quantity <= 0:
        ui.notify("Conversion not available for zero quantity positions.", type="warning")
        return

    symbol = position.get('Symbol', 'N/A')
    current_product = position.get('Product', 'N/A')
    target_product = conversion_option["target"]

    with ui.dialog() as dialog, ui.card().classes("dashboard-card"):
        with ui.column().classes("p-4 gap-4"):
            ui.label("Convert Position").classes("text-xl font-bold theme-text-primary")
            ui.label(
                conversion_option.get(
                    "description",
                    f"Convert {symbol} from {current_product} to {target_product}?"
                )
            ).classes("theme-text-secondary")

            with ui.column().classes("gap-2 theme-surface-card rounded-lg p-3"):
                ui.label(f"Symbol: {symbol}").classes("theme-text-primary font-semibold")
                ui.label(f"Broker: {broker}").classes("theme-text-secondary text-sm")
                ui.label(f"Current Product: {current_product}").classes("theme-text-secondary text-sm")
                ui.label(f"Target Product: {target_product}").classes("theme-text-secondary text-sm")
                ui.label(f"Quantity: {quantity:,.0f}").classes("theme-text-secondary text-sm")

            with ui.row().classes("gap-2 justify-end"):
                ui.button("Cancel", on_click=dialog.close).classes("theme-text-secondary")
                ui.button(
                    conversion_option.get("action_label", f"Convert to {target_product}"),
                    color="cyan",
                    on_click=lambda opt=conversion_option, d=dialog, p=position, b=broker: execute_convert_position(
                        d, p, b, opt, fetch_api
                    )
                )

    dialog.open()


async def execute_convert_position(dialog, position, broker, conversion_option, fetch_api):
    """Execute the conversion and refresh the positions view."""
    try:
        quantity_raw = position.get('Quantity', 0)
        quantity = float(quantity_raw) if quantity_raw is not None else 0
        payload = {
            "symbol": position.get('Symbol'),
            "exchange": position.get('Exchange'),
            "current_product": position.get('Product'),
            "target_product": conversion_option["target"],
            "quantity": quantity,
            "instrument_token": position.get('InstrumentToken'),
            "transaction_type": "BUY" if quantity >= 0 else "SELL"
        }

        response = await fetch_api(f"/positions/{broker}/convert", method="POST", data=payload)

        if response and response.get('status') == 'success':
            invalidate_on_position_change(broker, 'current_user', position.get('Symbol'))
            ui.notify(
                conversion_option.get(
                    "success_message",
                    f"Position converted to {conversion_option['target']} successfully."
                ),
                type="positive"
            )
            dialog.close()
            ui.navigate.to('/positions')
        else:
            error_message = "Unknown error"
            if response:
                error = response.get('error')
                if isinstance(error, dict):
                    error_message = error.get('message', error_message)
                elif isinstance(error, str):
                    error_message = error
                else:
                    error_message = response.get('detail', error_message)
            ui.notify(f"Conversion failed: {error_message}", type="negative")
    except Exception as e:
        logger.error(f"Error converting position: {e}")
        ui.notify(f"Error converting position: {str(e)}", type="negative")


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
                ui.label("Close Position").classes("text-xl font-bold theme-text-primary")
                ui.label(f"Are you sure you want to close your {symbol} position?").classes("theme-text-secondary")

                with ui.column().classes("gap-2 theme-surface-card rounded-lg p-2"):
                    ui.label(f"Symbol: {symbol}").classes("theme-text-primary")
                    ui.label(f"Quantity: {abs_quantity:,}").classes("theme-text-primary")
                    ui.label(f"Action: {transaction_type}").classes("theme-text-primary")

                with ui.row().classes("gap-2 justify-end w-full"):
                    ui.button("Cancel", on_click=dialog.close).classes("theme-text-secondary")
                    ui.button(
                        "Close Position",
                        color="red",
                        on_click=lambda: execute_close_position(dialog, symbol, transaction_type, abs_quantity, broker,
                                                                fetch_api)
                    )

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
            # Use centralized cache invalidation for position changes
            # Note: In a real implementation, user_id should be passed from the calling context
            user_id = 'current_user'  # Placeholder - should be actual user ID
            invalidate_on_position_change(broker, user_id, symbol)
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
            ui.label("Position Details").classes("text-xl font-bold theme-text-primary")

            with ui.column().classes("gap-2"):
                ui.label(f"Symbol: {position.get('Symbol', 'N/A')}").classes("theme-text-primary font-semibold")
                ui.label(f"Exchange: {position.get('Exchange', 'N/A')}").classes("theme-text-secondary")
                ui.label(f"Product: {position.get('Product', 'N/A')}").classes("theme-text-secondary")
                ui.label(f"Quantity: {position.get('Quantity', 0):,}").classes("theme-text-primary")
                ui.label(f"Average Price: ₹{float(position.get('AvgPrice', 0)):,.2f}").classes("theme-text-primary text-mono")
                ui.label(f"Last Price: ₹{float(position.get('LastPrice', 0)):,.2f}").classes("theme-text-primary text-mono")

                pnl = float(position.get('PnL', 0))
                pnl_color = "theme-text-success" if pnl >= 0 else "theme-text-error"
                ui.label(f"Unrealized P&L: ₹{pnl:,.2f}").classes(f"{pnl_color} font-semibold text-mono")

                # Calculate additional metrics
                investment_value = float(position.get('AvgPrice', 0)) * float(position.get('Quantity', 0))
                current_value = float(position.get('LastPrice', 0)) * float(position.get('Quantity', 0))

                ui.label(f"Investment Value: ₹{investment_value:,.2f}").classes("theme-text-secondary text-mono")
                ui.label(f"Current Value: ₹{current_value:,.2f}").classes("theme-text-primary text-mono")

            ui.button("Close", on_click=dialog.close).classes("self-end")

    dialog.open()
