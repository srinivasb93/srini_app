"""
Enhanced Dashboard Module - dashboard.py
Integrates with existing modules and provides modern UI
"""

from nicegui import ui, app
import asyncio
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import existing modules to utilize their functionality
from analytics import render_analytics_page

# Import market utilities
from market_utils import (
    fetch_batch_ltp_data, 
    fetch_indices_sectors,
    fetch_stocks_by_index,
    get_change_color_class,
    get_trend_symbol
)

# Import market data utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common_utils'))
try:
    from common_utils.market_data import MarketData
except ImportError:
    MarketData = None

logger = logging.getLogger(__name__)


# Remove unused global state - dashboard data is fetched on-demand


async def render_dashboard_page(fetch_api, user_storage, get_cached_instruments):
    """Main enhanced dashboard page with new layout structure"""

    broker = user_storage.get('default_broker', 'Zerodha')
    market_data = MarketData() if MarketData else None

    # CRITICAL: Main dashboard container with proper viewport handling
    with ui.column().classes("w-full"):
        # Enhanced Dashboard Title
        await render_enhanced_dashboard_title(broker, market_data)

        # Top Row - Portfolio Summary (40%) and Quick Trade (60%) - MUST BE IN SAME ROW
        with ui.row().classes("w-full"):
            with ui.column().classes("").style("flex: 0 0 35%; max-width: 35%; min-width: 35%;"):
                await render_compact_portfolio_widget(fetch_api, user_storage, broker)
            with ui.column().classes("").style("flex: 0 0 64%; max-width: 64%; min-width: 64%;"):
                await render_enhanced_quick_trade_section(fetch_api, user_storage, get_cached_instruments, broker)

        # Main Content Row: MUST BE Watchlist (20%) - Historical Chart (45%) - Live Market Data (35%)
        with ui.row().classes("w-full gap-3").style("min-height: calc(100vh - 320px); display: flex; align-items: flex-center;"):
            # Left Panel - Watchlist and Active Strategies (EXACTLY 20%)
            with ui.column().classes("").style("flex: 0 0 20%; max-width: 18.5%; min-width: 18.5%;"):
                await render_enhanced_watchlist_section(fetch_api, user_storage, get_cached_instruments, broker)
                await render_enhanced_strategies_section(fetch_api, user_storage, broker)

            # Center Panel - Historical Chart + Recent Orders (EXACTLY 45%)
            with ui.column().classes("").style("flex: 0 0 45%; max-width: 45%; min-width: 45%;"):
                # Historical Chart
                await render_historical_chart_section(fetch_api, user_storage, get_cached_instruments, broker)
                # Recent Orders below chart
                await render_enhanced_recent_orders_section(fetch_api, user_storage, broker)
            
            # Right Panel - Live Market Data + Heatmap (EXACTLY 35%)
            with ui.column().classes("").style("flex: 0 0 35%; max-width: 35%; min-width: 35%;"):
                await render_market_data_tables_section(fetch_api, user_storage, broker, market_data)

    # Real-time updates handled by individual widgets
    # Note: Broker switching shows notification in menu bar
    # Dashboard doesn't auto-refresh - users can manually refresh or navigate to portfolio/positions


async def render_enhanced_dashboard_title(broker, market_data):
    """Enhanced dashboard title with status indicators and live market summary"""
    with ui.row().classes("page-header-standard w-full justify-between items-center p-3"):
        # Left side - Title and subtitle (with proper spacing)
        with ui.column().classes("gap-2"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("candlestick_chart", size="1.5rem").classes("text-cyan-400")
                ui.label(f"{broker} Trading Dashboard").classes("page-title-standard theme-header-text text-xl")
                ui.chip("LIVE", color="green").classes("text-xs status-chip")

            ui.label("Real-time market data and portfolio management").classes("page-subtitle-standard text-sm")
        
        # Right side - Live market summary with real data
        await render_live_market_summary(market_data)
        
async def render_live_market_summary(market_data):
    """Render live market summary with real data"""
    with ui.row().classes("items-center gap-6"):
        try:
            # Key indices to display
            indices_config = [
                {"name": "NIFTY 50", "key": "NIFTY 50"},
                {"name": "NIFTY 500", "key": "NIFTY 500"},
                {"name": "NIFTY NEXT 50", "key": "NIFTY NEXT 50"},
                {"name": "NIFTY BANK", "key": "NIFTY BANK"},
                {"name": "NIFTY SMALLCAP 100", "key": "NIFTY SMALLCAP 100"},
                {"name": "NIFTY MIDCAP 100", "key": "NIFTY MIDCAP 100"}
            ]
            
            for idx_config in indices_config:
                try:
                    if market_data and hasattr(market_data, 'get_index_stocks_data'):
                        index_data = market_data.get_index_stocks_data(stock_idx=idx_config["key"], as_df=True)
                        if not index_data.empty:
                            index_row = index_data.iloc[0]  # Index data is first row
                            value = index_row.get("Close", 0)
                            change_pct = index_row.get("Pct_Change", 0)
                        else:
                            # Fallback values
                            value = 0 if "NIFTY 50" in idx_config["name"] else 0
                            change_pct = 0.0 if "NIFTY 50" in idx_config["name"] else 0.0
                    else:
                        # Fallback values when market_data not available
                        value = 0 if "NIFTY 50" in idx_config["name"] else 0
                        change_pct = 0.0 if "NIFTY 50" in idx_config["name"] else 0.0
                    
                    # Determine colors
                    change_color = "text-green-400" if change_pct >= 0 else "text-red-400"
                    trend_symbol = "▲" if change_pct >= 0 else "▼"
                    
                    with ui.column().classes("text-center min-w-20"):
                        ui.label(idx_config["name"]).classes("text-xs text-gray-400")
                        ui.label(f"{value:,.0f}").classes("text-sm font-bold theme-text-primary")
                        ui.label(f"{trend_symbol} {abs(change_pct):.2f}%").classes(f"text-xs font-bold {change_color}")
                        
                except Exception as e:
                    logger.error(f"Error loading {idx_config['name']}: {e}")
                    # Show error state
                    with ui.column().classes("text-center min-w-20"):
                        ui.label(idx_config["name"]).classes("text-xs text-gray-400")
                        ui.label("--").classes("text-sm font-bold text-gray-500")
                        ui.label("--").classes("text-xs text-gray-500")
                        
        except Exception as e:
            logger.error(f"Error in market summary: {e}")
            # Fallback display
            fallback_data = [
                {"name": "NIFTY 50", "value": 21736, "change_pct": 1.09},
                {"name": "BANK NIFTY", "value": 46816, "change_pct": 1.48}
            ]
            for data in fallback_data:
                change_color = "text-green-400" if data["change_pct"] >= 0 else "text-red-400"
                trend_symbol = "▲" if data["change_pct"] >= 0 else "▼"
                
                with ui.column().classes("text-center min-w-20"):
                    ui.label(data["name"]).classes("text-xs text-gray-400")
                    ui.label(f"{data['value']:,.0f}").classes("text-sm font-bold theme-text-primary")
                    ui.label(f"{trend_symbol} {abs(data['change_pct']):.2f}%").classes(f"text-xs font-bold {change_color}")

async def render_enhanced_watchlist_section(fetch_api, user_storage, get_cached_instruments, broker):
    """Watchlist section using backend watchlist routes"""

    with ui.card().classes("dashboard-card watchlist-card w-full"):
        # Header
        with ui.row().classes("card-header w-full justify-between items-center"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("visibility", size="1.2rem").classes("text-cyan-400")
                ui.label("Watchlist").classes("card-title text-sm")

            with ui.row().classes("items-center gap-1"):
                ui.button("Full", on_click=lambda: ui.navigate.to('/watchlist')).props("flat size=xs").classes("text-cyan-400 text-xs")
                ui.button(icon="refresh", on_click=lambda: asyncio.create_task(refresh_watchlist())).props("flat round size=xs").classes("text-gray-400")

        # Watchlist content container with fixed height and scroll
        watchlist_container = ui.column().classes("watchlist-content w-full p-2 h-64 overflow-y-auto")

        async def refresh_watchlist():
            await render_watchlist_content()

        async def render_watchlist_content():
            watchlist_container.clear()
            
            try:
                # Get watchlist symbols from backend
                watchlist_response = await fetch_api("/api/watchlist/Default/symbols/", params={"page": 0, "page_size": 8})
                
                # Debug logging
                logger.info(f"Watchlist response: {watchlist_response}")
                
                if watchlist_response and "symbols" in watchlist_response:
                    symbols = [item["symbol"] for item in watchlist_response["symbols"]]
                    
                    if symbols:
                        # Use batch LTP data fetching for better performance
                        ltp_data_map = await fetch_batch_ltp_data(symbols, get_cached_instruments, broker, fetch_api)
                        
                        with watchlist_container:
                            for symbol in symbols:
                                await render_compact_watchlist_item(symbol, ltp_data_map, fetch_api, broker)
                    else:
                        with watchlist_container:
                            ui.label("Watchlist empty").classes("text-gray-500 text-center text-xs p-2")
                            ui.label("Add symbols from full page").classes("text-gray-400 text-center text-xs")
                else:
                    # Try alternative approach - use sample data for now
                    with watchlist_container:
                        ui.label("Using sample watchlist data").classes("text-gray-400 text-center text-xs p-1")
                        sample_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
                        ltp_data_map = await fetch_batch_ltp_data(sample_symbols, get_cached_instruments, broker, fetch_api)
                        
                        for symbol in sample_symbols:
                            await render_compact_watchlist_item(symbol, ltp_data_map, fetch_api, broker)
                        
            except Exception as e:
                logger.error(f"Error loading watchlist from backend: {e}")
                with watchlist_container:
                    ui.label("Loading sample watchlist").classes("text-yellow-500 text-center text-xs p-1")
                    # Fallback to sample data
                    try:
                        sample_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
                        ltp_data_map = await fetch_batch_ltp_data(sample_symbols, get_cached_instruments, broker, fetch_api)
                        
                        for symbol in sample_symbols:
                            await render_compact_watchlist_item(symbol, ltp_data_map, fetch_api, broker)
                    except Exception as fallback_error:
                        logger.error(f"Fallback watchlist error: {fallback_error}")
                        ui.label("Error loading watchlist").classes("text-red-500 text-center text-xs p-2")

        # Initial render
        await render_watchlist_content()
        
        return watchlist_container


async def render_compact_watchlist_item(symbol, ltp_data_map, fetch_api, broker):
    """Render compact watchlist item for sidebar using batch LTP data"""

    try:
        # Get LTP data from the batch map
        ltp_data = ltp_data_map.get(symbol, {})
        price = ltp_data.get('last_price', 0.0)
        previous_close = ltp_data.get('previous_close', 0.0)
        
        # Calculate change
        change = price - previous_close if previous_close > 0 else 0.0
        change_pct = (change / previous_close) * 100 if previous_close > 0 else 0.0

        # Determine styling based on change using utility functions
        change_class = get_change_color_class(change)
        trend_icon = get_trend_symbol(change)

        with ui.row().classes("w-full items-center py-1 px-2 hover:bg-gray-800/50 rounded gap-1"):
            # Display as single row: Symbol | LTP | Change | Change% with proper spacing
            ui.label(symbol).classes("font-medium theme-text-primary text-xs flex-1 min-w-0 truncate")
            ui.label(f"Rs.{price:,.0f}").classes("theme-text-secondary text-xs w-16 text-right")
            ui.label(f"{change:+,.1f}").classes(f"text-xs {change_class} w-14 text-right")
            ui.label(f"{trend_icon}{abs(change_pct):.1f}%").classes(f"text-xs {change_class} font-medium w-16 text-right")

    except Exception as e:
        logger.error(f"Error rendering compact watchlist item {symbol}: {e}")
        with ui.row().classes("w-full p-2"):
            ui.label(symbol).classes("theme-text-primary text-xs")
            ui.label("Error").classes("text-red-500 text-xs")


# render_enhanced_watchlist_item function removed - unused (only render_compact_watchlist_item is used)


# render_enhanced_chart_section and create_dashboard_chart functions removed - unused


async def render_compact_portfolio_widget(fetch_api, user_storage, broker):
    """Compact portfolio widget taking minimal space"""

    with ui.card().classes("dashboard-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center justify-between .6rem"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("account_balance_wallet", size="1.2rem").classes("text-green-400")
                ui.label("Portfolio Summary").classes("card-title text-lg")
            
            ui.button("View Details", icon="launch", 
                     on_click=lambda: ui.navigate.to('/portfolio')).props("flat size=sm").classes("text-cyan-400")

        # Compact metrics in single row
        portfolio_metrics_container = ui.row().classes("w-full p-3 gap-4 justify-between")

        try:
            # Fetch portfolio data using existing API calls
            funds_data = await fetch_api(f"/funds/{broker}")
            portfolio_data = await fetch_api(f"/portfolio/{broker}") or []
            positions_data = await fetch_api(f"/positions/{broker}") or []

            # Calculate metrics
            available_funds = 0.0
            if funds_data and isinstance(funds_data, dict):
                equity = funds_data.get('equity', {})
                available = equity.get('available', 0.0)
                available_funds = float(available) if isinstance(available, (int, float)) else 0.0

            portfolio_value = sum(
                h.get("Quantity", 0) * h.get("LastPrice", 0) for h in portfolio_data if isinstance(h, dict)
            )

            total_invested = sum(
                h.get("Quantity", 0) * h.get("AvgPrice", 0) for h in portfolio_data if isinstance(h, dict)
            )

            daily_pnl = sum(
                h.get("DayChange", 0) for h in portfolio_data if isinstance(h, dict)
            )
            daily_pnl_pct = (daily_pnl / total_invested * 100) if total_invested > 0 else 0

            total_pnl = portfolio_value - total_invested
            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

            with portfolio_metrics_container:
                # Available Funds - Compact
                with ui.column().classes("flex-1 text-center"):
                    ui.label("Available").classes("text-xs text-gray-400")
                    ui.label(f"₹{available_funds:,.0f}").classes("text-sm font-bold text-blue-400")

                # Portfolio Value - Compact
                with ui.column().classes("flex-1 text-center"):
                    ui.label("Portfolio").classes("text-xs text-gray-400")
                    ui.label(f"₹{portfolio_value:,.0f}").classes("text-sm font-bold theme-text-primary")

                # Total P&L - Compact
                total_pnl_color = "text-green-400" if total_pnl >= 0 else "text-red-400"
                with ui.column().classes("flex-1 text-center"):
                    ui.label("Total P&L").classes("text-xs text-gray-400")
                    ui.label(f"₹{total_pnl:+,.0f}").classes(f"text-sm font-bold {total_pnl_color}")

                # Daily P&L - Compact
                daily_pnl_color = "text-green-400" if daily_pnl >= 0 else "text-red-400"
                with ui.column().classes("flex-1 text-center"):
                    ui.label("Day P&L").classes("text-xs text-gray-400")
                    ui.label(f"₹{daily_pnl:+,.0f}").classes(f"text-sm font-bold {daily_pnl_color}")

                # Positions - Compact
                open_positions = len([p for p in positions_data if isinstance(p, dict) and p.get("Quantity", 0) != 0])
                with ui.column().classes("flex-1 text-center"):
                    ui.label("Positions").classes("text-xs text-gray-400")
                    ui.label(str(open_positions)).classes("text-sm font-bold text-orange-400")

        except Exception as e:
            logger.error(f"Error loading portfolio data: {e}")
            with portfolio_metrics_container:
                ui.label("Error loading portfolio data").classes("text-red-500 text-center p-4")


async def render_enhanced_quick_trade_section(fetch_api, user_storage, get_cached_instruments, broker):
    """Compact quick trade section for sidebar"""

    with ui.card().classes("dashboard-card trading-card w-full"):
        # Header
        with ui.row().classes("card-header w-full justify-between items-center p-3"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("flash_on", size="1.2rem").classes("text-yellow-400")
                ui.label("Quick Trade").classes("card-title text-sm")
            
            ui.button("Full", on_click=lambda: ui.navigate.to('/order-management')).props("flat size=xs").classes("text-yellow-400 text-xs")

        # Enhanced trade form - Better layout with proper spacing and full widget utilization
        with ui.column().classes("quick-trade-form p-4 gap-3"):
            # Row 1: Symbol input with real-time price display
            with ui.row().classes("w-full gap-3 items-center"):
                symbol_input = ui.input("Symbol", placeholder="e.g. RELIANCE").classes("flex-1").props("outlined dense")
                
                # with ui.column().classes("min-w-0 text-center"):
                price_display = ui.label("--").classes("text-sm font-bold text-cyan-400")
                last_updated = ui.label("").classes("text-xs text-gray-500")
                quantity_input = ui.number("Qty", value=1, min=1, max=10000).classes("flex-1").props(
                    "outlined dense")
                order_type = ui.select(["MKT", "LMT", "SL", "SLM"], value="MKT", label="Type").classes(
                    "flex-1").props("outlined dense")
                product_type = ui.select(
                    ["CNC", "MIS", "NRM"] if broker == "Zerodha" else ["D", "I", "T"],
                    value="CNC" if broker == "Zerodha" else "D",
                    label="Product"
                ).classes("flex-1").props("outlined dense")

                price_input = ui.number("Price", value=0, step=0.05).classes("flex-1").props("outlined dense")
                price_input.visible = False

                trigger_input = ui.number("Trigger", value=0, step=0.05).classes("flex-1").props("outlined dense")
                trigger_input.visible = False

                buy_button = ui.button("BUY", icon="trending_up").classes(
                    "flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3")
                sell_button = ui.button("SELL", icon="trending_down").classes(
                    "flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3")
            # # Row 2: Trading parameters in proper grid
            # with ui.row().classes("w-full gap-2"):
            #     quantity_input = ui.number("Qty", value=1, min=1, max=10000).classes("flex-1").props("outlined dense")
            #     order_type = ui.select(["MKT", "LMT", "SL", "SLM"], value="MKT", label="Type").classes("flex-1").props("outlined dense")
            #     product_type = ui.select(
            #         ["CNC", "MIS", "NRM"] if broker == "Zerodha" else ["D", "I", "T"],
            #         value="CNC" if broker == "Zerodha" else "D",
            #         label="Product"
            #     ).classes("flex-1").props("outlined dense")
            #
            # # Row 3: Conditional inputs (price and trigger) - full width when visible
            # with ui.row().classes("w-full gap-2"):
            #     price_input = ui.number("Price", value=0, step=0.05).classes("flex-1").props("outlined dense")
            #     price_input.visible = False
            #
            #     trigger_input = ui.number("Trigger", value=0, step=0.05).classes("flex-1").props("outlined dense")
            #     trigger_input.visible = False
            #
            # # Row 4: Action buttons - full width and properly sized
            # with ui.row().classes("w-full gap-3 mt-2"):
            #     buy_button = ui.button("BUY", icon="trending_up").classes("flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3")
            #     sell_button = ui.button("SELL", icon="trending_down").classes("flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3")

            def toggle_inputs():
                price_input.visible = order_type.value in ["LMT", "SL"]
                trigger_input.visible = order_type.value in ["SL", "SLM"]
                update_buttons()

            order_type.on_value_change(toggle_inputs)
            
            async def fetch_symbol_price():
                await update_symbol_data()
            
            async def update_symbol_data():
                if not symbol_input.value:
                    price_display.text = "Price: --"
                    last_updated.text = ""
                    update_buttons()
                    return
                    
                try:
                    # Get live price
                    all_instruments_map = await get_cached_instruments(broker)
                    instrument_token = all_instruments_map.get(symbol_input.value.upper())
                    
                    if instrument_token:
                        ltp_response = await fetch_api(f"/ltp/{broker}", params={"instruments": instrument_token})
                        if ltp_response and isinstance(ltp_response, list) and ltp_response:
                            ltp_data = ltp_response[0]
                            current_price = ltp_data.get('last_price', 0.0)
                            price_display.text = f"LTP: ₹{current_price:,.2f}"
                            last_updated.text = datetime.now().strftime("%H:%M:%S")
                            
                            # Autofill price for LIMIT orders
                            if order_type.value == "LIMIT":
                                price_input.value = current_price
                            # Autofill trigger for SL orders (slightly below current price)
                            elif order_type.value in ["SL", "SL-M"]:
                                trigger_input.value = current_price * 0.95  # 5% below current price
                        else:
                            price_display.text = "Price: Not found"
                            last_updated.text = "Error"
                    else:
                        price_display.text = "Symbol not found"
                        last_updated.text = "Invalid"
                except Exception as e:
                    price_display.text = "Price: Error"
                    last_updated.text = "Error"
                    logger.error(f"Error fetching symbol price: {e}")
                    
                update_buttons()
                
            def update_buttons():
                symbol = symbol_input.value.upper() if symbol_input.value else ""
                qty = quantity_input.value if quantity_input.value else 1
                
                if symbol:
                    buy_button.text = f"BUY {qty} {symbol}"
                    sell_button.text = f"SELL {qty} {symbol}"
                    # Fix: Remove existing classes first, then add new ones
                    buy_button.classes(remove="bg-gray-600")
                    buy_button.classes("bg-green-600 hover:bg-green-700 text-white flex-1")
                    sell_button.classes(remove="bg-gray-600")
                    sell_button.classes("bg-red-600 hover:bg-red-700 text-white flex-1")
                else:
                    buy_button.text = "BUY"
                    sell_button.text = "SELL"
                    # Fix: Remove existing classes first, then add new ones
                    buy_button.classes(remove="bg-green-600 hover:bg-green-700")
                    buy_button.classes("bg-gray-600 text-white flex-1")
                    sell_button.classes(remove="bg-red-600 hover:bg-red-700")
                    sell_button.classes("bg-gray-600 text-white flex-1")

            symbol_input.on_value_change(lambda: update_symbol_data())
            quantity_input.on_value_change(update_buttons)
            
            async def place_quick_order(transaction_type):
                """Place real quick order using backend API"""
                try:
                    # Validation
                    if not symbol_input.value:
                        ui.notify("Enter symbol", type="warning")
                        return

                    symbol = symbol_input.value.upper()
                    
                    # Get instrument token from cached instruments
                    all_instruments_map = await get_cached_instruments(broker)
                    if symbol not in all_instruments_map:
                        ui.notify(f"Symbol {symbol} not found", type="negative")
                        return
                    
                    instrument_token = all_instruments_map[symbol]
                    qty = int(quantity_input.value)
                    order_type_val = order_type.value
                    product_val = product_type.value
                    
                    # Validate quantity
                    if qty <= 0:
                        ui.notify("Quantity must be greater than 0", type="negative")
                        return
                    
                    # Validate price for limit orders
                    if order_type_val in ["LMT", "SL"] and (not price_input.value or price_input.value <= 0):
                        ui.notify("Price must be greater than 0 for LIMIT and SL orders", type="negative")
                        return
                    
                    # Validate trigger price for stop loss orders
                    if order_type_val in ["SL", "SLM"] and (not trigger_input.value or trigger_input.value <= 0):
                        ui.notify("Trigger price must be greater than 0 for SL orders", type="negative")
                        return
                    
                    # Build order data structure matching order_management.py
                    order_data = {
                        "trading_symbol": symbol,
                        "instrument_token": instrument_token,
                        "quantity": qty,
                        "transaction_type": transaction_type,
                        "order_type": order_type_val,
                        "product_type": product_val,
                        "price": float(price_input.value) if order_type_val in ["LMT", "SL"] else 0,
                        "trigger_price": float(trigger_input.value) if order_type_val in ["SL", "SLM"] else 0,
                        "validity": "DAY",  # Default to DAY validity for quick orders
                        "disclosed_quantity": 0,  # No disclosed quantity for quick orders
                        "is_amo": False,  # Quick orders are regular orders
                        "broker": broker,
                        # No risk management for quick orders to keep it simple
                        "stop_loss": None,
                        "target": None,
                        "is_trailing_stop_loss": False
                    }
                    
                    # Build confirmation message
                    order_details = f"{transaction_type} {qty} {symbol} ({order_type_val})"
                    if order_data["price"] > 0:
                        order_details += f" at ₹{order_data['price']:.2f}"
                    if order_data["trigger_price"] > 0:
                        order_details += f" trigger ₹{order_data['trigger_price']:.2f}"
                    order_details += f" [{product_val}]"
                    
                    # Show quick confirmation and place order
                    with ui.dialog() as dialog, ui.card().classes('p-4 min-w-80'):
                        ui.label('Quick Order Confirmation').classes('text-lg font-bold mb-3')
                        ui.label(order_details).classes('text-white mb-4')
                        
                        with ui.row().classes('gap-3 w-full justify-end'):
                            ui.button('Cancel', on_click=dialog.close).classes('bg-gray-600 text-white px-4 py-2 rounded')
                            
                            async def confirm_quick_order():
                                dialog.close()
                                
                                # Show loading notification
                                ui.notify("Placing order...", type="info")
                                
                                try:
                                    # Place order via backend API
                                    response = await fetch_api("/orders", method="POST", data=order_data)
                                    
                                    if response and response.get('order_id'):
                                        ui.notify(f"Order placed successfully: {response['order_id']}", type='positive')
                                        # Don't clear form for quick repeat orders
                                        await update_symbol_data()
                                    else:
                                        error_msg = response.get('error', 'Unknown error') if response else 'Failed to place order'
                                        ui.notify(f"Order failed: {error_msg}", type='negative')
                                        
                                except Exception as api_error:
                                    logger.error(f"API error placing order: {api_error}")
                                    ui.notify(f"API Error: {str(api_error)}", type='negative')
                            
                            ui.button('Place Order', on_click=confirm_quick_order).classes('bg-green-600 text-white px-4 py-2 rounded')
                    
                    dialog.open()
                    
                except Exception as e:
                    logger.error(f"Error in quick order placement: {e}")
                    ui.notify(f"Error: {str(e)}", type="negative")

            buy_button.on_click(lambda: place_quick_order("BUY"))
            sell_button.on_click(lambda: place_quick_order("SELL"))
            
            # Initial setup
            update_buttons()


async def render_enhanced_recent_orders_section(fetch_api, user_storage, broker):
    """Enhanced recent orders section for sidebar"""

    with ui.card().classes("dashboard-card orders-card w-full"):
        # Header
        with ui.row().classes("card-header w-full justify-between items-center p-3"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("list_alt", size="1.2rem").classes("text-blue-400")
                ui.label("Recent Orders").classes("card-title text-sm")
            
            ui.button("All", on_click=lambda: ui.navigate.to('/order-book')).props("flat size=xs").classes("text-cyan-400 text-xs")

        # Orders container with headers and fixed height
        with ui.column().classes("w-full p-3"):
            # Column headers - optimized grid layout with better spacing
            with ui.row().classes("w-full items-center border-b border-gray-700 pb-2 mb-2").style("display: grid; grid-template-columns: 1fr 0.8fr 0.8fr 0.8fr 1fr; gap: 0.25rem; align-items: center;"):
                ui.label("Symbol").classes("text-gray-400 text-xs font-semibold")
                ui.label("Trans.Type").classes("text-gray-400 text-xs font-semibold text-center")
                ui.label("Qty").classes("text-gray-400 text-xs font-semibold text-left")
                ui.label("Price").classes("text-gray-400 text-xs font-semibold text-left")
                ui.label("Status").classes("text-gray-400 text-xs font-semibold text-left")
            
            orders_container = ui.column().classes("w-full gap-1 h-40 overflow-y-auto")

        try:
            # Fetch recent orders
            orders_data = await fetch_api(f"/orders/{broker}") or []
            recent_orders = orders_data[-5:] if orders_data else []  # Last 5 orders for compact view

            with orders_container:
                if recent_orders:
                    for order in recent_orders:
                        symbol = order.get("trading_symbol", "N/A")[:8]  # Truncate symbol
                        transaction_type = order.get("transaction_type", "N/A")
                        quantity = order.get("quantity", 0)
                        status = order.get("status", "UNKNOWN")
                        order_type = order.get("order_type", "N/A")
                        price = order.get("price", 0)

                        # Enhanced status and transaction type colors
                        status_color = {
                            "COMPLETE": "text-green-400",
                            "OPEN": "text-yellow-400", 
                            "CANCELLED": "text-red-400",
                            "REJECTED": "text-red-400"
                        }.get(status, "text-gray-400")
                        
                        # Transaction type colors
                        trans_color = "text-green-400" if transaction_type == "BUY" else "text-red-400" if transaction_type == "SELL" else "text-gray-400"
                        
                        # Background based on status
                        bg_class = ""
                        if status == "COMPLETE":
                            bg_class = "bg-green-900/20 border-l-2 border-green-400"
                        elif status == "OPEN":
                            bg_class = "bg-yellow-900/20 border-l-2 border-yellow-400"
                        elif status in ["CANCELLED", "REJECTED"]:
                            bg_class = "bg-red-900/20 border-l-2 border-red-400"
                        else:
                            bg_class = "bg-gray-900/20 border-l-2 border-gray-400"

                        # Display as properly aligned grid with optimized spacing
                        with ui.row().classes(f"w-full p-2 rounded {bg_class} hover:bg-gray-800/30 transition-all mb-1").style("display: grid; grid-template-columns: 1fr 0.8fr 0.8fr 0.8fr 1fr; gap: 0.25rem; align-items: center; min-height: 32px;"):
                            # Symbol - Column 1 (wider for symbol names)
                            ui.label(symbol).classes("theme-text-primary text-xs font-bold truncate")
                            
                            # Transaction Type - Column 2 (show full text)
                            ui.label(transaction_type).classes(f"{trans_color} text-xs font-bold text-center")
                            
                            # Quantity - Column 3
                            ui.label(f"{quantity}").classes("text-gray-300 text-xs text-left")
                            
                            # Price - Column 4
                            if price > 0:
                                ui.label(f"₹{price:.1f}").classes("text-gray-400 text-xs text-left")
                            else:
                                ui.label("--").classes("text-gray-500 text-xs text-left")
                            
                            # Status - Column 5
                            ui.label(status).classes(f"{status_color} text-xs font-semibold text-left")
                else:
                    ui.label("No recent orders").classes("text-gray-400 text-center text-xs p-2")

        except Exception as e:
            logger.error(f"Error loading orders: {e}")
            with orders_container:
                ui.label("Error loading orders").classes("text-red-500 text-center text-xs p-2")


async def render_enhanced_order_book_section(fetch_api, user_storage, broker):
    """Enhanced order book section"""

    with ui.card().classes("dashboard-card orders-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center p-4"):
            ui.icon("list_alt", size="1.5rem").classes("text-blue-400")
            ui.label("Recent Orders").classes("card-title")

        # Orders container
        orders_container = ui.column().classes("w-full p-4 gap-2")

        try:
            # Fetch recent orders
            orders_data = await fetch_api(f"/orders/{broker}") or []
            recent_orders = orders_data[-5:] if orders_data else []  # Last 5 orders

            with orders_container:
                if recent_orders:
                    for order in recent_orders:
                        symbol = order.get("trading_symbol", "N/A")
                        order_type = order.get("order_type", "N/A")
                        transaction_type = order.get("transaction_type", "N/A")
                        quantity = order.get("quantity", 0)
                        status = order.get("status", "UNKNOWN")

                        # Status color
                        status_color = {
                            "COMPLETE": "text-green-400",
                            "OPEN": "text-yellow-400",
                            "CANCELLED": "text-red-400",
                            "REJECTED": "text-red-400"
                        }.get(status, "text-gray-400")

                        with ui.row().classes("w-full justify-between items-center py-2 border-b border-gray-700"):
                            with ui.column().classes("flex-1"):
                                ui.label(f"{symbol} ({transaction_type})").classes("theme-text-primary font-medium text-sm")
                                ui.label(f"{order_type} • Qty: {quantity}").classes("text-gray-400 text-xs")

                            ui.label(status).classes(f"{status_color} text-xs font-semibold")
                else:
                    ui.label("No recent orders").classes("text-gray-400 text-center p-4")

        except Exception as e:
            logger.error(f"Error loading orders: {e}")
            with orders_container:
                ui.label("Error loading orders").classes("text-red-500 text-center p-4")


async def render_enhanced_strategies_section(fetch_api, user_storage, broker):
    """Enhanced strategies section"""

    with ui.card().classes("dashboard-card strategies-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center"):
            ui.icon("psychology", size="1.5rem").classes("text-purple-400")
            ui.label("Active Strategies").classes("card-title")

        # Strategies container with fixed height and scroll
        strategies_container = ui.column().classes("w-full p-4 gap-2 h-48 overflow-y-auto")

        try:
            # Fetch active strategies
            strategies_data = await fetch_api(f"/strategies/all/{broker}") or []
            active_strategies = [s for s in strategies_data if s.get("status") == "ACTIVE"]

            with strategies_container:
                if active_strategies:
                    for strategy in active_strategies[:3]:  # Show top 3 active strategies
                        name = strategy.get("name", "N/A")
                        pnl = strategy.get("pnl", 0)
                        status = strategy.get("status", "INACTIVE")

                        pnl_color = "text-green-400" if pnl > 0 else "text-red-400"
                        status_color = "text-green-400" if status == "ACTIVE" else "text-gray-400"

                        with ui.row().classes("w-full justify-between items-center py-2 border-b border-gray-700"):
                            with ui.column().classes("flex-1"):
                                ui.label(name).classes("theme-text-primary font-medium text-sm")
                                ui.label(status).classes(f"{status_color} text-xs")

                            ui.label(f"₹{pnl:+.2f}").classes(f"{pnl_color} text-sm font-semibold")
                else:
                    ui.label("No active strategies").classes("text-gray-400 text-center p-4")

                # Quick action button
                ui.button("Manage Strategies", icon="settings",
                         on_click=lambda: ui.navigate.to('/strategies')).classes("w-full mt-2 text-purple-400").props("outline")

        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            with strategies_container:
                ui.label("Error loading strategies").classes("text-red-500 text-center p-4")


async def render_historical_chart_section(fetch_api, user_storage, get_cached_instruments, broker):
    """Historical chart section with time period selection in header"""
    
    with ui.card().classes("dashboard-card w-full"):
        # Chart Header with time period selection
        with ui.row().classes("card-header w-full justify-between items-center p-4"):
            with ui.row().classes("items-center gap-4"):
                ui.label("Historical Chart").classes("card-title")
                
                # Index/Sector selector
                indices_sectors = await fetch_indices_sectors(fetch_api)
                chart_symbol_select = ui.select(
                    options=indices_sectors[:10],  # First 10 options
                    value=indices_sectors[0] if indices_sectors else "NIFTY_50",
                    label="Index/Sector"
                ).classes("w-48")
                
            # Time period selection buttons in header
            with ui.row().classes("items-center gap-1"):
                time_periods = [
                    ("1M", "1 Month"),
                    ("6M", "6 Months"), 
                    ("1Y", "1 Year"),
                    ("3Y", "3 Years"),
                    ("5Y", "5 Years"),
                    ("MAX", "Max")
                ]
                
                selected_period = [{"value": "6M"}]  # Use list to make it mutable in closures
                
                period_buttons = []
                for period_key, period_label in time_periods:
                    active_class = "bg-cyan-600" if period_key == selected_period[0]["value"] else "bg-gray-700 hover:bg-gray-600"
                    period_btn = ui.button(period_key, on_click=lambda pk=period_key: asyncio.create_task(update_chart_period(pk))).classes(f"text-xs {active_class}")
                    period_buttons.append((period_btn, period_key))

        # Chart container
        chart_container = ui.column().classes("w-full h-80 p-4")
        
        async def update_chart_period(period):
            selected_period[0]["value"] = period
            # Update button styles - recreate buttons with proper styling
            for btn, key in period_buttons:
                if key == period:
                    # Remove existing classes and add active classes
                    btn.classes(remove="bg-gray-700 hover:bg-gray-600")
                    btn.classes("bg-cyan-600")
                else:
                    # Remove existing classes and add inactive classes
                    btn.classes(remove="bg-cyan-600")
                    btn.classes("bg-gray-700 hover:bg-gray-600")
            await render_historical_chart(chart_container, chart_symbol_select.value, period, fetch_api)
        
        async def update_chart_symbol():
            await render_historical_chart(chart_container, chart_symbol_select.value, selected_period[0]["value"], fetch_api)
            
        chart_symbol_select.on_value_change(lambda: asyncio.create_task(update_chart_symbol()))
        
        # Initial chart render
        await render_historical_chart(chart_container, chart_symbol_select.value, selected_period[0]["value"], fetch_api)


async def render_historical_chart(container, symbol, period, fetch_api):
    """Render historical chart for selected symbol and period"""
    container.clear()
    
    try:
        # Calculate date range based on period
        end_date = datetime.now()
        if period == "1M":
            start_date = end_date - timedelta(days=30)
        elif period == "6M":
            start_date = end_date - timedelta(days=180)
        elif period == "1Y":
            start_date = end_date - timedelta(days=365)
        elif period == "3Y":
            start_date = end_date - timedelta(days=365*3)
        elif period == "5Y":
            start_date = end_date - timedelta(days=365*5)
        else:  # MAX
            start_date = end_date - timedelta(days=365*10)
        
        # Convert symbol with spaces to underscores for database lookup
        db_symbol = symbol.replace(" ", "_")
        
        # Fetch historical data from database 
        response = await fetch_api("/api/data/fetch-table-data/", params={
            "table_name": db_symbol,
            "columns": ["timestamp", "close", "high", "low", "open", "volume"],
            "filters": f"timestamp >= '{start_date.strftime('%Y-%m-%d')}' AND timestamp <= '{end_date.strftime('%Y-%m-%d')}'",
            "required_db": "nsedata",
            "order_by": "timestamp"
        })
        
        if response and "data" in response and response["data"]:
            data = response["data"]
            dates = [row["timestamp"] for row in data]
            closes = [float(row["close"]) if row["close"] else 0.0 for row in data]
            
            # Create line chart using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=closes,
                mode='lines',
                name=symbol,
                line=dict(color='#00BCD4', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Price: ₹%{y:,.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{symbol} - {period} Historical Data",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=40, t=40, b=40),
                showlegend=False,
                hovermode='x unified'
            )
            
            with container:
                ui.plotly(fig).classes("w-full h-full")
        else:
            # Fallback: Show sample chart
            with container:
                sample_dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, 0, -1)]
                sample_prices = [21000 + (x * 50) + (x % 7 * 100) for x in range(30)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sample_dates,
                    y=sample_prices,
                    mode='lines',
                    name=symbol,
                    line=dict(color='#00BCD4', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} - Sample Data",
                    xaxis_title="Date", 
                    yaxis_title="Price (₹)",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    showlegend=False
                )
                
                ui.plotly(fig).classes("w-full h-full")
                
    except Exception as e:
        logger.error(f"Error rendering historical chart: {e}")
        with container:
            ui.label("Error loading chart data").classes("text-red-500 text-center p-8")


async def render_market_data_tables_section(fetch_api, user_storage, broker, market_data):
    """Market data tables section with Nifty 50 and FNO data"""
    
    with ui.card().classes("dashboard-card w-full"):
        # Header with radio selection and FNO link
        with ui.row().classes("card-header w-full justify-between items-center p-4"):
            with ui.row().classes("items-center gap-4"):
                ui.icon("table_chart", size="1.5rem").classes("text-blue-400")
                ui.label("Live Data").classes("card-title")
            
            with ui.row().classes("items-center gap-4"):
                # Dropdown selection for key indices and sectors only
                key_indices = [
                    "Nifty 50", "FNO Snapshot", 
                    "NIFTY 50", "NIFTY 500", "NIFTY NEXT 50", "NIFTY BANK", 
                    "NIFTY SMALLCAP 100", "NIFTY MIDCAP 100", "NIFTY IT",
                    "NIFTY FMCG", "NIFTY AUTO", "NIFTY PHARMA", "NIFTY METAL"
                ]
                data_type_select = ui.select(
                    options=key_indices,
                    value="Nifty 50",
                    label="Select Index/Sector"
                ).classes("w-48")
                
                # FNO Snapshot modal link
                ui.button("View F&O Snapshot", icon="open_in_new", 
                         on_click=lambda: show_fno_modal_handler()).props("flat size=sm").classes("text-cyan-400")
        
        # Table container
        table_container = ui.column().classes("w-full p-4")
        
        # Pagination state
        current_page = [{"value": 0}]  # Use list to make it mutable in closures
        items_per_page = 50
        
        async def update_market_data_table():
            await render_market_data_table(table_container, data_type_select.value, current_page[0]["value"], items_per_page, market_data)
        
        # Remove the old event handler since we're replacing it
        # data_type_select.on_value_change(lambda: asyncio.create_task(update_market_data_table()))
        
        # Function to show FNO modal with proper UI context
        def show_fno_modal_handler():
            """Handler for FNO modal that maintains UI context"""
            try:
                # Create modal dialog with proper sizing (not maximized)
                with ui.dialog().props("persistent").classes("fno-modal") as dialog:
                    with ui.card().classes("fno-modal-card").style("width: 90vw; max-width: 1200px; height: 80vh; background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.98)); border: 1px solid rgba(34, 197, 252, 0.3); border-radius: 16px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);"):
                        # Modal header
                        with ui.row().classes("w-full justify-between items-center p-4 border-b border-gray-700"):
                            with ui.row().classes("items-center gap-2"):
                                ui.icon("bar_chart", size="1.5rem").classes("text-cyan-400")
                                ui.label("F&O Snapshot - Live Data").classes("text-xl font-bold text-white")
                            ui.button(icon="close", on_click=dialog.close).props("flat round").classes("text-gray-400")
                        
                        # Modal content
                        modal_container = ui.column().classes("w-full h-full p-4 overflow-auto")
                        
                        # Load FNO data in modal
                        with modal_container:
                            if market_data and hasattr(market_data, 'nse_get_fno_snapshot_live'):
                                try:
                                    fno_data = market_data.nse_get_fno_snapshot_live(mode="pandas")
                                    if isinstance(fno_data, dict) and fno_data.get('data'):
                                        fno_data = pd.DataFrame(fno_data['data'])
                                    
                                    if isinstance(fno_data, pd.DataFrame) and not fno_data.empty:
                                        # Enhanced columns for modal view
                                        columns = [
                                            {"name": "symbol", "label": "Symbol", "field": "symbol", "align": "left", "sortable": True},
                                            {"name": "ltp", "label": "LTP", "field": "lastPrice", "align": "right", "sortable": True},
                                            {"name": "change", "label": "Change", "field": "change", "align": "right", "sortable": True},
                                            {"name": "change_pct", "label": "Change %", "field": "pChange", "align": "right", "sortable": True},
                                            {"name": "volume", "label": "Volume", "field": "totalTradedVolume", "align": "right", "sortable": True},
                                            {"name": "high", "label": "High", "field": "high", "align": "right", "sortable": True},
                                            {"name": "low", "label": "Low", "field": "low", "align": "right", "sortable": True},
                                            {"name": "open", "label": "Open", "field": "open", "align": "right", "sortable": True}
                                        ]
                                        
                                        rows = fno_data.to_dict('records')
                                        
                                        # Format data with color coding
                                        for row in rows:
                                            # Price formatting
                                            for key in ['lastPrice', 'change', 'high', 'low', 'open']:
                                                if key in row and row[key] is not None:
                                                    try:
                                                        value = float(row[key])
                                                        row[key] = f"₹{value:,.2f}"
                                                    except (ValueError, TypeError):
                                                        row[key] = "₹0.00"
                                            
                                            # Percentage formatting (plain text to avoid JSON issues)
                                            if 'pChange' in row and row['pChange'] is not None:
                                                try:
                                                    pct_val = float(row['pChange'])
                                                    row['pChange'] = f"{pct_val:+.2f}%"
                                                except (ValueError, TypeError):
                                                    row['pChange'] = "0.00%"
                                            
                                            # Volume formatting
                                            if 'totalTradedVolume' in row and row['totalTradedVolume'] is not None:
                                                try:
                                                    vol_val = int(float(row['totalTradedVolume']))
                                                    if vol_val >= 10000000:  # 1 Cr+
                                                        row['totalTradedVolume'] = f"{vol_val/10000000:.1f}Cr"
                                                    elif vol_val >= 100000:  # 1 Lakh+
                                                        row['totalTradedVolume'] = f"{vol_val/100000:.1f}L"
                                                    else:
                                                        row['totalTradedVolume'] = f"{vol_val:,}"
                                                except (ValueError, TypeError):
                                                    row['totalTradedVolume'] = "0"
                                        
                                        # Enhanced modal table with beautiful styling
                                        table = ui.table(
                                            columns=columns, 
                                            rows=rows,
                                            pagination=50,
                                            row_key="symbol"
                                        ).classes("w-full fno-modal-table")
                                        
                                        table.props("flat dense dark virtual-scroll").style(
                                            "background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95)); "
                                            "border: 1px solid rgba(34, 197, 252, 0.3); "
                                            "border-radius: 12px; "
                                            "max-height: 60vh; "
                                            "box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);"
                                        )
                                        
                                        # Apply same beautiful styling as Live Market data table
                                        table.props("table-header-style='background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(45, 55, 72, 0.95)); color: #22d3ee; font-weight: 700; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em;'")
                                        table.props("table-style='background: transparent;'")
                                        
                                        # Add enhanced CSS styling for F&O modal table
                                        ui.add_head_html('''
                                        <style>
                                            .fno-modal-table .q-table__container { 
                                                border-radius: 12px; 
                                                overflow: hidden;
                                            }
                                            .fno-modal-table .q-table__top {
                                                background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(45, 55, 72, 0.9));
                                                border-radius: 12px 12px 0 0;
                                            }
                                            .fno-modal-table tbody tr { 
                                                transition: all 0.3s ease;
                                                border-bottom: 1px solid rgba(148, 163, 184, 0.1);
                                            }
                                            .fno-modal-table tbody tr:hover { 
                                                background: linear-gradient(135deg, rgba(34, 197, 252, 0.15), rgba(59, 130, 246, 0.15)) !important; 
                                                transform: translateY(-1px);
                                                box-shadow: 0 4px 12px rgba(34, 197, 252, 0.2);
                                            }
                                            
                                            /* Enhanced change value styling for F&O */
                                            .fno-modal-table .positive-change {
                                                color: #10b981 !important;
                                                font-weight: 700;
                                                text-shadow: 0 0 6px rgba(16, 185, 129, 0.4);
                                                background: rgba(16, 185, 129, 0.1);
                                                padding: 2px 6px;
                                                border-radius: 4px;
                                            }
                                            .fno-modal-table .negative-change {
                                                color: #f87171 !important;
                                                font-weight: 700;
                                                text-shadow: 0 0 6px rgba(248, 113, 113, 0.4);
                                                background: rgba(248, 113, 113, 0.1);
                                                padding: 2px 6px;
                                                border-radius: 4px;
                                            }
                                            .fno-modal-table .neutral-change {
                                                color: #94a3b8 !important;
                                                font-weight: 500;
                                                background: rgba(148, 163, 184, 0.1);
                                                padding: 2px 6px;
                                                border-radius: 4px;
                                            }
                                            
                                            /* Enhanced cell styling for F&O */
                                            .fno-modal-table tbody td {
                                                padding: 10px 8px;
                                                font-size: 0.875rem;
                                                color: #e2e8f0;
                                                vertical-align: middle;
                                            }
                                            
                                            /* Symbol styling for better visibility in modal */
                                            .fno-modal-table tbody td:first-child {
                                                font-weight: 600;
                                                font-size: 0.9rem;
                                                color: #22d3ee;
                                            }
                                            
                                            /* Modal enhancements */
                                            .fno-modal-card {
                                                backdrop-filter: blur(20px);
                                                -webkit-backdrop-filter: blur(20px);
                                            }
                                        </style>
                                        ''')
                                        
                                        # Add search functionality
                                        with ui.row().classes("w-full mt-4 gap-4"):
                                            search_input = ui.input("Search symbols...").classes("flex-1")
                                            ui.label(f"Total: {len(rows)} F&O instruments").classes("text-sm text-gray-400")
                                            
                                        def filter_table():
                                            search_term = search_input.value.upper() if search_input.value else ""
                                            if search_term:
                                                filtered_rows = [row for row in rows if search_term in row.get('symbol', '').upper()]
                                                table.rows = filtered_rows
                                            else:
                                                table.rows = rows
                                                
                                        search_input.on_value_change(filter_table)
                                        
                                    else:
                                        ui.label("No F&O data available").classes("text-gray-400 text-center p-8")
                                        
                                except Exception as e:
                                    logger.error(f"Error in FNO modal: {e}")
                                    ui.label(f"Error loading F&O data: {str(e)}").classes("text-red-400 p-4")
                                    ui.label("Please check if market data service is running").classes("text-gray-400 text-center text-sm")
                            else:
                                ui.label("Market data service not available").classes("text-gray-400 text-center p-8")
                                ui.label("F&O data requires MarketData module to be imported successfully").classes("text-gray-500 text-center text-sm")
                
                dialog.open()
                
            except Exception as e:
                logger.error(f"Error showing FNO modal: {e}")
                ui.notify(f"Error: {str(e)}", type="negative")
        
        # Initial table render
        await render_market_data_table(table_container, data_type_select.value, current_page[0]["value"], items_per_page, market_data)
        
        # Heatmap container below the table
        heatmap_container = ui.column().classes("w-full mt-4")
        
        async def render_index_heatmap(selected_index):
            """Render heatmap for the selected index/sector"""
            heatmap_container.clear()
            
            with heatmap_container:
                if selected_index not in ["FNO Snapshot"] and market_data:
                    try:
                        # Get data for heatmap
                        index_key = selected_index if selected_index != "Nifty 50" else "NIFTY 50"
                        index_data = market_data.get_index_stocks_data(stock_idx=index_key, as_df=True)
                        
                        if not index_data.empty and len(index_data) > 1:
                            stock_data = index_data.iloc[1:] if index_data.iloc[0].get('Symbol', '').startswith('NIFTY') else index_data
                            
                            # Take top 20 stocks for heatmap
                            top_stocks = stock_data.head(20)
                            
                            ui.label(f"{index_key} Heatmap - Top 20 Stocks").classes("text-sm font-bold text-cyan-400 mb-2")
                            
                            # Create enhanced heatmap grid with better spacing and responsiveness
                            with ui.grid(columns=5).classes("w-full gap-3").style("grid-template-columns: repeat(5, 1fr); justify-items: center; align-items: center;"):
                                for _, stock in top_stocks.iterrows():
                                    symbol = stock.get('Symbol', 'N/A')
                                    pct_change = float(stock.get('Pct_Change', 0))
                                    
                                    # Enhanced color coding with better contrast and visibility
                                    if pct_change > 3:
                                        bg_color = "linear-gradient(135deg, #059669, #047857)"  # Dark green gradient for strong gains
                                        text_color = "#ffffff"
                                        border_color = "#10b981"
                                        glow_color = "0 0 15px rgba(16, 185, 129, 0.6)"
                                    elif pct_change > 1:
                                        bg_color = "linear-gradient(135deg, #10b981, #059669)"  # Green gradient for moderate gains
                                        text_color = "#ffffff" 
                                        border_color = "#34d399"
                                        glow_color = "0 0 10px rgba(52, 211, 153, 0.5)"
                                    elif pct_change > 0:
                                        bg_color = "linear-gradient(135deg, #34d399, #10b981)"  # Light green gradient for small gains
                                        text_color = "#ffffff"
                                        border_color = "#6ee7b7"
                                        glow_color = "0 0 8px rgba(110, 231, 183, 0.4)"
                                    elif pct_change < -3:
                                        bg_color = "linear-gradient(135deg, #dc2626, #b91c1c)"  # Dark red gradient for strong losses
                                        text_color = "#ffffff"
                                        border_color = "#ef4444"
                                        glow_color = "0 0 15px rgba(239, 68, 68, 0.6)"
                                    elif pct_change < -1:
                                        bg_color = "linear-gradient(135deg, #ef4444, #dc2626)"  # Red gradient for moderate losses
                                        text_color = "#ffffff"
                                        border_color = "#f87171"
                                        glow_color = "0 0 10px rgba(248, 113, 113, 0.5)"
                                    elif pct_change < 0:
                                        bg_color = "linear-gradient(135deg, #f87171, #ef4444)"  # Light red gradient for small losses
                                        text_color = "#ffffff"
                                        border_color = "#fca5a5"
                                        glow_color = "0 0 8px rgba(252, 165, 165, 0.4)"
                                    else:
                                        bg_color = "linear-gradient(135deg, #6b7280, #4b5563)"  # Gray gradient for no change
                                        text_color = "#ffffff"
                                        border_color = "#9ca3af"
                                        glow_color = "0 0 5px rgba(156, 163, 175, 0.3)"
                                    
                                    # Enhanced heatmap tile with proper text display - FIXED
                                    with ui.card().classes("text-center transition-all duration-300 hover:scale-105 hover:shadow-lg cursor-pointer").style(f"""
                                        background: {bg_color}; 
                                        color: {text_color}; 
                                        border: 2px solid {border_color}; 
                                        border-radius: 12px; 
                                        padding: 12px 8px; 
                                        min-height: 80px; 
                                        min-width: 85px;
                                        max-height: 80px;
                                        box-shadow: {glow_color}, 0 4px 6px rgba(0, 0, 0, 0.3);
                                        display: flex; 
                                        flex-direction: column; 
                                        justify-content: center; 
                                        align-items: center;
                                        position: relative;
                                        overflow: visible;
                                        transform: translateZ(0);
                                        backface-visibility: hidden;
                                    """):
                                        # Symbol with better spacing and visibility
                                        ui.label(symbol[:7] if len(symbol) > 7 else symbol).classes("font-bold").style(f"""
                                            font-size: 11px; 
                                            line-height: 1.2; 
                                            margin-bottom: 4px; 
                                            white-space: nowrap;
                                            color: {text_color};
                                            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
                                        """)
                                        # Percentage with better visibility and formatting
                                        ui.label(f"{pct_change:+.1f}%").classes("font-bold").style(f"""
                                            font-size: 12px; 
                                            line-height: 1.1;
                                            color: {text_color};
                                            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
                                            font-weight: 700;
                                        """)
                    except Exception as e:
                        logger.error(f"Error rendering heatmap for {selected_index}: {e}")
                        ui.label("Heatmap unavailable").classes("text-gray-400 text-center text-sm")
        
        # Update heatmap when index changes
        async def update_data_and_heatmap():
            await render_market_data_table(table_container, data_type_select.value, current_page[0]["value"], items_per_page, market_data)
            await render_index_heatmap(data_type_select.value)
        
        data_type_select.on_value_change(lambda: asyncio.create_task(update_data_and_heatmap()))
        
        # Initial heatmap render
        await render_index_heatmap(data_type_select.value)


async def render_market_data_table(container, data_type, page, items_per_page, market_data):
    """Render market data table with pagination"""
    container.clear()
    
    try:
        with container:
            if not market_data:
                ui.label("Market data not available").classes("text-gray-400 text-center p-4")
                ui.label("Check MarketData import").classes("text-xs text-gray-500 text-center")
                return
                
            if data_type in ["Nifty 50", "NIFTY 50", "NIFTY NEXT 50", "NIFTY BANK", "NIFTY IT", "NIFTY SMALLCAP 100", "NIFTY MIDCAP 100", "NIFTY FMCG", "NIFTY AUTO", "NIFTY PHARMA", "NIFTY METAL"]:
                # Fetch selected index data
                try:
                    # Map display name to actual index key
                    index_key = data_type if data_type != "Nifty 50" else "NIFTY 50"
                    index_data = market_data.get_index_stocks_data(stock_idx=index_key, as_df=True)
                    if not index_data.empty and len(index_data) > 1:  # Ensure we have actual stock data, not just index
                        # Get actual stocks (skip the index row at position 0)
                        stock_data = index_data.iloc[1:] if index_data.iloc[0].get('Symbol', '').startswith('NIFTY') else index_data
                        
                        # Paginate data
                        start_idx = page * items_per_page
                        end_idx = start_idx + items_per_page
                        page_data = stock_data.iloc[start_idx:end_idx]
                        
                        if not page_data.empty:
                            # Create simplified table with basic columns
                            columns = [
                                {"name": "symbol", "label": "Symbol", "field": "Symbol"},
                                {"name": "ltp", "label": "LTP", "field": "Close"},
                                {"name": "change", "label": "Change", "field": "Day_Change"},
                                {"name": "change_pct", "label": "Change %", "field": "Pct_Change"},
                                {"name": "volume", "label": "Volume", "field": "Traded_Volume"}
                            ]
                            
                            rows = page_data.to_dict('records')
                            
                            # Format the data for display with color coding
                            for row in rows:
                                # Price formatting
                                for key in ['Close', 'Day_Change', 'High', 'Low']:
                                    if key in row and row[key] is not None:
                                        try:
                                            value = float(row[key])
                                            row[key] = f"₹{value:,.2f}"
                                        except (ValueError, TypeError):
                                            row[key] = "₹0.00"
                                
                                # Percentage formatting with CSS class assignment
                                if 'Pct_Change' in row and row['Pct_Change'] is not None:
                                    try:
                                        pct_val = float(row['Pct_Change'])
                                        row['Pct_Change'] = f"{pct_val:+.2f}%"
                                        # Add CSS class for styling (this would need custom cell rendering)
                                        row['_pct_change_class'] = 'positive-change' if pct_val >= 0 else 'negative-change'
                                    except (ValueError, TypeError):
                                        row['Pct_Change'] = "0.00%"
                                        row['_pct_change_class'] = 'neutral-change'
                                
                                # Day change with CSS class
                                if 'Day_Change' in row and row['Day_Change'] is not None:
                                    try:
                                        change_val = float(row['Day_Change'].replace('₹', '').replace(',', ''))
                                        row['Day_Change'] = f"₹{change_val:+,.2f}"
                                        row['_day_change_class'] = 'positive-change' if change_val >= 0 else 'negative-change'
                                    except (ValueError, TypeError):
                                        row['_day_change_class'] = 'neutral-change'
                                        
                                # Volume formatting
                                if 'Traded_Volume' in row and row['Traded_Volume'] is not None:
                                    try:
                                        vol_val = int(float(row['Traded_Volume']))
                                        if vol_val >= 10000000:  # 1 Cr+
                                            row['Traded_Volume'] = f"{vol_val/10000000:.1f}Cr"
                                        elif vol_val >= 100000:  # 1 Lakh+
                                            row['Traded_Volume'] = f"{vol_val/100000:.1f}L"
                                        else:
                                            row['Traded_Volume'] = f"{vol_val:,}"
                                    except (ValueError, TypeError):
                                        row['Traded_Volume'] = "0"
                            
                            # Enhanced table with stunning visual styling and gainer/loser context
                            table = ui.table(columns=columns, rows=rows).classes("w-full market-data-table enhanced-table")
                            table.props("flat dense dark virtual-scroll separator='cell'").style(
                                "background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.98)); "
                                "border: 2px solid rgba(34, 197, 252, 0.4); "
                                "border-radius: 16px; "
                                "max-height: 420px; "
                                "box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(34, 197, 252, 0.1);"
                            )
                            
                            # Enhanced table styling with modern premium theme
                            table.props("table-header-style='background: linear-gradient(135deg, rgba(30, 41, 59, 0.98), rgba(45, 55, 72, 0.98)); color: #22d3ee; font-weight: 800; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.08em; padding: 16px 12px; border-bottom: 2px solid rgba(34, 197, 252, 0.3);'")
                            table.props("table-style='background: transparent;'")
                            
                            # Apply row classes for dynamic styling
                            for i, row in enumerate(rows):
                                try:
                                    pct_change = float(row.get('Pct_Change', '0%').replace('%', '').replace('+', ''))
                                    if pct_change > 1:
                                        table.props(f"row-key='{i}' body-cell-class='gainer-row-strong'")
                                    elif pct_change > 0:
                                        table.props(f"row-key='{i}' body-cell-class='gainer-row'")
                                    elif pct_change < -1:
                                        table.props(f"row-key='{i}' body-cell-class='loser-row-strong'")
                                    elif pct_change < 0:
                                        table.props(f"row-key='{i}' body-cell-class='loser-row'")
                                    else:
                                        table.props(f"row-key='{i}' body-cell-class='neutral-row'")
                                except:
                                    pass
                            
                            # Add stunning CSS styling with advanced visual effects
                            ui.add_head_html('''
                            <style>
                                .enhanced-table .q-table__container { 
                                    border-radius: 16px; 
                                    overflow: hidden;
                                    backdrop-filter: blur(10px);
                                    -webkit-backdrop-filter: blur(10px);
                                }
                                .enhanced-table .q-table__top {
                                    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(45, 55, 72, 0.95));
                                    border-radius: 16px 16px 0 0;
                                    border-bottom: 2px solid rgba(34, 197, 252, 0.3);
                                }
                                
                                /* Enhanced row styling with animations */
                                .enhanced-table tbody tr { 
                                    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                                    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
                                    position: relative;
                                }
                                .enhanced-table tbody tr:hover { 
                                    background: linear-gradient(135deg, rgba(34, 197, 252, 0.12), rgba(59, 130, 246, 0.12)) !important; 
                                    transform: translateY(-2px) scale(1.01);
                                    box-shadow: 0 8px 25px rgba(34, 197, 252, 0.25), 0 0 15px rgba(34, 197, 252, 0.1);
                                    z-index: 10;
                                }
                                
                                /* Premium change value styling with glow effects */
                                .enhanced-table .positive-change {
                                    color: #00f5a0 !important;
                                    font-weight: 800;
                                    text-shadow: 0 0 8px rgba(0, 245, 160, 0.6);
                                    background: linear-gradient(135deg, rgba(0, 245, 160, 0.15), rgba(16, 185, 129, 0.08));
                                    padding: 4px 8px;
                                    border-radius: 6px;
                                    border: 1px solid rgba(0, 245, 160, 0.2);
                                    animation: pulse-green 2s infinite;
                                }
                                .enhanced-table .negative-change {
                                    color: #ff6b6b !important;
                                    font-weight: 800;
                                    text-shadow: 0 0 8px rgba(255, 107, 107, 0.6);
                                    background: linear-gradient(135deg, rgba(255, 107, 107, 0.15), rgba(248, 113, 113, 0.08));
                                    padding: 4px 8px;
                                    border-radius: 6px;
                                    border: 1px solid rgba(255, 107, 107, 0.2);
                                    animation: pulse-red 2s infinite;
                                }
                                .enhanced-table .neutral-change {
                                    color: #94a3b8 !important;
                                    font-weight: 600;
                                    background: rgba(148, 163, 184, 0.08);
                                    padding: 4px 8px;
                                    border-radius: 6px;
                                    border: 1px solid rgba(148, 163, 184, 0.15);
                                }
                                
                                /* Premium cell styling */
                                .enhanced-table tbody td {
                                    padding: 14px 10px;
                                    font-size: 0.9rem;
                                    color: #e2e8f0;
                                    vertical-align: middle;
                                    font-weight: 500;
                                    letter-spacing: 0.01em;
                                }
                                
                                /* Enhanced row performance styling */
                                .enhanced-table .gainer-row-strong {
                                    background: linear-gradient(90deg, rgba(0, 245, 160, 0.12), rgba(16, 185, 129, 0.03)) !important;
                                    border-left: 5px solid #00f5a0;
                                    box-shadow: inset 0 0 30px rgba(0, 245, 160, 0.08);
                                }
                                .enhanced-table .gainer-row {
                                    background: linear-gradient(90deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.02)) !important;
                                    border-left: 4px solid #10b981;
                                    box-shadow: inset 0 0 20px rgba(16, 185, 129, 0.05);
                                }
                                .enhanced-table .loser-row-strong {
                                    background: linear-gradient(90deg, rgba(255, 107, 107, 0.12), rgba(248, 113, 113, 0.03)) !important;
                                    border-left: 5px solid #ff6b6b;
                                    box-shadow: inset 0 0 30px rgba(255, 107, 107, 0.08);
                                }
                                .enhanced-table .loser-row {
                                    background: linear-gradient(90deg, rgba(248, 113, 113, 0.08), rgba(248, 113, 113, 0.02)) !important;
                                    border-left: 4px solid #f87171;
                                    box-shadow: inset 0 0 20px rgba(248, 113, 113, 0.05);
                                }
                                .enhanced-table .neutral-row {
                                    background: linear-gradient(90deg, rgba(148, 163, 184, 0.05), rgba(148, 163, 184, 0.01)) !important;
                                    border-left: 3px solid #94a3b8;
                                }
                                
                                /* Premium symbol styling */
                                .enhanced-table tbody td:first-child {
                                    font-weight: 700;
                                    font-size: 1rem;
                                    color: #22d3ee;
                                    text-shadow: 0 0 4px rgba(34, 211, 238, 0.3);
                                }
                                
                                /* Animations */
                                @keyframes pulse-green {
                                    0%, 100% { box-shadow: 0 0 5px rgba(0, 245, 160, 0.3); }
                                    50% { box-shadow: 0 0 10px rgba(0, 245, 160, 0.5), 0 0 15px rgba(0, 245, 160, 0.2); }
                                }
                                @keyframes pulse-red {
                                    0%, 100% { box-shadow: 0 0 5px rgba(255, 107, 107, 0.3); }
                                    50% { box-shadow: 0 0 10px rgba(255, 107, 107, 0.5), 0 0 15px rgba(255, 107, 107, 0.2); }
                                }
                            </style>
                            ''')
                            ui.label(f"Showing {len(rows)} of {len(stock_data)} {index_key} stocks").classes("text-xs text-gray-500 mt-2")
                        else:
                            ui.label("No data in current page").classes("text-gray-400 text-center p-4")
                    else:
                        ui.label(f"No {index_key} data available").classes("text-gray-400 text-center p-4")
                except Exception as e:
                    logger.error(f"Error fetching {index_key} data: {e}")
                    ui.label(f"Error loading {index_key} data").classes("text-red-400 text-center p-4")
                    ui.label(f"Error: {str(e)[:50]}...").classes("text-xs text-red-300 text-center")
            
            elif data_type == "FNO Snapshot":
                # Fetch FNO snapshot data
                try:
                    fno_data = market_data.nse_get_fno_snapshot_live(mode="pandas")
                    # Handle both dict and DataFrame responses
                    if isinstance(fno_data, dict):
                        if fno_data.get('data'):
                            fno_data = pd.DataFrame(fno_data['data'])
                        else:
                            fno_data = pd.DataFrame()
                    
                    if isinstance(fno_data, pd.DataFrame) and not fno_data.empty:
                        # Paginate data
                        start_idx = page * items_per_page
                        end_idx = start_idx + items_per_page
                        page_data = fno_data.iloc[start_idx:end_idx]
                    
                    # Create table without lambda functions to avoid JSON serialization error
                    columns = [
                        {"name": "symbol", "label": "Symbol", "field": "symbol", "required": True, "align": "left"},
                        {"name": "ltp", "label": "LTP", "field": "lastPrice", "required": True, "align": "right"},
                        {"name": "change", "label": "Change", "field": "change", "required": True, "align": "right"},
                        {"name": "change_pct", "label": "Change %", "field": "pChange", "required": True, "align": "right"},
                        {"name": "volume", "label": "Volume", "field": "totalTradedVolume", "required": True, "align": "right"},
                        {"name": "high", "label": "High", "field": "high", "required": True, "align": "right"},
                        {"name": "low", "label": "Low", "field": "low", "required": True, "align": "right"}
                    ]
                    
                    rows = page_data.to_dict('records')
                    
                    # Format the data for display with color coding
                    for row in rows:
                        # Price formatting
                        for key in ['lastPrice', 'high', 'low']:
                            if key in row and row[key] is not None:
                                try:
                                    value = float(row[key])
                                    row[key] = f"₹{value:,.2f}"
                                except (ValueError, TypeError):
                                    row[key] = "₹0.00"
                        
                        # Change with CSS class assignment
                        if 'change' in row and row['change'] is not None:
                            try:
                                change_val = float(row['change'])
                                row['change'] = f"₹{change_val:+,.2f}"
                                row['_change_class'] = 'positive-change' if change_val >= 0 else 'negative-change'
                            except (ValueError, TypeError):
                                row['change'] = "₹0.00"
                                row['_change_class'] = 'neutral-change'
                        
                        # Percentage with CSS class
                        if 'pChange' in row and row['pChange'] is not None:
                            try:
                                pct_val = float(row['pChange'])
                                row['pChange'] = f"{pct_val:+.2f}%"
                                row['_pchange_class'] = 'positive-change' if pct_val >= 0 else 'negative-change'
                            except (ValueError, TypeError):
                                row['pChange'] = "0.00%"
                                row['_pchange_class'] = 'neutral-change'
                                
                        # Volume formatting
                        if 'totalTradedVolume' in row and row['totalTradedVolume'] is not None:
                            try:
                                vol_val = int(float(row['totalTradedVolume']))
                                if vol_val >= 10000000:  # 1 Cr+
                                    row['totalTradedVolume'] = f"{vol_val/10000000:.1f}Cr"
                                elif vol_val >= 100000:  # 1 Lakh+
                                    row['totalTradedVolume'] = f"{vol_val/100000:.1f}L"
                                else:
                                    row['totalTradedVolume'] = f"{vol_val:,}"
                            except (ValueError, TypeError):
                                row['totalTradedVolume'] = "0"
                    
                        # Enhanced FNO table with proper theme styling
                        table = ui.table(columns=columns, rows=rows).classes("w-full market-data-table")
                        table.props("flat bordered dense dark virtual-scroll").style(
                            "background: rgba(15, 23, 42, 0.6); "
                            "border: 1px solid rgba(148, 163, 184, 0.2); "
                            "border-radius: 8px; "
                            "max-height: 400px;"
                        )
                        
                        # Custom table styling
                        table.props("table-header-style='background: rgba(30, 41, 59, 0.8); color: #cbd5e1; font-weight: 600;'")
                        table.props("table-style='background: rgba(15, 23, 42, 0.6);'")
                        ui.label(f"Showing {len(rows)} of {len(fno_data)} FNO stocks").classes("text-xs text-gray-500 mt-2")
                    else:
                        ui.label("No FNO data available").classes("text-gray-400 text-center p-4")
                        ui.label("Check market_data.py connection").classes("text-xs text-gray-500 text-center")
                except Exception as e:
                    logger.error(f"Error fetching FNO data: {e}")
                    ui.label("Error loading FNO data").classes("text-red-400 text-center p-4")
                    ui.label(f"Error: {str(e)[:50]}...").classes("text-xs text-red-300 text-center")
    
    except Exception as e:
        logger.error(f"Error rendering market data table: {e}")
        with container:
            ui.label("Error loading market data").classes("text-red-500 text-center p-4")

# setup_dashboard_updates function removed - unused


