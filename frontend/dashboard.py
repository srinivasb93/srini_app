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

# Import existing modules to utilize their functionality
from order_management import render_order_management
from analytics import render_analytics_page
from watchlist import render_watchlist_page
from portfolio import render_portfolio_page
from orderbook import render_order_book_page
from positions import render_positions_page
from strategies import render_strategies_page
from livetrading import render_live_trading_page


logger = logging.getLogger(__name__)

# Global state variables for dashboard data
dashboard_state = {
    "funds_data": {},
    "portfolio_data": [],
    "positions_data": [],
    "watchlist_data": [],
    "order_book_data": [],
    "strategies_data": [],
    "live_trades_data": [],
    "market_news_data": []
}


async def render_dashboard_page(fetch_api, user_storage, get_cached_instruments):
    """Main enhanced dashboard page with fixed layout"""

    broker = user_storage.get('default_broker', 'Zerodha')

    # CRITICAL: Main dashboard container with proper viewport handling
    with ui.column().classes("w-full"):
        # Enhanced Dashboard Title
        render_enhanced_dashboard_title(broker)

        # FIXED: Responsive grid layout that prevents truncation
        with ui.row().classes("w-full gap-4 p-4").style("min-height: calc(100vh - 120px); overflow: visible;"):
            # Left Panel - Watchlist and Quick Trade (flexible width)
            with ui.column().classes("flex-none").style("width: 300px; min-width: 280px;"):
                await render_enhanced_watchlist_section(fetch_api, user_storage, get_cached_instruments, broker)
                await render_enhanced_quick_trade_section(fetch_api, user_storage, get_cached_instruments, broker)

            # Center Panel - Portfolio Overview (expandable)
            with ui.column().classes("flex-1").style("min-width: 400px;"):
                await render_enhanced_portfolio_section(fetch_api, user_storage, broker)

            # Right Panel - Orders, Strategies, Market Summary (flexible width)
            with ui.column().classes("flex-none").style("width: 300px; min-width: 280px;"):
                await render_enhanced_order_book_section(fetch_api, user_storage, broker)
                await render_enhanced_strategies_section(fetch_api, user_storage, broker)
                await render_enhanced_market_summary_section(fetch_api, user_storage, broker)

    # Initialize real-time updates
    await setup_dashboard_updates(fetch_api, user_storage, get_cached_instruments, broker)


def render_enhanced_dashboard_title(broker):
    """Enhanced dashboard title with status indicators"""
    with ui.row().classes("page-header-standard w-full justify-between items-center"):
        # Left side - Title and subtitle
        with ui.column().classes("gap-2"):
            with ui.row().classes("items-center gap-1"):
                ui.icon("candlestick_chart", size="1rem").classes("text-cyan-400")
                ui.label(f"{broker} Trading Dashboard").classes("page-title-standard theme-header-text")
                ui.chip("LIVE", color="green").classes("text-xs status-chip")

            ui.label("Real-time market data and portfolio management").classes("page-subtitle-standard")

async def render_enhanced_watchlist_section(fetch_api, user_storage, get_cached_instruments, broker):
    """Enhanced watchlist section using existing watchlist.py functionality"""

    with ui.card().classes("dashboard-card watchlist-card w-full"):
        # Header
        with ui.row().classes("card-header w-full justify-between items-center p-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("visibility", size="1.5rem").classes("text-cyan-400")
                ui.label("Watchlist").classes("card-title")

            with ui.row().classes("items-center gap-2"):
                ui.button(icon="add").props("flat round").classes("text-cyan-400 add-button")
                ui.button(icon="refresh").props("flat round").classes("text-gray-400 refresh-button")

        ui.separator().classes("card-separator")

        # Watchlist content container
        watchlist_container = ui.column().classes("watchlist-content w-full p-2")

        # Get instruments and watchlist data
        try:
            all_instruments_map = await get_cached_instruments(broker)
            watchlist_symbols = user_storage.get("STORAGE_WATCHLIST_KEY", [])

            with watchlist_container:
                if not watchlist_symbols:
                    ui.label("Your watchlist is empty").classes("text-gray-500 text-center p-4")
                else:
                    for symbol in watchlist_symbols:
                        await render_enhanced_watchlist_item(symbol, all_instruments_map, fetch_api, broker)

        except Exception as e:
            logger.error(f"Error rendering watchlist: {e}")
            with watchlist_container:
                ui.label("Error loading watchlist").classes("text-red-500 text-center p-4")

        return watchlist_container


async def render_enhanced_watchlist_item(symbol, instruments_map, fetch_api, broker):
    """Render individual enhanced watchlist item"""

    try:
        # Get LTP data
        instrument_token = instruments_map.get(symbol)
        price = 0.0
        change = 0.0
        change_pct = 0.0

        if instrument_token:
            ltp_response = await fetch_api(f"/ltp/{broker}", params={"instruments": instrument_token})
            if ltp_response and isinstance(ltp_response, list) and ltp_response:
                ltp_data = ltp_response[0]
                price = ltp_data.get('last_price', 0.0)
                change = ltp_data.get('net_change', 0.0)
                change_pct = ltp_data.get('percentage_change', 0.0)

        # Determine styling based on change
        change_class = "positive-change" if change > 0 else "negative-change" if change < 0 else "neutral-change"
        trend_icon = "trending_up" if change > 0 else "trending_down" if change < 0 else "trending_flat"
        border_class = "watchlist-positive" if change > 0 else "watchlist-negative" if change < 0 else "watchlist-neutral"

        with ui.row().classes(f"watchlist-item {border_class} w-full justify-between items-center p-3 mb-2"):
            # Left side - Symbol and price
            with ui.column().classes("gap-1 flex-1"):
                ui.label(symbol).classes("font-semibold theme-text-primary text-sm symbol-text")
                ui.label(f"₹{price:,.2f}").classes("theme-text-secondary text-xs price-text")

            # Right side - Change and percentage
            with ui.column().classes("gap-1 items-end"):
                with ui.row().classes("items-center gap-1"):
                    ui.icon(trend_icon, size="0.75rem").classes(change_class)
                    ui.label(f"{change:+.2f}").classes(f"text-sm {change_class} change-text")
                ui.label(f"({change_pct:+.2f}%)").classes(f"text-xs {change_class} change-pct-text")

    except Exception as e:
        logger.error(f"Error rendering watchlist item {symbol}: {e}")
        with ui.row().classes("watchlist-item watchlist-error w-full p-3 mb-2"):
            ui.label(symbol).classes("theme-text-primary")
            ui.label("Error").classes("text-red-500")


async def render_enhanced_chart_section(fetch_api, user_storage, instruments_map, broker):
    """Enhanced chart section using existing analytics.py functionality"""

    with ui.card().classes("dashboard-card chart-card w-full"):
        # Chart Header
        with ui.row().classes("card-header w-full justify-between items-center p-4"):
            with ui.row().classes("items-center gap-4"):
                ui.icon("show_chart", size="1.5rem").classes("text-green-400")
                ui.label("Live Trading Chart").classes("card-title")

                # Symbol selector
                symbol_select = ui.select(
                    options=["NIFTY50", "BANKNIFTY", "RELIANCE", "TCS", "INFY"],
                    value="NIFTY50"
                ).classes("chart-symbol-select").props("outlined dense dark")

            # Timeframe buttons
            with ui.row().classes("chart-timeframe-buttons gap-1"):
                timeframes = ["1m", "5m", "15m", "1h", "1d"]
                for i, tf in enumerate(timeframes):
                    active_class = "timeframe-active" if tf == "5m" else "timeframe-inactive"
                    ui.button(tf).classes(f"timeframe-btn {active_class}")

        ui.separator().classes("card-separator")

        # Chart container
        chart_container = ui.column().classes("chart-content w-full")

        with chart_container:
            # Create lightweight trading chart using existing analytics functionality
            await create_dashboard_chart(fetch_api, user_storage, instruments_map, broker)

        return chart_container


async def create_dashboard_chart(fetch_api, user_storage, instruments_map, broker):
    """Create dashboard chart using existing analytics functionality"""

    try:
        # Use existing analytics chart but in compact dashboard mode
        with ui.row().classes("chart-display w-full h-80 items-center justify-center"):
            # This would integrate with your existing analytics.py chart
            # For now, showing a placeholder that matches your reference design
            from analytics import render_analytics_page

            # Create a compact version of the analytics chart
            with ui.element('div').classes('chart-container'):
                # This would call a simplified version of your analytics chart
                await render_analytics_page(fetch_api, user_storage, instruments_map, broker)

        # Chart controls
        with ui.row().classes("chart-controls w-full justify-center gap-2 mt-2"):
            ui.button("Fullscreen Chart", icon="fullscreen").classes("chart-control-btn")
            ui.button("Indicators", icon="show_chart").classes("chart-control-btn")
            ui.button("Drawing Tools", icon="edit").classes("chart-control-btn")

    except Exception as e:
        logger.error(f"Error creating dashboard chart: {e}")
        ui.label("Error loading chart").classes("text-red-500 text-center p-4")


async def render_enhanced_portfolio_section(fetch_api, user_storage, broker):
    """Enhanced portfolio section using existing portfolio.py functionality"""

    with ui.card().classes("dashboard-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center p-2"):
            ui.icon("account_balance_wallet", size="1.5rem").classes("text-green-400")
            ui.label("Portfolio Overview").classes("card-title")

        ui.separator().classes("card-separator")

        # Portfolio metrics container
        portfolio_metrics_container = ui.row().classes("w-full p-2 gap-2")

        try:
            # Fetch portfolio data using existing API calls
            funds_data = await fetch_api(f"/funds/{broker}")
            portfolio_data = await fetch_api(f"/portfolio/{broker}") or []
            positions_data = await fetch_api(f"/positions/{broker}") or []

            # Calculate metrics using existing logic
            available_funds = "N/A"
            if funds_data and isinstance(funds_data, dict):
                equity = funds_data.get('equity', {})
                available = equity.get('available', 0.0)
                available_funds = f"₹{float(available):,.2f}" if isinstance(available, (int, float)) else "N/A"

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

            open_positions = len([p for p in positions_data if isinstance(p, dict) and p.get("Quantity", 0) != 0])

            with portfolio_metrics_container:
                # Available Funds
                with ui.column().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("account_balance", size="2rem").classes("text-blue-400 mb-2")
                        ui.label("Available Funds").classes("text-sm text-gray-400")
                        ui.label(available_funds).classes("text-2xl font-bold theme-text-primary")
                        ui.label("Cash Balance").classes("text-sm text-gray-400")

                # Invested Value
                with ui.column().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("currency_rupee", size="2rem").classes("text-purple-400 mb-2")
                        ui.label("Invested Value").classes("text-sm text-gray-400")
                        ui.label(f"₹{total_invested:,.2f}").classes("text-2xl font-bold theme-text-primary")
                        ui.label("Current Investment").classes("text-sm text-gray-400")

                # Portfolio Value
                with ui.column().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("account_balance_wallet", size="2rem").classes("text-cyan-400 mb-2")
                        ui.label("Portfolio Value").classes("text-sm text-gray-400")
                        ui.label(f"₹{portfolio_value:,.2f}").classes("text-2xl font-bold theme-text-primary")
                        ui.label("Current Holdings").classes("text-sm text-gray-400")

                # Overall P&L
                total_pnl_color = "text-green-400" if total_pnl >= 0 else "text-red-400"
                pnl_icon = "trending_up" if total_pnl >= 0 else "trending_down"
                with ui.column().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon(pnl_icon, size="2rem").classes(f"{total_pnl_color} mb-2")
                        ui.label("Overall P&L").classes("text-sm text-gray-400")
                        ui.label(f"₹{total_pnl:,.2f}").classes(f"text-2xl font-bold {total_pnl_color}")
                        ui.label(f"({total_pnl_pct:+.2f}%)").classes(f"text-sm {total_pnl_color}")

                # Daily P&L
                daily_pnl_color = "text-green-400" if daily_pnl >= 0 else "text-red-400"
                daily_pnl_icon = "trending_up" if daily_pnl >= 0 else "trending_down"
                with ui.column().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon(daily_pnl_icon, size="2rem").classes(f"{daily_pnl_color} mb-2")
                        ui.label("Daily P&L").classes("text-sm text-gray-400")
                        ui.label(f"₹{daily_pnl:,.2f}").classes(f"text-2xl font-bold {daily_pnl_color}")
                        ui.label(f"({daily_pnl_pct:+.2f}%)").classes(f"text-sm {daily_pnl_color}")

                # Open Positions
                with ui.column().classes("dashboard-card metric-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("donut_small", size="2rem").classes("text-orange-400 mb-2")
                        ui.label("Open Positions").classes("text-sm text-gray-400")
                        ui.label(str(open_positions)).classes("text-2xl font-bold theme-text-primary")
                        ui.label("Active Trades").classes("text-sm text-gray-400")

        except Exception as e:
            logger.error(f"Error loading portfolio data: {e}")
            with portfolio_metrics_container:
                ui.label("Error loading portfolio data").classes("text-red-500 text-center p-4")


async def render_enhanced_quick_trade_section(fetch_api, user_storage, get_cached_instruments, broker):
    """Enhanced quick trade section"""

    with ui.card().classes("dashboard-card trading-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center p-4"):
            ui.icon("flash_on", size="1.5rem").classes("text-yellow-400")
            ui.label("Quick Trade").classes("card-title")

        ui.separator().classes("card-separator")

        # Quick trade form
        with ui.column().classes("quick-trade-form p-4 gap-3"):
            # Symbol and quantity row
            with ui.row().classes("w-full gap-2"):
                symbol_input = ui.input("Symbol", placeholder="RELIANCE").classes("flex-1")
                quantity_input = ui.number("Qty", value=1, min=1).classes("w-24")

            # Order type and product row
            with ui.row().classes("w-full gap-2"):
                order_type = ui.select(["MARKET", "LIMIT"], value="MARKET").classes("flex-1")
                product_type = ui.select(["CNC", "MIS"] if broker == "Zerodha" else ["D", "I"],
                                       value="CNC" if broker == "Zerodha" else "D").classes("flex-1")

            # Price input (conditional)
            price_input = ui.number("Price", value=0, step=0.05).classes("w-full")
            price_input.visible = False

            def toggle_price_input():
                price_input.visible = order_type.value == "LIMIT"

            order_type.on_value_change(toggle_price_input)

            # Action buttons
            with ui.row().classes("trade-buttons w-full gap-2 mt-3"):
                ui.button("BUY", on_click=lambda: place_quick_order("BUY")).classes("buy-button flex-1")
                ui.button("SELL", on_click=lambda: place_quick_order("SELL")).classes("sell-button flex-1")

        async def place_quick_order(transaction_type):
            """Place quick order"""
            if not symbol_input.value:
                ui.notify("Please enter a symbol", type="warning")
                return

            # Get instrument token
            instruments = await get_cached_instruments(broker)
            if symbol_input.value not in instruments:
                ui.notify("Symbol not found", type="warning")
                return

            order_data = {
                "trading_symbol": symbol_input.value,
                "instrument_token": instruments[symbol_input.value],
                "quantity": int(quantity_input.value),
                "transaction_type": transaction_type,
                "order_type": order_type.value,
                "product_type": product_type.value,
                "price": float(price_input.value) if order_type.value == "LIMIT" else 0,
                "validity": "DAY",
                "broker": broker
            }

            response = await fetch_api("/orders", method="POST", data=order_data)
            if response and response.get('order_id'):
                ui.notify(f"Order placed: {response['order_id']}", type='positive')
                # Clear form
                symbol_input.value = ""
                quantity_input.value = 1
                price_input.value = 0
            else:
                ui.notify("Failed to place order", type='negative')


async def render_enhanced_order_book_section(fetch_api, user_storage, broker):
    """Enhanced order book section"""

    with ui.card().classes("dashboard-card orders-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center p-4"):
            ui.icon("list_alt", size="1.5rem").classes("text-blue-400")
            ui.label("Recent Orders").classes("card-title")

        ui.separator().classes("card-separator")

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
        with ui.row().classes("card-header w-full items-center p-4"):
            ui.icon("psychology", size="1.5rem").classes("text-purple-400")
            ui.label("Active Strategies").classes("card-title")

        ui.separator().classes("card-separator")

        # Strategies container
        strategies_container = ui.column().classes("w-full p-4 gap-2")

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


async def render_enhanced_market_summary_section(fetch_api, user_storage, broker):
    """Enhanced market summary section with proper height management"""

    with ui.card().classes("dashboard-card market-summary-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center p-4"):
            ui.icon("trending_up", size="1.5rem").classes("text-green-400")
            ui.label("Market Summary").classes("card-title")

        ui.separator().classes("card-separator")

        # Market data container with proper height constraints
        market_container = ui.column().classes("w-full p-4 gap-3 max-h-80 overflow-y-auto")

        try:
            # Market indices data
            indices_data = [
                {"name": "NIFTY 50", "value": 21736.60, "change": +234.50, "change_pct": +1.09},
                {"name": "SENSEX", "value": 72240.26, "change": +759.49, "change_pct": +1.06},
                {"name": "NIFTY BANK", "value": 46816.35, "change": +682.25, "change_pct": +1.48},
                {"name": "NIFTY IT", "value": 36245.80, "change": -125.30, "change_pct": -0.34},
                {"name": "NIFTY AUTO", "value": 18963.45, "change": +289.15, "change_pct": +1.55}
            ]

            with market_container:
                ui.label("Major Indices").classes("theme-text-primary font-semibold mb-2")

                for index in indices_data:
                    change_color = "text-green-400" if index["change"] > 0 else "text-red-400"
                    change_icon = "▲" if index["change"] > 0 else "▼"

                    with ui.row().classes("w-full justify-between items-center py-2"):
                        with ui.column().classes("flex-1"):
                            ui.label(index["name"]).classes("theme-text-primary font-medium text-sm")
                            ui.label(f"{index['value']:,.2f}").classes("text-gray-300 text-sm")

                        with ui.column().classes("items-end"):
                            ui.label(f"{change_icon} {abs(index['change']):,.2f}").classes(f"{change_color} text-sm font-semibold")
                            ui.label(f"({index['change_pct']:+.2f}%)").classes(f"{change_color} text-xs")

                # Sector performance
                ui.separator().classes("my-3")
                ui.label("Top Movers").classes("theme-text-primary font-semibold mb-2")

                movers_data = [
                    {"symbol": "RELIANCE", "change_pct": +2.85, "price": 2456.30},
                    {"symbol": "TCS", "change_pct": -1.24, "price": 3789.45},
                    {"symbol": "HDFC BANK", "change_pct": +1.67, "price": 1543.20},
                    {"symbol": "INFOSYS", "change_pct": -0.89, "price": 1678.90}
                ]

                for mover in movers_data:
                    change_color = "text-green-400" if mover["change_pct"] > 0 else "text-red-400"
                    change_icon = "▲" if mover["change_pct"] > 0 else "▼"

                    with ui.row().classes("w-full justify-between items-center py-1"):
                        with ui.column().classes("flex-1"):
                            ui.label(mover["symbol"]).classes("theme-text-primary font-medium text-sm")
                            ui.label(f"₹{mover['price']:,.2f}").classes("text-gray-300 text-xs")

                        ui.label(f"{change_icon} {abs(mover['change_pct']):.2f}%").classes(f"{change_color} text-sm font-semibold")

                # Market status at bottom with proper padding
                ui.separator().classes("my-3")

                market_open = 9 <= datetime.now().hour < 16
                status_text = "Market Open" if market_open else "Market Closed"
                status_color = "text-green-400" if market_open else "text-red-400"

                with ui.row().classes("w-full justify-center items-center p-2"):
                    ui.icon("circle", size="0.5rem").classes(status_color)
                    ui.label(status_text).classes(f"{status_color} font-semibold text-sm ml-2")

        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            with market_container:
                ui.label("Error loading market data").classes("text-red-500 text-center p-4")
async def setup_dashboard_updates(fetch_api, user_storage, get_cached_instruments, broker):
    """Setup real-time dashboard updates"""

    async def update_dashboard_data():
        """Update all dashboard data"""
        try:
            # Update global dashboard state with real data
            dashboard_state["funds_data"] = await fetch_api(f"/funds/{broker}")
            dashboard_state["portfolio_data"] = await fetch_api(f"/portfolio/{broker}") or []
            dashboard_state["positions_data"] = await fetch_api(f"/positions/{broker}") or []

            # Update watchlist LTPs
            all_instruments_map = await get_cached_instruments(broker)
            watchlist_symbols = user_storage.get("STORAGE_WATCHLIST_KEY",
                                                 ["NIFTY50", "BANKNIFTY", "RELIANCE", "TCS", "INFY"])

            for symbol in watchlist_symbols:
                if symbol in all_instruments_map:
                    instrument_token = all_instruments_map[symbol]
                    try:
                        ltp_response = await fetch_api(f"/ltp/{broker}", params={"instruments": instrument_token})
                        if ltp_response and isinstance(ltp_response, list) and ltp_response:
                            # Update watchlist data in global state
                            # This would trigger UI updates in a production system
                            pass
                    except Exception as e:
                        logger.error(f"Error updating LTP for {symbol}: {e}")

            logger.debug("Dashboard data updated successfully")

        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")

    # Initial data load
    await update_dashboard_data()

    # Setup periodic updates (30 seconds like existing dashboard)
    def is_client_valid():
        """Check if the current client is still valid"""
        try:
            from nicegui import context
            return hasattr(context, 'client') and context.client and not context.client.is_deleted
        except Exception:
            return False

    # Use this before creating UI elements:
    if is_client_valid():
        ui.timer(300.0, update_dashboard_data)


# Utility functions for dashboard functionality

def format_currency(value):
    """Format currency values consistently"""
    if isinstance(value, (int, float)):
        return f"���{value:,.2f}"
    return "₹0.00"


def format_percentage(value):
    """Format percentage values consistently"""
    if isinstance(value, (int, float)):
        return f"{value:+.2f}%"
    return "0.00%"


def get_change_class(value):
    """Get CSS class based on value change"""
    if value > 0:
        return "positive-change"
    elif value < 0:
        return "negative-change"
    else:
        return "neutral-change"


def get_trend_icon(value):
    """Get trend icon based on value"""
    if value > 0:
        return "trending_up"
    elif value < 0:
        return "trending_down"
    else:
        return "trending_flat"