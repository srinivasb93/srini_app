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
from order_management import render_regular_orders
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
    """Main enhanced dashboard page"""

    broker = user_storage.get('broker', 'Zerodha')

    # Apply enhanced dashboard theme
    apply_enhanced_dashboard_styles()

    # Main dashboard container
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):
        # Enhanced Dashboard Title
        render_enhanced_dashboard_title(broker)

        # Main Dashboard Grid Layout (matches reference image)
        with ui.element('div').classes("dashboard-grid w-full p-4"):
            # Left Panel - Enhanced Watchlist (25% width)
            with ui.element('div').classes("dashboard-left-panel"):
                await render_enhanced_watchlist_section(fetch_api, user_storage, get_cached_instruments, broker)
                await render_enhanced_quick_trade_section(fetch_api, user_storage, get_cached_instruments, broker)

            # Center Panel - Chart and Portfolio Overview (50% width)
            with ui.element('div').classes("dashboard-center-panel"):
                # await render_enhanced_chart_section(fetch_api, user_storage, get_cached_instruments,broker)
                await render_enhanced_portfolio_section(fetch_api, user_storage, broker)

            # Right Panel - Quick Trade, Orders, Strategies, News (25% width)
            with ui.element('div').classes("dashboard-right-panel"):
                await render_enhanced_order_book_section(fetch_api, user_storage, broker)
                await render_enhanced_strategies_section(fetch_api, user_storage, broker)
                await render_enhanced_market_summary_section(fetch_api, user_storage, broker)
                # await render_enhanced_news_section()

    # Initialize real-time updates
    await setup_dashboard_updates(fetch_api, user_storage, get_cached_instruments, broker)


def apply_enhanced_dashboard_styles():
    """Apply enhanced CSS styles for the dashboard"""
    ui.add_css('static/styles.css')

def render_enhanced_dashboard_title(broker):
    """Enhanced dashboard title with status indicators"""
    with ui.row().classes("dashboard-title-section w-full justify-between items-center p-2"):
        # Left side - Title and subtitle
        with ui.column().classes("gap-2"):
            with ui.row().classes("items-center gap-1"):
                ui.icon("candlestick_chart", size="1rem").classes("text-cyan-400")
                ui.label(f"{broker} Trading Dashboard").classes("text-3xl font-bold text-white dashboard-title")
                ui.chip("LIVE", color="green").classes("text-xs status-chip")

            ui.label("Real-time market data and portfolio management").classes("text-gray-400 dashboard-subtitle")

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
                ui.label(symbol).classes("font-semibold text-white text-sm symbol-text")
                ui.label(f"₹{price:,.2f}").classes("text-gray-300 text-xs price-text")

            # Right side - Change and percentage
            with ui.column().classes("gap-1 items-end"):
                with ui.row().classes("items-center gap-1"):
                    ui.icon(trend_icon, size="0.75rem").classes(change_class)
                    ui.label(f"{change:+.2f}").classes(f"text-sm {change_class} change-text")
                ui.label(f"({change_pct:+.2f}%)").classes(f"text-xs {change_class} change-pct-text")

    except Exception as e:
        logger.error(f"Error rendering watchlist item {symbol}: {e}")
        with ui.row().classes("watchlist-item watchlist-error w-full p-3 mb-2"):
            ui.label(symbol).classes("text-white")
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

    with ui.card().classes("dashboard-card portfolio-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center p-2"):
            ui.icon("account_balance_wallet", size="1.5rem").classes("text-green-400")
            ui.label("Portfolio Overview").classes("card-title")

        ui.separator().classes("card-separator")

        # Portfolio metrics container
        portfolio_metrics_container = ui.row().classes("portfolio-metrics w-full p-2 gap-2")

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
                with ui.column().classes("metric-card flex-1"):
                    ui.label("Available Funds").classes("metric-label")
                    ui.label(available_funds).classes("metric-value")
                    ui.label("Cash Balance").classes("metric-sublabel")

                # Invested Value
                with ui.column().classes("metric-card flex-1"):
                    ui.label("Invested Value").classes("metric-label")
                    ui.label(f"₹{total_invested:,.2f}").classes("metric-value")
                    ui.label("Current Investment").classes("metric-sublabel")

                # Portfolio Value
                with ui.column().classes("metric-card flex-1"):
                    ui.label("Portfolio Value").classes("metric-label")
                    ui.label(f"₹{portfolio_value:,.2f}").classes("metric-value")
                    ui.label("Current Holdings").classes("metric-sublabel")

                # Overall P&L
                total_pnl_class = "positive-change" if total_pnl > 0 else "negative-change"
                with ui.column().classes("metric-card flex-1"):
                    ui.label("Overall P&L").classes("metric-label")
                    ui.label(f"₹{total_pnl:,.2f}").classes(f"metric-value {total_pnl_class}")
                    ui.label(f"({total_pnl_pct:,.2f}%)").classes(f"metric-sublabel {total_pnl_class}")

                # Daily P&L
                daily_pnl_class = "positive-change" if daily_pnl > 0 else "negative-change"
                with ui.column().classes("metric-card flex-1"):
                    ui.label("Daily P&L").classes("metric-label")
                    ui.label(f"₹{daily_pnl:,.2f}").classes(f"metric-value {daily_pnl_class}")
                    ui.label(f"({daily_pnl_pct:,.2f}%)").classes(f"metric-sublabel {daily_pnl_class}")

                # Open Positions
                with ui.column().classes("metric-card flex-1"):
                    ui.label("Open Positions").classes("metric-label")
                    ui.label(str(open_positions)).classes("metric-value")
                    ui.label("Active Trades").classes("metric-sublabel")

        except Exception as e:
            logger.error(f"Error rendering portfolio section: {e}")
            with portfolio_metrics_container:
                ui.label("Error loading portfolio data").classes("text-red-500 text-center p-4")

        return portfolio_metrics_container


async def render_enhanced_quick_trade_section(fetch_api, user_storage, get_cached_instruments, broker):
    """Enhanced quick trade section using existing order_management.py functionality"""

    with ui.card().classes("dashboard-card quick-trade-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center p-4"):
            ui.icon("bolt", size="1.5rem").classes("text-yellow-400")
            ui.label("Quick Trade").classes("card-title")

        ui.separator().classes("card-separator")

        # Quick trade form
        with ui.column().classes("quick-trade-form w-full p-4 gap-3"):
            # Symbol selection
            symbol_select = ui.select(
                options=["NIFTY50", "BANKNIFTY", "RELIANCE", "TCS", "INFY"],
                value="NIFTY50",
                label="Symbol"
            ).classes("trade-input").props("outlined dark")

            # Order type
            order_type_select = ui.select(
                options=["LIMIT", "MARKET", "SL", "SL-M"],
                value="LIMIT",
                label="Order Type"
            ).classes("trade-input").props("outlined dark")

            # Quantity
            quantity_input = ui.number(
                label="Quantity",
                value=100,
                min=1
            ).classes("trade-input").props("outlined dark")

            # Price (for limit orders)
            price_input = ui.number(
                label="Price",
                value=19850.75,
                step=0.01
            ).classes("trade-input").props("outlined dark")

            # Action buttons
            with ui.row().classes("trade-buttons w-full gap-2 mt-4"):
                all_instruments_map = await get_cached_instruments(broker)
                async def place_quick_order(buy_sell='BUY'):
                    # Use existing order_management.py functionality
                    order_data = {
                        "trading_symbol": symbol_select.value,
                        "instrument_token": all_instruments_map[symbol_select.value],
                        "quantity": int(quantity_input.value),
                        "order_type": order_type_select.value,
                        "transaction_type": buy_sell,
                        "product_type": "MIS",
                        "price": float(price_input.value) if order_type_select.value == "LIMIT" else 0,
                        "trigger_price": 0,
                        "validity": "DAY",
                        "disclosed_quantity": 0,
                        "is_amo": False,
                        "broker": broker
                    }

                    # Call existing API
                    result = await fetch_api(f"/orders", method="POST", data=order_data)
                    if result:
                        ui.notify("Order placed successfully!", type="positive")
                    else:
                        ui.notify("Order placement failed!", type="negative")

                ui.button("BUY", icon="trending_up", on_click=lambda : place_quick_order('BUY')).classes(
                    "buy-button trade-action-btn flex-1")
                ui.button("SELL", icon="trending_down", on_click=lambda : place_quick_order('SELL')).classes(
                    "sell-button trade-action-btn flex-1")


async def render_enhanced_order_book_section(fetch_api, user_storage, broker):
    """Enhanced order book section - COMPACT for right panel"""

    with ui.card().classes("dashboard-card order-book-card w-full"):
        # Header
        with ui.row().classes("card-header w-full justify-between items-center p-3"):
            ui.label("Order Book").classes("card-title text-sm")
            ui.label("NIFTY50").classes("text-xs text-cyan-400")

        ui.separator().classes("card-separator")

        # Order book content
        order_book_container = ui.column().classes("order-book-content w-full p-2")

        try:
            # Use your existing order book API integration
            order_book = await fetch_api(f"/order-book/{broker}", method="GET")

            with order_book_container:
                if order_book and len(order_book) > 0:
                    # Show top 5 orders for compact view
                    for order in order_book[:5]:
                        side_class = "buy-order" if order["TransType"] == "BUY" else "sell-order"
                        side_color = "text-green-400" if order["TransType"] == "BUY" else "text-red-400"

                        with ui.row().classes(f"order-row {side_class} w-full justify-between items-center p-1 mb-1"):
                            ui.label(f"₹{order['Price']:,.0f}").classes("order-price text-xs text-white")
                            ui.label(str(order["Quantity"])).classes("order-size text-xs text-gray-300")
                            ui.label(order["TransType"]).classes(f"order-side text-xs {side_color}")
                            ui.label(order["Status"]).classes("text-xs text-gray-400")
                else:
                    # Fallback sample data
                    sample_orders = [
                        {"price": 19852.25, "size": 150, "side": "SELL"},
                        {"price": 19851.00, "size": 200, "side": "SELL"},
                        {"price": 19850.75, "size": 100, "side": "BUY"},
                        {"price": 19850.25, "size": 250, "side": "BUY"},
                        {"price": 19849.50, "size": 180, "side": "BUY"},
                    ]

                    for order in sample_orders:
                        side_class = "buy-order" if order["side"] == "BUY" else "sell-order"
                        side_color = "text-green-400" if order["side"] == "BUY" else "text-red-400"

                        with ui.row().classes(f"order-row {side_class} w-full justify-between items-center p-1 mb-1"):
                            ui.label(f"₹{order['price']:,.0f}").classes("order-price text-xs text-white")
                            ui.label(str(order["size"])).classes("order-size text-xs text-gray-300")
                            ui.label(order["side"]).classes(f"order-side text-xs {side_color}")

        except Exception as e:
            logger.error(f"Error rendering order book: {e}")
            with order_book_container:
                ui.label("Error loading order book").classes("text-red-500 text-center p-4 text-xs")

        return order_book_container


async def render_enhanced_strategies_section(fetch_api, user_storage, broker):
    """Enhanced strategies section - COMPACT for right panel"""

    with ui.card().classes("dashboard-card strategies-card w-full"):
        # Header
        with ui.row().classes("card-header w-full justify-between items-center p-3"):
            ui.label("Active Strategies").classes("card-title text-sm")
            ui.button(icon="add").props("flat round").classes("text-cyan-400")

        ui.separator().classes("card-separator")

        # Strategies content
        strategies_container = ui.column().classes("strategies-content w-full p-2")

        try:
            # TODO: Integrate with your existing strategies.py and livetrading.py
            # active_strategies = await fetch_api(f"/strategies/{broker}/active")
            # live_trades = await fetch_api(f"/live-trades/{broker}")

            # Sample strategy data (replace with real data)
            sample_strategies = [
                {"name": "RSI Mean Reversion", "status": "active", "pnl": "+2.5%", "trades": 3},
                {"name": "Moving Average Cross", "status": "paused", "pnl": "+1.2%", "trades": 1},
                {"name": "Bollinger Bands", "status": "active", "pnl": "-0.8%", "trades": 2},
            ]

            with strategies_container:
                for strategy in sample_strategies:
                    status_class = "strategy-active" if strategy["status"] == "active" else "strategy-paused"
                    status_color = "green" if strategy["status"] == "active" else "orange"
                    pnl_class = "positive-change" if "+" in strategy["pnl"] else "negative-change"

                    with ui.column().classes(f"strategy-card {status_class} w-full p-2 mb-2"):
                        # Strategy header
                        with ui.row().classes("w-full justify-between items-center"):
                            ui.label(strategy["name"]).classes("strategy-name font-medium text-white text-xs")
                            ui.chip(strategy["status"].upper(), color=status_color).classes("strategy-status text-xs")

                        # Strategy metrics
                        with ui.row().classes("strategy-metrics w-full justify-between items-center mt-1"):
                            with ui.column().classes("gap-0"):
                                ui.label("P&L Today").classes("text-xs text-gray-400")
                                ui.label(strategy["pnl"]).classes(f"text-xs {pnl_class} font-semibold")
                            with ui.column().classes("gap-0"):
                                ui.label("Trades").classes("text-xs text-gray-400")
                                ui.label(str(strategy["trades"])).classes("text-xs text-white font-semibold")

        except Exception as e:
            logger.error(f"Error rendering strategies: {e}")
            with strategies_container:
                ui.label("Error loading strategies").classes("text-red-500 text-center p-4 text-xs")

        return strategies_container


async def render_enhanced_news_section():
    """Enhanced market news section (new feature)"""

    with ui.card().classes("dashboard-card news-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center p-4"):
            ui.icon("newspaper", size="1.5rem").classes("text-blue-400")
            ui.label("Market News").classes("card-title")

        ui.separator().classes("card-separator")

        # News content
        news_container = ui.column().classes("news-content w-full p-2")

        # Sample news data (this would come from a news API in real implementation)
        sample_news = [
            {"title": "Fed announces interest rate decision", "time": "2 hours ago", "impact": "high"},
            {"title": "NIFTY reports strong quarterly earnings", "time": "4 hours ago", "impact": "medium"},
            {"title": "Tech sector shows resilience amid volatility", "time": "6 hours ago", "impact": "low"}
        ]

        with news_container:
            for news in sample_news:
                impact_color = {"high": "news-high-impact", "medium": "news-medium-impact", "low": "news-low-impact"}[
                    news["impact"]]

                with ui.column().classes(f"news-item {impact_color} w-full p-3 mb-2"):
                    ui.label(news["title"]).classes("news-title text-sm text-white font-medium")
                    with ui.row().classes("news-meta w-full justify-between items-center mt-1"):
                        ui.label(news["time"]).classes("news-time text-xs text-gray-500")
                        ui.chip(news["impact"].upper()).classes(f"news-impact-chip text-xs")

        return news_container


async def render_enhanced_market_summary_section(fetch_api, user_storage, broker):
    """Enhanced market summary section with analytics"""

    with ui.card().classes("dashboard-card market-summary-card w-full"):
        # Header
        with ui.row().classes("card-header w-full items-center p-4"):
            ui.icon("trending_up", size="1.5rem").classes("text-green-400")
            ui.label("Market Summary").classes("card-title")

        ui.separator().classes("card-separator")

        # Market summary content
        market_summary_container = ui.column().classes("market-summary-content w-full p-2")

        try:
            # Fetch real market data (you can integrate with your existing APIs)
            # market_data = await fetch_api(f"/market-summary/{broker}")

            # For now, using sample data with real market analytics structure
            with market_summary_container:
                # Market Indices Section
                with ui.column().classes("market-indices w-full mb-3"):
                    ui.label("Major Indices").classes("text-sm font-semibold text-gray-300 mb-2")

                    indices = [
                        {"name": "NIFTY 50", "value": 19850.75, "change": 125.30, "change_pct": 0.63},
                        {"name": "BANK NIFTY", "value": 44250.25, "change": -89.50, "change_pct": -0.20},
                        {"name": "SENSEX", "value": 66589.93, "change": 245.86, "change_pct": 0.37}
                    ]

                    for index in indices:
                        change_class = "positive-change" if index["change"] > 0 else "negative-change"
                        trend_icon = "trending_up" if index["change"] > 0 else "trending_down"

                        with ui.row().classes("index-item w-full justify-between items-center py-1"):
                            with ui.column().classes("flex-1 gap-0"):
                                ui.label(index["name"]).classes("text-xs font-medium text-white")
                                ui.label(f"{index['value']:,.2f}").classes("text-xs text-gray-300")

                            with ui.column().classes("items-end gap-0"):
                                with ui.row().classes("items-center gap-1"):
                                    ui.icon(trend_icon, size="0.6rem").classes(change_class)
                                    ui.label(f"{index['change']:+.2f}").classes(f"text-xs {change_class}")
                                ui.label(f"({index['change_pct']:+.2f}%)").classes(f"text-xs {change_class}")

                ui.separator().classes("my-2 opacity-30")

                # Market Statistics Section
                with ui.column().classes("market-stats w-full mb-3"):
                    ui.label("Market Stats").classes("text-sm font-semibold text-gray-300 mb-2")

                    # Market breadth
                    with ui.row().classes("market-breadth w-full justify-between items-center py-1"):
                        ui.label("Advances").classes("text-xs text-gray-400")
                        ui.label("1,247").classes("text-xs positive-change font-semibold")

                    with ui.row().classes("market-breadth w-full justify-between items-center py-1"):
                        ui.label("Declines").classes("text-xs text-gray-400")
                        ui.label("823").classes("text-xs negative-change font-semibold")

                    with ui.row().classes("market-breadth w-full justify-between items-center py-1"):
                        ui.label("Unchanged").classes("text-xs text-gray-400")
                        ui.label("145").classes("text-xs text-gray-300 font-semibold")

                ui.separator().classes("my-2 opacity-30")

                # Market Sentiment Section
                with ui.column().classes("market-sentiment w-full mb-3"):
                    ui.label("Market Sentiment").classes("text-sm font-semibold text-gray-300 mb-2")

                    # VIX
                    with ui.row().classes("sentiment-item w-full justify-between items-center py-1"):
                        ui.label("VIX").classes("text-xs text-gray-400")
                        with ui.row().classes("items-center gap-2"):
                            ui.label("16.85").classes("text-xs text-white font-semibold")
                            ui.chip("Low Vol", color="green").classes("text-xs h-4")

                    # FII/DII Activity
                    with ui.row().classes("sentiment-item w-full justify-between items-center py-1"):
                        ui.label("FII Flow").classes("text-xs text-gray-400")
                        ui.label("₹+245 Cr").classes("text-xs positive-change font-semibold")

                    with ui.row().classes("sentiment-item w-full justify-between items-center py-1"):
                        ui.label("DII Flow").classes("text-xs text-gray-400")
                        ui.label("₹+186 Cr").classes("text-xs positive-change font-semibold")

            ui.separator().classes("my-2 opacity-30")

            # Market Trend Analysis
            with ui.column().classes("market-trend w-full"):
                ui.label("Trend Analysis").classes("text-sm font-semibold text-gray-300 mb-2")

                # Overall market trend
                with ui.row().classes("trend-summary w-full items-center justify-center py-2"):
                    ui.icon("trending_up", size="1rem").classes("text-green-400")
                    ui.label("BULLISH").classes("text-sm font-bold text-green-400 ml-2")

                # Key levels
                with ui.column().classes("key-levels w-full gap-1"):
                    with ui.row().classes("level-item w-full justify-between"):
                        ui.label("Support").classes("text-xs text-gray-400")
                        ui.label("19,750").classes("text-xs text-blue-400")

                    with ui.row().classes("level-item w-full justify-between"):
                        ui.label("Resistance").classes("text-xs text-gray-400")
                        ui.label("19,950").classes("text-xs text-orange-400")

        except Exception as e:
            logger.error(f"Error rendering market summary: {e}")
            with market_summary_container:
                ui.label("Error loading market data").classes("text-red-500 text-center p-4 text-xs")

        return market_summary_container

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
    ui.timer(300.0, update_dashboard_data)


# Utility functions for dashboard functionality

def format_currency(value):
    """Format currency values consistently"""
    if isinstance(value, (int, float)):
        return f"₹{value:,.2f}"
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