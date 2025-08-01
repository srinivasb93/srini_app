# Enhanced Live Trading Module - livetrading.py
# Applying beautiful dashboard.py styling consistently

from nicegui import ui
import logging
import asyncio
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def apply_enhanced_dashboard_styles():
    """Apply enhanced CSS styles matching dashboard.py"""
    ui.add_css('static/styles.css')


async def render_live_trading_page(fetch_api, user_storage, get_cached_instruments):
    """Enhanced live trading page with beautiful dashboard styling"""

    apply_enhanced_dashboard_styles()

    # Get broker from user storage
    broker = user_storage.get('default_broker', 'Zerodha')

    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):
        # Enhanced title section (matching dashboard.py)
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("speed", size="2rem").classes("text-cyan-400")
                    ui.label(f"Live Trading - {broker}").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("ACTIVE", color="green").classes("text-xs status-chip animate-pulse")

                ui.label("Real-time algorithmic trading execution and monitoring").classes(
                    "text-gray-400 dashboard-subtitle")

            # Right side - Control buttons
            with ui.row().classes("items-center gap-4"):
                ui.button("Emergency Stop", icon="emergency", color="red").classes("text-white animate-pulse")
                ui.button("Pause All", icon="pause", color="yellow").classes("text-black")
                ui.button("Start All", icon="play_arrow", color="green").classes("text-white")

        # Trading status and metrics
        await render_enhanced_trading_status(fetch_api, user_storage, broker)

        # Main content in grid layout
        with ui.row().classes("w-full gap-4 p-4"):
            # Active strategies panel (left)
            with ui.card().classes("dashboard-card w-1/2"):
                with ui.row().classes("card-header w-full justify-between items-center p-4"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("auto_awesome", size="1.5rem").classes("text-purple-400")
                        ui.label("Live Strategies").classes("card-title")

                        # Live indicator
                        with ui.row().classes("items-center gap-1 ml-2"):
                            ui.element('div').classes("w-2 h-2 bg-purple-400 rounded-full animate-pulse")
                            ui.label("Active").classes("text-xs text-purple-400")

                    ui.button("Manage", icon="settings", on_click=lambda: ui.navigate.to('/strategies')).props(
                        "flat").classes("text-cyan-400")

                ui.separator().classes("card-separator")

                strategies_container = ui.column().classes("w-full p-4")
                await render_enhanced_live_strategies(fetch_api, user_storage, broker, strategies_container)

            # Live trades panel (right)
            with ui.card().classes("dashboard-card flex-1"):
                with ui.row().classes("card-header w-full justify-between items-center p-4"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("trending_up", size="1.5rem").classes("text-green-400")
                        ui.label("Live Trades").classes("card-title")

                        # Live trades counter
                        trades_counter = ui.chip("0", color="green").classes("text-xs")

                    with ui.row().classes("items-center gap-2"):
                        ui.button("Export", icon="download").props("flat").classes("text-cyan-400")
                        ui.button("Refresh", icon="refresh").props("flat").classes("text-gray-400")

                ui.separator().classes("card-separator")

                trades_container = ui.column().classes("w-full p-4")
                await render_enhanced_live_trades(fetch_api, user_storage, broker, trades_container)

        # Risk monitoring section
        await render_enhanced_risk_monitoring(fetch_api, user_storage, broker)


async def render_enhanced_trading_status(fetch_api, user_storage, broker):
    """Enhanced trading status metrics"""

    with ui.row().classes("w-full gap-4 p-4"):
        try:
            # Fetch live trading data - this would be from your API
            # trading_data = await fetch_api(f"/live-trading/{broker}/status")

            # Sample data for demonstration
            total_pnl = 15420.75
            trades_today = 23
            active_strategies = 4
            success_rate = 68.5
            current_exposure = 125000.00

            # Total P&L Today
            pnl_color = "text-green-400" if total_pnl >= 0 else "text-red-400"
            pnl_icon = "trending_up" if total_pnl >= 0 else "trending_down"
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon(pnl_icon, size="2rem").classes(f"{pnl_color} mb-2")
                    ui.label("Today's P&L").classes("text-sm text-gray-400")
                    ui.label(f"₹{total_pnl:,.2f}").classes(f"text-2xl font-bold {pnl_color}")

            # Trades Today
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("swap_horiz", size="2rem").classes("text-blue-400 mb-2")
                    ui.label("Trades Today").classes("text-sm text-gray-400")
                    ui.label(str(trades_today)).classes("text-2xl font-bold text-blue-400")

            # Active Strategies
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("psychology", size="2rem").classes("text-purple-400 mb-2")
                    ui.label("Active Strategies").classes("text-sm text-gray-400")
                    ui.label(str(active_strategies)).classes("text-2xl font-bold text-purple-400")

            # Success Rate
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("verified", size="2rem").classes("text-orange-400 mb-2")
                    ui.label("Success Rate").classes("text-sm text-gray-400")
                    ui.label(f"{success_rate:.1f}%").classes("text-2xl font-bold text-orange-400")

            # Current Exposure
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("account_balance", size="2rem").classes("text-cyan-400 mb-2")
                    ui.label("Current Exposure").classes("text-sm text-gray-400")
                    ui.label(f"₹{current_exposure:,.0f}").classes("text-2xl font-bold text-cyan-400")

        except Exception as e:
            logger.error(f"Error fetching trading status: {e}")
            with ui.card().classes("dashboard-card w-full"):
                ui.label("Error loading trading status").classes("text-red-500 text-center p-4")


async def render_enhanced_live_strategies(fetch_api, user_storage, broker, container):
    """Render enhanced live strategies list"""

    try:
        # Fetch live strategies - this would be from your API
        # strategies_data = await fetch_api(f"/strategies/{broker}/live")

        # Sample strategies data
        live_strategies = [
            {
                "id": "live_001",
                "name": "RSI Scalper",
                "status": "RUNNING",
                "symbol": "NIFTY50",
                "pnl_today": 2450.75,
                "trades_today": 8,
                "last_signal": "BUY @ 19850.50",
                "last_trade_time": "14:32:15"
            },
            {
                "id": "live_002",
                "name": "Momentum Breakout",
                "status": "RUNNING",
                "symbol": "BANKNIFTY",
                "pnl_today": -450.25,
                "trades_today": 3,
                "last_signal": "SELL @ 45150.75",
                "last_trade_time": "14:28:45"
            },
            {
                "id": "live_003",
                "name": "Mean Reversion",
                "status": "WAITING",
                "symbol": "RELIANCE",
                "pnl_today": 875.50,
                "trades_today": 2,
                "last_signal": "No Signal",
                "last_trade_time": "13:45:20"
            }
        ]

        if not live_strategies:
            # Enhanced empty state
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("auto_awesome_motion", size="3rem").classes("text-gray-500 mb-4")
                    ui.label("No live strategies").classes("text-lg text-gray-400 mb-2")
                    ui.label("Start some strategies to begin live trading").classes("text-sm text-gray-500")
                    ui.button("Go to Strategies", icon="psychology",
                              on_click=lambda: ui.navigate.to('/strategies')).classes("mt-4")
            return

        with container:
            for strategy in live_strategies:
                await render_enhanced_strategy_card(strategy, fetch_api, broker)

    except Exception as e:
        logger.error(f"Error rendering live strategies: {e}")
        with container:
            with ui.column().classes("w-full text-center p-4"):
                ui.icon("error", size="2rem").classes("text-red-500 mb-2")
                ui.label("Error loading strategies").classes("text-red-400")


async def render_enhanced_strategy_card(strategy, fetch_api, broker):
    """Render individual enhanced strategy card"""

    try:
        strategy_id = strategy.get('id', 'N/A')
        name = strategy.get('name', 'N/A')
        status = strategy.get('status', 'UNKNOWN').upper()
        symbol = strategy.get('symbol', 'N/A')
        pnl_today = float(strategy.get('pnl_today', 0))
        trades_today = strategy.get('trades_today', 0)
        last_signal = strategy.get('last_signal', 'No Signal')
        last_trade_time = strategy.get('last_trade_time', 'N/A')

        # Determine status styling
        if status == 'RUNNING':
            status_color = "text-green-400"
            status_bg = "bg-green-900/20"
            border_color = "border-green-500/30"
            status_icon = "play_circle"
        elif status == 'WAITING':
            status_color = "text-yellow-400"
            status_bg = "bg-yellow-900/20"
            border_color = "border-yellow-500/30"
            status_icon = "schedule"
        else:
            status_color = "text-red-400"
            status_bg = "bg-red-900/20"
            border_color = "border-red-500/30"
            status_icon = "stop_circle"

        # Determine P&L styling
        pnl_color = "text-green-400" if pnl_today >= 0 else "text-red-400"
        pnl_icon = "trending_up" if pnl_today >= 0 else "trending_down"

        with ui.card().classes(
                f"live-strategy-card w-full mb-3 border {border_color} hover:bg-gray-800/20 transition-all"):
            with ui.column().classes("p-3"):
                # Header row
                with ui.row().classes("w-full justify-between items-center mb-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(status_icon, size="1.2rem").classes(status_color)
                        ui.label(name).classes("text-white font-semibold")
                        ui.chip(status, color=None).classes(f"{status_color} {status_bg} text-xs")

                    with ui.row().classes("items-center gap-1"):
                        ui.icon(pnl_icon, size="1rem").classes(pnl_color)
                        ui.label(f"₹{pnl_today:,.2f}").classes(f"{pnl_color} font-semibold text-mono text-sm")

                # Details row
                with ui.row().classes("w-full justify-between items-center text-sm"):
                    with ui.column().classes("gap-1"):
                        ui.label(f"Symbol: {symbol}").classes("text-gray-300")
                        ui.label(f"Trades: {trades_today}").classes("text-gray-300")

                    with ui.column().classes("gap-1 items-end"):
                        ui.label(last_signal).classes("text-cyan-400 text-xs")
                        ui.label(last_trade_time).classes("text-gray-500 text-xs")

    except Exception as e:
        logger.error(f"Error rendering strategy card: {e}")
        with ui.card().classes("live-strategy-card w-full mb-3 border border-red-500/30"):
            ui.label("Error loading strategy").classes("text-red-400 p-3")


async def render_enhanced_live_trades(fetch_api, user_storage, broker, container):
    """Render enhanced live trades list"""

    try:
        # Fetch live trades - this would be from your API
        # trades_data = await fetch_api(f"/live-trading/{broker}/trades")

        # Sample live trades data
        live_trades = [
            {
                "id": "trade_001",
                "strategy": "RSI Scalper",
                "symbol": "NIFTY50",
                "side": "BUY",
                "quantity": 50,
                "entry_price": 19850.50,
                "current_price": 19875.25,
                "pnl": 1237.50,
                "time": "14:32:15",
                "status": "OPEN"
            },
            {
                "id": "trade_002",
                "strategy": "Momentum Breakout",
                "symbol": "BANKNIFTY",
                "side": "SELL",
                "quantity": 25,
                "entry_price": 45150.75,
                "current_price": 45125.50,
                "pnl": 631.25,
                "time": "14:28:45",
                "status": "OPEN"
            },
            {
                "id": "trade_003",
                "strategy": "Mean Reversion",
                "symbol": "RELIANCE",
                "side": "BUY",
                "quantity": 100,
                "entry_price": 2450.25,
                "current_price": 2458.75,
                "pnl": 850.00,
                "time": "13:45:20",
                "status": "CLOSED"
            }
        ]

        if not live_trades:
            # Enhanced empty state
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("trending_flat", size="3rem").classes("text-gray-500 mb-4")
                    ui.label("No live trades").classes("text-lg text-gray-400 mb-2")
                    ui.label("Live trades will appear here when strategies execute").classes("text-sm text-gray-500")
            return

        with container:
            # Table header
            with ui.row().classes(
                    "trades-header w-full p-2 text-xs font-semibold text-gray-400 border-b border-gray-700 mb-2"):
                ui.label("Strategy").classes("w-32")
                ui.label("Symbol").classes("w-24")
                ui.label("Side").classes("w-16")
                ui.label("P&L").classes("w-24 text-right")
                ui.label("Time").classes("w-20")
                ui.label("Status").classes("w-20")

            # Render trade rows
            for trade in live_trades:
                await render_enhanced_trade_row(trade, fetch_api, broker)

    except Exception as e:
        logger.error(f"Error rendering live trades: {e}")
        with container:
            with ui.column().classes("w-full text-center p-4"):
                ui.icon("error", size="2rem").classes("text-red-500 mb-2")
                ui.label("Error loading trades").classes("text-red-400")


async def render_enhanced_trade_row(trade, fetch_api, broker):
    """Render individual enhanced trade row"""

    try:
        strategy = trade.get('strategy', 'N/A')
        symbol = trade.get('symbol', 'N/A')
        side = trade.get('side', 'N/A').upper()
        quantity = trade.get('quantity', 0)
        entry_price = float(trade.get('entry_price', 0))
        current_price = float(trade.get('current_price', 0))
        pnl = float(trade.get('pnl', 0))
        time = trade.get('time', 'N/A')
        status = trade.get('status', 'UNKNOWN').upper()

        # Determine side styling
        side_color = "text-green-400" if side == "BUY" else "text-red-400"
        side_icon = "trending_up" if side == "BUY" else "trending_down"

        # Determine P&L styling
        pnl_color = "text-green-400" if pnl >= 0 else "text-red-400"

        # Determine status styling
        status_color = "text-green-400" if status == "OPEN" else "text-gray-400"

        with ui.row().classes(
                "trade-row w-full p-2 hover:bg-gray-800/30 transition-all border-l-2 border-cyan-500/20 mb-1 rounded-r-lg"):
            # Strategy
            ui.label(strategy).classes("w-32 text-white text-xs")

            # Symbol
            ui.label(symbol).classes("w-24 text-cyan-400 text-xs font-semibold")

            # Side
            with ui.row().classes("w-16 items-center gap-1"):
                ui.icon(side_icon, size="0.8rem").classes(side_color)
                ui.label(side).classes(f"{side_color} text-xs font-semibold")

            # P&L
            ui.label(f"₹{pnl:,.0f}").classes(f"w-24 text-right {pnl_color} text-xs font-mono font-semibold")

            # Time
            ui.label(time).classes("w-20 text-gray-400 text-xs")

            # Status
            ui.label(status).classes(f"w-20 {status_color} text-xs")

    except Exception as e:
        logger.error(f"Error rendering trade row: {e}")
        with ui.row().classes("trade-row w-full p-2 border-l-2 border-red-500/20"):
            ui.label("Error loading trade").classes("text-red-400 text-xs")


async def render_enhanced_risk_monitoring(fetch_api, user_storage, broker):
    """Enhanced risk monitoring section"""

    with ui.card().classes("dashboard-card w-full m-4"):
        with ui.row().classes("card-header w-full justify-between items-center p-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("security", size="1.5rem").classes("text-red-400")
                ui.label("Risk Monitoring").classes("card-title")

                # Risk status indicator
                with ui.row().classes("items-center gap-1 ml-2"):
                    ui.element('div').classes("w-2 h-2 bg-green-400 rounded-full animate-pulse")
                    ui.label("Normal").classes("text-xs text-green-400")

            with ui.row().classes("items-center gap-2"):
                ui.button("Risk Settings", icon="tune").props("flat").classes("text-cyan-400")
                ui.button("Alert History", icon="history").props("flat").classes("text-gray-400")

        ui.separator().classes("card-separator")

        # Risk metrics
        with ui.row().classes("w-full gap-4 p-4"):
            try:
                # Sample risk data
                max_drawdown = 5.2
                var_95 = 8750.25
                exposure_limit = 85.5
                daily_loss_limit = 15000.00
                current_loss = 2450.75

                # Max Drawdown
                drawdown_color = "text-yellow-400" if max_drawdown > 3 else "text-green-400"
                with ui.card().classes("dashboard-card risk-metric-card flex-1"):
                    with ui.column().classes("p-3 text-center"):
                        ui.icon("trending_down", size="1.5rem").classes(f"{drawdown_color} mb-2")
                        ui.label("Max Drawdown").classes("text-xs text-gray-400")
                        ui.label(f"{max_drawdown:.1f}%").classes(f"text-lg font-bold {drawdown_color}")

                # VaR (95%)
                with ui.card().classes("dashboard-card risk-metric-card flex-1"):
                    with ui.column().classes("p-3 text-center"):
                        ui.icon("assessment", size="1.5rem").classes("text-purple-400 mb-2")
                        ui.label("VaR (95%)").classes("text-xs text-gray-400")
                        ui.label(f"₹{var_95:,.0f}").classes("text-lg font-bold text-purple-400")

                # Exposure Limit
                exposure_color = "text-red-400" if exposure_limit > 90 else "text-yellow-400" if exposure_limit > 70 else "text-green-400"
                with ui.card().classes("dashboard-card risk-metric-card flex-1"):
                    with ui.column().classes("p-3 text-center"):
                        ui.icon("pie_chart", size="1.5rem").classes(f"{exposure_color} mb-2")
                        ui.label("Exposure Used").classes("text-xs text-gray-400")
                        ui.label(f"{exposure_limit:.1f}%").classes(f"text-lg font-bold {exposure_color}")

                # Daily Loss Limit
                loss_percentage = (current_loss / daily_loss_limit) * 100
                loss_color = "text-red-400" if loss_percentage > 80 else "text-yellow-400" if loss_percentage > 60 else "text-green-400"
                with ui.card().classes("dashboard-card risk-metric-card flex-1"):
                    with ui.column().classes("p-3 text-center"):
                        ui.icon("warning", size="1.5rem").classes(f"{loss_color} mb-2")
                        ui.label("Loss Limit").classes("text-xs text-gray-400")
                        ui.label(f"₹{current_loss:,.0f}").classes(f"text-lg font-bold {loss_color}")
                        ui.label(f"of ₹{daily_loss_limit:,.0f}").classes("text-xs text-gray-500")

                # Risk Alerts
                with ui.card().classes("dashboard-card risk-metric-card flex-1"):
                    with ui.column().classes("p-3 text-center"):
                        ui.icon("notifications", size="1.5rem").classes("text-orange-400 mb-2")
                        ui.label("Active Alerts").classes("text-xs text-gray-400")
                        ui.label("2").classes("text-lg font-bold text-orange-400")
                        ui.label("Low Priority").classes("text-xs text-gray-500")

            except Exception as e:
                logger.error(f"Error rendering risk metrics: {e}")
                ui.label("Error loading risk data").classes("text-red-500 text-center p-4")


# Control functions
def emergency_stop_all():
    """Emergency stop all trading"""
    ui.notify("EMERGENCY STOP ACTIVATED - All trading halted!", type="negative")
    # This would call your API to immediately stop all trading


def pause_all_strategies():
    """Pause all active strategies"""
    ui.notify("All strategies paused", type="warning")
    # This would call your API to pause all strategies


def start_all_strategies():
    """Start all strategies"""
    ui.notify("Starting all strategies...", type="positive")
    # This would call your API to start strategies


def show_strategy_details(strategy_id):
    """Show detailed strategy information"""
    ui.notify(f"Opening details for strategy {strategy_id}", type="info")
    # This would open a detailed view or navigate to strategy page


def close_trade(trade_id):
    """Close a specific trade"""
    ui.notify(f"Closing trade {trade_id}", type="info")
    # This would call your API to close the trade