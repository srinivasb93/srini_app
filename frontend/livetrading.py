# Enhanced Live Trading Module - livetrading.py
# Applying beautiful dashboard.py styling consistently

from nicegui import ui
import logging
import asyncio
from datetime import datetime
import json
from cache_manager import frontend_cache, cached_fetch_api, FrontendCacheConfig, TradingDataCache
from cache_invalidation import invalidate_on_strategy_action

logger = logging.getLogger(__name__)

async def safe_api_call(fetch_api, endpoint, method="GET", timeout=10.0, fallback=None, **kwargs):
    """
    Safely call an API endpoint with proper error handling and timeout
    """
    try:
        if method.upper() == "GET":
            task = fetch_api(endpoint)
        else:
            task = fetch_api(endpoint, method=method, **kwargs)
            
        response = await asyncio.wait_for(task, timeout=timeout)
        return response if response is not None else fallback
        
    except asyncio.TimeoutError:
        logger.warning(f"API timeout for {endpoint}")
        return fallback
    except Exception as e:
        logger.error(f"API error for {endpoint}: {e}")
        return fallback

async def cached_safe_api_call(fetch_api, endpoint, method="GET", timeout=10.0, fallback=None, ttl=None, **kwargs):
    """
    Cached version of safe_api_call for frequently accessed data
    """
    if method.upper() != "GET" or ttl is None:
        return await safe_api_call(fetch_api, endpoint, method, timeout, fallback, **kwargs)
    
    cache_key = frontend_cache.generate_cache_key("safe_api", endpoint, method, str(kwargs))
    cached_result = frontend_cache.get(cache_key)
    
    if cached_result is not None:
        logger.debug(f"Cache hit for {endpoint}")
        return cached_result
    
    result = await safe_api_call(fetch_api, endpoint, method, timeout, fallback, **kwargs)
    
    if result is not None and result != fallback:
        frontend_cache.set(cache_key, result, ttl)
    
    return result


async def render_live_trading_page(fetch_api, user_storage, get_cached_instruments):
    """Enhanced live trading page with beautiful dashboard styling"""
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

            # Right side - Control buttons with proper functionality
            with ui.row().classes("items-center gap-4"):
                ui.button("Emergency Stop", icon="emergency", color="red",
                         on_click=lambda: emergency_stop_all(fetch_api)).classes("text-white animate-pulse")
                ui.button("Pause All", icon="pause", color="yellow",
                         on_click=lambda: pause_all_strategies(fetch_api)).classes("text-black")
                ui.button("Start All", icon="play_arrow", color="green",
                         on_click=lambda: start_all_strategies(fetch_api)).classes("text-white")

        # Trading status and metrics
        await render_enhanced_trading_status(fetch_api, user_storage, broker)

        # Main content in grid layout - FIXED to use settings-layout for side-by-side panels
        with ui.row().classes("settings-layout w-full gap-4 p-4"):
            # Active strategies panel (left)
            with ui.card().classes("dashboard-card flex-1"):
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
            # Fetch real trading data
            total_pnl = 0
            trades_today = 0
            active_strategies = 0
            success_rate = 0
            current_exposure = 0
            
            try:
                # Get real execution data with caching - short TTL for real-time accuracy
                executions_response = await cached_safe_api_call(
                    fetch_api, "/executions", timeout=10.0, 
                    ttl=FrontendCacheConfig.LIVE_QUOTES
                )
                strategies_response = await cached_safe_api_call(
                    fetch_api, f"/strategies/{broker}/execution-status", timeout=10.0,
                    ttl=FrontendCacheConfig.STRATEGY_LIST
                )
                
                if executions_response and isinstance(executions_response, list):
                    # Calculate metrics from real data with safe conversions
                    running_executions = [e for e in executions_response if e.get('status') == 'running']
                    active_strategies = len(running_executions)
                    
                    # Calculate total P&L from all executions with error handling
                    total_pnl = 0
                    for execution in executions_response:
                        try:
                            pnl_value = execution.get('pnl', 0)
                            total_pnl += float(pnl_value) if pnl_value else 0
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid P&L value in execution {execution.get('execution_id')}: {pnl_value}")
                    
                    # Calculate total trades executed with error handling
                    trades_today = 0
                    for execution in executions_response:
                        try:
                            trades_value = execution.get('trades_executed', 0)
                            trades_today += int(trades_value) if trades_value else 0
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid trades value in execution {execution.get('execution_id')}: {trades_value}")
                    
                    # Calculate success rate (completed vs failed)
                    completed = len([e for e in executions_response if e.get('status') == 'completed'])
                    failed = len([e for e in executions_response if e.get('status') == 'failed'])
                    total_finished = completed + failed
                    success_rate = (completed / total_finished * 100) if total_finished > 0 else 0
                    
                    # Current exposure - sum of running execution values (approximate)
                    current_exposure = active_strategies * 50000  # Rough estimate
                else:
                    # Handle when no executions are returned (empty response or None)
                    logger.debug(f"No executions found or empty response: {executions_response}")

            except Exception as e:
                logger.error(f"Error fetching real trading data: {e}")
                # Keep default values

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
    """Render enhanced live strategies list with real execution data"""

    try:
        # Fetch real strategy executions data
        live_strategies = []
        try:
            # Get active strategy executions with caching
            executions_response = await cached_safe_api_call(
                fetch_api, "/executions", timeout=8.0, 
                ttl=FrontendCacheConfig.LIVE_QUOTES
            )
            strategies_response = await cached_safe_api_call(
                fetch_api, f"/strategies/{broker}/execution-status", timeout=8.0,
                ttl=FrontendCacheConfig.STRATEGY_LIST
            )
            
            if executions_response and strategies_response:
                # Create a map of strategy names
                strategy_map = {s['strategy_id']: s['name'] for s in strategies_response}
                
                # Process running executions
                running_executions = [e for e in executions_response if e.get('status') == 'running']
                
                for execution in running_executions:
                    strategy_name = strategy_map.get(execution.get('strategy_id'), 'Unknown Strategy')
                    trading_symbol = execution.get('trading_symbol', 'N/A')
                    pnl = execution.get('pnl', 0)
                    signals_generated = execution.get('signals_generated', 0)
                    trades_executed = execution.get('trades_executed', 0)
                    started_at = execution.get('started_at', '')
                    last_signal_at = execution.get('last_signal_at', '')
                    
                    live_strategy = {
                        "id": execution.get('execution_id'),
                        "strategy_id": execution.get('strategy_id'),
                        "name": strategy_name,
                        "status": "RUNNING",
                        "symbol": trading_symbol,
                        "pnl_today": float(pnl),
                        "trades_today": trades_executed,
                        "signals_today": signals_generated,
                        "last_signal": f"{signals_generated} signals generated" if signals_generated > 0 else "No signals",
                        "last_trade_time": last_signal_at or started_at,
                        "execution": execution  # Store full execution data
                    }
                    live_strategies.append(live_strategy)
        except Exception as e:
            logger.error(f"Error fetching real strategy data: {e}")
            # Fallback to empty list
            live_strategies = []

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
                        if strategy.get('signals_today'):
                            ui.label(f"Signals: {strategy.get('signals_today')}").classes("text-gray-300")

                    with ui.column().classes("gap-1 items-end"):
                        ui.label(last_signal).classes("text-cyan-400 text-xs")
                        ui.label(last_trade_time).classes("text-gray-500 text-xs")

                # Action buttons row
                with ui.row().classes("w-full justify-between items-center mt-3 pt-2 border-t border-gray-700"):
                    with ui.row().classes("gap-1"):
                        ui.button("Stop", icon="stop", on_click=lambda sid=strategy_id: asyncio.create_task(stop_execution_from_live(sid, fetch_api))).props("size=sm color=negative flat").classes("text-xs")
                        ui.button("Metrics", icon="analytics", on_click=lambda sid=strategy_id: asyncio.create_task(view_metrics_from_live(sid, fetch_api))).props("size=sm flat").classes("text-xs text-cyan-400")
                    
                    with ui.row().classes("gap-1"):
                        if strategy.get('strategy_id'):
                            ui.button("Manage", icon="settings", on_click=lambda: ui.navigate.to('/strategies')).props("size=sm flat").classes("text-xs text-blue-400")
                        ui.button("Details", icon="visibility", on_click=lambda: navigate_to_strategy_details(strategy)).props("size=sm flat").classes("text-xs text-gray-400")

    except Exception as e:
        logger.error(f"Error rendering strategy card: {e}")
        with ui.card().classes("live-strategy-card w-full mb-3 border border-red-500/30"):
            ui.label("Error loading strategy").classes("text-red-400 p-3")


async def render_enhanced_live_trades(fetch_api, user_storage, broker, container):
    """Render enhanced live trades list"""

    try:
        # Fetch real live trades from API
        live_trades = []
        try:
            # Get current positions and recent orders for live trades with caching
            positions_response = await cached_safe_api_call(
                fetch_api, f"/positions/{broker}", timeout=8.0,
                ttl=FrontendCacheConfig.POSITION_DATA
            )
            orders_response = await cached_safe_api_call(
                fetch_api, f"/orders/{broker}", timeout=8.0,
                ttl=FrontendCacheConfig.ORDER_STATUS
            )
            
            if positions_response and isinstance(positions_response, list):
                # Convert positions to live trades format
                for position in positions_response:
                    if position.get('quantity', 0) != 0:  # Only show open positions
                        live_trade = {
                            "id": f"pos_{position.get('instrument_token', 'unknown')}",
                            "strategy": position.get('strategy_name', 'Manual Trade'),
                            "symbol": position.get('trading_symbol', 'Unknown'),
                            "side": "BUY" if position.get('quantity', 0) > 0 else "SELL",
                            "quantity": abs(position.get('quantity', 0)),
                            "entry_price": position.get('average_price', 0),
                            "current_price": position.get('last_price', position.get('average_price', 0)),
                            "pnl": position.get('pnl', 0),
                            "time": position.get('last_updated', ''),
                            "status": "OPEN"
                        }
                        live_trades.append(live_trade)
            
            # Add recent completed orders from today
            if orders_response:
                from datetime import date
                today_str = date.today().strftime('%Y-%m-%d')
                completed_orders = [
                    order for order in orders_response 
                    if order.get('status') in ['COMPLETE', 'FILLED'] and 
                    order.get('order_timestamp', '').startswith(today_str)
                ]
                
                for order in completed_orders[-10:]:  # Last 10 completed orders
                    live_trade = {
                        "id": f"order_{order.get('order_id', 'unknown')}",
                        "strategy": order.get('strategy_name', 'Manual Trade'),
                        "symbol": order.get('trading_symbol', 'Unknown'),
                        "side": order.get('transaction_type', 'BUY'),
                        "quantity": order.get('filled_quantity', order.get('quantity', 0)),
                        "entry_price": order.get('average_price', order.get('price', 0)),
                        "current_price": order.get('average_price', order.get('price', 0)),
                        "pnl": 0,  # Completed orders don't show running P&L
                        "time": order.get('order_timestamp', '').split('T')[1][:8] if 'T' in order.get('order_timestamp', '') else '',
                        "status": "CLOSED"
                    }
                    live_trades.append(live_trade)
                    
        except Exception as e:
            logger.error(f"Error fetching live trades from API: {e}")
            # Fall back to empty list if API fails
            live_trades = []

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
                # Fetch real risk metrics from API with proper error handling
                risk_metrics = {}
                try:
                    risk_task = fetch_api("/risk-management/metrics")
                    risk_response = await asyncio.wait_for(risk_task, timeout=5.0)
                    
                    if risk_response and isinstance(risk_response, dict):
                        risk_metrics = risk_response
                    elif risk_response is not None:
                        logger.warning(f"Unexpected risk metrics response format: {type(risk_response)}")
                        
                except asyncio.TimeoutError:
                    logger.warning("Timeout fetching risk metrics - using defaults")
                except Exception as e:
                    logger.error(f"Error fetching risk metrics: {e}")
                
                # Extract risk data with safe conversions and fallbacks
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value is not None else default
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid numeric value: {value}, using default {default}")
                        return default
                
                def safe_int(value, default=0):
                    try:
                        return int(value) if value is not None else default
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid integer value: {value}, using default {default}")
                        return default
                
                max_drawdown = safe_float(risk_metrics.get('max_drawdown'), 0.0)
                var_95 = safe_float(risk_metrics.get('var_95'), 0.0)
                exposure_used_pct = safe_float(risk_metrics.get('exposure_used_pct'), 0.0)
                daily_loss_limit = safe_float(risk_metrics.get('daily_loss_limit'), 15000.0)
                daily_pnl = safe_float(risk_metrics.get('daily_pnl'), 0.0)
                current_loss = abs(daily_pnl) if daily_pnl < 0 else 0.0
                alert_count = safe_int(risk_metrics.get('active_alerts'), 0)

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
                exposure_color = "text-red-400" if exposure_used_pct > 90 else "text-yellow-400" if exposure_used_pct > 70 else "text-green-400"
                with ui.card().classes("dashboard-card risk-metric-card flex-1"):
                    with ui.column().classes("p-3 text-center"):
                        ui.icon("pie_chart", size="1.5rem").classes(f"{exposure_color} mb-2")
                        ui.label("Exposure Used").classes("text-xs text-gray-400")
                        ui.label(f"{exposure_used_pct:.1f}%").classes(f"text-lg font-bold {exposure_color}")

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
                alert_color = "text-red-400" if alert_count > 5 else "text-orange-400" if alert_count > 0 else "text-green-400"
                alert_priority = "High Priority" if alert_count > 5 else "Medium Priority" if alert_count > 2 else "Low Priority" if alert_count > 0 else "No Alerts"
                with ui.card().classes("dashboard-card risk-metric-card flex-1"):
                    with ui.column().classes("p-3 text-center"):
                        ui.icon("notifications", size="1.5rem").classes(f"{alert_color} mb-2")
                        ui.label("Active Alerts").classes("text-xs text-gray-400")
                        ui.label(str(alert_count)).classes(f"text-lg font-bold {alert_color}")
                        ui.label(alert_priority).classes("text-xs text-gray-500")

            except Exception as e:
                logger.error(f"Error rendering risk metrics: {e}")
                ui.label("Error loading risk data").classes("text-red-500 text-center p-4")


# Control functions
async def emergency_stop_all(fetch_api):
    """Emergency stop all trading with robust error handling"""
    try:
        response = await safe_api_call(
            fetch_api, 
            "/executions/emergency-stop", 
            method="POST", 
            timeout=15.0,  # Longer timeout for critical operation
            fallback=None
        )
        
        if response and not response.get("error"):
            # Use centralized cache invalidation for emergency stop
            invalidate_on_strategy_action("stop", "all")
            ui.notify("🚨 EMERGENCY STOP ACTIVATED - All trading halted!", type="negative")
            return True
        elif response is None:
            ui.notify("⚠️ Emergency stop request failed - No response from server", type="negative")
            return False
        else:
            error_msg = "Unknown error"
            if isinstance(response, dict):
                error_info = response.get('error', {})
                if isinstance(error_info, dict):
                    error_msg = error_info.get('message', 'Unknown error')
                elif isinstance(error_info, str):
                    error_msg = error_info
            ui.notify(f"Failed to activate emergency stop: {error_msg}", type="negative")
            return False
    except Exception as e:
        logger.error(f"Emergency stop critical error: {e}")
        ui.notify(f"Emergency stop critical error: {str(e)}", type="negative")
        return False


async def pause_all_strategies(fetch_api):
    """Pause all active strategies"""
    try:
        response = await fetch_api("/strategies/pause-all", method="POST")
        if response and not response.get("error"):
            # Use centralized cache invalidation for pause
            invalidate_on_strategy_action("stop", "all")
            ui.notify("All strategies paused", type="warning")
            return True
        else:
            ui.notify(f"Failed to pause strategies: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
            return False
    except Exception as e:
        ui.notify(f"Pause error: {str(e)}", type="negative")
        return False


async def start_all_strategies(fetch_api):
    """Start all strategies"""
    try:
        response = await fetch_api("/strategies/start-all", method="POST")
        if response and not response.get("error"):
            # Use centralized cache invalidation for start
            invalidate_on_strategy_action("start", "all")
            ui.notify("Starting all strategies...", type="positive")
            return True
        else:
            ui.notify(f"Failed to start strategies: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
            return False
    except Exception as e:
        ui.notify(f"Start error: {str(e)}", type="negative")
        return False


async def stop_execution_from_live(execution_id, fetch_api=None):
    """Stop a strategy execution from live trading dashboard"""
    try:
        if not fetch_api:
            ui.notify("API not available", type="negative")
            return
            
        # Call the actual stop endpoint
        response = await fetch_api(f"/executions/{execution_id}/stop", method="POST")
        if response and not response.get("error"):
            ui.notify(f"Execution {execution_id} stopped successfully", type="positive")
            # Refresh page to show updated status
            await asyncio.sleep(1)
            ui.navigate.reload()
        else:
            error_msg = response.get("error", {}).get("message", "Unknown error") if response else "Failed to stop execution"
            ui.notify(f"Failed to stop execution: {error_msg}", type="negative")
    except Exception as e:
        logger.error(f"Error stopping execution {execution_id}: {e}")
        ui.notify(f"Error stopping execution: {str(e)}", type="negative")


async def view_metrics_from_live(execution_id, fetch_api=None):
    """View metrics for a strategy execution from live dashboard"""
    try:
        if not fetch_api:
            ui.notify("API not available", type="negative")
            return
            
        # Fetch execution metrics
        response = await fetch_api(f"/executions/{execution_id}/metrics")
        if response and not response.get("error"):
            metrics = response.get("metrics", {})
            
            # Show metrics in a dialog
            with ui.dialog() as dialog, ui.card().classes("dashboard-card min-w-[600px]"):
                with ui.column().classes("p-6 gap-4"):
                    ui.label(f"Execution Metrics: {execution_id}").classes("text-xl font-bold text-white")
                    
                    # Display key metrics
                    with ui.grid(columns=2).classes("w-full gap-4"):
                        with ui.card().classes("p-4 bg-gray-800/50"):
                            ui.label("Total P&L").classes("text-gray-400 text-sm")
                            pnl = float(metrics.get("total_pnl", 0))
                            pnl_color = "text-green-400" if pnl >= 0 else "text-red-400"
                            ui.label(f"₹{pnl:,.2f}").classes(f"text-lg font-bold {pnl_color}")
                        
                        with ui.card().classes("p-4 bg-gray-800/50"):
                            ui.label("Trades Executed").classes("text-gray-400 text-sm")
                            ui.label(str(metrics.get("trades_executed", 0))).classes("text-lg font-bold text-blue-400")
                        
                        with ui.card().classes("p-4 bg-gray-800/50"):
                            ui.label("Win Rate").classes("text-gray-400 text-sm")
                            win_rate = float(metrics.get("win_rate", 0))
                            ui.label(f"{win_rate:.1f}%").classes("text-lg font-bold text-purple-400")
                        
                        with ui.card().classes("p-4 bg-gray-800/50"):
                            ui.label("Duration").classes("text-gray-400 text-sm")
                            duration = metrics.get("duration", "Unknown")
                            ui.label(str(duration)).classes("text-lg font-bold text-cyan-400")
                    
                    ui.button("Close", on_click=dialog.close).classes("self-end mt-4")
            
            dialog.open()
        else:
            error_msg = response.get("error", {}).get("message", "Unknown error") if response else "Failed to fetch metrics"
            ui.notify(f"Failed to fetch metrics: {error_msg}", type="negative")
    except Exception as e:
        logger.error(f"Error viewing metrics for execution {execution_id}: {e}")
        ui.notify(f"Error viewing metrics: {str(e)}", type="negative")


def navigate_to_strategy_details(strategy):
    """Navigate to strategy details or execution view"""
    try:
        # Navigate to strategies page with focus on this strategy
        ui.navigate.to('/strategies')
        ui.notify(f"Navigating to strategy management for {strategy.get('name', 'Unknown')}", type="info")
    except Exception as e:
        ui.notify(f"Error navigating: {str(e)}", type="negative")