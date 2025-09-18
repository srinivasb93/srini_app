"""
Backtesting Module for NiceGUI Algo Trading Application
Implements UI for running backtests on defined strategies with enhanced UX and visualizations.
"""

from nicegui import ui
import pandas as pd
import numpy as np
import asyncio
import json
import logging
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from plotly.subplots import make_subplots
import aiohttp
import time
from functools import lru_cache
from uuid import uuid4
from optimization_ui import OptimizationUI

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

PREDEFINED_STRATEGIES = {
    "MACD Crossover": "MACD Crossover",
    "Bollinger Bands": "Bollinger Bands",
    "RSI Oversold/Overbought": "RSI Oversold/Overbought",
    "Short Sell Optimization": "Short Sell Optimization"
}

class BacktestMetrics:
    """Calculates comprehensive performance metrics from tradebook."""

    @staticmethod
    @lru_cache(maxsize=16)
    def calculate_metrics(tradebook_json: str, initial_capital: float) -> Dict:
        """Calculate metrics with caching for improved performance"""
        if not tradebook_json:
            return {}

        tradebook = json.loads(tradebook_json)
        if not tradebook:
            return {}

        df = pd.DataFrame(tradebook)
        if df.empty:
            return {}

        # Correctly convert 'Date' column to datetime objects for calculations
        df["timestamp"] = pd.to_datetime(df["Date"])
        df.sort_values(by="timestamp", inplace=True)
        df["returns"] = df["Profit"].fillna(0) / df["PortfolioValue"].shift(1).fillna(initial_capital)

        # --- Trade Cycle Analysis for Accurate Metrics ---
        buy_indices = df.index[df['Action'] == 'BUY'].tolist()
        trade_cycles = []
        if not buy_indices:
            return {"TotalProfit": 0, "WinRate": 0, "TotalTrades": 0}

        for i, buy_index in enumerate(buy_indices):
            start_index = buy_index
            end_index = buy_indices[i + 1] if i + 1 < len(buy_indices) else len(df)
            cycle_df = df.iloc[start_index:end_index]
            total_profit = cycle_df[cycle_df['Action'] == 'SELL']['Profit'].sum()
            trade_cycles.append({"profit": total_profit, "win": total_profit > 0})

        cycle_df = pd.DataFrame(trade_cycles)
        total_trades = len(cycle_df)
        winning_trades = cycle_df['win'].sum()

        # --- Equity Curve and Drawdown ---
        df['CumulativeProfit'] = df['Profit'].fillna(0).cumsum()
        df['PortfolioValue'] = initial_capital + df['CumulativeProfit']

        rolling_max = df['PortfolioValue'].cummax()
        drawdown = (df['PortfolioValue'] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100

        # --- Sharpe Ratio ---
        sharpe_ratio = 0
        if not df['returns'].empty and df['returns'].std() != 0:
            excess_returns = df['returns'] - (0.05 / 252)  # Assuming 5% risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / df['returns'].std()

        # --- Streaks ---
        streaks = (cycle_df['win'] != cycle_df['win'].shift()).cumsum()
        winning_streak = streaks[cycle_df['win']].value_counts().max() if winning_trades > 0 else 0
        losing_streak = streaks[~cycle_df['win']].value_counts().max() if total_trades > winning_trades else 0

        # --- CAGR and Calmar ---
        years = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days / 365.25 if len(df["timestamp"]) > 1 else 0
        cagr = 0
        if years > 0:
            cagr = ((df['PortfolioValue'].iloc[-1] / initial_capital) ** (1 / years)) - 1
        calmar_ratio = (cagr * 100) / max_drawdown if max_drawdown > 0 else 0

        # --- Final Metrics Dictionary ---
        return {
            "TotalProfit": float(cycle_df["profit"].sum()),
            "WinRate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "TotalTrades": total_trades,
            "FinalPortfolioValue": float(df["PortfolioValue"].iloc[-1]),
            "MaxDrawdown": float(max_drawdown),
            "SharpeRatio": float(sharpe_ratio),
            "ProfitFactor": float(cycle_df[cycle_df['profit'] > 0]['profit'].sum() / abs(
                cycle_df[cycle_df['profit'] < 0]['profit'].sum()) if abs(
                cycle_df[cycle_df['profit'] < 0]['profit'].sum()) > 0 else float('inf')),
            "AverageWin": float(cycle_df[cycle_df['win']]['profit'].mean()) if winning_trades > 0 else 0,
            "AverageLoss": float(cycle_df[~cycle_df['win']]['profit'].mean()) if total_trades > winning_trades else 0,
            "LargestWin": float(cycle_df["profit"].max()) if not cycle_df.empty else 0,
            "LargestLoss": float(cycle_df["profit"].min()) if not cycle_df.empty else 0,
            "WinningStreak": int(winning_streak),
            "LosingStreak": int(losing_streak),
            "CAGR": float(cagr),
            "AnnualizedVolatility": float(df['returns'].std() * np.sqrt(252) if not df['returns'].empty else 0),
            "CalmarRatio": float(calmar_ratio)
        }

def convert_timestamps_to_iso(data):
    """Recursively convert pandas Timestamps and datetime objects to ISO strings"""
    if isinstance(data, (pd.Timestamp, datetime)):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: convert_timestamps_to_iso(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_timestamps_to_iso(item) for item in data]
    elif isinstance(data, pd.DataFrame):
        return convert_timestamps_to_iso(data.to_dict('records'))
    elif isinstance(data, pd.Series):
        return convert_timestamps_to_iso(data.to_dict())
    else:
        return data

async def fetch_ohlc_data(fetch_api, instrument, from_date, to_date, interval="1day"):
    """
    Fetch historical OHLC data for candlestick charts using the existing API

    Args:
        fetch_api: Function to make API calls
        instrument: Trading symbol or instrument token
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
        interval: Data interval (1day, 1hour, etc.)

    Returns:
        DataFrame with OHLC data or empty DataFrame if error
    """
    try:
        # Format parameters for API call
        params = {
            "instrument": instrument,
            "from_date": from_date,
            "to_date": to_date,
            "interval": interval.split("day")[0] if "day" in interval else interval,
            "unit": "days" if "day" in interval else "minutes",
            "source": "default"  # Can be "default", "db", "upstox", or "openchart"
        }

        # Make API request
        response = await fetch_api("/historical-data/Upstox", params=params)

        if response and not response.get("error"):
            candles = response.get("data", [])

            # Convert to DataFrame
            df = pd.DataFrame(candles)

            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df
        else:
            error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
            logger.warning(f"Failed to fetch OHLC data: {error_msg}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error fetching OHLC data: {str(e)}")
        return pd.DataFrame()

def create_candlestick_chart(df, tradebook, theme_mode="Dark"):
    """Create candlestick chart with buy/sell markers using ECharts"""

    # Prepare candlestick data
    candlestick_data = []
    dates = []
    buy_points = []
    sell_points = []

    # Format candlestick data
    for index, row in df.iterrows():
        candlestick_data.append([
            row['open'],
            row['close'],
            row['low'],
            row['high']
        ])
        dates.append(row['timestamp'])

    # Extract buy/sell points from tradebook
    for trade in tradebook:
        if isinstance(trade.get("Date"), str):
            trade_date = datetime.strptime(trade["Date"].split("T")[0], "%Y-%m-%d")
        else:
            trade_date = trade.get("Date")

        if trade_date:
            date_str = trade_date.strftime('%Y-%m-%d')
            price = trade.get("EntryPrice", 0)

            if trade.get("Action") == "BUY":
                buy_points.append([date_str, price])
            elif trade.get("Action") == "SELL":
                sell_points.append([date_str, price])

    # Configure chart colors based on theme
    up_color = '#ef5350'
    down_color = '#26a69a'
    bg_color = 'transparent'
    text_color = '#ffffff' if theme_mode == "Dark" else '#333333'

    # Create ECharts options
    options = {
        "animation": True,
        "backgroundColor": bg_color,
        "textStyle": {
            "color": text_color
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "cross",
                "label": {
                    "backgroundColor": "#6a7985"
                }
            }
        },
        "legend": {
            "data": ["K-Line", "Buy", "Sell"],
            "textStyle": {
                "color": text_color
            }
        },
        "grid": {
            "left": "3%",
            "right": "3%",
            "bottom": "3%",
            "containLabel": True
        },
        "xAxis": {
            "type": "category",
            "data": dates,
            "axisLine": {
                "lineStyle": {
                    "color": text_color
                }
            },
            "splitLine": {
                "show": False
            }
        },
        "yAxis": {
            "type": "value",
            "scale": True,
            "axisLine": {
                "lineStyle": {
                    "color": text_color
                }
            },
            "splitLine": {
                "lineStyle": {
                    "color": "rgba(128, 128, 128, 0.2)"
                }
            }
        },
        "series": [
            {
                "name": "K-Line",
                "type": "candlestick",
                "data": candlestick_data,
                "itemStyle": {
                    "color": up_color,
                    "color0": down_color,
                    "borderColor": up_color,
                    "borderColor0": down_color
                }
            },
            {
                "name": "Buy",
                "type": "scatter",
                "data": buy_points,
                "symbol": "triangle",
                "symbolSize": 12,
                "itemStyle": {
                    "color": "#4caf50"
                }
            },
            {
                "name": "Sell",
                "type": "scatter",
                "data": sell_points,
                "symbol": "triangle-down",
                "symbolSize": 12,
                "itemStyle": {
                    "color": "#f44336"
                }
            }
        ]
    }

    return options

# Add this special converter for Plotly figures
def convert_plotly_timestamps(fig):
    """Convert timestamps in a Plotly figure to ISO format strings"""
    # Make a deep copy to avoid modifying the original
    import copy
    fig_copy = copy.deepcopy(fig)

    # Process each trace in the figure
    for trace in fig_copy.data:
        # Handle x values
        if hasattr(trace, 'x') and trace.x is not None:
            if isinstance(trace.x, (list, tuple)):
                trace.x = [x.isoformat() if isinstance(x, (pd.Timestamp, datetime)) else x for x in trace.x]

        # Handle y values
        if hasattr(trace, 'y') and trace.y is not None:
            if isinstance(trace.y, (list, tuple)):
                trace.y = [y.isoformat() if isinstance(y, (pd.Timestamp, datetime)) else y for y in trace.y]

    return fig_copy

def process_tradebook(tradebook):
    """Process tradebook once for all visualizations and displays"""
    if not tradebook:
        return pd.DataFrame()

    df = pd.DataFrame(tradebook)
    df["timestamp"] = pd.to_datetime(df["Date"])
    df["value"] = pd.to_numeric(df["PortfolioValue"])
    df["Action"] = df.get("Action", pd.Series(["BUY"] * len(df)))
    df["Quantity"] = df.get("Quantity", pd.Series([1] * len(df)))
    df = df.sort_values("timestamp")
    return df

def process_tradebook_for_display(tradebook: List[Dict]) -> List[Dict]:
    if not tradebook: return []
    df = pd.DataFrame(tradebook)
    df['timestamp'] = pd.to_datetime(df['Date'])

    buy_indices = df.index[df['Action'] == 'BUY'].tolist()
    completed_trades = []

    for i, buy_index in enumerate(buy_indices):
        start_index = buy_index
        end_index = buy_indices[i+1] if i+1 < len(buy_indices) else len(df)
        trade_cycle = df.iloc[start_index:end_index]

        entry_trade = trade_cycle.iloc[0]
        exit_trades = trade_cycle[trade_cycle['Action'] == 'SELL']

        if not exit_trades.empty:
            exit_price = (exit_trades['Price'] * exit_trades['Quantity']).sum() / exit_trades['Quantity'].sum()
            exit_reason = ", ".join(exit_trades['Reason'].unique())
            exit_date = exit_trades['timestamp'].max()
            pnl = exit_trades['Profit'].sum()
        else:
            exit_price, exit_reason, exit_date, pnl = None, "Open", None, 0

        holding_period = (exit_date - entry_trade['timestamp']) if exit_date else "N/A"

        completed_trades.append({
            "EntryDate": entry_trade['timestamp'].isoformat(),
            "ExitDate": exit_date.isoformat() if exit_date else "N/A",
            "EntryPrice": entry_trade['Price'], "ExitPrice": exit_price,
            "Quantity": entry_trade['Quantity'], "PNL": pnl,
            "ExitReason": exit_reason, "HoldingPeriod": str(holding_period)
        })
    return completed_trades

async def connect_backtest_websocket(user_id: str, progress_bar, progress_label, max_retries=5, initial_backoff=2, timeout=30):
    ws_url = f"{BASE_URL.replace('http', 'ws')}/ws/backtest/{user_id}"
    retry_count = 0

    # Function to safely update UI elements
    def safe_notify(message, type_):
        # Use ui.timer to ensure UI updates run in the correct context
        ui.timer(0, lambda: ui.notify(message, type=type_), once=True)

    while retry_count < max_retries:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.ws_connect(ws_url) as ws:
                    logger.info(f"WebSocket connected to {ws_url}")
                    safe_notify("Backtest progress updates connected.", "positive")
                    retry_count = 0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                progress = min(data.get("progress", 0), 1.0)
                                # Update UI elements safely
                                ui.timer(0, lambda: setattr(progress_bar, 'value', progress), once=True)
                                ui.timer(0, lambda: setattr(progress_label, 'text', f"Progress: {progress*100:.1f}%"), once=True)
                                logger.debug(f"WebSocket progress: {progress*100:.1f}%")
                                if progress >= 1:
                                    return
                            except json.JSONDecodeError:
                                logger.error(f"Invalid WebSocket message: {msg.data}")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info("WebSocket connection closed.")
                            break
        except aiohttp.ClientError as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                backoff = initial_backoff * (2 ** retry_count)
                safe_notify(f"Progress updates disconnected, retrying in {backoff} seconds...", "warning")
                await asyncio.sleep(backoff)
            else:
                safe_notify("Failed to connect to progress updates. Continuing without real-time updates.", "negative")
                ui.timer(0, lambda: setattr(progress_label, 'text', "Progress: N/A"), once=True)
                break


async def render_backtesting_page(fetch_api, user_storage, instruments):
    """Renders the backtesting UI with a modern two-column layout."""
    broker = user_storage.get("default_broker", "Zerodha")
    strategy_options = {}
    partial_exit_rows = []

    with ui.splitter(value=25, limits=(23, 35)).classes("w-full h-screen") as splitter:

        with splitter.before:
            with ui.card().classes("w-full h-full p-4 overflow-auto"):
                ui.label("Backtest Configuration").classes("text-h6 mb-3")

                with ui.expansion("Instrument & Strategy", icon="tune", value=True).classes("w-full"):
                    instrument_select = ui.select(options=sorted(list(instruments.keys())),
                                                  label="Select Instrument", with_input=True).props(
                        "clearable dense").classes("w-full")
                    strategies_select = ui.select(options=strategy_options, label="Select Strategy").props(
                        "dense").classes("w-full")
                    timeframe_select = ui.select(
                        options=["1min", "5min", "15min", "30min", "60min", "day", "week"],
                        label="Data Timeframe",
                        value="day"
                    ).props("dense hint='Timeframe for backtesting data'").classes("w-full")

                with ui.expansion("Date Range & Capital", icon="date_range", value=True).classes("w-full"):
                    with ui.row().classes("w-full gap-2"):
                        start_date = ui.input("Start Date", value=(datetime.now() - timedelta(days=365)).strftime(
                            "%Y-%m-%d")).props("dense type=date").classes("flex-1 min-w-0")
                        end_date = ui.input("End Date", value=datetime.now().strftime("%Y-%m-%d")).props(
                            "dense type=date").classes("flex-1 min-w-0")
                    initial_capital = ui.number(label="Initial Capital", value=100000, format="%.0f").props(
                        "dense").classes("w-full")

                with ui.expansion("Risk Management", icon="shield", value=True).classes("w-full"):
                    stop_loss_percent = ui.number("Stop Loss (%)", value=2.0, format="%.1f", min=0).props(
                        "dense").classes("w-full")
                    take_profit_percent = ui.number("Take Profit (%)", value=5.0, format="%.1f", min=0).props(
                        "dense").classes("w-full")

                    with ui.row().classes("w-full gap-2"):
                        trailing_stop_loss_percent = ui.number("Trailing Stop (%)", value=1.5, format="%.1f",
                                                               min=0).props("dense").classes("w-full")
                        position_sizing_percent = ui.number("Position Sizing (% of Capital)", value=10.0,
                                                            format="%.1f").props("dense").classes("w-full")
                    ui.separator().classes("my-2")

                    ui.label("Advanced Exit Strategies").classes("text-subtitle2 font-bold")
                    with ui.expansion("Partial Exits", icon="trending_down").classes("w-full"):
                        partial_exits_container = ui.column().classes("w-full")

                        def add_partial_exit():
                            with partial_exits_container:
                                with ui.row().classes("w-full items-center gap-2"):
                                    target = ui.number("Target %", value=5.0, format="%.1f").props("dense").classes(
                                        "flex-1 min-w-0")
                                    qty_percent = ui.number("Qty %", value=50.0, format="%.1f").props(
                                        "dense").classes("flex-1 min-w-0")
                                    remove_btn = ui.button(icon="delete",
                                                           on_click=lambda: remove_partial_exit(row_data)).props(
                                        "flat color=negative dense").classes("shrink-0")

                                row_data = {"target": target, "qty_percent": qty_percent, "remove_btn": remove_btn}
                                partial_exit_rows.append(row_data)

                        def remove_partial_exit(row_data):
                            if row_data in partial_exit_rows:
                                partial_exit_rows.remove(row_data)
                                row_data["target"].delete()
                                row_data["qty_percent"].delete()
                                row_data["remove_btn"].delete()

                        ui.button("Add Partial Exit", icon="add", on_click=add_partial_exit).props(
                            "outline").classes("w-full")

                with ui.expansion("Parameter Optimization", icon="auto_awesome").classes("w-full"):
                    enable_optimization = ui.switch("Enable Optimization", value=False).classes("w-full")
                    optimization_iterations = ui.number("Iterations", value=10, format="%.0f").props(
                        "dense").classes("w-full")

                    ui.label("Stop Loss Range").classes("text-subtitle2")
                    with ui.row().classes("w-full gap-2"):
                        stop_loss_min = ui.number("SL Min (%)", value=1.0, min=0.1, format="%.1f").props(
                            "dense").classes("flex-1 min-w-0")
                        stop_loss_max = ui.number("SL Max (%)", value=5.0, min=0.1, format="%.1f").props(
                            "dense").classes("flex-1 min-w-0")

                    with ui.row().classes("w-full gap-2"):
                        ui.label("Take Profit Range").classes("text-subtitle2")
                        use_take_profit_opt = ui.switch("Optimize TP").classes("shrink-0")
                    with ui.row().classes("w-full gap-2"):
                        take_profit_min = ui.number("TP Min (%)", value=2.0, min=0.0, format="%.1f").props(
                            "dense").classes("flex-1 min-w-0")
                        take_profit_max = ui.number("TP Max (%)", value=8.0, min=0.0, format="%.1f").props(
                            "dense").classes("flex-1 min-w-0")

                    with ui.row().classes("w-full gap-2"):
                        ui.label("Trail Stop Range").classes("text-subtitle2")
                        use_trail = ui.switch("Optimize Trail SL").classes("shrink-0")
                    with ui.row().classes("w-full gap-2"):
                        trail_stop_loss_min = ui.number("Trail SL Min (%)", value=1.0, min=0.0,
                                                        format="%.1f").props("dense").classes("flex-1 min-w-0")
                        trail_stop_loss_max = ui.number("Trail SL Max (%)", value=5.0, min=0.0,
                                                        format="%.1f").props("dense").classes("flex-1 min-w-0")

                # Optimization UI
                optimization_ui = OptimizationUI()
                optimization_ui.render_optimization_controls()

                run_button = ui.button("Run Backtest", on_click=lambda: run_backtest()).props(
                    "color=primary icon=play_arrow").classes("w-full mt-4")

        with splitter.after:
            with ui.card().classes("w-full h-full p-2"):
                results_tabs = ui.tabs().classes("w-full")
                with results_tabs:
                    performance_tab = ui.tab("Performance", icon="trending_up")
                    trades_tab = ui.tab("Trades", icon="list_alt")
                    metrics_tab = ui.tab("Metrics", icon="analytics")
                    optimization_tab = ui.tab("Optimization", icon="auto_awesome")

                with ui.tab_panels(results_tabs, value=performance_tab).classes("w-full mt-2 overflow-auto").style("height: calc(100vh - 120px);"):
                    with ui.tab_panel(performance_tab) as performance_panel:
                        ui.label("Run a backtest to see performance charts.").classes("absolute-center text-gray-500")
                    with ui.tab_panel(trades_tab) as trades_panel:
                        ui.label("Run a backtest to see the trade log.").classes("absolute-center text-gray-500")
                    with ui.tab_panel(metrics_tab) as metrics_panel:
                        ui.label("Run a backtest to see performance metrics.").classes("absolute-center text-gray-500")
                    with ui.tab_panel(optimization_tab) as optimization_panel:
                        ui.label("Run an optimization to see comparison results.")

    async def fetch_strategies_for_backtest():
        nonlocal strategy_options
        try:
            custom_strategies_list = await fetch_api(f"/strategies/all/{broker}")
            custom_strategies = {s["strategy_id"]: s["name"] for s in
                                 custom_strategies_list} if custom_strategies_list else {}
            strategy_options.clear()
            strategy_options.update(PREDEFINED_STRATEGIES)
            strategy_options.update(custom_strategies)
            strategies_select.options = strategy_options
            strategies_select.props("disabled=false")
            strategies_select.update()
        except Exception as e:
            ui.notify(f"Error fetching strategies: {e}", type="negative")

    instrument_select.on("update:model-value", fetch_strategies_for_backtest)

    async def run_backtest():
        try:
            if not strategies_select.value or not instrument_select.value:
                ui.notify("Please select a strategy and instrument.", type="negative");
                return

            for panel in [performance_panel, trades_panel, metrics_panel, optimization_panel]:
                panel.clear()
                with panel: ui.spinner(size='lg').classes('absolute-center')
            run_button.props("loading=true disabled=true")

            params = {
                "initial_investment": float(initial_capital.value),
                "stop_loss_percent": float(stop_loss_percent.value),
                "take_profit_percent": float(take_profit_percent.value),
                "trailing_stop_loss_percent": float(trailing_stop_loss_percent.value),
                "position_sizing_percent": float(position_sizing_percent.value),
                "enable_optimization": enable_optimization.value,
                "partial_exits": [{"target": r["target"].value, "qty_percent": r["qty_percent"].value} for r in
                                  partial_exit_rows]
            }

            # NEW: Add optimization config
            opt_config = optimization_ui.get_optimization_config()
            if opt_config:
                params["optimization_config"] = opt_config
                ui.notify("Parameter optimization enabled!", type="info")

            if enable_optimization.value:
                if stop_loss_min.value >= stop_loss_max.value:
                    ui.notify("Stop Loss Min must be less than Max.", type="negative")
                    return

                params["optimization_iterations"] = int(optimization_iterations.value)
                params["stop_loss_range"] = [float(stop_loss_min.value), float(stop_loss_max.value)]

                # Take profit optimization
                if use_take_profit_opt.value and take_profit_max.value >= take_profit_min.value > 0:
                    params["take_profit_range"] = [float(take_profit_min.value), float(take_profit_max.value)]

                # Trailing stop optimization
                if use_trail.value and trail_stop_loss_max.value >= trail_stop_loss_min.value > 0:
                    params["trailing_stop_range"] = [float(trail_stop_loss_min.value), float(trail_stop_loss_max.value)]

            strategy_id = strategies_select.value
            strategy_value = strategy_id
            if strategy_id not in PREDEFINED_STRATEGIES:
                strategy_response = await fetch_api(f"/strategies/{strategy_id}")
                if not (strategy_response and not strategy_response.get("error")):
                    ui.notify("Failed to fetch custom strategy definition.", type="negative");
                    return
                strategy_value = json.dumps(strategy_response)

            backtest_payload = {
                "trading_symbol": instrument_select.value,
                "instrument_token": instruments[instrument_select.value],
                "timeframe": timeframe_select.value,
                "strategy": strategy_value,
                "params": params,
                "start_date": start_date.value,
                "end_date": end_date.value
            }

            response = await fetch_api("/algo-trading/backtest", method="POST", data=backtest_payload)
            # logger.debug(f"ALGO RESPONSE: {json.dumps(response)}")
            for panel in [performance_panel, trades_panel, metrics_panel, optimization_panel]:
                panel.clear()

            if response and not response.get("error"):
                if response.get("optimization_enabled"):
                    ui.notify("Optimization complete! Displaying best result.", type="positive")
                    display_backtest_results(performance_panel, trades_panel, metrics_panel,
                                             response.get("best_result", {}), user_storage, fetch_api)
                    display_optimization_runs(optimization_panel, response.get("all_runs", []))
                    results_tabs.set_value(optimization_tab)
                else:
                    ui.notify("Backtest complete!", type="positive")
                    display_backtest_results(performance_panel, trades_panel, metrics_panel, response, user_storage, fetch_api)
                    results_tabs.set_value(performance_tab)
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error")
                with performance_panel:
                    ui.label(f"Backtest failed: {error_msg}").classes("text-negative")
        finally:
            run_button.props("loading=false disabled=false")
            run_button._props["loading"] = False
            run_button._props["disabled"] = False
            run_button.update()
            ui.update()


def create_enhanced_performance_dashboard(df, metrics, results, theme_mode="Dark"):
    """Create an enhanced performance dashboard with modern design"""

    # Theme-aware colors
    if theme_mode == "Dark":
        colors = {
            'font': '#ffffff',
            'bg_primary': 'rgba(26, 32, 44, 0.95)',
            'bg_secondary': 'rgba(45, 55, 72, 0.8)',
            'accent_blue': '#3182ce',
            'accent_green': '#38a169',
            'accent_red': '#e53e3e',
            'accent_orange': '#dd6b20',
            'accent_purple': '#805ad5',
            'grid': 'rgba(255, 255, 255, 0.1)',
            'border': 'rgba(255, 255, 255, 0.2)'
        }
    else:
        colors = {
            'font': '#2d3748',
            'bg_primary': 'rgba(247, 250, 252, 0.95)',
            'bg_secondary': 'rgba(237, 242, 247, 0.8)',
            'accent_blue': '#2b6cb0',
            'accent_green': '#2f855a',
            'accent_red': '#c53030',
            'accent_orange': '#c05621',
            'accent_purple': '#6b46c1',
            'grid': 'rgba(0, 0, 0, 0.1)',
            'border': 'rgba(0, 0, 0, 0.2)'
        }

    return colors


def create_performance_overview_cards(metrics, results, theme_mode="Dark"):
    """Create enhanced KPI cards with trend indicators"""
    colors = create_enhanced_performance_dashboard(None, None, None, theme_mode)

    # Calculate additional metrics
    total_profit = metrics.get('TotalProfit', 0)
    initial_investment = results.get('InitialInvestment', 100000)
    roi_percent = (total_profit / initial_investment * 100) if initial_investment > 0 else 0
    win_rate = metrics.get('WinRate', 0)
    total_trades = metrics.get('TotalTrades', 0)
    sharpe_ratio = metrics.get('SharpeRatio', 0)
    max_drawdown = metrics.get('MaxDrawdown', 0)

    # Performance grade calculation
    def get_performance_grade(roi, sharpe, win_rate, max_dd):
        score = 0
        if roi > 15:
            score += 25
        elif roi > 8:
            score += 15
        elif roi > 0:
            score += 10

        if sharpe > 2:
            score += 25
        elif sharpe > 1:
            score += 15
        elif sharpe > 0:
            score += 10

        if win_rate > 70:
            score += 25
        elif win_rate > 50:
            score += 15
        elif win_rate > 30:
            score += 10

        if max_dd < 5:
            score += 25
        elif max_dd < 10:
            score += 15
        elif max_dd < 20:
            score += 10

        if score >= 80:
            return "A+", colors['accent_green']
        elif score >= 70:
            return "A", colors['accent_green']
        elif score >= 60:
            return "B+", colors['accent_blue']
        elif score >= 50:
            return "B", colors['accent_blue']
        elif score >= 40:
            return "C", colors['accent_orange']
        else:
            return "D", colors['accent_red']

    grade, grade_color = get_performance_grade(roi_percent, sharpe_ratio, win_rate, max_drawdown)

    return {
        'total_return': {
            'value': f"₹{total_profit:,.0f}",
            'subtitle': f"{roi_percent:.1f}% ROI",
            'icon': 'trending_up' if total_profit > 0 else 'trending_down',
            'color': colors['accent_green'] if total_profit > 0 else colors['accent_red'],
            'trend': 'positive' if total_profit > 0 else 'negative'
        },
        'win_rate': {
            'value': f"{win_rate:.1f}%",
            'subtitle': f"{int(total_trades * win_rate / 100) if total_trades > 0 else 0}/{total_trades} wins",
            'icon': 'military_tech',
            'color': colors['accent_blue'],
            'trend': 'positive' if win_rate > 50 else 'negative'
        },
        'sharpe_ratio': {
            'value': f"{sharpe_ratio:.2f}",
            'subtitle': 'Risk-Adjusted Return',
            'icon': 'analytics',
            'color': colors['accent_purple'],
            'trend': 'positive' if sharpe_ratio > 1 else 'negative'
        },
        'performance_grade': {
            'value': grade,
            'subtitle': 'Overall Performance',
            'icon': 'grade',
            'color': grade_color,
            'trend': 'positive' if grade in ['A+', 'A', 'B+'] else 'negative'
        }
    }


def create_enhanced_main_chart(df, metrics, theme_mode="Dark"):
    """Create enhanced main performance chart with multiple visualizations"""
    if df.empty:
        return None

    colors = create_enhanced_performance_dashboard(None, None, None, theme_mode)

    # Create 2x2 subplot layout for better organization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("📈 Portfolio Value & Signals", "📊 Monthly Performance Heatmap",
                        "📉 Rolling Drawdown", "🎯 Trade Distribution"),
        specs=[[{"secondary_y": True}, {"type": "heatmap"}],
               [{"secondary_y": False}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    x_values = [t.isoformat() if isinstance(t, (pd.Timestamp, datetime)) else t for t in df["timestamp"]]

    # 1. Portfolio Value with Volume (Top Left)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=df["value"],
        mode="lines",
        name="Portfolio Value",
        line=dict(color=colors['accent_blue'], width=3),
        hovertemplate="<b>Portfolio</b><br>Date: %{x}<br>Value: ₹%{y:,.0f}<extra></extra>"
    ), row=1, col=1)

    # Add trade volume on secondary y-axis
    trade_volumes = df.groupby(df["timestamp"].dt.date).size().reindex(
        pd.date_range(df["timestamp"].min().date(), df["timestamp"].max().date(), freq='D'), fill_value=0
    )

    fig.add_trace(go.Bar(
        x=[d.isoformat() for d in trade_volumes.index],
        y=trade_volumes.values,
        name="Daily Trades",
        marker_color=colors['accent_orange'],
        opacity=0.3,
        yaxis="y2",
        hovertemplate="<b>Trade Volume</b><br>Date: %{x}<br>Trades: %{y}<extra></extra>"
    ), row=1, col=1, secondary_y=True)

    # Buy/Sell signals
    for idx, trade in df.iterrows():
        action = trade.get("Action")
        if action in ["BUY", "SELL"]:
            color = colors['accent_green'] if action == "BUY" else colors['accent_red']
            symbol = "triangle-up" if action == "BUY" else "triangle-down"

            fig.add_trace(go.Scatter(
                x=[trade["timestamp"].isoformat()],
                y=[trade["value"]],
                mode="markers",
                name=f"{action}",
                marker=dict(color=color, size=12, symbol=symbol),
                showlegend=False,
                hovertemplate=f"<b>{action}</b><br>Date: %{{x}}<br>Price: ₹{trade.get('EntryPrice', 0):.2f}<extra></extra>"
            ), row=1, col=1)

    # 2. Monthly Performance Heatmap (Top Right)
    try:
        monthly_data = df.groupby([df["timestamp"].dt.year, df["timestamp"].dt.month])["Profit"].sum().unstack(
            fill_value=0)

        fig.add_trace(go.Heatmap(
            z=monthly_data.values,
            x=[f"Month {i}" for i in range(1, 13)],
            y=monthly_data.index,
            colorscale=[[0, colors['accent_red']], [0.5, '#ffffff'], [1, colors['accent_green']]],
            hovertemplate="<b>Monthly Return</b><br>Year: %{y}<br>Month: %{x}<br>Return: ₹%{z:,.0f}<extra></extra>",
            showscale=False
        ), row=1, col=2)
    except:
        # Fallback if monthly grouping fails
        fig.add_trace(go.Scatter(
            x=[1], y=[1], mode="text",
            text=["Monthly data unavailable"],
            textfont=dict(color=colors['font'])
        ), row=1, col=2)

    # 3. Rolling Drawdown (Bottom Left)
    rolling_max = df["value"].expanding().max()
    drawdowns = (df["value"] / rolling_max - 1) * 100

    fig.add_trace(go.Scatter(
        x=x_values,
        y=drawdowns,
        mode="lines",
        name="Drawdown",
        line=dict(color=colors['accent_red'], width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(colors['accent_red'][1:3], 16)}, {int(colors['accent_red'][3:5], 16)}, {int(colors['accent_red'][5:7], 16)}, 0.3)",
        hovertemplate="<b>Drawdown</b><br>Date: %{x}<br>DD: %{y:.2f}%<extra></extra>",
        showlegend=False
    ), row=2, col=1)

    # 4. Trade Distribution Scatter (Bottom Right)
    trade_profits = df["Profit"].fillna(0)
    trade_returns = (trade_profits / df["value"].shift(1) * 100).fillna(0)

    fig.add_trace(go.Scatter(
        x=trade_returns,
        y=trade_profits,
        mode="markers",
        name="Trades",
        marker=dict(
            color=[colors['accent_green'] if p > 0 else colors['accent_red'] for p in trade_profits],
            size=[abs(p) / 100 + 5 for p in trade_profits],  # Size based on profit magnitude
            opacity=0.7,
            line=dict(color=colors['font'], width=1)
        ),
        hovertemplate="<b>Trade Analysis</b><br>Return: %{x:.1f}%<br>Profit: ₹%{y:,.0f}<extra></extra>",
        showlegend=False
    ), row=2, col=2)

    # Layout updates
    fig.update_layout(
        height=700,
        title=dict(
            text="📊 Advanced Portfolio Performance Dashboard",
            font=dict(color=colors['font'], size=20),
            x=0.5
        ),
        font=dict(color=colors['font']),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=colors['bg_secondary'],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(color=colors['font'])
        ),
        hoverlabel=dict(
            bgcolor=colors['bg_primary'],
            bordercolor=colors['border'],
            font_color=colors['font']
        )
    )

    # Update axes
    fig.update_xaxes(gridcolor=colors['grid'], title_font_color=colors['font'], tickfont_color=colors['font'])
    fig.update_yaxes(gridcolor=colors['grid'], title_font_color=colors['font'], tickfont_color=colors['font'])

    return fig


def create_performance_panel(performance_panel, df, metrics, results, theme_mode="Dark"):
    """Create the enhanced performance panel with modern design"""

    with performance_panel:
        with ui.scroll_area().classes("w-full h-full"):
            with ui.column().classes("w-full gap-6 p-2"):

                if not df.empty:
                    # 1. KPI Cards Section
                    ui.label("📊 Performance Overview").classes("text-xl font-bold mb-2")

                    kpi_data = create_performance_overview_cards(metrics, results, theme_mode)

                    with ui.grid(columns=4).classes("w-full gap-4 mb-2"):
                        for key, kpi in kpi_data.items():
                            with ui.card().classes("p-4 text-center hover:shadow-lg transition-all duration-300"):
                                # Icon and trend indicator
                                with ui.row().classes("items-center justify-between mb-2"):
                                    ui.icon(kpi['icon'], size="1rem").style(f"color: {kpi['color']};")
                                    trend_icon = "arrow_upward" if kpi['trend'] == 'positive' else "arrow_downward"
                                    trend_color = "#10b981" if kpi['trend'] == 'positive' else "#ef4444"
                                    ui.icon(trend_icon, size="1rem").style(f"color: {trend_color};")

                                # Main value
                                ui.label(kpi['value']).classes("text-2xl font-bold").style(f"color: {kpi['color']};")
                                ui.label(kpi['subtitle']).classes("text-sm text-gray-500 mt-1")

                    # 2. Advanced Chart Dashboard
                    with ui.card().classes("w-full p-2 mb-3"):
                        ui.label("📈 Advanced Performance Analytics").classes("text-xl font-bold mb-2")

                        enhanced_fig = create_enhanced_main_chart(df, metrics, theme_mode)
                        if enhanced_fig:
                            ui.plotly(enhanced_fig).classes("w-full")

                    # 3. Quick Insights Section
                    with ui.card().classes("w-full p-3"):
                        ui.label("🔍 Quick Insights").classes("text-xl font-bold mb-4")

                        with ui.grid(columns=2).classes("w-full gap-6"):
                            # Left column - Risk Analysis
                            with ui.column().classes("gap-3"):
                                ui.label("⚠️ Risk Analysis").classes("text-lg font-semibold mb-2")

                                max_dd = metrics.get('MaxDrawdown', 0)
                                if max_dd < 5:
                                    risk_level = "Low Risk"
                                    risk_color = "#10b981"
                                elif max_dd < 15:
                                    risk_level = "Moderate Risk"
                                    risk_color = "#f59e0b"
                                else:
                                    risk_level = "High Risk"
                                    risk_color = "#ef4444"

                                ui.chip(f"Risk Level: {risk_level}", color=risk_color).props("outline")
                                ui.chip(f"Max Drawdown: {max_dd:.1f}%",
                                        color=risk_color if max_dd > 10 else "#10b981").props("outline")

                                volatility = metrics.get('Volatility', 0)
                                ui.chip(f"Volatility: {volatility:.1f}%", color="#6366f1").props("outline")

                            # Right column - Performance Summary
                            with ui.column().classes("gap-3"):
                                ui.label("📈 Performance Summary").classes("text-lg font-semibold mb-2")

                                total_trades = metrics.get('TotalTrades', 0)
                                if total_trades > 0:
                                    avg_trade = metrics.get('TotalProfit', 0) / total_trades
                                    ui.chip(f"Avg Trade: ₹{avg_trade:,.0f}", color="#3b82f6").props("outline")

                                profit_factor = metrics.get('ProfitFactor', 0)
                                pf_color = "#10b981" if profit_factor > 1.5 else "#f59e0b" if profit_factor > 1 else "#ef4444"
                                ui.chip(f"Profit Factor: {profit_factor:.2f}", color=pf_color).props("outline")

                                ui.chip(f"Total Trades: {total_trades}", color="#8b5cf6").props("outline")

                else:
                    # Empty state with better design
                    with ui.card().classes("w-full p-12 text-center"):
                        ui.icon("bar_chart", size="4rem").classes("text-gray-400 mb-4")
                        ui.label("No Performance Data Available").classes("text-2xl font-bold text-gray-500 mb-2")
                        ui.label("Run a backtest to see detailed performance analytics").classes("text-gray-400 mb-6")
                        ui.button("Start Backtesting", icon="play_arrow").props("color=primary size=lg")

def display_backtest_results(performance_panel, trades_panel, metrics_panel, results, user_storage, fetch_api):
    """Displays backtest results with metrics and charts in a tabbed interface."""
    # Clear panels
    performance_panel.clear()
    trades_panel.clear()
    metrics_panel.clear()

    theme_mode = "Dark" if user_storage.get("dark_mode", True) else "Light"
    tradebook = results.get("Tradebook", [])

    # Process tradebook once for all visualizations
    df = process_tradebook(tradebook)

    # Calculate metrics from processed data
    tradebook_json = json.dumps(tradebook)
    metrics = BacktestMetrics.calculate_metrics(tradebook_json, results.get("InitialInvestment", 100000))

    # If BacktestMetrics returns empty or missing keys, use the direct results
    if not metrics or "TotalProfit" not in metrics:
        metrics = {
            "TotalProfit": results.get("TotalProfit", 0),
            "WinRate": results.get("WinRate", 0),
            "TotalTrades": results.get("TotalTrades", 0),
            "FinalPortfolioValue": results.get("EndValue", results.get("StartValue", 100000)),
            "MaxDrawdown": results.get("MaxDrawdown", 0),
            "SharpeRatio": results.get("SharpeRatio", 0),
            "ProfitFactor": results.get("ProfitFactor", 1.0),
            "AverageWin": results.get("AverageWin", 0),
            "AverageLoss": results.get("AverageLoss", 0),
            "LargestWin": results.get("LargestWin", 0),
            "LargestLoss": results.get("LargestLoss", 0),
            "WinningStreak": results.get("WinningStreak", 0),
            "LosingStreak": results.get("LosingStreak", 0),
            "CalmarRatio": results.get("CalmarRatio", 0),
            "CAGR": 0
        }

    # Store data for export
    user_storage["backtest_tradebook"] = convert_timestamps_to_iso(tradebook)

    # 1. Metrics Panel - Show key metrics
    with metrics_panel:
        with ui.card().classes("w-full p-4 mb-4"):
            ui.label(f"Strategy: {results.get('StrategyName', 'Unknown Strategy')}").classes("text-h6")
            optimized_params = results.get("OptimizedParameters", {})
            is_optimized = bool(optimized_params)

            if is_optimized:
                ui.separator().classes("my-3")
                ui.label("🎯 Optimized Parameters").classes("text-subtitle1 font-bold text-primary")

                with ui.grid(columns=4).classes("gap-2 mb-3"):
                    for param_name, param_value in optimized_params.items():
                        with ui.card().classes("p-2 bg-primary-50"):
                            param_display_name = param_name.replace('_', ' ').replace('percent', '%').title()
                            ui.label(param_display_name).classes("text-caption font-medium")
                            if 'percent' in param_name:
                                ui.label(f"{param_value:.2f}%").classes("text-subtitle2 font-bold text-primary")
                            else:
                                ui.label(f"{param_value}").classes("text-subtitle2 font-bold text-primary")
            ui.label(
                f"Period: {datetime.strptime(results.get('StartDate', ''), '%Y-%m-%d').strftime('%d %b %Y') if results.get('StartDate') else 'N/A'} to {datetime.strptime(results.get('EndDate', ''), '%Y-%m-%d').strftime('%d %b %Y') if results.get('EndDate') else 'N/A'}").classes("text-subtitle2")

        with ui.grid(columns=4).classes("gap-4 mb-4"):
            def display_metric(label, value, format_spec=",.2f", help_text=None, color=None):
                with ui.card().classes(f"items-center p-4 {color or ''}"):
                    ui.label(label).classes("text-caption")
                    formatted_value = f"{float(value):{format_spec}}" if isinstance(value, (int, float)) else str(value)
                    ui.label(formatted_value).classes("text-subtitle1 font-bold")
                    if help_text:
                        ui.button(icon="help_outline", on_click=lambda: ui.notify(help_text)).props("flat dense round").classes("absolute top-1 right-1")

            # Key performance metrics
            display_metric("Total Profit (₹)", metrics["TotalProfit"], help_text="Sum of all profits/losses from trades")
            display_metric("Win Rate (%)", metrics["WinRate"], help_text="Percentage of winning trades")
            display_metric("Total Trades", metrics["TotalTrades"], ",.0f", help_text="Number of trades executed")
            display_metric("Final Portfolio Value (₹)", metrics["FinalPortfolioValue"], help_text="Final capital after all trades")

            # Risk metrics
            color = "bg-red-50" if metrics["MaxDrawdown"] > 20 else ""
            display_metric("Max Drawdown (%)", metrics["MaxDrawdown"], help_text="Maximum percentage decline from peak", color=color)
            color = "bg-green-50" if metrics["SharpeRatio"] > 1 else ""
            display_metric("Sharpe Ratio", metrics["SharpeRatio"], help_text="Risk-adjusted return (higher is better)", color=color)
            display_metric("Profit Factor", metrics["ProfitFactor"], help_text="Gross profit divided by gross loss")
            display_metric("CAGR (%)", metrics.get("CAGR", 0) * 100, help_text="Compound Annual Growth Rate")

            # Enhanced metric display with additional context
            def get_trade_date_by_profit(tradebook, find_max=True):
                if not tradebook:
                    return "N/A"
                try:
                    compare_func = max if find_max else min
                    extreme_trade = compare_func(tradebook, key=lambda x: x.get("Profit", 0))
                    return datetime.strptime(extreme_trade.get("Date", ""), "%Y-%m-%dT%H:%M:%S").strftime(
                        "%d %b %Y") if extreme_trade.get("Date") else "N/A"
                except:
                    return "N/A"

            largest_win_date = get_trade_date_by_profit(tradebook, True)
            largest_loss_date = get_trade_date_by_profit(tradebook, False)

            # Trade statistics
            display_metric("Average Win (₹)", metrics["AverageWin"], help_text="Average profit on winning trades")
            display_metric("Average Loss (₹)", metrics["AverageLoss"], help_text="Average loss on losing trades")
            # Update the metrics display with dates
            display_metric("Largest Win (₹)", metrics["LargestWin"],
                           help_text=f"Biggest profitable trade on {largest_win_date}")
            display_metric("Largest Loss (₹)", metrics["LargestLoss"],
                           help_text=f"Biggest losing trade on {largest_loss_date}")

            # Streak information
            display_metric("Winning Streak", metrics["WinningStreak"], ",.0f", help_text="Longest consecutive winning trades")
            display_metric("Losing Streak", metrics["LosingStreak"], ",.0f", help_text="Longest consecutive losing trades")
            display_metric("Calmar Ratio", metrics.get("CalmarRatio", 0), help_text="CAGR divided by max drawdown")
            display_metric("Annual Volatility (%)", metrics.get("AnnualizedVolatility", 0) * 100, help_text="Annualized standard deviation of returns")

    # 2. Performance Panel - Show charts and performance overview
    create_performance_panel(performance_panel, df, metrics, results, theme_mode)

    # 3. Trades Panel - Show detailed trade list
    with trades_panel:
        completed_trades = process_tradebook_for_display(tradebook)
        if completed_trades:
            # Container with proper height distribution
            with ui.column().classes("w-full h-full gap-4"):
                # Single table with constrained height
                ui.aggrid({
                    "columnDefs": [
                        {"headerName": "Entry Date", "field": "EntryDate",
                         "valueFormatter": "value ? new Date(value).toLocaleDateString() : 'N/A'",
                         "width": 130},
                        {"headerName": "Exit Date", "field": "ExitDate",
                         "valueFormatter": "value ? new Date(value).toLocaleDateString() : 'N/A'",
                         "width": 130},
                        {"headerName": "Entry ₹", "field": "EntryPrice",
                         "valueFormatter": "params.value.toFixed(2)", "width": 110},
                        {"headerName": "Exit ₹", "field": "ExitPrice",
                         "valueFormatter": "params.value ? params.value.toFixed(2) : 'N/A'", "width": 110},
                        {"headerName": "Quantity", "field": "Quantity",
                         "valueFormatter": "params.value.toFixed(0)", "width": 100},
                        {"headerName": "P&L", "field": "PNL",
                         "valueFormatter": "params.value.toFixed(2)",
                         "cellStyle": "params.value >= 0 ? {'color': '#16a34a', 'font-weight': 'bold'} : {'color': '#dc2626', 'font-weight': 'bold'}",
                         "width": 120},
                        {"headerName": "Exit Reason", "field": "ExitReason", "width": 140},
                        {"headerName": "Duration", "field": "HoldingPeriod", "width": 120}
                    ],
                    "rowData": completed_trades,
                    "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
                    "domLayout": "normal",
                    "pagination": True,
                    "paginationPageSize": 20,
                    "rowHeight": 40,
                    "headerHeight": 50
                }).classes("w-full").style("height: 450px;")  # Fixed height instead of h-full

                # Quick stats chips
                with ui.row().classes("w-full gap-4 flex-wrap"):
                    ui.chip(f"Total Profit: ₹{metrics['TotalProfit']:,.2f}", icon="account_balance").props("outline")
                    ui.chip(f"Win Rate: {metrics['WinRate']:.1f}%", icon="percent").props("outline")
                    ui.chip(f"Trades: {metrics['TotalTrades']}", icon="format_list_numbered").props("outline")
                    ui.chip(f"Winning Trades: {int(metrics['TotalTrades'] * metrics['WinRate'] / 100)}",
                            icon="thumb_up").props("outline")
                    ui.chip(
                        f"Losing Trades: {metrics['TotalTrades'] - int(metrics['TotalTrades'] * metrics['WinRate'] / 100)}",
                        icon="thumb_down").props("outline")

                # TRADE STATISTICS (Always visible now)
                with ui.card().classes("w-full"):
                    with ui.card_section().classes("p-4"):
                        ui.label("📊 Trade Statistics").classes("text-h6 font-bold mb-3")

                        with ui.grid(columns=4).classes("gap-4"):
                            def create_stat_card(title, value, color="blue"):
                                with ui.card().classes(f"p-3 bg-{color}-50 border border-{color}-200"):
                                    ui.label(title).classes("text-sm font-medium text-gray-700")
                                    ui.label(str(value)).classes(f"text-xl font-bold text-{color}-700")

                            # Calculate statistics
                            avg_profit = metrics['TotalProfit'] / metrics['TotalTrades'] if metrics[
                                                                                            'TotalTrades'] > 0 else 0
                            win_loss_ratio = abs(metrics["AverageWin"] / metrics["AverageLoss"]) if metrics[
                                                                                                    "AverageLoss"] != 0 else 0
                            create_stat_card("Avg Profit/Trade", f"₹{avg_profit:.2f}", "green" if avg_profit > 0 else "red")
                            create_stat_card("Win/Loss Ratio", f"{win_loss_ratio:.2f}", "purple")
                            create_stat_card("Largest Win", f"₹{metrics['LargestWin']:.2f}", "green")
                            create_stat_card("Largest Loss", f"₹{abs(metrics['LargestLoss']):.2f}", "red")

        else:
            with ui.column().classes("w-full h-full items-center justify-center"):
                ui.icon("trending_down", size="4rem").classes("text-gray-400 mb-4")
                ui.label("No completed trades in this backtest").classes("text-h5 text-gray-500")


def display_optimization_runs(container, all_runs):
    """Enhanced optimization results display showing ALL optimization parameters."""
    with container:
        if not all_runs:
            ui.label("No optimization data available.").classes("absolute-center text-gray-500")
            return

        ui.label("Parameter Optimization Results").classes("text-h6 mb-4")

        # Analyze which parameters were optimized
        sample_params = all_runs[0].get('parameters', {}) if all_runs else {}
        optimized_params = list(sample_params.keys())

        ui.label(f"Optimized Parameters: {', '.join(optimized_params)}").classes("text-subtitle2 mb-2")

        # Enhanced row data to include ALL optimized parameters
        rowData = []
        for i, r in enumerate(all_runs):
            params = r.get('parameters', {})
            row = {
                "run_number": i + 1,
                "sl_percent": params.get('stop_loss_percent', 0),
                "tp_percent": params.get('take_profit_percent', 0),
                "trail_percent": params.get('trailing_stop_loss_percent', 0),
                "pnl": r.get('TotalPNL', 0),
                "win_rate": r.get('WinRate', 0),
                "trades": r.get('TotalTrades', 0),
                "sharpe": r.get('SharpeRatio', 0),
                "max_drawdown": r.get('MaxDrawdown', 0)
            }
            rowData.append(row)

        # Dynamic column definitions based on optimized parameters
        columnDefs = [
            {"headerName": "Run #", "field": "run_number", "width": 80, "pinned": "left"}
        ]

        # Add parameter columns based on what was actually optimized
        if any(r.get('sl_percent', 0) > 0 for r in rowData):
            columnDefs.append({
                "headerName": "Stop Loss (%)",
                "field": "sl_percent",
                "valueFormatter": "params.value.toFixed(2)",
                "width": 120,
                "cellStyle": "{'background-color': '#ffebee'}"
            })

        if any(r.get('tp_percent', 0) > 0 for r in rowData):
            columnDefs.append({
                "headerName": "Take Profit (%)",
                "field": "tp_percent",
                "valueFormatter": "params.value.toFixed(2)",
                "width": 130,
                "cellStyle": "{'background-color': '#e8f5e8'}"
            })

        if any(r.get('trail_percent', 0) > 0 for r in rowData):
            columnDefs.append({
                "headerName": "Trail Stop (%)",
                "field": "trail_percent",
                "valueFormatter": "params.value.toFixed(2)",
                "width": 120,
                "cellStyle": "{'background-color': '#fff3e0'}"
            })

        # Performance columns
        columnDefs.extend([
            {
                "headerName": "Total PNL (₹)",
                "field": "pnl",
                "valueFormatter": "params.value.toFixed(2)",
                "sort": "desc",
                "cellStyle": "params.value > 0 ? {'color': '#4caf50', 'font-weight': 'bold'} : {'color': '#f44336', 'font-weight': 'bold'}",
                "width": 130
            },
            {
                "headerName": "Win Rate (%)",
                "field": "win_rate",
                "valueFormatter": "params.value.toFixed(1)",
                "width": 110
            },
            {
                "headerName": "Trades",
                "field": "trades",
                "width": 80
            }
        ])

        ui.aggrid({
            "columnDefs": columnDefs,
            "rowData": rowData,
            "domLayout": 'autoHeight',
            "defaultColDef": {
                "sortable": True,
                "filter": True,
                "resizable": True
            },
            "rowSelection": "single",
            "pagination": True,
            "paginationPageSize": 20
        }).classes("w-full")

def display_strategy_comparison(container, all_results, user_storage):
    """Displays a comparison of multiple strategy backtests with enhanced visuals."""
    theme_mode = "Dark" if user_storage.get("dark_mode", True) else "Light"
    font_color = "white" if theme_mode == "Dark" else "black"
    grid_color = "rgba(128, 128, 128, 0.5)" if theme_mode == "Dark" else "rgba(200, 200, 200, 0.5)"

    with container:
        ui.label("Strategy Comparison").classes("text-h6 mb-4")

        # Prepare comparison data
        comparison_data = []
        for result in all_results:
            tradebook_json = json.dumps(result.get("Tradebook", []))
            metrics = BacktestMetrics.calculate_metrics(tradebook_json, result.get("InitialInvestment", 100000))

            for k, v in metrics.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    metrics[k] = v.isoformat()

            comparison_data.append({
                "Strategy": result.get("StrategyName", "Unknown"),
                "Total Profit": float(metrics["TotalProfit"]),
                "Win Rate": float(metrics["WinRate"]),
                "Sharpe Ratio": float(metrics["SharpeRatio"]),
                "Max Drawdown": float(metrics["MaxDrawdown"]),
                "Profit Factor": float(metrics["ProfitFactor"]),
                "CAGR": float(metrics.get("CAGR", 0) * 100),
                "Trades": int(metrics["TotalTrades"])
            })

        # Create comparison metrics table
        try:
            # Before creating the comparison grid
            comparison_data = convert_timestamps_to_iso(comparison_data)
            comparison_grid = ui.aggrid({
                "defaultColDef": {
                    "resizable": True,
                    "sortable": True,
                    "filter": True
                },
                "columnDefs": [
                    {"headerName": "Strategy", "field": "Strategy", "pinned": "left"},
                    {"headerName": "Total Profit (₹)", "field": "Total Profit",
                        "valueFormatter": "value.toFixed(2)",
                        "cellStyle": "params.value > 0 ? {'color': '#4caf50'} : {'color': '#f44336'}"
                    },
                    {"headerName": "Win Rate (%)", "field": "Win Rate", "valueFormatter": "value.toFixed(2)"},
                    {"headerName": "Sharpe Ratio", "field": "Sharpe Ratio",
                        "valueFormatter": "value.toFixed(3)",
                        "cellStyle": "params.value > 1 ? {'color': '#4caf50'} : params.value < 0 ? {'color': '#f44336'} : {}"
                    },
                    {"headerName": "Max Drawdown (%)", "field": "Max Drawdown",
                        "valueFormatter": "value.toFixed(2)",
                        "cellStyle": "params.value > 20 ? {'color': '#f44336'} : {}"
                    },
                    {"headerName": "Profit Factor", "field": "Profit Factor", "valueFormatter": "value.toFixed(2)"},
                    {"headerName": "CAGR (%)", "field": "CAGR", "valueFormatter": "value.toFixed(2)"},
                    {"headerName": "Trades", "field": "Trades"}
                ],
                "rowData": comparison_data,
                "domLayout": 'autoHeight'
            }).classes("w-full mb-4")
        except Exception as e:
            logger.error(f"Error rendering comparison grid: {str(e)}")
            ui.label(f"Failed to display strategy comparison grid: {str(e)}").classes("text-negative")

        # Prepare equity curves for comparison chart
        comparison_data_serializable = []
        for result in all_results:
            df = process_tradebook(result.get("Tradebook", []))
            if not df.empty:
                strategy_name = result.get("StrategyName", "Unknown")
                comparison_data_serializable.append({
                    "name": strategy_name,
                    "timestamps": [t.isoformat() if isinstance(t, (pd.Timestamp, datetime)) else t for t in df["timestamp"]],
                    "values": df["value"].tolist()
                })

        if comparison_data_serializable:
            # Create the equity curve comparison chart
            fig = go.Figure()

            # Add a baseline for initial capital
            initial_capital = all_results[0].get("InitialInvestment", 100000)
            fig.add_trace(go.Scatter(
                x=[comparison_data_serializable[0]["timestamps"][0], comparison_data_serializable[0]["timestamps"][-1]],
                y=[initial_capital, initial_capital],
                mode="lines",
                name="Initial Capital",
                line=dict(color="#9e9e9e", dash="dash")
            ))

            # Add each strategy's equity curve
            for strategy_data in comparison_data_serializable:
                fig.add_trace(go.Scatter(
                    x=strategy_data["timestamps"],
                    y=strategy_data["values"],
                    mode="lines",
                    name=strategy_data["name"]
                ))

            fig.update_layout(
                title="Strategy Performance Comparison",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                height=500,
                font_color=font_color,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(gridcolor=grid_color),
                yaxis=dict(gridcolor=grid_color)
            )

            try:
                ui.plotly(fig).classes("w-full h-96 mb-4")
            except Exception as e:
                logger.error(f"Error rendering comparison chart: {str(e)}")
                ui.label(f"Failed to display strategy comparison chart: {str(e)}").classes("text-negative")

            # Add drawdown comparison chart
            fig_drawdown = go.Figure()

            for result in all_results:
                df = process_tradebook(result.get("Tradebook", []))
                if not df.empty:
                    strategy_name = result.get("StrategyName", "Unknown")

                    # Calculate drawdown for this strategy
                    rolling_max = df["value"].expanding().max()
                    drawdowns = (df["value"] / rolling_max - 1) * 100

                    fig_drawdown.add_trace(go.Scatter(
                        x=df["timestamp"],
                        y=drawdowns,
                        mode="lines",
                        name=strategy_name,
                        fill="tozeroy",
                        opacity=0.7
                    ))

            fig_drawdown.update_layout(
                title="Drawdown Comparison",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300,
                font_color=font_color,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                xaxis=dict(gridcolor=grid_color),
                yaxis=dict(gridcolor=grid_color)
            )

            ui.plotly(convert_plotly_timestamps(fig_drawdown)).classes("w-full h-64")

            # Add key metrics comparison in visual format
            ui.label("Key Performance Metrics Comparison").classes("text-subtitle1 mt-4")

            # Extract metrics for radar chart
            strategies = [d["Strategy"] for d in comparison_data]
            metrics_to_compare = ["Win Rate", "Sharpe Ratio", "Profit Factor", "CAGR"]

            # Normalize values for radar chart
            radar_data = []
            for metric in metrics_to_compare:
                values = [d[metric] for d in comparison_data]
                max_val = max(values) if values else 1
                normalized = [int(v)/max_val*100 for v in values]
                radar_data.append(normalized)

            # Create radar chart for metrics comparison
            fig_radar = go.Figure()

            for i, strategy in enumerate(strategies):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[radar_data[j][i] for j in range(len(metrics_to_compare))],
                    theta=metrics_to_compare,
                    fill='toself',
                    name=strategy
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color=font_color
            )

            ui.plotly(fig_radar).classes("w-full h-80")
        else:
            ui.label("No data available for strategy comparison.").classes("text-warning")

