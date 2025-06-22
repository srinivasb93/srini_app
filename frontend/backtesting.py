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
            "unit": "days" if "day" in interval else "minutes"
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


def create_performance_charts(df, metrics, theme_mode="Dark"):
    """Create performance charts from processed tradebook data"""
    if df.empty:
        return None

    font_color = "white" if theme_mode == "Dark" else "black"
    grid_color = "rgba(128, 128, 128, 0.5)" if theme_mode == "Dark" else "rgba(200, 200, 200, 0.5)"

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=("Portfolio Value", "Drawdown", "Monthly Returns")
    )

    # Equity curve
    x_values = [t.isoformat() if isinstance(t, (pd.Timestamp, datetime)) else t for t in df["timestamp"]]

    fig.add_trace(go.Scatter(
        x=x_values,
        y=df["value"],
        mode="lines",
        name="Portfolio Value",
        line=dict(color="#2196f3")
    ), row=1, col=1)

    # Add baseline for initial capital
    initial_capital = df["value"].iloc[0] if not df.empty else 100000
    fig.add_trace(go.Scatter(
        x=[df["timestamp"].min(), df["timestamp"].max()],
        y=[initial_capital, initial_capital],
        mode="lines",
        name="Initial Capital",
        line=dict(color="#9e9e9e", dash="dash")
    ), row=1, col=1)

    # Buy/sell markers with hover info
    for idx, trade in df.iterrows():
        action = trade["Action"]
        if action:
            marker_color = "#4caf50" if action == "BUY" else "#f44336"
            fig.add_trace(go.Scatter(
                x=[trade["timestamp"]],
                y=[trade["value"]],
                mode="markers",
                name=action,
                marker=dict(color=marker_color, size=10, symbol="triangle-up" if action == "BUY" else "triangle-down"),
                hovertemplate=f"{action}<br>Date: %{{x}}<br>Price: {trade.get('EntryPrice', 0):.2f}<br>Quantity: {trade.get('Quantity', 0):.0f}<extra></extra>",
                showlegend=False
            ), row=1, col=1)

    # Drawdown chart
    rolling_max = df["value"].expanding().max()
    drawdowns = (df["value"] / rolling_max - 1) * 100
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=drawdowns,
        mode="lines",
        name="Drawdown",
        line=dict(color="#ff9800"),
        fill="tozeroy",
        hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>"
    ), row=2, col=1)

    # Add max drawdown line
    max_dd = drawdowns.min()
    fig.add_trace(go.Scatter(
        x=[df["timestamp"].min(), df["timestamp"].max()],
        y=[max_dd, max_dd],
        mode="lines",
        name="Max Drawdown",
        line=dict(color="#f44336", dash="dash"),
        hovertemplate=f"Max Drawdown: {max_dd:.2f}%<extra></extra>"
    ), row=2, col=1)

    # Monthly returns
    monthly_returns = df.groupby(df["timestamp"].dt.to_period("M"))["Profit"].sum()
    monthly_colors = ["#4caf50" if ret > 0 else "#f44336" for ret in monthly_returns.values]

    fig.add_trace(go.Bar(
        x=monthly_returns.index.to_timestamp(),
        y=monthly_returns.values,
        name="Monthly Returns",
        marker_color=monthly_colors,
        hovertemplate="Month: %{x}<br>Return: ₹%{y:.2f}<extra></extra>"
    ), row=3, col=1)

    # Add annotations for key metrics
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"Total Profit: ₹{metrics['TotalProfit']:.2f}<br>Win Rate: {metrics['WinRate']:.1f}%<br>Sharpe: {metrics['SharpeRatio']:.2f}",
        showarrow=False,
        font=dict(color=font_color),
        align="left",
        bgcolor="rgba(0,0,0,0.5)" if theme_mode == "Dark" else "rgba(255,255,255,0.5)",
        bordercolor="#9e9e9e",
        borderwidth=1,
        borderpad=4
    )

    fig.update_layout(
        height=1000,
        title="Backtest Performance Analysis",
        showlegend=True,
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
        )
    )

    for row in range(1, 4):
        fig.update_xaxes(gridcolor=grid_color, row=row, col=1)
        fig.update_yaxes(gridcolor=grid_color, row=row, col=1)
    fig.update_yaxes(title="Portfolio Value (₹)", row=1, col=1)
    fig.update_yaxes(title="Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title="Returns (₹)", row=3, col=1)

    return fig


async def render_backtesting_page(fetch_api, user_storage, instruments):
    """Renders the backtesting UI with a modern two-column layout."""
    broker = user_storage.get("default_broker", "Zerodha")
    strategy_options = {}
    partial_exit_rows = []

    with ui.splitter(value=30).classes("w-full h-screen") as splitter:
        with splitter.before:
            with ui.card().classes("w-full h-full p-4 overflow-auto"):
                ui.label("Backtest Configuration").classes("text-h6 mb-4")
                # ... Rest of the configuration UI elements from the original file ...
                with ui.expansion("Instrument & Strategy", icon="tune", value=True):
                    instrument_select = ui.select(options=sorted(list(instruments.keys())), label="Select Instrument", with_input=True).props("clearable dense").classes("w-full")
                    strategies_select = ui.select(options=strategy_options, label="Select Strategy").props("dense disabled").classes("w-full")

                with ui.expansion("Date Range & Capital", icon="date_range", value=True):
                    with ui.row():
                        start_date = ui.input("Start Date", value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")).props("dense type=date").classes("flex-1")
                        end_date = ui.input("End Date", value=datetime.now().strftime("%Y-%m-%d")).props("dense type=date").classes("flex-1")
                    initial_capital = ui.number(label="Initial Capital", value=100000, format="%.0f").props("dense").classes("w-full")

                with ui.expansion("Risk Management", icon="shield", value=True):
                    stop_loss_percent = ui.number("Stop Loss (%)", value=2.0, format="%.1f", min=0).props("dense")
                    trailing_stop_loss_percent = ui.number("Trailing Stop (%)", value=1.5, format="%.1f", min=0).props("dense")
                    position_sizing_percent = ui.number("Position Sizing (% of Capital)", value=10.0, format="%.1f").props("dense")
                    ui.separator().classes("my-2")
                    ui.label("Partial Exits").classes("text-subtitle2")
                    partial_exits_container = ui.column().classes("w-full gap-1")

                    def add_partial_exit_row():
                        row_id = str(uuid4())
                        with partial_exits_container:
                            with ui.row().classes("w-full items-center gap-2") as pe_row:
                                target = ui.number("Target (%)", value=5.0, min=0.1).props("dense").classes("flex-grow")
                                qty_percent = ui.number("Qty (%)", value=50.0, min=1, max=100).props("dense").classes("flex-grow")
                                ui.button(icon="delete", on_click=lambda: remove_partial_exit(row_id)).props("flat round dense color=negative text-xs")
                        partial_exit_rows.append({"id": row_id, "row": pe_row, "target": target, "qty_percent": qty_percent})

                    def remove_partial_exit(row_id):
                        row_to_remove = next((r for r in partial_exit_rows if r["id"] == row_id), None)
                        if row_to_remove:
                            partial_exits_container.remove(row_to_remove["row"])
                            partial_exit_rows.remove(row_to_remove)

                    ui.button("Add Partial Exit", icon="add", on_click=add_partial_exit_row).props("outline size=sm").classes("w-full mt-1")

                with ui.expansion("Optimization Settings", icon="settings"):
                    enable_optimization = ui.switch("Enable Parameter Optimization")
                    with ui.column().classes("w-full").bind_visibility_from(enable_optimization, 'value'):
                        optimization_iterations = ui.number("Iterations", value=20, min=5, max=200).props("dense")
                        with ui.row():
                            stop_loss_min = ui.number("SL Min (%)", value=1.0, min=0.1, format="%.1f").props("dense")
                            stop_loss_max = ui.number("SL Max (%)", value=5.0, min=0.1, format="%.1f").props("dense")

                run_button = ui.button("Run Backtest", on_click=lambda: run_backtest()).props("color=primary icon=play_arrow").classes("w-full mt-4")

        with splitter.after:
            with ui.card().classes("w-full h-full p-2"):
                results_tabs = ui.tabs().classes("w-full")
                with results_tabs:
                    performance_tab = ui.tab("Performance", icon="trending_up")
                    trades_tab = ui.tab("Trades", icon="list_alt")
                    metrics_tab = ui.tab("Metrics", icon="analytics")
                    optimization_tab = ui.tab("Optimization", icon="auto_awesome")

                with ui.tab_panels(results_tabs, value=performance_tab).classes("w-full mt-2 h-full overflow-auto"):
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
            custom_strategies_list = await fetch_api(f"/strategies/broker/{broker}")
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
                "trailing_stop_loss_percent": float(trailing_stop_loss_percent.value),
                "position_sizing_percent": float(position_sizing_percent.value),
                "enable_optimization": enable_optimization.value,
                "partial_exits": [{"target": r["target"].value, "qty_percent": r["qty_percent"].value} for r in
                                  partial_exit_rows]
            }
            if enable_optimization.value:
                if stop_loss_min.value >= stop_loss_max.value:
                    ui.notify("Stop Loss Min must be less than Max.", type="negative")
                    return
                params["optimization_iterations"] = int(optimization_iterations.value)
                params["stop_loss_range"] = [float(stop_loss_min.value), float(stop_loss_max.value)]

            strategy_id = strategies_select.value
            strategy_value = strategy_id
            if strategy_id not in PREDEFINED_STRATEGIES:
                strategy_response = await fetch_api(f"/strategies/{strategy_id}")
                if not (strategy_response and not strategy_response.get("error")):
                    ui.notify("Failed to fetch custom strategy definition.", type="negative");
                    return
                strategy_value = json.dumps(strategy_response)

            backtest_payload = {
                "instrument_token": instrument_select.value, "timeframe": "day", "strategy": strategy_value,
                "params": params, "start_date": start_date.value, "end_date": end_date.value
            }

            response = await fetch_api("/algo-trading/backtest", method="POST", data=backtest_payload)
            logger.debug(f"ALGO RESPONSE: {json.dumps(response)}")
            for panel in [performance_panel, trades_panel, metrics_panel, optimization_panel]: panel.clear()

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


def display_backtest_results(performance_panel, trades_panel, metrics_panel, results, user_storage, fetch_api):
    """Displays backtest results with metrics and charts in a tabbed interface."""
    theme_mode = "Dark" if user_storage.get("dark_mode", True) else "Light"
    tradebook = results.get("Tradebook", [])

    # Process tradebook once for all visualizations
    df = process_tradebook(tradebook)

    # Calculate metrics from processed data
    tradebook_json = json.dumps(tradebook)
    metrics = BacktestMetrics.calculate_metrics(tradebook_json, results.get("InitialInvestment", 100000))

    # Store data for export
    user_storage["backtest_tradebook"] = convert_timestamps_to_iso(tradebook)

    # 1. Metrics Panel - Show key metrics
    with metrics_panel:
        with ui.card().classes("w-full p-4 mb-4"):
            ui.label(f"Strategy: {results.get('StrategyName', 'Unknown Strategy')}").classes("text-h6")
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

    # 2. Performance Panel - Show charts
    with performance_panel:
        if not df.empty:
            # Create performance charts
            fig = create_performance_charts(df, metrics, theme_mode)
            ui.plotly(convert_plotly_timestamps(fig)).classes("w-full h-128")

            # Add candlestick chart
            with ui.card().classes("w-full p-4 my-4"):
                with ui.row().classes("items-center justify-between"):
                    ui.label("Price Chart with Signals").classes("text-h6")
                    ui.button("Refresh", icon="refresh", on_click=lambda: ui.notify("Chart refreshed")).props("outline")

                # Fetch OHLC data if not available in results
                ohlc_data = results.get("OHLC", pd.DataFrame())

                if not isinstance(ohlc_data, pd.DataFrame) and isinstance(ohlc_data, list):
                    ohlc_data = pd.DataFrame(ohlc_data)

                # If no OHLC data in results, fetch it
                if ohlc_data.empty:
                    instrument = results.get("Instrument", "")
                    from_date = results.get("StartDate", "")
                    to_date = results.get("EndDate", "")

                    # Show loading indicator
                    chart_loading = ui.spinner("dots", size="lg").classes("self-center")

                    # Create a background task to fetch data
                    async def fetch_and_update_chart():
                        nonlocal ohlc_data
                        if instrument and from_date and to_date:
                            # Fetch data from API
                            ohlc_data = await fetch_ohlc_data(
                                fetch_api,
                                instrument,
                                from_date,
                                to_date,
                                interval=results.get("Timeframe", "1day")
                            )

                            # Remove loading indicator
                            chart_loading.delete()

                            if not ohlc_data.empty:
                                # Create and display candlestick chart
                                candlestick_options = create_candlestick_chart(ohlc_data, tradebook, theme_mode)
                                ui.echart(candlestick_options).classes("w-full h-96")
                            else:
                                ui.label("Could not retrieve price chart data.").classes("text-subtitle2")
                        else:
                            chart_loading.delete()
                            ui.label("Missing instrument or date information for chart.").classes("text-subtitle2")

                    # Start background task
                    ui.timer(0, fetch_and_update_chart, once=True)
                else:
                    # Use existing OHLC data
                    candlestick_options = create_candlestick_chart(ohlc_data, tradebook, theme_mode)
                    ui.echart(candlestick_options).classes("w-full h-96")

            # Update the Performance Summary section to prevent overlap:
            with ui.card().classes("w-full p-4 mb-4"):
                with ui.row().classes("items-center justify-between"):
                    ui.label("Performance Summary").classes("text-h6")
                    ui.button("Refresh Chart", icon="refresh", on_click=lambda: ui.notify("Charts refreshed")).props("outline")

                ui.separator().classes("my-2")

                # Fix the layout with consistent column widths
                with ui.grid(columns=3).classes("gap-4 w-full"):
                    # Quick summary metrics
                    with ui.card().classes("p-3"):
                        ui.label("Returns").classes("text-subtitle2")
                        color = "text-positive" if metrics["TotalProfit"] > 0 else "text-negative"
                        ui.label(f"₹{metrics['TotalProfit']:,.2f} ({metrics['TotalProfit']/results.get('InitialInvestment', 100000)*100:.1f}%)").classes(f"{color} text-h6")

                    with ui.card().classes("p-3"):
                        ui.label("Winning Trades").classes("text-subtitle2")
                        ui.label(f"{metrics['WinRate']:.1f}% ({int(metrics['TotalTrades'] * metrics['WinRate']/100)}/{metrics['TotalTrades']})").classes("text-h6")

                    with ui.card().classes("p-3"):
                        ui.label("Risk-Adjusted Return").classes("text-subtitle2")
                        color = "text-positive" if metrics["SharpeRatio"] > 1 else "text-negative"
                        ui.label(f"Sharpe: {metrics['SharpeRatio']:.2f}").classes(f"{color} text-h6")
        else:
            ui.label("No trades executed in this backtest.").classes("text-warning")

    # 3. Trades Panel - Show detailed trade list
    with trades_panel:
        completed_trades = process_tradebook_for_display(tradebook)
        if completed_trades:
            ui.aggrid({
                "columnDefs": [
                    {"headerName": "Entry Date", "field": "EntryDate",
                     "valueFormatter": "value ? new Date(value).toLocaleString() : ''"},
                    {"headerName": "Exit Date", "field": "ExitDate",
                     "valueFormatter": "value ? new Date(value).toLocaleString() : ''"},
                    {"headerName": "Entry Price", "field": "EntryPrice",
                     "valueFormatter": "'₹' + params.value.toFixed(2)"},
                    {"headerName": "Exit Price", "field": "ExitPrice",
                     "valueFormatter": "params.value ? '₹' + params.value.toFixed(2) : 'N/A'"},
                    {"headerName": "PNL", "field": "PNL", "valueFormatter": "'₹' + params.value.toFixed(2)",
                     "cellStyle": "params.value >= 0 ? {'color': '#4caf50'} : {'color': '#f44336'}"},
                    {"headerName": "Exit Reason", "field": "ExitReason"},
                    {"headerName": "Holding Period", "field": "HoldingPeriod"},
                ],
                "rowData": completed_trades, "domLayout": 'autoHeight'
            }).classes("w-full")

            # Display metrics *below* the table
            with ui.card().classes("w-full mt-4 p-4"):
                ui.label("Trade Statistics").classes("text-h6 mb-2")
                with ui.grid(columns=4).classes("gap-4"):
                    avg_holding_period = pd.to_timedelta(
                        [t['HoldingPeriod'] for t in completed_trades if t['HoldingPeriod'] != "N/A"]).mean()
                    display_metric("Avg. Holding Period", str(avg_holding_period).split('.')[0], format_spec="s",
                                   help_text="Average time a position is held.")
                    display_metric("Avg. Profit per Trade",
                                   metrics['TotalProfit'] / metrics['TotalTrades'] if metrics['TotalTrades'] > 0 else 0,
                                   help_text="Average PNL across all trades.")
                    win_loss_ratio = abs(metrics["AverageWin"] / metrics["AverageLoss"]) if metrics[
                                                                                                "AverageLoss"] != 0 else 0
                    display_metric("Win/Loss Ratio", win_loss_ratio,
                                   help_text="Average win amount divided by average loss amount.")
                    expectancy = (metrics["WinRate"] / 100 * metrics["AverageWin"]) + (
                                (1 - metrics["WinRate"] / 100) * abs(metrics["AverageLoss"]))
                    display_metric("Expectancy (₹)", expectancy, help_text="Expected PNL per trade.")

            with ui.scroll_area().classes('w-full h-full border p-2 rounded-lg'):
                ui.label("Trade Execution Log").classes("text-h6 p-2")
                for trade in tradebook:
                    # Each trade event gets its own card
                    with ui.card().classes('w-full mb-2 p-3'):
                        # Display primary trade info in a row
                        with ui.row().classes('w-full items-center justify-between text-sm'):
                            action_color = "text-positive" if trade['Action'] == "BUY" else "text-negative"
                            ui.label(f"{trade['Action']}").classes(f'text-md font-bold {action_color}')
                            ui.label(f"On: {trade['Date']}")
                            ui.label(f"Price: ₹{trade['Price']:.2f}")
                            ui.label(f"Reason: {trade['Reason']}")
                            ui.label(f"PnL: ₹{trade['Profit']:.2f}").classes('font-mono')

                        # Check for and display indicators in an expandable section
                        indicators = trade.get("Indicators")
                        if indicators and isinstance(indicators, dict) and any(indicators.values()):
                            with ui.expansion('View Indicators', icon='insights').classes('w-full text-xs mt-2'):
                                with ui.grid(columns=3).classes('w-full gap-2 p-2'):
                                    for key, value in indicators.items():
                                        with ui.card().classes('items-center p-1 bg-blue-grey-1 dark:bg-blue-grey-8'):
                                            ui.label(key).classes('text-xs text-gray-500')
                                            ui.label(f"{value}").classes('font-bold')
        else:
            ui.label("No completed trades in this backtest.").classes("text-warning")


def display_optimization_runs(container, all_runs):
    with container:
        if not all_runs:
            ui.label("No optimization data available.").classes("absolute-center text-gray-500")
            return

        ui.label("Optimization Run Summary").classes("text-h6 mb-2")
        rowData = [{"sl_percent": r['parameters'].get('stop_loss_percent', 0),
                    "pnl": r.get('TotalPNL', 0), "win_rate": r.get('WinRate', 0),
                    "trades": r.get('TotalTrades', 0)} for r in all_runs]
        ui.aggrid({
            "columnDefs": [
                {"headerName": "Stop Loss (%)", "field": "sl_percent", "valueFormatter": "params.value.toFixed(2)"},
                {"headerName": "Total PNL (₹)", "field": "pnl", "valueFormatter": "params.value.toFixed(2)", "sort": "desc"},
                {"headerName": "Win Rate (%)", "field": "win_rate", "valueFormatter": "params.value.toFixed(2)"},
                {"headerName": "Total Trades", "field": "trades"},
            ],
            "rowData": rowData, "domLayout": 'autoHeight'
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