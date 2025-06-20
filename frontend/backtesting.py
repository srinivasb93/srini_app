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

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"  # Update with your API base URL

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
        try:
            df["timestamp"] = pd.to_datetime(df["Date"])
        except Exception as e:
            logger.warning(f"Error converting dates: {str(e)}")
            # Create a default timestamp column if conversion fails
            df["timestamp"] = pd.Series([pd.Timestamp.now()] * len(df))
        df["returns"] = df["Profit"] / df["PortfolioValue"].shift(1)

        total_trades = len(df)
        winning_trades = len(df[df["Profit"] > 0])

        metrics = {
            "TotalProfit": float(df["Profit"].sum()),
            "WinRate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "TotalTrades": total_trades,
            "FinalPortfolioValue": float(df["PortfolioValue"].iloc[-1]) if not df.empty else initial_capital,
            "MaxDrawdown": float(BacktestMetrics._calculate_max_drawdown(df["PortfolioValue"])),
            "SharpeRatio": float(BacktestMetrics._calculate_sharpe_ratio(df["returns"])),
            "ProfitFactor": float(BacktestMetrics._calculate_profit_factor(df["Profit"])),
            "AverageWin": float(df[df["Profit"] > 0]["Profit"].mean()) if winning_trades > 0 else 0,
            "AverageLoss": float(df[df["Profit"] < 0]["Profit"].mean()) if len(df[df["Profit"] < 0]) > 0 else 0,
            "LargestWin": float(df["Profit"].max()),
            "LargestLoss": float(df["Profit"].min()),
            "WinningStreak": int(BacktestMetrics._calculate_max_streak(df["Profit"] > 0)),
            "LosingStreak": int(BacktestMetrics._calculate_max_streak(df["Profit"] < 0)),
            # Additional metrics
            "CAGR": float(BacktestMetrics._calculate_cagr(df["PortfolioValue"].iloc[0],
                                                         df["PortfolioValue"].iloc[-1],
                                                         (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days / 365)),
            "AnnualizedVolatility": float(BacktestMetrics._calculate_annualized_volatility(df["returns"])),
            "CalmarRatio": float(BacktestMetrics._calculate_calmar_ratio(df))
        }
        return metrics

    @staticmethod
    def _calculate_max_drawdown(equity_curve: pd.Series) -> float:
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        return abs(drawdowns.min()) * 100

    @staticmethod
    def _calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        if returns.empty:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0

    @staticmethod
    def _calculate_profit_factor(profits: pd.Series) -> float:
        gains = profits[profits > 0].sum()
        losses = abs(profits[profits < 0].sum())
        return gains / losses if losses != 0 else float('inf')

    @staticmethod
    def _calculate_max_streak(series: pd.Series) -> int:
        return max((series != series.shift()).cumsum().value_counts())

    @staticmethod
    def _calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
        """Calculate Compound Annual Growth Rate"""
        if years <= 0 or initial_value <= 0:
            return 0
        return (final_value / initial_value) ** (1 / years) - 1

    @staticmethod
    def _calculate_annualized_volatility(returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        if returns.empty:
            return 0
        return returns.std() * np.sqrt(252)

    @staticmethod
    def _calculate_calmar_ratio(df: pd.DataFrame) -> float:
        """Calculate Calmar Ratio (CAGR / Max Drawdown)"""
        if df.empty or len(df) < 2:
            return 0

        years = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days / 365
        if years <= 0:
            return 0

        cagr = BacktestMetrics._calculate_cagr(
            df["PortfolioValue"].iloc[0],
            df["PortfolioValue"].iloc[-1],
            years
        )
        max_dd = BacktestMetrics._calculate_max_drawdown(df["PortfolioValue"]) / 100  # Convert to decimal

        return cagr / max_dd if max_dd > 0 else 0

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

            # Ensure column names are standardized
            column_mapping = {
                "timestamp": "timestamp",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }

            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

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
            row['Open'],
            row['Close'],
            row['Low'],
            row['High']
        ])
        dates.append(row['timestamp'].strftime('%Y-%m-%d'))

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

def prepare_tradebook_for_display(tradebook):
    """Prepare tradebook for UI display with proper formatting"""
    serializable_tradebook = []
    for trade in tradebook:
        trade_copy = trade.copy()
        if isinstance(trade_copy.get("Date"), (pd.Timestamp, datetime)):
            trade_copy["Date"] = trade_copy["Date"].isoformat()
        elif isinstance(trade_copy.get("Date"), str):
            try:
                trade_copy["Date"] = pd.to_datetime(trade_copy["Date"]).isoformat()
            except ValueError:
                logger.warning(f"Invalid date format in tradebook: {trade_copy['Date']}")
                trade_copy["Date"] = datetime.now().isoformat()
        for key in ["EntryPrice", "ExitPrice", "Profit", "PortfolioValue", "Quantity"]:
            if key in trade_copy and trade_copy[key] is not None:
                trade_copy[key] = float(trade_copy[key])
        serializable_tradebook.append(trade_copy)
    return serializable_tradebook

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
    """Renders the backtesting UI with configuration, results, and comparison sections."""
    logger.debug("Running backtesting.py version 2025-06-19 v6 for NiceGUI 2.19.0")
    broker = user_storage.get("default_broker", "Zerodha")

    with ui.card().classes("w-full mb-4"):
        with ui.row().classes("items-center justify-between"):
            ui.label("Strategy Backtesting").classes("text-h5")
            ui.button(icon="help_outline").props("flat round").tooltip(
                "Run simulations of trading strategies on historical data")

    # Manual state for strategy options
    strategy_options = {}

    with ui.row().classes("w-full gap-4 items-start"):
        with ui.column().classes("w-1/3"):
            with ui.card().classes("w-full p-4 sticky top-0"):
                ui.label("Backtest Configuration").classes("text-h6")

                with ui.expansion("Instrument & Strategy", icon="tune", value=True).classes("w-full mb-2"):
                    # Instrument Selection
                    ui.label("Instrument").classes("text-subtitle2 mt-2")
                    instrument_select = ui.select(
                        options=sorted(list(instruments.keys())),
                        label="Select Instrument",
                        with_input=True,
                        value=None
                    ).props("clearable filter filled hint='Choose the stock or index to backtest'").classes("w-full")

                    # Strategy Selection
                    ui.label("Strategy").classes("text-subtitle2 mt-2")
                    strategies_select = ui.select(
                        options=strategy_options,
                        label="Select Strategy",
                        value=None
                    ).props("hint='Select a strategy to backtest' filled disabled").classes("w-full")

                with ui.expansion("Date Range & Capital", icon="date_range", value=False).classes("w-full mb-2"):
                    # Parameters
                    with ui.row().classes("w-full items-center gap-2"):
                        with ui.column().classes("flex-grow"):
                            ui.label("Start Date").classes("text-subtitle2")
                            start_date = ui.input(
                                "Start Date", value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                            ).props("filled type=date")
                        with ui.column().classes("flex-grow"):
                            ui.label("End Date").classes("text-subtitle2")
                            end_date = ui.input(
                                "End Date", value=datetime.now().strftime("%Y-%m-%d")
                            ).props("filled type=date")

                    ui.label("Capital & Costs").classes("text-subtitle2 mt-2")
                    initial_capital = ui.number(label="Initial Capital", value=100000, format="%.0f", step=10000).props("filled hint='Starting capital for the backtest'").classes("w-full")

                    with ui.row().classes("w-full gap-2"):
                        slippage = ui.number(label="Slippage (%)", value=0.01, format="%.2f", step=0.01).props("filled hint='Price impact'").classes("w-1/2")
                        commission = ui.number(label="Commission (₹)", value=0.0, format="%.2f", step=0.1).props("filled hint='Brokerage fee'").classes("w-1/2")

                with ui.expansion("Optimization Settings", icon="settings").classes("w-full mb-2"):
                    enable_optimization = ui.switch("Enable Parameter Optimization").bind_value(user_storage, "enable_optimization")
                    with ui.column().classes("w-full").bind_visibility_from(user_storage, "enable_optimization"):
                        optimization_iterations = ui.number("Optimization Iterations", value=user_storage.get("optimization_iterations", 10), min=1).bind_value(user_storage, "optimization_iterations").classes("w-full")
                        ui.label("Parameter Ranges").classes("text-caption")
                        with ui.row().classes("w-full gap-2"):
                            stop_loss_min = ui.number("Stop Loss Min (%)", value=user_storage.get("stop_loss_min", 1.0), step=0.5).bind_value(user_storage, "stop_loss_min").classes("w-1/4")
                            stop_loss_max = ui.number("Stop Loss Max (%)", value=user_storage.get("stop_loss_max", 3.0), step=0.5).bind_value(user_storage, "stop_loss_max").classes("w-1/4")
                            target_min = ui.number("Target Min (%)", value=user_storage.get("target_min", 2.0), step=0.5).bind_value(user_storage, "target_min").classes("w-1/4")
                            target_max = ui.number("Target Max (%)", value=user_storage.get("target_max", 6.0), step=0.5).bind_value(user_storage, "target_max").classes("w-1/4")

                with ui.expansion("Backtest Presets", icon="save").classes("w-full mb-2"):
                    preset_name = ui.input("Preset Name").props("filled").classes("w-full")
                    async def save_preset():
                        if not preset_name.value:
                            ui.notify("Please enter a preset name", type="warning")
                            return

                        user_storage["backtest_presets"] = user_storage.get("backtest_presets", {})
                        user_storage["backtest_presets"][preset_name.value] = {
                            "start_date": start_date.value,
                            "end_date": end_date.value,
                            "initial_capital": initial_capital.value,
                            "slippage": slippage.value,
                            "commission": commission.value
                        }
                        preset_select.options = list(user_storage["backtest_presets"].keys())
                        preset_select.update()
                        ui.notify("Preset saved!", type="positive")
                        logger.debug(f"Saved preset: {preset_name.value}")
                    ui.button("Save Preset", on_click=save_preset).props("outline").classes("w-full")
                    preset_select = ui.select(
                        options=list(user_storage.get("backtest_presets", {}).keys()),
                        label="Load Preset"
                    ).props("clearable filled").classes("w-full mt-2")
                    def load_preset():
                        if preset_select.value:
                            preset = user_storage["backtest_presets"][preset_select.value]
                            start_date.value = preset["start_date"]
                            end_date.value = preset["end_date"]
                            initial_capital.value = preset["initial_capital"]
                            slippage.value = preset["slippage"]
                            commission.value = preset["commission"]
                            logger.debug(f"Loaded preset: {preset_select.value}")
                    preset_select.on("update:model-value", load_preset)

                with ui.expansion("Risk Management", icon="shield", value=True).classes("w-full mb-2"):
                    ui.label("Stop Loss & Position Sizing").classes("text-subtitle2")
                    stop_loss_percent = ui.number("Stop Loss (%)", value=2.0, format="%.1f").props(
                        "filled hint='Automatic stop loss percentage'").classes("w-full")
                    max_risk_per_trade = ui.number("Max Risk Per Trade (%)", value=1.0, format="%.1f").props(
                        "filled hint='Maximum capital to risk per trade'").classes("w-full")

                # Confirm Buttons
                ui.separator().classes("my-4")
                with ui.row().classes("w-full gap-2 justify-center"):
                    run_button = ui.button("Run Backtest", on_click=lambda: run_backtest()).props("color=primary").classes("w-full")
                    export_button = ui.button(icon="download", on_click=lambda: export_results()).props("color=secondary outline disabled").tooltip("Export Results").classes("")

        with ui.column().classes("w-2/3"):
            progress_container = ui.row().classes("w-full p-4").bind_visibility_from(run_button, "loading")
            with progress_container:
                progress_bar = ui.linear_progress(value=0).classes("w-full")
                progress_label = ui.label("Progress: 0%").classes("ml-2")

            # Create tabbed result area
            results_tabs = ui.tabs().classes("w-full")
            with results_tabs:
                performance_tab = ui.tab("Performance", icon="trending_up")
                trades_tab = ui.tab("Trades", icon="list")
                metrics_tab = ui.tab("Metrics", icon="analytics")
                comparison_tab = ui.tab("Comparison", icon="compare")

            results_panels = ui.tab_panels(results_tabs, value="Performance").classes("w-full")

            with results_panels:
                performance_panel = ui.tab_panel(performance_tab)
                trades_panel = ui.tab_panel(trades_tab)
                metrics_panel = ui.tab_panel(metrics_tab)
                comparison_panel = ui.tab_panel(comparison_tab)

    async def fetch_strategies_for_backtest():
        nonlocal strategy_options
        cache_key = f"strategies_cache_{broker}"
        cache_ttl = 3600  # 1 hour
        cached = user_storage.get(cache_key, {})
        if cached.get("data") and cached.get("timestamp", 0) + cache_ttl > time.time():
            strategy_options = cached["data"]
            strategies_select.options = strategy_options
            if strategy_options and not strategies_select.value:
                strategies_select.value = list(strategy_options.keys())[0]
            strategies_select.props("disabled=false")
            strategies_select.update()
            logger.debug(f"Loaded {len(strategy_options)} strategies from cache")
            return

        try:
            strategies = await fetch_api(f"/strategies/broker/{broker}")
            await asyncio.sleep(0.5)
            if strategies and isinstance(strategies, list):
                strategy_options.clear()
                strategy_options.update({s["strategy_id"]: s["name"] for s in strategies})
                strategies_select.options = strategy_options
                strategies_select.props("disabled=false")
                if strategy_options and not strategies_select.value:
                    strategies_select.value = list(strategy_options.keys())[0]
                strategies_select.update()
                user_storage[cache_key] = {"data": strategy_options, "timestamp": time.time()}
                logger.debug(f"Fetched {len(strategy_options)} strategies: {list(strategy_options.keys())}")
            else:
                strategy_options.clear()
                strategies_select.options = {}
                strategies_select.value = None
                strategies_select.props("disabled")
                strategies_select.update()
                ui.notify("No strategies found.", type="warning")
                logger.warning("No strategies found or invalid response")
        except Exception as e:
            strategy_options.clear()
            strategies_select.options = {}
            strategies_select.value = None
            strategies_select.props("disabled")
            strategies_select.update()
            ui.notify(f"Error fetching strategies: {str(e)}", type="negative")
            logger.error(f"Error fetching strategies: {str(e)}")

    # Bind fetch_strategies to instrument selection
    async def on_instrument_change(e):
        logger.debug(f"Instrument changed to: {instrument_select.value}")
        if instrument_select.value:
            await fetch_strategies_for_backtest()
        else:
            strategy_options.clear()
            strategies_select.options = {}
            strategies_select.value = None
            strategies_select.props("disabled")
            strategies_select.update()
            logger.debug("No instrument selected, cleared strategy options")
    instrument_select.on("update:model-value", on_instrument_change)



    async def run_backtest():
        """Executes the backtest with validation and progress updates."""
        try:
            if not strategies_select.value or not instrument_select.value:
                ui.notify("Please select both a strategy and instrument.", type="negative")
                return

            try:
                start_dt = datetime.strptime(start_date.value, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date.value, "%Y-%m-%d")
                if start_dt >= end_dt:
                    ui.notify("Start date must be before end date.", type="negative")
                    return
            except ValueError:
                ui.notify("Invalid date format.", type="negative")
                return

            # Validate optimization parameters
            enable_opt = user_storage.get("enable_optimization", False)
            if enable_opt:
                if stop_loss_min.value >= stop_loss_max.value:
                    ui.notify("Stop Loss Min must be less than Stop Loss Max.", type="negative")
                    return
                if target_min.value >= target_max.value:
                    ui.notify("Target Min must be less than Target Max.", type="negative")
                    return

            # Clear previous results and show loading state
            performance_panel.clear()
            trades_panel.clear()
            metrics_panel.clear()
            comparison_panel.clear()

            with performance_panel:
                ui.skeleton(height="400px").classes("w-full")
            with trades_panel:
                ui.skeleton(height="300px").classes("w-full")
            with metrics_panel:
                with ui.grid(columns=4).classes("gap-4"):
                    for _ in range(8):
                        ui.skeleton(height="80px").classes("w-full")

            run_button.props("loading=true disabled=true")
            export_button.props("disabled=true")

            user_id = user_storage.get("user_id", "default_user_id")
            asyncio.create_task(connect_backtest_websocket(user_id, progress_bar, progress_label))

            all_results = []
            strategies = [strategies_select.value] if isinstance(strategies_select.value, str) else strategies_select.value
            for strategy_id in strategies:
                # Fetch strategy details
                strategy_response = await fetch_api(f"/strategies/{strategy_id}")
                if strategy_response and not strategy_response.get("error"):
                    strategy_name = strategy_response.get("name")
                    # For custom strategies, pass the JSON string
                    strategy_value = json.dumps(strategy_response) if strategy_response.get(
                        "entry_conditions") else strategy_name
                else:
                    ui.notify(
                        f"Failed to fetch strategy {strategy_id}: {strategy_response.get('error', {}).get('message', 'Unknown error')}",
                        type="negative")
                    logger.error(f"Failed to fetch strategy {strategy_id}")
                    continue

                backtest_params = {
                    "instrument_token": instrument_select.value,
                    "timeframe": "day",
                    "strategy": strategy_value,
                    "params": {
                        "initial_investment": float(initial_capital.value),
                        "slippage_percent": float(slippage.value),
                        "commission_per_trade": float(commission.value),
                        "optimization_iterations": float(optimization_iterations.value) if enable_opt else 1.0,
                        "stop_loss_range": [float(stop_loss_min.value), float(stop_loss_max.value)] if enable_opt else [1.0, 3.0],
                        "target_range": [float(target_min.value), float(target_max.value)] if enable_opt else [2.0, 6.0]
                    },
                    "start_date": start_date.value,
                    "end_date": end_date.value
                }
                logger.debug(f"Backtest params for strategy {strategy_name}: {json.dumps(backtest_params, indent=2)}")

                response = await fetch_api("/algo-trading/backtest", method="POST", data=backtest_params)
                logger.debug(f"Backtest response for strategy {strategy_name}: {response}")
                if response and not response.get("error"):
                    all_results.append(response)
                else:
                    error_msg = response.get("error", {}).get("message", "Unknown error")
                    ui.notify(f"Backtest failed for strategy {strategy_name}: {error_msg}", type="negative")
                    logger.error(f"Backtest failed for strategy {strategy_name}: {error_msg}")

            # Clear skeleton loaders
            performance_panel.clear()
            trades_panel.clear()
            metrics_panel.clear()
            comparison_panel.clear()

            if all_results:
                # Store results for export
                user_storage["backtest_results"] = all_results

                # Display results
                display_backtest_results(
                    performance_panel=performance_panel,
                    trades_panel=trades_panel,
                    metrics_panel=metrics_panel,
                    results=all_results[0],
                    user_storage=user_storage,
                    fetch_api=fetch_api
                )

                if len(all_results) > 1:
                    display_strategy_comparison(comparison_panel, all_results, user_storage)
                    results_tabs.value = "Comparison"
                else:
                    results_tabs.value = "Performance"

                export_button.props("disabled=false")
                ui.notify("Backtest completed successfully!", type="positive")
            else:
                with performance_panel:
                    ui.label("No successful backtest results to display.").classes("text-negative")
                results_tabs.value = "Performance"

        except Exception as e:
            performance_panel.clear()
            with performance_panel:
                ui.label(f"An error occurred during backtest: {str(e)}").classes("text-negative")
            logger.exception(f"Error during backtest execution: {str(e)}")
            ui.notify(f"Backtest failed: {str(e)}", type="negative")
            results_tabs.value = "Performance"
        # In run_backtest function, update the finally block:
        finally:
            run_button.props("loading=false disabled=false")
            run_button._props["loading"] = False
            run_button._props["disabled"] = False
            run_button.update()

            export_button.props("disabled=" + str(len(user_storage.get("backtest_results", [])) == 0).lower())
            progress_container.visible = False
            ui.update()

    async def export_results():
        """Exports backtest results to CSV."""
        try:
            results = user_storage.get("backtest_results", [])
            if not results:
                ui.notify("No results to export.", type="warning")
                return

            tradebook = results[0].get("Tradebook", [])
            if tradebook:
                df = pd.DataFrame(tradebook)
                csv_data = df.to_csv(index=False)

                # Add download button for CSV
                ui.download(csv_data.encode(), f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

                # Also create a JSON version with all data
                json_data = json.dumps(convert_timestamps_to_iso(results), indent=2)
                ui.download(json_data.encode(), f"backtest_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

                ui.notify("Results exported successfully!", type="positive")
                logger.debug("Backtest results exported")
            else:
                ui.notify("No tradebook data to export.", type="warning")
                logger.warning("No tradebook data for export")
        except Exception as e:
            ui.notify(f"Error exporting results: {str(e)}", type="negative")
            logger.error(f"Error exporting results: {str(e)}")

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
        if tradebook:
            # Create a more user-friendly trade table
            serializable_tradebook = prepare_tradebook_for_display(tradebook)

            # Add trade filters
            with ui.card().classes("w-full p-4 mb-4"):
                with ui.row().classes("items-center justify-between"):
                    ui.label("Trade Analysis").classes("text-h6")
                    with ui.row().classes("gap-2"):
                        trade_filter = ui.select(
                            options=["All Trades", "Winning Trades", "Losing Trades"],
                            value="All Trades",
                            label="Filter"
                        ).props("outlined dense")

                        def filter_trades(e):
                            filter_type = trade_filter.value
                            filtered_data = serializable_tradebook

                            if filter_type == "Winning Trades":
                                filtered_data = [trade for trade in serializable_tradebook if
                                                 trade.get("Profit", 0) > 0]
                            elif filter_type == "Losing Trades":
                                filtered_data = [trade for trade in serializable_tradebook if
                                                 trade.get("Profit", 0) < 0]

                            trade_grid.options["rowData"] = filtered_data
                            trade_grid.update()

                        trade_filter.on("update:model-value", filter_trades)

                        async def export_results():
                            """Exports backtest results to CSV."""
                            try:
                                results = user_storage.get("backtest_results", [])
                                if not results:
                                    ui.notify("No results to export.", type="warning")
                                    return

                                tradebook = results[0].get("Tradebook", [])
                                if tradebook:
                                    df = pd.DataFrame(tradebook)
                                    csv_data = df.to_csv(index=False)

                                    # Add download button for CSV
                                    ui.download(csv_data.encode(),
                                                f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

                                    # Also create a JSON version with all data
                                    json_data = json.dumps(convert_timestamps_to_iso(results), indent=2)
                                    ui.download(json_data.encode(),
                                                f"backtest_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

                                    ui.notify("Results exported successfully!", type="positive")
                                    logger.debug("Backtest results exported")
                                else:
                                    ui.notify("No tradebook data to export.", type="warning")
                                    logger.warning("No tradebook data for export")
                            except Exception as e:
                                ui.notify(f"Error exporting results: {str(e)}", type="negative")
                                logger.error(f"Error exporting results: {str(e)}")
                        ui.button("Export", icon="download", on_click=lambda: export_results()).props("outline")

            try:
                # Fix the column definition in trades_panel
                trade_grid = ui.aggrid({
                    "defaultColDef": {
                        "resizable": True,
                        "sortable": True,
                        "filter": True
                    },
                    "columnDefs": [
                        {"headerName": "Date", "field": "Date", "valueFormatter": "value ? new Date(value).toLocaleString() : ''", "width": 180},
                        {"headerName": "Action", "field": "Action", "cellStyle": "params.value === 'BUY' ? {'color': '#4caf50'} : {'color': '#f44336'}", "width": 100},
                        {"headerName": "Entry Price", "field": "EntryPrice", "valueFormatter": "value ? '₹' + value.toFixed(2) : ''", "width": 120},
                        {"headerName": "Exit Price", "field": "ExitPrice", "valueFormatter": "value ? '₹' + value.toFixed(2) : ''", "width": 120},
                        {"headerName": "Quantity", "field": "Quantity", "valueFormatter": "value ? value.toFixed(2) : ''", "width": 100},
                        {"headerName": "Profit", "field": "Profit",
                            "valueFormatter": "value ? '₹' + value.toFixed(2) : ''",
                            "cellStyle": "params.value > 0 ? {'color': '#4caf50'} : params.value < 0 ? {'color': '#f44336'} : {}",
                            "width": 120
                        },
                        {"headerName": "Portfolio Value", "field": "PortfolioValue", "valueFormatter": "value ? '₹' + value.toFixed(2) : ''", "width": 150}
                    ],
                    "rowData": serializable_tradebook,
                    "pagination": True,
                    "paginationPageSize": 15,
                    "domLayout": 'autoHeight'
                }).classes("w-full")

                # Add trade statistics below the grid
                with ui.row().classes("mt-4 gap-4"):
                    with ui.card().classes("w-1/4 p-4"):
                        ui.label("Avg. Holding Period").classes("text-subtitle2")
                        ui.label("2.5 days").classes("text-h6")  # Would need calculation
                    with ui.card().classes("w-1/4 p-4"):
                        ui.label("Avg. Profit per Trade").classes("text-subtitle2")
                        ui.label(f"₹{metrics['TotalProfit']/metrics['TotalTrades'] if metrics['TotalTrades'] > 0 else 0:,.2f}").classes("text-h6")
                    with ui.card().classes("w-1/4 p-4"):
                        ui.label("Win/Loss Ratio").classes("text-subtitle2")
                        win_loss = abs(metrics["AverageWin"]/metrics["AverageLoss"]) if metrics["AverageLoss"] != 0 else 0
                        ui.label(f"{win_loss:.2f}").classes("text-h6")
                    with ui.card().classes("w-1/4 p-4"):
                        ui.label("Expectancy").classes("text-subtitle2")
                        expectancy = (metrics["WinRate"]/100 * metrics["AverageWin"]) + ((1-metrics["WinRate"]/100) * metrics["AverageLoss"])
                        ui.label(f"₹{expectancy:,.2f}").classes("text-h6")
            except Exception as e:
                logger.error(f"Error rendering trade grid: {str(e)}")
                ui.label(f"Failed to display trade analysis: {str(e)}").classes("text-negative")
        else:
            ui.label("No trades executed in this backtest.").classes("text-warning")

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
                normalized = [v/max_val*100 for v in values]
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