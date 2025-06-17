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
from typing import Dict, List, Optional
import plotly.subplots as make_subplots
import aiohttp

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"  # Update with your API base URL

class BacktestMetrics:
    @staticmethod
    def calculate_metrics(tradebook: pd.DataFrame, initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics from tradebook."""
        if tradebook.empty:
            return {}

        df = pd.DataFrame(tradebook)
        df["timestamp"] = pd.to_datetime(df["Date"])
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
            "LosingStreak": int(BacktestMetrics._calculate_max_streak(df["Profit"] < 0))
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
        excess_returns = returns - risk_free_rate/252  # Assuming daily returns
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0

    @staticmethod
    def _calculate_profit_factor(profits: pd.Series) -> float:
        gains = profits[profits > 0].sum()
        losses = abs(profits[profits < 0].sum())
        return gains / losses if losses != 0 else float('inf')

    @staticmethod
    def _calculate_max_streak(series: pd.Series) -> int:
        return max((series != series.shift()).cumsum().value_counts())

async def connect_backtest_websocket(user_id: str, progress_bar, progress_label):
    """Connect to WebSocket for backtest progress updates."""
    ws_url = f"{BASE_URL.replace('http', 'ws')}/ws/backtest/{user_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            progress = data.get("progress", 0)
                            progress_bar.value = progress
                            progress_label.text = f"Progress: {progress*100:.1f}%"
                            if progress >= 1:
                                break
                        except json.JSONDecodeError:
                            logger.error(f"Invalid WebSocket message: {msg.data}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
    except Exception as e:
        logger.error(f"WebSocket connection failed: {str(e)}")
        ui.notify("Failed to connect to progress updates.", type="negative")

async def render_backtesting_page(fetch_api, user_storage, instruments):
    broker = user_storage.get("default_broker", "Zerodha")
    ui.label("Strategy Backtesting").classes("text-h5 q-pa-md")

    with ui.row().classes("w-full gap-4 items-start"):
        with ui.card().classes("w-1/3 p-4"):
            ui.label("Backtest Configuration").classes("text-h6")
            with ui.stepper().props("vertical").classes("w-full") as stepper:
                with ui.step("Instrument"):
                    instrument_select = ui.select(
                        options=sorted(list(instruments.keys())),
                        label="Select Instrument",
                        with_input=True
                    ).props("clearable filter hint='Choose the stock or index to backtest'").classes("w-full")
                    ui.stepper_navigation()

                with ui.step("Strategies"):
                    strategies_select = ui.select(options={}, label="Select Strategy", multiple=True).props("hint='Select one or more strategies to compare'").classes("w-full")
                    async def fetch_strategies_for_backtest():
                        strategies = await fetch_api(f"/strategies/{broker}")
                        if strategies and isinstance(strategies, list):
                            strategy_options = {s["strategy_id"]: s["name"] for s in strategies}
                            strategies_select.options = strategy_options
                            strategies_select.update()
                        else:
                            ui.notify("Could not fetch strategies.", type="warning")
                    await fetch_strategies_for_backtest()
                    ui.stepper_navigation()

                with ui.step("Parameters"):
                    with ui.row().classes("w-full items-center gap-2"):
                        with ui.column().classes("flex-grow"):
                            ui.label("Start Date")
                            start_date = ui.date(value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")).props("hint='Start date for historical data'").classes("w-full")
                        with ui.column().classes("flex-grow"):
                            ui.label("End Date")
                            end_date = ui.date(value=datetime.now().strftime("%Y-%m-%d")).props("hint='End date for historical data'").classes("w-full")
                    initial_capital = ui.number(label="Initial Capital", value=100000, format="%.0f", step=10000).props("hint='Starting capital for the backtest'").classes("w-full")
                    slippage = ui.number(label="Slippage (% per trade)", value=0.01, format="%.2f", step=0.01).props("hint='Percentage price impact per trade'").classes("w-full")
                    commission = ui.number(label="Commission (₹ per trade)", value=0.0, format="%.2f", step=0.1).props("hint='Brokerage fee per trade'").classes("w-full")
                    with ui.expansion("Optimization Settings", icon="settings").classes("w-full"):
                        enable_optimization = ui.switch("Enable Parameter Optimization").bind_value(user_storage, "enable_optimization")
                        if user_storage.get("enable_optimization", False):
                            optimization_iterations = ui.number("Optimization Iterations", value=10, min=1).bind_value(user_storage, "optimization_iterations")
                            ui.label("Parameter Ranges").classes("text-caption")
                            with ui.row().classes("w-full gap-2"):
                                stop_loss_min = ui.number("Stop Loss Min (%)", value=1.0, step=0.5).classes("w-1/3")
                                stop_loss_max = ui.number("Stop Loss Max (%)", value=3.0, step=0.5).classes("w-1/3")
                                target_min = ui.number("Target Min (%)", value=2.0, step=0.5).classes("w-1/3")
                                target_max = ui.number("Target Max (%)", value=6.0, step=0.5).classes("w-1/3")
                    ui.stepper_navigation()

                with ui.step("Confirm"):
                    run_button = ui.button("Run Backtest", on_click=lambda: run_backtest()).props("color=primary").classes("w-full mt-4")
                    export_button = ui.button("Export Results", on_click=lambda: export_results()).props("color=secondary disabled").classes("w-full mt-2")

    progress_container = ui.row().classes("w-full p-4").bind_visibility_from(run_button, "loading")
    with progress_container:
        progress_bar = ui.linear_progress(value=0).classes("w-full")
        progress_label = ui.label("Progress: 0%").classes("ml-2")

    results_container = ui.column().classes("w-full gap-4")
    comparison_container = ui.column().classes("w-full gap-4")

    async def run_backtest():
        if not strategies_select.value or not instrument_select.value:
            ui.notify("Please select at least one strategy and instrument.", type="negative")
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

        results_container.clear()
        comparison_container.clear()

        with results_container:
            with ui.row().classes("items-center"):
                ui.spinner(size="lg")
                ui.label("Preparing backtest...")

        run_button.props("loading=true disabled=true")
        export_button.props("disabled=true")

        try:
            user_id = user_storage.get("user_id", "default_user_id")
            asyncio.create_task(connect_backtest_websocket(user_id, progress_bar, progress_label))

            all_results = []
            for strategy_id in (strategies_select.value if isinstance(strategies_select.value, list) else [strategies_select.value]):
                backtest_params = {
                    "instrument_token": instruments[instrument_select.value],
                    "timeframe": "day",
                    "strategy": strategy_id,
                    "params": {
                        "initial_investment": float(initial_capital.value),
                        "slippage_percent": float(slippage.value),
                        "commission_per_trade": float(commission.value),
                        "optimization_iterations": optimization_iterations.value if user_storage.get("enable_optimization", False) else 1,
                        "stop_loss_range": [stop_loss_min.value, stop_loss_max.value] if user_storage.get("enable_optimization", False) else None,
                        "target_range": [target_min.value, target_max.value] if user_storage.get("enable_optimization", False) else None
                    },
                    "start_date": start_date.value,
                    "end_date": end_date.value,
                    "enable_optimization": user_storage.get("enable_optimization", False)
                }

                response = await fetch_api("/algo-trading/backtest", method="POST", data=backtest_params)
                if response and not response.get("error"):
                    all_results.append(response)
                else:
                    ui.notify(f"Backtest failed for strategy {strategy_id}: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")

            results_container.clear()
            if all_results:
                display_backtest_results(results_container, all_results[0], user_storage)
                if len(all_results) > 1:
                    display_strategy_comparison(comparison_container, all_results, user_storage)
                export_button.props("disabled=false")
                ui.notify("Backtest completed successfully!", type="positive")
            else:
                with results_container:
                    ui.label("No successful backtest results to display.").classes("text-negative")

        except Exception as e:
            results_container.clear()
            with results_container:
                ui.label(f"An error occurred during backtest: {str(e)}").classes("text-negative")
            logger.exception("Error during backtest execution")
            ui.notify(f"Backtest failed: {str(e)}", type="negative")
        finally:
            run_button.props("loading=false disabled=false")
            progress_container.visible = False

    async def export_results():
        try:
            tradebook = results_container.storage.get("tradebook", [])
            if tradebook:
                df = pd.DataFrame(tradebook)
                csv_data = df.to_csv(index=False)
                ui.download(csv_data.encode(), f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                ui.notify("Results exported successfully!", type="positive")
            else:
                ui.notify("No results to export.", type="negative")
        except Exception as e:
            ui.notify(f"Error exporting results: {str(e)}", type="negative")
            logger.error(f"Export error: {str(e)}")

def display_backtest_results(container, results, user_storage):
    with container:
        metrics = BacktestMetrics.calculate_metrics(pd.DataFrame(results.get("Tradebook", [])), results.get("InitialCapital", 100000))
        container.storage["tradebook"] = results.get("Tradebook", [])

        with ui.grid(columns=4).classes("gap-4 mb-4"):
            def display_metric(label, value, format_spec=",.2f"):
                with ui.card().classes("items-center p-4"):
                    ui.label(label).classes("text-caption")
                    formatted_value = f"{float(value):{format_spec}}" if isinstance(value, (int, float)) else str(value)
                    ui.label(formatted_value).classes("text-subtitle1 font-bold")

            display_metric("Total Profit (₹)", metrics["TotalProfit"])
            display_metric("Win Rate (%)", metrics["WinRate"])
            display_metric("Total Trades", metrics["TotalTrades"], ",.0f")
            display_metric("Final Portfolio Value (₹)", metrics["FinalPortfolioValue"])
            display_metric("Max Drawdown (%)", metrics["MaxDrawdown"])
            display_metric("Sharpe Ratio", metrics["SharpeRatio"])
            display_metric("Profit Factor", metrics["ProfitFactor"])
            display_metric("Average Win (₹)", metrics["AverageWin"])

        ui.separator()
        ui.label("Performance Charts").classes("text-subtitle1 mt-4")

        tradebook = results.get("Tradebook", [])
        if tradebook:
            df = pd.DataFrame(tradebook)
            df["timestamp"] = pd.to_datetime(df["Date"])
            df["value"] = pd.to_numeric(df["PortfolioValue"])
            df = df.sort_values("timestamp")

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.3, 0.2],
                subplot_titles=("Portfolio Value", "Drawdown", "Monthly Returns")
            )

            # Equity curve
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#2196f3")
            ), row=1, col=1)

            # Buy/sell markers
            for idx, trade in df.iterrows():
                if "Action" in trade:
                    marker_color = "#4caf50" if trade["Action"] == "BUY" else "#f44336"
                    fig.add_trace(go.Scatter(
                        x=[trade["timestamp"]],
                        y=[trade["value"]],
                        mode="markers",
                        name=trade["Action"],
                        marker=dict(color=marker_color, size=10, symbol="triangle-up" if trade["Action"] == "BUY" else "triangle-down"),
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
                fill="tonexty"
            ), row=2, col=1)

            # Monthly returns
            monthly_returns = df.groupby(df["timestamp"].dt.to_period("M"))["Profit"].sum()
            fig.add_trace(go.Bar(
                x=monthly_returns.index.to_timestamp(),
                y=monthly_returns.values,
                name="Monthly Returns",
                marker_color="#3f51b5"
            ), row=3, col=1)

            theme_mode = user_storage.get("app_theme", "Dark")
            font_color = "white" if theme_mode == "Dark" else "black"

            fig.update_layout(
                height=1000,
                title="Backtest Performance Analysis",
                showlegend=True,
                font_color=font_color,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )

            for row in range(1, 4):
                fig.update_xaxes(gridcolor="rgba(128, 128, 128, 0.2)", row=row, col=1)
                fig.update_yaxes(gridcolor="rgba(128, 128, 128, 0.2)", row=row, col=1)
            fig.update_yaxes(title="Drawdown (%)", row=2, col=1)
            fig.update_yaxes(title="Returns (₹)", row=3, col=1)

            ui.plotly(fig).classes("w-full h-128")

            ui.label("Trade Analysis").classes("text-subtitle1 mt-4")
            trade_grid = ui.aggrid({
                "columnDefs": [
                    {"headerName": "Date", "field": "Date", "valueFormatter": "value ? new Date(value).toLocaleString() : ''"},
                    {"headerName": "Action", "field": "Action"},
                    {"headerName": "Entry Price", "field": "EntryPrice", "valueFormatter": "value ? value.toFixed(2) : ''"},
                    {"headerName": "Exit Price", "field": "ExitPrice", "valueFormatter": "value ? value.toFixed(2) : ''"},
                    {"headerName": "Quantity", "field": "Quantity"},
                    {"headerName": "Profit", "field": "Profit", "valueFormatter": "value ? value.toFixed(2) : ''"},
                    {"headerName": "Portfolio Value", "field": "PortfolioValue", "valueFormatter": "value ? value.toFixed(2) : ''"}
                ],
                "rowData": tradebook,
                "pagination": True,
                "paginationPageSize": 10
            }).classes("w-full")
        else:
            ui.label("No trade data available.").classes("text-warning")

def display_strategy_comparison(container, all_results, user_storage):
    with container:
        ui.label("Strategy Comparison").classes("text-subtitle1 mt-4")

        comparison_data = []
        for result in all_results:
            metrics = BacktestMetrics.calculate_metrics(pd.DataFrame(result.get("Tradebook", [])), result.get("InitialCapital", 100000))
            comparison_data.append({
                "Strategy": result.get("StrategyName", "Unknown"),
                "Total Profit": metrics["TotalProfit"],
                "Win Rate": metrics["WinRate"],
                "Sharpe Ratio": metrics["SharpeRatio"],
                "Max Drawdown": metrics["MaxDrawdown"],
                "Profit Factor": metrics["ProfitFactor"]
            })

        comparison_grid = ui.aggrid({
            "columnDefs": [
                {"headerName": "Strategy", "field": "Strategy"},
                {"headerName": "Total Profit (₹)", "field": "Total Profit", "valueFormatter": "value.toFixed(2)"},
                {"headerName": "Win Rate (%)", "field": "Win Rate", "valueFormatter": "value.toFixed(2)"},
                {"headerName": "Sharpe Ratio", "field": "Sharpe Ratio", "valueFormatter": "value.toFixed(3)"},
                {"headerName": "Max Drawdown (%)", "field": "Max Drawdown", "valueFormatter": "value.toFixed(2)"},
                {"headerName": "Profit Factor", "field": "Profit Factor", "valueFormatter": "value.toFixed(2)"}
            ],
            "rowData": comparison_data
        }).classes("w-full")

        df_comparison = pd.DataFrame([{
            "timestamp": pd.to_datetime(trade["Date"]),
            "value": trade["PortfolioValue"],
            "strategy": result.get("StrategyName", "Unknown")
        } for result in all_results for trade in result.get("Tradebook", [])])

        if not df_comparison.empty:
            fig = go.Figure()
            for strategy in df_comparison["strategy"].unique():
                strategy_data = df_comparison[df_comparison["strategy"] == strategy]
                fig.add_trace(go.Scatter(
                    x=strategy_data["timestamp"],
                    y=strategy_data["value"],
                    mode="lines",
                    name=strategy
                ))

            theme_mode = user_storage.get("app_theme", "Dark")
            font_color = "white" if theme_mode == "Dark" else "black"

            fig.update_layout(
                title="Strategy Performance Comparison",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=400,
                font_color=font_color,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )

            ui.plotly(fig).classes("w-full")