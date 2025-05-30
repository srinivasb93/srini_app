"""
Backtesting Module for NiceGUI Algo Trading Application
Implements UI for running backtests on defined strategies and viewing results.
"""

from nicegui import ui
import pandas as pd
import asyncio
import json
import logging
import plotly.graph_objects as go
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# --- Backtesting Page --- #

async def render_backtesting_page(fetch_api, user_storage):
    """Render the main page for running and viewing backtests."""
    broker = user_storage.get("broker", "Zerodha")
    ui.label("Strategy Backtesting").classes("text-h5 q-pa-md")

    with ui.row().classes("w-full gap-4 items-start"):
        # --- Configuration Card --- #
        with ui.card().classes("w-1/3"):
            ui.label("Backtest Configuration").classes("text-h6")

            # Fetch available strategies
            strategies_select = ui.select(options=[], label="Select Strategy").classes("w-full")
            
            async def fetch_strategies_for_backtest():
                strategies = await fetch_api(f"/strategies/{broker}")
                if strategies and isinstance(strategies, list):
                    strategy_options = {s["strategy_id"]: s["name"] for s in strategies}
                    strategies_select.options = strategy_options
                    strategies_select.update()
                else:
                    strategies_select.options = {}
                    strategies_select.update()
                    ui.notify("Could not fetch strategies.", type="warning")
            
            await fetch_strategies_for_backtest()

            # Date Range Selection
            with ui.row().classes("w-full items-center"): 
                start_date = ui.date(label="Start Date", value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")).classes("flex-grow")
                end_date = ui.date(label="End Date", value=datetime.now().strftime("%Y-%m-%d")).classes("flex-grow")

            # Initial Capital
            initial_capital = ui.number(label="Initial Capital", value=100000, format="%.0f", step=10000).classes("w-full")

            # Slippage and Commission (Optional)
            slippage = ui.number(label="Slippage (% per trade)", value=0.01, format="%.2f", step=0.01).classes("w-full")
            commission = ui.number(label="Commission (₹ per trade)", value=0.0, format="%.2f", step=0.1).classes("w-full")

            # Run Backtest Button
            run_button = ui.button("Run Backtest", on_click=lambda: run_backtest()).props("color=primary").classes("w-full mt-4")

        # --- Results Card --- #
        with ui.card().classes("flex-grow"):
            ui.label("Backtest Results").classes("text-h6")
            results_container = ui.column().classes("w-full")

            with results_container:
                ui.label("Run a backtest to see results.").classes("text-caption")

    # --- Backtest Execution Logic --- #
    async def run_backtest():
        if not strategies_select.value:
            ui.notify("Please select a strategy.", type="negative")
            return
        
        strategy_id = strategies_select.value
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date.value, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date.value, "%Y-%m-%d")
            if start_dt >= end_dt:
                ui.notify("Start date must be before end date.", type="negative")
                return
        except ValueError:
            ui.notify("Invalid date format.", type="negative")
            return

        backtest_params = {
            "strategy_id": strategy_id,
            "start_date": start_date.value,
            "end_date": end_date.value,
            "initial_capital": float(initial_capital.value),
            "slippage_percent": float(slippage.value),
            "commission_per_trade": float(commission.value),
            "broker": broker
        }

        results_container.clear()
        with results_container:
            with ui.row().classes("items-center"): 
                ui.spinner(size="lg")
                ui.label("Running backtest...").classes("ml-2")
        
        run_button.props("loading=true disabled=true")
        
        try:
            # Assuming a backend endpoint /backtest/
            response = await fetch_api("/backtest/", method="POST", data=backtest_params)
            
            results_container.clear()
            if response and isinstance(response, dict) and "metrics" in response and "equity_curve" in response:
                display_backtest_results(results_container, response)
                ui.notify("Backtest completed successfully!", type="positive")
            else:
                with results_container:
                    ui.label("Failed to run backtest or invalid response received.").classes("text-negative")
                logger.error(f"Backtest failed or returned invalid data: {response}")
                ui.notify("Backtest failed.", type="negative")
        except Exception as e:
            results_container.clear()
            with results_container:
                ui.label(f"An error occurred during backtest: {e}").classes("text-negative")
            logger.exception("Error during backtest execution")
            ui.notify("An error occurred during backtest.", type="negative")
        finally:
             run_button.props("loading=false disabled=false")

# --- Helper Function to Display Results --- #
def display_backtest_results(container, results):
    """Populates the results container with metrics and the equity curve."""
    metrics = results.get("metrics", {})
    equity_curve_data = results.get("equity_curve", []) # Expecting list of {"timestamp": ..., "value": ...}

    with container:
        # Display Key Metrics
        with ui.grid(columns=2).classes("gap-4 mb-4"):
            def display_metric(label, value, format_spec=",.2f"):
                with ui.column().classes("items-center"):
                    ui.label(label).classes("text-caption")
                    try:
                        # Attempt to format as float, fallback to string
                        formatted_value = f"{float(value):{format_spec}}"
                    except (ValueError, TypeError):
                        formatted_value = str(value)
                    ui.label(formatted_value).classes("text-subtitle1 font-bold")
            
            display_metric("Total Return (%) ", metrics.get("total_return_pct", "N/A"))
            display_metric("Annualized Return (%) ", metrics.get("annualized_return_pct", "N/A"))
            display_metric("Max Drawdown (%) ", metrics.get("max_drawdown_pct", "N/A"))
            display_metric("Sharpe Ratio", metrics.get("sharpe_ratio", "N/A"))
            display_metric("Sortino Ratio", metrics.get("sortino_ratio", "N/A"))
            display_metric("Win Rate (%) ", metrics.get("win_rate_pct", "N/A"))
            display_metric("Profit Factor", metrics.get("profit_factor", "N/A"))
            display_metric("Total Trades", metrics.get("total_trades", "N/A"), format_spec=",.0f")
            # Add more metrics as needed

        ui.separator()

        # Display Equity Curve Chart
        ui.label("Equity Curve").classes("text-subtitle1 mt-4")
        equity_chart = ui.plotly().classes("w-full h-64")

        if equity_curve_data:
            try:
                df = pd.DataFrame(equity_curve_data)
                # Ensure timestamp is datetime and value is numeric
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["value"] = pd.to_numeric(df["value"])
                df = df.sort_values("timestamp")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["value"], mode="lines", name="Equity"))
                
                # Determine theme for chart colors
                user_storage = ui.context.client.storage.user # Access user storage
                theme_mode = user_storage.get("theme", "Dark")
                font_color = "white" if theme_mode == "Dark" else "black"
                paper_bgcolor = "rgba(0,0,0,0)" # Transparent background
                plot_bgcolor = "rgba(0,0,0,0)"
                grid_color = "rgba(128, 128, 128, 0.5)" # Grey grid lines

                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value",
                    margin=dict(l=20, r=20, t=40, b=20),
                    font_color=font_color,
                    paper_bgcolor=paper_bgcolor,
                    plot_bgcolor=plot_bgcolor,
                    xaxis=dict(gridcolor=grid_color),
                    yaxis=dict(gridcolor=grid_color)
                )
                equity_chart.update(fig)
            except Exception as e:
                logger.error(f"Error creating equity curve chart: {e}")
                equity_chart.clear()
                ui.label("Error displaying equity curve chart.").classes("text-negative")
        else:
            equity_chart.clear()
            ui.label("No equity curve data available.").classes("text-warning")

        # TODO: Add trade log display (e.g., using AG Grid)
        # trade_log = results.get("trade_log", [])
        # if trade_log:
        #     ui.label("Trade Log").classes("text-subtitle1 mt-4")
        #     trade_grid = ui.aggrid({...}) # Define columns and data
        #     await trade_grid.update_grid_options({"rowData": trade_log})

