"""
Analytics Module for NiceGUI Algo Trading Application
Implements UI for viewing historical data charts with technical indicators.
"""

from nicegui import ui
import pandas as pd
import asyncio
import logging
import plotly.graph_objects as go
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# --- Analytics Page --- #

async def render_analytics_page(fetch_api, user_storage, instruments):
    """Render the main page for analytics and charting."""
    broker = user_storage.get("broker", "Zerodha")
    ui.label("Analytics & Charting").classes("text-h5 q-pa-md")

    # --- Chart Configuration --- #
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-end gap-4"):
            # Instrument Selection
            instrument_select = ui.select(options=sorted(list(instruments.keys())),
                                          label="Select Instrument",
                                          with_input=True,
                                          value=list(instruments.keys())[0] if instruments else None)\
                                          .classes("w-48")

            # Timeframe Selection
            timeframe_select = ui.select(options=["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"],
                                         label="Timeframe",
                                         value="day").classes("w-32")

            # Date Range Selection (Optional - could fetch last N candles instead)
            with ui.row().classes("items-center"): 
                start_date = ui.date(label="Start Date", value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")).classes("flex-grow")
                end_date = ui.date(label="End Date", value=datetime.now().strftime("%Y-%m-%d")).classes("flex-grow")

            # Indicator Selection (Multi-select Checkboxes)
            with ui.column():
                ui.label("Indicators").classes("text-caption")
                indicators_group = {}
                with ui.row(): # Layout checkboxes horizontally
                    indicators_group["SMA_20"] = ui.checkbox("SMA(20)")
                    indicators_group["EMA_50"] = ui.checkbox("EMA(50)")
                    indicators_group["RSI_14"] = ui.checkbox("RSI(14)")
                    indicators_group["MACD"] = ui.checkbox("MACD(12,26,9)")
                    # Add more indicators as needed

            # Update Chart Button
            update_button = ui.button("Update Chart", on_click=lambda: update_chart()).props("color=primary")

    # --- Chart Display --- #
    chart_container = ui.column().classes("w-full mt-4")
    with chart_container:
        chart = ui.plotly().classes("w-full h-96") # Adjust height as needed
        status_label = ui.label("Select parameters and update chart.").classes("text-caption q-pa-md")

    # --- Chart Update Logic --- #
    async def update_chart():
        selected_symbol = instrument_select.value
        selected_timeframe = timeframe_select.value
        selected_start = start_date.value
        selected_end = end_date.value
        selected_indicators = [name for name, cb in indicators_group.items() if cb.value]

        if not selected_symbol or selected_symbol not in instruments:
            ui.notify("Please select a valid instrument.", type="negative")
            return
        
        instrument_token = instruments[selected_symbol]

        # Validate dates
        try:
            start_dt = datetime.strptime(selected_start, "%Y-%m-%d")
            end_dt = datetime.strptime(selected_end, "%Y-%m-%d")
            if start_dt >= end_dt:
                ui.notify("Start date must be before end date.", type="negative")
                return
        except ValueError:
            ui.notify("Invalid date format.", type="negative")
            return

        status_label.text = f"Loading {selected_symbol} ({selected_timeframe}) data..."
        chart.clear()
        update_button.props("loading=true disabled=true")

        try:
            # Construct API endpoint - Adjust based on your backend API structure
            # Example: /historical_data/{broker}/{instrument_token}?timeframe=...&from=...&to=...&indicators=SMA_20,RSI_14
            api_endpoint = f"/historical_data/{broker}/{instrument_token}"
            params = {
                "timeframe": selected_timeframe,
                "from_date": selected_start,
                "to_date": selected_end,
                "indicators": ",".join(selected_indicators) # Pass indicators as comma-separated string
            }
            
            # Use GET request with query parameters
            response = await fetch_api(api_endpoint, method="GET", data=params) # Pass params in data for GET helper

            if response and isinstance(response, list) and len(response) > 0:
                df = pd.DataFrame(response)
                # Ensure required columns exist and have correct types
                required_cols = {"timestamp": "datetime64[ns]", "open": float, "high": float, "low": float, "close": float}
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                for col, dtype in required_cols.items():
                    if col not in df.columns:
                        raise ValueError(f"Missing required column: {col}")
                    if col != "timestamp":
                         df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=list(required_cols.keys()))
                df = df.sort_values("timestamp")

                if df.empty:
                    status_label.text = "No historical data found for the selected parameters."
                    ui.notify("No data found.", type="warning")
                    return

                # Create Plotly figure
                fig = create_candlestick_chart(df, selected_symbol, selected_indicators, user_storage)
                chart.update(fig)
                status_label.text = f"Displaying {selected_symbol} ({selected_timeframe}) from {selected_start} to {selected_end}"
                ui.notify("Chart updated successfully!", type="positive")
            elif isinstance(response, list) and len(response) == 0:
                status_label.text = "No historical data found for the selected parameters."
                ui.notify("No data found.", type="warning")
            else:
                status_label.text = "Failed to load chart data."
                logger.error(f"Failed to fetch or parse historical data: {response}")
                ui.notify("Failed to load chart data.", type="negative")

        except Exception as e:
            status_label.text = f"Error loading chart: {e}"
            logger.exception("Error updating analytics chart")
            ui.notify(f"Error loading chart: {e}", type="negative")
        finally:
            update_button.props("loading=false disabled=false")

    # Initial chart load (optional, can wait for user interaction)
    # await update_chart()

# --- Helper Function to Create Chart --- #
def create_candlestick_chart(df, symbol, indicators, user_storage):
    """Creates the Plotly candlestick chart with selected indicators."""
    fig = go.Figure()

    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(x=df["timestamp"],
                               open=df["open"],
                               high=df["high"],
                               low=df["low"],
                               close=df["close"],
                               name="Candlestick"))

    # 2. Add Indicator Traces
    indicator_colors = ["orange", "purple", "cyan", "magenta", "yellow", "lime"]
    color_index = 0

    # Subplots for indicators like RSI and MACD
    subplot_traces = []
    subplot_needed = any(ind in indicators for ind in ["RSI_14", "MACD"])

    for indicator in indicators:
        indicator_col = indicator # Assume column name matches indicator key
        if indicator_col in df.columns:
            df[indicator_col] = pd.to_numeric(df[indicator_col], errors="coerce") # Ensure numeric
            color = indicator_colors[color_index % len(indicator_colors)]
            color_index += 1

            if indicator == "RSI_14":
                # Add RSI to a subplot
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df[indicator_col], mode="lines", name="RSI", line=dict(color=color)), row=2, col=1)
                subplot_traces.append("RSI")
            elif indicator == "MACD":
                 # MACD requires multiple columns (macd, signal, histogram)
                 if all(c in df.columns for c in ["macd", "macd_signal", "macd_hist"]):
                     df["macd"] = pd.to_numeric(df["macd"], errors="coerce")
                     df["macd_signal"] = pd.to_numeric(df["macd_signal"], errors="coerce")
                     df["macd_hist"] = pd.to_numeric(df["macd_hist"], errors="coerce")
                     # Add MACD lines to subplot 3
                     fig.add_trace(go.Scatter(x=df["timestamp"], y=df["macd"], mode="lines", name="MACD", line=dict(color=color)), row=3, col=1)
                     fig.add_trace(go.Scatter(x=df["timestamp"], y=df["macd_signal"], mode="lines", name="Signal", line=dict(color=indicator_colors[color_index % len(indicator_colors)])), row=3, col=1)
                     color_index += 1
                     # Add MACD histogram to subplot 3
                     hist_colors = ["green" if v >= 0 else "red" for v in df["macd_hist"]]
                     fig.add_trace(go.Bar(x=df["timestamp"], y=df["macd_hist"], name="Histogram", marker_color=hist_colors), row=3, col=1)
                     subplot_traces.append("MACD")
                 else:
                     logger.warning("MACD selected but required columns (macd, macd_signal, macd_hist) not found in data.")
            else:
                # Add other indicators (like SMA, EMA) to the main chart
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df[indicator_col], mode="lines", name=indicator, line=dict(color=color)))
        else:
            logger.warning(f"Indicator ", {indicator}, " selected but corresponding column not found in data.")

    # 3. Configure Layout
    theme_mode = user_storage.get("theme", "Dark")
    font_color = "white" if theme_mode == "Dark" else "black"
    paper_bgcolor = "rgba(0,0,0,0)" # Transparent background
    plot_bgcolor = "rgba(0,0,0,0)"
    grid_color = "rgba(128, 128, 128, 0.3)"

    # Define layout with subplots if needed
    layout_args = {
        "title": f"{symbol} Candlestick Chart",
        "xaxis_title": "Date",
        "yaxis_title": "Price",
        "xaxis_rangeslider_visible": False,
        "margin": dict(l=40, r=40, t=50, b=40),
        "font_color": font_color,
        "paper_bgcolor": paper_bgcolor,
        "plot_bgcolor": plot_bgcolor,
        "xaxis": dict(gridcolor=grid_color),
        "yaxis": dict(gridcolor=grid_color),
        "hovermode": "x unified",
    }

    if subplot_needed:
        rows = 1 + ("RSI" in subplot_traces) + ("MACD" in subplot_traces)
        row_heights = [0.6] + ([0.2] * (rows - 1)) # Main chart takes more space
        fig.update_layout(grid=dict(rows=rows, columns=1, pattern="independent"), yaxis_domain=[0.65, 1.0]) # Adjust domain for main chart
        if "RSI" in subplot_traces:
            fig.update_layout(yaxis2_title="RSI", yaxis2=dict(domain=[0.3, 0.6], gridcolor=grid_color))
        if "MACD" in subplot_traces:
             fig.update_layout(yaxis3_title="MACD", yaxis3=dict(domain=[0.0, 0.25], gridcolor=grid_color))
        # Hide x-axis labels on upper plots for cleaner look
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        if "RSI" in subplot_traces and "MACD" in subplot_traces:
             fig.update_xaxes(showticklabels=False, row=2, col=1)

    fig.update_layout(**layout_args)

    return fig

