"""
Analytics Module for NiceGUI Algo Trading Application
Implements a feature-rich TradingView Lightweight Charts using ui.html and ui.run_javascript.
Replicates chart capabilities from stock_analysis.py, including clean display, legends, indicators, subcharts, drawing tools, replay, and more.
Fetches historical data from a custom API endpoint.
"""

import json
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nicegui import ui, app
import aiohttp

logger = logging.getLogger(__name__)

# Theme configuration for the chart
THEME_CONFIG = {
    "dark": {
        "bg": "#161616",
        "text": "#FFFFFF",
        "grid": "#262616",
        "watermark": "rgba(255, 255, 255, 0.1)",
        "candle_up": "#6FB76F",
        "candle_down": "#FF6F6F",
        "volume": "rgba(38, 166, 154, 0.5)",
        "sma": "#F59E0B",
        "ema": "#8B5CF6",
        "rsi": "#06B6D4",
        "macd": "#2596be",
        "macd_signal": "#FF0000",
        "macd_hist": "#2596be",
        "bb_upper": "#FF0000",
        "bb_middle": "#00FFFF",
        "bb_lower": "#00FF00",
        "linreg": "#FF00FF"
    },
    "light": {
        "bg": "#FFFFFF",
        "text": "#000000",
        "grid": "#D3D3D3",
        "watermark": "rgba(0, 0, 0, 0.1)",
        "candle_up": "#6FB76F",
        "candle_down": "#FF6F6F",
        "volume": "rgba(38, 166, 154, 0.5)",
        "sma": "#F59E0B",
        "ema": "#8B5CF6",
        "rsi": "#06B6D4",
        "macd": "#2596be",
        "macd_signal": "#FF0000",
        "macd_hist": "#2596be",
        "bb_upper": "#FF0000",
        "bb_middle": "#00FFFF",
        "bb_lower": "#00FF00",
        "linreg": "#FF00FF"
    }
}

# Function to calculate SMA
def calculate_sma(series: pd.Series, period: int) -> list:
    sma = series.rolling(window=period).mean()
    return sma.tolist()

# Function to calculate EMA
def calculate_ema(series: pd.Series, period: int) -> list:
    ema = series.ewm(span=period, adjust=False).mean()
    return ema.tolist()

# Function to calculate RSI
def calculate_rsi(series: pd.Series, period: int) -> list:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.tolist()

# Function to calculate BBANDS
def calculate_bbands(series: pd.Series, period: int, std: float) -> dict:
    mean = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper = mean + (std_dev * std)
    lower = mean - (std_dev * std)
    return {
        "upper": upper.tolist(),
        "middle": mean.tolist(),
        "lower": lower.tolist()
    }

# Function to calculate MACD
def calculate_macd(series: pd.Series, fast: int, slow: int, signal: int) -> dict:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return {
        "macd": macd.tolist(),
        "signal": signal_line.tolist(),
        "hist": histogram.tolist()
    }

# Function to calculate LINREG slope (optimized)
def calculate_linreg(series: pd.Series, period: int) -> list:
    result = [None] * (period - 1)
    x = np.arange(period)
    x_mean = x.mean()
    x_squared = np.sum((x - x_mean) ** 2)
    for i in range(period - 1, len(series)):
        y = series[i - period + 1:i + 1].values
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / x_squared if x_squared != 0 else 0
        result.append(slope)
    return result

# Merge default state with stored state
def merge_state(stored_state, default_state):
    merged = default_state.copy()
    if stored_state:
        for key, value in stored_state.items():
            if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = merge_state(value, merged[key])
            else:
                merged[key] = value
    return merged

async def render_analytics_page(fetch_api, user_storage, instruments):
    broker = user_storage.get("broker", "Zerodha")
    ui.label("Analytics & Charting").classes("text-h4")

    # Default state
    default_state = {
        "indicators": {},
        "drawings": [],
        "templates": {"Default": {"indicators": {}, "drawings": []}},
        "current_template": "Default",
        "watchlist": list(instruments.keys()),
        "current_replay_index": -1,
        "last_replay_date": None,
        "is_playing": False
    }

    # Merge stored state with default state
    stored_state = user_storage.get("chart_state", {})
    chart_state = merge_state(stored_state, default_state)
    user_storage["chart_state"] = chart_state

    with ui.card().classes("card"):
        with ui.row().classes("w-full items-end gap-4"):
            with ui.column().classes("w-48"):
                ui.label("Select Instrument").classes("text-subtitle1 mb-2")
                instrument_select = ui.select(
                    options=sorted(list(instruments.keys())),
                    with_input=True,
                    value=list(instruments.keys())[0] if instruments else None
                ).classes("input")
            with ui.column().classes("w-32"):
                ui.label("Timeframe").classes("text-subtitle1 mb-2")
                timeframe_select = ui.select(
                    options=["1m", "5m", "15m", "1h", "1d"],
                    value="1d"
                ).classes("input")
            with ui.column().classes("flex-grow"):
                ui.label("Start Date").classes("text-subtitle1 mb-2")
                start_date = ui.date(
                    value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                ).classes("input")
            with ui.column().classes("flex-grow"):
                ui.label("End Date").classes("text-subtitle1 mb-2")
                end_date = ui.date(
                    value=datetime.now().strftime("%Y-%m-%d")
                ).classes("input")

        # Indicators, Drawing Tools, Watchlist, Replay, and Templates UI
        with ui.row().classes("w-full items-center gap-4 mt-2"):
            ui.label("Indicators").classes("text-subtitle1")
            with ui.column().classes("space-y-2"):
                with ui.row().classes("space-x-2"):
                    selected_indicator = ui.select(
                        options=["EMA", "SMA", "BBANDS", "MACD", "RSI", "LINREG"],
                        value="EMA",
                        label="Indicator"
                    ).classes("w-32")
                    params = {}
                    if selected_indicator.value in ["EMA", "SMA", "RSI", "LINREG"]:
                        period_input = ui.number(value=20, min=2, max=100, label="Period").props("dense").classes("w-32")
                        params["period"] = period_input
                    elif selected_indicator.value == "BBANDS":
                        period_input = ui.number(value=20, min=2, max=100, label="Period").props("dense").classes("w-32")
                        std_input = ui.number(value=2.0, min=0.1, max=5.0, step=0.1, label="Std Dev").props("dense").classes("w-32")
                        params["period"] = period_input
                        params["std"] = std_input
                    elif selected_indicator.value == "MACD":
                        fast_input = ui.number(value=12, min=2, max=100, label="Fast").props("dense").classes("w-32")
                        slow_input = ui.number(value=26, min=2, max=100, label="Slow").props("dense").classes("w-32")
                        signal_input = ui.number(value=9, min=2, max=100, label="Signal").props("dense").classes("w-32")
                        params["fast"] = fast_input
                        params["slow"] = slow_input
                        params["signal"] = signal_input
                    add_button = ui.button("Add", on_click=lambda: add_indicator(selected_indicator.value, params))

            ui.label("Drawing Tools").classes("text-subtitle1 ml-4")
            drawing_tool = ui.select(
                options=["Trendline", "Horizontal Line", "Rectangle"],
                value="Trendline",
                label="Tool"
            ).classes("w-32")
            draw_button = ui.button("Draw", on_click=lambda: start_drawing(drawing_tool.value))

            ui.label("Watchlist").classes("text-subtitle1 ml-4")
            watchlist_select = ui.select(
                options=chart_state["watchlist"],
                value=instrument_select.value,
                on_change=lambda e: update_instrument(e.value)
            ).classes("w-32")

            with ui.row().classes("space-x-2"):
                prev_button = ui.button("Previous Day", on_click=lambda: adjust_replay(-1))
                next_button = ui.button("Next Day", on_click=lambda: adjust_replay(1))
                play_button = ui.button("Play/Pause", on_click=lambda: toggle_play())

            ui.label("Templates").classes("text-subtitle1 ml-4")
            template_select = ui.select(
                options=list(chart_state["templates"].keys()),
                value=chart_state["current_template"],
                on_change=lambda e: load_template(e.value)
            ).classes("w-32")
            template_name = ui.input("Template Name").props("dense").classes("w-32")
            save_template_button = ui.button("Save Template", on_click=lambda: save_template(template_name.value))

        update_button = ui.button(
            "Update Chart",
            on_click=lambda: update_chart()
        ).classes("button-primary ml-4")

    chart_container = ui.column().classes("w-full mt-4").style("width: 100%; height: 60vh; visibility: visible; display: block;")
    with chart_container:
        chart_element = ui.html("")  # Placeholder for chart container
        status_label = ui.label(
            "Select parameters and update chart."
        ).classes("text-subtitle1 mt-2 text-gray-400")

    # Drawing state
    drawing_mode = {"active": False, "tool": None, "start_point": None, "end_point": None}

    def start_drawing(tool):
        drawing_mode["active"] = True
        drawing_mode["tool"] = tool
        drawing_mode["start_point"] = None
        drawing_mode["end_point"] = None
        ui.notify(f"Started drawing {tool}. Click on the chart to draw.")
        # Notify JavaScript about drawing mode
        ui.run_javascript(f"""
        window.drawingMode = {{ active: true, tool: '{tool}', start_point: null, end_point: null }};
        """)

    async def add_indicator(indicator_type, params):
        if indicator_type not in chart_state["indicators"]:
            chart_state["indicators"][indicator_type] = []
        param_dict = {}
        if indicator_type in ["EMA", "SMA", "RSI", "LINREG"]:
            param_dict["period"] = params["period"].value
        elif indicator_type == "BBANDS":
            param_dict["period"] = params["period"].value
            param_dict["std"] = params["std"].value
        elif indicator_type == "MACD":
            param_dict["fast"] = params["fast"].value
            param_dict["slow"] = params["slow"].value
            param_dict["signal"] = params["signal"].value
        chart_state["indicators"][indicator_type].append(param_dict)
        user_storage["chart_state"] = chart_state
        await update_chart()
        ui.notify(f"Added {indicator_type} indicator", type="positive")

    async def update_instrument(new_symbol):
        instrument_select.value = new_symbol
        chart_state["current_replay_index"] = -1  # Reset replay on instrument change
        user_storage["chart_state"] = chart_state
        await update_chart()

    async def adjust_replay(step):
        chart_state["is_playing"] = False
        chart_state["current_replay_index"] = max(0, min(len(df) - 1, chart_state["current_replay_index"] + step))
        user_storage["chart_state"] = chart_state
        await ui.run_javascript(f"""
        window.isPlaying_{instrument_select.value} = false;
        startReplay();
        """)
        await update_chart()

    async def toggle_play():
        chart_state["is_playing"] = not chart_state["is_playing"]
        user_storage["chart_state"] = chart_state
        ui.notify("Starting bar replay..." if chart_state["is_playing"] else "Paused bar replay")
        await ui.run_javascript(f"""
        window.isPlaying_{instrument_select.value} = {json.dumps(chart_state["is_playing"])};
        startReplay();
        """)

    async def save_template(name):
        if not name:
            ui.notify("Please enter a template name.", type="negative")
            return
        chart_state["templates"][name] = {
            "indicators": chart_state["indicators"],
            "drawings": chart_state["drawings"]
        }
        chart_state["current_template"] = name
        template_select.options = list(chart_state["templates"].keys())
        template_select.value = name
        user_storage["chart_state"] = chart_state
        ui.notify(f"Template '{name}' saved successfully!", type="positive")

    async def load_template(name):
        if name in chart_state["templates"]:
            chart_state["indicators"] = chart_state["templates"][name]["indicators"]
            chart_state["drawings"] = chart_state["templates"][name]["drawings"]
            chart_state["current_template"] = name
            user_storage["chart_state"] = chart_state
            await update_chart()
            ui.notify(f"Template '{name}' loaded successfully!", type="positive")

    df = pd.DataFrame()  # Define df globally for access

    async def update_chart():
        nonlocal df
        selected_symbol = instrument_select.value
        selected_timeframe = timeframe_select.value
        selected_start = start_date.value
        selected_end = end_date.value

        if not selected_symbol or selected_symbol not in instruments:
            status_label.text = "Invalid instrument selection."
            ui.notify("Please select a valid instrument.", type="negative")
            return

        try:
            start_dt = datetime.strptime(selected_start, "%Y-%m-%d")
            end_dt = datetime.strptime(selected_end, "%Y-%m-%d")
            if start_dt >= end_dt:
                status_label.text = "Invalid date range."
                ui.notify("Start date must be before end date.", type="negative")
                return
        except ValueError:
            status_label.text = "Invalid date format."
            ui.notify("Invalid date format.", type="negative")
            return

        theme = user_storage.get("theme", "dark")
        theme_config = THEME_CONFIG.get(theme.lower(), THEME_CONFIG["dark"])

        status_label.text = f"Loading {selected_symbol} ({selected_timeframe}) data..."
        update_button.props("loading=true disabled=true")

        try:
            # Map timeframe to unit and interval for API
            timeframe_mapping = {
                "1m": ("minute", "1"),
                "5m": ("minute", "5"),
                "15m": ("minute", "15"),
                "1h": ("hour", "1"),
                "1d": ("days", "1")
            }
            unit, interval = timeframe_mapping.get(selected_timeframe, ("days", "1"))

            # Fetch historical data from custom API endpoint
            historical_endpoint = f"/historical-data/Upstox"
            historical_params = {
                "instrument": instruments[selected_symbol],
                "from_date": selected_start,
                "to_date": selected_end,
                "unit": unit,
                "interval": interval
            }
            response = await fetch_api(historical_endpoint, method="GET", params=historical_params)
            logger.info(f"Historical response: {response}")

            if not response or not isinstance(response, dict) or "data" not in response:
                status_label.text = "Invalid historical data response."
                ui.notify("Invalid data from server.", type="warning")
                logger.error(f"Invalid historical data response: {response}")
                raise ValueError("Invalid historical data response")

            if not response["data"]:
                status_label.text = "No historical data available."
                ui.notify("No historical data for the selected parameters.", type="warning")
                logger.warning("Empty historical data received.")
                raise ValueError("Empty historical data received")

            # Process historical data
            df = pd.DataFrame([{
                "time": int(pd.to_datetime(point["timestamp"]).timestamp()),  # seconds for Lightweight Charts
                "open": float(point.get("open", 0)),
                "high": float(point.get("high", 0)),
                "low": float(point.get("low", 0)),
                "close": float(point.get("close", 0)),
                "volume": int(point.get("volume", 0))
            } for point in response["data"] if point.get("timestamp")])
            if df.empty:
                status_label.text = "No data to display."
                ui.notify("No data available after processing.", type="warning")
                logger.warning("DataFrame is empty after processing.")
                raise ValueError("DataFrame is empty after processing")
            df = df.sort_values("time")

            # Apply replay index
            replay_index = chart_state.get("current_replay_index", -1)
            if replay_index == -1:
                replay_index = len(df) - 1
            replay_data = df.iloc[:replay_index + 1].copy()

            # Calculate indicators
            close_series = replay_data["close"]
            indicators = {}
            for indicator_type, instances in chart_state["indicators"].items():
                indicators[indicator_type] = []
                for params in instances:
                    if indicator_type == "SMA":
                        sma_data = calculate_sma(close_series, params["period"])
                        sma_series = [{"time": row["time"], "value": val} for row, val in zip(replay_data.to_dict("records"), sma_data) if val is not None]
                        indicators[indicator_type].append({
                            "name": f"SMA ({params['period']})",
                            "data": sma_series,
                            "color": theme_config["sma"],
                            "pane": 0,
                            "type": "line",
                            "enabled": True
                        })
                    elif indicator_type == "EMA":
                        ema_data = calculate_ema(close_series, params["period"])
                        ema_series = [{"time": row["time"], "value": val} for row, val in zip(replay_data.to_dict("records"), ema_data) if val is not None]
                        indicators[indicator_type].append({
                            "name": f"EMA ({params['period']})",
                            "data": ema_series,
                            "color": theme_config["ema"],
                            "pane": 0,
                            "type": "line",
                            "enabled": True
                        })
                    elif indicator_type == "RSI":
                        rsi_data = calculate_rsi(close_series, params["period"])
                        rsi_series = [{"time": row["time"], "value": val} for row, val in zip(replay_data.to_dict("records"), rsi_data) if val is not None]
                        indicators[indicator_type].append({
                            "name": f"RSI ({params['period']})",
                            "data": rsi_series,
                            "color": theme_config["rsi"],
                            "pane": 1,
                            "type": "line",
                            "enabled": True
                        })
                    elif indicator_type == "BBANDS":
                        bbands = calculate_bbands(close_series, params["period"], params["std"])
                        for band, color_key in [("upper", "bb_upper"), ("middle", "bb_middle"), ("lower", "bb_lower")]:
                            band_series = [{"time": row["time"], "value": val} for row, val in zip(replay_data.to_dict("records"), bbands[band]) if val is not None]
                            indicators[indicator_type].append({
                                "name": f"BB_{band.upper()} ({params['period']},{params['std']})",
                                "data": band_series,
                                "color": theme_config[color_key],
                                "pane": 0,
                                "type": "line",
                                "enabled": True
                            })
                    elif indicator_type == "MACD":
                        macd = calculate_macd(close_series, params["fast"], params["slow"], params["signal"])
                        for key, color_key, type_key in [("macd", "macd", "line"), ("signal", "macd_signal", "line"), ("hist", "macd_hist", "histogram")]:
                            series_data = [{"time": row["time"], "value": val} for row, val in zip(replay_data.to_dict("records"), macd[key]) if val is not None]
                            indicators[indicator_type].append({
                                "name": f"MACD_{key.upper()} ({params['fast']},{params['slow']},{params['signal']})",
                                "data": series_data,
                                "color": theme_config[color_key],
                                "pane": 2 if key != "hist" else 3,
                                "type": type_key,
                                "enabled": True
                            })
                    elif indicator_type == "LINREG":
                        linreg_data = calculate_linreg(close_series, params["period"])
                        linreg_series = [{"time": row["time"], "value": val} for row, val in zip(replay_data.to_dict("records"), linreg_data) if val is not None]
                        indicators[indicator_type].append({
                            "name": f"LINREG ({params['period']})",
                            "data": linreg_series,
                            "color": theme_config["linreg"],
                            "pane": 0,
                            "type": "line",
                            "enabled": True
                        })

            # Prepare chart data
            ohlc_data = replay_data[["time", "open", "high", "low", "close"]].to_dict("records")
            volume_data = [{"time": row["time"], "value": row["volume"]} for row in replay_data.to_dict("records")]

            # Calculate total height based on subcharts
            num_subcharts = sum(1 for ind in chart_state["indicators"] if ind in ["RSI", "MACD"])
            total_height = 600 + (num_subcharts * 100)
            main_chart_height = 600 - (num_subcharts * 100) if num_subcharts > 0 else 600
            scale_margins_bottom = 0.1 + (num_subcharts * 0.1)

            # Render the chart container using ui.html
            chart_container_html = f"""
            <div id="chart_{selected_symbol}" style="width: 100%; height: {total_height}px; position: relative;">
                <div id="chart_topbar_{selected_symbol}" style="position: absolute; top: 0px; left: 50%; transform: translateX(-50%); z-index: 1000; color: {theme_config['text']}; font-size: 20px;"></div>
                <div id="chart_legend_{selected_symbol}" style="position: absolute; top: 30px; left: 10px; z-index: 1000;"></div>
                <div id="chart_controls_{selected_symbol}" style="position: absolute; top: 30px; right: 10px; z-index: 1000;">
                    <select id="indicator_select_{selected_symbol}">
                        <option value="">Add Indicator</option>
                        <option value="SMA">SMA</option>
                        <option value="EMA">EMA</option>
                        <option value="RSI">RSI</option>
                        <option value="BBANDS">BBANDS</option>
                        <option value="MACD">MACD</option>
                        <option value="LINREG">LINREG</option>
                    </select>
                    <input id="indicator_period_{selected_symbol}" type="number" value="20" min="1" style="width: 60px; margin-left: 5px;">
                    <button onclick="addIndicator('{selected_symbol}')">Add</button>
                </div>
            </div>
            """
            with chart_container:
                chart_element.clear()  # Clear previous chart
                ui.html(chart_container_html).style(f"width: 100%; height: {total_height}px;")

            # Load Lightweight Charts library and initialize the chart using ui.run_javascript
            chart_init_js = f"""
            // Check if Lightweight Charts is already loaded
            if (typeof LightweightCharts === 'undefined') {{
                console.log('Loading Lightweight Charts library...');
                const script = document.createElement('script');
                script.src = 'https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js';
                script.async = false; // Ensure synchronous loading
                script.onload = function() {{
                    console.log('Lightweight Charts Loaded:', typeof LightweightCharts !== 'undefined');
                    initializeChartWithRetry();
                }};
                script.onerror = function() {{
                    console.error('Failed to load Lightweight Charts library');
                }};
                document.head.appendChild(script);
            }} else {{
                console.log('Lightweight Charts already loaded, initializing chart...');
                initializeChartWithRetry();
            }}

            let chartInstance_{selected_symbol} = null;
            let candlestickSeries_{selected_symbol} = null;
            let volumeSeries_{selected_symbol} = null;
            let indicatorSeries_{selected_symbol} = {{}};
            let replayInterval_{selected_symbol} = null;
            let subcharts_{selected_symbol} = {{}};
            window.isPlaying_{selected_symbol} = {json.dumps(chart_state["is_playing"])};

            window.drawingMode = window.drawingMode || {{ active: false, tool: null, start_point: null, end_point: null }};

            function initializeChartWithRetry(attempt = 1, maxAttempts = 5) {{
                try {{
                    const chartContainer = document.getElementById('chart_{selected_symbol}');
                    if (!chartContainer) {{
                        if (attempt < maxAttempts) {{
                            console.warn('Chart container not found, retrying in 500ms... Attempt ' + attempt);
                            setTimeout(() => initializeChartWithRetry(attempt + 1, maxAttempts), 500);
                        }} else {{
                            console.error('Chart container not found after ' + maxAttempts + ' attempts');
                        }}
                        return;
                    }}
                    if (chartContainer.offsetWidth === 0 || chartContainer.offsetHeight === 0) {{
                        if (attempt < maxAttempts) {{
                            console.warn('Chart container has zero dimensions, retrying in 500ms... Attempt ' + attempt);
                            setTimeout(() => initializeChartWithRetry(attempt + 1, maxAttempts), 500);
                        }} else {{
                            console.error('Chart container has zero dimensions after ' + maxAttempts + ' attempts');
                        }}
                        return;
                    }}
                    if (typeof LightweightCharts.createChart !== 'function') {{
                        console.error('LightweightCharts.createChart is not a function');
                        return;
                    }}

                    console.log('Container found with dimensions:', chartContainer.offsetWidth, 'x', chartContainer.offsetHeight);
                    chartInstance_{selected_symbol} = LightweightCharts.createChart(chartContainer, {{
                        width: chartContainer.offsetWidth,
                        height: {main_chart_height},
                        layout: {{
                            background: {{ color: '{theme_config['bg']}' }},
                            textColor: '{theme_config['text']}',
                            fontFamily: 'Arial, sans-serif'
                        }},
                        grid: {{
                            vertLines: {{ color: '{theme_config['grid']}', style: LightweightCharts.LineStyle.Dashed }},
                            horzLines: {{ color: '{theme_config['grid']}', style: LightweightCharts.LineStyle.Dashed }}
                        }},
                        crosshair: {{
                            mode: LightweightCharts.CrosshairMode.Normal,
                        }},
                        rightPriceScale: {{
                            borderColor: '{theme_config['grid']}',
                            scaleMargins: {{ top: 0.1, bottom: {scale_margins_bottom} }}
                        }},
                        timeScale: {{
                            timeVisible: true,
                            secondsVisible: false,
                            borderColor: '{theme_config['grid']}'
                        }},
                        watermark: {{
                            text: '{selected_symbol}',
                            color: '{theme_config['watermark']}',
                            fontSize: 48,
                            visible: true
                        }}
                    }});

                    // Set topbar symbol
                    const topbar = document.getElementById('chart_topbar_{selected_symbol}');
                    topbar.innerText = '{selected_symbol}';

                    if (typeof chartInstance_{selected_symbol}.addCandlestickSeries !== 'function') {{
                        console.error('chartInstance_{selected_symbol}.addCandlestickSeries is not a function');
                        return;
                    }}
                    candlestickSeries_{selected_symbol} = chartInstance_{selected_symbol}.addCandlestickSeries({{
                        upColor: '{theme_config['candle_up']}',
                        downColor: '{theme_config['candle_down']}',
                        borderUpColor: '{theme_config['candle_up']}',
                        borderDownColor: '{theme_config['candle_down']}',
                        wickUpColor: '{theme_config['candle_up']}',
                        wickDownColor: '{theme_config['candle_down']}',
                        priceFormat: {{ type: 'price', precision: 2, minMove: 0.01 }}
                    }});
                    candlestickSeries_{selected_symbol}.setData({json.dumps(ohlc_data)});

                    volumeSeries_{selected_symbol} = chartInstance_{selected_symbol}.addHistogramSeries({{
                        color: '{theme_config['volume']}',
                        priceFormat: {{ type: 'volume' }},
                        priceScaleId: 'volume',
                        scaleMargins: {{ top: 0.8, bottom: 0 }}
                    }});
                    volumeSeries_{selected_symbol}.setData({json.dumps(volume_data)});

                    // Create subcharts for RSI and MACD
                    const indicators = {json.dumps(indicators)};
                    let paneIndex = 0;
                    if (indicators.RSI) {{
                        paneIndex++;
                        subcharts_{selected_symbol}.rsi = paneIndex;
                    }}
                    if (indicators.MACD) {{
                        paneIndex++;
                        subcharts_{selected_symbol}.macd = paneIndex;
                        subcharts_{selected_symbol}.macd_hist = paneIndex;
                    }}

                    // Add indicators
                    Object.keys(indicators).forEach(indType => {{
                        indicators[indType].forEach(ind => {{
                            if (ind.enabled) {{
                                if (indType === 'BBANDS') {{
                                    ['upper', 'middle', 'lower'].forEach((band, idx) => {{
                                        const series = chartInstance_{selected_symbol}.addLineSeries({{
                                            color: ind.color,
                                            lineWidth: 1.25,
                                            pane: 0,
                                            priceLineVisible: true
                                        }});
                                        const bandData = ind.data.map(d => ({{ time: d.time, value: d.value }}));
                                        series.setData(bandData);
                                        indicatorSeries_{selected_symbol}[ind.name + '_' + band] = series;
                                    }});
                                }} else if (indType === 'MACD') {{
                                    const pane = subcharts_{selected_symbol}[ind.type === 'histogram' ? 'macd_hist' : 'macd'];
                                    if (ind.type === 'line') {{
                                        const series = chartInstance_{selected_symbol}.addLineSeries({{
                                            color: ind.color,
                                            lineWidth: 1.5,
                                            priceLineVisible: false
                                        }});
                                        series.setData(ind.data);
                                        indicatorSeries_{selected_symbol}[ind.name] = series;
                                    }} else if (ind.type === 'histogram') {{
                                        const series = chartInstance_{selected_symbol}.addHistogramSeries({{
                                            color: ind.color,
                                            priceLineVisible: false
                                        }});
                                        series.setData(ind.data);
                                        indicatorSeries_{selected_symbol}[ind.name] = series;
                                    }}
                                }} else {{
                                    const series = chartInstance_{selected_symbol}.addLineSeries({{
                                        color: ind.color,
                                        lineWidth: 1.5,
                                        pane: ind.pane,
                                        priceLineVisible: ind.pane === 0
                                    }});
                                    series.setData(ind.data);
                                    indicatorSeries_{selected_symbol}[ind.name] = series;
                                }}
                            }}
                        }});
                    }});

                    // Build legend
                    buildLegend();

                    // Add drawings
                    const drawings = {json.dumps(chart_state["drawings"])};
                    chartInstance_{selected_symbol}.subscribeClick(function(param) {{
                        if (!window.drawingMode.active) return;
                        const point = {{ x: param.time, y: param.point.y }};
                        if (!window.drawingMode.start_point) {{
                            window.drawingMode.start_point = point;
                        }} else {{
                            window.drawingMode.end_point = point;
                            const newDrawing = createDrawing(window.drawingMode.tool, window.drawingMode.start_point, window.drawingMode.end_point);
                            drawings.push(newDrawing);
                            window.drawingMode.start_point = null;
                            window.drawingMode.end_point = null;
                            window.drawingMode.active = false;
                            // Save drawings to user_storage
                            fetch('/_nicegui/storage/user/chart_state', {{
                                method: 'POST',
                                body: JSON.stringify({{drawings: drawings, indicators: {json.dumps(chart_state["indicators"])}, templates: {json.dumps(chart_state["templates"])}, current_template: '{chart_state["current_template"]}', watchlist: {json.dumps(chart_state["watchlist"])}}}),
                                headers: {{'Content-Type': 'application/json'}}
                            }}).then(response => response.json()).then(data => {{
                                console.log('Drawings saved:', data);
                            }});
                        }}
                    }});

                    chartInstance_{selected_symbol}.timeScale().fitContent();
                    console.log('Lightweight Charts rendered successfully');
                }} catch (e) {{
                    console.error('Lightweight Charts rendering error:', e);
                }}
            }}

            function createDrawing(tool, startPoint, endPoint) {{
                if (tool === 'Trendline') {{
                    return {{
                        type: 'trendline',
                        start: startPoint,
                        end: endPoint,
                        color: '#FF0000',
                        width: 2
                    }};
                }} else if (tool === 'Horizontal Line') {{
                    return {{
                        type: 'horizontal_line',
                        price: startPoint.y,
                        color: '#00FF00',
                        width: 2
                    }};
                }} else if (tool === 'Rectangle') {{
                    return {{
                        type: 'rectangle',
                        start: startPoint,
                        end: endPoint,
                        color: 'rgba(0, 0, 255, 0.2)',
                        borderColor: '#0000FF',
                        width: 2
                    }};
                }}
                return null;
            }}

            function buildLegend() {{
                const legendContainer = document.getElementById('chart_legend_{selected_symbol}');
                legendContainer.innerHTML = '';
                const lastCandle = {json.dumps(ohlc_data[-1] if ohlc_data else {"close": 0, "open": 0})};
                const colorBasedOnCandle = lastCandle.close >= lastCandle.open ? '{theme_config['candle_up']}' : '{theme_config['candle_down']}'
                const series = [
                    {{ name: 'Candlestick', series: candlestickSeries_{selected_symbol}, color: colorBasedOnCandle, visible: true }},
                    {{ name: 'Volume', series: volumeSeries_{selected_symbol}, color: '{theme_config['volume']}', visible: true }},
                    ...Object.entries(indicatorSeries_{selected_symbol}).map(([name, series]) => ({{ name, series, color: series.options().color, visible: true }}))
                ];
                series.forEach(item => {{
                    const div = document.createElement('div');
                    div.style.display = 'flex';
                    div.style.alignItems = 'center';
                    div.style.marginRight = '10px';
                    div.style.cursor = 'pointer';
                    div.style.fontSize = '20px';
                    div.innerHTML = `
                        <span style="width: 12px; height: 12px; background-color: ${{item.color}}; margin-right: 5px;"></span>
                        <span style="color: '{theme_config['text']}';">${{item.name}}</span>
                    `;
                    div.onclick = () => {{
                        item.visible = !item.visible;
                        item.series.applyOptions({{ visible: item.visible }});
                        div.style.opacity = item.visible ? 1 : 0.5;
                    }};
                    legendContainer.appendChild(div);
                }});
            }}

            function startReplay() {{
                if (!window.isPlaying_{selected_symbol}) {{
                    clearInterval(replayInterval_{selected_symbol});
                    replayInterval_{selected_symbol} = null;
                    candlestickSeries_{selected_symbol}.setData({json.dumps(ohlc_data)});
                    volumeSeries_{selected_symbol}.setData({json.dumps(volume_data)});
                    Object.values(indicatorSeries_{selected_symbol}).forEach(series => {{
                        const ind = Object.values(indicators).flat().find(i => i.name === series.name);
                        series.setData(ind.data);
                    }});
                    chartInstance_{selected_symbol}.timeScale().fitContent();
                    console.log('Replay stopped');
                    return;
                }}
                const data = {json.dumps(ohlc_data)};
                const volumeData = {json.dumps(volume_data)};
                const indicatorData = {json.dumps(indicators)};
                let index = {chart_state["current_replay_index"]};
                candlestickSeries_{selected_symbol}.setData([]);
                volumeSeries_{selected_symbol}.setData([]);
                Object.values(indicatorSeries_{selected_symbol}).forEach(series => series.setData([]));
                replayInterval_{selected_symbol} = setInterval(() => {{
                    if (!window.isPlaying_{selected_symbol} || index >= data.length) {{
                        clearInterval(replayInterval_{selected_symbol});
                        replayInterval_{selected_symbol} = null;
                        console.log('Replay finished');
                        window.isPlaying_{selected_symbol} = false;
                        fetch('/_nicegui/storage/user/chart_state', {{
                            method: 'POST',
                            body: JSON.stringify({{...state, is_playing: false}}),
                            headers: {{'Content-Type': 'application/json'}}
                        }});
                        return;
                    }}
                    candlestickSeries_{selected_symbol}.update(data[index]);
                    volumeSeries_{selected_symbol}.update(volumeData[index]);
                    Object.values(indicatorData).flat().forEach(ind => {{
                        if (indicatorSeries_{selected_symbol}[ind.name] && ind.data[index]) {{
                            indicatorSeries_{selected_symbol}[ind.name].update(ind.data[index]);
                        }}
                    }});
                    chartInstance_{selected_symbol}.timeScale().fitContent();
                    index++;
                    fetch('/_nicegui/storage/user/chart_state', {{
                        method: 'POST',
                        body: JSON.stringify({{...state, current_replay_index: index}}),
                        headers: {{'Content-Type': 'application/json'}}
                    }});
                }}, 500);
                console.log('Replay started');
            }}
            """
            # Execute the JavaScript to initialize the chart
            await ui.run_javascript(chart_init_js)

            # Save chart state
            user_storage["chart_state"] = chart_state

            status_label.text = f"Displaying {selected_symbol} ({selected_timeframe})"
            ui.notify("Chart updated successfully!", type="positive")
        except Exception as e:
            status_label.text = f"Unexpected error: {str(e)}"
            ui.notify(f"Unexpected error: {e}", type="negative")
            logger.exception("Error updating chart")
        finally:
            update_button.props("loading=false disabled=false")
            chart_container.update()
            ui.update()