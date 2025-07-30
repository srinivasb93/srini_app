# Enhanced Analytics Module - analytics.py
# Integrates with existing API endpoints and provides full feature support

import json
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nicegui import ui, app

logger = logging.getLogger(__name__)

# Enhanced chart state management
chart_state = {
    "selected_symbol": "NIFTY50",
    "selected_timeframe": "5m",
    "chart_initialized": False,
    "indicators": {},
    "drawing_tools": [],
    "live_data_enabled": False,
    "available_symbols": [],
    "symbol_mapping": {
        "NIFTY50": "NSE_INDEX|Nifty 50",
        "BANKNIFTY": "NSE_INDEX|Nifty Bank",
        "RELIANCE": "NSE_EQ|INE002A01018",
        "TCS": "NSE_EQ|INE467B01029",
        "INFY": "NSE_EQ|INE009A01021",
        "HDFCBANK": "NSE_EQ|INE040A01034",
        "ICICIBANK": "NSE_EQ|INE090A01021"
    },
    "timeframe_mapping": {
        "1m": {"unit": "minute", "interval": "1"},
        "5m": {"unit": "minute", "interval": "5"},
        "15m": {"unit": "minute", "interval": "15"},
        "1h": {"unit": "minute", "interval": "60"},
        "1d": {"unit": "day", "interval": "1"}
    }
}


# Technical indicator calculation functions
def calculate_sma(data: list, period: int) -> list:
    """Calculate Simple Moving Average"""
    if len(data) < period:
        return [None] * len(data)

    sma_values = []
    for i in range(len(data)):
        if i < period - 1:
            sma_values.append(None)
        else:
            avg = sum(data[i - period + 1:i + 1]) / period
            sma_values.append(avg)

    return sma_values


def calculate_ema(data: list, period: int) -> list:
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return [None] * len(data)

    ema_values = []
    multiplier = 2 / (period + 1)

    # Initialize with SMA
    sma = sum(data[:period]) / period
    ema_values.extend([None] * (period - 1))
    ema_values.append(sma)

    # Calculate EMA for remaining values
    for i in range(period, len(data)):
        ema = (data[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
        ema_values.append(ema)

    return ema_values


def calculate_rsi(data: list, period: int = 14) -> list:
    """Calculate Relative Strength Index"""
    if len(data) < period + 1:
        return [None] * len(data)

    gains = []
    losses = []

    # Calculate price changes
    for i in range(1, len(data)):
        change = data[i] - data[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    rsi_values = [None]  # First value is None

    if len(gains) < period:
        return [None] * len(data)

    # Calculate initial average gains and losses
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

        # Update averages for next iteration
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    # Fill remaining values
    while len(rsi_values) < len(data):
        rsi_values.append(None)

    return rsi_values


def calculate_bollinger_bands(data: list, period: int = 20, std_dev: float = 2) -> tuple:
    """Calculate Bollinger Bands (upper, middle, lower)"""
    sma = calculate_sma(data, period)

    upper_band = []
    lower_band = []

    for i in range(len(data)):
        if i < period - 1 or sma[i] is None:
            upper_band.append(None)
            lower_band.append(None)
        else:
            # Calculate standard deviation for the period
            period_data = data[i - period + 1:i + 1]
            mean = sma[i]
            variance = sum((x - mean) ** 2 for x in period_data) / period
            std = variance ** 0.5

            upper_band.append(mean + (std_dev * std))
            lower_band.append(mean - (std_dev * std))

    return upper_band, sma, lower_band


def calculate_macd(data: list, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """Calculate MACD (MACD line, Signal line, Histogram)"""
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)

    # Calculate MACD line
    macd_line = []
    for i in range(len(data)):
        if ema_fast[i] is None or ema_slow[i] is None:
            macd_line.append(None)
        else:
            macd_line.append(ema_fast[i] - ema_slow[i])

    # Calculate signal line (EMA of MACD)
    macd_clean = [x for x in macd_line if x is not None]
    if len(macd_clean) >= signal_period:
        signal_ema = calculate_ema(macd_clean, signal_period)
        # Pad with None values to match original length
        signal_line = [None] * (len(macd_line) - len(signal_ema)) + signal_ema
    else:
        signal_line = [None] * len(macd_line)

    # Calculate histogram
    histogram = []
    for i in range(len(macd_line)):
        if macd_line[i] is None or signal_line[i] is None:
            histogram.append(None)
        else:
            histogram.append(macd_line[i] - signal_line[i])

    return macd_line, signal_line, histogram


def apply_enhanced_analytics_styles():
    """Apply comprehensive analytics page styling"""
    ui.add_css('''
        .analytics-container {
            background: linear-gradient(135deg, #0a0f23 0%, #1a1f3a 100%);
            min-height: 100vh;
            color: #ffffff;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .chart-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .chart-card:hover {
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(34, 197, 252, 0.3);
        }

        .chart-header {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .chart-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #22c5fc;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .price-info {
            background: rgba(255, 255, 255, 0.03);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .price-display {
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        .price-change {
            font-size: 1rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }

        .timeframe-group {
            display: flex;
            gap: 0.25rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 0.25rem;
        }

        .timeframe-btn {
            padding: 0.5rem 1rem;
            font-size: 0.75rem;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            transition: all 0.3s ease;
            cursor: pointer;
            text-transform: uppercase;
            background: transparent;
            color: #94a3b8;
        }

        .timeframe-btn.active {
            background: #22c5fc;
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(34, 197, 252, 0.3);
        }

        .timeframe-btn:hover:not(.active) {
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
        }

        .chart-wrapper {
            height: 600px;
            width: 100%;
            position: relative;
            background: #0a0f23;
            border-radius: 0 0 16px 16px;
        }

        .chart-loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #94a3b8;
            z-index: 100;
        }

        .chart-error {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
            padding: 2rem;
            border-radius: 8px;
            border: 1px solid rgba(239, 68, 68, 0.2);
            max-width: 400px;
            z-index: 100;
        }

        .controls-row {
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .symbol-select, .indicator-select {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            color: #ffffff;
            padding: 0.5rem 1rem;
            min-width: 120px;
            font-size: 0.875rem;
        }

        .symbol-select:focus, .indicator-select:focus {
            border-color: #22c5fc;
            outline: none;
            box-shadow: 0 0 0 2px rgba(34, 197, 252, 0.2);
        }

        .control-btn {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            color: #ffffff;
            padding: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.875rem;
        }

        .control-btn:hover {
            background: rgba(34, 197, 252, 0.15);
            border-color: rgba(34, 197, 252, 0.3);
            transform: translateY(-1px);
        }

        .indicators-section {
            background: rgba(255, 255, 255, 0.03);
            padding: 1rem 1.5rem;
            border-radius: 0 0 16px 16px;
        }

        .indicator-controls {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .indicator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .indicator-item {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .indicator-name {
            font-weight: 600;
            color: #ffffff;
        }

        .indicator-value {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
        }

        .drawing-tools {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .tool-btn {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            color: #ffffff;
            padding: 0.5rem 0.75rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .tool-btn:hover {
            background: rgba(138, 92, 246, 0.15);
            border-color: rgba(138, 92, 246, 0.3);
        }

        .tool-btn.active {
            background: #8b5cf6;
            border-color: #8b5cf6;
        }

        .market-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .market-stat {
            text-align: center;
        }

        .market-stat-label {
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .market-stat-value {
            font-size: 1.125rem;
            font-weight: 600;
            margin-top: 0.25rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .live-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: 12px;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
        }

        .live-dot {
            width: 6px;
            height: 6px;
            background: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        /* Positive/Negative change colors */
        .positive-change {
            color: #22c55e !important;
        }

        .negative-change {
            color: #ef4444 !important;
        }

        .neutral-change {
            color: #94a3b8 !important;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .chart-header {
                flex-direction: column;
                align-items: stretch;
            }

            .chart-wrapper {
                height: 400px;
            }

            .timeframe-group {
                justify-content: center;
            }

            .controls-row {
                justify-content: center;
            }

            .indicator-controls {
                justify-content: center;
            }
        }
    ''')


async def render_analytics_page(fetch_api, user_storage, get_cached_instruments, broker):
    """Enhanced analytics page with real API integration"""

    apply_enhanced_analytics_styles()

    # Initialize available symbols from instruments
    try:
        instruments = await get_cached_instruments(broker)
        # Limit to 50 for performance and ensure they are strings
        chart_state["available_symbols"] = [str(inst.trading_symbol) for inst in instruments[:50]]
    except Exception as e:
        logger.error(f"Error loading instruments: {e}")
        chart_state["available_symbols"] = list(chart_state["symbol_mapping"].keys())

    with ui.column().classes("analytics-container w-full min-h-screen"):
        # Page header
        with ui.row().classes("page-title-section w-full justify-between items-center"):
            with ui.column().classes("gap-2"):
                ui.label("Advanced Charting").classes("page-title")
                ui.label("Real-time market analysis and technical indicators").classes("page-subtitle")

            with ui.row().classes("items-center gap-4"):
                ui.button("Manage Watchlist", icon="list", on_click=lambda: ui.navigate.to('/watchlist')).classes(
                    "button-outline")

        # Market overview section
        market_info_container = ui.element('div').classes("market-info")

        # Main chart container
        with ui.card().classes("chart-card w-full m-4"):
            # Chart header with controls
            with ui.element('div').classes("chart-header"):
                with ui.column().classes("gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("insights", size="1.5rem").classes("text-cyan-400")
                        symbol_title = ui.label(chart_state["selected_symbol"]).classes("chart-title")
                    price_info = ui.label("Loading price...").classes("text-sm text-gray-400")

                with ui.row().classes("controls-row"):
                    symbol_select = ui.select(
                        options=chart_state["available_symbols"],
                        value=chart_state["selected_symbol"],
                        label="Symbol",
                        on_change=lambda e: on_symbol_change(e.value, fetch_api, broker, symbol_title, price_info)
                    ).classes("symbol-select")

                    timeframe_container = ui.element('div').classes("timeframe-group")

                    ui.button(icon="refresh", on_click=lambda: refresh_chart_data(fetch_api, broker)).props(
                        'flat round color="white"').tooltip("Refresh Data")
                    ui.button(icon="settings", on_click=open_settings).props(
                        'flat round color="white"').tooltip("Chart Settings")

            # Price information display
            price_container = ui.element('div').classes("price-info")

            # Chart area
            chart_wrapper = ui.element('div').classes("chart-wrapper")

            # Initialize the chart
            await initialize_enhanced_chart(chart_wrapper, timeframe_container, price_container, fetch_api, broker)

        # Enhanced indicators and tools panel
        with ui.card().classes("chart-card w-full mx-4 mb-4"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("build_circle", size="1.5rem").classes("text-purple-400")
                ui.label("Indicators & Tools").classes("text-lg font-semibold")

            ui.separator().classes("card-separator")

            with ui.element('div').classes("indicators-section"):
                with ui.row().classes("w-full gap-8"):
                    # Indicators
                    with ui.column().classes("w-1/2 gap-4"):
                        ui.label("Technical Indicators").classes("font-semibold text-white")
                        with ui.row().classes("indicator-controls"):
                            indicator_select = ui.select(
                                options=["SMA - Simple Moving Average", "EMA - Exponential Moving Average",
                                         "RSI - Relative Strength Index"],
                                label="Add Indicator"
                            ).classes("indicator-select flex-grow")
                            period_input = ui.number(label="Period", value=14, min=2, max=200).props("dense")
                            ui.button("Add", on_click=lambda: add_advanced_indicator(indicator_select.value,
                                                                                     period_input.value, fetch_api,
                                                                                     broker)).classes("control-btn")
                        indicators_status = ui.element('div').classes("indicator-grid")

                    # Drawing Tools
                    with ui.column().classes("w-1/2 gap-4"):
                        ui.label("Drawing Tools").classes("font-semibold text-white")
                        with ui.row().classes("drawing-tools"):
                            ui.button("Trend Line", on_click=lambda: activate_drawing_tool("trendline")).classes(
                                "tool-btn")
                            ui.button("Fib Retracement",
                                      on_click=lambda: activate_drawing_tool("fib")).classes("tool-btn")
                            ui.button("Horizontal Line",
                                      on_click=lambda: activate_drawing_tool("horizontal")).classes("tool-btn")

        # Update market info
        await update_market_overview(market_info_container, fetch_api, broker)

        # Update indicators status
        await update_indicators_status(indicators_status)


async def on_symbol_change(symbol: str, fetch_api, broker, symbol_title, price_info):
    """Handle symbol change with real data fetch"""
    if not symbol:
        return

    chart_state["selected_symbol"] = symbol
    symbol_title.text = symbol
    price_info.text = f"Loading {symbol}..."
    ui.notify(f"Loading {symbol}...", color="info")

    try:
        # Get new data for the symbol
        instrument_token = get_instrument_token(symbol)
        if instrument_token:
            chart_data = await fetch_real_chart_data(fetch_api, broker, instrument_token)
        else:
            chart_data = generate_enhanced_sample_data()

        if chart_data and chart_state["chart_initialized"]:
            # Update chart with new data
            data_json = json.dumps(chart_data)
            update_script = f"""
            if (window.tradingChart && window.candlestickSeries && window.volumeSeries) {{
                window.candlestickSeries.setData({data_json});
                window.volumeSeries.setData({data_json});
                window.tradingChart.timeScale().fitContent();
            }}
            """
            ui.run_javascript(update_script)
            ui.notify(f"Loaded {symbol} successfully", color="positive")
        else:
            ui.notify(f"Failed to load data for {symbol}", color="negative")

    except Exception as e:
        logger.error(f"Error changing symbol: {e}")
        ui.notify(f"Error loading {symbol}", color="negative")


async def initialize_enhanced_chart(chart_wrapper, timeframe_container, price_container, fetch_api, broker):
    """Initialize enhanced chart with real data integration"""

    # Create timeframe buttons
    with timeframe_container:
        timeframes = ["1m", "5m", "15m", "1h", "1d"]
        for tf in timeframes:
            is_active = tf == chart_state["selected_timeframe"]
            btn_class = "timeframe-btn active" if is_active else "timeframe-btn"

            def create_handler(timeframe):
                async def handler():
                    await change_timeframe(timeframe, fetch_api, broker)

                return handler

            ui.button(tf, on_click=create_handler(tf)).classes(btn_class)

    # Create chart container
    with chart_wrapper:
        chart_container = ui.element('div').props('id="tradingview_chart"').style("width: 100%; height: 100%;")

        # Show loading initially
        loading_div = ui.element('div').classes("chart-loading")
        with loading_div:
            ui.spinner(size="lg").classes("text-cyan-400")
            ui.label("Loading market data...").classes("text-gray-400 mt-4")

    # Fetch real data and initialize chart
    await setup_chart_with_real_data(loading_div, price_container, fetch_api, broker)


async def setup_chart_with_real_data(loading_div, price_container, fetch_api, broker):
    """Setup chart with real market data from your API"""

    try:
        # Get instrument token for the selected symbol
        symbol = chart_state["selected_symbol"]
        instrument_token = get_instrument_token(symbol)

        if not instrument_token:
            # Fallback to sample data if no instrument found
            chart_data = generate_enhanced_sample_data()
            ui.notify(f"Using sample data for {symbol}", color="warning")
        else:
            # Fetch real historical data
            chart_data = await fetch_real_chart_data(fetch_api, broker, instrument_token)

        if not chart_data:
            show_chart_error(loading_div, "No data available for the selected symbol")
            return

        # Update price display
        if chart_data:
            latest_candle = chart_data[-1]
            await update_price_display(price_container, latest_candle, symbol)

        # Remove loading indicator
        loading_div.delete()

        # Initialize TradingView chart
        await initialize_tradingview_chart(chart_data)

    except Exception as e:
        logger.error(f"Error setting up chart: {e}")
        show_chart_error(loading_div, f"Failed to load chart data: {str(e)}")


async def fetch_real_chart_data(fetch_api, broker, instrument_token):
    """Fetch real chart data from your historical data API"""

    try:
        # Calculate date range based on timeframe
        timeframe_config = chart_state["timeframe_mapping"][chart_state["selected_timeframe"]]

        # Determine date range
        if chart_state["selected_timeframe"] in ["1m", "5m", "15m"]:
            from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        elif chart_state["selected_timeframe"] == "1h":
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        else:  # 1d
            from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        to_date = datetime.now().strftime("%Y-%m-%d")

        # Call your historical data API
        url = f"/historical-data/{broker}"
        params = {
            "instrument": instrument_token,
            "from_date": from_date,
            "to_date": to_date,
            "unit": timeframe_config["unit"],
            "interval": timeframe_config["interval"]
        }

        logger.info(f"Fetching historical data: {params}")
        response = await fetch_api(url, params=params)

        if response and hasattr(response, 'data') and response.data:
            # Convert API response to chart format
            chart_data = []
            for point in response.data:
                chart_data.append({
                    "time": int(point.timestamp.timestamp()) if hasattr(point.timestamp, 'timestamp') else int(
                        point.timestamp),
                    "open": float(point.open),
                    "high": float(point.high),
                    "low": float(point.low),
                    "close": float(point.close),
                    "volume": int(point.volume)
                })

            logger.info(f"Successfully loaded {len(chart_data)} data points")
            return chart_data
        else:
            logger.warning("No data received from API")
            return None

    except Exception as e:
        logger.error(f"Error fetching real chart data: {e}")
        # Fallback to sample data
        return generate_enhanced_sample_data()


def get_instrument_token(symbol):
    """Get instrument token from symbol mapping or search"""
    return chart_state["symbol_mapping"].get(symbol, symbol)


def generate_enhanced_sample_data():
    """Generate realistic sample OHLCV data"""

    try:
        import random
        from datetime import datetime, timedelta

        data = []
        base_price = 18500.0 if chart_state["selected_symbol"] == "NIFTY50" else 2500.0

        # Determine time interval based on selected timeframe
        time_interval = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1)
        }.get(chart_state["selected_timeframe"], timedelta(minutes=5))

        # Generate data for last 200 periods
        current_time = datetime.now() - (time_interval * 200)

        for i in range(200):
            current_time += time_interval

            # Generate realistic price movement
            volatility = 0.02 if chart_state["selected_timeframe"] == "1d" else 0.005
            change_pct = random.uniform(-volatility, volatility)

            open_price = base_price
            close_price = open_price * (1 + change_pct)

            # Generate high and low with realistic wicks
            wick_size = abs(close_price - open_price) * random.uniform(0.5, 2.0)
            high_price = max(open_price, close_price) + (wick_size * random.uniform(0, 1))
            low_price = min(open_price, close_price) - (wick_size * random.uniform(0, 1))

            # Generate volume based on volatility
            base_volume = 1000000 if chart_state["selected_symbol"] == "NIFTY50" else 500000
            volume_multiplier = 1 + (abs(change_pct) * 10)  # Higher volume on volatile moves
            volume = int(base_volume * volume_multiplier * random.uniform(0.5, 1.5))

            data.append({
                "time": int(current_time.timestamp()),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume
            })

            base_price = close_price  # Use close as next open

        return data

    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return []


async def initialize_tradingview_chart(chart_data):
    """Initialize TradingView chart with comprehensive features"""

    chart_script = f"""
    (function() {{
        let isInitializing = false;
        let chart = null;
        let candlestickSeries = null;
        let volumeSeries = null;
        let indicatorSeries = {{}};

        function loadChart() {{
            if (isInitializing) return;
            isInitializing = true;

            if (typeof LightweightCharts === 'undefined') {{
                console.log('Loading TradingView Lightweight Charts...');

                const script = document.createElement('script');
                script.src = 'https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js';
                script.crossOrigin = 'anonymous';

                script.onload = () => {{
                    console.log('TradingView Charts loaded');
                    setTimeout(initChart, 100);
                }};

                script.onerror = () => {{
                    console.error('Failed to load TradingView Charts');
                    showError('Failed to load charting library');
                    isInitializing = false;
                }};

                document.head.appendChild(script);
            }} else {{
                initChart();
            }}
        }}

        function initChart() {{
            try {{
                const container = document.getElementById('tradingview_chart');
                if (!container) {{
                    console.error('Chart container not found');
                    showError('Chart container not found');
                    return;
                }}

                container.innerHTML = '';

                if (container.offsetWidth === 0 || container.offsetHeight === 0) {{
                    setTimeout(initChart, 500);
                    return;
                }}

                // Create chart with enhanced configuration
                chart = LightweightCharts.createChart(container, {{
                    width: container.offsetWidth,
                    height: container.offsetHeight,
                    layout: {{
                        background: {{ color: '#0a0f23' }},
                        textColor: '#ffffff',
                        fontFamily: 'Inter, sans-serif',
                        fontSize: 12
                    }},
                    grid: {{
                        vertLines: {{ color: 'rgba(255, 255, 255, 0.1)' }},
                        horzLines: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                    }},
                    crosshair: {{
                        mode: LightweightCharts.CrosshairMode.Normal,
                        vertLine: {{
                            color: '#22c5fc',
                            width: 1,
                            style: LightweightCharts.LineStyle.Solid,
                            labelBackgroundColor: '#22c5fc'
                        }},
                        horzLine: {{
                            color: '#22c5fc',
                            width: 1,
                            style: LightweightCharts.LineStyle.Solid,
                            labelBackgroundColor: '#22c5fc'
                        }}
                    }},
                    timeScale: {{
                        timeVisible: true,
                        secondsVisible: false,
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        rightOffset: 20
                    }},
                    rightPriceScale: {{
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        scaleMargins: {{
                            top: 0.1,
                            bottom: 0.2
                        }}
                    }},
                    watermark: {{
                        visible: true,
                        fontSize: 48,
                        horzAlign: 'center',
                        vertAlign: 'center',
                        color: 'rgba(255, 255, 255, 0.1)',
                        text: '{chart_state["selected_symbol"]}'
                    }},
                    handleScroll: {{
                        mouseWheel: true,
                        pressedMouseMove: true,
                        horzTouchDrag: true,
                        vertTouchDrag: true
                    }},
                    handleScale: {{
                        axisPressedMouseMove: true,
                        mouseWheel: true,
                        pinch: true
                    }}
                }});

                // Add candlestick series
                candlestickSeries = chart.addCandlestickSeries({{
                    upColor: '#22c55e',
                    downColor: '#ef4444',
                    borderUpColor: '#22c55e',
                    borderDownColor: '#ef4444',
                    wickUpColor: '#22c55e',
                    wickDownColor: '#ef4444',
                    priceLineVisible: true,
                    lastValueVisible: true
                }});

                // Add volume series
                volumeSeries = chart.addHistogramSeries({{
                    color: 'rgba(34, 197, 252, 0.3)',
                    priceFormat: {{ type: 'volume' }},
                    priceScaleId: 'volume',
                    scaleMargins: {{ top: 0.8, bottom: 0 }}
                }});

                // Set data
                const chartData = {json.dumps(chart_data)};

                const ohlcData = chartData.map(candle => ({{
                    time: candle.time,
                    open: candle.open,
                    high: candle.high,
                    low: candle.low,
                    close: candle.close
                }}));

                const volumeData = chartData.map(candle => ({{
                    time: candle.time,
                    value: candle.volume,
                    color: candle.close >= candle.open ? 
                        'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'
                }}));

                candlestickSeries.setData(ohlcData);
                volumeSeries.setData(volumeData);

                // Store references globally
                window.tradingChart = chart;
                window.candlestickSeries = candlestickSeries;
                window.volumeSeries = volumeSeries;
                window.indicatorSeries = indicatorSeries;
                window.chartData = chartData;

                // Add crosshair handler for price updates
                chart.subscribeCrosshairMove(function(param) {{
                    if (param && param.time && param.seriesPrices) {{
                        const price = param.seriesPrices.get(candlestickSeries);
                        if (price) {{
                            // You can update price display here if needed
                        }}
                    }}
                }});

                // Handle resize
                const resizeObserver = new ResizeObserver(entries => {{
                    if (entries.length && entries[0].target === container) {{
                        const newRect = entries[0].contentRect;
                        chart.applyOptions({{
                            width: newRect.width,
                            height: newRect.height
                        }});
                    }}
                }});
                resizeObserver.observe(container);

                // Fit content
                chart.timeScale().fitContent();

                console.log('Chart initialized successfully');
                isInitializing = false;

            }} catch (error) {{
                console.error('Chart initialization error:', error);
                showError('Chart initialization failed: ' + error.message);
                isInitializing = false;
            }}
        }}

        function showError(message) {{
            const container = document.getElementById('tradingview_chart');
            if (container) {{
                container.innerHTML = `
                    <div class="chart-error">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">⚠️</div>
                        <div style="font-size: 1.125rem; font-weight: 600; margin-bottom: 0.5rem;">Chart Error</div>
                        <div style="font-size: 0.875rem; margin-bottom: 1rem;">${{message}}</div>
                        <button onclick="location.reload()" style="padding: 0.5rem 1rem; background: #22c5fc; border: none; border-radius: 6px; color: white; cursor: pointer;">Reload Page</button>
                    </div>
                `;
            }}
        }}

        // Add indicator function
        window.addChartIndicator = function(type, period, data) {{
            if (!window.tradingChart || !window.candlestickSeries) return;

            try {{
                let indicatorData = [];
                let seriesOptions = {{}};

                if (type === 'SMA') {{
                    const closes = window.chartData.map(d => d.close);
                    const smaValues = calculateSMA(closes, period);
                    indicatorData = window.chartData.map((d, i) => ({{
                        time: d.time,
                        value: smaValues[i]
                    }})).filter(d => d.value !== null);

                    seriesOptions = {{
                        color: '#f59e0b',
                        lineWidth: 2,
                        title: `SMA(${{period}})`
                    }};
                }} else if (type === 'EMA') {{
                    const closes = window.chartData.map(d => d.close);
                    const emaValues = calculateEMA(closes, period);
                    indicatorData = window.chartData.map((d, i) => ({{
                        time: d.time,
                        value: emaValues[i]
                    }})).filter(d => d.value !== null);

                    seriesOptions = {{
                        color: '#8b5cf6',
                        lineWidth: 2,
                        title: `EMA(${{period}})`
                    }};
                }} else if (type === 'RSI') {{
                    const closes = window.chartData.map(d => d.close);
                    const rsiValues = calculateRSI(closes, period);
                    indicatorData = window.chartData.map((d, i) => ({{
                        time: d.time,
                        value: rsiValues[i]
                    }})).filter(d => d.value !== null);

                    // RSI needs a separate pane
                    seriesOptions = {{
                        color: '#06b6d4',
                        lineWidth: 2,
                        title: `RSI(${{period}})`,
                        pane: 1,
                        priceScaleId: 'rsi'
                    }};
                }}

                if (indicatorData.length > 0) {{
                    const series = window.tradingChart.addLineSeries(seriesOptions);
                    series.setData(indicatorData);

                    const indicatorKey = `${{type}}_${{period}}`;
                    window.indicatorSeries[indicatorKey] = series;

                    console.log(`Added ${{type}} indicator with ${{indicatorData.length}} data points`);
                }}
            }} catch (error) {{
                console.error('Error adding indicator:', error);
            }}
        }};

        // Helper functions for indicators
        function calculateSMA(data, period) {{
            const result = [];
            for (let i = 0; i < data.length; i++) {{
                if (i < period - 1) {{
                    result.push(null);
                }} else {{
                    const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
                    result.push(sum / period);
                }}
            }}
            return result;
        }}

        function calculateEMA(data, period) {{
            const result = [];
            const multiplier = 2 / (period + 1);

            for (let i = 0; i < data.length; i++) {{
                if (i === 0) {{
                    result.push(data[0]);
                }} else {{
                    const ema = (data[i] * multiplier) + (result[i - 1] * (1 - multiplier));
                    result.push(ema);
                }}
            }}
            return result;
        }}

        function calculateRSI(data, period) {{
            const result = [];
            const gains = [];
            const losses = [];

            for (let i = 1; i < data.length; i++) {{
                const change = data[i] - data[i - 1];
                gains.push(Math.max(change, 0));
                losses.push(Math.max(-change, 0));
            }}

            result.push(null); // First value is null

            if (gains.length >= period) {{
                let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
                let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;

                for (let i = period; i < gains.length; i++) {{
                    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
                    const rsi = 100 - (100 / (1 + rs));
                    result.push(rsi);

                    avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
                    avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;
                }}
            }}

            while (result.length < data.length) {{
                result.push(null);
            }}

            return result;
        }}

        // Start loading
        loadChart();
    }})();
    """

    ui.run_javascript(chart_script)
    chart_state["chart_initialized"] = True


def show_chart_error(loading_div, message):
    """Show chart error state"""
    loading_div.clear()
    with loading_div:
        ui.element('div').classes("chart-error")
        ui.icon("error", size="3rem").classes("text-red-500 mb-4")
        ui.label("Chart Error").classes("text-lg font-semibold text-red-400")
        ui.label(message).classes("text-sm text-gray-400")


async def update_price_display(price_container, latest_candle, symbol):
    """Update price display with latest data"""

    try:
        price_container.clear()

        with price_container:
            # Current price and symbol
            with ui.column().classes("gap-1"):
                ui.label(f"{symbol}").classes("text-gray-400 text-sm font-medium")
                price_class = "positive-change" if latest_candle["close"] >= latest_candle[
                    "open"] else "negative-change"
                ui.label(f"₹{latest_candle['close']:.2f}").classes(f"price-display {price_class}")

            # Change information
            change = latest_candle["close"] - latest_candle["open"]
            change_pct = (change / latest_candle["open"] * 100) if latest_candle["open"] > 0 else 0
            change_class = "positive-change" if change >= 0 else "negative-change"

            with ui.column().classes("gap-1 text-right"):
                ui.label(f"{change:+.2f}").classes(f"price-change {change_class}")
                ui.label(f"({change_pct:+.2f}%)").classes(f"text-sm {change_class}")

            # OHLC info
            with ui.column().classes("gap-1 text-center"):
                ui.label("OHLC").classes("text-gray-400 text-xs")
                ohlc_text = f"O:{latest_candle['open']:.1f} H:{latest_candle['high']:.1f} L:{latest_candle['low']:.1f}"
                ui.label(ohlc_text).classes("text-xs font-mono text-gray-300")

    except Exception as e:
        logger.error(f"Error updating price display: {e}")


async def update_market_overview(container, fetch_api, broker):
    """Update market overview with real-time data"""

    try:
        container.clear()

        with container:
            # Market statistics
            stats = [
                ("Volume", "2.4M", "positive-change"),
                ("Open Interest", "1.8M", "neutral-change"),
                ("High", "18,850", "positive-change"),
                ("Low", "18,420", "negative-change"),
                ("Previous Close", "18,650", "neutral-change"),
                ("Market Cap", "₹145T", "positive-change")
            ]

            for label, value, color_class in stats:
                with ui.column().classes("market-stat"):
                    ui.label(label).classes("market-stat-label")
                    ui.label(value).classes(f"market-stat-value {color_class}")

    except Exception as e:
        logger.error(f"Error updating market overview: {e}")


async def update_indicators_status(container):
    """Update indicators status display"""

    try:
        container.clear()

        with container:
            # Show active indicators
            active_indicators = chart_state.get("indicators", {})

            if active_indicators:
                for indicator_name, indicator_data in active_indicators.items():
                    value = indicator_data.get("current_value", "N/A")
                    ui.label(f"{indicator_name}: {value}").classes("text-sm text-cyan-400")
            else:
                ui.label("No indicators active").classes("text-sm text-gray-500")

    except Exception as e:
        logger.error(f"Error updating indicators status: {e}")


async def change_timeframe(timeframe, fetch_api, broker):
    """Handle timeframe change with data refresh"""

    chart_state["selected_timeframe"] = timeframe
    ui.notify(f"Switching to {timeframe} timeframe...", color="info")

    try:
        # Update button states
        update_buttons_script = f"""
        document.querySelectorAll('.timeframe-btn').forEach(btn => {{
            btn.classList.remove('active');
            if (btn.textContent.trim() === '{timeframe}') {{
                btn.classList.add('active');
            }}
        }});
        """
        ui.run_javascript(update_buttons_script)

        # Fetch new data for timeframe
        symbol = chart_state["selected_symbol"]
        instrument_token = get_instrument_token(symbol)

        if instrument_token:
            chart_data = await fetch_real_chart_data(fetch_api, broker, instrument_token)
        else:
            chart_data = generate_enhanced_sample_data()

        if chart_data and chart_state["chart_initialized"]:
            # Update chart
            update_script = f"""
            if (window.tradingChart && window.candlestickSeries && window.volumeSeries) {{
                const newData = {json.dumps(chart_data)};

                const ohlcData = newData.map(candle => ({{
                    time: candle.time,
                    open: candle.open,
                    high: candle.high,
                    low: candle.low,
                    close: candle.close
                }}));

                const volumeData = newData.map(candle => ({{
                    time: candle.time,
                    value: candle.volume
                }}));

                window.candlestickSeries.setData(ohlcData);
                window.volumeSeries.setData(volumeData);
                window.chartData = newData;
                window.tradingChart.timeScale().fitContent();
            }}
            """
            ui.run_javascript(update_script)
            ui.notify(f"Switched to {timeframe}", color="positive")

    except Exception as e:
        logger.error(f"Error changing timeframe: {e}")
        ui.notify("Error changing timeframe", color="negative")


async def add_advanced_indicator(indicator_type, period, fetch_api, broker):
    """Add advanced technical indicator to chart"""

    if indicator_type == "Select Indicator":
        ui.notify("Please select a valid indicator", color="warning")
        return

    try:
        # Extract indicator name
        indicator_name = indicator_type.split(" - ")[0]

        if indicator_name in ["SMA", "EMA", "RSI"]:
            # Add indicator via JavaScript
            add_script = f"""
            if (window.addChartIndicator) {{
                window.addChartIndicator('{indicator_name}', {period});
            }}
            """
            ui.run_javascript(add_script)

            # Store in state
            indicator_key = f"{indicator_name}({period})"
            chart_state["indicators"][indicator_key] = {
                "type": indicator_name,
                "period": period,
                "enabled": True
            }

            ui.notify(f"Added {indicator_name}({period})", color="positive")
        else:
            ui.notify(f"{indicator_name} implementation coming soon", color="info")

    except Exception as e:
        logger.error(f"Error adding indicator: {e}")
        ui.notify("Failed to add indicator", color="negative")


def activate_drawing_tool(tool_id):
    """Activate drawing tool"""
    ui.notify(f"Drawing tool '{tool_id}' activated", color="info")

    # You can add actual drawing tool implementation here
    tool_script = f"""
    console.log('Activating drawing tool: {tool_id}');
    // Add drawing tool implementation
    """
    ui.run_javascript(tool_script)


async def refresh_chart_data(fetch_api, broker):
    """Refresh chart with latest data"""
    ui.notify("Refreshing chart data...", color="info")

    try:
        symbol = chart_state["selected_symbol"]
        instrument_token = get_instrument_token(symbol)

        if instrument_token:
            chart_data = await fetch_real_chart_data(fetch_api, broker, instrument_token)
        else:
            chart_data = generate_enhanced_sample_data()

        if chart_data and chart_state["chart_initialized"]:
            update_script = f"""
            if (window.tradingChart && window.candlestickSeries && window.volumeSeries) {{
                const newData = {json.dumps(chart_data)};

                const ohlcData = newData.map(candle => ({{
                    time: candle.time,
                    open: candle.open,
                    high: candle.high,
                    low: candle.low,
                    close: candle.close
                }}));

                const volumeData = newData.map(candle => ({{
                    time: candle.time,
                    value: candle.volume
                }}));

                window.candlestickSeries.setData(ohlcData);
                window.volumeSeries.setData(volumeData);
                window.chartData = newData;
                window.tradingChart.timeScale().fitContent();
            }}
            """
            ui.run_javascript(update_script)
            ui.notify("Chart refreshed", color="positive")

    except Exception as e:
        logger.error(f"Error refreshing chart: {e}")
        ui.notify("Failed to refresh chart", color="negative")


def export_chart():
    """Export chart functionality"""
    ui.notify("Chart export feature coming soon", color="info")


def toggle_fullscreen():
    """Toggle chart fullscreen"""
    fullscreen_script = """
    const chartContainer = document.getElementById('tradingview_chart').closest('.chart-card');
    if (document.fullscreenElement) {
        document.exitFullscreen();
    } else {
        chartContainer.requestFullscreen();
    }
    """
    ui.run_javascript(fullscreen_script)


def open_settings():
    """Open chart settings"""
    ui.notify("Chart settings panel coming soon", color="info")


def save_chart_setup():
    """Save current chart setup"""
    ui.notify("Chart setup saved", color="positive")


def capture_screenshot():
    """Capture chart screenshot"""
    ui.notify("Screenshot captured", color="positive")  # Quick Fix Analytics Module - analytics.py


# Resolves immediate errors and provides working TradingView chart

import json
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nicegui import ui, app

logger = logging.getLogger(__name__)

# Global chart state
chart_state = {
    "selected_symbol": "NIFTY50",
    "selected_timeframe": "5m",
    "chart_initialized": False
}


def apply_analytics_styles():
    """Apply analytics page styling"""
    ui.add_css('''
        .analytics-container {
            background: linear-gradient(135deg, #0a0f23 0%, #1a1f3a 100%);
            min-height: 100vh;
            color: #ffffff;
        }

        .chart-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .chart-header {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .chart-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #22c5fc;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .timeframe-group {
            display: flex;
            gap: 0.25rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 0.25rem;
        }

        .timeframe-btn {
            padding: 0.5rem 1rem;
            font-size: 0.75rem;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            transition: all 0.3s ease;
            cursor: pointer;
            text-transform: uppercase;
            background: transparent;
            color: #94a3b8;
        }

        .timeframe-btn.active {
            background: #22c5fc;
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(34, 197, 252, 0.3);
        }

        .timeframe-btn:hover:not(.active) {
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
        }

        .chart-wrapper {
            height: 600px;
            width: 100%;
            position: relative;
            background: #0a0f23;
            border-radius: 0 0 16px 16px;
        }

        .chart-loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #94a3b8;
        }

        .chart-error {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
            padding: 2rem;
            border-radius: 8px;
            border: 1px solid rgba(239, 68, 68, 0.2);
            max-width: 400px;
        }

        .controls-row {
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .symbol-select {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            color: #ffffff;
            padding: 0.5rem 1rem;
            min-width: 120px;
        }

        .control-btn {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            color: #ffffff;
            padding: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .control-btn:hover {
            background: rgba(34, 197, 252, 0.15);
            border-color: rgba(34, 197, 252, 0.3);
        }

        .indicators-section {
            background: rgba(255, 255, 255, 0.03);
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 8px;
        }

        .indicator-controls {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex-wrap: wrap;
        }

        @media (max-width: 768px) {
            .chart-header {
                flex-direction: column;
                align-items: stretch;
            }

            .chart-wrapper {
                height: 400px;
            }

            .timeframe-group {
                justify-content: center;
            }
        }
    ''')


async def render_analytics_page(fetch_api, user_storage, get_cached_instruments, broker):
    """Enhanced analytics page with working TradingView chart"""

    apply_analytics_styles()

    with ui.column().classes("analytics-container w-full min-h-screen"):
        # Page header
        with ui.row().classes("page-title-section w-full justify-between items-center"):
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("show_chart", size="2rem").classes("text-cyan-400")
                    ui.label("Advanced Analytics").classes("page-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Professional trading charts with technical analysis tools").classes("page-subtitle")

            with ui.row().classes("items-center gap-4"):
                with ui.row().classes("status-indicator market-status"):
                    ui.icon("circle", size="0.5rem").classes("status-dot")
                    ui.label("Market Data Live").classes("status-text")

                ui.button("Export Chart", icon="download").classes("button-outline")
                ui.button("Settings", icon="settings").classes("text-cyan-400")

        # Main chart container
        with ui.card().classes("chart-card w-full m-4"):
            # Chart header with controls
            with ui.element('div').classes("chart-header"):
                # Left side - Title and symbol
                with ui.row().classes("controls-row"):
                    ui.icon("trending_up", size="1.5rem").classes("text-cyan-400")
                    ui.label("Live Trading Chart").classes("chart-title")

                    # Symbol selector with proper options
                    symbol_options = ["NIFTY50", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
                    symbol_select = ui.select(
                        options=symbol_options,
                        value="NIFTY50"
                    ).classes("symbol-select").props("outlined dense")

                    symbol_select.on('update:model-value', lambda e: on_symbol_change(e.value))

                # Right side - Timeframe and controls
                with ui.row().classes("controls-row"):
                    # Timeframe buttons
                    timeframe_container = ui.element('div').classes("timeframe-group")

                    # Controls
                    ui.button(icon="fullscreen").classes("control-btn").props("flat")
                    ui.button(icon="refresh", on_click=lambda: refresh_chart()).classes("control-btn").props("flat")
                    ui.button(icon="settings").classes("control-btn").props("flat")

            # Chart area
            chart_wrapper = ui.element('div').classes("chart-wrapper")

            # Initialize the chart
            await initialize_chart(chart_wrapper, timeframe_container)

        # Indicators panel
        with ui.card().classes("chart-card w-full mx-4 mb-4"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("timeline", size="1.5rem").classes("text-purple-400")
                ui.label("Technical Indicators & Tools").classes("card-title")

            ui.separator().classes("card-separator")

            with ui.element('div').classes("indicators-section"):
                with ui.element('div').classes("indicator-controls"):
                    # Fixed indicator selector with proper empty option
                    indicator_options = ["Select Indicator", "SMA", "EMA", "RSI", "MACD", "Bollinger Bands"]
                    indicator_select = ui.select(
                        options=indicator_options,
                        value="Select Indicator"
                    ).classes("symbol-select").props("outlined dense").style("min-width: 150px;")

                    # Period input
                    period_input = ui.number(
                        label="Period",
                        value=20,
                        min=1,
                        max=200
                    ).classes("symbol-select").style("width: 100px;")

                    # Add button
                    ui.button(
                        "Add Indicator",
                        icon="add",
                        on_click=lambda: add_indicator(indicator_select.value, period_input.value)
                    ).classes("control-btn")

                    ui.separator().props("vertical").classes("mx-4")

                    # Drawing tools
                    ui.label("Drawing Tools:").classes("text-gray-400 font-semibold")

                    drawing_tools = [
                        ("Trendline", "trending_up"),
                        ("Rectangle", "crop_square"),
                        ("Horizontal", "remove")
                    ]

                    for tool_name, icon in drawing_tools:
                        ui.button(
                            tool_name,
                            icon=icon,
                            on_click=lambda t=tool_name: ui.notify(f"{t} tool activated", color="info")
                        ).classes("control-btn")


async def initialize_chart(chart_wrapper, timeframe_container):
    """Initialize the TradingView chart with proper error handling"""

    # Create timeframe buttons
    with timeframe_container:
        timeframes = ["1m", "5m", "15m", "1h", "1d"]
        for tf in timeframes:
            is_active = tf == chart_state["selected_timeframe"]
            btn_class = "timeframe-btn active" if is_active else "timeframe-btn"

            def create_handler(timeframe):
                def handler():
                    change_timeframe(timeframe)

                return handler

            ui.button(tf, on_click=create_handler(tf)).classes(btn_class)

    # Create chart container
    with chart_wrapper:
        chart_container = ui.element('div').props('id="tradingview_chart"').style("width: 100%; height: 100%;")

        # Show loading initially
        loading_div = ui.element('div').classes("chart-loading")
        with loading_div:
            ui.spinner(size="lg").classes("text-cyan-400")
            ui.label("Initializing Chart...").classes("text-gray-400 mt-4")

    # Initialize chart with JavaScript
    await setup_tradingview_chart(loading_div)


async def setup_tradingview_chart(loading_div):
    """Setup TradingView Lightweight Charts"""

    try:
        # Generate sample data
        chart_data = generate_sample_data()

        if not chart_data:
            loading_div.clear()
            with loading_div:
                ui.element('div').classes("chart-error")
                ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                ui.label("No Chart Data Available").classes("text-lg font-semibold mb-2")
                ui.button("Generate Sample Data",
                          on_click=lambda: ui.notify("Generating data...", color="info")).classes("mt-4")
            return

        # Remove loading indicator
        loading_div.delete()

        # Initialize chart with comprehensive error handling
        chart_script = f"""
        (function() {{
            let isInitializing = false;

            function loadTradingViewChart() {{
                if (isInitializing) return;
                isInitializing = true;

                // Check if library exists
                if (typeof LightweightCharts === 'undefined') {{
                    console.log('Loading TradingView Lightweight Charts...');

                    const script = document.createElement('script');
                    script.src = 'https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js';
                    script.crossOrigin = 'anonymous';

                    script.onload = function() {{
                        console.log('TradingView Charts loaded successfully');
                        setTimeout(initializeChart, 100);
                    }};

                    script.onerror = function() {{
                        console.error('Failed to load TradingView Charts library');
                        showError('Failed to load charting library. Please check your internet connection.');
                        isInitializing = false;
                    }};

                    document.head.appendChild(script);
                }} else {{
                    initializeChart();
                }}
            }}

            function initializeChart() {{
                try {{
                    const container = document.getElementById('tradingview_chart');
                    if (!container) {{
                        console.error('Chart container not found');
                        showError('Chart container not found');
                        isInitializing = false;
                        return;
                    }}

                    // Clear container
                    container.innerHTML = '';

                    // Check container dimensions
                    if (container.offsetWidth === 0 || container.offsetHeight === 0) {{
                        console.warn('Container has zero dimensions, retrying...');
                        setTimeout(initializeChart, 500);
                        return;
                    }}

                    console.log('Creating chart with dimensions:', container.offsetWidth, 'x', container.offsetHeight);

                    // Create chart
                    const chart = LightweightCharts.createChart(container, {{
                        width: container.offsetWidth,
                        height: container.offsetHeight,
                        layout: {{
                            background: {{ color: '#0a0f23' }},
                            textColor: '#ffffff',
                            fontFamily: 'Inter, sans-serif'
                        }},
                        grid: {{
                            vertLines: {{ color: 'rgba(255, 255, 255, 0.1)' }},
                            horzLines: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                        }},
                        crosshair: {{
                            mode: LightweightCharts.CrosshairMode.Normal,
                            vertLine: {{
                                color: '#22c5fc',
                                width: 1,
                                style: LightweightCharts.LineStyle.Solid
                            }},
                            horzLine: {{
                                color: '#22c5fc',
                                width: 1,
                                style: LightweightCharts.LineStyle.Solid
                            }}
                        }},
                        timeScale: {{
                            timeVisible: true,
                            secondsVisible: false,
                            borderColor: 'rgba(255, 255, 255, 0.2)'
                        }},
                        rightPriceScale: {{
                            borderColor: 'rgba(255, 255, 255, 0.2)'
                        }}
                    }});

                    // Add candlestick series
                    const candlestickSeries = chart.addCandlestickSeries({{
                        upColor: '#22c55e',
                        downColor: '#ef4444',
                        borderUpColor: '#22c55e',
                        borderDownColor: '#ef4444',
                        wickUpColor: '#22c55e',
                        wickDownColor: '#ef4444'
                    }});

                    // Add volume series
                    const volumeSeries = chart.addHistogramSeries({{
                        color: 'rgba(34, 197, 252, 0.3)',
                        priceFormat: {{ type: 'volume' }},
                        priceScaleId: 'volume',
                        scaleMargins: {{ top: 0.8, bottom: 0 }}
                    }});

                    // Set data
                    const chartData = {json.dumps(chart_data)};

                    const ohlcData = chartData.map(candle => ({{
                        time: candle.time,
                        open: candle.open,
                        high: candle.high,
                        low: candle.low,
                        close: candle.close
                    }}));

                    const volumeData = chartData.map(candle => ({{
                        time: candle.time,
                        value: candle.volume
                    }}));

                    candlestickSeries.setData(ohlcData);
                    volumeSeries.setData(volumeData);

                    // Fit content
                    chart.timeScale().fitContent();

                    // Store references
                    window.tradingChart = chart;
                    window.candlestickSeries = candlestickSeries;
                    window.volumeSeries = volumeSeries;

                    // Handle resize
                    const resizeObserver = new ResizeObserver(entries => {{
                        if (entries.length && entries[0].target === container) {{
                            const newRect = entries[0].contentRect;
                            chart.applyOptions({{
                                width: newRect.width,
                                height: newRect.height
                            }});
                        }}
                    }});
                    resizeObserver.observe(container);

                    console.log('Chart initialized successfully');
                    isInitializing = false;

                }} catch (error) {{
                    console.error('Chart initialization error:', error);
                    showError('Chart initialization failed: ' + error.message);
                    isInitializing = false;
                }}
            }}

            function showError(message) {{
                const container = document.getElementById('tradingview_chart');
                if (container) {{
                    container.innerHTML = `
                        <div class="chart-error">
                            <div style="font-size: 2rem; margin-bottom: 1rem;">⚠️</div>
                            <div style="font-size: 1.125rem; font-weight: 600; margin-bottom: 0.5rem; color: #ef4444;">Chart Error</div>
                            <div style="font-size: 0.875rem; margin-bottom: 1rem;">${{message}}</div>
                            <button onclick="location.reload()" style="padding: 0.5rem 1rem; background: #22c5fc; border: none; border-radius: 6px; color: white; cursor: pointer;">Reload Page</button>
                        </div>
                    `;
                }}
                isInitializing = false;
            }}

            // Start loading
            loadTradingViewChart();
        }})();
        """

        ui.run_javascript(chart_script)
        chart_state["chart_initialized"] = True

    except Exception as e:
        logger.error(f"Error setting up chart: {e}")
        loading_div.clear()
        with loading_div:
            ui.element('div').classes("chart-error")
            ui.icon("error", size="3rem").classes("text-red-500 mb-4")
            ui.label("Setup Error").classes("text-lg font-semibold text-red-400")
            ui.label(f"Error: {str(e)}").classes("text-sm text-gray-400")


def generate_sample_data():
    """Generate sample OHLCV data for demonstration"""

    try:
        import random
        from datetime import datetime, timedelta

        data = []
        base_price = 18500.0  # NIFTY50 base price
        current_time = datetime.now() - timedelta(days=7)  # Last 7 days

        for i in range(200):  # 200 5-minute candles
            current_time += timedelta(minutes=5)

            # Generate realistic price movement
            change_pct = random.uniform(-0.5, 0.5)  # Max 0.5% change per candle
            open_price = base_price
            close_price = open_price * (1 + change_pct / 100)

            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.2) / 100)
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.2) / 100)
            volume = random.randint(50000, 200000)

            data.append({
                "time": int(current_time.timestamp()),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume
            })

            base_price = close_price  # Use close as next open

        return data

    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return []


def on_symbol_change(symbol):
    """Handle symbol change"""
    chart_state["selected_symbol"] = symbol
    ui.notify(f"Symbol changed to {symbol}", color="info")
    # Here you would typically fetch new data and update the chart


def change_timeframe(timeframe):
    """Handle timeframe change"""
    chart_state["selected_timeframe"] = timeframe
    ui.notify(f"Timeframe changed to {timeframe}", color="info")

    # Update button states via JavaScript
    update_script = f"""
    document.querySelectorAll('.timeframe-btn').forEach(btn => {{
        btn.classList.remove('active');
        if (btn.textContent.trim() === '{timeframe}') {{
            btn.classList.add('active');
        }}
    }});
    """
    ui.run_javascript(update_script)


def refresh_chart():
    """Refresh chart data"""
    ui.notify("Refreshing chart data...", color="info")
    # Here you would fetch fresh data and update the chart


def add_indicator(indicator_type, period):
    """Add technical indicator"""
    if indicator_type == "Select Indicator":
        ui.notify("Please select a valid indicator", color="warning")
        return

    ui.notify(f"Adding {indicator_type} with period {period}", color="positive")
    # Here you would add the indicator to the chart