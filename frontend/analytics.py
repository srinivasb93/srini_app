"""
Enhanced Analytics Module - analytics.py
Comprehensive stock analysis with TradingView widgets and advanced metrics
"""

from nicegui import ui, app
import asyncio
from datetime import datetime, timedelta
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import talib
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Enhanced chart state with more comprehensive data
chart_state = {
    "selected_symbol": "RELIANCE",
    "selected_timeframe": "1d",
    "chart_type": "candlestick",
    "indicators": {
        "sma": {"enabled": False, "period": 20, "color": "#ff6b6b"},
        "ema": {"enabled": False, "period": 20, "color": "#4ecdc4"},
        "rsi": {"enabled": False, "period": 14, "overbought": 70, "oversold": 30},
        "macd": {"enabled": False, "fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"enabled": False, "period": 20, "std_dev": 2},
        "volume": {"enabled": True, "type": "histogram"},
        "atr": {"enabled": False, "period": 14},
        "stochastic": {"enabled": False, "k_period": 14, "d_period": 3},
        "williams_r": {"enabled": False, "period": 14},
        "cci": {"enabled": False, "period": 20}
    },
    "drawing_tools": {"enabled": False, "tool": "line"},
    "analysis_metrics": {},
    "current_data": [],
    "symbol_info": {},
    "market_data": {}
}


@dataclass
class StockMetrics:
    """Comprehensive stock metrics for trading decisions"""
    symbol: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    avg_volume: int
    market_cap: float
    pe_ratio: float
    pb_ratio: float
    dividend_yield: float
    beta: float
    fifty_two_week_high: float
    fifty_two_week_low: float

    # Technical indicators
    rsi: float
    macd_signal: str
    bollinger_position: str
    sma_20: float
    sma_50: float
    sma_200: float

    # Trading signals
    trend_signal: str
    momentum_signal: str
    volume_signal: str
    overall_signal: str
    confidence_score: float


async def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch comprehensive stock data using yfinance and other sources"""
    try:
        # Convert symbol format for yfinance (add .NS for NSE stocks)
        yf_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol

        stock = yf.Ticker(yf_symbol)

        # Fetch historical data
        hist_data = stock.history(period=period)

        if hist_data.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()

        # Add technical indicators using talib
        hist_data = add_technical_indicators(hist_data)

        return hist_data

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators to the dataframe"""
    try:
        # Ensure we have the required columns
        if df.empty or len(df) < 50:
            return df

        # Moving Averages
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
        df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)

        # RSI
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])

        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])

        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])

        # Williams %R
        df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'])

        # CCI
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])

        # ATR
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])

        # Volume indicators
        df['Volume_SMA'] = talib.SMA(df['Volume'], timeperiod=20)
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])

        return df

    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        return df


async def calculate_stock_metrics(symbol: str, df: pd.DataFrame) -> StockMetrics:
    """Calculate comprehensive stock metrics for trading decisions"""
    try:
        if df.empty:
            return None

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest

        # Basic metrics
        current_price = latest['Close']
        change = current_price - previous['Close']
        change_percent = (change / previous['Close']) * 100

        # Volume analysis
        avg_volume = df['Volume'].tail(20).mean()
        volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1

        # Technical signals
        rsi = latest.get('RSI', 50)
        sma_20 = latest.get('SMA_20', current_price)
        sma_50 = latest.get('SMA_50', current_price)
        sma_200 = latest.get('SMA_200', current_price)

        # Generate trading signals
        trend_signal = generate_trend_signal(current_price, sma_20, sma_50, sma_200)
        momentum_signal = generate_momentum_signal(rsi, latest.get('MACD', 0), latest.get('MACD_Signal', 0))
        volume_signal = generate_volume_signal(volume_ratio)

        # Calculate overall signal and confidence
        overall_signal, confidence_score = calculate_overall_signal(
            trend_signal, momentum_signal, volume_signal, rsi
        )

        # Get additional metrics (mock data for now - integrate with real APIs)
        stock_info = await get_stock_fundamentals(symbol)

        return StockMetrics(
            symbol=symbol,
            current_price=current_price,
            change=change,
            change_percent=change_percent,
            volume=int(latest['Volume']),
            avg_volume=int(avg_volume),
            market_cap=stock_info.get('market_cap', 0),
            pe_ratio=stock_info.get('pe_ratio', 0),
            pb_ratio=stock_info.get('pb_ratio', 0),
            dividend_yield=stock_info.get('dividend_yield', 0),
            beta=stock_info.get('beta', 1.0),
            fifty_two_week_high=df['High'].max(),
            fifty_two_week_low=df['Low'].min(),
            rsi=rsi,
            macd_signal=get_macd_signal(latest.get('MACD', 0), latest.get('MACD_Signal', 0)),
            bollinger_position=get_bollinger_position(current_price, latest.get('BB_Upper', 0),
                                                      latest.get('BB_Lower', 0)),
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            trend_signal=trend_signal,
            momentum_signal=momentum_signal,
            volume_signal=volume_signal,
            overall_signal=overall_signal,
            confidence_score=confidence_score
        )

    except Exception as e:
        logger.error(f"Error calculating stock metrics: {e}")
        return None


def generate_trend_signal(price, sma_20, sma_50, sma_200):
    """Generate trend signal based on moving averages"""
    if price > sma_20 > sma_50 > sma_200:
        return "Strong Bullish"
    elif price > sma_20 > sma_50:
        return "Bullish"
    elif price < sma_20 < sma_50 < sma_200:
        return "Strong Bearish"
    elif price < sma_20 < sma_50:
        return "Bearish"
    else:
        return "Neutral"


def generate_momentum_signal(rsi, macd, macd_signal):
    """Generate momentum signal based on RSI and MACD"""
    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    macd_signal_trend = "Bullish" if macd > macd_signal else "Bearish"

    if rsi < 30 and macd_signal_trend == "Bullish":
        return "Strong Buy"
    elif rsi > 70 and macd_signal_trend == "Bearish":
        return "Strong Sell"
    elif rsi < 40 and macd_signal_trend == "Bullish":
        return "Buy"
    elif rsi > 60 and macd_signal_trend == "Bearish":
        return "Sell"
    else:
        return "Hold"


def generate_volume_signal(volume_ratio):
    """Generate volume signal based on volume ratio"""
    if volume_ratio > 2.0:
        return "High Interest"
    elif volume_ratio > 1.5:
        return "Above Average"
    elif volume_ratio < 0.5:
        return "Low Interest"
    else:
        return "Normal"


def calculate_overall_signal(trend_signal, momentum_signal, volume_signal, rsi):
    """Calculate overall trading signal and confidence score"""
    buy_signals = 0
    sell_signals = 0

    # Trend analysis
    if "Bullish" in trend_signal:
        buy_signals += 2 if "Strong" in trend_signal else 1
    elif "Bearish" in trend_signal:
        sell_signals += 2 if "Strong" in trend_signal else 1

    # Momentum analysis
    if "Buy" in momentum_signal:
        buy_signals += 2 if "Strong" in momentum_signal else 1
    elif "Sell" in momentum_signal:
        sell_signals += 2 if "Strong" in momentum_signal else 1

    # Volume confirmation
    if volume_signal in ["High Interest", "Above Average"]:
        if buy_signals > sell_signals:
            buy_signals += 1
        elif sell_signals > buy_signals:
            sell_signals += 1

    # Determine overall signal
    if buy_signals > sell_signals + 1:
        overall_signal = "BUY"
        confidence_score = min(90, (buy_signals / (buy_signals + sell_signals)) * 100)
    elif sell_signals > buy_signals + 1:
        overall_signal = "SELL"
        confidence_score = min(90, (sell_signals / (buy_signals + sell_signals)) * 100)
    else:
        overall_signal = "HOLD"
        confidence_score = 50

    return overall_signal, confidence_score


def get_macd_signal(macd, macd_signal):
    """Get MACD signal interpretation"""
    if macd > macd_signal:
        return "Bullish Crossover"
    elif macd < macd_signal:
        return "Bearish Crossover"
    else:
        return "Neutral"


def get_bollinger_position(price, upper, lower):
    """Get Bollinger Bands position"""
    if upper == 0 or lower == 0:
        return "No Data"

    if price > upper:
        return "Above Upper Band"
    elif price < lower:
        return "Below Lower Band"
    else:
        return "Within Bands"


async def get_stock_fundamentals(symbol: str) -> Dict:
    """Get stock fundamental data (integrate with real APIs)"""
    try:
        # This is mock data - integrate with actual APIs like Alpha Vantage, EOD Historical Data, etc.
        fundamentals = {
            'market_cap': 1500000000000,  # 1.5T
            'pe_ratio': 25.5,
            'pb_ratio': 3.2,
            'dividend_yield': 2.1,
            'beta': 1.15,
            'eps': 85.2,
            'book_value': 650.0,
            'debt_to_equity': 0.45
        }
        return fundamentals
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        return {}


async def create_tradingview_widget(symbol: str, timeframe: str) -> str:
    """Create TradingView widget HTML"""
    widget_html = f'''
    <div class="tradingview-widget-container">
        <div id="tradingview_chart_{symbol.lower()}"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
            new TradingView.widget({{
                "width": "100%",
                "height": 600,
                "symbol": "NSE:{symbol}",
                "interval": "{timeframe}",
                "timezone": "Asia/Kolkata",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#1e293b",
                "enable_publishing": false,
                "hide_top_toolbar": false,
                "hide_legend": false,
                "save_image": true,
                "container_id": "tradingview_chart_{symbol.lower()}",
                "studies": [
                    "RSI@tv-basicstudies",
                    "MACD@tv-basicstudies",
                    "Volume@tv-basicstudies"
                ],
                "overrides": {{
                    "paneProperties.background": "#0f172a",
                    "paneProperties.vertGridProperties.color": "#334155",
                    "paneProperties.horzGridProperties.color": "#334155",
                    "symbolWatermarkProperties.transparency": 90,
                    "scalesProperties.textColor": "#94a3b8"
                }}
            }});
        </script>
    </div>
    '''
    return widget_html


async def render_analytics_page(fetch_api, user_storage, get_cached_instruments, broker):
    """Enhanced analytics page with comprehensive stock analysis"""
    with ui.column().classes("analytics-container w-full min-h-screen p-4"):
        # Enhanced header
        with ui.element('div').classes("analytics-header"):
            with ui.row().classes("w-full justify-between items-center"):
                with ui.column().classes("gap-2"):
                    ui.label("Advanced Stock Analysis").classes("page-title")
                    ui.label("Professional-grade technical analysis with TradingView integration").classes(
                        "page-subtitle")

                with ui.row().classes("gap-4"):
                    ui.button("Screener", icon="filter_list", on_click=lambda: ui.navigate.to('/screener')).classes(
                        "px-6 py-2")
                    ui.button("Alerts", icon="notifications", on_click=lambda: ui.navigate.to('/alerts')).classes(
                        "px-6 py-2")

        # Main chart container with TradingView widget
        with ui.element('div').classes("main-chart-container"):
            # Chart controls header
            with ui.element('div').classes("chart-controls-header"):
                with ui.row().classes("w-full justify-between items-center"):
                    # Symbol selector
                    with ui.row().classes("items-center gap-4"):
                        ui.label("Symbol:").classes("text-sm font-medium")
                        symbol_select = ui.select(
                            options=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "LT", "WIPRO",
                                     "ONGC"],
                            value=chart_state["selected_symbol"],
                            on_change=lambda e: update_symbol(e.value)
                        ).classes("symbol-selector")

                    # Timeframe buttons
                    with ui.element('div').classes("timeframe-buttons"):
                        for tf in ["1m", "5m", "15m", "1h", "1d", "1w", "1M"]:
                            btn_class = "timeframe-btn active" if tf == chart_state[
                                "selected_timeframe"] else "timeframe-btn"
                            ui.button(tf, on_click=lambda t=tf: update_timeframe(t)).classes(btn_class)

            # TradingView widget container
            chart_container = ui.element('div').classes("tradingview-widget-container")

            # Drawing tools (optional)
            with ui.element('div').classes("drawing-tools"):
                ui.label("Drawing Tools:").classes("text-sm font-medium mr-4")
                for tool in ["line", "horizontal_line", "trend_line", "rectangle", "circle"]:
                    ui.button(tool.replace("_", " ").title(), on_click=lambda t=tool: toggle_drawing_tool(t)).classes(
                        "drawing-tool-btn text-xs")

        # Stock metrics grid
        metrics_container = ui.column().classes("w-full")

        # Technical indicators panel
        with ui.element('div').classes("indicators-panel"):
            ui.label("Technical Indicators").classes("indicator-group-title")

            with ui.element('div').classes("indicator-controls"):
                # Trend indicators
                with ui.element('div').classes("indicator-control"):
                    ui.label("Trend Indicators").classes("text-sm font-semibold mb-2")
                    ui.checkbox("SMA (20, 50, 200)", value=chart_state["indicators"]["sma"]["enabled"],
                                on_change=lambda e: toggle_indicator("sma", e.value))
                    ui.checkbox("EMA (20, 50)", value=chart_state["indicators"]["ema"]["enabled"],
                                on_change=lambda e: toggle_indicator("ema", e.value))
                    ui.checkbox("Bollinger Bands", value=chart_state["indicators"]["bollinger"]["enabled"],
                                on_change=lambda e: toggle_indicator("bollinger", e.value))

                # Momentum indicators
                with ui.element('div').classes("indicator-control"):
                    ui.label("Momentum Indicators").classes("text-sm font-semibold mb-2")
                    ui.checkbox("RSI (14)", value=chart_state["indicators"]["rsi"]["enabled"],
                                on_change=lambda e: toggle_indicator("rsi", e.value))
                    ui.checkbox("MACD (12,26,9)", value=chart_state["indicators"]["macd"]["enabled"],
                                on_change=lambda e: toggle_indicator("macd", e.value))
                    ui.checkbox("Stochastic", value=chart_state["indicators"]["stochastic"]["enabled"],
                                on_change=lambda e: toggle_indicator("stochastic", e.value))

                # Volume indicators
                with ui.element('div').classes("indicator-control"):
                    ui.label("Volume & Others").classes("text-sm font-semibold mb-2")
                    ui.checkbox("Volume", value=chart_state["indicators"]["volume"]["enabled"],
                                on_change=lambda e: toggle_indicator("volume", e.value))
                    ui.checkbox("ATR", value=chart_state["indicators"]["atr"]["enabled"],
                                on_change=lambda e: toggle_indicator("atr", e.value))
                    ui.checkbox("CCI", value=chart_state["indicators"]["cci"]["enabled"],
                                on_change=lambda e: toggle_indicator("cci", e.value))

        # Analysis summary
        analysis_container = ui.column().classes("w-full")

    # Initialize with default symbol
    await update_chart_and_metrics(chart_state["selected_symbol"])


async def update_symbol(symbol: str):
    """Update selected symbol and refresh data"""
    chart_state["selected_symbol"] = symbol
    await update_chart_and_metrics(symbol)


async def update_timeframe(timeframe: str):
    """Update selected timeframe and refresh chart"""
    chart_state["selected_timeframe"] = timeframe
    await update_chart_and_metrics(chart_state["selected_symbol"])


def toggle_indicator(indicator: str, enabled: bool):
    """Toggle technical indicator on/off"""
    chart_state["indicators"][indicator]["enabled"] = enabled
    # Refresh chart with updated indicators
    asyncio.create_task(update_chart_and_metrics(chart_state["selected_symbol"]))


def toggle_drawing_tool(tool: str):
    """Toggle drawing tool"""
    chart_state["drawing_tools"]["tool"] = tool
    chart_state["drawing_tools"]["enabled"] = True


async def update_chart_and_metrics(symbol: str):
    """Update both chart and metrics for the selected symbol"""
    try:
        # Fetch stock data
        df = await fetch_stock_data(symbol)
        if df.empty:
            logger.warning(f"No data available for {symbol}")
            return

        # Calculate comprehensive metrics
        metrics = await calculate_stock_metrics(symbol, df)
        if not metrics:
            logger.warning(f"Could not calculate metrics for {symbol}")
            return

        # Update chart with TradingView widget
        await update_tradingview_chart(symbol, chart_state["selected_timeframe"])

        # Update metrics display
        await update_metrics_display(metrics)

        # Update analysis summary
        await update_analysis_summary(metrics, df)

    except Exception as e:
        logger.error(f"Error updating chart and metrics: {e}")


async def update_tradingview_chart(symbol: str, timeframe: str):
    """Update TradingView chart widget"""
    try:
        widget_html = await create_tradingview_widget(symbol, timeframe)

        # Update the chart container with new widget
        # Note: In a real implementation, you'd need to properly handle DOM updates
        chart_script = f'''
        <script>
            // Clear existing chart
            const container = document.querySelector('.tradingview-widget-container');
            if (container) {{
                container.innerHTML = `{widget_html}`;
            }}
        </script>
        '''

        ui.add_head_html(chart_script)

    except Exception as e:
        logger.error(f"Error updating TradingView chart: {e}")


async def update_metrics_display(metrics: StockMetrics):
    """Update the stock metrics display"""
    try:
        # Clear existing metrics container and rebuild
        with ui.element('div').classes("stock-metrics-grid"):
            # Price & Change Card
            with ui.element('div').classes("metric-card"):
                ui.label("Price & Change").classes("metric-card-title")
                with ui.row().classes("items-baseline gap-2"):
                    ui.label(f"₹{metrics.current_price:.2f}").classes("metric-value")
                    change_class = "positive" if metrics.change >= 0 else "negative"
                    ui.label(f"{metrics.change:+.2f} ({metrics.change_percent:+.2f}%)").classes(
                        f"metric-change {change_class}")

            # Volume Analysis Card
            with ui.element('div').classes("metric-card"):
                ui.label("Volume Analysis").classes("metric-card-title")
                ui.label(f"{metrics.volume:,}").classes("metric-value text-blue-400")
                volume_ratio = metrics.volume / metrics.avg_volume if metrics.avg_volume > 0 else 1
                ui.label(f"Avg: {metrics.avg_volume:,} ({volume_ratio:.1f}x)").classes("metric-change neutral")
                ui.label(metrics.volume_signal).classes(
                    f"signal-badge signal-{get_signal_class(metrics.volume_signal)}")

            # Technical Indicators Card
            with ui.element('div').classes("metric-card"):
                ui.label("Technical Indicators").classes("metric-card-title")

                with ui.column().classes("gap-2"):
                    # RSI
                    with ui.row().classes("justify-between items-center"):
                        ui.label("RSI (14)")
                        rsi_class = "negative" if metrics.rsi > 70 else "positive" if metrics.rsi < 30 else "neutral"
                        ui.label(f"{metrics.rsi:.1f}").classes(f"font-bold {rsi_class}")

                    # MACD
                    with ui.row().classes("justify-between items-center"):
                        ui.label("MACD Signal")
                        ui.label(metrics.macd_signal).classes(
                            f"signal-badge signal-{get_signal_class(metrics.macd_signal)}")

                    # Bollinger Bands
                    with ui.row().classes("justify-between items-center"):
                        ui.label("Bollinger Position")
                        ui.label(metrics.bollinger_position).classes("text-sm neutral")

            # Fundamental Data Card
            with ui.element('div').classes("metric-card"):
                ui.label("Fundamentals").classes("metric-card-title")

                with ui.column().classes("gap-1"):
                    with ui.row().classes("justify-between"):
                        ui.label("P/E Ratio")
                        ui.label(f"{metrics.pe_ratio:.2f}").classes("font-mono")

                    with ui.row().classes("justify-between"):
                        ui.label("P/B Ratio")
                        ui.label(f"{metrics.pb_ratio:.2f}").classes("font-mono")

                    with ui.row().classes("justify-between"):
                        ui.label("Dividend Yield")
                        ui.label(f"{metrics.dividend_yield:.2f}%").classes("font-mono")

                    with ui.row().classes("justify-between"):
                        ui.label("Beta")
                        ui.label(f"{metrics.beta:.2f}").classes("font-mono")

            # 52-Week Range Card
            with ui.element('div').classes("metric-card"):
                ui.label("52-Week Range").classes("metric-card-title")

                with ui.column().classes("gap-2"):
                    ui.label(f"High: ₹{metrics.fifty_two_week_high:.2f}").classes("text-green-400")
                    ui.label(f"Low: ₹{metrics.fifty_two_week_low:.2f}").classes("text-red-400")

                    # Calculate position in range
                    range_position = ((metrics.current_price - metrics.fifty_two_week_low) /
                                      (metrics.fifty_two_week_high - metrics.fifty_two_week_low)) * 100
                    ui.label(f"Position: {range_position:.1f}%").classes("text-blue-400")

            # Moving Averages Card
            with ui.element('div').classes("metric-card"):
                ui.label("Moving Averages").classes("metric-card-title")

                with ui.column().classes("gap-1"):
                    sma_20_signal = "positive" if metrics.current_price > metrics.sma_20 else "negative"
                    with ui.row().classes("justify-between"):
                        ui.label("SMA 20")
                        ui.label(f"₹{metrics.sma_20:.2f}").classes(f"font-mono {sma_20_signal}")

                    sma_50_signal = "positive" if metrics.current_price > metrics.sma_50 else "negative"
                    with ui.row().classes("justify-between"):
                        ui.label("SMA 50")
                        ui.label(f"₹{metrics.sma_50:.2f}").classes(f"font-mono {sma_50_signal}")

                    sma_200_signal = "positive" if metrics.current_price > metrics.sma_200 else "negative"
                    with ui.row().classes("justify-between"):
                        ui.label("SMA 200")
                        ui.label(f"₹{metrics.sma_200:.2f}").classes(f"font-mono {sma_200_signal}")

    except Exception as e:
        logger.error(f"Error updating metrics display: {e}")


async def update_analysis_summary(metrics: StockMetrics, df: pd.DataFrame):
    """Update the comprehensive analysis summary"""
    try:
        with ui.element('div').classes("analysis-summary"):
            with ui.row().classes("w-full justify-between items-start"):
                # Overall Signal
                with ui.column().classes("gap-4"):
                    ui.label("Trading Analysis Summary").classes("text-xl font-bold mb-4")

                    with ui.row().classes("items-center gap-4"):
                        ui.label("Overall Signal:").classes("text-lg")
                        signal_class = get_signal_class(metrics.overall_signal)
                        ui.label(metrics.overall_signal).classes(
                            f"signal-badge signal-{signal_class} text-lg px-4 py-2")

                    # Confidence meter
                    ui.label(f"Confidence: {metrics.confidence_score:.1f}%").classes("text-sm text-gray-300")
                    with ui.element('div').classes("confidence-meter"):
                        confidence_color = "#22c55e" if metrics.confidence_score > 70 else "#f59e0b" if metrics.confidence_score > 50 else "#ef4444"
                        ui.element('div').classes("confidence-fill").style(
                            f"width: {metrics.confidence_score}%; background-color: {confidence_color};"
                        )

                # Signal Breakdown
                with ui.column().classes("gap-2"):
                    ui.label("Signal Breakdown").classes("text-lg font-semibold mb-2")

                    with ui.column().classes("gap-1"):
                        ui.label(f"Trend: {metrics.trend_signal}").classes(
                            f"signal-badge signal-{get_signal_class(metrics.trend_signal)}")
                        ui.label(f"Momentum: {metrics.momentum_signal}").classes(
                            f"signal-badge signal-{get_signal_class(metrics.momentum_signal)}")
                        ui.label(f"Volume: {metrics.volume_signal}").classes(
                            f"signal-badge signal-{get_signal_class(metrics.volume_signal)}")

            # Key Levels and Recommendations
            with ui.element('div').classes("mt-6 p-4 bg-slate-800 rounded-lg"):
                ui.label("Key Levels & Recommendations").classes("text-lg font-semibold mb-3")

                recommendations = generate_trading_recommendations(metrics, df)

                with ui.column().classes("gap-2"):
                    for rec in recommendations:
                        with ui.row().classes("items-center gap-2"):
                            ui.icon(rec["icon"], size="sm").classes(f"text-{rec['color']}-400")
                            ui.label(rec["text"]).classes("text-sm")

    except Exception as e:
        logger.error(f"Error updating analysis summary: {e}")


def get_signal_class(signal: str) -> str:
    """Get CSS class for signal styling"""
    signal_lower = signal.lower()

    if any(word in signal_lower for word in ["buy", "bullish", "strong", "high", "above"]):
        return "buy"
    elif any(word in signal_lower for word in ["sell", "bearish", "weak", "low", "below"]):
        return "sell"
    else:
        return "hold"


def generate_trading_recommendations(metrics: StockMetrics, df: pd.DataFrame) -> List[Dict]:
    """Generate actionable trading recommendations"""
    recommendations = []

    try:
        # Support and Resistance levels
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()

        recommendations.append({
            "icon": "trending_up",
            "color": "green",
            "text": f"Resistance Level: ₹{recent_high:.2f} - Watch for breakout above this level"
        })

        recommendations.append({
            "icon": "trending_down",
            "color": "red",
            "text": f"Support Level: ₹{recent_low:.2f} - Strong buying opportunity if price holds"
        })

        # RSI based recommendations
        if metrics.rsi > 70:
            recommendations.append({
                "icon": "warning",
                "color": "yellow",
                "text": "RSI indicates overbought conditions - Consider taking profits"
            })
        elif metrics.rsi < 30:
            recommendations.append({
                "icon": "shopping_cart",
                "color": "green",
                "text": "RSI indicates oversold conditions - Potential buying opportunity"
            })

        # Volume based recommendations
        if "High" in metrics.volume_signal:
            recommendations.append({
                "icon": "volume_up",
                "color": "blue",
                "text": "High volume confirms price movement - Strong conviction in current trend"
            })
        elif "Low" in metrics.volume_signal:
            recommendations.append({
                "icon": "volume_down",
                "color": "gray",
                "text": "Low volume suggests weak conviction - Wait for volume confirmation"
            })

        # Trend based recommendations
        if "Strong Bullish" in metrics.trend_signal:
            recommendations.append({
                "icon": "rocket_launch",
                "color": "green",
                "text": "Strong uptrend - Consider adding to positions on dips"
            })
        elif "Strong Bearish" in metrics.trend_signal:
            recommendations.append({
                "icon": "trending_down",
                "color": "red",
                "text": "Strong downtrend - Avoid catching falling knife, wait for reversal"
            })

        # Risk management
        atr_value = df['ATR'].iloc[-1] if 'ATR' in df.columns else metrics.current_price * 0.02
        stop_loss = metrics.current_price - (2 * atr_value)
        target = metrics.current_price + (3 * atr_value)

        recommendations.append({
            "icon": "shield",
            "color": "orange",
            "text": f"Suggested Stop Loss: ₹{stop_loss:.2f} (2x ATR)"
        })

        recommendations.append({
            "icon": "flag",
            "color": "purple",
            "text": f"Suggested Target: ₹{target:.2f} (3x ATR)"
        })

        return recommendations

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []


# Additional utility functions for enhanced features

async def create_price_alerts_section():
    """Create price alerts section"""
    with ui.element('div').classes("metric-card"):
        ui.label("Price Alerts").classes("metric-card-title")

        with ui.column().classes("gap-3"):
            # Create alert form
            with ui.row().classes("gap-2 items-end"):
                alert_price = ui.number("Alert Price", value=0, format="%.2f").classes("flex-1")
                alert_type = ui.select(["Above", "Below"], value="Above").classes("w-24")
                ui.button("Set Alert", icon="add_alert").classes("px-4")

            # Existing alerts list (mock data)
            ui.label("Active Alerts").classes("text-sm font-semibold mt-3")

            mock_alerts = [
                {"price": 2500.00, "type": "Above", "status": "Active"},
                {"price": 2300.00, "type": "Below", "status": "Active"}
            ]

            for alert in mock_alerts:
                with ui.row().classes("justify-between items-center p-2 bg-slate-700 rounded"):
                    ui.label(f"₹{alert['price']:.2f} ({alert['type']})").classes("text-sm")
                    with ui.row().classes("gap-1"):
                        ui.chip(alert["status"], color="green" if alert["status"] == "Active" else "gray").classes(
                            "text-xs")
                        ui.button(icon="delete", on_click=lambda: None).classes("text-red-400 p-1")


async def create_news_sentiment_section(symbol: str):
    """Create news and sentiment analysis section"""
    with ui.element('div').classes("metric-card"):
        ui.label("News & Sentiment").classes("metric-card-title")

        # Mock news data - integrate with real news APIs
        news_items = [
            {
                "headline": f"{symbol} reports strong Q4 earnings, beats estimates",
                "sentiment": "Positive",
                "time": "2 hours ago",
                "source": "Economic Times"
            },
            {
                "headline": f"Analysts upgrade {symbol} target price to ₹2800",
                "sentiment": "Positive",
                "time": "5 hours ago",
                "source": "Moneycontrol"
            },
            {
                "headline": f"Market volatility affects {symbol} trading volumes",
                "sentiment": "Neutral",
                "time": "1 day ago",
                "source": "Business Standard"
            }
        ]

        with ui.column().classes("gap-2"):
            for news in news_items[:3]:  # Show top 3 news
                with ui.element('div').classes("p-2 bg-slate-700 rounded"):
                    ui.label(news["headline"]).classes("text-sm font-medium")
                    with ui.row().classes("justify-between items-center mt-1"):
                        ui.label(news["source"]).classes("text-xs text-gray-400")
                        sentiment_color = "green" if news["sentiment"] == "Positive" else "red" if news[
                                                                                                       "sentiment"] == "Negative" else "gray"
                        ui.chip(news["sentiment"], color=sentiment_color).classes("text-xs")
                    ui.label(news["time"]).classes("text-xs text-gray-500")


async def create_peer_comparison_section(symbol: str):
    """Create peer comparison section"""
    with ui.element('div').classes("metric-card"):
        ui.label("Peer Comparison").classes("metric-card-title")

        # Mock peer data - integrate with real financial data APIs
        peer_data = [
            {"symbol": "TCS", "price": 3500.00, "change": 1.2, "pe": 22.5},
            {"symbol": "INFY", "price": 1650.00, "change": -0.8, "pe": 24.1},
            {"symbol": "WIPRO", "price": 420.00, "change": 0.5, "pe": 18.9},
        ]

        with ui.column().classes("gap-2"):
            for peer in peer_data:
                with ui.row().classes("justify-between items-center p-2 bg-slate-700 rounded"):
                    ui.label(peer["symbol"]).classes("font-medium")
                    with ui.column().classes("items-end"):
                        ui.label(f"₹{peer['price']:.2f}").classes("text-sm")
                        change_color = "text-green-400" if peer["change"] >= 0 else "text-red-400"
                        ui.label(f"{peer['change']:+.1f}%").classes(f"text-xs {change_color}")


# Enhanced screener functionality

async def create_stock_screener():
    """Create advanced stock screener"""
    with ui.element('div').classes("screener-container p-4"):
        ui.label("Stock Screener").classes("text-2xl font-bold mb-4")

        # Screening criteria
        with ui.element('div').classes("screening-criteria bg-slate-800 p-4 rounded-lg mb-4"):
            ui.label("Screening Criteria").classes("text-lg font-semibold mb-3")

            with ui.grid(columns=3).classes("gap-4"):
                # Technical criteria
                with ui.column().classes("gap-2"):
                    ui.label("Technical").classes("font-semibold")
                    ui.number("RSI Min", value=30, min=0, max=100)
                    ui.number("RSI Max", value=70, min=0, max=100)
                    ui.select(["Above SMA20", "Below SMA20", "Any"], value="Any", label="SMA20 Position")

                # Fundamental criteria  
                with ui.column().classes("gap-2"):
                    ui.label("Fundamental").classes("font-semibold")
                    ui.number("P/E Max", value=25)
                    ui.number("Market Cap Min (Cr)", value=1000)
                    ui.number("ROE Min %", value=15)

                # Volume criteria
                with ui.column().classes("gap-2"):
                    ui.label("Volume & Price").classes("font-semibold")
                    ui.number("Volume Ratio Min", value=1.5)
                    ui.number("Price Min", value=100)
                    ui.number("Price Max", value=5000)

            with ui.row().classes("gap-2 mt-4"):
                ui.button("Run Screener", icon="search").classes("bg-blue-600 px-6")
                ui.button("Save Criteria", icon="save").classes("bg-gray-600 px-4")
                ui.button("Load Preset", icon="folder_open").classes("bg-gray-600 px-4")

        # Results table
        with ui.element('div').classes("screener-results"):
            ui.label("Screening Results").classes("text-lg font-semibold mb-3")

            # This would be populated with actual screening results
            mock_results = [
                {"symbol": "TATAMOTORS", "price": 650.50, "change": 2.1, "rsi": 45.2, "pe": 12.5, "volume_ratio": 2.3},
                {"symbol": "MARUTI", "price": 9800.00, "change": -1.2, "rsi": 38.5, "pe": 18.2, "volume_ratio": 1.8},
                {"symbol": "BAJFINANCE", "price": 7200.00, "change": 3.5, "rsi": 65.8, "pe": 22.1, "volume_ratio": 2.8},
            ]

            # Create results table
            with ui.element('div').classes("overflow-x-auto"):
                with ui.element('table').classes("w-full text-sm"):
                    # Header
                    with ui.element('thead'):
                        with ui.element('tr').classes("border-b border-gray-600"):
                            for header in ["Symbol", "Price", "Change %", "RSI", "P/E", "Volume Ratio", "Action"]:
                                ui.element('th').classes("p-2 text-left").add(ui.label(header))

                    # Body
                    with ui.element('tbody'):
                        for result in mock_results:
                            with ui.element('tr').classes("border-b border-gray-700 hover:bg-slate-700"):
                                ui.element('td').classes("p-2").add(ui.label(result["symbol"]).classes("font-medium"))
                                ui.element('td').classes("p-2").add(ui.label(f"₹{result['price']:.2f}"))
                                change_color = "text-green-400" if result["change"] >= 0 else "text-red-400"
                                ui.element('td').classes(f"p-2 {change_color}").add(
                                    ui.label(f"{result['change']:+.1f}%"))
                                ui.element('td').classes("p-2").add(ui.label(f"{result['rsi']:.1f}"))
                                ui.element('td').classes("p-2").add(ui.label(f"{result['pe']:.1f}"))
                                ui.element('td').classes("p-2").add(ui.label(f"{result['volume_ratio']:.1f}x"))
                                ui.element('td').classes("p-2").add(
                                    ui.button("Analyze", icon="analytics",
                                              on_click=lambda s=result["symbol"]: analyze_stock(s)).classes(
                                        "bg-blue-600 px-3 py-1 text-xs")
                                )


async def analyze_stock(symbol: str):
    """Navigate to detailed analysis for a specific stock"""
    chart_state["selected_symbol"] = symbol
    await update_chart_and_metrics(symbol)
    ui.navigate.to('/analytics')


# Export functionality for reports

async def export_analysis_report(symbol: str, metrics: StockMetrics, df: pd.DataFrame):
    """Export comprehensive analysis report"""
    try:
        report_data = {
            "symbol": symbol,
            "analysis_date": datetime.now().isoformat(),
            "current_price": metrics.current_price,
            "overall_signal": metrics.overall_signal,
            "confidence_score": metrics.confidence_score,
            "technical_indicators": {
                "rsi": metrics.rsi,
                "macd_signal": metrics.macd_signal,
                "bollinger_position": metrics.bollinger_position,
                "trend_signal": metrics.trend_signal,
                "momentum_signal": metrics.momentum_signal
            },
            "fundamental_data": {
                "pe_ratio": metrics.pe_ratio,
                "pb_ratio": metrics.pb_ratio,
                "market_cap": metrics.market_cap,
                "dividend_yield": metrics.dividend_yield
            },
            "recommendations": generate_trading_recommendations(metrics, df)
        }

        # Convert to JSON for export
        report_json = json.dumps(report_data, indent=2)

        # In a real implementation, you'd save this to a file or send via API
        logger.info(f"Analysis report generated for {symbol}")
        return report_json

    except Exception as e:
        logger.error(f"Error exporting analysis report: {e}")
        return None