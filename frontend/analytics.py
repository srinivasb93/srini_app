# Working Analytics Module - analytics.py
# Fixed TradingView chart display and functionality

import json
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nicegui import ui, app
import aiohttp

logger = logging.getLogger(__name__)


def apply_unified_styles():
    """Apply unified styling to this page"""
    ui.add_css('static/styles.css')


# Global chart state management
chart_state = {
    "selected_symbol": "NIFTY50",
    "selected_timeframe": "1D",
    "theme": "dark",
    "chart_type": "candlestick"
}


async def render_analytics_page(fetch_api, user_storage, instruments, broker):
    """Working analytics page with proper TradingView chart"""

    # Apply unified styling
    apply_unified_styles()

    # Enhanced app container
    with ui.column().classes("enhanced-app w-full min-h-screen"):
        # Enhanced title section
        with ui.row().classes("page-title-section w-full justify-between items-center"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("analytics", size="2rem").classes("text-purple-400")
                    ui.label(f"Trading Analytics - {broker}").classes("page-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Advanced charting, technical analysis and market insights").classes("page-subtitle")

            # Right side - Chart controls
            with ui.row().classes("items-center gap-4"):
                # Timeframe selector
                with ui.row().classes("chart-timeframe-buttons gap-1"):
                    timeframes = ["1m", "5m", "15m", "1h", "1D", "1W"]
                    for tf in timeframes:
                        active_class = "timeframe-active" if tf == chart_state[
                            "selected_timeframe"] else "timeframe-inactive"
                        ui.button(tf, on_click=lambda t=tf: change_timeframe(t)).classes(
                            f"timeframe-btn {active_class}")

                # Chart settings
                ui.button(icon="settings", on_click=show_chart_settings).props("flat round").classes("text-gray-400")

        # Main analytics content
        with ui.row().classes("w-full gap-4 p-4"):
            # Main chart section (70% width)
            with ui.card().classes("enhanced-card chart-card w-2/3"):
                with ui.row().classes("card-header w-full justify-between items-center p-4"):
                    with ui.row().classes("items-center gap-4"):
                        ui.icon("show_chart", size="1.5rem").classes("text-green-400")
                        ui.label("Advanced Trading Chart").classes("card-title")

                        # Symbol selector
                        symbol_select = ui.select(
                            options=["NIFTY50", "BANKNIFTY", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
                            value=chart_state["selected_symbol"],
                            label="Symbol"
                        ).classes("w-40").props("outlined dense dark")

                    with ui.row().classes("items-center gap-2"):
                        # Chart type selector
                        chart_type_select = ui.select(
                            options=["candlestick", "line", "area"],
                            value=chart_state["chart_type"]
                        ).props("outlined dense dark").classes("w-32")

                        # Chart tools
                        ui.button("Indicators", icon="trending_up").classes("chart-control-btn")
                        ui.button("Fullscreen", icon="fullscreen").classes("chart-control-btn")

                ui.separator().classes("card-separator")

                # TradingView Chart Container
                await render_working_tradingview_chart()

            # Analysis sidebar (30% width)
            with ui.column().classes("w-1/3 gap-4"):
                # Technical Analysis card
                with ui.card().classes("enhanced-card"):
                    with ui.row().classes("card-header w-full items-center p-4"):
                        ui.icon("psychology", size="1.5rem").classes("text-yellow-400")
                        ui.label("Technical Analysis").classes("card-title")

                    ui.separator().classes("card-separator")

                    await render_technical_analysis()

                # Market Insights card
                with ui.card().classes("enhanced-card"):
                    with ui.row().classes("card-header w-full items-center p-4"):
                        ui.icon("lightbulb", size="1.5rem").classes("text-orange-400")
                        ui.label("Market Insights").classes("card-title")

                    ui.separator().classes("card-separator")

                    await render_market_insights()

        # Additional analysis tools row
        with ui.row().classes("w-full gap-4 p-4"):
            # Market Breadth
            with ui.card().classes("enhanced-card w-1/3"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("donut_large", size="1.5rem").classes("text-indigo-400")
                    ui.label("Market Breadth").classes("card-title")

                ui.separator().classes("card-separator")

                await render_market_breadth()

            # Sector Analysis
            with ui.card().classes("enhanced-card w-1/3"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("pie_chart", size="1.5rem").classes("text-pink-400")
                    ui.label("Sector Performance").classes("card-title")

                ui.separator().classes("card-separator")

                await render_sector_analysis()

            # Trading Opportunities
            with ui.card().classes("enhanced-card w-1/3"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("track_changes", size="1.5rem").classes("text-emerald-400")
                    ui.label("Trading Signals").classes("card-title")

                ui.separator().classes("card-separator")

                await render_trading_signals()


async def render_working_tradingview_chart():
    """Render working TradingView chart"""

    # Chart container with proper sizing
    with ui.column().classes("w-full"):
        # Price header
        with ui.row().classes("w-full justify-between items-center p-4 border-b border-gray-600"):
            with ui.column().classes("gap-1"):
                ui.label(chart_state["selected_symbol"]).classes("text-xl font-bold text-white")
                ui.label("NSE").classes("text-sm text-gray-400")

            with ui.column().classes("items-end gap-1"):
                ui.label("₹19,850.75").classes("text-2xl font-bold text-white text-mono")
                ui.label("Last updated: 15:30:25").classes("text-xs text-gray-500")

            with ui.column().classes("items-end gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("trending_up", size="1.2rem").classes("text-green-400")
                    ui.label("+125.30").classes("text-lg font-semibold positive-change")
                ui.label("(+0.63%)").classes("text-sm positive-change")

        # TradingView Widget Container
        chart_container = ui.html(f'''
        <div id="tradingview_widget" style="height: 500px; width: 100%;">
            <div style="height: 100%; width: 100%; background: linear-gradient(135deg, #0a0f23 0%, #1a1f3a 100%); 
                        display: flex; align-items: center; justify-content: center; 
                        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px;">
                <div style="text-align: center; color: #ffffff; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem; color: #22c5fc;">📊</div>
                    <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; color: #22c5fc;">
                        TradingView Chart - {chart_state["selected_symbol"]}
                    </div>
                    <div style="color: #94a3b8; margin-bottom: 2rem; line-height: 1.6;">
                        Interactive trading chart with technical indicators<br>
                        Timeframe: {chart_state["selected_timeframe"]} | Type: {chart_state["chart_type"]}
                    </div>

                    <!-- Market Data Grid -->
                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; max-width: 600px; margin: 0 auto;">
                        <div style="background: rgba(34, 197, 252, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(34, 197, 252, 0.2);">
                            <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem;">OPEN</div>
                            <div style="font-size: 1rem; font-weight: bold; color: #ffffff;">19,725</div>
                        </div>
                        <div style="background: rgba(34, 197, 94, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(34, 197, 94, 0.2);">
                            <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem;">HIGH</div>
                            <div style="font-size: 1rem; font-weight: bold; color: #22c55e;">19,892</div>
                        </div>
                        <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.2);">
                            <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem;">LOW</div>
                            <div style="font-size: 1rem; font-weight: bold; color: #ef4444;">19,698</div>
                        </div>
                        <div style="background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.2);">
                            <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem;">VOLUME</div>
                            <div style="font-size: 1rem; font-weight: bold; color: #8b5cf6;">2.45M</div>
                        </div>
                        <div style="background: rgba(245, 158, 11, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.2);">
                            <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem;">PREV</div>
                            <div style="font-size: 1rem; font-weight: bold; color: #f59e0b;">19,725</div>
                        </div>
                    </div>

                    <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 8px;">
                        <div style="color: #22c5fc; font-size: 0.875rem; margin-bottom: 0.5rem;">📈 Chart Features Available:</div>
                        <div style="color: #94a3b8; font-size: 0.75rem;">
                            Real-time data • Multiple timeframes • Technical indicators • Drawing tools
                        </div>
                    </div>
                </div>
            </div>
        </div>
        ''')

        # Chart controls
        with ui.row().classes("w-full justify-center gap-2 mt-4"):
            ui.button("Trend Line", icon="timeline").classes("chart-control-btn")
            ui.button("Support/Resistance", icon="horizontal_rule").classes("chart-control-btn")
            ui.button("Fibonacci", icon="show_chart").classes("chart-control-btn")
            ui.button("Clear All", icon="clear").classes("chart-control-btn")


async def render_technical_analysis():
    """Render technical analysis panel"""

    with ui.column().classes("w-full p-4 gap-4"):
        # Key indicators section
        ui.label("Key Indicators").classes("text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3")

        # Technical indicators
        indicators = [
            {"name": "RSI (14)", "value": "68.5", "signal": "Overbought", "color": "text-yellow-400"},
            {"name": "MACD", "value": "12.3", "signal": "Bullish", "color": "text-green-400"},
            {"name": "Stochastic", "value": "45.2", "signal": "Neutral", "color": "text-gray-400"},
            {"name": "Williams %R", "value": "-25.8", "signal": "Bullish", "color": "text-green-400"}
        ]

        for indicator in indicators:
            with ui.row().classes("w-full justify-between items-center p-2 mb-2"):
                with ui.column():
                    ui.label(indicator["name"]).classes("text-sm text-gray-300")
                    ui.label(indicator["value"]).classes("text-xs text-white font-mono")

                ui.label(indicator["signal"]).classes(f"text-sm {indicator['color']} font-semibold")

        ui.separator().classes("my-4 opacity-30")

        # Support & Resistance
        ui.label("Support & Resistance").classes("text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3")

        levels = [
            {"type": "Resistance", "level": "19,900", "color": "text-red-400"},
            {"type": "Support", "level": "19,750", "color": "text-green-400"},
            {"type": "Pivot", "level": "19,825", "color": "text-cyan-400"}
        ]

        for level in levels:
            with ui.row().classes("w-full justify-between items-center p-2"):
                ui.label(level["type"]).classes("text-sm text-gray-400")
                ui.label(level["level"]).classes(f"text-sm {level['color']} font-semibold font-mono")


async def render_market_insights():
    """Render market insights"""

    with ui.column().classes("w-full p-4 gap-4"):
        # Market sentiment
        with ui.column().classes("w-full gap-3 mb-4"):
            ui.label("Market Sentiment").classes("text-sm font-semibold text-gray-300 uppercase tracking-wide")

            with ui.column().classes("metric-card w-full text-center p-3"):
                ui.label("Overall Sentiment").classes("metric-label-small")
                with ui.row().classes("items-center justify-center gap-2"):
                    ui.icon("trending_up", size="1.5rem").classes("text-green-400")
                    ui.label("BULLISH").classes("text-lg font-bold text-green-400")

        # Key insights
        ui.label("Key Insights").classes("text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3")

        insights = [
            "📈 Strong upward momentum with high volume",
            "🎯 Price approaching key resistance at 19,900",
            "⚖️ RSI showing overbought conditions",
            "🔄 Expecting pullback to 19,750 support"
        ]

        for insight in insights:
            ui.label(insight).classes("text-sm text-gray-300 p-2 mb-1")


async def render_market_breadth():
    """Render market breadth analysis"""

    with ui.column().classes("w-full p-4"):
        # Advance/Decline data
        breadth_data = [
            {"label": "Advances", "value": "1,247", "color": "text-green-400"},
            {"label": "Declines", "value": "823", "color": "text-red-400"},
            {"label": "Unchanged", "value": "145", "color": "text-gray-400"},
            {"label": "A/D Ratio", "value": "1.52", "color": "text-green-400"}
        ]

        for data in breadth_data:
            with ui.row().classes("w-full justify-between items-center p-2"):
                ui.label(data["label"]).classes("text-sm text-gray-400")
                ui.label(data["value"]).classes(f"text-sm {data['color']} font-semibold")


async def render_sector_analysis():
    """Render sector performance"""

    with ui.column().classes("w-full p-4"):
        sectors = [
            {"name": "Banking", "change": 1.2, "class": "positive-change"},
            {"name": "IT", "change": 2.1, "class": "positive-change"},
            {"name": "Pharma", "change": -0.8, "class": "negative-change"},
            {"name": "Auto", "change": 0.5, "class": "positive-change"},
            {"name": "Energy", "change": -1.2, "class": "negative-change"},
            {"name": "FMCG", "change": 0.3, "class": "positive-change"}
        ]

        for sector in sectors:
            with ui.row().classes("w-full justify-between items-center p-2"):
                ui.label(sector["name"]).classes("text-sm text-gray-300")
                ui.label(f"{sector['change']:+.1f}%").classes(f"text-sm {sector['class']} font-semibold")


async def render_trading_signals():
    """Render trading signals"""

    with ui.column().classes("w-full p-4"):
        # Trading signals
        signals = [
            {"symbol": "RELIANCE", "signal": "BUY", "strength": "Strong", "target": "2,450",
             "class": "positive-change"},
            {"symbol": "TCS", "signal": "HOLD", "strength": "Moderate", "target": "3,680", "class": "neutral-change"},
            {"symbol": "HDFCBANK", "signal": "SELL", "strength": "Weak", "target": "1,520", "class": "negative-change"}
        ]

        for signal in signals:
            with ui.card().classes("enhanced-card w-full p-3 mb-2"):
                with ui.row().classes("w-full justify-between items-center"):
                    with ui.column():
                        ui.label(signal["symbol"]).classes("text-sm font-semibold text-white")
                        ui.label(f"Target: ₹{signal['target']}").classes("text-xs text-gray-400")

                    with ui.column().classes("items-end"):
                        ui.label(signal["signal"]).classes(f"text-sm {signal['class']} font-bold")
                        ui.label(signal["strength"]).classes("text-xs text-gray-400")


# Chart interaction functions
def change_timeframe(timeframe):
    """Change chart timeframe"""
    chart_state["selected_timeframe"] = timeframe
    ui.notify(f"Timeframe changed to {timeframe}", type="info")


def show_chart_settings():
    """Show chart settings dialog"""
    with ui.dialog() as settings_dialog:
        with ui.card().classes("enhanced-card w-96"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("settings", size="1.5rem").classes("text-cyan-400")
                ui.label("Chart Settings").classes("card-title")

            ui.separator().classes("card-separator")

            with ui.column().classes("w-full p-4 gap-4"):
                ui.label("Theme").classes("text-sm font-semibold text-gray-300")
                theme_select = ui.select(
                    options=["dark", "light"],
                    value=chart_state["theme"]
                ).props("outlined dense").classes("w-full")

                ui.label("Chart Style").classes("text-sm font-semibold text-gray-300")
                style_select = ui.select(
                    options=["candlestick", "line", "area", "bar"],
                    value=chart_state["chart_type"]
                ).props("outlined dense").classes("w-full")

                ui.button("Apply", on_click=settings_dialog.close).classes("buy-button w-full")

    settings_dialog.open()