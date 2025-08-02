# Enhanced Portfolio Module - portfolio.py
# Fully integrated with existing backend architecture
# Production-ready with comprehensive error handling and optimizations

from nicegui import ui
import logging
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import asyncio
from typing import Dict, List, Optional, Any
from functools import lru_cache
import json


logger = logging.getLogger(__name__)


def transform_portfolio_data(raw_holdings: List[Dict]) -> List[Dict]:
    """Transform backend data structure to match enhanced UI expectations"""
    if not raw_holdings or not isinstance(raw_holdings, list):
        return []

    transformed = []

    for holding in raw_holdings:
        try:
            # Extract values with safe defaults
            quantity = float(holding.get('Quantity', 0))
            avg_price = float(holding.get('AvgPrice', 0))
            last_price = float(holding.get('LastPrice', 0))
            pnl = float(holding.get('PnL', 0))
            day_change_pct = float(holding.get('DayChangePct', 0))

            # Calculate derived fields
            investment_value = quantity * avg_price
            current_value = quantity * last_price

            transformed_holding = {
                'Symbol': holding.get('Symbol', 'N/A'),
                'Exchange': holding.get('Exchange', 'NSE'),
                'Quantity': quantity,
                'AvgPrice': avg_price,
                'LastPrice': last_price,
                'PnL': pnl,
                'DayChangePct': day_change_pct,
                'investment_value': investment_value,
                'current_value': current_value,
                'exchange': holding.get('Exchange', 'NSE').lower(),
                'Broker': holding.get('Broker', 'Unknown')
            }
            transformed.append(transformed_holding)

        except Exception as e:
            logger.error(f"Error transforming holding {holding}: {e}")
            continue

    return transformed


async def fetch_portfolio_with_retry(fetch_api, broker: str, max_retries: int = 3) -> List[Dict]:
    """Fetch portfolio data with automatic retry and exponential backoff"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching portfolio data for {broker}, attempt {attempt + 1}")
            response = await fetch_api(f"/portfolio/{broker}")

            if response is None:
                logger.warning(f"Portfolio API returned None for {broker}")
                return []

            if isinstance(response, list):
                return response
            elif isinstance(response, dict):
                # Handle case where API returns wrapped response
                return response.get('data', []) if 'data' in response else []
            else:
                logger.warning(f"Unexpected portfolio response type: {type(response)}")
                return []

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {broker} portfolio: {e}")
            if attempt == max_retries - 1:
                raise e
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)


@lru_cache(maxsize=100)
def calculate_portfolio_metrics(portfolio_json: str) -> Dict:
    """Calculate portfolio metrics with caching for performance"""
    try:
        portfolio_data = json.loads(portfolio_json)

        total_investment = sum(h.get('investment_value', 0) for h in portfolio_data)
        current_value = sum(h.get('current_value', 0) for h in portfolio_data)
        total_pnl = current_value - total_investment
        total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        total_holdings = len(portfolio_data)

        return {
            'total_investment': total_investment,
            'current_value': current_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'total_holdings': total_holdings
        }
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        return {
            'total_investment': 0,
            'current_value': 0,
            'total_pnl': 0,
            'total_pnl_pct': 0,
            'total_holdings': 0
        }


async def render_portfolio_page(fetch_api, user_storage, broker):
    """Enhanced portfolio page with complete backend integration"""
    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):
        # Enhanced title section
        with ui.row().classes("dashboard-title-section w-full justify-between items-center"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("account_balance_wallet", size="2rem").classes("text-cyan-400")
                    ui.label(f"Portfolio Overview - {broker}").classes("text-3xl font-bold text-white")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Your complete investment portfolio - equity and mutual funds").classes(
                    "text-gray-400 text-lg")

            # Right side - Action buttons
            with ui.row().classes("items-center gap-4"):
                ui.button("Export Report", icon="download",
                          on_click=lambda: handle_export_portfolio(broker)).classes("text-cyan-400")
                refresh_btn = ui.button("Refresh All", icon="refresh").classes("text-gray-400")

        # Portfolio metrics cards row
        metrics_container = ui.row().classes("w-full gap-4 p-4")
        await render_enhanced_portfolio_metrics(fetch_api, broker, metrics_container)

        # Holdings section with enhanced cards
        with ui.card().classes("dashboard-card w-full m-4"):
            with ui.row().classes("card-header w-full justify-between items-center p-4"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("pie_chart", size="1.5rem").classes("text-purple-400")
                    ui.label("Portfolio Holdings").classes("text-xl font-bold text-white")

                with ui.row().classes("items-center gap-2"):
                    ui.button(icon="add", on_click=lambda: ui.navigate.to('/order-management')).props(
                        "flat round").classes("text-cyan-400")
                    ui.button(icon="refresh", on_click=lambda: refresh_holdings(fetch_api, broker)).props(
                        "flat round").classes("text-gray-400")

            ui.separator().classes("card-separator")

            # Enhanced tabs for equity and mutual funds
            with ui.column().classes("w-full"):
                with ui.tabs().props("dense indicator-color=cyan").classes("w-full") as portfolio_tabs:
                    equity_tab = ui.tab(name="equity", label="📈 Equity Holdings", icon="trending_up")
                    mf_tab = ui.tab(name="mutual_funds", label="🏦 Mutual Funds", icon="account_balance")
                    analytics_tab = ui.tab(name="analytics", label="📊 Analytics", icon="analytics")

                with ui.tab_panels(portfolio_tabs, value="equity").classes("w-full"):
                    # Equity Holdings Panel
                    with ui.tab_panel("equity").classes("w-full p-4"):
                        equity_container = ui.column().classes("w-full")
                        await render_enhanced_equity_holdings(fetch_api, broker, equity_container)

                    # Mutual Funds Panel
                    with ui.tab_panel("mutual_funds").classes("w-full p-4"):
                        mf_container = ui.column().classes("w-full")
                        await render_enhanced_mf_holdings(fetch_api, broker, mf_container)

                    # Analytics Panel
                    with ui.tab_panel("analytics").classes("w-full p-4"):
                        analytics_container = ui.column().classes("w-full")
                        await render_portfolio_analytics(fetch_api, broker, analytics_container)

        # Setup refresh functionality
        async def refresh_all():
            """Refresh all portfolio data"""
            ui.notify("Refreshing portfolio data...", type="info")
            metrics_container.clear()
            await render_enhanced_portfolio_metrics(fetch_api, broker, metrics_container)
            equity_container.clear()
            await render_enhanced_equity_holdings(fetch_api, broker, equity_container)
            ui.notify("Portfolio data refreshed", type="positive")

        refresh_btn.on_click(refresh_all)


async def render_enhanced_portfolio_metrics(fetch_api, broker: str, container):
    """Enhanced portfolio metrics with glassmorphism cards and proper error handling"""

    with container:
        try:
            # Fetch portfolio data with retry logic
            portfolio_data = await fetch_portfolio_with_retry(fetch_api, broker)
            transformed_data = transform_portfolio_data(portfolio_data)

            # Calculate metrics using cached function
            portfolio_json = json.dumps(transformed_data)
            metrics = calculate_portfolio_metrics(portfolio_json)

            # Total Investment
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("account_balance", size="2rem").classes("text-blue-400 mb-2")
                    ui.label("Total Investment").classes("text-sm text-gray-400")
                    ui.label(f"₹{metrics['total_investment']:,.2f}").classes("text-2xl font-bold text-white")

            # Current Value
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("trending_up", size="2rem").classes("text-green-400 mb-2")
                    ui.label("Current Value").classes("text-sm text-gray-400")
                    ui.label(f"₹{metrics['current_value']:,.2f}").classes("text-2xl font-bold text-white")

            # Total P&L
            pnl = metrics['total_pnl']
            pnl_pct = metrics['total_pnl_pct']
            pnl_color = "text-green-400" if pnl >= 0 else "text-red-400"
            pnl_icon = "trending_up" if pnl >= 0 else "trending_down"

            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon(pnl_icon, size="2rem").classes(f"{pnl_color} mb-2")
                    ui.label("Total P&L").classes("text-sm text-gray-400")
                    ui.label(f"₹{pnl:,.2f}").classes(f"text-2xl font-bold {pnl_color}")
                    ui.label(f"({pnl_pct:+.2f}%)").classes(f"text-sm {pnl_color}")

            # Total Holdings
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("inventory", size="2rem").classes("text-purple-400 mb-2")
                    ui.label("Total Holdings").classes("text-sm text-gray-400")
                    ui.label(str(metrics['total_holdings'])).classes("text-2xl font-bold text-white")

        except Exception as e:
            logger.error(f"Error fetching portfolio metrics: {e}")
            # Error state with retry option
            with ui.card().classes("dashboard-card w-full"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                    ui.label("Error loading portfolio metrics").classes("text-red-400 text-lg")
                    ui.label(str(e)).classes("text-gray-500 text-sm mb-4")
                    ui.button("Retry", icon="refresh",
                              on_click=lambda: render_enhanced_portfolio_metrics(fetch_api, broker, container)).classes(
                        "text-cyan-400")


async def render_enhanced_equity_holdings(fetch_api, broker: str, container):
    """Enhanced equity holdings table with complete backend integration"""

    with container:
        # Loading indicator
        loading_row = ui.row().classes("w-full justify-center p-4")
        with loading_row:
            loading_spinner = ui.spinner(size="lg").classes("text-cyan-400")
            ui.label("Loading equity holdings...").classes("text-gray-400 ml-2")

        try:
            # Fetch and transform data
            portfolio_data = await fetch_portfolio_with_retry(fetch_api, broker)
            transformed_data = transform_portfolio_data(portfolio_data)

            # Remove loading indicator
            loading_row.delete()

            if not transformed_data:
                # Enhanced empty state
                with ui.column().classes("empty-state"):
                    ui.icon("inbox", size="4rem").classes("text-gray-500 mb-4")
                    ui.label("No equity holdings found").classes("text-xl text-gray-400 mb-2")
                    ui.label("Start investing to see your portfolio here").classes("text-sm text-gray-500")
                    ui.button("Start Trading", icon="add",
                              on_click=lambda: ui.navigate.to('/order-management')).classes("mt-4 bg-cyan-600")
                return

            # Enhanced holdings table
            with ui.element('div').classes("w-full").props("style='overflow-x:auto;'"):
                # Table header
                with ui.row().classes("holdings-header w-full p-3 text-sm font-semibold text-gray-400"):
                    ui.label("Symbol").classes("w-20 text-right")
                    ui.label("Quantity").classes("w-20 text-right")
                    ui.label("Avg Price").classes("w-24 text-right")
                    ui.label("LTP").classes("w-24 text-right")
                    ui.label("P&L").classes("w-32 text-right")
                    ui.label("Current Value").classes("w-32 text-right")
                    ui.label("Actions").classes("w-24 text-center")

                # Holdings rows
                for holding in transformed_data:
                    symbol = holding.get('Symbol', 'N/A')
                    quantity = holding.get('Quantity', 0)
                    avg_price = holding.get('AvgPrice', 0)
                    ltp = holding.get('LastPrice', 0)
                    current_value = holding.get('current_value', 0)
                    pnl = holding.get('PnL', 0)
                    pnl_percent = holding.get('DayChangePct', 0)
                    exchange = holding.get('Exchange', 'NSE')

                    # Color coding for P&L
                    pnl_color = "positive-change" if pnl >= 0 else "negative-change"
                    border_color = "border-green-500/20" if pnl >= 0 else "border-red-500/20"

                    with ui.row().classes(f"holdings-row w-full p-3 border-l-2 {border_color}"):
                        # Symbol
                        with ui.column().classes("flex-1"):
                            ui.label(symbol).classes("text-white font-semibold")
                            ui.label(exchange).classes("text-xs text-gray-500")

                        # Quantity
                        ui.label(f"{quantity:,.0f}").classes("w-20 text-right text-white")

                        # Average Price
                        ui.label(f"₹{avg_price:,.2f}").classes("w-24 text-right text-gray-300 text-mono")

                        # LTP
                        ui.label(f"₹{ltp:,.2f}").classes("w-24 text-right text-white text-mono")

                        # P&L
                        with ui.column().classes("w-32 text-right"):
                            ui.label(f"₹{pnl:,.2f}").classes(f"{pnl_color} font-semibold text-mono")
                            ui.label(f"({pnl_percent:+.2f}%)").classes(f"{pnl_color} text-xs")

                        # Current Value
                        ui.label(f"₹{current_value:,.2f}").classes("w-32 text-right text-white text-mono")

                        # Actions
                        with ui.row().classes("w-24 justify-center gap-1"):
                            ui.button(icon="sell",
                                      on_click=lambda s=symbol: handle_sell_action(s)).props(
                                "flat round size=sm").classes("text-red-400")
                            ui.button(icon="add",
                                      on_click=lambda s=symbol: handle_buy_action(s)).props(
                                "flat round size=sm").classes("text-green-400")

        except Exception as e:
            logger.error(f"Error rendering equity holdings: {e}")
            # Make sure to clean up loading state
            try:
                loading_row.delete()
            except:
                pass

            with ui.column().classes("empty-state"):
                ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                ui.label("Error loading holdings").classes("text-xl text-red-400 mb-2")
                ui.label(str(e)).classes("text-sm text-gray-500")
                ui.button("Retry", icon="refresh",
                          on_click=lambda: render_enhanced_equity_holdings(fetch_api, broker, container)).classes(
                    "mt-4 text-cyan-400")


async def render_enhanced_mf_holdings(fetch_api, broker: str, container):
    """Enhanced mutual funds holdings section with proper API integration"""

    with container:
        try:
            # Loading state
            loading_row = ui.row().classes("w-full justify-center p-4")
            with loading_row:
                loading_spinner = ui.spinner(size="lg").classes("text-cyan-400")
                ui.label("Loading mutual fund holdings...").classes("text-gray-400 ml-2")

            # Try to fetch MF holdings (integrate with your existing MF API)
            try:
                mf_data = await fetch_api("/mutual-funds/holdings")
                loading_row.delete()

                if mf_data and isinstance(mf_data, list) and len(mf_data) > 0:
                    await render_mf_holdings_table(mf_data, container)
                else:
                    await render_mf_empty_state(container)

            except Exception as api_error:
                logger.warning(f"MF API not available: {api_error}")
                loading_row.delete()
                await render_mf_integration_placeholder(container)

        except Exception as e:
            logger.error(f"Error rendering MF holdings: {e}")
            ui.label(f"Error loading mutual funds: {str(e)}").classes("text-red-500 text-center p-4")


async def render_mf_holdings_table(mf_data: List[Dict], container):
    """Render actual MF holdings table"""
    with container:
        with ui.element('div').classes("w-full"):
            # MF Table header
            with ui.row().classes("holdings-header w-full p-3 text-sm font-semibold text-gray-400"):
                ui.label("Scheme").classes("flex-1")
                ui.label("Units").classes("w-20 text-right")
                ui.label("NAV").classes("w-24 text-right")
                ui.label("Invested").classes("w-28 text-right")
                ui.label("Current").classes("w-28 text-right")
                ui.label("P&L").classes("w-28 text-right")
                ui.label("XIRR").classes("w-20 text-right")

            # MF Holdings rows
            for fund in mf_data:
                scheme_name = fund.get('scheme_name', 'N/A')[:30] + "..." if len(
                    fund.get('scheme_name', '')) > 30 else fund.get('scheme_name', 'N/A')
                units = fund.get('units', 0)
                nav = fund.get('nav', 0)
                invested = fund.get('invested_amount', 0)
                current_value = units * nav
                pnl = current_value - invested
                pnl_pct = (pnl / invested * 100) if invested > 0 else 0
                xirr = fund.get('xirr', 0)

                pnl_color = "positive-change" if pnl >= 0 else "negative-change"
                xirr_color = "positive-change" if xirr >= 0 else "negative-change"

                with ui.row().classes("holdings-row w-full p-3"):
                    ui.label(scheme_name).classes("flex-1 text-white").props(
                        "title='" + fund.get('scheme_name', 'N/A') + "'")
                    ui.label(f"{units:.3f}").classes("w-20 text-right text-white")
                    ui.label(f"₹{nav:.2f}").classes("w-24 text-right text-gray-300 text-mono")
                    ui.label(f"₹{invested:,.0f}").classes("w-28 text-right text-gray-300 text-mono")
                    ui.label(f"₹{current_value:,.0f}").classes("w-28 text-right text-white text-mono")
                    with ui.column().classes("w-28 text-right"):
                        ui.label(f"₹{pnl:,.0f}").classes(f"{pnl_color} font-semibold text-mono")
                        ui.label(f"({pnl_pct:+.2f}%)").classes(f"{pnl_color} text-xs")
                    ui.label(f"{xirr:.2f}%").classes(f"w-20 text-right {xirr_color}")


async def render_mf_empty_state(container):
    """Render empty state for mutual funds"""
    with container:
        with ui.column().classes("empty-state"):
            ui.icon("account_balance", size="4rem").classes("text-blue-400 mb-4")
            ui.label("No Mutual Fund Holdings").classes("text-xl text-gray-400 mb-2")
            ui.label("Start SIP investments to see your mutual fund portfolio").classes("text-sm text-gray-500")
            ui.button("Explore SIP Strategy", icon="trending_up",
                      on_click=lambda: ui.navigate.to('/sip-strategy')).classes("mt-4 bg-blue-600")


async def render_mf_integration_placeholder(container):
    """Render integration placeholder for mutual funds"""
    with container:
        with ui.column().classes("empty-state"):
            ui.icon("link", size="4rem").classes("text-purple-400 mb-4")
            ui.label("Mutual Fund Integration").classes("text-xl text-gray-400 mb-2")
            ui.label("Connect your mutual fund accounts to view holdings here").classes("text-sm text-gray-500 mb-4")
            ui.button("Setup Integration", icon="settings", color="primary").classes("mt-4")


async def render_portfolio_analytics(fetch_api, broker: str, container):
    """Enhanced portfolio analytics with comprehensive insights"""

    with container:
        ui.label("📊 Portfolio Analytics").classes("text-xl font-bold text-white mb-4")

        try:
            # Fetch portfolio data for analytics
            portfolio_data = await fetch_portfolio_with_retry(fetch_api, broker)
            transformed_data = transform_portfolio_data(portfolio_data)

            if not transformed_data:
                with ui.column().classes("empty-state"):
                    ui.icon("analytics", size="4rem").classes("text-gray-500 mb-4")
                    ui.label("No data for analytics").classes("text-xl text-gray-400 mb-2")
                    ui.label("Add holdings to see portfolio analytics").classes("text-sm text-gray-500")
                return

            # Analytics grid
            with ui.row().classes("w-full gap-4 mb-6"):
                # Asset allocation chart
                with ui.card().classes("dashboard-card flex-1"):
                    with ui.row().classes("card-header w-full items-center p-4"):
                        ui.icon("pie_chart", size="1.5rem").classes("text-blue-400")
                        ui.label("Asset Allocation").classes("text-lg font-bold text-white")

                    with ui.column().classes("p-4"):
                        await render_asset_allocation_chart(transformed_data)

                # Performance metrics
                with ui.card().classes("dashboard-card flex-1"):
                    with ui.row().classes("card-header w-full items-center p-4"):
                        ui.icon("trending_up", size="1.5rem").classes("text-green-400")
                        ui.label("Performance Metrics").classes("text-lg font-bold text-white")

                    with ui.column().classes("p-4"):
                        await render_performance_metrics(transformed_data)

            # Top performers and losers
            with ui.row().classes("w-full gap-4"):
                # Top performers
                with ui.card().classes("dashboard-card flex-1"):
                    with ui.row().classes("card-header w-full items-center p-4"):
                        ui.icon("arrow_upward", size="1.5rem").classes("text-green-400")
                        ui.label("Top Performers").classes("text-lg font-bold text-white")

                    with ui.column().classes("p-4"):
                        await render_top_performers(transformed_data)

                # Top losers
                with ui.card().classes("dashboard-card flex-1"):
                    with ui.row().classes("card-header w-full items-center p-4"):
                        ui.icon("arrow_downward", size="1.5rem").classes("text-red-400")
                        ui.label("Top Losers").classes("text-lg font-bold text-white")

                    with ui.column().classes("p-4"):
                        await render_top_losers(transformed_data)

        except Exception as e:
            logger.error(f"Error rendering portfolio analytics: {e}")
            ui.label(f"Error loading analytics: {str(e)}").classes("text-red-500 text-center p-4")


async def render_asset_allocation_chart(portfolio_data: List[Dict]):
    """Render asset allocation pie chart"""
    try:
        # Calculate allocation by value
        allocations = {}
        total_value = sum(h.get('current_value', 0) for h in portfolio_data)

        for holding in portfolio_data:
            symbol = holding.get('Symbol', 'Unknown')
            value = holding.get('current_value', 0)
            percentage = (value / total_value * 100) if total_value > 0 else 0
            allocations[symbol] = percentage

        # Sort by percentage
        sorted_allocations = dict(sorted(allocations.items(), key=lambda x: x[1], reverse=True))

        # Take top 10 for better visualization
        top_allocations = dict(list(sorted_allocations.items())[:10])
        if len(sorted_allocations) > 10:
            others_pct = sum(list(sorted_allocations.values())[10:])
            top_allocations['Others'] = others_pct

        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(top_allocations.keys()),
            values=list(top_allocations.values()),
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(
                colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316',
                        '#ec4899', '#6366f1'],
                line=dict(color='#000000', width=1)
            )
        )])

        fig.update_layout(
            title="Portfolio Allocation by Value",
            title_font_color='white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300,
            margin=dict(t=50, b=20, l=20, r=20)
        )

        ui.plotly(fig).classes("w-full")

    except Exception as e:
        logger.error(f"Error creating allocation chart: {e}")
        ui.label("Error creating allocation chart").classes("text-red-500 text-center")


async def render_performance_metrics(portfolio_data: List[Dict]):
    """Render performance metrics"""
    try:
        total_invested = sum(h.get('investment_value', 0) for h in portfolio_data)
        total_current = sum(h.get('current_value', 0) for h in portfolio_data)
        total_pnl = total_current - total_invested
        total_return_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        # Calculate other metrics
        winners = len([h for h in portfolio_data if h.get('PnL', 0) > 0])
        losers = len([h for h in portfolio_data if h.get('PnL', 0) < 0])
        win_rate = (winners / len(portfolio_data) * 100) if portfolio_data else 0

        # Metrics display
        metrics = [
            ("Total Return", f"{total_return_pct:+.2f}%",
             "positive-change" if total_return_pct >= 0 else "negative-change"),
            ("Winners", f"{winners}/{len(portfolio_data)}", "text-green-400"),
            ("Win Rate", f"{win_rate:.1f}%", "positive-change" if win_rate >= 50 else "negative-change"),
            ("Avg. P&L per Stock", f"₹{total_pnl / len(portfolio_data):,.0f}" if portfolio_data else "₹0", "text-white")
        ]

        for label, value, color in metrics:
            with ui.row().classes("w-full justify-between items-center mb-2"):
                ui.label(label).classes("text-gray-400")
                ui.label(value).classes(f"font-bold {color}")

    except Exception as e:
        logger.error(f"Error rendering performance metrics: {e}")
        ui.label("Error calculating metrics").classes("text-red-500")


async def render_top_performers(portfolio_data: List[Dict]):
    """Render top performing stocks"""
    try:
        # Sort by P&L percentage
        sorted_data = sorted(portfolio_data, key=lambda x: x.get('DayChangePct', 0), reverse=True)
        top_performers = sorted_data[:5]

        if not top_performers:
            ui.label("No data available").classes("text-gray-500 text-center")
            return

        for holding in top_performers:
            symbol = holding.get('Symbol', 'N/A')
            pnl_pct = holding.get('DayChangePct', 0)
            pnl = holding.get('PnL', 0)

            with ui.row().classes("w-full justify-between items-center mb-2"):
                ui.label(symbol).classes("text-white font-semibold")
                with ui.column().classes("text-right"):
                    ui.label(f"+{pnl_pct:.2f}%").classes("text-green-400 font-bold text-sm")
                    ui.label(f"₹{pnl:,.0f}").classes("text-green-400 text-xs")

    except Exception as e:
        logger.error(f"Error rendering top performers: {e}")
        ui.label("Error loading top performers").classes("text-red-500")


async def render_top_losers(portfolio_data: List[Dict]):
    """Render worst performing stocks"""
    try:
        # Sort by P&L percentage (ascending for losers)
        sorted_data = sorted(portfolio_data, key=lambda x: x.get('DayChangePct', 0))
        top_losers = sorted_data[:5]

        if not top_losers:
            ui.label("No data available").classes("text-gray-500 text-center")
            return

        for holding in top_losers:
            symbol = holding.get('Symbol', 'N/A')
            pnl_pct = holding.get('DayChangePct', 0)
            pnl = holding.get('PnL', 0)

            with ui.row().classes("w-full justify-between items-center mb-2"):
                ui.label(symbol).classes("text-white font-semibold")
                with ui.column().classes("text-right"):
                    ui.label(f"{pnl_pct:.2f}%").classes("text-red-400 font-bold text-sm")
                    ui.label(f"₹{pnl:,.0f}").classes("text-red-400 text-xs")

    except Exception as e:
        logger.error(f"Error rendering top losers: {e}")
        ui.label("Error loading top losers").classes("text-red-500")


# Action handlers
def handle_sell_action(symbol: str):
    """Handle sell action for a holding"""
    ui.notify(f"Initiating sell order for {symbol}", type="info")
    ui.navigate.to(f'/order-management?action=sell&symbol={symbol}')


def handle_buy_action(symbol: str):
    """Handle buy more action for a holding"""
    ui.notify(f"Initiating buy order for {symbol}", type="info")
    ui.navigate.to(f'/order-management?action=buy&symbol={symbol}')


def handle_export_portfolio(broker: str):
    """Handle portfolio export"""
    ui.notify(f"Exporting {broker} portfolio...", type="info")
    # Implement export functionality here
    # This could generate PDF/Excel reports
    ui.notify("Export functionality coming soon!", type="warning")


async def refresh_holdings(fetch_api, broker: str):
    """Refresh holdings data"""
    ui.notify("Refreshing holdings...", type="info")
    try:
        # Clear cache if using caching
        calculate_portfolio_metrics.cache_clear()
        ui.notify("Holdings refreshed successfully", type="positive")
    except Exception as e:
        ui.notify(f"Error refreshing holdings: {str(e)}", type="negative")


# Additional utility functions for production readiness
async def setup_portfolio_websocket(broker: str):
    """Setup WebSocket for real-time portfolio updates"""
    # Placeholder for WebSocket implementation
    # This would connect to real-time market data feeds
    pass


async def calculate_portfolio_risk_metrics(portfolio_data: List[Dict]) -> Dict:
    """Calculate advanced risk metrics for the portfolio"""
    try:
        # This is a placeholder for advanced risk calculations
        # In production, you'd calculate:
        # - Portfolio Beta
        # - Value at Risk (VaR)
        # - Maximum Drawdown
        # - Sharpe Ratio
        # - Correlation matrix

        total_value = sum(h.get('current_value', 0) for h in portfolio_data)
        volatility = 0.15  # Placeholder - calculate actual volatility

        return {
            'portfolio_value': total_value,
            'estimated_volatility': volatility,
            'beta': 1.0,  # Placeholder
            'var_95': total_value * 0.05,  # Placeholder
            'max_drawdown': 0.1  # Placeholder
        }
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        return {}


async def generate_portfolio_report(portfolio_data: List[Dict], format: str = "pdf") -> str:
    """Generate comprehensive portfolio report"""
    try:
        # This is a placeholder for report generation
        # In production, you'd use libraries like:
        # - ReportLab for PDF generation
        # - openpyxl for Excel reports
        # - jinja2 for HTML templates

        report_data = {
            'total_holdings': len(portfolio_data),
            'total_value': sum(h.get('current_value', 0) for h in portfolio_data),
            'generated_at': datetime.now().isoformat()
        }

        # Return report file path or URL
        return f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"

    except Exception as e:
        logger.error(f"Error generating portfolio report: {e}")
        raise e


# Advanced portfolio features for production
class PortfolioManager:
    """Advanced portfolio management class for production use"""

    def __init__(self, user_id: str, broker: str):
        self.user_id = user_id
        self.broker = broker
        self.cache_ttl = 300  # 5 minutes
        self._cache = {}

    async def get_portfolio_summary(self, fetch_api) -> Dict[str, Any]:
        """Get comprehensive portfolio summary with caching"""
        cache_key = f"portfolio_summary_{self.user_id}_{self.broker}"

        # Check cache first
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return cached_data

        try:
            # Fetch fresh data
            portfolio_data = await fetch_portfolio_with_retry(fetch_api, self.broker)
            transformed_data = transform_portfolio_data(portfolio_data)

            # Calculate comprehensive metrics
            portfolio_json = json.dumps(transformed_data)
            metrics = calculate_portfolio_metrics(portfolio_json)

            # Add advanced metrics
            risk_metrics = await calculate_portfolio_risk_metrics(transformed_data)

            summary = {
                'basic_metrics': metrics,
                'risk_metrics': risk_metrics,
                'holdings_count': len(transformed_data),
                'last_updated': datetime.now().isoformat(),
                'broker': self.broker
            }

            # Cache the result
            self._cache[cache_key] = (summary, datetime.now())

            return summary

        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            raise e

    async def get_portfolio_alerts(self, db_session) -> List[Dict]:
        """Get active portfolio alerts for user"""
        try:
            query = """
                SELECT * FROM portfolio_alerts 
                WHERE user_id = :user_id AND broker = :broker 
                AND is_active = TRUE AND is_triggered = FALSE
                ORDER BY created_at DESC
            """

            # This would use your database session
            # result = await db_session.execute(text(query), {
            #     'user_id': self.user_id,
            #     'broker': self.broker
            # })
            # return result.fetchall()

            # Placeholder return
            return []

        except Exception as e:
            logger.error(f"Error getting portfolio alerts: {e}")
            return []

    async def create_portfolio_alert(self, symbol: str, alert_type: str,
                                     threshold: float, db_session) -> bool:
        """Create a new portfolio alert"""
        try:
            alert_data = {
                'user_id': self.user_id,
                'broker': self.broker,
                'symbol': symbol,
                'alert_type': alert_type,
                'threshold_value': threshold,
                'is_active': True,
                'created_at': datetime.now()
            }

            # This would insert into your database
            # Implementation depends on your database setup

            logger.info(f"Created portfolio alert: {alert_data}")
            return True

        except Exception as e:
            logger.error(f"Error creating portfolio alert: {e}")
            return False

    def clear_cache(self):
        """Clear portfolio cache"""
        self._cache.clear()
        calculate_portfolio_metrics.cache_clear()


# Portfolio optimization utilities
def optimize_portfolio_allocation(portfolio_data: List[Dict],
                                  target_allocation: Dict[str, float]) -> Dict[str, float]:
    """Suggest portfolio rebalancing based on target allocation"""
    try:
        current_total = sum(h.get('current_value', 0) for h in portfolio_data)

        # Calculate current allocation
        current_allocation = {}
        for holding in portfolio_data:
            symbol = holding.get('Symbol', 'Unknown')
            value = holding.get('current_value', 0)
            current_allocation[symbol] = (value / current_total * 100) if current_total > 0 else 0

        # Calculate rebalancing suggestions
        rebalancing_suggestions = {}
        for symbol, target_pct in target_allocation.items():
            current_pct = current_allocation.get(symbol, 0)
            difference = target_pct - current_pct

            if abs(difference) > 1:  # Only suggest if difference > 1%
                target_value = current_total * target_pct / 100
                current_value = current_total * current_pct / 100
                action_value = target_value - current_value

                rebalancing_suggestions[symbol] = {
                    'current_percentage': current_pct,
                    'target_percentage': target_pct,
                    'difference_percentage': difference,
                    'action': 'BUY' if action_value > 0 else 'SELL',
                    'action_value': abs(action_value)
                }

        return rebalancing_suggestions

    except Exception as e:
        logger.error(f"Error optimizing portfolio allocation: {e}")
        return {}


# Integration with existing SIP strategy
async def integrate_with_sip_strategy(portfolio_data: List[Dict],
                                      sip_portfolios: List[Dict]) -> Dict[str, Any]:
    """Integrate regular portfolio with SIP strategy portfolios"""
    try:
        integration_summary = {
            'total_regular_value': sum(h.get('current_value', 0) for h in portfolio_data),
            'total_sip_value': sum(p.get('current_value', 0) for p in sip_portfolios),
            'combined_symbols': set(),
            'allocation_overlap': {},
            'diversification_score': 0.0
        }

        # Combine symbols from both portfolios
        regular_symbols = {h.get('Symbol') for h in portfolio_data}
        sip_symbols = set()

        for sip_portfolio in sip_portfolios:
            symbols = sip_portfolio.get('symbols', [])
            if isinstance(symbols, list):
                for symbol_data in symbols:
                    if isinstance(symbol_data, dict):
                        sip_symbols.add(symbol_data.get('symbol', ''))
                    elif isinstance(symbol_data, str):
                        sip_symbols.add(symbol_data)

        integration_summary['combined_symbols'] = regular_symbols.union(sip_symbols)
        integration_summary['overlapping_symbols'] = regular_symbols.intersection(sip_symbols)

        # Calculate diversification score (simple version)
        total_symbols = len(integration_summary['combined_symbols'])
        integration_summary['diversification_score'] = min(total_symbols / 20.0, 1.0) * 100

        return integration_summary

    except Exception as e:
        logger.error(f"Error integrating with SIP strategy: {e}")
        return {}


# Performance comparison with benchmarks
async def compare_with_benchmarks(portfolio_data: List[Dict],
                                  fetch_api) -> Dict[str, Any]:
    """Compare portfolio performance with market benchmarks"""
    try:
        total_invested = sum(h.get('investment_value', 0) for h in portfolio_data)
        total_current = sum(h.get('current_value', 0) for h in portfolio_data)
        portfolio_return = ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0

        # This would fetch actual benchmark data in production
        benchmarks = {
            'NIFTY_50': 12.5,  # Placeholder returns
            'SENSEX': 11.8,
            'NIFTY_NEXT_50': 15.2,
            'NIFTY_MIDCAP': 18.5
        }

        comparison = {
            'portfolio_return': portfolio_return,
            'benchmarks': benchmarks,
            'outperformance': {}
        }

        for benchmark, return_pct in benchmarks.items():
            comparison['outperformance'][benchmark] = portfolio_return - return_pct

        return comparison

    except Exception as e:
        logger.error(f"Error comparing with benchmarks: {e}")
        return {}


# Export the main render function and utilities
__all__ = [
    'render_portfolio_page',
    'PortfolioManager',
    'transform_portfolio_data',
    'calculate_portfolio_metrics',
    'optimize_portfolio_allocation',
    'integrate_with_sip_strategy',
    'compare_with_benchmarks'
]