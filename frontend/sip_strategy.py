"""
SIP Strategy UI Module for NiceGUI Algo Trading Application
Implements comprehensive SIP strategy interface with backtesting, portfolio management, and signals
"""

from nicegui import ui
import pandas as pd
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import aiohttp

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

class SIPStrategyUI:
    """Main SIP Strategy UI Controller"""

    def __init__(self):
        self.current_results = {}
        self.active_portfolios = []
        self.signals = []

    async def render_sip_main_page(self, fetch_api, user_storage):
        """Render the main SIP strategy page with tabs"""

        with ui.tabs().classes('w-full') as tabs:
            backtest_tab = ui.tab('📊 Backtest Strategy')
            portfolio_tab = ui.tab('💼 My SIP Portfolios')
            signals_tab = ui.tab('🎯 Investment Signals')
            analytics_tab = ui.tab('📈 Performance Analytics')
            config_tab = ui.tab('⚙️ Strategy Configuration')

        with ui.tab_panels(tabs).classes('w-full'):
            with ui.tab_panel(backtest_tab):
                await self.render_backtest_panel(fetch_api, user_storage)

            with ui.tab_panel(portfolio_tab):
                await self.render_portfolio_panel(fetch_api, user_storage)

            with ui.tab_panel(signals_tab):
                await self.render_signals_panel(fetch_api, user_storage)

            with ui.tab_panel(analytics_tab):
                await self.render_analytics_panel(fetch_api, user_storage)

            with ui.tab_panel(config_tab):
                await self.render_config_panel(fetch_api, user_storage)

    async def render_backtest_panel(self, fetch_api, user_storage):
        """Render SIP strategy backtesting interface"""

        ui.label("SIP Strategy Backtesting").classes("text-2xl font-bold mb-4")

        with ui.card().classes("w-full mb-4"):
            ui.label("Configure Backtest Parameters").classes("text-lg font-semibold mb-2")

            with ui.row().classes("w-full gap-4"):
                with ui.column().classes("flex-1"):
                    # Symbol selection
                    symbols_input = ui.textarea(
                        label="Symbols (one per line)",
                        placeholder="ICICIB22\nHDFCNEXT50\nMOTILALOSML",
                        value="ICICIB22\nHDFCNEXT50"
                    ).classes("w-full")

                    # Date range
                    with ui.row().classes("w-full gap-2"):
                        start_date = ui.date(
                            value="2020-01-01"
                        ).classes("flex-1")

                        end_date = ui.date(
                            value=datetime.now().strftime("%Y-%m-%d")
                        ).classes("flex-1")

                with ui.column().classes("flex-1"):
                    # Strategy parameters
                    fixed_investment = ui.number(
                        label="Fixed Investment Amount",
                        value=5000,
                        min=100,
                        step=100
                    ).classes("w-full")

                    drawdown_threshold = ui.number(
                        label="Drawdown Threshold (%)",
                        value=-10,
                        min=-50,
                        max=-1,
                        step=1
                    ).classes("w-full")

                    with ui.row().classes("w-full gap-2"):
                        multiplier_2x = ui.number(
                            label="2x Multiplier",
                            value=2.0,
                            min=1.0,
                            step=0.1
                        ).classes("flex-1")

                        multiplier_3x = ui.number(
                            label="3x Multiplier",
                            value=3.0,
                            min=1.0,
                            step=0.1
                        ).classes("flex-1")

                        multiplier_5x = ui.number(
                            label="5x Multiplier",
                            value=5.0,
                            min=1.0,
                            step=0.1
                        ).classes("flex-1")

        # Results container
        results_container = ui.column().classes("w-full")

        # Run backtest button
        async def run_backtest():
            with results_container:
                results_container.clear()

                # Show loading
                with ui.card().classes("w-full"):
                    ui.spinner(size="lg")
                    ui.label("Running backtest... This may take a few minutes").classes("text-lg")

                try:
                    # Prepare request data
                    symbol_list = [s.strip() for s in symbols_input.value.split('\n') if s.strip()]

                    request_data = {
                        "symbols": symbol_list,
                        "start_date": start_date.value,
                        "end_date": end_date.value,
                        "config": {
                            "fixed_investment": fixed_investment.value,
                            "drawdown_threshold_1": drawdown_threshold.value,
                            "investment_multiplier_1": multiplier_2x.value,
                            "investment_multiplier_2": multiplier_3x.value,
                            "investment_multiplier_3": multiplier_5x.value
                        }
                    }

                    # Call API
                    response = await fetch_api("/sip/backtest", method="POST", data=request_data)

                    if response:
                        self.current_results = {item['symbol']: item for item in response}
                        await self.display_backtest_results(results_container)
                    else:
                        results_container.clear()
                        with ui.card().classes("w-full"):
                            ui.label("No results returned. Please check your symbols and date range.").classes("text-red-500")

                except Exception as e:
                    results_container.clear()
                    with ui.card().classes("w-full"):
                        ui.label(f"Error running backtest: {str(e)}").classes("text-red-500")

        ui.button("🚀 Run Backtest", on_click=run_backtest).classes("bg-blue-500 text-white px-6 py-2 rounded")

        # Display results container
        results_container

    async def display_backtest_results(self, container):
        """Display backtest results in a formatted table and charts"""

        container.clear()

        if not self.current_results:
            return

        # Summary statistics
        with ui.card().classes("w-full mb-4"):
            ui.label("Backtest Results Summary").classes("text-xl font-bold mb-4")

            # Create summary table
            summary_data = []
            for symbol, result in self.current_results.items():
                summary_data.append({
                    'Symbol': symbol,
                    'Total Investment': f"₹{result['total_investment']:,.0f}",
                    'Final Value': f"₹{result['final_portfolio_value']:,.0f}",
                    'Total Return': f"{((result['final_portfolio_value']/result['total_investment']-1)*100):,.1f}%",
                    'CAGR': f"{result['cagr']*100:.2f}%",
                    'Max Drawdown': f"{result['max_drawdown']*100:.2f}%" if result['max_drawdown'] else "N/A",
                    'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}" if result['sharpe_ratio'] else "N/A",
                    'Trades': result['num_trades']
                })

            # Display as table
            with ui.element('div').classes('overflow-x-auto'):
                table = ui.table(
                    columns=[
                        {'name': 'symbol', 'label': 'Symbol', 'field': 'Symbol'},
                        {'name': 'investment', 'label': 'Total Investment', 'field': 'Total Investment'},
                        {'name': 'value', 'label': 'Final Value', 'field': 'Final Value'},
                        {'name': 'return', 'label': 'Total Return', 'field': 'Total Return'},
                        {'name': 'cagr', 'label': 'CAGR', 'field': 'CAGR'},
                        {'name': 'drawdown', 'label': 'Max Drawdown', 'field': 'Max Drawdown'},
                        {'name': 'sharpe', 'label': 'Sharpe Ratio', 'field': 'Sharpe Ratio'},
                        {'name': 'trades', 'label': 'Trades', 'field': 'Trades'}
                    ],
                    rows=summary_data
                ).classes('w-full')

        # Performance comparison chart
        with ui.card().classes("w-full mb-4"):
            ui.label("Performance Comparison").classes("text-lg font-bold mb-2")

            # Create comparison chart
            symbols = list(self.current_results.keys())
            cagr_values = [self.current_results[s]['cagr'] * 100 for s in symbols]
            total_returns = [((self.current_results[s]['final_portfolio_value']/self.current_results[s]['total_investment'])-1)*100 for s in symbols]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='CAGR (%)',
                x=symbols,
                y=cagr_values,
                marker_color='lightblue'
            ))

            fig.add_trace(go.Bar(
                name='Total Return (%)',
                x=symbols,
                y=total_returns,
                marker_color='lightgreen'
            ))

            fig.update_layout(
                title='Strategy Performance Comparison',
                xaxis_title='Symbols',
                yaxis_title='Return (%)',
                barmode='group',
                height=400
            )

            ui.plotly(fig).classes('w-full')

        # Action buttons
        with ui.row().classes("gap-4 mt-4"):
            async def create_portfolio_from_backtest():
                # Implementation for creating live portfolio from backtest
                ui.notify("Portfolio creation feature coming soon!", type="info")

            async def export_results():
                # Implementation for exporting results
                ui.notify("Export feature coming soon!", type="info")

            ui.button("💼 Create Live Portfolio", on_click=create_portfolio_from_backtest).classes("bg-green-500 text-white")
            ui.button("📊 Export Results", on_click=export_results).classes("bg-gray-500 text-white")

    async def render_portfolio_panel(self, fetch_api, user_storage):
        """Render SIP portfolio management interface"""

        ui.label("My SIP Portfolios").classes("text-2xl font-bold mb-4")

        # Create new portfolio section
        with ui.card().classes("w-full mb-4"):
            ui.label("Create New SIP Portfolio").classes("text-lg font-semibold mb-2")

            with ui.row().classes("w-full gap-4"):
                symbol_input = ui.input(
                    label="Symbol",
                    placeholder="e.g., ICICIB22"
                ).classes("flex-1")

                portfolio_name_input = ui.input(
                    label="Portfolio Name (Optional)",
                    placeholder="e.g., My Conservative SIP"
                ).classes("flex-1")

                investment_amount = ui.number(
                    label="Monthly Investment",
                    value=5000,
                    min=100
                ).classes("flex-1")

            async def create_portfolio():
                try:
                    request_data = {
                        "symbol": symbol_input.value,
                        "portfolio_name": portfolio_name_input.value,
                        "config": {
                            "fixed_investment": investment_amount.value
                        }
                    }

                    response = await fetch_api("/sip/portfolio", method="POST", data=request_data)

                    if response:
                        ui.notify(f"Portfolio created successfully! ID: {response['portfolio_id']}", type="positive")
                        await refresh_portfolios()
                        # Clear inputs
                        symbol_input.value = ""
                        portfolio_name_input.value = ""

                except Exception as e:
                    ui.notify(f"Error creating portfolio: {str(e)}", type="negative")

            ui.button("✨ Create Portfolio", on_click=create_portfolio).classes("bg-blue-500 text-white")

        # Existing portfolios
        portfolios_container = ui.column().classes("w-full")

        async def refresh_portfolios():
            """Refresh portfolio list"""
            try:
                portfolios = await fetch_api("/sip/portfolio")

                portfolios_container.clear()

                if portfolios:
                    self.active_portfolios = portfolios

                    with portfolios_container:
                        ui.label("Active Portfolios").classes("text-lg font-semibold mb-2")

                        for portfolio in portfolios:
                            await self.render_portfolio_card(portfolio, fetch_api)
                else:
                    with portfolios_container:
                        ui.label("No portfolios found. Create your first SIP portfolio above!").classes("text-gray-500")

            except Exception as e:
                logger.error(f"Error fetching portfolios: {e}")
                with portfolios_container:
                    ui.label("Error loading portfolios").classes("text-red-500")

        # Load portfolios on page load
        await refresh_portfolios()

        # Refresh button
        ui.button("🔄 Refresh", on_click=refresh_portfolios).classes("bg-gray-500 text-white mt-2")

    async def render_portfolio_card(self, portfolio, fetch_api):
        """Render individual portfolio card"""

        with ui.card().classes("w-full mb-4"):
            with ui.row().classes("w-full justify-between items-center"):
                with ui.column():
                    ui.label(f"{portfolio['symbol']} - {portfolio.get('portfolio_name', 'Unnamed Portfolio')}").classes("text-lg font-bold")
                    ui.label(f"Status: {portfolio['status'].title()}").classes("text-sm")

                with ui.column().classes("text-right"):
                    ui.label(f"Total Invested: ₹{portfolio['total_invested']:,.0f}").classes("text-sm")
                    ui.label(f"Current Value: ₹{portfolio['current_value']:,.0f}").classes("text-sm")
                    ui.label(f"Units: {portfolio['current_units']:.2f}").classes("text-sm")

            with ui.row().classes("gap-2 mt-2"):
                async def view_performance():
                    await self.show_portfolio_performance(portfolio['portfolio_id'], fetch_api)

                async def get_signals():
                    await self.show_portfolio_signals(portfolio['portfolio_id'], fetch_api)

                async def invest_now():
                    await self.show_investment_dialog(portfolio['portfolio_id'], fetch_api)

                ui.button("📈 Performance", on_click=view_performance).classes("bg-blue-500 text-white")
                ui.button("🎯 Signals", on_click=get_signals).classes("bg-green-500 text-white")
                ui.button("💰 Invest Now", on_click=invest_now).classes("bg-orange-500 text-white")

    async def show_portfolio_performance(self, portfolio_id, fetch_api):
        """Show detailed portfolio performance in a dialog"""

        try:
            performance = await fetch_api(f"/sip/performance/{portfolio_id}")

            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label("Portfolio Performance").classes("text-xl font-bold mb-4")

                # Performance metrics
                with ui.column().classes("w-full gap-2"):
                    ui.label(f"Symbol: {performance['symbol']}").classes("text-lg")
                    ui.separator()

                    with ui.row().classes("w-full justify-between"):
                        ui.label("Total Invested:")
                        ui.label(f"₹{performance['total_invested']:,.0f}").classes("font-bold")

                    with ui.row().classes("w-full justify-between"):
                        ui.label("Current Value:")
                        ui.label(f"₹{performance['current_value']:,.0f}").classes("font-bold")

                    with ui.row().classes("w-full justify-between"):
                        ui.label("Total Return:")
                        return_color = "text-green-500" if performance['total_return_percent'] >= 0 else "text-red-500"
                        ui.label(f"{performance['total_return_percent']:.2f}%").classes(f"font-bold {return_color}")

                    with ui.row().classes("w-full justify-between"):
                        ui.label("CAGR:")
                        cagr_color = "text-green-500" if performance['cagr_percent'] >= 0 else "text-red-500"
                        ui.label(f"{performance['cagr_percent']:.2f}%").classes(f"font-bold {cagr_color}")

                    with ui.row().classes("w-full justify-between"):
                        ui.label("Average Buy Price:")
                        ui.label(f"₹{performance['average_buy_price']:.2f}").classes("font-bold")

                    with ui.row().classes("w-full justify-between"):
                        ui.label("Number of Investments:")
                        ui.label(f"{performance['num_investments']}").classes("font-bold")

                    with ui.row().classes("w-full justify-between"):
                        ui.label("Days Invested:")
                        ui.label(f"{performance['days_invested']}").classes("font-bold")

                ui.button("Close", on_click=dialog.close).classes("mt-4")

            dialog.open()

        except Exception as e:
            ui.notify(f"Error loading performance: {str(e)}", type="negative")

    async def show_portfolio_signals(self, portfolio_id, fetch_api):
        """Show investment signals for portfolio"""

        try:
            signals = await fetch_api(f"/sip/signals/{portfolio_id}")

            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label("Investment Signals").classes("text-xl font-bold mb-4")

                if signals.get('should_invest'):
                    with ui.column().classes("w-full gap-2"):
                        ui.label("🎯 Investment Opportunity Detected!").classes("text-green-500 font-bold")
                        ui.separator()

                        with ui.row().classes("w-full justify-between"):
                            ui.label("Trade Type:")
                            ui.label(signals['trade_type']).classes("font-bold")

                        with ui.row().classes("w-full justify-between"):
                            ui.label("Recommended Amount:")
                            ui.label(f"₹{signals['recommended_amount']:,.0f}").classes("font-bold text-green-500")

                        with ui.row().classes("w-full justify-between"):
                            ui.label("Multiplier:")
                            ui.label(f"{signals['investment_multiplier']}x").classes("font-bold")

                        with ui.row().classes("w-full justify-between"):
                            ui.label("Current Price:")
                            ui.label(f"₹{signals['current_price']:.2f}").classes("font-bold")

                        if signals.get('drawdown_100'):
                            with ui.row().classes("w-full justify-between"):
                                ui.label("Drawdown:")
                                ui.label(f"{signals['drawdown_100']:.2f}%").classes("font-bold text-red-500")

                        with ui.row().classes("w-full justify-between"):
                            ui.label("RSI:")
                            ui.label(f"{signals.get('rsi', 0):.2f}").classes("font-bold")
                else:
                    ui.label("📅 No immediate investment signal. Next fallback investment:").classes("text-blue-500")
                    if signals.get('next_fallback_date'):
                        ui.label(signals['next_fallback_date']).classes("font-bold")

                ui.button("Close", on_click=dialog.close).classes("mt-4")

            dialog.open()

        except Exception as e:
            ui.notify(f"Error loading signals: {str(e)}", type="negative")

    async def show_investment_dialog(self, portfolio_id, fetch_api):
        """Show manual investment dialog"""

        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Manual Investment").classes("text-xl font-bold mb-4")

            amount_input = ui.number(
                label="Investment Amount",
                value=5000,
                min=100,
                step=100
            ).classes("w-full")

            ui.label("This will execute a manual SIP investment for this portfolio.").classes("text-sm text-gray-600 mt-2")

            async def execute_investment():
                try:
                    response = await fetch_api(
                        f"/sip/execute/{portfolio_id}",
                        method="POST",
                        data={"amount": amount_input.value}
                    )

                    if response:
                        ui.notify(
                            f"Investment successful! ₹{response['investment_amount']} invested, "
                            f"{response['units_purchased']:.4f} units purchased at ₹{response['price']:.2f}",
                            type="positive"
                        )
                        dialog.close()

                except Exception as e:
                    ui.notify(f"Error executing investment: {str(e)}", type="negative")

            with ui.row().classes("gap-2 mt-4"):
                ui.button("💰 Invest", on_click=execute_investment).classes("bg-green-500 text-white")
                ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white")

        dialog.open()

    async def render_signals_panel(self, fetch_api, user_storage):
        """Render investment signals dashboard"""

        ui.label("Investment Signals Dashboard").classes("text-2xl font-bold mb-4")

        signals_container = ui.column().classes("w-full")

        async def refresh_signals():
            """Refresh all investment signals"""
            try:
                signals = await fetch_api("/sip/signals?active_only=true")

                signals_container.clear()

                if signals:
                    self.signals = signals

                    with signals_container:
                        # Summary stats
                        with ui.card().classes("w-full mb-4"):
                            ui.label("Signal Summary").classes("text-lg font-semibold mb-2")

                            high_signals = [s for s in signals if s['signal_strength'] == 'high']
                            medium_signals = [s for s in signals if s['signal_strength'] == 'medium']

                            with ui.row().classes("gap-4"):
                                with ui.card().classes("bg-red-100 p-4"):
                                    ui.label(f"{len(high_signals)}").classes("text-2xl font-bold text-red-600")
                                    ui.label("High Priority").classes("text-sm")

                                with ui.card().classes("bg-yellow-100 p-4"):
                                    ui.label(f"{len(medium_signals)}").classes("text-2xl font-bold text-yellow-600")
                                    ui.label("Medium Priority").classes("text-sm")

                                with ui.card().classes("bg-blue-100 p-4"):
                                    ui.label(f"{len(signals)}").classes("text-2xl font-bold text-blue-600")
                                    ui.label("Total Active").classes("text-sm")

                        # Individual signals
                        ui.label("Active Investment Signals").classes("text-lg font-semibold mb-2")

                        for signal in signals:
                            await self.render_signal_card(signal, fetch_api)
                else:
                    with signals_container:
                        ui.label("No active investment signals found.").classes("text-gray-500")
                        ui.label("Signals will appear here when investment opportunities are detected.").classes("text-sm text-gray-400")

            except Exception as e:
                logger.error(f"Error fetching signals: {e}")
                with signals_container:
                    ui.label("Error loading signals").classes("text-red-500")

        # Load signals on page load
        await refresh_signals()

        # Auto-refresh toggle
        auto_refresh = ui.switch("Auto-refresh (30s)", value=False)

        async def auto_refresh_loop():
            while auto_refresh.value:
                await asyncio.sleep(30)
                if auto_refresh.value:
                    await refresh_signals()

        auto_refresh.on('update:value', lambda: asyncio.create_task(auto_refresh_loop()))

        ui.button("🔄 Refresh Now", on_click=refresh_signals).classes("bg-blue-500 text-white mt-2")

    async def render_signal_card(self, signal, fetch_api):
        """Render individual signal card"""

        # Determine card color based on signal strength
        card_color = {
            'high': 'border-l-4 border-red-500',
            'medium': 'border-l-4 border-yellow-500',
            'low': 'border-l-4 border-green-500'
        }.get(signal['signal_strength'], 'border-l-4 border-gray-500')

        with ui.card().classes(f"w-full mb-2 {card_color}"):
            with ui.row().classes("w-full justify-between items-center"):
                with ui.column():
                    ui.label(f"{signal['symbol']} - {signal['signal_type']}").classes("text-lg font-bold")
                    ui.label(f"Strength: {signal['signal_strength'].title()}").classes("text-sm")
                    ui.label(f"Created: {signal['created_at'][:19]}").classes("text-xs text-gray-500")

                with ui.column().classes("text-right"):
                    ui.label(f"Recommended: ₹{signal['recommended_amount']:,.0f}").classes("font-bold text-green-600")
                    ui.label(f"Multiplier: {signal['multiplier']}x").classes("text-sm")
                    ui.label(f"Price: ₹{signal['current_price']:.2f}").classes("text-sm")

                    if signal.get('drawdown_percent'):
                        ui.label(f"Drawdown: {signal['drawdown_percent']:.1f}%").classes("text-sm text-red-500")

    async def render_analytics_panel(self, fetch_api, user_storage):
        """Render performance analytics and reporting"""

        ui.label("Performance Analytics").classes("text-2xl font-bold mb-4")

        # Backtest history
        with ui.card().classes("w-full mb-4"):
            ui.label("Recent Backtest Results").classes("text-lg font-semibold mb-2")

            try:
                backtest_history = await fetch_api("/sip/backtest/history?limit=10")

                if backtest_history:
                    # Create table for backtest history
                    table_data = []
                    for bt in backtest_history:
                        table_data.append({
                            'Date': bt['created_at'][:10],
                            'Symbol': bt['symbol'],
                            'Strategy': bt['strategy_name'],
                            'Investment': f"₹{bt['total_investment']:,.0f}",
                            'Final Value': f"₹{bt['final_portfolio_value']:,.0f}",
                            'CAGR': f"{bt['cagr']*100:.2f}%",
                            'Trades': bt['num_trades']
                        })

                    ui.table(
                        columns=[
                            {'name': 'date', 'label': 'Date', 'field': 'Date'},
                            {'name': 'symbol', 'label': 'Symbol', 'field': 'Symbol'},
                            {'name': 'strategy', 'label': 'Strategy', 'field': 'Strategy'},
                            {'name': 'investment', 'label': 'Investment', 'field': 'Investment'},
                            {'name': 'value', 'label': 'Final Value', 'field': 'Final Value'},
                            {'name': 'cagr', 'label': 'CAGR', 'field': 'CAGR'},
                            {'name': 'trades', 'label': 'Trades', 'field': 'Trades'}
                        ],
                        rows=table_data
                    ).classes('w-full')
                else:
                    ui.label("No backtest history found. Run some backtests to see results here.").classes("text-gray-500")

            except Exception as e:
                ui.label("Error loading backtest history").classes("text-red-500")

        # Portfolio performance summary
        with ui.card().classes("w-full"):
            ui.label("Live Portfolio Summary").classes("text-lg font-semibold mb-2")

            try:
                portfolios = await fetch_api("/sip/portfolio")

                if portfolios:
                    total_invested = sum(p['total_invested'] for p in portfolios)
                    total_value = sum(p['current_value'] for p in portfolios)
                    total_return = ((total_value / total_invested) - 1) * 100 if total_invested > 0 else 0

                    with ui.row().classes("gap-4"):
                        with ui.card().classes("bg-blue-100 p-4"):
                            ui.label(f"₹{total_invested:,.0f}").classes("text-xl font-bold text-blue-600")
                            ui.label("Total Invested").classes("text-sm")

                        with ui.card().classes("bg-green-100 p-4"):
                            ui.label(f"₹{total_value:,.0f}").classes("text-xl font-bold text-green-600")
                            ui.label("Current Value").classes("text-sm")

                        with ui.card().classes("bg-purple-100 p-4"):
                            return_color = "text-green-600" if total_return >= 0 else "text-red-600"
                            ui.label(f"{total_return:.2f}%").classes(f"text-xl font-bold {return_color}")
                            ui.label("Total Return").classes("text-sm")

                        with ui.card().classes("bg-gray-100 p-4"):
                            ui.label(f"{len(portfolios)}").classes("text-xl font-bold text-gray-600")
                            ui.label("Active Portfolios").classes("text-sm")
                else:
                    ui.label("No active portfolios found.").classes("text-gray-500")

            except Exception as e:
                ui.label("Error loading portfolio summary").classes("text-red-500")

    async def render_config_panel(self, fetch_api, user_storage):
        """Render strategy configuration templates"""

        ui.label("Strategy Configuration").classes("text-2xl font-bold mb-4")

        with ui.card().classes("w-full mb-4"):
            ui.label("Default SIP Parameters").classes("text-lg font-semibold mb-2")
            ui.label("Configure default parameters for new SIP strategies").classes("text-sm text-gray-600 mb-4")

            # Configuration form
            with ui.row().classes("w-full gap-4"):
                with ui.column().classes("flex-1"):
                    fixed_inv = ui.number(label="Fixed Investment Amount", value=5000, min=100).classes("w-full")
                    drawdown_1 = ui.number(label="Primary Drawdown Threshold (%)", value=-10, max=-1).classes("w-full")
                    drawdown_2 = ui.number(label="Secondary Drawdown Step (%)", value=-4, max=-1).classes("w-full")

                with ui.column().classes("flex-1"):
                    mult_1 = ui.number(label="2x Investment Multiplier", value=2.0, min=1.0, step=0.1).classes("w-full")
                    mult_2 = ui.number(label="3x Investment Multiplier", value=3.0, min=1.0, step=0.1).classes("w-full")
                    mult_3 = ui.number(label="5x Investment Multiplier", value=5.0, min=1.0, step=0.1).classes("w-full")

                with ui.column().classes("flex-1"):
                    rolling_win = ui.number(label="Rolling Window (days)", value=100, min=20, max=200).classes("w-full")
                    fallback_day = ui.number(label="Fallback Day of Month", value=22, min=1, max=28).classes("w-full")

            async def save_config():
                config = {
                    'fixed_investment': fixed_inv.value,
                    'drawdown_threshold_1': drawdown_1.value,
                    'drawdown_threshold_2': drawdown_2.value,
                    'investment_multiplier_1': mult_1.value,
                    'investment_multiplier_2': mult_2.value,
                    'investment_multiplier_3': mult_3.value,
                    'rolling_window': rolling_win.value,
                    'fallback_day': fallback_day.value
                }

                # Save to user storage (you could also save to database)
                user_storage['sip_default_config'] = config
                ui.notify("Configuration saved successfully!", type="positive")

            ui.button("💾 Save Configuration", on_click=save_config).classes("bg-green-500 text-white mt-4")

        # Strategy templates (future enhancement)
        with ui.card().classes("w-full"):
            ui.label("Strategy Templates").classes("text-lg font-semibold mb-2")
            ui.label("Pre-configured strategy templates for different risk profiles").classes("text-sm text-gray-600 mb-4")

            with ui.row().classes("gap-4"):
                with ui.card().classes("p-4 cursor-pointer hover:bg-gray-50").on('click', lambda: ui.notify("Conservative template applied")):
                    ui.label("🛡️ Conservative").classes("font-bold text-green-600")
                    ui.label("Lower multipliers, higher thresholds").classes("text-sm")

                with ui.card().classes("p-4 cursor-pointer hover:bg-gray-50").on('click', lambda: ui.notify("Balanced template applied")):
                    ui.label("⚖️ Balanced").classes("font-bold text-blue-600")
                    ui.label("Moderate risk, balanced returns").classes("text-sm")

                with ui.card().classes("p-4 cursor-pointer hover:bg-gray-50").on('click', lambda: ui.notify("Aggressive template applied")):
                    ui.label("🚀 Aggressive").classes("font-bold text-red-600")
                    ui.label("Higher multipliers, lower thresholds").classes("text-sm")

# Main integration function
async def render_sip_strategy_page(fetch_api, user_storage, instruments=None):
    """Main function to render SIP strategy page"""

    sip_ui = SIPStrategyUI()
    await sip_ui.render_sip_main_page(fetch_api, user_storage)