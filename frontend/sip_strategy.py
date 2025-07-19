"""
Enhanced SIP Strategy Frontend with comprehensive improvements:
- Multi-symbol portfolio support
- Enhanced signal display with GTT status
- Better portfolio management with scheduler integration
- Improved user experience with benchmark comparisons
- Standalone implementation (no BaseStrategy dependency)
- Added monthly limits and GTT handling
- Scheduler status and control integration
- Enhanced reports with quick and comprehensive options
- Analytics with performance metrics and comparisons
- Added UI for batch backtest multi-configs, config optimization, quick test, templates, benchmark test, symbol availability, and scheduler job config
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui

logger = logging.getLogger(__name__)

class EnhancedSIPStrategy:
    """Enhanced SIP Strategy UI - Standalone Implementation"""

    def __init__(self):
        self.strategy_name = "Enhanced SIP Strategy"
        self.description = """
        Advanced SIP strategy with:
        • Multi-symbol portfolio support  
        • Dynamic investment amounts based on market drawdowns
        • Minimum 5-day gap between investments
        • Daily 8AM signal processing with GTT orders
        • Enhanced technical analysis (RSI, MACD, Bollinger Bands)
        • Comprehensive portfolio analytics with benchmarks
        • Monthly investment limits and force investments
        • Scheduler integration for automated processing
        """
        self.active_portfolios = []
        self.default_config = {
            "fixed_investment": 5000,
            "major_drawdown_threshold": -10,
            "minor_drawdown_threshold": -4,
            "extreme_drawdown_threshold": -15,
            "minor_drawdown_inv_multiplier": 1.75,
            "major_drawdown_inv_multiplier": 3,
            "extreme_drawdown_inv_multiplier": 4,
            "rolling_window": 100,
            "fallback_day": 28,
            "min_investment_gap_days": 5,
            "max_amount_in_a_month": None,  # Will default to 4x fixed_investment
            "price_reduction_threshold": 4.0,
            "force_remaining_investment": True
        }

        # UI state management
        self.is_loading = False
        self.last_error = None
        self.fetch_api = None

        logger.info(f"Initialized {self.__class__.__name__}")

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def show_loading(self, message: str = "Loading...") -> None:
        """Show loading state in UI"""
        self.is_loading = True
        ui.notify(message, type="info", timeout=1000)

    def hide_loading(self) -> None:
        """Hide loading state"""
        self.is_loading = False

    def show_error(self, error: str, details: Optional[str] = None) -> None:
        """Show error message to user"""
        self.last_error = error
        error_message = f"❌ {error}"
        if details:
            error_message += f"\nDetails: {details}"
        ui.notify(error_message, type="negative", timeout=5000)
        logger.error(f"{self.strategy_name} Error: {error} - {details}")

    def show_success(self, message: str) -> None:
        """Show success message to user"""
        ui.notify(f"✅ {message}", type="positive", timeout=3000)
        logger.info(f"{self.strategy_name} Success: {message}")

    def show_warning(self, message: str) -> None:
        """Show warning message to user"""
        ui.notify(f"⚠️ {message}", type="warning", timeout=4000)
        logger.warning(f"{self.strategy_name} Warning: {message}")

    async def safe_api_call(self, fetch_api: Callable, endpoint: str,
                           method: str = "GET", data: Optional[Dict] = None,
                           error_message: str = "API call failed") -> Optional[Dict]:
        """Safely make API calls with error handling"""
        try:
            self.show_loading()

            if method.upper() == "GET":
                response = await fetch_api(endpoint)
            else:
                response = await fetch_api(endpoint, method=method, data=data)

            self.hide_loading()
            return response

        except Exception as e:
            self.hide_loading()
            self.show_error(error_message, str(e))
            return None

    def format_currency(self, amount: float, currency: str = "₹") -> str:
        """Format currency amount for display"""
        if amount is None:
            return f"{currency}0"
        return f"{currency}{amount:,.2f}"

    def format_percentage(self, value: float, decimals: int = 2) -> str:
        """Format percentage for display"""
        if value is None:
            return "0.00%"
        return f"{value:.{decimals}f}%"

    async def render(self, fetch_api, user_storage):
        """Render enhanced SIP strategy interface"""
        self.fetch_api = fetch_api
        # Ensure full width
        with ui.tabs().classes("w-full max-w-full") as tabs:
            backtest_tab = ui.tab("📊 Backtesting", icon="analytics")
            portfolio_tab = ui.tab("💼 Portfolios", icon="account_balance_wallet")
            multi_portfolio_tab = ui.tab("🎯 Multi-Portfolio", icon="dashboard")
            reports_tab = ui.tab("📋 Reports", icon="description")
            signals_tab = ui.tab("📡 Signals", icon="notifications_active")
            analytics_tab = ui.tab("📈 Analytics", icon="trending_up")
            config_tab = ui.tab("⚙️ Configuration", icon="settings")
            scheduler_tab = ui.tab("⏰ Scheduler", icon="schedule")  # NEW TAB for scheduler controls

        with ui.tab_panels(tabs, value=backtest_tab).classes("w-full max-w-full"):
            with ui.tab_panel(backtest_tab).classes("w-full max-w-full"):
                await self.render_enhanced_backtest_panel(fetch_api, user_storage)

            with ui.tab_panel(portfolio_tab).classes("w-full max-w-full"):
                await self.render_enhanced_portfolio_panel(fetch_api, user_storage)

            with ui.tab_panel(multi_portfolio_tab).classes("w-full max-w-full"):
                await self.render_multi_portfolio_panel(fetch_api, user_storage)

            with ui.tab_panel(reports_tab).classes("w-full max-w-full"):
                await self.render_investment_reports_panel(fetch_api, user_storage)

            with ui.tab_panel(signals_tab).classes("w-full max-w-full"):
                await self.render_enhanced_signals_panel(fetch_api, user_storage)

            with ui.tab_panel(analytics_tab).classes("w-full max-w-full"):
                await self.render_enhanced_analytics_panel(fetch_api, user_storage)

            with ui.tab_panel(config_tab).classes("w-full max-w-full"):
                await self.render_enhanced_config_panel(fetch_api, user_storage)

            with ui.tab_panel(scheduler_tab).classes("w-full max-w-full"):  # NEW TAB
                await self.render_scheduler_panel(fetch_api, user_storage)

    async def render_scheduler_panel(self, fetch_api, user_storage):
        """Scheduler management panel with job trades view"""

        ui.label("⏰ Scheduler Management").classes("text-2xl font-bold mb-4")
        ui.label("Control and monitor the automated signal processing scheduler").classes("text-gray-600 mb-6")

        scheduler_container = ui.column().classes("w-full")

        async def refresh_scheduler_status():
            try:
                status = await self.safe_api_call(fetch_api, "/sip/scheduler/status")
                if status:
                    await self.display_scheduler_status(status, scheduler_container)
                else:
                    ui.notify("❌ Failed to fetch scheduler status", type="negative")
            except Exception as e:
                ui.notify(f"❌ Error: {str(e)}", type="negative")

        # Initial load
        await refresh_scheduler_status()

        # Control buttons
        with ui.row().classes("gap-4 mt-6"):
            ui.button("🔄 Refresh Status", on_click=refresh_scheduler_status).classes("bg-blue-500 text-white")
            ui.button("▶️ Start Scheduler", on_click=lambda: self.control_scheduler(fetch_api, "start")).classes("bg-green-500 text-white")
            ui.button("⏸️ Pause Scheduler", on_click=lambda: self.control_scheduler(fetch_api, "pause")).classes("bg-yellow-500 text-white")
            ui.button("🛑 Stop Scheduler", on_click=lambda: self.control_scheduler(fetch_api, "shutdown")).classes("bg-red-500 text-white")

        # Job list
        jobs_container = ui.column().classes("w-full mt-8")
        await self.display_scheduler_jobs(fetch_api, jobs_container)

        # Daily signal check configuration section
        with ui.card().classes("w-full mt-8 p-6"):
            ui.label("🛠️ Configure Daily Signal Check Job").classes("text-lg font-bold mb-4")

            with ui.row().classes("w-full gap-4"):
                trigger_type = ui.select(
                    options=["cron", "interval"],
                    value="interval",
                    label="Trigger Type"
                ).classes("flex-1")

                cron_expression = ui.input(
                    label="Cron Expression (for cron trigger)",
                    placeholder="0 9 * * 1-5",
                    value="0 9 * * 1-5"
                ).classes("flex-1")

                interval_minutes = ui.number(
                    label="Interval Minutes (for interval trigger)",
                    value=1,
                    min=1
                ).classes("flex-1")

                timezone = ui.input(
                    label="Timezone",
                    value="Asia/Kolkata"
                ).classes("flex-1")

                max_instances = ui.number(
                    label="Max Instances",
                    value=1,
                    min=1
                ).classes("flex-1")

            async def configure_daily_job():
                try:
                    request_data = {
                        "trigger_type": trigger_type.value,
                        "cron_expression": cron_expression.value if trigger_type.value == "cron" else None,
                        "interval_minutes": interval_minutes.value if trigger_type.value == "interval" else None,
                        "timezone": timezone.value,
                        "max_instances": max_instances.value
                    }

                    response = await self.safe_api_call(
                        fetch_api,
                        "/sip/scheduler/jobs/daily-signal-check/configure",
                        method="POST",
                        data=request_data
                    )

                    if response:
                        self.show_success(response.get("message", "Job configured successfully"))
                        await refresh_scheduler_status()
                    else:
                        self.show_error("Failed to configure job")

                except Exception as e:
                    self.show_error("Job configuration failed", str(e))

            ui.button("⚙️ Configure Job", on_click=configure_daily_job).classes("bg-purple-500 text-white mt-4")

        # View trades from daily signal check
        trades_container = ui.column().classes("w-full mt-8")

        async def view_scheduler_trades():
            try:
                trades = await self.safe_api_call(fetch_api, "/sip/scheduler/trades")  # Assuming this endpoint

                trades_container.clear()

                if trades:
                    with trades_container:
                        ui.label("📝 Trades from Daily Signal Check").classes("text-lg font-bold mb-4")
                        table = ui.table(columns=[
                            {'name': 'symbol', 'label': 'Symbol', 'field': 'symbol'},
                            {'name': 'amount', 'label': 'Amount', 'field': 'amount'},
                            {'name': 'timestamp', 'label': 'Date', 'field': 'timestamp'},
                            {'name': 'status', 'label': 'Status', 'field': 'execution_status'},
                            {'name': 'price', 'label': 'Price', 'field': 'price'},
                            {'name': 'units', 'label': 'Units', 'field': 'units'}
                        ], rows=trades).classes("w-full")
                else:
                    with trades_container:
                        ui.label("No trades found from scheduler").classes("text-gray-500")

            except Exception as e:
                self.show_error("Failed to load scheduler trades", str(e))

        ui.button("📝 View Scheduler Trades", on_click=view_scheduler_trades).classes("bg-indigo-500 text-white mt-4")

    async def display_scheduler_status(self, status: Dict, container: ui.column):
        container.clear()
        with container:
            with ui.card().classes("w-full p-4 mb-4"):
                ui.label("Scheduler Status").classes("text-lg font-bold mb-2")
                status_color = "text-green-600" if status["scheduler_running"] else "text-red-600"
                ui.label(f"Running: {status['scheduler_running']}").classes(f"text-sm {status_color}")
                ui.label(f"Total Jobs: {status['total_jobs']}").classes("text-sm")
                ui.label(f"Timezone: {status['timezone']}").classes("text-sm")
                ui.label(f"Uptime: {status['uptime_seconds'] / 3600:.1f} hours").classes("text-sm")

            # Active jobs
            if status["active_jobs"]:
                ui.label("Active Jobs").classes("text-md font-semibold mb-2")
                for job in status["active_jobs"]:
                    with ui.card().classes("p-2 mb-2"):
                        ui.label(f"Job ID: {job['job_id']}").classes("text-sm")
                        ui.label(f"Status: {job['status']}").classes("text-sm")
                        ui.label(f"Next Run: {job['next_run_time']}").classes("text-sm")

    async def display_scheduler_jobs(self, fetch_api, container: ui.column):
        container.clear()
        with container:
            ui.label("Scheduled Jobs").classes("text-lg font-bold mb-4")

            jobs = await self.safe_api_call(fetch_api, "/sip/scheduler/jobs")
            if jobs and jobs["jobs"]:
                for job in jobs["jobs"]:
                    with ui.card().classes("p-4 mb-2"):
                        ui.label(f"Job: {job['function_name']} ({job['job_id']})").classes("font-bold")
                        ui.label(f"Trigger: {job['trigger_type']} - {job['trigger_info']}").classes("text-sm")
                        ui.label(f"Next Run: {job['next_run_time']}").classes("text-sm")
                        with ui.row().classes("gap-2 mt-2"):
                            ui.button("▶️ Run Now", on_click=lambda j=job['job_id']: self.run_job_now(fetch_api, j)).classes("bg-blue-500 text-white text-xs")
                            ui.button("⏸️ Pause", on_click=lambda j=job['job_id']: self.control_job(fetch_api, j, "pause")).classes("bg-yellow-500 text-white text-xs")
                            ui.button("▶️ Resume", on_click=lambda j=job['job_id']: self.control_job(fetch_api, j, "resume")).classes("bg-green-500 text-white text-xs")
            else:
                ui.label("No jobs scheduled").classes("text-gray-500")

    async def control_scheduler(self, fetch_api, action: str):
        endpoint = f"/sip/scheduler/{action}"
        result = await self.safe_api_call(fetch_api, endpoint, method="POST")
        if result:
            self.show_success(result.get("message", f"Scheduler {action} successful"))
        else:
            self.show_error(f"Failed to {action} scheduler")

    async def run_job_now(self, fetch_api, job_id: str):
        result = await self.safe_api_call(fetch_api, f"/sip/scheduler/jobs/{job_id}/run", method="POST")
        if result:
            self.show_success(result.get("message", "Job triggered successfully"))
        else:
            self.show_error("Failed to trigger job")

    async def control_job(self, fetch_api, job_id: str, action: str):
        result = await self.safe_api_call(fetch_api, f"/sip/scheduler/jobs/{job_id}/{action}", method="POST")
        if result:
            self.show_success(result.get("message", f"Job {action} successful"))
        else:
            self.show_error(f"Failed to {action} job")

    async def render_enhanced_backtest_panel(self, fetch_api, user_storage):
        """Enhanced backtesting interface with benchmark comparison and monthly limits, using sub-tabs"""

        ui.label("🚀 Enhanced SIP Strategy Backtesting").classes("text-2xl font-bold mb-4")
        ui.label("Test your SIP strategy with dynamic investments, monthly limits, and benchmark comparison").classes("text-gray-600 mb-6")

        # Sub-tabs for different backtesting sections
        with ui.tabs().classes("w-full") as sub_tabs:
            main_backtest = ui.tab("Main Backtest")
            multi_configs = ui.tab("Batch Multi-Configs")
            optimize = ui.tab("Optimize Config")
            quick_test = ui.tab("Quick Test")
            benchmark = ui.tab("Benchmark Test")
            symbols = ui.tab("Symbols Search")

        with ui.tab_panels(sub_tabs, value=main_backtest).classes("w-full"):
            with ui.tab_panel(main_backtest).classes("w-full"):
                await self.render_main_backtest_section(fetch_api, user_storage)

            with ui.tab_panel(multi_configs).classes("w-full"):
                await self.render_batch_multi_configs_section(fetch_api, user_storage)

            with ui.tab_panel(optimize).classes("w-full"):
                await self.render_optimize_config_section(fetch_api, user_storage)

            with ui.tab_panel(quick_test).classes("w-full"):
                await self.render_quick_test_section(fetch_api, user_storage)

            with ui.tab_panel(benchmark).classes("w-full"):
                await self.render_benchmark_test_section(fetch_api, user_storage)

            with ui.tab_panel(symbols).classes("w-full"):
                await self.render_symbols_search_section(fetch_api, user_storage)

    async def clear_results(self):
        self.results_container.clear()
        ui.notify("Results cleared", type="info")

    async def render_main_backtest_section(self, fetch_api, user_storage):
        """Render main backtest configuration and results"""

        with ui.card().classes("w-full mb-4 p-6"):
            ui.label("📋 Backtest Configuration").classes("text-lg font-semibold mb-4")

            with ui.row().classes("w-full gap-6"):
                with ui.column().classes("flex-1"):
                    ui.label("Basic Settings").classes("font-medium mb-2")

                    self.symbols_input = ui.textarea(
                        label="📈 Symbols (one per line)",
                        placeholder="ICICIB22\nHDFCNEXT50\nMOTILALOSML\nNIFTYBEES",
                        value="ICICIB22\nGOLDBEES"
                    ).classes("w-full")

                    with ui.row().classes("w-full gap-2"):
                        self.start_date = ui.date(
                            value="2020-01-01"
                        ).classes("flex-1")

                        self.end_date = ui.date(
                            value=datetime.now().strftime("%Y-%m-%d")
                        ).classes("flex-1")

                    self.enable_monthly_limits = ui.checkbox("Enable Monthly Limits", value=True).classes("mt-2")

                with ui.column().classes("flex-1"):
                    ui.label("Strategy Parameters").classes("font-medium mb-2")

                    self.fixed_investment = ui.number(
                        label="💰 Monthly Investment (₹)",
                        value=self.default_config["fixed_investment"],
                        min=1000, step=500
                    ).classes("w-full")

                    with ui.row().classes("w-full gap-2"):
                        self.major_dd = ui.number(
                            label="Major Drawdown (%)",
                            value=self.default_config["major_drawdown_threshold"],
                            min=-50, max=0, step=1
                        ).classes("flex-1")

                        self.major_mult = ui.number(
                            label="Major Multiplier",
                            value=self.default_config["major_drawdown_inv_multiplier"],
                            min=1, max=10, step=0.5
                        ).classes("flex-1")

                    with ui.row().classes("w-full gap-2"):
                        self.minor_dd = ui.number(
                            label="Minor Drawdown (%)",
                            value=self.default_config["minor_drawdown_threshold"],
                            min=-20, max=0, step=1
                        ).classes("flex-1")

                        self.minor_mult = ui.number(
                            label="Minor Multiplier",
                            value=self.default_config["minor_drawdown_inv_multiplier"],
                            min=1, max=8, step=0.5
                        ).classes("flex-1")

                    with ui.row().classes("w-full gap-2"):
                        self.extreme_dd = ui.number(
                            label="Extreme Drawdown (%)",
                            value=self.default_config["extreme_drawdown_threshold"],
                            min=-50, max=0, step=1
                        ).classes("flex-1")

                        self.extreme_mult = ui.number(
                            label="Extreme Multiplier",
                            value=self.default_config["extreme_drawdown_inv_multiplier"],
                            min=1, max=10, step=0.5
                        ).classes("flex-1")

                    with ui.row().classes("w-full gap-2"):
                        self.fallback_day = ui.number(
                            label="📅 Investment Day",
                            value=self.default_config["fallback_day"],
                            min=1, max=28, step=1
                        ).classes("flex-1")

                        self.min_gap_days = ui.number(
                            label="⏰ Min Gap (days)",
                            value=self.default_config["min_investment_gap_days"],
                            min=1, max=30, step=1
                        ).classes("flex-1")

                    self.rolling_window = ui.number(
                        label="🔄 Rolling Window (days)",
                        value=self.default_config["rolling_window"],
                        min=20, max=365, step=5
                    ).classes("w-full")

                    self.price_reduction_threshold = ui.number(
                        label="📉 Price Reduction Threshold (%)",
                        value=self.default_config["price_reduction_threshold"],
                        min=1, max=10, step=0.5
                    ).classes("w-full")

                    self.max_monthly = ui.number(
                        label="📅 Max Monthly Amount (₹)",
                        value=self.default_config["fixed_investment"] * 4,
                        min=1000, step=1000
                    ).classes("w-full")

        # Results container with clear button
        self.results_container = ui.column().classes("w-full")

        ui.button("🗑️ Clear Results", on_click=self.clear_results).classes("bg-red-500 text-white mb-2")

        # Action buttons
        with ui.row().classes("gap-4 mt-4"):
            ui.button("🚀 Run Enhanced Backtest", on_click=self.run_enhanced_backtest).classes("bg-blue-600 text-white px-6 py-2")
            ui.button("📊 Load Template", on_click=self.load_backtest_template).classes("bg-gray-500 text-white px-4 py-2")
            ui.button("📈 Compare Strategies", on_click=self.run_strategy_comparison).classes("bg-purple-500 text-white px-4 py-2")

    async def run_enhanced_backtest(self):
        await self.clear_results()
        try:
            ui.notify("🚀 Starting enhanced backtest...", type="info")

            symbols_list = [s.strip() for s in self.symbols_input.value.split('\n') if s.strip()]

            if not symbols_list:
                ui.notify("❌ Please enter at least one symbol", type="negative")
                return

            config = {
                "fixed_investment": self.fixed_investment.value,
                "major_drawdown_threshold": self.major_dd.value,
                "minor_drawdown_threshold": self.minor_dd.value,
                "extreme_drawdown_threshold": self.extreme_dd.value,
                "minor_drawdown_inv_multiplier": self.minor_mult.value,
                "major_drawdown_inv_multiplier": self.major_mult.value,
                "extreme_drawdown_inv_multiplier": self.extreme_mult.value,
                "rolling_window": self.rolling_window.value,
                "fallback_day": self.fallback_day.value,
                "min_investment_gap_days": self.min_gap_days.value,
                "max_amount_in_a_month": self.max_monthly.value if self.enable_monthly_limits.value else None,
                "price_reduction_threshold": self.price_reduction_threshold.value,
                "force_remaining_investment": True,
                "enable_monthly_limits": self.enable_monthly_limits.value
            }

            request_data = {
                "symbols": symbols_list,
                "start_date": self.start_date.value,
                "end_date": self.end_date.value,
                "config": config
            }

            # Run backtest
            endpoint = "/sip/backtest"
            results = await self.safe_api_call(self.fetch_api, endpoint, method="POST", data=request_data)

            if results:
                await self.display_enhanced_backtest_results(results, self.results_container)
                ui.notify("✅ Backtest completed successfully!", type="positive")
            else:
                ui.notify("❌ Backtest failed - no results returned", type="negative")

        except Exception as e:
            logger.error(f"Backtest error: {e}")
            ui.notify(f"❌ Backtest failed: {str(e)}", type="negative")

    async def run_strategy_comparison(self, fetch_api, symbols_text: str, start_date: str, end_date: str,
                                      enable_monthly_limits: bool):
        try:
            symbols_list = [s.strip() for s in symbols_text.split('\n') if s.strip()]
            if not symbols_list:
                self.show_warning("No symbols provided for comparison")
                return

            if len(symbols_list) > 1:
                self.show_warning("Comparison limited to first symbol")
                symbol = symbols_list[0]
            else:
                symbol = symbols_list[0]

            comparison = await self.safe_api_call(
                fetch_api,
                f"/sip/strategies/compare/{symbol}?start_date={start_date}&end_date={end_date}&enable_monthly_limits={enable_monthly_limits}")
            if comparison:
                await self.show_strategy_comparison_dialog(comparison)
        except Exception as e:
            self.show_error("Strategy comparison failed", str(e))

    async def show_strategy_comparison_dialog(self, comparison: Dict):
        with ui.dialog() as dialog, ui.card().classes("w-[800px]"):
            ui.label("📊 Strategy Comparison").classes("text-xl font-bold mb-4")
            ui.label(f"Symbol: {comparison['symbol']} | Period: {comparison['analysis_period']}").classes("text-sm mb-4")

            # Benchmark
            with ui.card().classes("w-full p-4 mb-4 bg-blue-50"):
                ui.label("Benchmark (Fixed SIP)").classes("font-bold mb-2")
                bench = comparison["benchmark"]
                ui.label(f"Return: {bench['total_return_percent']:.2f}% | CAGR: {bench['cagr_percent']:.2f}%").classes("text-sm")
                ui.label(f"Invested: ₹{bench['total_investment']:,.0f} | Trades: {bench['num_trades']}").classes("text-sm")

            # Strategies grid
            with ui.grid(columns=3).classes("w-full gap-4"):
                for name, strat in comparison["strategies"].items():
                    with ui.card().classes("p-4"):
                        ui.label(name).classes("font-bold mb-2")
                        ui.label(f"Return: {strat['total_return_percent']:.2f}% | CAGR: {strat['cagr_percent']:.2f}%").classes("text-sm")
                        ui.label(f"Outperformance: {strat['vs_benchmark']['return_outperformance']:.2f}%").classes("text-sm text-green-600")
                        ui.label(f"Invested: ₹{strat['total_investment']:,.0f} | Trades: {strat['num_trades']}").classes("text-sm")

            ui.button("Close", on_click=dialog.close).classes("mt-4 bg-gray-500 text-white")

        dialog.open()

    async def render_multi_portfolio_panel(self, fetch_api, user_storage):
        """Multi-symbol portfolio creation interface with enhanced config"""

        ui.label("🎯 Multi-Symbol Portfolio Creation").classes("text-2xl font-bold mb-4")
        ui.label("Create portfolios with multiple symbols, automatic allocation, and monthly limits").classes("text-gray-600 mb-6")

        with ui.card().classes("w-full mb-6 p-6"):
            ui.label("Portfolio Configuration").classes("text-lg font-semibold mb-4")

            # Portfolio basic info
            portfolio_name_input = ui.input(
                label="📁 Portfolio Name",
                placeholder="My Balanced SIP Portfolio",
                value=""
            ).classes("w-full mb-4")

            # Symbols configuration
            symbols_container = ui.column().classes("w-full")
            symbols_data = []

            def add_symbol_row():
                with symbols_container:
                    with ui.card().classes("w-full p-4 mb-2 border-l-4 border-blue-500"):
                        with ui.row().classes("w-full items-center gap-4"):
                            symbol_input = ui.input(
                                label="Symbol",
                                placeholder="ICICIB22"
                            ).classes("flex-1")

                            allocation_input = ui.number(
                                label="Allocation %",
                                value=25.0,
                                min=0.1, max=100, step=0.1
                            ).classes("w-32")

                            def remove_symbol():
                                parent_card = symbol_input.parent_slot.parent
                                symbols_container.remove(parent_card)
                                symbols_data[:] = [s for s in symbols_data if s != (symbol_input, allocation_input)]

                            ui.button("🗑️", on_click=remove_symbol).classes("bg-red-500 text-white")

                        symbols_data.append((symbol_input, allocation_input))

            # Initial symbols
            with ui.row().classes("w-full justify-between items-center mb-4"):
                ui.label("Symbols & Allocation").classes("font-medium")
                ui.button("➕ Add Symbol", on_click=add_symbol_row).classes("bg-green-500 text-white")

            # Add default symbols
            for _ in range(3):
                add_symbol_row()

            # Default configuration with new fields
            with ui.row().classes("w-full gap-4 mt-6"):
                with ui.column().classes("flex-1"):
                    ui.label("Default Strategy Config").classes("font-medium mb-2")

                    default_investment = ui.number(
                        label="Monthly Investment (₹)",
                        value=10000,
                        min=1000, step=1000
                    ).classes("w-full")

                    default_major_dd = ui.number(
                        label="Major Drawdown (%)",
                        value=-10.0,
                        min=-50, max=0
                    ).classes("w-full")

                    default_minor_dd = ui.number(
                        label="Minor Drawdown (%)",
                        value=-4.0,
                        min=-20, max=0
                    ).classes("w-full")

                    default_extreme_dd = ui.number(
                        label="Extreme Drawdown (%)",
                        value=-15.0,
                        min=-50, max=0
                    ).classes("w-full")

                with ui.column().classes("flex-1"):
                    ui.label("Portfolio Settings").classes("font-medium mb-2")

                    auto_rebalance = ui.checkbox(
                        "Auto Rebalance",
                        value=False
                    )

                    rebalance_frequency = ui.number(
                        label="Rebalance Frequency (days)",
                        value=30,
                        min=7, max=365
                    ).classes("w-full")

                    max_monthly = ui.number(
                        label="Max Monthly Amount (₹)",
                        value=40000,
                        min=1000, step=1000
                    ).classes("w-full")

                    force_invest = ui.checkbox(
                        "Force Remaining Investment",
                        value=True
                    )

        async def create_multi_portfolio():
            try:
                # Validate inputs
                if not portfolio_name_input.value.strip():
                    ui.notify("❌ Please enter portfolio name", type="negative")
                    return

                # Collect symbols data
                portfolio_symbols = []
                total_allocation = 0

                for symbol_input, allocation_input in symbols_data:
                    if symbol_input.value.strip() and allocation_input.value > 0:
                        portfolio_symbols.append({
                            "symbol": symbol_input.value.strip().upper(),
                            "allocation_percentage": allocation_input.value
                        })
                        total_allocation += allocation_input.value

                if not portfolio_symbols:
                    ui.notify("❌ Please add at least one symbol", type="negative")
                    return

                if abs(total_allocation - 100.0) > 0.01:
                    ui.notify(f"❌ Total allocation must equal 100% (current: {total_allocation}%)", type="negative")
                    return

                # Prepare request with enhanced config
                request_data = {
                    "portfolio_name": portfolio_name_input.value.strip(),
                    "symbols": portfolio_symbols,
                    "default_config": {
                        "fixed_investment": default_investment.value,
                        "major_drawdown_threshold": default_major_dd.value,
                        "minor_drawdown_threshold": default_minor_dd.value,
                        "extreme_drawdown_threshold": default_extreme_dd.value,
                        "minor_drawdown_inv_multiplier": 1.75,
                        "major_drawdown_inv_multiplier": 3.0,
                        "extreme_drawdown_inv_multiplier": 4.0,
                        "rolling_window": 100,
                        "fallback_day": 22,
                        "min_investment_gap_days": 5,
                        "max_amount_in_a_month": max_monthly.value,
                        "price_reduction_threshold": 4.0,
                        "force_remaining_investment": force_invest.value
                    },
                    "auto_rebalance": auto_rebalance.value,
                    "rebalance_frequency_days": rebalance_frequency.value
                }

                response = await self.safe_api_call(
                    fetch_api, "/sip/portfolio/multi", method="POST", data=request_data)

                if response:
                    ui.notify(f"✅ Multi-portfolio created: {response['portfolio_id']}", type="positive")
                    # Clear form
                    portfolio_name_input.value = ""
                    for symbol_input, allocation_input in symbols_data:
                        symbol_input.value = ""
                        allocation_input.value = 25.0
                else:
                    ui.notify("❌ Failed to create multi-portfolio", type="negative")

            except Exception as e:
                ui.notify(f"❌ Error creating portfolio: {str(e)}", type="negative")

        ui.button("🎯 Create Multi-Portfolio", on_click=create_multi_portfolio).classes(
            "bg-purple-600 text-white px-6 py-2 mt-4")

    async def render_enhanced_portfolio_panel(self, fetch_api, user_storage):
        """Enhanced portfolio management interface with GTT status"""

        ui.label("💼 SIP Portfolio Management").classes("text-2xl font-bold mb-4")

        # Single symbol portfolio creation with enhanced config
        with ui.card().classes("w-full mb-6 p-6"):
            ui.label("Create Single-Symbol Portfolio").classes("text-lg font-semibold mb-4")

            with ui.row().classes("w-full gap-4"):
                symbol_input = ui.input(
                    label="📈 Symbol",
                    placeholder="ICICIB22"
                ).classes("flex-1")

                portfolio_name_input = ui.input(
                    label="📁 Portfolio Name (optional)",
                    placeholder="My ICICI Bank SIP"
                ).classes("flex-1")

            with ui.row().classes("w-full gap-4 mt-4"):
                investment_amount = ui.number(
                    label="💰 Monthly Investment (₹)",
                    value=5000,
                    min=500, step=500
                ).classes("flex-1")

                fallback_day = ui.number(
                    label="📅 Investment Day",
                    value=22,
                    min=1, max=28
                ).classes("flex-1")

                min_gap = ui.number(
                    label="⏰ Min Gap (days)",
                    value=5,
                    min=1, max=30
                ).classes("flex-1")

            max_monthly = ui.number(
                label="📅 Max Monthly Amount (₹)",
                value=20000,
                min=1000, step=1000
            ).classes("w-full mt-2")

            async def create_single_portfolio():
                try:
                    if not symbol_input.value.strip():
                        ui.notify("❌ Please enter a symbol", type="negative")
                        return

                    config = {
                        "fixed_investment": investment_amount.value,
                        "major_drawdown_threshold": -10.0,
                        "minor_drawdown_threshold": -4.0,
                        "extreme_drawdown_threshold": -15.0,
                        "minor_drawdown_inv_multiplier": 1.75,
                        "major_drawdown_inv_multiplier": 3.0,
                        "extreme_drawdown_inv_multiplier": 4.0,
                        "rolling_window": 100,
                        "fallback_day": fallback_day.value,
                        "min_investment_gap_days": min_gap.value,
                        "max_amount_in_a_month": max_monthly.value,
                        "price_reduction_threshold": 4.0,
                        "force_remaining_investment": True
                    }

                    request_data = {
                        "symbol": symbol_input.value.strip().upper(),
                        "portfolio_name": portfolio_name_input.value.strip() or None,
                        "config": config
                    }

                    response = await self.safe_api_call(fetch_api, "/sip/portfolio", method="POST", data=request_data)

                    if response:
                        ui.notify(f"✅ Portfolio created: {response['portfolio_id']}", type="positive")
                        await refresh_portfolios()
                        symbol_input.value = ""
                        portfolio_name_input.value = ""
                    else:
                        ui.notify("❌ Failed to create portfolio", type="negative")

                except Exception as e:
                    ui.notify(f"❌ Error creating portfolio: {str(e)}", type="negative")

            ui.button("✨ Create Portfolio", on_click=create_single_portfolio).classes("bg-blue-500 text-white mt-4")

        # Existing portfolios
        portfolios_container = ui.column().classes("w-full")

        async def refresh_portfolios():
            """Refresh portfolio list with enhanced display including GTT status"""
            try:
                portfolios = await self.safe_api_call(fetch_api, "/sip/portfolio")

                portfolios_container.clear()

                if portfolios:
                    self.active_portfolios = portfolios

                    with portfolios_container:
                        ui.label("📊 Your SIP Portfolios").classes("text-lg font-semibold mb-4")

                        for portfolio in portfolios:
                            await self.render_enhanced_portfolio_card(portfolio, fetch_api, refresh_portfolios)
                else:
                    with portfolios_container:
                        ui.label("📝 No portfolios found. Create your first SIP portfolio above!").classes(
                            "text-gray-500 text-center py-8")

            except Exception as e:
                logger.error(f"Error fetching portfolios: {e}")
                with portfolios_container:
                    ui.label("❌ Error loading portfolios").classes("text-red-500")

        # Load portfolios on page load
        await refresh_portfolios()

        # Refresh button
        ui.button("🔄 Refresh Portfolios", on_click=refresh_portfolios).classes("bg-gray-500 text-white mt-4")

    async def render_enhanced_portfolio_card(self, portfolio, fetch_api, refresh_callback):
        """Render enhanced portfolio card with GTT status and monthly tracking"""

        portfolio_type = portfolio.get('portfolio_type', 'single')
        symbols = portfolio.get('symbols', [])
        status = portfolio.get('status', 'active')

        # Determine card styling based on status
        card_classes = "w-full mb-4 p-4"
        if status == 'active':
            card_classes += " border-l-4 border-green-500"
        elif status == 'cancelled':
            card_classes += " border-l-4 border-yellow-500 opacity-75"
        else:
            card_classes += " border-l-4 border-gray-400 opacity-50"

        with ui.card().classes(card_classes):
            # Header with portfolio info
            with ui.row().classes("w-full justify-between items-start"):
                with ui.column().classes("flex-1"):
                    # Portfolio name and type
                    portfolio_name = portfolio.get('portfolio_name', 'Unnamed Portfolio')
                    type_icon = "🎯" if portfolio_type == 'multi' else "📈"
                    ui.label(f"{type_icon} {portfolio_name}").classes("text-lg font-bold")

                    # Status and symbols info
                    status_color = "text-green-600" if status == 'active' else "text-yellow-600" if status == 'cancelled' else "text-gray-500"
                    ui.label(f"Status: {status.title()}").classes(f"text-sm {status_color}")

                    if portfolio_type == 'multi' and symbols:
                        symbols_text = ", ".join(
                            [f"{s['symbol']} ({s['allocation_percentage']}%)" for s in symbols[:3]])
                        if len(symbols) > 3:
                            symbols_text += f" +{len(symbols) - 3} more"
                        ui.label(f"Symbols: {symbols_text}").classes("text-sm text-gray-600")
                    elif symbols:
                        ui.label(f"Symbol: {symbols[0]['symbol']}").classes("text-sm text-gray-600")

                # Performance metrics
                with ui.column().classes("text-right"):
                    total_invested = portfolio.get('total_invested', 0)
                    current_value = portfolio.get('current_value', 0)
                    current_units = portfolio.get('current_units', 0)

                    ui.label(f"💰 Invested: ₹{total_invested:,.0f}").classes("text-sm")
                    ui.label(f"📊 Value: ₹{current_value:,.0f}").classes("text-sm")
                    ui.label(f"📦 Units: {current_units:.2f}").classes("text-sm")

                    # Calculate and display return
                    if total_invested > 0:
                        return_pct = ((current_value - total_invested) / total_invested) * 100
                        return_color = "text-green-600" if return_pct >= 0 else "text-red-600"
                        ui.label(f"📈 Return: {return_pct:+.2f}%").classes(f"text-sm {return_color} font-medium")

            # Next investment and monthly status
            next_investment = portfolio.get('next_investment_date')
            if next_investment:
                ui.label(f"📅 Next Investment: {next_investment}").classes("text-sm text-blue-600 mt-2")

            monthly_invested = portfolio.get('monthly_invested_so_far', 0)
            monthly_limit = portfolio.get('monthly_limit', 0)
            if monthly_limit > 0:
                ui.label(f"📅 Monthly Invested: ₹{monthly_invested:,.0f} / ₹{monthly_limit:,.0f}").classes("text-sm mt-1")

            # Action buttons with GTT management
            with ui.row().classes("gap-2 mt-4"):
                async def view_performance():
                    await self.show_enhanced_portfolio_performance(portfolio['portfolio_id'], fetch_api)

                ui.button("📊 Performance", on_click=view_performance).classes(
                    "bg-blue-500 text-white text-xs px-3 py-1")

                async def get_signals():
                    await self.show_enhanced_portfolio_signals(portfolio['portfolio_id'], fetch_api)

                ui.button("📡 Signals & GTT", on_click=get_signals).classes("bg-purple-500 text-white text-xs px-3 py-1")

                if status == 'active':
                    async def invest_now():
                        await self.show_enhanced_investment_dialog(portfolio['portfolio_id'], fetch_api,
                                                                   refresh_callback)

                    ui.button("💰 Invest", on_click=invest_now).classes("bg-green-500 text-white text-xs px-3 py-1")

                if status == 'active':
                    async def cancel_portfolio():
                        await self.cancel_portfolio(portfolio['portfolio_id'], fetch_api, refresh_callback)

                    ui.button("⏸️ Cancel", on_click=cancel_portfolio).classes(
                        "bg-yellow-500 text-white text-xs px-3 py-1")

                async def delete_portfolio():
                    await self.delete_portfolio(portfolio['portfolio_id'], fetch_api, refresh_callback)

                ui.button("🗑️ Delete", on_click=delete_portfolio).classes("bg-red-500 text-white text-xs px-3 py-1")

    async def cancel_portfolio(self, portfolio_id, fetch_api, refresh_callback):
        """Cancel SIP portfolio (mark as cancelled)"""
        try:
            with ui.dialog() as dialog, ui.card():
                ui.label("⏸️ Cancel SIP Portfolio").classes("text-lg font-bold mb-4")
                ui.label("This will stop future automated investments but keep the portfolio data.").classes(
                    "text-gray-600 mb-4")
                ui.label("You can still manually invest or reactivate later.").classes("text-blue-600 mb-4")

                with ui.row().classes("gap-2"):
                    ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white")

                    async def confirm_cancel():
                        response = await self.safe_api_call(fetch_api, f"/sip/portfolio/{portfolio_id}/cancel", method="PUT")
                        if response:
                            ui.notify("✅ Portfolio cancelled successfully", type="positive")
                            await refresh_callback()
                        else:
                            ui.notify("❌ Failed to cancel portfolio", type="negative")
                        dialog.close()

                    ui.button("⏸️ Confirm Cancel", on_click=confirm_cancel).classes("bg-yellow-500 text-white")

            dialog.open()

        except Exception as e:
            ui.notify(f"❌ Error cancelling portfolio: {str(e)}", type="negative")

    async def delete_portfolio(self, portfolio_id, fetch_api, refresh_callback):
        """Permanently delete SIP portfolio"""
        try:
            with ui.dialog() as dialog, ui.card():
                ui.label("🗑️ Delete SIP Portfolio").classes("text-lg font-bold mb-4")
                ui.label("⚠️ This will permanently delete the portfolio and ALL related data.").classes(
                    "text-red-600 mb-4")
                ui.label("This action cannot be undone!").classes("text-red-600 font-bold mb-4")

                with ui.row().classes("gap-2"):
                    ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white")

                    async def confirm_delete():
                        response = await self.safe_api_call(fetch_api, f"/sip/portfolio/{portfolio_id}", method="DELETE")
                        if response:
                            ui.notify("✅ Portfolio deleted permanently", type="positive")
                            await refresh_callback()
                        else:
                            ui.notify("❌ Failed to delete portfolio", type="negative")
                        dialog.close()

                    ui.button("🗑️ Confirm Delete", on_click=confirm_delete).classes("bg-red-600 text-white")

            dialog.open()

        except Exception as e:
            ui.notify(f"❌ Error deleting portfolio: {str(e)}", type="negative")

    async def show_enhanced_portfolio_signals(self, portfolio_id, fetch_api):
        """Show enhanced portfolio signals with GTT status and monthly tracking"""
        try:
            signals = await self.safe_api_call(fetch_api, f"/sip/signals/{portfolio_id}")

            with ui.dialog() as dialog, ui.card().classes("w-[600px]"):
                ui.label("📡 Investment Signals & GTT Status").classes("text-xl font-bold mb-4")

                if signals and signals.get('signal') != 'NO_DATA':
                    # Signal strength indicator
                    signal_type = signals.get('signal', 'NORMAL')
                    confidence = signals.get('confidence', 0)

                    # Color coding for signal strength
                    if signal_type == 'STRONG_BUY':
                        signal_color = "bg-green-600"
                        signal_text = "🟢 STRONG BUY"
                    elif signal_type == 'BUY':
                        signal_color = "bg-green-500"
                        signal_text = "🟡 BUY"
                    elif signal_type == 'WEAK_BUY':
                        signal_color = "bg-yellow-500"
                        signal_text = "🟡 WEAK BUY"
                    elif signal_type == 'AVOID':
                        signal_color = "bg-red-500"
                        signal_text = "🔴 AVOID"
                    else:
                        signal_color = "bg-blue-500"
                        signal_text = "🔵 NORMAL"

                    with ui.card().classes(f"w-full p-4 {signal_color} text-white mb-4"):
                        ui.label(signal_text).classes("text-lg font-bold")
                        ui.label(f"Confidence: {confidence * 100:.1f}%").classes("text-sm")
                        ui.label(signals.get('message', '')).classes("text-sm mt-2")

                    # GTT status
                    gtt_status = signals.get('gtt_status', 'NONE')
                    gtt_color = "bg-green-100 text-green-800" if gtt_status == 'ACTIVE' else "bg-red-100 text-red-800" if gtt_status == 'NONE' else "bg-yellow-100 text-yellow-800"
                    ui.label(f"GTT Status: {gtt_status}").classes(f"text-sm px-3 py-1 rounded {gtt_color} mb-4")

                    # Detailed metrics
                    with ui.grid(columns=2).classes("w-full gap-4 mb-4"):
                        # Left column - Investment details
                        with ui.card().classes("p-4"):
                            ui.label("💰 Investment Details").classes("font-bold mb-2")
                            ui.label(f"Recommended Amount: ₹{signals.get('recommended_amount', 0):,.0f}").classes(
                                "text-sm")
                            ui.label(f"Multiplier: {signals.get('investment_multiplier', 1)}x").classes("text-sm")
                            ui.label(f"Current Price: ₹{signals.get('current_price', 0):.2f}").classes("text-sm")
                            ui.label(f"Trade Type: {signals.get('trade_type', 'Regular')}").classes("text-sm")
                            ui.label(f"Monthly Invested: ₹{signals.get('monthly_invested_so_far', 0):,.0f} / ₹{signals.get('monthly_limit', 0):,.0f}").classes("text-sm")

                        # Right column - Technical indicators
                        with ui.card().classes("p-4"):
                            ui.label("📊 Technical Indicators").classes("font-bold mb-2")

                            drawdown = signals.get('drawdown_100', 0)
                            drawdown_color = "text-red-500" if drawdown < -5 else "text-yellow-500" if drawdown < 0 else "text-green-500"
                            ui.label(f"Drawdown (100d): {drawdown:.2f}%").classes(f"text-sm {drawdown_color}")

                            rsi = signals.get('rsi', 50)
                            rsi_color = "text-green-500" if rsi < 30 else "text-red-500" if rsi > 70 else "text-blue-500"
                            ui.label(f"RSI: {rsi:.1f}").classes(f"text-sm {rsi_color}")

                            volatility = signals.get('volatility', 0)
                            ui.label(f"Volatility: {volatility:.3f}").classes("text-sm")

                            bb_pos = signals.get('bb_position', 0.5)
                            ui.label(f"BB Position: {bb_pos:.2f}").classes("text-sm")

                    # Market conditions
                    market_conditions = signals.get('market_conditions', [])
                    if market_conditions:
                        ui.label("🔍 Market Analysis").classes("font-bold mb-2")
                        with ui.card().classes("p-4 bg-gray-50"):
                            for condition in market_conditions:
                                ui.label(f"• {condition}").classes("text-sm")

                    # Next dates
                    next_fallback = signals.get('next_fallback_date')
                    if next_fallback:
                        ui.label(f"📅 Next Fallback Date: {next_fallback}").classes("text-sm text-blue-600 mt-4")

                else:
                    ui.label("📊 No immediate signals detected").classes("text-gray-600")
                    ui.label("Market conditions appear normal for regular SIP investment").classes(
                        "text-sm text-gray-500")

                ui.button("Close", on_click=dialog.close).classes("mt-4 bg-gray-500 text-white")

            dialog.open()

        except Exception as e:
            ui.notify(f"❌ Error loading signals: {str(e)}", type="negative")

    async def show_enhanced_investment_dialog(self, portfolio_id, fetch_api, refresh_callback):
        """Show enhanced manual investment dialog with gap checking and monthly limits"""
        try:
            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label("💰 Manual SIP Investment").classes("text-xl font-bold mb-4")

                amount_input = ui.number(
                    label="Investment Amount (₹)",
                    value=5000,
                    min=100,
                    step=100
                ).classes("w-full")

                ui.label("⚠️ Note: System will check minimum 5-day gap and monthly limits").classes(
                    "text-xs text-yellow-600 mt-2")
                ui.label("💡 This will execute immediately regardless of market conditions").classes(
                    "text-xs text-blue-600")

                async def execute_investment():
                    try:
                        response = await self.safe_api_call(
                            fetch_api,
                            f"/sip/execute/{portfolio_id}",
                            method="POST",
                            data={"amount": amount_input.value}
                        )

                        if response:
                            ui.notify(
                                f"✅ Investment successful! ₹{response.get('total_investment_amount', 0):,.0f} invested",
                                type="positive")
                            await refresh_callback()
                            dialog.close()
                        else:
                            ui.notify("❌ Investment failed", type="negative")

                    except Exception as e:
                        error_msg = str(e)
                        if "Minimum" in error_msg and "days gap" in error_msg:
                            ui.notify(f"⏰ {error_msg}", type="warning")
                        elif "monthly limit" in error_msg.lower():
                            ui.notify(f"📅 {error_msg}", type="warning")
                        else:
                            ui.notify(f"❌ Investment failed: {error_msg}", type="negative")

                with ui.row().classes("gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white")
                    ui.button("💰 Invest Now", on_click=execute_investment).classes("bg-green-500 text-white")

            dialog.open()

        except Exception as e:
            ui.notify(f"❌ Error showing investment dialog: {str(e)}", type="negative")

    async def render_enhanced_signals_panel(self, fetch_api, user_storage):
        """Enhanced signals panel with real-time updates and GTT status"""

        ui.label("📡 Investment Signals Dashboard").classes("text-2xl font-bold mb-4")
        ui.label("Monitor investment opportunities across all your portfolios with GTT status").classes("text-gray-600 mb-6")

        signals_container = ui.column().classes("w-full")

        async def refresh_signals():
            """Refresh all signals with enhanced display including GTT"""
            try:
                signals = await self.safe_api_call(fetch_api, "/sip/signals")

                signals_container.clear()

                if signals:
                    with signals_container:
                        ui.label(f"📊 Found {len(signals)} Active Signals").classes("text-lg font-semibold mb-4")

                        for signal in signals:
                            await self.render_signal_card(signal)
                else:
                    with signals_container:
                        with ui.card().classes("w-full p-8 text-center"):
                            ui.label("🔍 No active signals found").classes("text-lg text-gray-500")
                            ui.label("Check back later or trigger manual scan").classes("text-sm text-gray-400")

            except Exception as e:
                self.show_error("Failed to refresh signals", str(e))

        ui.button("🔄 Refresh Signals", on_click=refresh_signals).classes("bg-blue-500 text-white mb-4")

        await refresh_signals()

    async def render_signal_card(self, signal: Dict):
        """Render individual signal card with GTT status"""
        symbol = signal.get('symbol', 'Unknown')
        signal_type = signal.get('signal_type', 'NORMAL')
        recommended_amount = signal.get('recommended_amount', 0)
        current_price = signal.get('current_price', 0)
        drawdown = signal.get('drawdown_percent', 0)
        gtt_status = signal.get('gtt_status', 'NONE')

        card_color = {
            'STRONG_BUY': 'border-green-500',
            'BUY': 'border-blue-500',
            'WEAK_BUY': 'border-yellow-500',
            'AVOID': 'border-red-500',
            'NORMAL': 'border-gray-500'
        }.get(signal_type, 'border-gray-500')

        with ui.card().classes(f"w-full mb-4 p-4 border-l-4 {card_color}"):
            with ui.row().classes("w-full justify-between items-center"):
                ui.label(f"📡 {symbol} Signal").classes("text-lg font-bold")
                ui.label(signal_type).classes("text-sm font-medium px-2 py-1 rounded bg-gray-100")

            ui.label(f"💰 Recommended: ₹{recommended_amount:,.0f}").classes("text-sm")
            ui.label(f"📈 Price: ₹{current_price:.2f}").classes("text-sm")
            ui.label(f"🔽 Drawdown: {drawdown:.2f}%").classes("text-sm text-red-600")

            # GTT status badge
            gtt_color = "bg-green-200 text-green-800" if gtt_status == 'ACTIVE' else "bg-red-200 text-red-800"
            ui.label(gtt_status).classes(f"text-xs px-2 py-1 rounded {gtt_color} mt-2")

            ui.label(f"Created: {signal.get('created_at', '')}").classes("text-xs text-gray-500 mt-2")

    async def render_batch_multi_configs_section(self, fetch_api, user_storage):
        """Render batch multi-configs backtest UI with improved input"""

        ui.label("🔄 Batch Backtest with Multiple Configurations").classes("text-lg font-bold mb-4")
        ui.label("Test multiple strategy configurations in batch and find the best one").classes("text-gray-600 mb-4")

        symbols_input = ui.textarea(
            label="Symbols (one per line)",
            value="ICICIB22\nGOLDBEES"
        ).classes("w-full mb-4")

        # Configs table
        configs_table = ui.table(columns=[
            {'name': 'fixed_investment', 'label': 'Monthly Inv (₹)', 'field': 'fixed_investment', 'editable': True},
            {'name': 'major_dd', 'label': 'Major DD (%)', 'field': 'major_drawdown_threshold', 'editable': True},
            {'name': 'minor_dd', 'label': 'Minor DD (%)', 'field': 'minor_drawdown_threshold', 'editable': True},
            {'name': 'extreme_dd', 'label': 'Extreme DD (%)', 'field': 'extreme_drawdown_threshold', 'editable': True},
            {'name': 'minor_mult', 'label': 'Minor Mult', 'field': 'minor_drawdown_inv_multiplier', 'editable': True},
            {'name': 'major_mult', 'label': 'Major Mult', 'field': 'major_drawdown_inv_multiplier', 'editable': True},
            {'name': 'extreme_mult', 'label': 'Extreme Mult', 'field': 'extreme_drawdown_inv_multiplier', 'editable': True},
            {'name': 'actions', 'label': 'Actions'}
        ], rows=[], row_key='id').classes("w-full")

        configs_table.add_rows([{'id': 1, 'fixed_investment': 5000, 'major_drawdown_threshold': -10.0, 'minor_drawdown_threshold': -4.0, 'extreme_drawdown_threshold': -15.0, 'minor_drawdown_inv_multiplier': 1.75, 'major_drawdown_inv_multiplier': 3.0, 'extreme_drawdown_inv_multiplier': 4.0}])

        def add_config_row():
            new_id = max([r['id'] for r in configs_table.rows] or [0]) + 1
            new_row = {'id': new_id, 'fixed_investment': 5000, 'major_drawdown_threshold': -10.0, 'minor_drawdown_threshold': -4.0, 'extreme_drawdown_threshold': -15.0, 'minor_drawdown_inv_multiplier': 1.75, 'major_drawdown_inv_multiplier': 3.0, 'extreme_drawdown_inv_multiplier': 4.0}
            configs_table.add_rows([new_row])

        ui.button("➕ Add Config", on_click=add_config_row).classes("bg-green-500 text-white mb-4")

        with ui.row().classes("gap-4"):
            start_date = ui.date(value="2020-01-01")
            end_date = ui.date(value=datetime.now().strftime("%Y-%m-%d"))

        batch_results = ui.column().classes("w-full mt-4")

        async def run_batch_multi():
            try:
                symbols = [s.strip() for s in symbols_input.value.split('\n') if s.strip()]
                if not symbols:
                    self.show_warning("No symbols")
                    return

                configs = [ {k: r[k] for k in r if k != 'id'} for r in configs_table.rows ]

                if not configs:
                    self.show_warning("No configs added")
                    return

                request_data = {
                    "symbols": symbols,
                    "start_date": start_date.value,
                    "end_date": end_date.value,
                    "configs": configs
                }

                results = await self.safe_api_call(fetch_api, "/sip/batch-backtest/multi-configs", "POST", request_data)

                if results:
                    batch_results.clear()
                    with batch_results:
                        ui.label("Batch Results").classes("font-bold mb-2")
                        for idx, res in enumerate(results):
                            with ui.expansion(f"Config {idx+1}").classes("w-full mb-2"):
                                await self.display_enhanced_backtest_results(res, batch_results)
                        if 'best_config' in results:
                            with ui.card().classes("mt-4 p-4 bg-green-50"):
                                ui.label("Best Configuration").classes("font-bold")
                                for k, v in results['best_config'].items():
                                    ui.label(f"{k}: {v}").classes("text-sm")
            except Exception as e:
                self.show_error("Batch run failed", str(e))

        ui.button("🚀 Run Batch", on_click=run_batch_multi).classes("bg-orange-500 text-white")

    async def render_optimize_config_section(self, fetch_api, user_storage):
        """Render config optimization section"""

        ui.label("⚙️ Optimize Configuration").classes("text-lg font-bold mb-4")
        ui.label("Find the optimal strategy parameters based on historical data").classes("text-gray-600 mb-4")

        symbols_input = ui.textarea(
            label="Symbols (one per line)",
            value="ICICIB22\nGOLDBEES"
        ).classes("w-full mb-4")

        with ui.row().classes("gap-4"):
            start_date = ui.date(value="2020-01-01")
            end_date = ui.date(value=datetime.now().strftime("%Y-%m-%d"))

        optimize_results = ui.column().classes("w-full mt-4")

        async def run_optimize():
            try:
                symbols = [s.strip() for s in symbols_input.value.split('\n') if s.strip()]
                if not symbols:
                    self.show_warning("No symbols")
                    return

                request_data = {
                    "symbols": symbols,
                    "start_date": start_date.value,
                    "end_date": end_date.value
                }

                optimized = await self.safe_api_call(fetch_api, "/sip/optimize-config", "POST", request_data)

                if optimized:
                    optimize_results.clear()
                    with optimize_results:
                        ui.label("Optimized Configuration").classes("font-bold mb-2")
                        config = optimized.get("optimized_config", {})
                        table = ui.table(columns=[{'name': 'param', 'label': 'Parameter', 'field': 'param'},
                                                 {'name': 'value', 'label': 'Value', 'field': 'value'}],
                                         rows=[{'param': k, 'value': v} for k, v in config.items()]).classes("w-full")
                        ui.label(f"Expected CAGR: {optimized.get('expected_cagr', 0):.2f}%").classes("mt-2")
            except Exception as e:
                self.show_error("Optimization failed", str(e))

        ui.button("⚙️ Optimize", on_click=run_optimize).classes("bg-purple-500 text-white")

    async def render_quick_test_section(self, fetch_api, user_storage):
        """Render quick test section"""

        ui.label("⚡ Quick SIP Test").classes("text-lg font-bold mb-4")
        ui.label("Rapid analysis with benchmark comparison").classes("text-gray-600 mb-4")

        symbols_input = ui.textarea(
            label="Symbols (one per line)",
            value="ICICIB22\nGOLDBEES"
        ).classes("w-full mb-4")

        with ui.row().classes("gap-4"):
            start_date = ui.date(value="2023-01-01")
            end_date = ui.date(value=datetime.now().strftime("%Y-%m-%d"))
            investment_amount = ui.number(label="Investment Amount (₹)", value=5000, min=1000)

        quick_results = ui.column().classes("w-full mt-4")

        async def run_quick():
            try:
                symbols = [s.strip() for s in symbols_input.value.split('\n') if s.strip()]
                if not symbols:
                    self.show_warning("No symbols")
                    return

                request_data = {
                    "symbols": symbols,
                    "start_date": start_date.value,
                    "end_date": end_date.value,
                    "investment_amount": investment_amount.value
                }

                results = await self.safe_api_call(fetch_api, "/sip/quick-test", "POST", request_data)

                if results:
                    quick_results.clear()
                    with quick_results:
                        ui.label("Quick Test Results").classes("font-bold mb-2")
                        for res in results.get("results", []):
                            with ui.card().classes("mb-4 p-4"):
                                ui.label(res['symbol']).classes("font-bold")
                                strat = res.get('strategy', {})
                                bench = res.get('benchmark', {})
                                ui.label(f"Strategy Return: {strat.get('return_percent', 0):.2f}%").classes("text-sm")
                                ui.label(f"Benchmark Return: {bench.get('return_percent', 0):.2f}%").classes("text-sm")
                                ui.label(f"Outperformance: {res.get('comparison', {}).get('outperformance', 0):.2f}%").classes("text-sm")
            except Exception as e:
                self.show_error("Quick test failed", str(e))

        ui.button("⚡ Run Quick Test", on_click=run_quick).classes("bg-green-500 text-white")

    async def render_benchmark_test_section(self, fetch_api, user_storage):
        """Render benchmark test section with full metrics"""

        ui.label("📊 Benchmark Test").classes("text-lg font-bold mb-4")
        ui.label("Test fixed SIP benchmark performance").classes("text-gray-600 mb-4")

        symbol_input = ui.input(label="Symbol", placeholder="ICICIB22").classes("w-full mb-4")

        with ui.row().classes("gap-4"):
            start_date = ui.date(value="2023-01-01")
            end_date = ui.date(value=datetime.now().strftime("%Y-%m-%d"))
            monthly_amount = ui.number(label="Monthly Amount (₹)", value=5000, min=1000)
            investment_day = ui.number(label="Investment Day", value=15, min=1, max=28)

        bench_results = ui.column().classes("w-full mt-4")

        async def run_bench_test():
            try:
                symbol = symbol_input.value.strip()
                if not symbol:
                    self.show_warning("No symbol")
                    return

                query = f"?start_date={start_date.value}&end_date={end_date.value}&monthly_amount={monthly_amount.value}&investment_day={investment_day.value}"

                results = await self.safe_api_call(fetch_api, f"/sip/benchmark/test/{symbol}{query}")

                if results:
                    bench_results.clear()
                    with bench_results:
                        ui.label("Benchmark Results").classes("font-bold mb-2")
                        res = results.get("benchmark_result", {})
                        with ui.grid(columns=2).classes("gap-4"):
                            ui.label(f"Total Investment: ₹{res.get('total_investment', 0):,.2f}").classes("text-sm")
                            ui.label(f"Final Value: ₹{res.get('final_portfolio_value', 0):,.2f}").classes("text-sm")
                            ui.label(f"Total Return: {res.get('total_return_percent', 0):.2f}%").classes("text-sm")
                            ui.label(f"CAGR: {res.get('cagr_percent', 0):.2f}%").classes("text-sm")
                            ui.label(f"Number of Trades: {res.get('num_trades', 0)}").classes("text-sm")
                            ui.label(f"Max Drawdown: {res.get('max_drawdown_percent', 0):.2f}%").classes("text-sm")
                            ui.label(f"Sharpe Ratio: {res.get('sharpe_ratio', 0):.2f}").classes("text-sm")
                            ui.label(f"Volatility: {res.get('volatility', 0):.2f}").classes("text-sm")
            except Exception as e:
                self.show_error("Benchmark test failed", str(e))

        ui.button("📊 Run Benchmark Test", on_click=run_bench_test).classes("bg-indigo-500 text-white mt-4")

    async def render_symbols_search_section(self, fetch_api, user_storage):
        """Render symbols search section"""

        ui.label("🔍 Search Available Symbols").classes("text-lg font-bold mb-4")
        ui.label("Search and view available symbols for backtesting").classes("text-gray-600 mb-4")

        symbol_search_input = ui.input(
            label="Search Symbols",
            placeholder="Enter symbol or keyword"
        ).classes("w-full mb-4")

        limit = ui.number(
            label="Results Limit",
            value=50,
            min=10, max=200
        ).classes("w-full mb-4")

        symbols_results = ui.column().classes("w-full")

        async def search_symbols():
            try:
                query_params = f"?limit={limit.value}"
                symbols_data = await self.safe_api_call(
                    fetch_api,
                    f"/sip/symbols{query_params}"
                )

                if symbols_data:
                    symbols_results.clear()
                    with symbols_results:
                        ui.label(f"Found {len(symbols_data['symbols'])} symbols").classes("mb-2")
                        table = ui.table(columns=[{'name': 'symbol', 'label': 'Symbol', 'field': 'symbol'}],
                                         rows=[{'symbol': s} for s in symbols_data['symbols']]).classes("w-full")
                else:
                    ui.notify("No symbols found", type="warning")

            except Exception as e:
                self.show_error("Symbol search failed", str(e))

        ui.button("🔍 Search Symbols", on_click=search_symbols).classes("bg-blue-500 text-white")

        # Market data for specific symbol
        specific_symbol = ui.input(
            label="Get Market Data for Symbol",
            placeholder="ICICIB22"
        ).classes("w-full mt-4 mb-4")

        market_data_results = ui.column().classes("w-full")

        async def get_market_data():
            symbol = specific_symbol.value.strip().upper()
            if not symbol:
                ui.notify("Enter a symbol", type="warning")
                return

            data = await self.safe_api_call(
                fetch_api,
                f"/sip/market-data/{symbol}"
            )

            if data:
                market_data_results.clear()
                with market_data_results:
                    ui.label(f"Market Data for {symbol}").classes("font-bold mb-2")
                    quality = data.get("data_quality", {})
                    ui.label(f"Records: {quality.get('total_records', 0)}").classes("text-sm")
                    ui.label(f"Period: {quality.get('data_start')} to {quality.get('data_end')}").classes("text-sm")
                    ui.label(f"Coverage Days: {quality.get('coverage_days', 0)}").classes("text-sm")

                    price_stats = data.get("price_stats", {})
                    ui.label(f"Avg Price: ₹{price_stats.get('avg_price', 0):.2f}").classes("text-sm")
                    ui.label(f"Min Price: ₹{price_stats.get('min_price', 0):.2f}").classes("text-sm")
                    ui.label(f"Max Price: ₹{price_stats.get('max_price', 0):.2f}").classes("text-sm")
                    ui.label(f"Avg Volume: {price_stats.get('avg_volume', 0):,.0f}").classes("text-sm")
            else:
                ui.notify("No market data found", type="warning")

        ui.button("📊 Get Market Data", on_click=get_market_data).classes("bg-green-500 text-white")

    async def display_enhanced_backtest_results(self, results, container):
        """Display enhanced backtest results with benchmark comparison, monthly metrics, and charts"""

        with container:
            ui.label("🎯 Enhanced Backtest Results").classes("text-xl font-bold mb-4")

            if not results:
                ui.label("❌ No results to display").classes("text-red-500")
                return

            # Summary statistics
            total_symbols = len(results)
            total_investment = sum(r.get('total_investment', 0) for r in results)
            total_final_value = sum(r.get('final_portfolio_value', 0) for r in results)
            avg_cagr = sum(r.get('cagr_percent', 0) for r in results) / total_symbols if total_symbols > 0 else 0

            with ui.card().classes("w-full p-6 mb-6 bg-gradient-to-r from-blue-50 to-green-50"):
                ui.label("📊 Portfolio Summary").classes("text-lg font-bold mb-4")

                with ui.grid(columns=4).classes("w-full gap-4"):
                    with ui.card().classes("p-4 text-center"):
                        ui.label("📈 Symbols").classes("text-sm text-gray-600")
                        ui.label(str(total_symbols)).classes("text-2xl font-bold text-blue-600")

                    with ui.card().classes("p-4 text-center"):
                        ui.label("💰 Total Investment").classes("text-sm text-gray-600")
                        ui.label(f"₹{total_investment:,.0f}").classes("text-2xl font-bold text-green-600")

                    with ui.card().classes("p-4 text-center"):
                        ui.label("💎 Final Value").classes("text-sm text-gray-600")
                        ui.label(f"₹{total_final_value:,.0f}").classes("text-2xl font-bold text-purple-600")

                    with ui.card().classes("p-4 text-center"):
                        ui.label("📈 Avg CAGR").classes("text-sm text-gray-600")
                        cagr_color = "text-green-600" if avg_cagr >= 0 else "text-red-600"
                        ui.label(f"{avg_cagr:.2f}%").classes(f"text-2xl font-bold {cagr_color}")

            # Individual symbol results with benchmark
            ui.label("🔍 Symbol-wise Performance & Benchmark").classes("text-lg font-bold mb-4")

            sorted_results = sorted(results, key=lambda x: x.get('cagr_percent', 0), reverse=True)

            for result in sorted_results:
                symbol = result.get('symbol', 'Unknown')
                total_investment = result.get('total_investment', 0)
                final_value = result.get('final_portfolio_value', 0)
                cagr = result.get('cagr_percent', 0)
                total_return = result.get('total_return_percent', 0)
                num_trades = result.get('num_trades', 0)
                monthly_limit_exceeded = result.get('monthly_limit_exceeded', 0)
                price_threshold_skipped = result.get('price_threshold_skipped', 0)
                trade_breakdown = result.get('trade_breakdown', {})
                period = result.get('period', 'Unknown period')

                performance_color = "border-green-500" if cagr >= 12 else "border-yellow-500" if cagr >= 8 else "border-red-500"

                with ui.card().classes(f"w-full mb-4 p-4 border-l-4 {performance_color}"):
                    with ui.row().classes("w-full justify-between items-center mb-4"):
                        ui.label(f"📈 {symbol}").classes("text-lg font-bold")

                        if cagr >= 15:
                            badge_class = "bg-green-600 text-white"
                            badge_text = "🚀 Excellent"
                        elif cagr >= 12:
                            badge_class = "bg-green-500 text-white"
                            badge_text = "✅ Good"
                        elif cagr >= 8:
                            badge_class = "bg-yellow-500 text-white"
                            badge_text = "⚠️ Average"
                        else:
                            badge_class = "bg-red-500 text-white"
                            badge_text = "❌ Poor"

                        ui.label(badge_text).classes(f"px-3 py-1 rounded text-xs {badge_class}")

                    with ui.grid(columns=4).classes("w-full gap-4 mb-4"):
                        with ui.column():
                            ui.label("💰 Investment").classes("text-xs text-gray-600")
                            ui.label(f"₹{total_investment:,.0f}").classes("text-sm font-bold")

                            ui.label("💎 Final Value").classes("text-xs text-gray-600 mt-2")
                            ui.label(f"₹{final_value:,.0f}").classes("text-sm font-bold")

                        with ui.column():
                            ui.label("📊 Total Return").classes("text-xs text-gray-600")
                            return_color = "text-green-600" if total_return >= 0 else "text-red-600"
                            ui.label(f"{total_return:+.2f}%").classes(f"text-sm font-bold {return_color}")

                            ui.label("📈 CAGR").classes("text-xs text-gray-600 mt-2")
                            cagr_color = "text-green-600" if cagr >= 8 else "text-red-600"
                            ui.label(f"{cagr:.2f}%").classes(f"text-sm font-bold {cagr_color}")

                        with ui.column():
                            ui.label("🔄 Total Trades").classes("text-xs text-gray-600")
                            ui.label(str(num_trades)).classes("text-sm font-bold")

                            ui.label("Trade Breakdown").classes("text-xs text-gray-600 mt-2")
                            ui.label(f"Regular: {trade_breakdown.get('regular_trades', 0)} | Extreme: {trade_breakdown.get('extreme_trades', 0)}").classes("text-xs")

                        with ui.column():
                            ui.label("📅 Monthly Limit Exceeded").classes("text-xs text-gray-600")
                            ui.label(str(monthly_limit_exceeded)).classes("text-sm font-bold")

                            ui.label("🚫 Price Threshold Skipped").classes("text-xs text-gray-600 mt-2")
                            ui.label(str(price_threshold_skipped)).classes("text-sm font-bold")

                    ui.label(f"📅 Period: {period}").classes("text-sm")

                    if 'benchmark' in result:
                        bench = result['benchmark']
                        outperformance = cagr - bench.get('cagr_percent', 0)
                        out_color = "text-green-600" if outperformance > 0 else "text-red-600"
                        ui.label(f"vs Benchmark CAGR: {outperformance:+.2f}%").classes(f"text-sm {out_color} mt-2")

                    if total_investment > 0:
                        profit_loss = final_value - total_investment
                        profit_color = "text-green-600" if profit_loss >= 0 else "text-red-600"
                        ui.label(f"💰 Absolute P&L: ₹{profit_loss:+,.0f}").classes(f"text-sm {profit_color}")

                    # Chart buttons
                    with ui.row().classes("gap-2 mt-2"):
                        async def show_trade_chart(res=result):
                            await self.display_trade_chart(res.get('symbol'), res.get('trades', []), res.get('benchmark', {}).get('trades', []))

                        ui.button("📉 Trade Chart", on_click=show_trade_chart).classes("bg-blue-500 text-white text-xs")

                        async def show_comparison_chart(res=result):
                            await self.display_comparison_chart(res.get('symbol'), res, res.get('benchmark', {}))

                        ui.button("📊 vs Benchmark", on_click=show_comparison_chart).classes("bg-purple-500 text-white text-xs")

    async def display_trade_chart(self, symbol: str, strategy_trades: List[Dict], benchmark_trades: List[Dict]):
        """Display candlestick chart with strategy and benchmark trade markers"""

        # Parse period from trades if available
        all_trades = strategy_trades + benchmark_trades
        if all_trades:
            min_date = min(t.get('date') for t in all_trades)
            max_date = max(t.get('date') for t in all_trades)
            endpoint = f"/sip/market-data/{symbol}?start_date={min_date}&end_date={max_date}"
        else:
            endpoint = f"/sip/market-data/{symbol}"

        data = await self.safe_api_call(self.fetch_api, endpoint)
        if not data or not data.get('historical_data', data.get('recent_data', [])):
            ui.notify("No data for chart", type="warning")
            return

        historical_data = data.get('historical_data', data.get('recent_data', []))

        # Prepare dataframe
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                                            open=df['ohlc'].apply(lambda x: x['open']),
                                            high=df['ohlc'].apply(lambda x: x['high']),
                                            low=df['ohlc'].apply(lambda x: x['low']),
                                            close=df['ohlc'].apply(lambda x: x['close']))])

        # Add strategy trade markers
        strategy_dates = [pd.to_datetime(t['date']) for t in strategy_trades]
        strategy_prices = [t['price'] for t in strategy_trades]
        fig.add_trace(go.Scatter(x=strategy_dates, y=strategy_prices, mode='markers',
                                 marker=dict(symbol='star', size=12, color='red'),
                                 name='Strategy Trades'))

        # Add benchmark trade markers
        benchmark_dates = [pd.to_datetime(t['date']) for t in benchmark_trades]
        benchmark_prices = [t['price'] for t in benchmark_trades]
        fig.add_trace(go.Scatter(x=benchmark_dates, y=benchmark_prices, mode='markers',
                                 marker=dict(symbol='circle', size=10, color='blue'),
                                 name='Benchmark Trades'))

        fig.update_layout(title=f"{symbol} Price Chart with Trades",
                          xaxis_title="Date",
                          yaxis_title="Price")

        with ui.dialog() as dialog, ui.card().classes("w-[800px] h-[600px]"):
            ui.plotly(fig).classes("w-full h-full")
            ui.button("Close", on_click=dialog.close)

        dialog.open()

    async def display_comparison_chart(self, symbol: str, strategy: Dict, benchmark: Dict):
        """Display line chart comparing strategy and benchmark performance"""

        # Fetch historical data for the period
        period = strategy.get('period', 'Unknown')
        start_date, end_date = period.split(' to ')
        data = await self.safe_api_call(self.fetch_api, f"/sip/market-data/{symbol}?start_date={start_date}&end_date={end_date}")
        if not data or not data.get('historical_data', data.get('recent_data', [])):
            ui.notify("No historical data for comparison chart", type="warning")
            return

        historical_data = data.get('historical_data', data.get('recent_data', []))
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')

        # Compute strategy portfolio values
        strategy_units = 0
        strategy_values = []
        strategy_trades_sorted = sorted(strategy.get('trades', []), key=lambda t: t['date'])
        trade_idx = 0
        for date, row in df.iterrows():
            # Add any trades on this date
            while trade_idx < len(strategy_trades_sorted) and strategy_trades_sorted[trade_idx]['date'] == str(date.date()):
                strategy_units += strategy_trades_sorted[trade_idx]['units']
                trade_idx += 1
            value = strategy_units * row['ohlc']['close']
            strategy_values.append({'date': date, 'value': value})

        # Compute benchmark portfolio values
        benchmark_units = 0
        benchmark_values = []
        benchmark_trades_sorted = sorted(benchmark.get('trades', []), key=lambda t: t['date'])
        trade_idx = 0
        for date, row in df.iterrows():
            # Add any trades on this date
            while trade_idx < len(benchmark_trades_sorted) and benchmark_trades_sorted[trade_idx]['date'] == str(date.date()):
                benchmark_units += benchmark_trades_sorted[trade_idx]['units']
                trade_idx += 1
            value = benchmark_units * row['ohlc']['close']
            benchmark_values.append({'date': date, 'value': value})

        strat_df = pd.DataFrame(strategy_values)
        bench_df = pd.DataFrame(benchmark_values)

        if strat_df.empty or bench_df.empty:
            ui.notify("No data for comparison chart", type="warning")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strat_df['date'], y=strat_df['value'], name='Strategy', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=bench_df['date'], y=bench_df['value'], name='Benchmark', line=dict(color='orange')))

        fig.update_layout(title=f"{symbol} Performance vs Benchmark",
                          xaxis_title="Date",
                          yaxis_title="Portfolio Value")

        with ui.dialog() as dialog, ui.card().classes("w-[800px] h-[600px]"):
            ui.plotly(fig).classes("w-full h-full")
            ui.button("Close", on_click=dialog.close)

        dialog.open()

    async def render_enhanced_analytics_panel(self, fetch_api, user_storage):
        """Enhanced analytics panel with comprehensive portfolio insights and benchmarks"""

        ui.label("📈 Portfolio Analytics Dashboard").classes("text-2xl font-bold mb-4")
        ui.label("Comprehensive analysis of your SIP portfolio performance with benchmarks").classes("text-gray-600 mb-6")

        # Portfolio selector
        portfolios = await self.safe_api_call(fetch_api, "/sip/portfolio")

        if not portfolios:
            with ui.card().classes("w-full p-8 text-center"):
                ui.label("📊 No portfolios found").classes("text-lg text-gray-500")
                ui.label("Create a portfolio first to view analytics").classes("text-sm text-gray-400")
            return

        portfolio_options = {
            p['portfolio_id']: f"{p.get('portfolio_name', 'Unnamed')} ({p.get('portfolio_type', 'single')})"
            for p in portfolios}

        with ui.card().classes("w-full p-4 mb-6"):
            ui.label("Select Portfolio for Analysis").classes("font-bold mb-2")

            selected_portfolio = ui.select(
                options=portfolio_options,
                value=list(portfolio_options.keys())[0] if portfolio_options else None,
                label="Portfolio"
            ).classes("w-full")

        analytics_container = ui.column().classes("w-full")

        async def load_analytics():
            """Load comprehensive analytics for selected portfolio with benchmarks"""
            if not selected_portfolio.value:
                return

            try:
                analytics = await self.safe_api_call(fetch_api, f"/sip/analytics/portfolio/{selected_portfolio.value}")

                analytics_container.clear()

                if analytics:
                    await self.display_portfolio_analytics(analytics, analytics_container)
                else:
                    with analytics_container:
                        ui.label("❌ No analytics data available").classes("text-red-500")

            except Exception as e:
                logger.error(f"Error loading analytics: {e}")
                with analytics_container:
                    ui.label(f"❌ Error loading analytics: {str(e)}").classes("text-red-500")

        # Load analytics when portfolio changes
        selected_portfolio.on_value_change(lambda: load_analytics())

        # Load initial analytics
        await load_analytics()

        # Refresh button
        ui.button("🔄 Refresh Analytics", on_click=load_analytics).classes("bg-blue-500 text-white mt-4")

    async def display_portfolio_analytics(self, analytics, container):
        """Display comprehensive portfolio analytics with benchmarks"""

        with container:
            portfolio_name = analytics.get('portfolio_name', 'Portfolio')
            ui.label(f"📊 Analytics for {portfolio_name}").classes("text-xl font-bold mb-4")

            # Overview metrics
            total_invested = analytics.get('total_invested', 0)
            current_value = analytics.get('current_value', 0)
            total_return_pct = analytics.get('total_return_percent', 0)
            cagr_pct = analytics.get('cagr_percent', 0)
            days_invested = analytics.get('days_invested', 0)
            total_trades = analytics.get('total_trades', 0)

            with ui.card().classes("w-full p-6 mb-6 bg-gradient-to-r from-purple-50 to-blue-50"):
                ui.label("📈 Performance Overview").classes("text-lg font-bold mb-4")

                with ui.grid(columns=3).classes("w-full gap-6"):
                    # Investment metrics
                    with ui.card().classes("p-4 text-center bg-white"):
                        ui.label("💰 Total Invested").classes("text-sm text-gray-600")
                        ui.label(f"₹{total_invested:,.0f}").classes("text-xl font-bold text-blue-600")

                        ui.label("💎 Current Value").classes("text-sm text-gray-600 mt-2")
                        ui.label(f"₹{current_value:,.0f}").classes("text-xl font-bold text-green-600")

                    # Returns metrics
                    with ui.card().classes("p-4 text-center bg-white"):
                        ui.label("📊 Total Return").classes("text-sm text-gray-600")
                        return_color = "text-green-600" if total_return_pct >= 0 else "text-red-600"
                        ui.label(f"{total_return_pct:+.2f}%").classes(f"text-xl font-bold {return_color}")

                        ui.label("📈 CAGR").classes("text-sm text-gray-600 mt-2")
                        cagr_color = "text-green-600" if cagr_pct >= 8 else "text-red-600"
                        ui.label(f"{cagr_pct:.2f}%").classes(f"text-xl font-bold {cagr_color}")

                    # Activity metrics
                    with ui.card().classes("p-4 text-center bg-white"):
                        ui.label("📅 Days Invested").classes("text-sm text-gray-600")
                        ui.label(str(days_invested)).classes("text-xl font-bold text-purple-600")

                        ui.label("🔄 Total Trades").classes("text-sm text-gray-600 mt-2")
                        ui.label(str(total_trades)).classes("text-xl font-bold text-orange-600")

            # Symbol-wise analytics (for multi-symbol portfolios)
            symbols_analytics = analytics.get('symbols_analytics', {})
            if symbols_analytics:
                ui.label("🎯 Symbol-wise Performance").classes("text-lg font-bold mb-4")

                for symbol, symbol_data in symbols_analytics.items():
                    with ui.card().classes("w-full mb-4 p-4 border-l-4 border-blue-500"):
                        with ui.row().classes("w-full justify-between items-center"):
                            ui.label(f"📈 {symbol}").classes("text-lg font-bold")
                            ui.label(f"Allocation: {symbol_data.get('allocation_percent', 0):.1f}%").classes(
                                "text-sm text-gray-600")

                        with ui.grid(columns=4).classes("w-full gap-4 mt-4"):
                            ui.label(f"Invested: ₹{symbol_data.get('invested', 0):,.0f}").classes("text-sm")
                            ui.label(f"Units: {symbol_data.get('units', 0):.2f}").classes("text-sm")
                            ui.label(f"Avg Price: ₹{symbol_data.get('avg_buy_price', 0):.2f}").classes("text-sm")

                            return_pct = symbol_data.get('return_percent', 0)
                            return_color = "text-green-600" if return_pct >= 0 else "text-red-600"
                            ui.label(f"Return: {return_pct:+.2f}%").classes(f"text-sm {return_color}")

    async def render_enhanced_config_panel(self, fetch_api, user_storage):
        """Enhanced configuration panel with templates and presets"""

        ui.label("⚙️ Strategy Configuration").classes("text-2xl font-bold mb-4")
        ui.label("Configure your SIP strategy parameters and save templates").classes("text-gray-600 mb-6")

        # Load default config
        try:
            default_config = await self.safe_api_call(fetch_api, "/sip/config/defaults")
            config_data = default_config.get('default_config', self.default_config)
        except:
            config_data = self.default_config

        with ui.card().classes("w-full p-6 mb-6"):
            ui.label("🎯 Strategy Parameters").classes("text-lg font-bold mb-4")

            with ui.grid(columns=2).classes("w-full gap-6"):
                # Left column - Investment settings
                with ui.column().classes("flex-1"):
                    ui.label("💰 Investment Settings").classes("font-medium mb-3")

                    fixed_investment = ui.number(
                        label="Monthly Investment (₹)",
                        value=config_data.get('fixed_investment', 5000),
                        min=500, step=500
                    ).classes("w-full mb-3")

                    fallback_day = ui.number(
                        label="Investment Day of Month",
                        value=config_data.get('fallback_day', 22),
                        min=1, max=28
                    ).classes("w-full mb-3")

                    min_gap_days = ui.number(
                        label="Minimum Gap Between Investments (days)",
                        value=config_data.get('min_investment_gap_days', 5),
                        min=1, max=30
                    ).classes("w-full")

                    max_monthly = ui.number(
                        label="Max Monthly Amount (₹)",
                        value=config_data.get('max_amount_in_a_month', 20000),
                        min=1000, step=1000
                    ).classes("w-full mb-3")

                # Right column - Drawdown settings
                with ui.column().classes("flex-1"):
                    ui.label("📉 Drawdown Thresholds").classes("font-medium mb-3")

                    major_dd = ui.number(
                        label="Major Drawdown Threshold (%)",
                        value=config_data.get('major_drawdown_threshold', -10),
                        min=-50, max=0
                    ).classes("w-full mb-3")

                    minor_dd = ui.number(
                        label="Minor Drawdown Threshold (%)",
                        value=config_data.get('minor_drawdown_threshold', -4),
                        min=-20, max=0
                    ).classes("w-full mb-3")

                    extreme_dd = ui.number(
                        label="Extreme Drawdown Threshold (%)",
                        value=config_data.get('extreme_drawdown_threshold', -15),
                        min=-50, max=0
                    ).classes("w-full mb-3")

                    rolling_window = ui.number(
                        label="Rolling Window (days)",
                        value=config_data.get('rolling_window', 100),
                        min=20, max=200
                    ).classes("w-full")

            # Investment multipliers
            with ui.card().classes("w-full p-4 mt-6 bg-yellow-50"):
                ui.label("🚀 Investment Multipliers").classes("font-medium mb-3")
                ui.label("How much to increase investment during different market conditions").classes(
                    "text-sm text-gray-600 mb-3")

                with ui.grid(columns=3).classes("w-full gap-4"):
                    minor_mult = ui.number(
                        label="Minor Dip Multiplier",
                        value=config_data.get('minor_drawdown_inv_multiplier', 1.75),
                        min=1.0, max=5.0, step=0.1
                    ).classes("w-full")

                    major_mult = ui.number(
                        label="Major Drawdown Multiplier",
                        value=config_data.get('major_drawdown_inv_multiplier', 3.0),
                        min=1.0, max=8.0, step=0.1
                    ).classes("w-full")

                    extreme_mult = ui.number(
                        label="Extreme Drawdown Multiplier",
                        value=config_data.get('extreme_drawdown_inv_multiplier', 4.0),
                        min=1.0, max=10.0, step=0.1
                    ).classes("w-full")

        # Configuration templates
        with ui.card().classes("w-full p-6"):
            ui.label("📋 Configuration Templates").classes("text-lg font-bold mb-4")

            async def load_templates():
                try:
                    templates_data = await self.safe_api_call(fetch_api, "/sip/templates")
                    if templates_data:
                        return templates_data.get("templates", {})
                except Exception as e:
                    self.show_error("Failed to load templates", str(e))
                    return {}

            templates = await load_templates()

            def apply_template(template_name):
                template_config = templates[template_name]["config"]

                # Update UI components
                major_dd.value = template_config["major_drawdown_threshold"]
                minor_dd.value = template_config["minor_drawdown_threshold"]
                minor_mult.value = template_config["minor_drawdown_inv_multiplier"]
                major_mult.value = template_config["major_drawdown_inv_multiplier"]
                extreme_mult.value = template_config.get("extreme_drawdown_multiplier", 4.0)  # Assuming key if present
                min_gap_days.value = template_config["min_investment_gap_days"]

                ui.notify(f"✅ Applied {template_name} template", type="positive")

            with ui.grid(columns=3).classes("w-full gap-4"):
                for template_name, template_data in templates.items():
                    with ui.card().classes("p-4 text-center hover:shadow-lg cursor-pointer"):
                        ui.label(template_name).classes("font-bold mb-2")
                        ui.label(template_data["description"]).classes("text-sm text-gray-600 mb-3")
                        ui.button(f"Apply {template_name}",
                                  on_click=lambda name=template_name: apply_template(name)).classes(
                            "w-full bg-blue-500 text-white text-sm")

    def load_backtest_template(self, fixed_investment, drawdown_1, drawdown_2, multiplier_2, multiplier_3, fallback_day,
                               min_gap_days):
        """Load a balanced template for backtesting"""
        fixed_investment.value = 5000
        drawdown_1.value = -10.0
        drawdown_2.value = -4.0
        multiplier_2.value = 3.0
        multiplier_3.value = 5.0
        fallback_day.value = 28
        min_gap_days.value = 5

        ui.notify("✅ Loaded balanced template", type="positive")

    async def render_investment_reports_panel(self, fetch_api, user_storage):
        """Render investment reports panel with report generation and history"""

        ui.label("📊 Investment Reports & Analysis").classes("text-2xl font-bold mb-4")
        ui.label("Generate comprehensive investment reports for multiple symbols").classes("text-gray-600 mb-6")

        # Report generation section
        with ui.card().classes("w-full mb-6 p-6"):
            ui.label("📋 Generate Investment Report").classes("text-lg font-semibold mb-4")

            with ui.row().classes("w-full gap-4"):
                # Left column - Symbol selection
                with ui.column().classes("flex-1"):
                    ui.label("Select Symbols").classes("font-medium mb-2")

                    symbols_input = ui.textarea(
                        label="📈 Symbols (one per line or comma-separated)",
                        placeholder="ICICIB22\nGOLDBEES\nITBEES",
                        value="ICICIB22\nCPSEETF\nITBEES"
                    ).classes("w-full")

                    # Report options
                    with ui.row().classes("w-full gap-2 mt-4"):
                        include_risk = ui.checkbox("Include Risk Assessment", value=True)
                        include_allocation = ui.checkbox("Include Allocation Suggestions", value=True)

                # Right column - Configuration
                with ui.column().classes("flex-1"):
                    ui.label("Report Configuration").classes("font-medium mb-2")

                    report_type = ui.select(
                        options={
                            "quick": "Quick Analysis",
                            "comprehensive": "Comprehensive Report",
                            "detailed": "Detailed Analysis"
                        },
                        value="comprehensive",
                        label="Report Type"
                    ).classes("w-full")

                    # SIP configuration (optional)
                    with ui.expansion("Advanced SIP Configuration", icon="settings").classes("w-full mt-4"):
                        with ui.column().classes("w-full gap-2"):
                            fixed_investment = ui.number(
                                label="Monthly Investment (₹)",
                                value=5000,
                                min=1000, step=500
                            ).classes("w-full")

                            drawdown_threshold = ui.number(
                                label="Drawdown Threshold (%)",
                                value=-10.0,
                                min=-50, max=0
                            ).classes("w-full")

        async def render_report_history_card(report, fetch_api):
            """Render individual report history card"""

            with ui.card().classes("w-full mb-4 p-4 border-l-4 border-blue-500"):
                with ui.row().classes("w-full justify-between items-center"):
                    with ui.column():
                        ui.label(f"📊 Report {report['report_id'][:12]}...").classes("text-lg font-bold")
                        ui.label(
                            f"Symbols: {', '.join(report['symbols'][:3])}{'...' if len(report['symbols']) > 3 else ''}").classes(
                            "text-sm text-gray-600")
                        ui.label(f"Generated: {report['generated_at'][:19]}").classes("text-xs text-gray-500")

                    with ui.column().classes("text-right"):
                        summary = report.get('summary', {})
                        ui.label(
                            f"Analyzed: {summary.get('analyzed_symbols', 0)}/{summary.get('total_symbols', 0)}").classes(
                            "text-sm")
                        ui.label(f"Action: {summary.get('overall_action', 'N/A')}").classes("text-sm font-medium")
                        ui.label(f"Risk: {summary.get('risk_level', 'N/A')}").classes("text-sm")

                async def view_report_details():
                    try:
                        details = await self.safe_api_call(fetch_api, f"/sip/reports/{report['report_id']}")
                        if details:
                            await display_investment_report(details['report_data'])
                        else:
                            ui.notify("❌ Could not load report details", type="negative")
                    except Exception as e:
                        ui.notify(f"❌ Error loading report: {str(e)}", type="negative")

                ui.button("👀 View Details", on_click=view_report_details).classes(
                    "bg-blue-500 text-white text-sm px-3 py-1 mt-2")

        async def display_investment_report(report):
            """Display comprehensive investment report in a dialog"""

            with ui.dialog() as dialog, ui.card().classes("w-[90vw] max-w-6xl h-[80vh] overflow-auto"):
                ui.label("📊 Investment Report").classes("text-2xl font-bold mb-4")

                # Report overview
                overall_metrics = report.get('overall_metrics', {})
                portfolio_rec = report.get('portfolio_recommendation', {})
                risk_assessment = report.get('risk_assessment', {})

                with ui.grid(columns=4).classes("w-full gap-4 mb-6"):
                    ui.label(f"Total Symbols: {overall_metrics.get('total_symbols', 0)}").classes(
                        "text-center p-2 bg-blue-100 rounded")
                    ui.label(f"Analyzed: {overall_metrics.get('analyzed_symbols', 0)}").classes(
                        "text-center p-2 bg-green-100 rounded")
                    ui.label(f"Action: {portfolio_rec.get('portfolio_action', 'N/A')}").classes(
                        "text-center p-2 bg-yellow-100 rounded")
                    ui.label(f"Risk: {risk_assessment.get('overall_risk_level', 'N/A')}").classes(
                        "text-center p-2 bg-red-100 rounded")

                # Portfolio recommendations
                with ui.expansion("🎯 Portfolio Recommendations", icon="recommend").classes("w-full mb-4"):
                    recommendations = portfolio_rec.get('recommendations', [])
                    for rec in recommendations:
                        ui.label(f"• {rec}").classes("text-sm mb-1")

                # Symbol reports
                symbol_reports = report.get('symbol_reports', {})
                with ui.expansion("📈 Symbol Analysis", icon="analytics").classes("w-full mb-4"):
                    for symbol, symbol_data in symbol_reports.items():
                        if symbol_data.get('status') == 'SUCCESS':
                            recommendation = symbol_data.get('recommendation', {})
                            signals = symbol_data.get('investment_signals', {})

                            with ui.card().classes("w-full mb-2 p-3"):
                                with ui.row().classes("w-full justify-between"):
                                    ui.label(f"📊 {symbol}").classes("font-bold")
                                    ui.label(recommendation.get('recommendation', 'N/A')).classes("text-sm")

                                ui.label(
                                    f"Signal: {signals.get('signal', 'N/A')} | Confidence: {signals.get('confidence', 0):.2f}").classes(
                                    "text-sm")
                                ui.label(
                                    f"Suggested Allocation: {recommendation.get('suggested_allocation', 0):.1f}%").classes(
                                    "text-sm")

                # Risk assessment
                with ui.expansion("⚠️ Risk Assessment", icon="warning").classes("w-full mb-4"):
                    risk_factors = risk_assessment.get('risk_factors', [])
                    mitigation = risk_assessment.get('mitigation_strategies', [])

                    ui.label("Risk Factors:").classes("font-medium")
                    for factor in risk_factors:
                        ui.label(f"• {factor}").classes("text-sm text-red-600")

                    ui.label("Mitigation Strategies:").classes("font-medium mt-4")
                    for strategy in mitigation:
                        ui.label(f"• {strategy}").classes("text-sm text-green-600")

                ui.button("Close", on_click=dialog.close).classes("mt-4 bg-gray-500 text-white")

            dialog.open()

        async def display_quick_report(report):
            """Display quick report in a dialog"""

            with ui.dialog() as dialog, ui.card().classes("w-[600px]"):
                ui.label("⚡ Quick Investment Report").classes("text-xl font-bold mb-4")

                symbols = report.get('symbols', [])
                ui.label(f"Symbols Analyzed: {', '.join(symbols)}").classes("text-sm text-gray-600 mb-4")

                # Portfolio recommendations
                portfolio_rec = report.get('portfolio_recommendations', {})
                overall_signal = portfolio_rec.get('overall_signal', 'N/A')
                overall_confidence = portfolio_rec.get('overall_confidence', 0)

                with ui.card().classes("w-full p-4 mb-4 bg-blue-50"):
                    ui.label(f"Overall Signal: {overall_signal}").classes("font-bold")
                    ui.label(f"Confidence: {overall_confidence:.2f}").classes("text-sm")

                # Individual symbol recommendations
                symbol_recs = portfolio_rec.get('symbol_recommendations', {})
                if symbol_recs:
                    ui.label("Symbol Signals:").classes("font-medium mb-2")
                    for symbol, rec in symbol_recs.items():
                        signal = rec.get('signal', 'N/A')
                        confidence = rec.get('confidence', 0)
                        ui.label(f"• {symbol}: {signal} ({confidence:.2f})").classes("text-sm")

                # Next steps
                next_steps = report.get('next_steps', [])
                if next_steps:
                    ui.label("Next Steps:").classes("font-medium mt-4 mb-2")
                    for step in next_steps:
                        ui.label(f"• {step}").classes("text-sm text-blue-600")

                ui.button("Close", on_click=dialog.close).classes("mt-4 bg-gray-500 text-white")

            dialog.open()

        # Report generation buttons
        with ui.row().classes("gap-4 mb-6"):
            async def generate_comprehensive_report():
                try:
                    ui.notify("🔄 Generating comprehensive investment report...", type="info")

                    # Parse symbols
                    symbols_text = symbols_input.value.strip()
                    if not symbols_text:
                        ui.notify("❌ Please enter at least one symbol", type="negative")
                        return

                    # Parse symbols (handle both newlines and commas)
                    symbols = []
                    for line in symbols_text.replace(',', '\n').split('\n'):
                        symbol = line.strip().upper()
                        if symbol:
                            symbols.append(symbol)

                    if not symbols:
                        ui.notify("❌ No valid symbols found", type="negative")
                        return

                    # Prepare request
                    config = {
                        "fixed_investment": fixed_investment.value,
                        "major_drawdown_threshold": drawdown_threshold.value,
                        "minor_drawdown_threshold": -4.0,
                        "minor_drawdown_inv_multiplier": 3.0,
                        "major_drawdown_inv_multiplier": 5.0,
                        "extreme_drawdown_inv_multiplier": 4.0,
                        "rolling_window": 100,
                        "fallback_day": 22,
                        "min_investment_gap_days": 5
                    }

                    request_data = {
                        "symbols": symbols,
                        "config": config,
                        "report_type": report_type.value,
                        "include_risk_assessment": include_risk.value,
                        "include_allocation_suggestions": include_allocation.value
                    }

                    # Generate report
                    report = await self.safe_api_call(fetch_api, "/sip/reports/investment", method="POST", data=request_data)

                    if report:
                        ui.notify("✅ Investment report generated successfully!", type="positive")
                        await display_investment_report(report)
                    else:
                        ui.notify("❌ Failed to generate report", type="negative")

                except Exception as e:
                    ui.notify(f"❌ Error generating report: {str(e)}", type="negative")

            async def generate_quick_report():
                try:
                    symbols_text = symbols_input.value.strip()
                    if not symbols_text:
                        ui.notify("❌ Please enter at least one symbol", type="negative")
                        return

                    # Parse and format symbols for URL
                    symbols = []
                    for line in symbols_text.replace(',', '\n').split('\n'):
                        symbol = line.strip().upper()
                        if symbol:
                            symbols.append(symbol)

                    if not symbols:
                        ui.notify("❌ No valid symbols found", type="negative")
                        return

                    symbols_param = ",".join(symbols[:10])  # Limit to 10 for quick report

                    ui.notify("🚀 Generating quick report...", type="info")

                    # Generate quick report
                    report = await self.safe_api_call(fetch_api, f"/sip/reports/quick/{symbols_param}")

                    if report:
                        ui.notify("✅ Quick report generated!", type="positive")
                        await display_quick_report(report)
                    else:
                        ui.notify("❌ Failed to generate quick report", type="negative")

                except Exception as e:
                    ui.notify(f"❌ Error: {str(e)}", type="negative")

            ui.button("📊 Generate Comprehensive Report", on_click=generate_comprehensive_report).classes(
                "bg-blue-600 text-white px-6 py-2")
            ui.button("⚡ Quick Report", on_click=generate_quick_report).classes("bg-green-600 text-white px-4 py-2")

        # Report history section
        reports_container = ui.column().classes("w-full")

        async def load_report_history():
            """Load and display report history"""
            try:
                reports = await self.safe_api_call(fetch_api, "/sip/reports/history?limit=10")

                reports_container.clear()

                if reports:
                    with reports_container:
                        ui.label("📋 Recent Reports").classes("text-lg font-semibold mb-4")

                        for report in reports:
                            await render_report_history_card(report, fetch_api)
                else:
                    with reports_container:
                        ui.label("📝 No reports generated yet").classes("text-gray-500 text-center py-4")

            except Exception as e:
                logger.error(f"Error loading report history: {e}")
                with reports_container:
                    ui.label("❌ Error loading report history").classes("text-red-500")

        # Load report history on page load
        await load_report_history()

        # Refresh button
        ui.button("🔄 Refresh History", on_click=load_report_history).classes("bg-gray-500 text-white mt-4")

    async def show_enhanced_portfolio_performance(self, portfolio_id, fetch_api):
        """Show enhanced portfolio performance with benchmarks"""
        try:
            performance = await self.safe_api_call(fetch_api, f"/sip/performance/{portfolio_id}")

            with ui.dialog() as dialog, ui.card().classes("w-[600px]"):
                ui.label("📊 Portfolio Performance").classes("text-xl font-bold mb-4")

                if performance:
                    summary = performance.get("performance_summary", {})
                    ui.label(f"Invested: ₹{summary.get('total_invested', 0):,.0f}").classes("text-sm")
                    ui.label(f"Current Value: ₹{summary.get('current_value', 0):,.0f}").classes("text-sm")
                    ui.label(f"Return: {summary.get('total_return_percent', 0):+.2f}%").classes("text-sm")
                    ui.label(f"CAGR: {summary.get('cagr_percent', 0):.2f}%").classes("text-sm")
                    ui.label(f"Days Invested: {summary.get('days_invested', 0)}").classes("text-sm")

                    # Benchmark comparison if available
                    if "benchmark" in performance:
                        bench = performance["benchmark"]
                        outperformance = summary.get('cagr_percent', 0) - bench.get('cagr_percent', 0)
                        out_color = "text-green-600" if outperformance > 0 else "text-red-600"
                        ui.label(f"vs Benchmark: {outperformance:+.2f}% CAGR").classes(f"text-sm {out_color} mt-4")

                else:
                    ui.label("No performance data available").classes("text-gray-500")

                ui.button("Close", on_click=dialog.close).classes("mt-4")

            dialog.open()

        except Exception as e:
            ui.notify(f"Error loading performance: {str(e)}", type="negative")

async def render_sip_strategy_page(fetch_api, user_storage):
    """Render the enhanced SIP strategy UI"""
    strategy = EnhancedSIPStrategy()
    await strategy.render(fetch_api, user_storage)