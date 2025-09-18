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
        # Debug logging to see what parameters are received
        logger.error(f"show_error called with - error: '{error}', details: '{details}'")
        
        # Create error message
        if details and details != "Unknown API error":
            error_message = f"❌ {error}\nDetails: {details}"
        else:
            error_message = f"❌ {error}"
            
        # Show notification
        ui.notify(error_message, type="negative", timeout=8000)
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
            
            # Check if response contains an error
            if isinstance(response, dict) and response.get("error"):
                # Debug logging to understand the response structure
                logger.error(f"API Error Response: {response}")
                
                # Extract error message from the response
                error_obj = response["error"]
                
                # Try to get the error message from different possible structures
                if isinstance(error_obj, dict):
                    error_detail = error_obj.get("message") or error_obj.get("detail") or error_obj.get("error")
                    logger.error(f"Error object keys: {list(error_obj.keys())}")
                    logger.error(f"Message value: {error_obj.get('message')}")
                    logger.error(f"Detail value: {error_obj.get('detail')}")
                elif isinstance(error_obj, str):
                    error_detail = error_obj
                else:
                    error_detail = None
                
                # If no error detail found in error object, try the response itself
                if not error_detail:
                    error_detail = response.get("detail") or response.get("message")
                
                logger.error(f"Final extracted error detail: '{error_detail}'")
                
                # Use the extracted error detail or a generic message
                if error_detail and error_detail != "Unknown API error":
                    self.show_error(error_message, error_detail)
                else:
                    self.show_error(error_message, "Backend returned an error")
                
                return None
            
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
        """Render enhanced SIP strategy interface with dramatic visual improvements"""
        self.fetch_api = fetch_api
        
        # Main wrapper with SIP strategy specific styling class
        with ui.column().classes("sip-strategy-page w-full min-h-screen p-2"):
            
            # Page header with tabs in the same row
            with ui.card().classes("w-full mb-4"):
                with ui.row().classes("w-full items-center justify-between p-4"):
                    with ui.column().classes("gap-2"):
                        ui.label("🚀 SIP Strategy Management").classes("page-title-standard text-2xl font-bold")
                        ui.label("Advanced systematic investment planning with AI-powered analytics").classes("page-subtitle-standard text-sm")
                    
                    # Tabs in the same row as header
                    with ui.tabs().classes("flex-1 ml-8") as tabs:
                        backtest_tab = ui.tab("📊 Backtesting", icon="analytics").classes("text-sm px-4 py-2")
                        portfolio_tab = ui.tab("💼 Portfolios", icon="account_balance_wallet").classes("text-sm px-4 py-2")
                        multi_portfolio_tab = ui.tab("🎯 Multi-Portfolio", icon="dashboard").classes("text-sm px-4 py-2")
                        reports_tab = ui.tab("📋 Reports", icon="description").classes("text-sm px-4 py-2")
                        signals_tab = ui.tab("📡 Signals", icon="notifications_active").classes("text-sm px-4 py-2")
                        analytics_tab = ui.tab("📈 Analytics", icon="trending_up").classes("text-sm px-4 py-2")
                        config_tab = ui.tab("⚙️ Configuration", icon="settings").classes("text-sm px-4 py-2")
                        scheduler_tab = ui.tab("⏰ Scheduler", icon="schedule").classes("text-sm px-4 py-2")

            with ui.tab_panels(tabs, value=backtest_tab).classes("w-full"):
                with ui.tab_panel(backtest_tab).classes("w-full"):
                    await self.render_enhanced_backtest_panel(fetch_api, user_storage)

                with ui.tab_panel(portfolio_tab).classes("w-full"):
                    await self.render_enhanced_portfolio_panel(fetch_api, user_storage)

                with ui.tab_panel(multi_portfolio_tab).classes("w-full"):
                    await self.render_multi_portfolio_panel(fetch_api, user_storage)

                with ui.tab_panel(reports_tab).classes("w-full"):
                    await self.render_investment_reports_panel(fetch_api, user_storage)

                with ui.tab_panel(signals_tab).classes("w-full"):
                    await self.render_enhanced_signals_panel(fetch_api, user_storage)

                with ui.tab_panel(analytics_tab).classes("w-full"):
                    await self.render_enhanced_analytics_panel(fetch_api, user_storage)

                with ui.tab_panel(config_tab).classes("w-full"):
                    await self.render_enhanced_config_panel(fetch_api, user_storage)

                with ui.tab_panel(scheduler_tab).classes("w-full"):
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
            ui.button("🔄 Refresh Status", on_click=refresh_scheduler_status).classes("bg-blue-500 btn-primary-text")
            ui.button("▶️ Start Scheduler", on_click=lambda: self.control_scheduler(fetch_api, "start")).classes("bg-green-500 btn-primary-text")
            ui.button("⏸️ Pause Scheduler", on_click=lambda: self.control_scheduler(fetch_api, "pause")).classes("bg-yellow-500 btn-primary-text")
            ui.button("🛑 Stop Scheduler", on_click=lambda: self.control_scheduler(fetch_api, "shutdown")).classes("bg-red-500 btn-primary-text")

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
        """Enhanced backtest panel with side menu navigation and modern styling"""
        
        # Side menu layout for backtesting options
        with ui.row().classes("w-full gap-4"):
            # Left sidebar menu
            with ui.card().classes("w-72 h-fit"):
                with ui.column().classes("w-full"):
                    with ui.row().classes("p-4 border-b border-slate-600/30"):
                        ui.icon("menu", size="1.2rem").classes("text-cyan-400")
                        ui.label("Backtesting Options").classes("ml-3 text-lg font-bold")
                    
                    # Menu items container
                    menu_container = ui.column().classes("w-full p-3")
                    
                    # Create menu state
                    selected_option = {"current": "main_backtest"}
                    content_area = None
                    
                    def create_modern_menu_item(key: str, label: str, icon: str, description: str, color: str = "slate"):
                        is_selected = selected_option["current"] == key
                        btn_classes = f"w-full justify-start p-3 mb-2 rounded-lg transition-all duration-300 {'bg-cyan-500/20 border-cyan-400/60' if is_selected else 'hover:bg-slate-700/60 border-slate-600/40 hover:border-cyan-400/40'} border"
                        
                        with ui.button().classes(btn_classes).props("flat no-caps") as btn:
                            with ui.row().classes("w-full items-center gap-3"):
                                ui.icon(icon, size="1.2rem").classes(f"text-{color}-400")
                                with ui.column().classes("gap-1 flex-1 text-left"):
                                    ui.label(label).classes(f"font-bold text-sm {'text-cyan-300' if is_selected else 'text-slate-200'}")
                                    ui.label(description).classes("text-xs text-slate-400")
                                if is_selected:
                                    ui.icon("chevron_right", size="1rem").classes("text-cyan-400")
                        
                        async def on_click():
                            selected_option["current"] = key
                            await update_content()
                            await refresh_menu()
                        
                        btn.on_click(on_click)
                        return btn
                    
                    async def refresh_menu():
                        menu_container.clear()
                        with menu_container:
                            create_modern_menu_item("main_backtest", "Main Backtest", "analytics", "Core strategy testing", "cyan")
                            create_modern_menu_item("multi_configs", "Batch Multi-Configs", "layers", "Test multiple configurations", "green")
                            create_modern_menu_item("optimize", "Optimize Config", "tune", "Parameter optimization", "purple")
                            create_modern_menu_item("quick_test", "Quick Test", "speed", "Rapid validation", "yellow")
                            create_modern_menu_item("benchmark", "Benchmark Test", "compare_arrows", "Compare with benchmarks", "orange")
                            create_modern_menu_item("symbols", "Symbols Search", "search", "Find trading symbols", "blue")
            
            # Right content area
            content_area = ui.column().classes("flex-1")
            
            # Content panels based on selection
            async def update_content():
                content_area.clear()
                with content_area:
                    with ui.card().classes("w-full p-4"):
                        if selected_option["current"] == "main_backtest":
                            await self.render_main_backtest_section(fetch_api, user_storage)
                        elif selected_option["current"] == "multi_configs":
                            await self.render_batch_multi_configs_section(fetch_api, user_storage)
                        elif selected_option["current"] == "optimize":
                            await self.render_optimize_config_section(fetch_api, user_storage)
                        elif selected_option["current"] == "quick_test":
                            await self.render_quick_test_section(fetch_api, user_storage)
                        elif selected_option["current"] == "benchmark":
                            await self.render_benchmark_test_section(fetch_api, user_storage)
                        elif selected_option["current"] == "symbols":
                            await self.render_symbols_search_section(fetch_api, user_storage)
            
            # Initialize menu and content
            await refresh_menu()
            await update_content()

    async def clear_results(self):
        self.results_container.clear()
        ui.notify("Results cleared", type="info")

    async def render_main_backtest_section(self, fetch_api, user_storage):
        """Render main backtest configuration and results with enhanced styling"""

        # Configuration header
        with ui.row().classes("w-full items-center mb-4"):
            ui.icon("settings", size="1.5rem").classes("text-cyan-400")
            ui.label("📋 Backtest Configuration").classes("text-lg font-bold ml-3")
            ui.separator().classes("flex-1 ml-4")
            ui.chip("Main Backtest", icon="analytics", color="primary").classes("text-sm px-3 py-1").props("outline")

            with ui.row().classes("w-full gap-6"):
                with ui.column().classes("flex-1"):
                    with ui.card().classes("w-full p-4 mb-4"):
                        with ui.row().classes("items-center mb-4"):
                            ui.icon("tune", size="1.5rem").classes("text-blue-400")
                            ui.label("Basic Settings").classes("font-bold text-lg ml-3 text-blue-300")

                        self.symbols_input = ui.textarea(
                            label="📈 Symbols (one per line)",
                            placeholder="ICICIB22\nHDFCNEXT50\nMOTILALOSML\nNIFTYBEES",
                            value="ICICIB22\nGOLDBEES"
                        ).classes("w-full").props("outlined dense")

                        with ui.row().classes("w-full gap-3"):
                            self.start_date = ui.input("Start Date", value="2020-01-01").props(
                                "outlined dense type=date").classes("flex-1 min-w-0")

                            self.end_date = ui.input("End Date", value=datetime.now().strftime("%Y-%m-%d")).props(
                                "outlined dense type=date").classes("flex-1 min-w-0")

                        self.enable_monthly_limits = ui.checkbox("Enable Monthly Limits", value=True).classes("mt-3")

                with ui.column().classes("flex-1"):
                    with ui.card().classes("w-full p-4 mb-4"):
                        with ui.row().classes("items-center mb-4"):
                            ui.icon("psychology", size="1.5rem").classes("text-green-400")
                            ui.label("Strategy Parameters").classes("font-bold text-lg ml-3 text-green-300")

                        self.fixed_investment = ui.number(
                            label="💰 Monthly Investment (₹)",
                            value=self.default_config["fixed_investment"],
                            min=1000, step=500
                        ).classes("w-full").props("outlined dense")

                        with ui.row().classes("w-full gap-3"):
                            self.major_dd = ui.number(
                                label="Major Drawdown (%)",
                                value=self.default_config["major_drawdown_threshold"],
                                min=-50, max=0, step=1
                            ).classes("flex-1").props("outlined dense")

                            self.major_mult = ui.number(
                                label="Major Multiplier",
                                value=self.default_config["major_drawdown_inv_multiplier"],
                                min=1, max=10, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        with ui.row().classes("w-full gap-3"):
                            self.minor_dd = ui.number(
                                label="Minor Drawdown (%)",
                                value=self.default_config["minor_drawdown_threshold"],
                                min=-20, max=0, step=1
                            ).classes("flex-1").props("outlined dense")

                            self.minor_mult = ui.number(
                                label="Minor Multiplier",
                                value=self.default_config["minor_drawdown_inv_multiplier"],
                                min=1, max=8, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        with ui.row().classes("w-full gap-3"):
                            self.extreme_dd = ui.number(
                                label="Extreme Drawdown (%)",
                                value=self.default_config["extreme_drawdown_threshold"],
                                min=-50, max=0, step=1
                            ).classes("flex-1").props("outlined dense")

                            self.extreme_mult = ui.number(
                                label="Extreme Multiplier",
                                value=self.default_config["extreme_drawdown_inv_multiplier"],
                                min=1, max=10, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        with ui.row().classes("w-full gap-3"):
                            self.fallback_day = ui.number(
                                label="📅 Investment Day",
                                value=self.default_config["fallback_day"],
                                min=1, max=28, step=1
                            ).classes("flex-1").props("outlined dense")

                            self.min_gap_days = ui.number(
                                label="⏰ Min Gap (days)",
                                value=self.default_config["min_investment_gap_days"],
                                min=1, max=30, step=1
                            ).classes("flex-1").props("outlined dense")

                        self.rolling_window = ui.number(
                            label="🔄 Rolling Window (days)",
                            value=self.default_config["rolling_window"],
                            min=20, max=365, step=5
                        ).classes("w-full").props("outlined dense")

                        self.price_reduction_threshold = ui.number(
                            label="📉 Price Reduction Threshold (%)",
                            value=self.default_config["price_reduction_threshold"],
                            min=1, max=10, step=0.5
                        ).classes("w-full").props("outlined dense")

                        self.max_monthly = ui.number(
                            label="📅 Max Monthly Amount (₹)",
                            value=self.default_config["fixed_investment"] * 4,
                            min=1000, step=1000
                        ).classes("w-full").props("outlined dense")

        # Action buttons
        with ui.card().classes("w-full p-4 mt-4"):
            with ui.row().classes("w-full items-center justify-between mb-4"):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("rocket_launch", size="1.5rem").classes("text-orange-400")
                    ui.label("Backtest Actions").classes("font-bold text-lg text-orange-300")
                
                ui.button("🗑️ Clear Results", on_click=self.clear_results).classes(
                    "bg-red-500 text-white px-4 py-2 rounded").props("outline")
            
            with ui.row().classes("gap-4 w-full justify-center"):
                ui.button("🚀 RUN BACKTEST", icon="analytics", on_click=self.run_enhanced_backtest).classes(
                    "bg-blue-500 text-white px-6 py-3 font-bold")
                ui.button("📈 COMPARE STRATEGIES", icon="compare_arrows", on_click=self.run_strategy_comparison).classes(
                    "bg-green-500 text-white px-6 py-3 font-bold")

        # Results container
        self.results_container = ui.column().classes("w-full mt-4")

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

                # Advanced Configuration Section
                with ui.expansion("⚙️ Advanced Configuration").classes("w-full mt-4"):
                    with ui.column().classes("gap-4"):
                        # Investment Multipliers
                        with ui.row().classes("w-full gap-4"):
                            default_minor_mult = ui.number(
                                label="📈 Minor Multiplier",
                                value=1.75,
                                min=1.0, max=5.0, step=0.25
                            ).classes("flex-1").props("outlined dense")
                            
                            default_major_mult = ui.number(
                                label="📈 Major Multiplier",
                                value=3.0,
                                min=1.0, max=10.0, step=0.5
                            ).classes("flex-1").props("outlined dense")
                            
                            default_extreme_mult = ui.number(
                                label="📈 Extreme Multiplier",
                                value=4.0,
                                min=1.0, max=15.0, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        # Technical Parameters
                        with ui.row().classes("w-full gap-4"):
                            default_rolling_window = ui.number(
                                label="📊 Rolling Window",
                                value=100,
                                min=20, max=500, step=10
                            ).classes("flex-1").props("outlined dense")
                            
                            default_price_threshold = ui.number(
                                label="💰 Price Reduction Threshold (%)",
                                value=4.0,
                                min=1.0, max=20.0, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        # Additional parameters
                        with ui.row().classes("w-full gap-4"):
                            default_fallback_day = ui.number(
                                label="📅 Fallback Day",
                                value=22,
                                min=1, max=31, step=1
                            ).classes("flex-1").props("outlined dense")
                            
                            default_min_gap = ui.number(
                                label="⏱️ Min Investment Gap (days)",
                                value=5,
                                min=1, max=30, step=1
                            ).classes("flex-1").props("outlined dense")

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
                        "minor_drawdown_inv_multiplier": default_minor_mult.value,
                        "major_drawdown_inv_multiplier": default_major_mult.value,
                        "extreme_drawdown_inv_multiplier": default_extreme_mult.value,
                        "rolling_window": int(default_rolling_window.value),
                        "fallback_day": int(default_fallback_day.value),
                        "min_investment_gap_days": int(default_min_gap.value),
                        "max_amount_in_a_month": max_monthly.value,
                        "price_reduction_threshold": default_price_threshold.value,
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

        # Existing multi-portfolios
        with ui.card().classes("w-full mt-6"):
            with ui.row().classes("w-full items-center p-4 border-b border-slate-600/30"):
                ui.icon("dashboard", size="1.5rem").classes("text-purple-400")
                ui.label("Your Multi-Portfolios").classes("text-lg ml-3 font-bold")
                ui.separator().classes("flex-1 ml-4")
                ui.chip("Multi-Symbol", icon="trending_up", color="purple").classes("text-sm px-3 py-1").props("outline")
            
            multi_portfolios_container = ui.column().classes("w-full p-4 gap-4")

        async def refresh_multi_portfolios():
            """Refresh multi-portfolio list with enhanced display"""
            try:
                portfolios = await self.safe_api_call(fetch_api, "/sip/portfolio")

                multi_portfolios_container.clear()

                if portfolios:
                    # Filter only multi-portfolios
                    multi_portfolios = [p for p in portfolios if p.get('portfolio_type') == 'multi']
                    
                    if multi_portfolios:
                        with multi_portfolios_container:
                            for portfolio in multi_portfolios:
                                await self.render_enhanced_portfolio_card(portfolio, fetch_api, refresh_multi_portfolios)
                    else:
                        with multi_portfolios_container:
                            with ui.column().classes("w-full items-center py-12 gap-4"):
                                ui.icon("dashboard", size="3rem").classes("text-slate-400")
                                ui.label("No multi-portfolios found").classes("text-lg font-medium text-slate-300")
                                ui.label("Create your first multi-portfolio above to get started!").classes(
                                    "text-sm text-slate-400 text-center")
                else:
                    with multi_portfolios_container:
                        with ui.column().classes("w-full items-center py-8 gap-3"):
                            ui.icon("error", size="2rem").classes("text-red-400")
                            ui.label("❌ Error loading portfolios").classes("text-lg font-medium text-red-400")
                            ui.label("Please try refreshing the page").classes("text-sm text-slate-400")

            except Exception as e:
                logger.error(f"Error fetching multi-portfolios: {e}")
                with multi_portfolios_container:
                    with ui.column().classes("w-full items-center py-8 gap-3"):
                        ui.icon("error", size="2rem").classes("text-red-400")
                        ui.label("❌ Error loading multi-portfolios").classes("text-lg font-medium text-red-400")
                        ui.label("Please try refreshing the page").classes("text-sm text-slate-400")

        # Load multi-portfolios on page load
        await refresh_multi_portfolios()

        # Refresh button
        ui.button("🔄 Refresh Multi-Portfolios", on_click=refresh_multi_portfolios).classes("bg-gray-500 text-white mt-4 px-4 py-2 rounded")

    async def render_enhanced_portfolio_panel(self, fetch_api, user_storage):
        """Enhanced portfolio management interface with modern styling and GTT status"""

        # Header
        with ui.card().classes("w-full mb-4"):
            with ui.row().classes("w-full items-center p-4"):
                ui.icon("account_balance_wallet", size="2rem").classes("text-green-400")
                with ui.column().classes("ml-4 gap-2"):
                    ui.label("💼 SIP Portfolio Management").classes("text-xl font-bold")
                    ui.label("Create and manage your systematic investment portfolios").classes("text-sm text-slate-300")

        # Single symbol portfolio creation
        with ui.card().classes("w-full mb-4"):
            with ui.row().classes("w-full items-center p-4 border-b border-slate-600/30"):
                ui.icon("add_circle", size="1.5rem").classes("text-cyan-400")
                ui.label("Create Single-Symbol Portfolio").classes("text-lg ml-3 font-bold")
                ui.separator().classes("flex-1 ml-4")
                ui.chip("Quick Setup", icon="speed", color="positive").classes("text-sm px-3 py-1").props("outline")
            
            with ui.column().classes("p-4 gap-4"):

                with ui.row().classes("w-full gap-4"):
                    symbol_input = ui.input(
                        label="📈 Symbol",
                        placeholder="ICICIB22"
                    ).classes("flex-1").props("outlined dense")

                    portfolio_name_input = ui.input(
                        label="📁 Portfolio Name (optional)",
                        placeholder="My ICICI Bank SIP"
                    ).classes("flex-1").props("outlined dense")

                with ui.row().classes("w-full gap-4 mt-4"):
                    investment_amount = ui.number(
                        label="💰 Monthly Investment (₹)",
                        value=5000,
                        min=500, step=500
                    ).classes("flex-1").props("outlined dense")

                    fallback_day = ui.number(
                        label="📅 Investment Day",
                        value=22,
                        min=1, max=28
                    ).classes("flex-1").props("outlined dense")

                    min_gap = ui.number(
                        label="⏰ Min Gap (days)",
                        value=5,
                        min=1, max=30
                    ).classes("flex-1").props("outlined dense")

                max_monthly = ui.number(
                    label="📅 Max Monthly Amount (₹)",
                    value=20000,
                    min=1000, step=1000
                ).classes("w-full mt-4").props("outlined dense")

                # Advanced Configuration Section
                with ui.expansion("⚙️ Advanced Configuration").classes("w-full mt-4"):
                    with ui.column().classes("gap-4"):
                        # Drawdown Thresholds
                        with ui.row().classes("w-full gap-4"):
                            minor_dd = ui.number(
                                label="📉 Minor Drawdown (%)",
                                value=-4.0,
                                min=-20, max=0, step=0.5
                            ).classes("flex-1").props("outlined dense")
                            
                            major_dd = ui.number(
                                label="📉 Major Drawdown (%)",
                                value=-10.0,
                                min=-30, max=0, step=0.5
                            ).classes("flex-1").props("outlined dense")
                            
                            extreme_dd = ui.number(
                                label="📉 Extreme Drawdown (%)",
                                value=-15.0,
                                min=-50, max=0, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        # Investment Multipliers
                        with ui.row().classes("w-full gap-4"):
                            minor_mult = ui.number(
                                label="📈 Minor Multiplier",
                                value=1.75,
                                min=1.0, max=5.0, step=0.25
                            ).classes("flex-1").props("outlined dense")
                            
                            major_mult = ui.number(
                                label="📈 Major Multiplier",
                                value=3.0,
                                min=1.0, max=10.0, step=0.5
                            ).classes("flex-1").props("outlined dense")
                            
                            extreme_mult = ui.number(
                                label="📈 Extreme Multiplier",
                                value=4.0,
                                min=1.0, max=15.0, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        # Technical Parameters
                        with ui.row().classes("w-full gap-4"):
                            rolling_window = ui.number(
                                label="📊 Rolling Window",
                                value=100,
                                min=20, max=500, step=10
                            ).classes("flex-1").props("outlined dense")
                            
                            price_threshold = ui.number(
                                label="💰 Price Reduction Threshold (%)",
                                value=4.0,
                                min=1.0, max=20.0, step=0.5
                            ).classes("flex-1").props("outlined dense")

            async def create_single_portfolio():
                try:
                    if not symbol_input.value.strip():
                        ui.notify("❌ Please enter a symbol", type="negative")
                        return

                    config = {
                        "fixed_investment": investment_amount.value,
                        "major_drawdown_threshold": major_dd.value,
                        "minor_drawdown_threshold": minor_dd.value,
                        "extreme_drawdown_threshold": extreme_dd.value,
                        "minor_drawdown_inv_multiplier": minor_mult.value,
                        "major_drawdown_inv_multiplier": major_mult.value,
                        "extreme_drawdown_inv_multiplier": extreme_mult.value,
                        "rolling_window": int(rolling_window.value),
                        "fallback_day": fallback_day.value,
                        "min_investment_gap_days": min_gap.value,
                        "max_amount_in_a_month": max_monthly.value,
                        "price_reduction_threshold": price_threshold.value,
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

            # Create Portfolio Button - placed outside the function
            with ui.row().classes("w-full justify-center mt-4"):
                ui.button("✨ Create Portfolio", icon="add_circle", on_click=create_single_portfolio).classes(
                    "bg-blue-500 text-white px-6 py-3 font-bold")

        # Existing portfolios
        with ui.card().classes("w-full mt-4"):
            with ui.row().classes("w-full items-center p-4 border-b border-slate-600/30"):
                ui.icon("folder", size="1.5rem").classes("text-green-400")
                ui.label("Your SIP Portfolios").classes("text-lg ml-3 font-bold")
                ui.separator().classes("flex-1 ml-4")
                ui.chip("Active", icon="trending_up", color="positive").classes("text-sm px-3 py-1").props("outline")
            
            portfolios_container = ui.column().classes("w-full p-4 gap-4")

        async def refresh_portfolios():
            """Refresh portfolio list with enhanced display including GTT status"""
            try:
                portfolios = await self.safe_api_call(fetch_api, "/sip/portfolio")

                portfolios_container.clear()

                if portfolios:
                    self.active_portfolios = portfolios

                    with portfolios_container:
                        for portfolio in portfolios:
                            await self.render_enhanced_portfolio_card(portfolio, fetch_api, refresh_portfolios)
                else:
                    with portfolios_container:
                        with ui.column().classes("w-full items-center py-12 gap-4"):
                            ui.icon("portfolio", size="3rem").classes("text-slate-400")
                            ui.label("No portfolios found").classes("text-lg font-medium text-slate-300")
                            ui.label("Create your first SIP portfolio above to get started!").classes(
                                "text-sm text-slate-400 text-center")

            except Exception as e:
                logger.error(f"Error fetching portfolios: {e}")
                with portfolios_container:
                    with ui.column().classes("w-full items-center py-8 gap-3"):
                        ui.icon("error", size="2rem").classes("text-red-400")
                        ui.label("❌ Error loading portfolios").classes("text-lg font-medium text-red-400")
                        ui.label("Please try refreshing the page").classes("text-sm text-slate-400")

        # Load portfolios on page load
        await refresh_portfolios()

        # Refresh button
        ui.button("🔄 Refresh Portfolios", on_click=refresh_portfolios).classes("bg-gray-500 text-white mt-4 px-4 py-2 rounded")

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

                async def edit_portfolio():
                    await self.show_portfolio_update_dialog(portfolio['portfolio_id'], fetch_api, refresh_callback)

                ui.button("✏️ Edit", on_click=edit_portfolio).classes("bg-orange-500 text-white text-xs px-3 py-1")

                if status == 'active':
                    async def invest_now():
                        await self.show_enhanced_investment_dialog(portfolio['portfolio_id'], fetch_api,
                                                                   refresh_callback)

                    ui.button("💰 Invest", on_click=invest_now).classes("bg-green-500 text-white text-xs px-3 py-1")

                if status == 'active':
                    async def pause_portfolio():
                        await self.pause_portfolio(portfolio['portfolio_id'], fetch_api, refresh_callback)

                    ui.button("⏸️ Pause", on_click=pause_portfolio).classes(
                        "bg-yellow-500 text-white text-xs px-3 py-1")

                    async def cancel_portfolio():
                        await self.cancel_portfolio(portfolio['portfolio_id'], fetch_api, refresh_callback)

                    ui.button("🚫 Cancel", on_click=cancel_portfolio).classes(
                        "bg-red-600 text-white text-xs px-3 py-1")

                elif status == 'paused':
                    async def resume_portfolio():
                        await self.resume_portfolio(portfolio['portfolio_id'], fetch_api, refresh_callback)

                    ui.button("▶️ Resume", on_click=resume_portfolio).classes(
                        "bg-green-500 text-white text-xs px-3 py-1")

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

    async def show_portfolio_update_dialog(self, portfolio_id, fetch_api, refresh_callback):
        """Show dialog to update portfolio parameters"""
        try:
            # Get current portfolio details
            portfolio_details = await self.safe_api_call(fetch_api, f"/sip/portfolio/{portfolio_id}")
            
            if not portfolio_details:
                ui.notify("❌ Failed to load portfolio details", type="negative")
                return

            # Check if it's a multi-portfolio
            portfolio_type = portfolio_details.get('portfolio_type', 'single')
            
            if portfolio_type == 'multi':
                await self.show_multi_portfolio_update_dialog(portfolio_id, portfolio_details, fetch_api, refresh_callback)
            else:
                await self.show_single_portfolio_update_dialog(portfolio_id, portfolio_details, fetch_api, refresh_callback)
        except Exception as e:
            ui.notify(f"❌ Error loading portfolio details: {str(e)}", type="negative")

    async def show_single_portfolio_update_dialog(self, portfolio_id, portfolio_details, fetch_api, refresh_callback):
        """Show dialog to update single portfolio parameters"""
        try:
            with ui.dialog() as dialog, ui.card().classes("w-[800px] max-h-[90vh] overflow-y-auto"):
                ui.label("✏️ Update Single Portfolio").classes("text-xl font-bold mb-4")
                
                # Portfolio name
                portfolio_name = ui.input(
                    label="Portfolio Name",
                    value=portfolio_details.get('portfolio_name', ''),
                    placeholder="Enter portfolio name"
                ).classes("w-full mb-4").props("outlined dense")

                # Status selection
                current_status = portfolio_details.get('status', 'active')
                status_options = ['active', 'paused', 'cancelled']
                status_select = ui.select(
                    label="Status",
                    options=status_options,
                    value=current_status
                ).classes("w-full mb-4").props("outlined dense")

                # Configuration section
                ui.label("⚙️ Configuration Parameters").classes("text-lg font-bold mt-6 mb-4")
                
                config = portfolio_details.get('config', {})
                
                # Basic parameters
                with ui.row().classes("w-full gap-4"):
                    fixed_investment = ui.number(
                        label="💰 Fixed Investment (₹)",
                        value=config.get('fixed_investment', 5000),
                        min=1000, step=1000
                    ).classes("flex-1").props("outlined dense")
                    
                    max_monthly = ui.number(
                        label="📅 Max Monthly Amount (₹)",
                        value=config.get('max_amount_in_a_month', 20000),
                        min=1000, step=1000
                    ).classes("flex-1").props("outlined dense")

                # Advanced Configuration Section
                with ui.expansion("⚙️ Advanced Configuration").classes("w-full mt-4"):
                    with ui.column().classes("gap-4"):
                        # Drawdown Thresholds
                        with ui.row().classes("w-full gap-4"):
                            minor_dd = ui.number(
                                label="📉 Minor Drawdown (%)",
                                value=config.get('minor_drawdown_threshold', -4.0),
                                min=-20, max=0, step=0.5
                            ).classes("flex-1").props("outlined dense")
                            
                            major_dd = ui.number(
                                label="📉 Major Drawdown (%)",
                                value=config.get('major_drawdown_threshold', -10.0),
                                min=-30, max=0, step=0.5
                            ).classes("flex-1").props("outlined dense")
                            
                            extreme_dd = ui.number(
                                label="📉 Extreme Drawdown (%)",
                                value=config.get('extreme_drawdown_threshold', -15.0),
                                min=-50, max=0, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        # Investment Multipliers
                        with ui.row().classes("w-full gap-4"):
                            minor_mult = ui.number(
                                label="📈 Minor Multiplier",
                                value=config.get('minor_drawdown_inv_multiplier', 1.75),
                                min=1.0, max=5.0, step=0.25
                            ).classes("flex-1").props("outlined dense")
                            
                            major_mult = ui.number(
                                label="📈 Major Multiplier",
                                value=config.get('major_drawdown_inv_multiplier', 3.0),
                                min=1.0, max=10.0, step=0.5
                            ).classes("flex-1").props("outlined dense")
                            
                            extreme_mult = ui.number(
                                label="📈 Extreme Multiplier",
                                value=config.get('extreme_drawdown_inv_multiplier', 4.0),
                                min=1.0, max=15.0, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        # Technical Parameters
                        with ui.row().classes("w-full gap-4"):
                            rolling_window = ui.number(
                                label="📊 Rolling Window",
                                value=config.get('rolling_window', 100),
                                min=20, max=500, step=10
                            ).classes("flex-1").props("outlined dense")
                            
                            price_threshold = ui.number(
                                label="💰 Price Reduction Threshold (%)",
                                value=config.get('price_reduction_threshold', 4.0),
                                min=1.0, max=20.0, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        # Additional parameters
                        with ui.row().classes("w-full gap-4"):
                            fallback_day = ui.number(
                                label="📅 Fallback Day",
                                value=config.get('fallback_day', 28),
                                min=1, max=31, step=1
                            ).classes("flex-1").props("outlined dense")
                            
                            min_gap = ui.number(
                                label="⏱️ Min Investment Gap (days)",
                                value=config.get('min_investment_gap_days', 5),
                                min=1, max=30, step=1
                            ).classes("flex-1").props("outlined dense")

                # Action buttons
                with ui.row().classes("gap-2 mt-6"):
                    ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white")

                    async def confirm_update():
                        try:
                            # Prepare update data
                            update_data = {
                                "portfolio_name": portfolio_name.value,
                                "status": status_select.value,
                                "config": {
                                    "fixed_investment": fixed_investment.value,
                                    "major_drawdown_threshold": major_dd.value,
                                    "minor_drawdown_threshold": minor_dd.value,
                                    "extreme_drawdown_threshold": extreme_dd.value,
                                    "minor_drawdown_inv_multiplier": minor_mult.value,
                                    "major_drawdown_inv_multiplier": major_mult.value,
                                    "extreme_drawdown_inv_multiplier": extreme_mult.value,
                                    "rolling_window": int(rolling_window.value),
                                    "fallback_day": int(fallback_day.value),
                                    "min_investment_gap_days": int(min_gap.value),
                                    "max_amount_in_a_month": max_monthly.value,
                                    "price_reduction_threshold": price_threshold.value,
                                    "force_remaining_investment": True
                                }
                            }

                            response = await self.safe_api_call(
                                fetch_api, 
                                f"/sip/portfolio/{portfolio_id}", 
                                method="PUT", 
                                data=update_data
                            )
                            
                            if response:
                                ui.notify("✅ Portfolio updated successfully", type="positive")
                                await refresh_callback()
                                dialog.close()
                            else:
                                ui.notify("❌ Failed to update portfolio", type="negative")
                                
                        except Exception as e:
                            ui.notify(f"❌ Error updating portfolio: {str(e)}", type="negative")

                    ui.button("💾 Save Changes", on_click=confirm_update).classes("bg-blue-500 text-white")

            dialog.open()

        except Exception as e:
            ui.notify(f"❌ Error loading portfolio details: {str(e)}", type="negative")

    async def show_multi_portfolio_update_dialog(self, portfolio_id, portfolio_details, fetch_api, refresh_callback):
        """Show dialog to update multi-portfolio parameters"""
        try:
            with ui.dialog() as dialog, ui.card().classes("w-[900px] max-h-[90vh] overflow-y-auto"):
                ui.label("✏️ Update Multi-Portfolio").classes("text-xl font-bold mb-4")
                
                # Portfolio name
                portfolio_name = ui.input(
                    label="Portfolio Name",
                    value=portfolio_details.get('portfolio_name', ''),
                    placeholder="Enter portfolio name"
                ).classes("w-full mb-4").props("outlined dense")

                # Status selection
                current_status = portfolio_details.get('status', 'active')
                status_options = ['active', 'paused', 'cancelled']
                status_select = ui.select(
                    label="Status",
                    options=status_options,
                    value=current_status
                ).classes("w-full mb-4").props("outlined dense")

                # Symbols section
                ui.label("📈 Symbols & Allocation").classes("text-lg font-bold mt-6 mb-4")
                
                symbols = portfolio_details.get('symbols', [])
                symbols_data = []
                
                # Create symbols container
                symbols_container = ui.column().classes("gap-4")
                
                def add_symbol_row():
                    with symbols_container:
                        with ui.row().classes("w-full gap-4 items-center"):
                            symbol_input = ui.input(
                                label="Symbol",
                                placeholder="ICICIB22"
                            ).classes("flex-1").props("outlined dense")
                            
                            allocation_input = ui.number(
                                label="Allocation %",
                                value=25.0,
                                min=0.1, max=100, step=0.1
                            ).classes("w-32").props("outlined dense")

                            def remove_symbol():
                                parent_card = symbol_input.parent_slot.parent
                                symbols_container.remove(parent_card)
                                symbols_data[:] = [s for s in symbols_data if s != (symbol_input, allocation_input)]

                            ui.button("🗑️", on_click=remove_symbol).classes("bg-red-500 text-white")

                        symbols_data.append((symbol_input, allocation_input))

                # Add existing symbols
                for symbol_data in symbols:
                    symbol_name = symbol_data.get('symbol', '')
                    allocation = symbol_data.get('allocation_percentage', 25.0)
                    
                    with symbols_container:
                        with ui.row().classes("w-full gap-4 items-center"):
                            symbol_input = ui.input(
                                label="Symbol",
                                value=symbol_name,
                                placeholder="ICICIB22"
                            ).classes("flex-1").props("outlined dense")
                            
                            allocation_input = ui.number(
                                label="Allocation %",
                                value=allocation,
                                min=0.1, max=100, step=0.1
                            ).classes("w-32").props("outlined dense")

                            def remove_symbol():
                                parent_card = symbol_input.parent_slot.parent
                                symbols_container.remove(parent_card)
                                symbols_data[:] = [s for s in symbols_data if s != (symbol_input, allocation_input)]

                            ui.button("🗑️", on_click=remove_symbol).classes("bg-red-500 text-white")

                        symbols_data.append((symbol_input, allocation_input))

                # Add symbol button
                with ui.row().classes("w-full justify-between items-center mb-4"):
                    ui.label("Symbols & Allocation").classes("font-medium")
                    ui.button("➕ Add Symbol", on_click=add_symbol_row).classes("bg-green-500 text-white")

                # Portfolio settings
                ui.label("⚙️ Portfolio Settings").classes("text-lg font-bold mt-6 mb-4")
                
                config = portfolio_details.get('config', {})
                
                with ui.row().classes("w-full gap-4"):
                    auto_rebalance = ui.checkbox(
                        "Auto Rebalance",
                        value=portfolio_details.get('auto_rebalance', False)
                    )

                    rebalance_frequency = ui.number(
                        label="Rebalance Frequency (days)",
                        value=portfolio_details.get('rebalance_frequency_days', 30),
                        min=7, max=365
                    ).classes("flex-1").props("outlined dense")

                # Default configuration
                ui.label("⚙️ Default Strategy Config").classes("text-lg font-bold mt-6 mb-4")
                
                with ui.row().classes("w-full gap-4"):
                    with ui.column().classes("flex-1"):
                        default_investment = ui.number(
                            label="Monthly Investment (₹)",
                            value=config.get('fixed_investment', 10000),
                            min=1000, step=1000
                        ).classes("w-full").props("outlined dense")

                        default_major_dd = ui.number(
                            label="Major Drawdown (%)",
                            value=config.get('major_drawdown_threshold', -10.0),
                            min=-50, max=0
                        ).classes("w-full").props("outlined dense")

                        default_minor_dd = ui.number(
                            label="Minor Drawdown (%)",
                            value=config.get('minor_drawdown_threshold', -4.0),
                            min=-20, max=0
                        ).classes("w-full").props("outlined dense")

                        default_extreme_dd = ui.number(
                            label="Extreme Drawdown (%)",
                            value=config.get('extreme_drawdown_threshold', -15.0),
                            min=-50, max=0
                        ).classes("w-full").props("outlined dense")

                    with ui.column().classes("flex-1"):
                        max_monthly = ui.number(
                            label="Max Monthly Amount (₹)",
                            value=config.get('max_amount_in_a_month', 40000),
                            min=1000, step=1000
                        ).classes("w-full").props("outlined dense")

                        force_invest = ui.checkbox(
                            "Force Remaining Investment",
                            value=config.get('force_remaining_investment', True)
                        )

                # Advanced Configuration Section
                with ui.expansion("⚙️ Advanced Configuration").classes("w-full mt-4"):
                    with ui.column().classes("gap-4"):
                        # Investment Multipliers
                        with ui.row().classes("w-full gap-4"):
                            default_minor_mult = ui.number(
                                label="📈 Minor Multiplier",
                                value=config.get('minor_drawdown_inv_multiplier', 1.75),
                                min=1.0, max=5.0, step=0.25
                            ).classes("flex-1").props("outlined dense")
                            
                            default_major_mult = ui.number(
                                label="📈 Major Multiplier",
                                value=config.get('major_drawdown_inv_multiplier', 3.0),
                                min=1.0, max=10.0, step=0.5
                            ).classes("flex-1").props("outlined dense")
                            
                            default_extreme_mult = ui.number(
                                label="📈 Extreme Multiplier",
                                value=config.get('extreme_drawdown_inv_multiplier', 4.0),
                                min=1.0, max=15.0, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        # Technical Parameters
                        with ui.row().classes("w-full gap-4"):
                            default_rolling_window = ui.number(
                                label="📊 Rolling Window",
                                value=config.get('rolling_window', 100),
                                min=20, max=500, step=10
                            ).classes("flex-1").props("outlined dense")
                            
                            default_price_threshold = ui.number(
                                label="💰 Price Reduction Threshold (%)",
                                value=config.get('price_reduction_threshold', 4.0),
                                min=1.0, max=20.0, step=0.5
                            ).classes("flex-1").props("outlined dense")

                        # Additional parameters
                        with ui.row().classes("w-full gap-4"):
                            default_fallback_day = ui.number(
                                label="📅 Fallback Day",
                                value=config.get('fallback_day', 22),
                                min=1, max=31, step=1
                            ).classes("flex-1").props("outlined dense")
                            
                            default_min_gap = ui.number(
                                label="⏱️ Min Investment Gap (days)",
                                value=config.get('min_investment_gap_days', 5),
                                min=1, max=30, step=1
                            ).classes("flex-1").props("outlined dense")

                # Action buttons
                with ui.row().classes("gap-2 mt-6"):
                    ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white")

                    async def confirm_update():
                        try:
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

                            # Prepare update data
                            update_data = {
                                "portfolio_name": portfolio_name.value,
                                "status": status_select.value,
                                "symbols": portfolio_symbols,
                                "default_config": {
                                    "fixed_investment": default_investment.value,
                                    "major_drawdown_threshold": default_major_dd.value,
                                    "minor_drawdown_threshold": default_minor_dd.value,
                                    "extreme_drawdown_threshold": default_extreme_dd.value,
                                    "minor_drawdown_inv_multiplier": default_minor_mult.value,
                                    "major_drawdown_inv_multiplier": default_major_mult.value,
                                    "extreme_drawdown_inv_multiplier": default_extreme_mult.value,
                                    "rolling_window": int(default_rolling_window.value),
                                    "fallback_day": int(default_fallback_day.value),
                                    "min_investment_gap_days": int(default_min_gap.value),
                                    "max_amount_in_a_month": max_monthly.value,
                                    "price_reduction_threshold": default_price_threshold.value,
                                    "force_remaining_investment": force_invest.value
                                },
                                "auto_rebalance": auto_rebalance.value,
                                "rebalance_frequency_days": rebalance_frequency.value
                            }

                            response = await self.safe_api_call(
                                fetch_api, 
                                f"/sip/portfolio/{portfolio_id}/multi", 
                                method="PUT", 
                                data=update_data
                            )
                            
                            if response:
                                ui.notify("✅ Multi-portfolio updated successfully", type="positive")
                                await refresh_callback()
                                dialog.close()
                            else:
                                ui.notify("❌ Failed to update multi-portfolio", type="negative")
                                
                        except Exception as e:
                            ui.notify(f"❌ Error updating multi-portfolio: {str(e)}", type="negative")

                    ui.button("💾 Save Changes", on_click=confirm_update).classes("bg-blue-500 text-white")

            dialog.open()

        except Exception as e:
            ui.notify(f"❌ Error loading multi-portfolio details: {str(e)}", type="negative")

    async def pause_portfolio(self, portfolio_id, fetch_api, refresh_callback):
        """Pause SIP portfolio (temporarily stop automated investments)"""
        try:
            with ui.dialog() as dialog, ui.card():
                ui.label("⏸️ Pause SIP Portfolio").classes("text-lg font-bold mb-4")
                ui.label("This will temporarily stop automated investments but keep the portfolio active.").classes(
                    "text-gray-600 mb-4")
                ui.label("You can resume automated investments later.").classes("text-blue-600 mb-4")

                with ui.row().classes("gap-2"):
                    ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white")

                    async def confirm_pause():
                        response = await self.safe_api_call(fetch_api, f"/sip/portfolio/{portfolio_id}/pause", method="PUT")
                        if response:
                            ui.notify("✅ Portfolio paused successfully", type="positive")
                            await refresh_callback()
                        else:
                            ui.notify("❌ Failed to pause portfolio", type="negative")
                        dialog.close()

                    ui.button("⏸️ Confirm Pause", on_click=confirm_pause).classes("bg-yellow-500 text-white")

            dialog.open()

        except Exception as e:
            ui.notify(f"❌ Error pausing portfolio: {str(e)}", type="negative")

    async def resume_portfolio(self, portfolio_id, fetch_api, refresh_callback):
        """Resume SIP portfolio (restart automated investments)"""
        try:
            with ui.dialog() as dialog, ui.card():
                ui.label("▶️ Resume SIP Portfolio").classes("text-lg font-bold mb-4")
                ui.label("This will restart automated investments for this portfolio.").classes(
                    "text-gray-600 mb-4")
                ui.label("The portfolio will begin processing signals again.").classes("text-blue-600 mb-4")

                with ui.row().classes("gap-2"):
                    ui.button("Cancel", on_click=dialog.close).classes("bg-gray-500 text-white")

                    async def confirm_resume():
                        response = await self.safe_api_call(fetch_api, f"/sip/portfolio/{portfolio_id}/resume", method="PUT")
                        if response:
                            ui.notify("✅ Portfolio resumed successfully", type="positive")
                            await refresh_callback()
                        else:
                            ui.notify("❌ Failed to resume portfolio", type="negative")
                        dialog.close()

                    ui.button("▶️ Confirm Resume", on_click=confirm_resume).classes("bg-green-500 text-white")

            dialog.open()

        except Exception as e:
            ui.notify(f"❌ Error resuming portfolio: {str(e)}", type="negative")

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

                # Order type selection
                ui.label("📋 Order Type").classes("font-bold mt-4 mb-2")
                
                order_type_select = ui.select(
                    label="Select Order Type",
                    options=[
                        "🔄 Market Order",
                        "🎯 Limit Order", 
                        "⏰ GTT Order"
                    ],
                    value="🔄 Market Order"
                ).classes("w-full")

                # Create a container for order type specific inputs (after the select)
                order_inputs_container = ui.column().classes("w-full gap-4")

                # Store references to dynamically created inputs
                limit_price_input = None
                gtt_trigger_input = None
                gtt_limit_input = None
                
                # Show/hide inputs based on order type
                def on_order_type_change():
                    nonlocal limit_price_input, gtt_trigger_input, gtt_limit_input
                    try:
                        selected_text = order_type_select.value
                        
                        # Clear the container first
                        order_inputs_container.clear()
                        
                        # Reset references
                        limit_price_input = None
                        gtt_trigger_input = None
                        gtt_limit_input = None
                        
                        # Market Order: No price inputs needed
                        if "Market Order" in selected_text:
                            # No inputs to show for market orders
                            pass
                            
                        # Limit Order: Show only limit price
                        elif "Limit Order" in selected_text:
                            with order_inputs_container:
                                limit_price_input = ui.number(
                                    label="Limit Price (₹)",
                                    value=0,
                                    min=0,
                                    step=0.01
                                ).classes("w-full")
                                
                        # GTT Order: Show GTT trigger and limit prices
                        elif "GTT Order" in selected_text:
                            with order_inputs_container:
                                gtt_trigger_input = ui.number(
                                    label="GTT Trigger Price (₹)",
                                    value=0,
                                    min=0,
                                    step=0.01
                                ).classes("w-full")
                                
                                gtt_limit_input = ui.number(
                                    label="GTT Limit Price (₹)",
                                    value=0,
                                    min=0,
                                    step=0.01
                                ).classes("w-full")
                        else:
                            logger.warning(f"Unknown order type: '{selected_text}'")
                    except Exception as e:
                        logger.error(f"Error in on_order_type_change: {e}")

                # Bind the change event
                order_type_select.on_value_change(on_order_type_change)
                
                # Set initial visibility based on default selection
                on_order_type_change()

                ui.label("⚠️ Note: System will check minimum 5-day gap and monthly limits").classes(
                    "text-xs text-yellow-600 mt-2")
                ui.label("💡 Market orders only work during market hours (9:15 AM - 3:30 PM)").classes(
                    "text-xs text-blue-600")
                ui.label("📋 Market: Immediate execution, Limit: At specified price, GTT: Good Till Triggered").classes(
                    "text-xs text-purple-600")
                ui.label("⏰ GTT orders are recommended when markets are closed").classes(
                    "text-xs text-green-600")

                async def execute_investment():
                    try:
                        # Map display text to order type values
                        selected_text = order_type_select.value
                        if "Market Order" in selected_text:
                            order_type = "MARKET"
                        elif "Limit Order" in selected_text:
                            order_type = "LIMIT"
                        elif "GTT Order" in selected_text:
                            order_type = "GTT"
                        else:
                            ui.notify("❌ Invalid order type selected", type="negative")
                            return
                        
                        # Prepare execution request based on order type
                        execution_data = {
                            "amount": amount_input.value,
                            "order_type": order_type
                        }
                        
                        # Add order type specific parameters
                        if order_type == "LIMIT":
                            # Use direct reference to limit price input
                            if not limit_price_input or not limit_price_input.value:
                                ui.notify("❌ Limit price is required for LIMIT orders", type="negative")
                                return
                            execution_data["limit_price"] = limit_price_input.value
                        elif order_type == "GTT":
                            # Use direct references to GTT inputs
                            if not gtt_trigger_input or not gtt_limit_input or not gtt_trigger_input.value or not gtt_limit_input.value:
                                ui.notify("❌ GTT trigger price and limit price are required for GTT orders", type="negative")
                                return
                            execution_data["gtt_trigger_price"] = gtt_trigger_input.value
                            execution_data["gtt_limit_price"] = gtt_limit_input.value

                        response = await self.safe_api_call(
                            fetch_api,
                            f"/sip/execute/{portfolio_id}",
                            method="POST",
                            data=execution_data
                        )

                        if response:
                            order_type_used = response.get('order_type_used', 'MARKET')
                            executed_trades = response.get('executed_trades', [])
                            failed_trades = response.get('failed_trades', [])
                            
                            # Build notification message
                            message_parts = []
                            
                            if executed_trades:
                                total_invested = response.get('total_investment_amount', 0)
                                message_parts.append(f"✅ ₹{total_invested:,.0f} invested using {order_type_used} orders")
                                message_parts.append(f"Successfully placed {len(executed_trades)} orders")
                            
                            if failed_trades:
                                message_parts.append(f"❌ Failed to place {len(failed_trades)} orders")
                                # Show details of failed trades with error messages
                                for trade in failed_trades:
                                    symbol = trade.get('symbol', 'Unknown')
                                    error = trade.get('error', 'Unknown error')
                                    message_parts.append(f"• {symbol}: {error}")
                            
                            if not executed_trades and failed_trades:
                                # All orders failed
                                error_message = " ".join(message_parts)
                                
                                # Add helpful suggestion for market closed error
                                if "Markets are closed" in error_message:
                                    error_message += "\n💡 Tip: Use GTT orders when markets are closed for automatic execution when markets open."
                                
                                ui.notify(error_message, type="negative")
                            elif executed_trades and failed_trades:
                                # Mixed success/failure
                                ui.notify(" ".join(message_parts), type="warning")
                            else:
                                # All orders succeeded
                                ui.notify(" ".join(message_parts), type="positive")
                            
                            await refresh_callback()
                            dialog.close()
                        else:
                            # safe_api_call already handled the error display
                            pass
                    except Exception as e:
                        # Handle any unexpected errors not caught by safe_api_call
                        ui.notify(f"❌ Unexpected error: {str(e)}", type="negative")

                with ui.row().classes("gap-4 mt-6 justify-center"):
                    ui.button("Cancel", on_click=dialog.close).classes("btn-modern-danger").props("outline")
                    ui.button("💰 Invest Now", icon="trending_up", on_click=execute_investment).classes(
                        "btn-modern-success px-8 py-3")

            dialog.open()

        except Exception as e:
            ui.notify(f"❌ Error showing investment dialog: {str(e)}", type="negative")

    async def render_enhanced_signals_panel(self, fetch_api, user_storage):
        """Enhanced signals panel with modern styling and real-time updates"""

        # Header
        with ui.card().classes("w-full mb-4"):
            with ui.row().classes("w-full items-center p-4"):
                ui.icon("notifications_active", size="2rem").classes("text-orange-400")
                with ui.column().classes("ml-4 gap-2"):
                    ui.label("📡 Investment Signals Dashboard").classes("text-xl font-bold")
                    ui.label("Monitor investment opportunities across all your portfolios").classes("text-sm text-slate-300")
        
        # Control panel for signal refresh
        with ui.card().classes("w-full mb-4 p-4"):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("refresh", size="1.2rem").classes("text-cyan-400")
                    ui.label("Signal Controls").classes("font-bold text-lg")
                
                ui.button("Refresh Signals", icon="refresh", on_click=lambda: refresh_signals()).classes(
                    "bg-blue-500 text-white px-4 py-2 rounded font-semibold")
        
        signals_container = ui.column().classes("w-full gap-4")

        async def refresh_signals():
            """Refresh all signals with enhanced display including GTT"""
            try:
                signals = await self.safe_api_call(fetch_api, "/sip/signals")

                signals_container.clear()

                if signals:
                    with signals_container:
                        with ui.card().classes("w-full p-4 mb-4"):
                            with ui.row().classes("w-full items-center gap-3"):
                                ui.icon("signal_cellular_alt", size="1.5rem").classes("text-green-400")
                                ui.label(f"📊 Found {len(signals['signals'])} Active Signals").classes("text-lg font-bold")
                                ui.separator().classes("flex-1 ml-4")
                                ui.chip(f"{len(signals['signals'])}", color="positive").classes("text-sm px-3 py-1").props("outline")

                        for signal in signals['signals']:
                            await self.render_signal_card(signal)
                else:
                    with signals_container:
                        with ui.card().classes("w-full p-8 text-center"):
                            ui.icon("search_off", size="2rem").classes("text-slate-400 mb-3")
                            ui.label("🔍 No active signals found").classes("text-lg font-bold text-slate-300 mb-2")
                            ui.label("Check back later or trigger manual scan").classes("text-sm text-slate-400")

            except Exception as e:
                self.show_error("Failed to refresh signals", str(e))

        # Remove duplicate button as it's already in the control panel above

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

        with ui.card().classes("w-full mb-3 p-3"):
            with ui.row().classes("w-full justify-between items-center"):
                ui.label(f"📡 {symbol} Signal").classes("text-base font-bold")
                ui.label(signal_type).classes("text-xs font-medium px-2 py-1 rounded bg-gray-100")

            ui.label(f"💰 Recommended: ₹{recommended_amount:,.0f}").classes("text-sm")
            ui.label(f"📈 Price: ₹{current_price:.2f}").classes("text-sm")
            ui.label(f"🔽 Drawdown: {drawdown:.2f}%").classes("text-sm text-red-600")

            # GTT status badge
            gtt_color = "bg-green-200 text-green-800" if gtt_status == 'ACTIVE' else "bg-red-200 text-red-800"
            ui.label(gtt_status).classes(f"text-xs px-2 py-1 rounded {gtt_color} mt-2")

            ui.label(f"Created: {signal.get('created_at', '')}").classes("text-xs text-gray-500 mt-2")

    async def render_batch_multi_configs_section(self, fetch_api, user_storage):
        """Enhanced batch multi-configs backtest UI aligned with endpoint response"""

        ui.label("🔄 Batch Backtest with Multiple Configurations").classes("text-lg font-bold mb-4")
        ui.label("Test multiple strategy configurations in batch and find the best one").classes("text-gray-600 mb-4")

        symbols_input = ui.textarea(
            label="Symbols (one per line)",
            value="ICICIB22\nCPSEETF",
            placeholder="Enter symbols separated by new lines"
        ).classes("w-full mb-4")

        with ui.row().classes("gap-4 mb-4"):
            start_date = ui.input("Start Date", value="2020-01-01").props("dense type=date").classes("flex-1 min-w-0")
            end_date = ui.input("End Date", value=datetime.now().strftime("%Y-%m-%d")).props(
                "dense type=date").classes("flex-1 min-w-0")

            # FIXED: Use individual input fields instead of editable table for better UX
            ui.label("⚙️ Configuration Management").classes("text-lg font-bold mb-2")
            ui.label("Add and customize multiple configurations to test").classes("text-gray-600 mb-4")

            # Configuration storage
            configs_data = []
            configs_container = ui.column().classes("w-full mb-4")

            def create_config_card(config_data: dict, config_index: int):
                """Create an editable configuration card"""

                with ui.card().classes("w-full mb-4 p-4 border"):
                    with ui.row().classes("w-full justify-between items-center mb-4"):
                        ui.label(f"Configuration {config_index + 1}").classes("text-lg font-bold")

                        def remove_config():
                            if len(configs_data) > 1:  # Keep at least one config
                                configs_data.pop(config_index)
                                refresh_configs_display()
                                ui.notify(f"Configuration {config_index + 1} removed", type="positive")
                            else:
                                ui.notify("Keep at least one configuration", type="warning")

                        if len(configs_data) > 1:
                            ui.button("🗑️ Remove", on_click=remove_config).classes("bg-red-500 text-white text-xs")

                    # Investment Settings
                    with ui.expansion("💰 Investment Settings", value=True).classes("w-full mb-2"):
                        with ui.row().classes("gap-4"):
                            fixed_investment = ui.number(
                                label="Monthly Investment (₹)",
                                value=config_data.get('fixed_investment', 5000),
                                step=100,
                                min=100
                            ).classes("w-48")

                            max_amount = ui.number(
                                label="Max Monthly Amount (₹)",
                                value=config_data.get('max_amount_in_a_month', 18000),
                                step=1000,
                                min=1000
                            ).classes("w-48")

                            force_remaining = ui.checkbox(
                                text="Force Remaining Investment",
                                value=config_data.get('force_remaining_investment', True)
                            )

                    # Drawdown Thresholds
                    with ui.expansion("📉 Drawdown Thresholds").classes("w-full mb-2"):
                        with ui.row().classes("gap-4"):
                            minor_dd = ui.number(
                                label="Minor Drawdown (%)",
                                value=config_data.get('minor_drawdown_threshold', -4),
                                step=0.5,
                                max=-1
                            ).classes("w-48")

                            major_dd = ui.number(
                                label="Major Drawdown (%)",
                                value=config_data.get('major_drawdown_threshold', -10),
                                step=0.5,
                                max=-5
                            ).classes("w-48")

                            extreme_dd = ui.number(
                                label="Extreme Drawdown (%)",
                                value=config_data.get('extreme_drawdown_threshold', -15),
                                step=0.5,
                                max=-10
                            ).classes("w-48")

                    # Investment Multipliers
                    with ui.expansion("📈 Investment Multipliers").classes("w-full mb-2"):
                        with ui.row().classes("gap-4"):
                            minor_mult = ui.number(
                                label="Minor Multiplier",
                                value=config_data.get('minor_drawdown_inv_multiplier', 1.75),
                                step=0.25,
                                min=1.0,
                                max=5.0
                            ).classes("w-48")

                            major_mult = ui.number(
                                label="Major Multiplier",
                                value=config_data.get('major_drawdown_inv_multiplier', 3.0),
                                step=0.25,
                                min=1.0,
                                max=5.0
                            ).classes("w-48")

                            extreme_mult = ui.number(
                                label="Extreme Multiplier",
                                value=config_data.get('extreme_drawdown_inv_multiplier', 4.0),
                                step=0.25,
                                min=1.0,
                                max=10.0
                            ).classes("w-48")

                    # Advanced Settings
                    with ui.expansion("🔧 Advanced Settings").classes("w-full mb-2"):
                        with ui.row().classes("gap-4"):
                            rolling_window = ui.number(
                                label="Rolling Window (days)",
                                value=config_data.get('rolling_window', 100),
                                step=10,
                                min=20,
                                max=200
                            ).classes("w-48")

                            fallback_day = ui.number(
                                label="Fallback Day",
                                value=config_data.get('fallback_day', 28),
                                step=1,
                                min=1,
                                max=31
                            ).classes("w-48")

                        with ui.row().classes("gap-4"):
                            min_gap_days = ui.number(
                                label="Min Investment Gap (days)",
                                value=config_data.get('min_investment_gap_days', 5),
                                step=1,
                                min=1,
                                max=30
                            ).classes("w-48")

                            price_threshold = ui.number(
                                label="Price Reduction Threshold (%)",
                                value=config_data.get('price_reduction_threshold', 5),
                                step=0.5,
                                min=0,
                                max=20
                            ).classes("w-48")

                    # Update config data when values change
                    def update_config():
                        configs_data[config_index] = {
                            'fixed_investment': fixed_investment.value,
                            'max_amount_in_a_month': max_amount.value,
                            'force_remaining_investment': force_remaining.value,
                            'minor_drawdown_threshold': minor_dd.value,
                            'major_drawdown_threshold': major_dd.value,
                            'extreme_drawdown_threshold': extreme_dd.value,
                            'minor_drawdown_inv_multiplier': minor_mult.value,
                            'major_drawdown_inv_multiplier': major_mult.value,
                            'extreme_drawdown_inv_multiplier': extreme_mult.value,
                            'rolling_window': int(rolling_window.value),
                            'fallback_day': int(fallback_day.value),
                            'min_investment_gap_days': int(min_gap_days.value),
                            'price_reduction_threshold': price_threshold.value
                        }

                    # Bind update function to all inputs
                    for input_field in [fixed_investment, max_amount, force_remaining, minor_dd, major_dd, extreme_dd,
                                        minor_mult, major_mult, extreme_mult, rolling_window, fallback_day,
                                        min_gap_days, price_threshold]:
                        input_field.on('update:model-value', lambda: update_config())

            def refresh_configs_display():
                """Refresh the configs display"""
                configs_container.clear()
                with configs_container:
                    for idx, config in enumerate(configs_data):
                        create_config_card(config, idx)

            # Initialize with default configurations
            default_configs = [
                {
                    'fixed_investment': 5000,
                    'max_amount_in_a_month': 18000,
                    'force_remaining_investment': True,
                    'minor_drawdown_threshold': -4.0,
                    'major_drawdown_threshold': -10.0,
                    'extreme_drawdown_threshold': -15.0,
                    'minor_drawdown_inv_multiplier': 1.75,
                    'major_drawdown_inv_multiplier': 3.0,
                    'extreme_drawdown_inv_multiplier': 4.0,
                    'rolling_window': 100,
                    'fallback_day': 28,
                    'min_investment_gap_days': 5,
                    'price_reduction_threshold': 5.0
                },
                {
                    'fixed_investment': 3000,
                    'max_amount_in_a_month': 20000,
                    'force_remaining_investment': True,
                    'minor_drawdown_threshold': -5.0,
                    'major_drawdown_threshold': -10.0,
                    'extreme_drawdown_threshold': -15.0,
                    'minor_drawdown_inv_multiplier': 2.0,
                    'major_drawdown_inv_multiplier': 3.0,
                    'extreme_drawdown_inv_multiplier': 4.0,
                    'rolling_window': 100,
                    'fallback_day': 28,
                    'min_investment_gap_days': 5,
                    'price_reduction_threshold': 4.0
                }
            ]

            configs_data.extend(default_configs)

            def add_new_config():
                """Add a new configuration"""
                new_config = {
                    'fixed_investment': 5000,
                    'max_amount_in_a_month': 18000,
                    'force_remaining_investment': True,
                    'minor_drawdown_threshold': -4.0,
                    'major_drawdown_threshold': -10.0,
                    'extreme_drawdown_threshold': -15.0,
                    'minor_drawdown_inv_multiplier': 1.75,
                    'major_drawdown_inv_multiplier': 3.0,
                    'extreme_drawdown_inv_multiplier': 4.0,
                    'rolling_window': 100,
                    'fallback_day': 28,
                    'min_investment_gap_days': 5,
                    'price_reduction_threshold': 5.0
                }
                configs_data.append(new_config)
                refresh_configs_display()
                ui.notify(f"Configuration {len(configs_data)} added", type="positive")

            def add_preset_config(preset_name: str):
                """Add preset configuration"""
                presets = {
                    'Conservative': {
                        'fixed_investment': 3000,
                        'max_amount_in_a_month': 15000,
                        'minor_drawdown_threshold': -3.0,
                        'major_drawdown_threshold': -8.0,
                        'extreme_drawdown_threshold': -12.0,
                        'minor_drawdown_inv_multiplier': 1.5,
                        'major_drawdown_inv_multiplier': 2.5,
                        'extreme_drawdown_inv_multiplier': 3.5,
                        'price_reduction_threshold': 3.0
                    },
                    'Aggressive': {
                        'fixed_investment': 7000,
                        'max_amount_in_a_month': 25000,
                        'minor_drawdown_threshold': -5.0,
                        'major_drawdown_threshold': -12.0,
                        'extreme_drawdown_threshold': -18.0,
                        'minor_drawdown_inv_multiplier': 2.0,
                        'major_drawdown_inv_multiplier': 3.5,
                        'extreme_drawdown_inv_multiplier': 5.0,
                        'price_reduction_threshold': 6.0
                    }
                }

                if preset_name in presets:
                    base_config = {
                        'fixed_investment': 5000,
                        'max_amount_in_a_month': 18000,
                        'force_remaining_investment': True,
                        'rolling_window': 100,
                        'fallback_day': 28,
                        'min_investment_gap_days': 5
                    }
                    base_config.update(presets[preset_name])
                    configs_data.append(base_config)
                    refresh_configs_display()
                    ui.notify(f"{preset_name} configuration added", type="positive")

            # Configuration management buttons
            with ui.row().classes("gap-2 mb-4"):
                ui.button("➕ Add Configuration", on_click=add_new_config).classes("bg-green-500 text-white")
                ui.button("🛡️ Add Conservative", on_click=lambda: add_preset_config('Conservative')).classes(
                    "bg-blue-500 text-white")
                ui.button("🚀 Add Aggressive", on_click=lambda: add_preset_config('Aggressive')).classes(
                    "bg-red-500 text-white")

            # Display configurations
            refresh_configs_display()

        # Results Section
        batch_results = ui.column().classes("w-full mt-4")

        async def run_batch_multi():
            try:
                symbols = [s.strip() for s in symbols_input.value.split('\n') if s.strip()]
                if not symbols:
                    ui.notify("Please enter at least one symbol", type="warning")
                    return

                if not configs_data:
                    ui.notify("Please add at least one configuration", type="warning")
                    return

                request_data = {
                    "symbols": symbols,
                    "configs": configs_data
                }

                ui.notify("Running batch backtest...", type="info")
                url = f"/sip/batch-backtest/multi-configs?start_date={start_date.value}&end_date={end_date.value}"
                results = await self.safe_api_call(fetch_api, url, "POST", request_data)

                if results:
                    await self.display_batch_multi_config_results(results, batch_results)
                else:
                    ui.notify("No results received", type="warning")

            except Exception as e:
                self.show_error("Batch backtest failed", str(e))

        ui.button("🚀 Run Batch Backtest", on_click=run_batch_multi).classes("bg-blue-600 text-white px-6 py-2")

        # Configuration summary
        with ui.card().classes("w-full mt-4 p-4 bg-gray-50"):
            ui.label("📊 Current Configurations Summary").classes("font-bold mb-2")

            def show_config_summary():
                """Show a summary of current configurations"""
                if configs_data:
                    summary_text = f"Total Configurations: {len(configs_data)}\n"
                    for i, config in enumerate(configs_data):
                        summary_text += f"Config {i + 1}: ₹{config['fixed_investment']:,}/month, "
                        summary_text += f"DD: {config['minor_drawdown_threshold']:.1f}%/{config['major_drawdown_threshold']:.1f}%/{config['extreme_drawdown_threshold']:.1f}%, "
                        summary_text += f"Mult: {config['minor_drawdown_inv_multiplier']:.1f}x/{config['major_drawdown_inv_multiplier']:.1f}x/{config['extreme_drawdown_inv_multiplier']:.1f}x\n"

                    ui.label(summary_text.strip()).classes("text-xs text-gray-600 whitespace-pre-line")
                else:
                    ui.label("No configurations added").classes("text-gray-500")

            show_config_summary()

    async def display_batch_multi_config_results(self, results: dict, container):
        """Display comprehensive batch multi-config results aligned with endpoint response"""

        container.clear()

        with container:
            # Header Information
            with ui.card().classes("w-full mb-4 p-4 bg-blue-50"):
                ui.label("📊 Batch Backtest Results").classes("text-xl font-bold mb-2")

                with ui.row().classes("gap-8"):
                    ui.label(f"🆔 Batch ID: {results.get('batch_id', 'N/A')}").classes("text-sm")
                    ui.label(f"📈 Symbols: {', '.join(results.get('symbols', []))}").classes("text-sm")
                    ui.label(f"📅 Period: {results.get('period', 'N/A')}").classes("text-sm")
                    ui.label(f"⚙️ Configs Tested: {results.get('configurations_tested', 0)}").classes("text-sm")

            # Best Configuration Recommendation
            best_config = results.get('best_config_recommendation', {})
            if best_config.get('recommended_config'):
                with ui.card().classes("w-full mb-4 p-4 bg-green-50 border-l-4 border-green-500"):
                    ui.label("🏆 Best Configuration Recommendation").classes("text-lg font-bold mb-2")
                    ui.label(f"📈 Average Outperformance: {best_config.get('average_outperformance', 0):.2f}%").classes(
                        "text-sm font-semibold text-green-700")
                    ui.label(f"💡 Reason: {best_config.get('reason', 'N/A')}").classes("text-sm mb-2")

                    # Display recommended config in a compact format
                    recommended = best_config['recommended_config']
                    with ui.expansion("View Recommended Parameters").classes("w-full"):
                        config_table = ui.table(
                            columns=[
                                {'name': 'parameter', 'label': 'Parameter', 'field': 'parameter'},
                                {'name': 'value', 'label': 'Value', 'field': 'value'}
                            ],
                            rows=[
                                {'parameter': k.replace('_', ' ').title(), 'value': v}
                                for k, v in recommended.items()
                            ]
                        ).classes("w-full")

            # Individual Configuration Results
            batch_results_data = results.get('batch_results', [])

            for config_idx, batch_result in enumerate(batch_results_data):
                with ui.expansion(f"Configuration {config_idx + 1} Results", value=config_idx == 0).classes(
                        "w-full mb-4"):

                    # Configuration Details
                    with ui.card().classes("w-full mb-4 p-4 bg-gray-50"):
                        ui.label("⚙️ Configuration Parameters").classes("font-semibold mb-2")
                        config = batch_result.get('config', {})

                        with ui.grid(columns=4).classes("gap-2"):
                            for key, value in config.items():
                                if isinstance(value, bool):
                                    display_value = "✅" if value else "❌"
                                elif isinstance(value, (int, float)):
                                    display_value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
                                else:
                                    display_value = str(value)

                                ui.label(f"{key.replace('_', ' ').title()}: {display_value}").classes("text-xs")

                    # Summary Statistics
                    summary = batch_result.get('summary', {})
                    benchmark_summary = batch_result.get('benchmark_summary', {})

                    with ui.row().classes("gap-4 mb-4"):
                        # Strategy Summary
                        with ui.card().classes("flex-1 p-4"):
                            ui.label("📊 Strategy Summary").classes("font-semibold mb-2")
                            ui.label(f"💰 Total Investment: ₹{summary.get('total_investment', 0):,.0f}").classes(
                                "text-sm")
                            ui.label(f"💎 Portfolio Value: ₹{summary.get('total_portfolio_value', 0):,.0f}").classes(
                                "text-sm")
                            ui.label(f"📈 Overall Return: {summary.get('overall_return_percent', 0):.2f}%").classes(
                                "text-sm font-semibold text-green-600")
                            ui.label(f"🔄 Total Trades: {summary.get('total_trades', 0)}").classes("text-sm")
                            ui.label(f"⏭️ Skipped: {summary.get('total_skipped', 0)}").classes("text-sm")

                        # Benchmark Summary
                        if benchmark_summary:
                            with ui.card().classes("flex-1 p-4"):
                                ui.label("📋 Benchmark Summary").classes("font-semibold mb-2")
                                ui.label(
                                    f"💰 Total Investment: ₹{benchmark_summary.get('total_investment', 0):,.0f}").classes(
                                    "text-sm")
                                ui.label(
                                    f"💎 Portfolio Value: ₹{benchmark_summary.get('total_portfolio_value', 0):,.0f}").classes(
                                    "text-sm")
                                ui.label(
                                    f"📈 Overall Return: {benchmark_summary.get('overall_return_percent', 0):.2f}%").classes(
                                    "text-sm font-semibold text-blue-600")
                                ui.label(f"🔄 Total Trades: {benchmark_summary.get('total_trades', 0)}").classes(
                                    "text-sm")

                    # Individual Symbol Results
                    symbol_results = batch_result.get('results', [])

                    for symbol_result in symbol_results:
                        symbol = symbol_result.get('symbol', 'Unknown')

                        with ui.expansion(f"📈 {symbol} Detailed Results").classes("w-full mb-2"):
                            await self.display_symbol_result_details(symbol_result)

    async def display_symbol_result_details(self, result: dict):
        """Display detailed results for a single symbol"""

        symbol = result.get('symbol', 'Unknown')

        # Key Metrics
        with ui.row().classes("gap-4 mb-4"):
            with ui.card().classes("flex-1 p-4 bg-blue-50"):
                ui.label("📊 Performance Metrics").classes("font-semibold mb-2")
                ui.label(f"💰 Total Investment: ₹{result.get('total_investment', 0):,.0f}").classes("text-sm")
                ui.label(f"💎 Final Value: ₹{result.get('final_portfolio_value', 0):,.0f}").classes("text-sm")
                ui.label(f"📈 Total Return: {result.get('total_return_percent', 0):.2f}%").classes("text-sm font-bold")
                ui.label(f"📊 CAGR: {result.get('cagr_percent', 0):.2f}%").classes("text-sm font-bold")
                ui.label(f"💹 Avg Buy Price: ₹{result.get('average_buy_price', 0):.2f}").classes("text-sm")

            # Benchmark Comparison
            comparison = result.get('comparison', {})
            if comparison:
                with ui.card().classes("flex-1 p-4 bg-green-50"):
                    ui.label("🆚 vs Benchmark").classes("font-semibold mb-2")
                    ui.label(
                        f"📈 Return Outperformance: {comparison.get('return_outperformance_percent', 0):.2f}%").classes(
                        "text-sm font-bold text-green-600")
                    ui.label(f"📊 CAGR Outperformance: {comparison.get('cagr_outperformance_percent', 0):.2f}%").classes(
                        "text-sm font-bold text-green-600")
                    ui.label(
                        f"⚡ Investment Efficiency: {comparison.get('investment_efficiency_percent', 0):.1f}%").classes(
                        "text-sm")
                    ui.label(f"💡 {comparison.get('performance_summary', 'N/A')}").classes("text-sm font-medium")

        # Trade Breakdown
        trade_breakdown = result.get('trade_breakdown', {})
        if trade_breakdown:
            with ui.card().classes("w-full mb-4 p-4"):
                ui.label("🔄 Trade Analysis").classes("font-semibold mb-2")

                with ui.row().classes("gap-4"):
                    ui.label(f"📅 Regular Trades: {trade_breakdown.get('regular_trades', 0)}").classes("text-sm")
                    ui.label(f"🚨 Extreme Trades: {trade_breakdown.get('extreme_trades', 0)}").classes(
                        "text-sm text-red-600")
                    ui.label(f"⚡ Force Trades: {trade_breakdown.get('force_trades', 0)}").classes(
                        "text-sm text-orange-600")
                    ui.label(f"🚫 Monthly Limits Hit: {result.get('monthly_limit_exceeded', 0)}").classes("text-sm")
                    ui.label(f"⏭️ Price Threshold Skipped: {result.get('price_threshold_skipped', 0)}").classes("text-sm")

        # Monthly Summary (show first few months as sample)
        monthly_summary = result.get('monthly_summary', {})
        if monthly_summary:
            with ui.expansion("📅 Monthly Investment Summary (Sample)").classes("w-full mb-4"):
                sample_months = list(monthly_summary.items())[:6]  # Show first 6 months

                monthly_table = ui.table(
                    columns=[
                        {'name': 'month', 'label': 'Month', 'field': 'month'},
                        {'name': 'invested', 'label': 'Invested (₹)', 'field': 'invested'},
                        {'name': 'investments', 'label': 'Count', 'field': 'investments'},
                        {'name': 'remaining', 'label': 'Remaining (₹)', 'field': 'remaining'},
                        {'name': 'utilization', 'label': 'Utilization %', 'field': 'utilization'}
                    ],
                    rows=[
                        {
                            'month': month,
                            'invested': f"₹{data.get('total_invested', 0):,.0f}",
                            'investments': data.get('num_investments', 0),
                            'remaining': f"₹{data.get('remaining_budget', 0):,.0f}",
                            'utilization': f"{data.get('budget_utilization_percent', 0):.1f}%"
                        }
                        for month, data in sample_months
                    ]
                ).classes("w-full")

                if len(monthly_summary) > 6:
                    ui.label(f"... and {len(monthly_summary) - 6} more months").classes("text-sm text-gray-500 mt-2")

        # Recent Trades (show last 5)
        trades = result.get('trades', [])
        if trades:
            with ui.expansion("💼 Recent Trades (Last 5)").classes("w-full"):
                recent_trades = trades[-5:]

                trades_table = ui.table(
                    columns=[
                        {'name': 'date', 'label': 'Date', 'field': 'date'},
                        {'name': 'price', 'label': 'Price (₹)', 'field': 'price'},
                        {'name': 'units', 'label': 'Units', 'field': 'units'},
                        {'name': 'amount', 'label': 'Amount (₹)', 'field': 'amount'},
                        {'name': 'type', 'label': 'Type', 'field': 'type'},
                        {'name': 'drawdown', 'label': 'Drawdown %', 'field': 'drawdown'}
                    ],
                    rows=[
                        {
                            'date': trade.get('date', ''),
                            'price': f"₹{trade.get('price', 0):.2f}",
                            'units': f"{trade.get('units', 0):.2f}",
                            'amount': f"₹{trade.get('amount', 0):,.0f}",
                            'type': trade.get('trade_type', '').replace('_', ' ').title(),
                            'drawdown': f"{trade.get('drawdown', 0):.2f}%" if trade.get('drawdown') else 'N/A'
                        }
                        for trade in recent_trades
                    ]
                ).classes("w-full")

            # Enhanced Features Applied
            enhancements = result.get('enhancements_applied', [])
            if enhancements:
                with ui.card().classes("w-full mt-4 p-4 bg-yellow-50"):
                    ui.label("✨ Applied Enhancements").classes("font-semibold mb-2")
                    for enhancement in enhancements:
                        ui.label(f"• {enhancement}").classes("text-sm")

    async def render_optimize_config_section(self, fetch_api, user_storage):
        """Render config optimization section"""

        ui.label("⚙️ Optimize Configuration").classes("text-lg font-bold mb-1")
        ui.label("Find the optimal strategy parameters based on historical data").classes("text-gray-600 mb-4")

        with ui.row().classes("gap-2 mb-2"):
            with ui.column().classes("w-full"):
                with ui.card().classes("w-full"):
                    symbol_input = ui.input(
                        label="Symbol",
                        value="ICICIB22"
                    ).classes("w-40 mb-1")

                    target_monthly_utilization = ui.number(
                        label="Target Monthly Utilization (%)",
                        value=80,
                        min=50,
                        max=100,
                        step=1
                    ).classes("w-40 mb-1")

                    risk_tolerance = ui.select(
                        label="Risk Tolerance",
                        options=[
                            "conservative",
                            "moderate",
                            "aggressive"
                        ],
                        value="conservative",
                    ).classes("w-40 mb-1")

            with ui.column().classes("gap-4 mb-4"):
                optimize_results = ui.column().classes("w-full mt-4")

        async def run_optimize():
            try:
                if not symbol_input:
                    self.show_warning("No symbols")
                    return

                url = f"/sip/optimize-config?symbol={symbol_input}&target_monthly_utilization={target_monthly_utilization.value}&risk_tolerance={risk_tolerance.value}"

                optimized = await self.safe_api_call(fetch_api, url, "POST", data={})

                if optimized:
                    optimize_results.clear()
                    with optimize_results:
                        ui.label("Optimized Configuration").classes("font-bold mb-2")
                        config = optimized.get("optimized_config", {})

                        with ui.card().classes("p-4 bg-cyan-50"):
                            with ui.grid(columns=5).classes("gap-4"):
                                for key, value in config.items():
                                    with ui.column().classes("p-2 border rounded bg-white"):
                                        ui.label(key.replace('_', ' ').title()).classes("text-xs text-gray-500")
                                        if isinstance(value, bool):
                                            display_value = "✅ Yes" if value else "❌ No"
                                            color = "text-green-600" if value else "text-red-600"
                                        elif isinstance(value, (int, float)):
                                            display_value = f"{value:,.2f}"
                                            color = "text-blue-600"
                                        else:
                                            display_value = str(value)
                                            color = "text-gray-800"
                                        ui.label(display_value).classes(f"font-semibold {color}")
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
            start_date = ui.input("Start Date", value="2023-01-01").props("dense type=date").classes("flex-1 min-w-0")
            end_date = ui.input("End Date", value=datetime.now().strftime("%Y-%m-%d")).props(
                "dense type=date").classes("flex-1 min-w-0")
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

        symbol_input = ui.input(label="ICICI Bank Symbol", placeholder="ICICIB22").classes("w-full mb-4")

        with ui.row().classes("gap-4"):
            start_date = ui.input("Start Date", value="2023-01-01").props("dense type=date").classes("flex-1 min-w-0")
            end_date = ui.input("End Date", value=datetime.now().strftime("%Y-%m-%d")).props(
                "dense type=date").classes("flex-1 min-w-0")
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
        """Display candlestick chart with strategy and benchmark trade markers (fixed endpoint + parsing)."""

        try:
            # Determine date range from trades (supports 'date' or 'timestamp')
            def extract_trade_date(t: Dict) -> Optional[str]:
                d = t.get('date') or t.get('timestamp')
                if not d:
                    return None
                try:
                    return pd.to_datetime(d).date().strftime('%Y-%m-%d')
                except Exception:
                    return None

            all_trades = strategy_trades + benchmark_trades
            trade_dates = [extract_trade_date(t) for t in all_trades]
            trade_dates = [d for d in trade_dates if d]

            if trade_dates:
                start_date = min(trade_dates)
                end_date = max(trade_dates)
            else:
                # Fallback to recent 365 days if no trades/dates
                end_date = datetime.now().date().strftime('%Y-%m-%d')
                start_date = (datetime.now().date() - timedelta(days=365)).strftime('%Y-%m-%d')

            # Use the backend market-data endpoint that exists
            # GET /historical-data/Upstox?trading_symbol=SYMBOL&from_date=YYYY-MM-DD&to_date=YYYY-MM-DD&unit=day&interval=1
            endpoint = (
                f"/historical-data/Upstox?trading_symbol={symbol}"
                f"&from_date={start_date}&to_date={end_date}&unit=day&interval=1"
            )

            data = await self.safe_api_call(self.fetch_api, endpoint)
            points = (data or {}).get('data', [])
            if not points:
                ui.notify("No market data available for chart", type="warning")
                return

            # Prepare OHLC dataframe
            try:
                df = pd.DataFrame(points)
                if df.empty:
                    ui.notify("No historical data available", type="warning")
                    return
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                # Ensure required OHLC columns exist (service returns open/high/low/close at top-level)
                required_cols = {'open', 'high', 'low', 'close'}
                if not required_cols.issubset(df.columns):
                    ui.notify("Market data missing OHLC columns", type="negative")
                    return
            except Exception as e:
                ui.notify(f"Error processing market data: {str(e)}", type="negative")
                return

            # Create candlestick chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=f"{symbol} Price",
                showlegend=True
            ))

            # Strategy trade markers (support 'date' or 'timestamp')
            if strategy_trades:
                s_dates, s_prices, s_amounts = [], [], []
                for t in strategy_trades:
                    d = extract_trade_date(t)
                    p = t.get('price')
                    if d and p is not None:
                        s_dates.append(pd.to_datetime(d))
                        s_prices.append(float(p))
                        s_amounts.append(t.get('amount', 0))

                if s_dates:
                    fig.add_trace(go.Scatter(
                        x=s_dates,
                        y=s_prices,
                        mode='markers+text',
                        marker=dict(symbol='star', size=14, color='red', line=dict(width=2, color='darkred')),
                        text=[f"₹{amt:,.0f}" for amt in s_amounts],
                        textposition="top center",
                        textfont=dict(size=10, color='red'),
                        name=f'Strategy Trades ({len(strategy_trades)})',
                        hovertemplate="<b>Strategy Trade</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<br>Amount: %{text}<extra></extra>",
                        showlegend=True
                    ))

            # Benchmark trade markers
            if benchmark_trades:
                b_dates, b_prices, b_amounts = [], [], []
                for t in benchmark_trades:
                    d = extract_trade_date(t)
                    p = t.get('price')
                    if d and p is not None:
                        b_dates.append(pd.to_datetime(d))
                        b_prices.append(float(p))
                        b_amounts.append(t.get('amount', 0))

                if b_dates:
                    fig.add_trace(go.Scatter(
                        x=b_dates,
                        y=b_prices,
                        mode='markers+text',
                        marker=dict(symbol='circle', size=12, color='blue', line=dict(width=2, color='darkblue')),
                        text=[f"₹{amt:,.0f}" for amt in b_amounts],
                        textposition="bottom center",
                        textfont=dict(size=9, color='blue'),
                        name=f'Benchmark Trades ({len(benchmark_trades)})',
                        hovertemplate="<b>Benchmark Trade</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<br>Amount: %{text}<extra></extra>",
                        showlegend=True
                    ))

            # Layout
            fig.update_layout(
                title=f"{symbol} - Price Chart with Trade Overlays",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=600,
                hovermode='x unified'
            )

            with ui.dialog() as dialog, ui.card().classes("w-[90vw] h-[80vh] max-w-none"):
                with ui.row().classes("w-full justify-between items-center mb-4"):
                    ui.label(f"{symbol} Trade Analysis Chart").classes("text-xl font-bold")
                    with ui.row().classes("gap-2"):
                        ui.button("📊 Download Chart",
                                  on_click=lambda: ui.download(fig.to_html(), f"{symbol}_trade_chart.html")).classes("text-xs")
                        ui.button("❌ Close", on_click=dialog.close).classes("text-xs")
                ui.plotly(fig).classes("w-full h-full")
            dialog.open()

        except Exception as e:
            logger.error(f"Error creating trade chart: {e}")
            ui.notify(f"Failed to create trade chart: {str(e)}", type="negative")

    async def display_comparison_chart(self, symbol: str, strategy: Dict, benchmark: Dict):
        """Display line chart comparing strategy and benchmark performance - FIXED VERSION"""

        try:
            # FIXED: Extract ROI data directly from the response structure
            strategy_roi_data = strategy.get('return_on_investment', [])
            benchmark_roi_data = benchmark.get('return_on_investment', [])

            # If ROI data is available, use it directly
            if strategy_roi_data and benchmark_roi_data:
                return await self._display_roi_comparison_chart(symbol, strategy_roi_data, benchmark_roi_data, strategy,
                                                                benchmark)

            # Fallback to historical data method if ROI not available
            return await self._display_historical_comparison_chart(symbol, strategy, benchmark)

        except Exception as e:
            logger.error(f"Error in comparison chart: {e}")
            ui.notify(f"Failed to create comparison chart: {str(e)}", type="negative")

    async def _display_roi_comparison_chart(self, symbol: str, strategy_roi: List, benchmark_roi: List, strategy: Dict,
                                            benchmark: Dict):
        """Create comparison chart using ROI data - NEW METHOD"""

        try:
            # Convert ROI data to dataframes
            strategy_df = pd.DataFrame(strategy_roi)
            benchmark_df = pd.DataFrame(benchmark_roi)

            if strategy_df.empty or benchmark_df.empty:
                ui.notify("Insufficient ROI data for comparison", type="warning")
                return

            # Ensure date columns are datetime
            strategy_df['date'] = pd.to_datetime(strategy_df['date'])
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])

            # Sort by date
            strategy_df = strategy_df.sort_values('date')
            benchmark_df = benchmark_df.sort_values('date')

            # Create enhanced comparison chart
            fig = go.Figure()

            # Add strategy performance line
            fig.add_trace(go.Scatter(
                x=strategy_df['date'],
                y=strategy_df['portfolio_value'],
                name='Strategy Portfolio',
                line=dict(color='#1f77b4', width=3),
                hovertemplate="<b>Strategy</b><br>" +
                              "Date: %{x}<br>" +
                              "Value: ₹%{y:,.0f}<br>" +
                              "<extra></extra>"
            ))

            # Add benchmark performance line
            fig.add_trace(go.Scatter(
                x=benchmark_df['date'],
                y=benchmark_df['portfolio_value'],
                name='Benchmark Portfolio',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                hovertemplate="<b>Benchmark</b><br>" +
                              "Date: %{x}<br>" +
                              "Value: ₹%{y:,.0f}<br>" +
                              "<extra></extra>"
            ))

            # Calculate and add performance metrics
            strategy_final = strategy_df['portfolio_value'].iloc[-1]
            benchmark_final = benchmark_df['portfolio_value'].iloc[-1]
            outperformance = ((strategy_final / benchmark_final) - 1) * 100

            # Enhanced layout
            fig.update_layout(
                title=f"{symbol} - Strategy vs Benchmark Performance<br>" +
                      f"<sub>Outperformance: {outperformance:+.2f}%</sub>",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )

            # Show performance summary
            strategy_return = strategy.get('total_return_percent', 0)
            benchmark_return = benchmark.get('total_return_percent', 0)

            with ui.dialog() as dialog, ui.card().classes("w-[90vw] h-[85vh] max-w-none"):
                with ui.row().classes("w-full justify-between items-center mb-4"):
                    ui.label(f"{symbol} Performance Comparison").classes("text-xl font-bold")
                    ui.button("❌ Close", on_click=dialog.close)

                # Performance metrics summary
                with ui.card().classes("w-full p-4 mb-4 bg-gray-50"):
                    ui.label("📊 Performance Summary").classes("text-lg font-semibold mb-2")
                    with ui.grid(columns=4).classes("w-full gap-4"):
                        with ui.card().classes("p-3 text-center"):
                            ui.label("Strategy Return").classes("text-sm text-gray-600")
                            color = "text-green-600" if strategy_return >= 0 else "text-red-600"
                            ui.label(f"{strategy_return:+.2f}%").classes(f"text-lg font-bold {color}")

                        with ui.card().classes("p-3 text-center"):
                            ui.label("Benchmark Return").classes("text-sm text-gray-600")
                            color = "text-green-600" if benchmark_return >= 0 else "text-red-600"
                            ui.label(f"{benchmark_return:+.2f}%").classes(f"text-lg font-bold {color}")

                        with ui.card().classes("p-3 text-center"):
                            ui.label("Outperformance").classes("text-sm text-gray-600")
                            color = "text-green-600" if outperformance >= 0 else "text-red-600"
                            ui.label(f"{outperformance:+.2f}%").classes(f"text-lg font-bold {color}")

                        with ui.card().classes("p-3 text-center"):
                            ui.label("Final Portfolio").classes("text-sm text-gray-600")
                            ui.label(f"₹{strategy_final:,.0f}").classes("text-lg font-bold text-blue-600")

                ui.plotly(fig).classes("w-full flex-1")

            dialog.open()

        except Exception as e:
            logger.error(f"Error creating ROI comparison chart: {e}")
            ui.notify(f"Failed to create ROI comparison: {str(e)}", type="negative")

    async def _display_historical_comparison_chart(self, symbol: str, strategy: Dict, benchmark: Dict):
        """Fallback comparison chart by reconstructing portfolio values from trades and daily prices."""

        try:
            strategy_trades = strategy.get('trades', []) or []
            benchmark_trades = benchmark.get('trades', []) or []

            if not strategy_trades and not benchmark_trades:
                ui.notify("No trade history available for comparison", type="warning")
                return

            # Helper to extract date string from trade
            def trade_date_str(t: Dict) -> Optional[str]:
                d = t.get('date') or t.get('timestamp')
                if not d:
                    return None
                try:
                    return pd.to_datetime(d).date().strftime('%Y-%m-%d')
                except Exception:
                    return None

            # Determine date range
            dates = [d for d in [trade_date_str(t) for t in (strategy_trades + benchmark_trades)] if d]
            if dates:
                start_date, end_date = min(dates), max(dates)
            else:
                # Try parse from period field like "YYYY-MM-DD to YYYY-MM-DD"
                period = strategy.get('period') or benchmark.get('period') or ''
                if 'to' in period:
                    parts = [p.strip() for p in period.split('to')]
                    start_date = parts[0]
                    end_date = parts[1] if len(parts) > 1 else datetime.now().date().strftime('%Y-%m-%d')
                else:
                    end_date = datetime.now().date().strftime('%Y-%m-%d')
                    start_date = (datetime.now().date() - timedelta(days=365)).strftime('%Y-%m-%d')

            # Fetch daily prices for the full range
            endpoint = (
                f"/historical-data/Upstox?trading_symbol={symbol}"
                f"&from_date={start_date}&to_date={end_date}&unit=day&interval=1"
            )
            data = await self.safe_api_call(self.fetch_api, endpoint)
            points = (data or {}).get('data', [])
            if not points:
                ui.notify("No market data available for comparison", type="warning")
                return

            price_df = pd.DataFrame(points)
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            price_df = price_df.sort_values('timestamp')

            # Index trades by date for quick accumulation
            def normalize_trades(trades: List[Dict]) -> Dict[str, Dict[str, float]]:
                acc: Dict[str, Dict[str, float]] = {}
                for t in trades:
                    d = trade_date_str(t)
                    if not d:
                        continue
                    amt = float(t.get('amount', 0) or 0)
                    units = float(t.get('units', 0) or 0)
                    if d not in acc:
                        acc[d] = {'amount': 0.0, 'units': 0.0}
                    acc[d]['amount'] += amt
                    acc[d]['units'] += units
                return acc

            strat_by_date = normalize_trades(strategy_trades)
            bench_by_date = normalize_trades(benchmark_trades)

            # Build daily portfolio values by accumulating units
            s_units = 0.0
            b_units = 0.0
            s_values = []
            b_values = []
            dates = []

            for _, row in price_df.iterrows():
                d = row['timestamp'].date().strftime('%Y-%m-%d')
                close = float(row['close'])

                if d in strat_by_date:
                    s_units += strat_by_date[d]['units']
                if d in bench_by_date:
                    b_units += bench_by_date[d]['units']

                dates.append(row['timestamp'])
                s_values.append(s_units * close)
                b_values.append(b_units * close)

            # Create comparison chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=s_values, name='Strategy Portfolio', line=dict(color='#1f77b4', width=3)))
            fig.add_trace(go.Scatter(x=dates, y=b_values, name='Benchmark Portfolio', line=dict(color='#ff7f0e', width=3, dash='dash')))

            # Metrics
            if b_values and b_values[-1] != 0:
                outperformance = ((s_values[-1] / b_values[-1]) - 1) * 100
            else:
                outperformance = 0.0

            fig.update_layout(
                title=f"{symbol} - Strategy vs Benchmark (Reconstructed)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )

            s_return = strategy.get('total_return_percent', 0)
            b_return = benchmark.get('total_return_percent', 0)

            with ui.dialog() as dialog, ui.card().classes("w-[90vw] h-[85vh] max-w-none"):
                with ui.row().classes("w-full justify-between items-center mb-4"):
                    ui.label(f"{symbol} Performance Comparison").classes("text-xl font-bold")
                    ui.button("❌ Close", on_click=dialog.close)

                with ui.card().classes("w-full p-4 mb-4 bg-gray-50"):
                    ui.label("📊 Performance Summary").classes("text-lg font-semibold mb-2")
                    with ui.grid(columns=4).classes("w-full gap-4"):
                        with ui.card().classes("p-3 text-center"):
                            ui.label("Strategy Return").classes("text-sm text-gray-600")
                            color = "text-green-600" if s_return >= 0 else "text-red-600"
                            ui.label(f"{s_return:+.2f}%").classes(f"text-lg font-bold {color}")
                        with ui.card().classes("p-3 text-center"):
                            ui.label("Benchmark Return").classes("text-sm text-gray-600")
                            color = "text-green-600" if b_return >= 0 else "text-red-600"
                            ui.label(f"{b_return:+.2f}%").classes(f"text-lg font-bold {color}")
                        with ui.card().classes("p-3 text-center"):
                            ui.label("Outperformance").classes("text-sm text-gray-600")
                            color = "text-green-600" if outperformance >= 0 else "text-red-600"
                            ui.label(f"{outperformance:+.2f}%").classes(f"text-lg font-bold {color}")
                        with ui.card().classes("p-3 text-center"):
                            ui.label("Final Portfolio").classes("text-sm text-gray-600")
                            ui.label(f"₹{s_values[-1]:,.0f}").classes("text-lg font-bold text-blue-600")

                ui.plotly(fig).classes("w-full flex-1")
            dialog.open()

        except Exception as e:
            logger.error(f"Error creating historical comparison chart: {e}")
            ui.notify(f"Failed to create comparison chart: {str(e)}", type="negative")

    async def render_enhanced_analytics_panel(self, fetch_api, user_storage):
        """Enhanced analytics panel with comprehensive portfolio insights and benchmarks"""

        ui.label("📈 Portfolio Analytics Dashboard").classes("text-xl font-bold mb-4")
        ui.label("Comprehensive analysis of your SIP portfolio performance").classes("text-sm text-slate-300 mb-4")

        # Portfolio selector
        portfolios = await self.safe_api_call(fetch_api, "/sip/portfolio")

        if not portfolios:
            with ui.card().classes("w-full p-8 text-center"):
                ui.icon("analytics", size="2rem").classes("text-slate-400 mb-3")
                ui.label("📊 No portfolios found").classes("text-lg font-bold text-slate-300 mb-2")
                ui.label("Create a portfolio first to view analytics").classes("text-sm text-slate-400")
            return

        portfolio_options = {
            p['portfolio_id']: f"{p.get('portfolio_name', 'Unnamed')} ({p.get('portfolio_type', 'single')})"
            for p in portfolios}

        with ui.card().classes("w-full p-4 mb-4"):
            with ui.row().classes("w-full items-center mb-3"):
                ui.icon("folder", size="1.5rem").classes("text-blue-400")
                ui.label("Select Portfolio for Analysis").classes("font-bold text-lg ml-3")

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
        ui.button("🔄 Refresh Analytics", on_click=load_analytics).classes("bg-blue-500 text-white px-4 py-2 rounded font-semibold mt-4")

    async def display_portfolio_analytics(self, analytics, container):
        """Display comprehensive portfolio analytics with benchmarks"""

        with container:
            portfolio_name = analytics.get('portfolio_name', 'Portfolio')
            ui.label(f"📊 Analytics for {portfolio_name}").classes("text-lg font-bold mb-4")

            # Overview metrics
            total_invested = analytics.get('total_invested', 0)
            current_value = analytics.get('current_value', 0)
            total_return_pct = analytics.get('total_return_percent', 0)
            cagr_pct = analytics.get('cagr_percent', 0)
            days_invested = analytics.get('days_invested', 0)
            total_trades = analytics.get('total_trades', 0)

            with ui.card().classes("w-full p-4 mb-4"):
                ui.label("📈 Performance Overview").classes("text-lg font-bold mb-4")

                with ui.grid(columns=3).classes("w-full gap-4"):
                    # Investment metrics
                    with ui.card().classes("p-4 text-center"):
                        ui.label("💰 Total Invested").classes("text-sm text-slate-400")
                        ui.label(f"₹{total_invested:,.0f}").classes("text-lg font-bold text-blue-400")

                        ui.label("💎 Current Value").classes("text-sm text-slate-400 mt-2")
                        ui.label(f"₹{current_value:,.0f}").classes("text-lg font-bold text-green-400")

                    # Returns metrics
                    with ui.card().classes("p-4 text-center"):
                        ui.label("📊 Total Return").classes("text-sm text-slate-400")
                        return_color = "text-green-400" if total_return_pct >= 0 else "text-red-400"
                        ui.label(f"{total_return_pct:+.2f}%").classes(f"text-lg font-bold {return_color}")

                        ui.label("📈 CAGR").classes("text-sm text-slate-400 mt-2")
                        cagr_color = "text-green-400" if cagr_pct >= 8 else "text-red-400"
                        ui.label(f"{cagr_pct:.2f}%").classes(f"text-lg font-bold {cagr_color}")

                    # Activity metrics
                    with ui.card().classes("p-4 text-center"):
                        ui.label("📅 Days Invested").classes("text-sm text-slate-400")
                        ui.label(str(days_invested)).classes("text-lg font-bold text-purple-400")

                        ui.label("🔄 Total Trades").classes("text-sm text-slate-400 mt-2")
                        ui.label(str(total_trades)).classes("text-lg font-bold text-orange-400")

            # Symbol-wise analytics (for multi-symbol portfolios)
            symbols_analytics = analytics.get('symbols_analytics', {})
            if symbols_analytics:
                ui.label("🎯 Symbol-wise Performance").classes("text-lg font-bold mb-4")

                for symbol, symbol_data in symbols_analytics.items():
                    with ui.card().classes("w-full mb-4 p-4 border-l-4 border-blue-500"):
                        with ui.row().classes("w-full justify-between items-center"):
                            ui.label(f"📈 {symbol}").classes("text-lg font-bold")
                            ui.label(f"Allocation: {symbol_data.get('allocation_percent', 0):.1f}%").classes(
                                "text-sm text-slate-400")

                        with ui.grid(columns=4).classes("w-full gap-4 mt-4"):
                            ui.label(f"Invested: ₹{symbol_data.get('invested', 0):,.0f}").classes("text-sm")
                            ui.label(f"Units: {symbol_data.get('units', 0):.2f}").classes("text-sm")
                            ui.label(f"Avg Price: ₹{symbol_data.get('avg_buy_price', 0):.2f}").classes("text-sm")

                            return_pct = symbol_data.get('return_percent', 0)
                            return_color = "text-green-400" if return_pct >= 0 else "text-red-400"
                            ui.label(f"Return: {return_pct:+.2f}%").classes(f"text-sm {return_color}")

    async def render_enhanced_config_panel(self, fetch_api, user_storage):
        """Enhanced configuration panel with all user input parameters - FIXED VERSION"""

        ui.label("⚙️ Enhanced SIP Configuration").classes("text-xl font-bold mb-4")
        ui.label("Complete configuration including all strategy parameters").classes("text-sm text-slate-300 mb-4")

        # Load current configuration
        try:
            config_data = await self.safe_api_call(fetch_api, "/sip/config/defaults")
            current_config = config_data.get('enhanced_config', {}) if config_data else {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            current_config = {
                "fixed_investment": 5000,
                "max_amount_in_a_month": 20000,
                "price_reduction_threshold": 4.0,
                "major_drawdown_threshold": -10.0,
                "minor_drawdown_threshold": -4.0,
                "rolling_window": 100,
                "fallback_day": 15,
                "min_investment_gap_days": 5
            }

        # FIXED: Complete configuration form with all parameters
        with ui.card().classes("w-full max-w-4xl mx-auto p-4"):
            ui.label("💰 Investment Parameters").classes("text-lg font-bold mb-4")

            with ui.grid(columns=2).classes("w-full gap-6"):
                # Left column - Basic Investment Settings
                with ui.column().classes("gap-4"):
                    fixed_investment = ui.number(
                        label="💵 Fixed Investment Amount (₹)",
                        value=current_config.get("fixed_investment", 5000),
                        min=500, max=100000, step=500
                    ).classes("w-full")

                    max_monthly = ui.number(
                        label="📅 Max Monthly Investment (₹)",
                        value=current_config.get("max_amount_in_a_month", 20000),
                        min=2000, max=500000, step=1000
                    ).classes("w-full")

                    # FIXED: Added missing price_reduction_threshold input
                    price_reduction_threshold = ui.number(
                        label="📉 Price Reduction Threshold (%)",
                        value=current_config.get("price_reduction_threshold", 4.0),
                        min=1.0, max=15.0, step=0.5
                    ).classes("w-full")

                    fallback_day = ui.number(
                        label="📆 Investment Day of Month",
                        value=current_config.get("fallback_day", 15),
                        min=1, max=28, step=1
                    ).classes("w-full")

                # Right column - Advanced Parameters
                with ui.column().classes("gap-4"):
                    # FIXED: Added missing rolling_window input
                    rolling_window = ui.number(
                        label="🔄 Rolling Window (days)",
                        value=current_config.get("rolling_window", 100),
                        min=20, max=365, step=10
                    ).classes("w-full")

                    min_gap_days = ui.number(
                        label="⏱️ Min Investment Gap (days)",
                        value=current_config.get("min_investment_gap_days", 5),
                        min=1, max=30, step=1
                    ).classes("w-full")

                    major_drawdown = ui.number(
                        label="📊 Major Drawdown Threshold (%)",
                        value=current_config.get("major_drawdown_threshold", -10.0),
                        min=-30.0, max=0.0, step=1.0
                    ).classes("w-full")

                    minor_drawdown = ui.number(
                        label="📈 Minor Drawdown Threshold (%)",
                        value=current_config.get("minor_drawdown_threshold", -4.0),
                        min=-15.0, max=0.0, step=0.5
                    ).classes("w-full")

        # FIXED: Enhanced parameter explanations
        with ui.card().classes("w-full max-w-4xl mx-auto p-4 mt-4"):
            ui.label("📚 Parameter Explanations").classes("text-xl font-bold mb-6 theme-header-text")

            with ui.grid(columns=2).classes("w-full gap-4"):
                with ui.column():
                    ui.label("🔄 Rolling Window").classes("font-medium")
                    ui.label(
                        "Number of days to look back for calculating moving averages and market conditions. Higher values = smoother signals, lower values = more responsive.").classes(
                        "text-sm text-gray-600 mb-3")

                    ui.label("📉 Price Reduction Threshold").classes("font-medium")
                    ui.label(
                        "Minimum price drop (%) required for additional investments within the same month. Prevents over-investing on minor fluctuations.").classes(
                        "text-sm text-gray-600 mb-3")

                with ui.column():
                    ui.label("📅 Max Monthly Investment").classes("font-medium")
                    ui.label(
                        "Maximum amount to invest per symbol per month. Usually 3-5x the fixed investment amount for drawdown opportunities.").classes(
                        "text-sm text-gray-600 mb-3")

                    ui.label("⏱️ Min Investment Gap").classes("font-medium")
                    ui.label(
                        "Minimum days between investments to avoid frequent trading and allow market movements to develop.").classes(
                        "text-sm text-gray-600")

        # Configuration templates section
        with ui.card().classes("w-full max-w-4xl mx-auto p-4 mt-4"):
            ui.label("📋 Configuration Templates").classes("text-xl font-bold mb-6 theme-header-text")

            async def apply_conservative_template():
                fixed_investment.value = 5000
                rolling_window.value = 150
                price_reduction_threshold.value = 5.0
                max_monthly.value = 20000
                min_gap_days.value = 7
                ui.notify("Conservative template applied", type="positive")

            async def apply_aggressive_template():
                fixed_investment.value = 7500
                rolling_window.value = 50
                price_reduction_threshold.value = 2.5
                max_monthly.value = 35000
                min_gap_days.value = 3
                ui.notify("Aggressive template applied", type="positive")

            with ui.row().classes("gap-6"):
                ui.button("🛡️ Conservative", on_click=apply_conservative_template).classes("btn-modern-success text-lg px-6 py-3 font-semibold")
                ui.button("⚡ Aggressive", on_click=apply_aggressive_template).classes("btn-modern-danger text-lg px-6 py-3 font-semibold")
                ui.button("⚖️ Balanced",
                          on_click=lambda: ui.notify("Balanced template coming soon", type="info")).classes(
                    "btn-modern-primary text-lg px-6 py-3 font-semibold")

        # Save configuration
        async def save_configuration():
            try:
                config = {
                    "fixed_investment": fixed_investment.value,
                    "max_amount_in_a_month": max_monthly.value,
                    "price_reduction_threshold": price_reduction_threshold.value,
                    "rolling_window": rolling_window.value,
                    "fallback_day": fallback_day.value,
                    "min_investment_gap_days": min_gap_days.value,
                    "major_drawdown_threshold": major_drawdown.value,
                    "minor_drawdown_threshold": minor_drawdown.value
                }

                # Save to user storage or send to backend
                if user_storage:
                    user_storage["sip_config"] = config

                ui.notify("✅ Configuration saved successfully!", type="positive")

            except Exception as e:
                ui.notify(f"❌ Failed to save configuration: {str(e)}", type="negative")

        ui.button("💾 Save Configuration", on_click=save_configuration).classes("btn-modern-primary text-xl px-8 py-4 font-bold mt-8")

    async def render_investment_reports_panel(self, fetch_api, user_storage):
        """Render investment reports panel with report generation and history"""

        ui.label("📊 Investment Reports & Analysis").classes("text-xl font-bold mb-4")
        ui.label("Generate comprehensive investment reports for multiple symbols").classes("text-sm text-slate-300 mb-4")

        # Report generation section
        with ui.card().classes("w-full mb-4 p-4"):
            ui.label("📋 Generate Investment Report").classes("text-lg font-bold mb-4")

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
                "bg-blue-500 text-white px-6 py-3 font-semibold")
            ui.button("⚡ Quick Report", on_click=generate_quick_report).classes("bg-green-500 text-white px-6 py-3 font-semibold")

        # Report history section
        reports_container = ui.column().classes("w-full")

        async def load_report_history():
            """Load and display report history"""
            try:
                reports = await self.safe_api_call(fetch_api, "/sip/reports/history?limit=10")

                reports_container.clear()

                if reports:
                    with reports_container:
                        ui.label("📋 Recent Reports").classes("text-lg font-bold mb-4")

                        for report in reports:
                            await render_report_history_card(report, fetch_api)
                else:
                    with reports_container:
                        with ui.card().classes("w-full p-8 text-center"):
                            ui.icon("description", size="2rem").classes("text-slate-400 mb-3")
                            ui.label("📝 No reports generated yet").classes("text-lg font-bold text-slate-300 mb-2")
                            ui.label("Generate your first report to get started").classes("text-sm text-slate-400")

            except Exception as e:
                logger.error(f"Error loading report history: {e}")
                with reports_container:
                    ui.label("❌ Error loading report history").classes("text-red-500")

        # Load report history on page load
        await load_report_history()

        # Refresh button
        ui.button("🔄 Refresh History", on_click=load_report_history).classes("bg-blue-500 text-white px-4 py-2 rounded font-semibold mt-4")

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
