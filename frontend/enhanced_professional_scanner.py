"""
Enhanced Professional Trading Analytics & Scanner UI
Comprehensive data-driven design with advanced visualizations and signal dashboard
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import aiohttp
from nicegui import ui, events
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class EnhancedProfessionalTradingAnalytics:
    """Enhanced Professional Trading Analytics Dashboard with comprehensive data visualization"""
    
    def __init__(self, backend_url="http://localhost:8000"):
        self.backend_url = backend_url
        self.auth_token = None
        
        # Data containers
        self.market_data = {}
        self.filtered_stocks = []
        self.analytics_summary = {}
        self.current_filters = {}
        self.signals_dashboard_data = {}
        self.comprehensive_analytics = {}
        
        # UI state
        self.current_view = "signals_dashboard"
        self.auto_refresh = False
        self.refresh_interval = 30  # seconds
        self.stat_cards = []
        self.selected_stock = None
        
        # Data tables
        self.agg_data = []
        self.eod_data = []
        
    def set_auth_token(self, token):
        """Set authentication token"""
        self.auth_token = token
    
    async def _api_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make authenticated API request with error handling"""
        if not self.auth_token:
            ui.notify("Authentication required", type="negative")
            return {}
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(f"{self.backend_url}{endpoint}", headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_msg = f"API Error {response.status}: {await response.text()}"
                            ui.notify(error_msg, type="negative")
                            return {}
                elif method.upper() == "POST":
                    async with session.post(f"{self.backend_url}{endpoint}", json=data, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_msg = f"API Error {response.status}: {await response.text()}"
                            ui.notify(error_msg, type="negative")
                            return {}
        except Exception as e:
            ui.notify(f"Request failed: {str(e)}", type="negative")
            return {}
    
    def create_main_interface(self):
        """Create the enhanced main professional analytics interface"""
        with ui.column().classes("w-full min-h-screen page-container"):
            self.create_enhanced_header()
            self.create_enhanced_navigation()
            self.create_main_content_area()
            
            # Initialize data loading
            ui.timer(2.0, self.initialize_data, once=True)
    
    def create_enhanced_header(self):
        """Create enhanced professional trading header with comprehensive live stats"""
        with ui.card().classes("w-full mb-4 enhanced-card"):
            with ui.row().classes("w-full items-center justify-between p-6"):
                # Title and Status
                with ui.column().classes("flex-none"):
                    ui.label("Enhanced Professional Trading Analytics").classes("text-3xl font-bold text-primary")
                    ui.label("Comprehensive Market Intelligence & Advanced Signal Analysis").classes("text-secondary text-lg")
                    with ui.row().classes("items-center gap-4 mt-2"):
                        self.market_status = ui.label("Market Status: Loading...").classes("text-success font-medium")
                        ui.separator().props("vertical inset")
                        self.last_update = ui.label(f"Last Update: {datetime.now().strftime('%H:%M:%S')}").classes("text-secondary text-sm")
                
                # Enhanced Live Market Stats Cards
                with ui.row().classes("gap-4 flex-none"):
                    self.create_enhanced_stat_card("Total Stocks", "Loading...", "assessment", "text-info")
                    self.create_enhanced_stat_card("Active Signals", "Loading...", "signal_cellular_alt", "text-success")
                    self.create_enhanced_stat_card("Volume Surge", "Loading...", "trending_up", "text-warning") 
                    self.create_enhanced_stat_card("Breakouts", "Loading...", "north_east", "text-purple")
                    self.create_enhanced_stat_card("Market Sentiment", "Loading...", "sentiment_satisfied", "text-info")
                
                # Enhanced Action Controls
                with ui.row().classes("gap-2 flex-none"):
                    ui.button("", icon="refresh", on_click=self.refresh_all_data).props("round dense").tooltip("Refresh All Data")
                    ui.button("", icon="download", on_click=self.export_data).props("round dense").tooltip("Export Data")
                    ui.button("", icon="settings", on_click=self.show_enhanced_settings).props("round dense").tooltip("Settings")
                    
                    # Auto-refresh toggle
                    with ui.row().classes("items-center gap-2"):
                        ui.label("Auto-refresh:")
                        ui.switch(on_change=self.toggle_auto_refresh).bind_value_from(self, 'auto_refresh')
    
    def create_enhanced_stat_card(self, title: str, value: str, icon: str, color_class: str):
        """Create an enhanced professional stat card with trend indicators"""
        with ui.card().classes("metric-card-modern min-w-40"):
            with ui.column().classes("w-full"):
                with ui.row().classes("items-center justify-between mb-2"):
                    ui.icon(icon).classes(f"{color_class} text-2xl")
                    ui.icon("trending_up").classes("text-success text-sm")  # Trend indicator
                
                ui.label(title).classes("text-secondary text-sm font-medium")
                value_label = ui.label(value).classes("text-primary font-bold text-xl")
                ui.label("vs yesterday").classes("text-secondary text-xs")
                
                # Store reference for updates
                if not hasattr(self, 'stat_cards'):
                    self.stat_cards = []
                
                card_type = ('total_stocks' if 'Total' in title else 
                           'signals' if 'Signal' in title else 
                           'volume' if 'Volume' in title else 
                           'breakouts' if 'Breakout' in title else 
                           'sentiment')
                self.stat_cards.append({'type': card_type, 'label': value_label})
    
    def create_enhanced_navigation(self):
        """Create enhanced navigation tabs for different analytics views"""
        with ui.card().classes("w-full mb-4 enhanced-card"):
            with ui.row().classes("w-full p-4 gap-2"):
                self.nav_buttons = {}
                
                self.nav_buttons["signals_dashboard"] = ui.button("Signals Dashboard", icon="dashboard", 
                         on_click=lambda: self.switch_view("signals_dashboard")).classes("chart-type-btn")
                self.nav_buttons["comprehensive_analytics"] = ui.button("Market Analytics", icon="analytics", 
                         on_click=lambda: self.switch_view("comprehensive_analytics")).classes("chart-type-btn")
                self.nav_buttons["data_explorer"] = ui.button("Data Explorer", icon="table_view", 
                         on_click=lambda: self.switch_view("data_explorer")).classes("chart-type-btn")
                self.nav_buttons["stock_analysis"] = ui.button("Stock Analysis", icon="show_chart", 
                         on_click=lambda: self.switch_view("stock_analysis")).classes("chart-type-btn")
                self.nav_buttons["advanced_scanner"] = ui.button("Advanced Scanner", icon="search", 
                         on_click=lambda: self.switch_view("advanced_scanner")).classes("chart-type-btn")
                
                # Set initial active button
                self.update_nav_active()
    
    def create_main_content_area(self):
        """Create the main content area that switches based on selected view"""
        self.content_container = ui.column().classes("w-full flex-1")
        self.switch_view(self.current_view)
    
    def switch_view(self, view_name: str):
        """Switch to different analytics view"""
        self.current_view = view_name
        self.content_container.clear()
        self.update_nav_active()
        
        with self.content_container:
            if view_name == "signals_dashboard":
                self.create_signals_dashboard_view()
            elif view_name == "comprehensive_analytics":
                self.create_comprehensive_analytics_view()
            elif view_name == "data_explorer":
                self.create_data_explorer_view()
            elif view_name == "stock_analysis":
                self.create_stock_analysis_view()
            elif view_name == "advanced_scanner":
                self.create_advanced_scanner_view()
    
    def update_nav_active(self):
        """Update navigation button active states"""
        if hasattr(self, 'nav_buttons'):
            for view_name, button in self.nav_buttons.items():
                if view_name == self.current_view:
                    button.classes("chart-type-btn active", remove="chart-type-btn")
                else:
                    button.classes("chart-type-btn", remove="chart-type-btn active")
    
    def create_signals_dashboard_view(self):
        """Create comprehensive signals dashboard with categorized stocks"""
        with ui.column().classes("w-full gap-4"):
            # Quick Stats Row
            with ui.row().classes("w-full gap-4 mb-4"):
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Bullish Momentum").classes("text-secondary mb-2")
                    self.bullish_count = ui.label("0").classes("text-2xl font-bold text-success")
                    ui.label("EMA + Volume + Regression").classes("text-sm text-secondary")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Volume Breakouts").classes("text-secondary mb-2")
                    self.volume_breakouts_count = ui.label("0").classes("text-2xl font-bold text-warning")
                    ui.label("20-day breakouts with volume").classes("text-sm text-secondary")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Support/Resistance").classes("text-secondary mb-2")
                    self.sr_breaks_count = ui.label("0").classes("text-2xl font-bold text-info")
                    ui.label("Key level signals").classes("text-sm text-secondary")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Narrow Range").classes("text-secondary mb-2")
                    self.narrow_range_count = ui.label("0").classes("text-2xl font-bold text-purple")
                    ui.label("Consolidation breakouts").classes("text-sm text-secondary")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Regression Signals").classes("text-secondary mb-2")
                    self.regression_count = ui.label("0").classes("text-2xl font-bold text-cyan")
                    ui.label("Momentum cross signals").classes("text-sm text-secondary")
            
            # Signals Dashboard Content
            with ui.card().classes("enhanced-card w-full flex-1"):
                ui.label("Live Trading Signals Dashboard").classes("text-xl font-bold mb-4 text-primary")
                self.signals_content_container = ui.column().classes("w-full")
                with self.signals_content_container:
                    ui.label("Loading signals dashboard...").classes("text-center text-secondary p-8")
        
        # Auto-load signals dashboard when view is created
        ui.timer(1.0, self.load_signals_dashboard, once=True)
    
    def create_comprehensive_analytics_view(self):
        """Create comprehensive market analytics view with charts and metrics"""
        with ui.column().classes("w-full gap-4"):
            # Timeframe Selection
            with ui.card().classes("enhanced-card"):
                with ui.row().classes("items-center gap-4 p-4"):
                    ui.label("Market Analytics Timeframe:").classes("font-semibold")
                    self.timeframe_select = ui.select(
                        ["daily", "weekly", "monthly"], 
                        value="daily",
                        on_change=self.update_comprehensive_analytics
                    ).classes("w-32")
                    ui.button("Refresh Analytics", on_click=self.update_comprehensive_analytics, 
                             icon="refresh", color="primary").classes("ml-4")
            
            # Analytics Content
            with ui.row().classes("w-full gap-4"):
                # Market Breadth
                with ui.column().classes("w-1/2"):
                    with ui.card().classes("enhanced-card"):
                        ui.label("Market Breadth Analysis").classes("text-lg font-bold mb-4 text-primary")
                        self.market_breadth_container = ui.column().classes("w-full")
                        with self.market_breadth_container:
                            ui.label("Loading market breadth...").classes("text-center text-secondary p-4")
                
                # Technical Summary
                with ui.column().classes("w-1/2"):
                    with ui.card().classes("enhanced-card"):
                        ui.label("Technical Summary").classes("text-lg font-bold mb-4 text-primary")
                        self.technical_summary_container = ui.column().classes("w-full")
                        with self.technical_summary_container:
                            ui.label("Loading technical summary...").classes("text-center text-secondary p-4")
            
            # Performance Metrics and Charts
            with ui.card().classes("enhanced-card w-full"):
                ui.label("Performance Metrics & Visualization").classes("text-lg font-bold mb-4 text-primary")
                self.performance_charts_container = ui.column().classes("w-full")
                with self.performance_charts_container:
                    ui.label("Loading performance charts...").classes("text-center text-secondary p-8")
    
    def create_data_explorer_view(self):
        """Create comprehensive data explorer with full AGG_DATA and EOD_Summary tables"""
        with ui.column().classes("w-full gap-4"):
            # Data Source Selection
            with ui.card().classes("enhanced-card"):
                with ui.row().classes("items-center gap-4 p-4"):
                    ui.label("Data Source:").classes("font-semibold")
                    self.data_source_select = ui.select(
                        {"AGG_DATA (Aggregated Stock Data)": "agg",
                         "EOD_Summary (Technical Analysis)": "eod"}, 
                        value="AGG_DATA (Aggregated Stock Data)",
                        on_change=self.load_selected_data_table
                    ).classes("w-64")
                    
                    ui.label("Sort by:").classes("font-semibold ml-4")
                    self.sort_column_select = ui.select(
                        ["Symbol", "close", "volume", "Pct_Chg_D"] if self.data_source_select.value == 'agg' else
                        ["Symbol", "close", "volume", "Pct_Chg"],
                        value="Symbol",
                        on_change=self.load_selected_data_table
                    ).classes("w-32")
                    
                    self.sort_order_select = ui.select(
                        {"Ascending": "asc", "Descending": "desc"}, 
                        value="Ascending",
                        on_change=self.load_selected_data_table
                    ).classes("w-32")
                    
                    ui.button("Load Data", on_click=self.load_selected_data_table, 
                             icon="table_view", color="primary").classes("ml-4")
            
            # Data Table
            with ui.card().classes("enhanced-card w-full flex-1"):
                ui.label("Data Explorer").classes("text-xl font-bold mb-4 text-primary")
                self.data_table_container = ui.column().classes("w-full")
                with self.data_table_container:
                    ui.label("Select data source and click Load Data").classes("text-center text-secondary p-8")
            
            # Auto-load data when view is created
            ui.timer(1.0, self.load_selected_data_table, once=True)
    
    def create_stock_analysis_view(self):
        """Create individual stock analysis view"""
        with ui.row().classes("w-full gap-4"):
            # Stock Selection
            with ui.column().classes("w-1/3"):
                with ui.card().classes("enhanced-card"):
                    ui.label("Stock Selection").classes("text-xl font-bold text-primary mb-4")
                    
                    # Stock search/select
                    self.stock_search = ui.input("Search Stock Symbol", 
                                               placeholder="Enter symbol (e.g., RELIANCE)",
                                               on_change=self.search_stocks).classes("w-full mb-4")
                    
                    self.stock_list_container = ui.column().classes("w-full max-h-64 overflow-y-auto")
                    with self.stock_list_container:
                        ui.label("Loading stock list...").classes("text-center text-secondary p-4")
                    
                    ui.button("Analyze Selected Stock", on_click=self.analyze_selected_stock,
                             icon="analytics", color="primary").classes("w-full mt-4")
            
            # Stock Analysis Results
            with ui.column().classes("w-2/3"):
                with ui.card().classes("enhanced-card w-full flex-1"):
                    ui.label("Stock Analysis Results").classes("text-xl font-bold mb-4 text-primary")
                    self.stock_analysis_container = ui.column().classes("w-full")
                    with self.stock_analysis_container:
                        ui.label("Select a stock to view comprehensive analysis").classes("text-center text-secondary p-8")
        
        # Auto-load stock list when view is created
        ui.timer(1.0, self.load_stock_list, once=True)
    
    def create_advanced_scanner_view(self):
        """Create advanced scanner with comprehensive filtering options"""
        with ui.row().classes("w-full gap-4"):
            # Advanced Filters Panel
            with ui.column().classes("w-1/3"):
                with ui.card().classes("enhanced-card"):
                    ui.label("Advanced Scanner Filters").classes("text-xl font-bold text-primary mb-4")
                    
                    # Signal-Based Filters (Based on EOD_analysis logic)
                    with ui.card().classes("metric-card p-4 mb-4"):
                        ui.label("Signal-Based Filters").classes("font-semibold mb-2 text-secondary")
                        ui.checkbox("Bullish EMA Alignment", on_change=lambda e: self.update_filter('bullish_ema_alignment', e.value))
                        ui.checkbox("EMA Cross Above Signals", on_change=lambda e: self.update_filter('ema_cross_above', e.value))
                        ui.checkbox("Regression Cross Up", on_change=lambda e: self.update_filter('reg_cross_up', e.value))
                        ui.checkbox("Support Bounce Signals", on_change=lambda e: self.update_filter('support_bounce', e.value))
                        ui.checkbox("Resistance Break Signals", on_change=lambda e: self.update_filter('resistance_break', e.value))
                    
                    # Volume & Momentum Filters
                    with ui.card().classes("metric-card p-4 mb-4"):
                        ui.label("Volume & Momentum").classes("font-semibold mb-2 text-secondary")
                        ui.number("Min Volume Ratio", value=1.2, min=0.5, max=5.0, step=0.1,
                                 on_change=lambda e: self.update_filter('min_volume_ratio', e.value)).classes("w-full mb-2")
                        ui.select({"Any": "", "Above Avg Volume": "Above Avg Volume", "Super Volume": "Super Volume"}, 
                                 label="Volume Signal", on_change=lambda e: self.update_filter('volume_signal', e.value))
                        ui.number("Min Daily Change %", value=None, placeholder="Any",
                                 on_change=lambda e: self.update_filter('min_daily_change', e.value)).classes("w-full")
                    
                    # Breakout & Pattern Filters
                    with ui.card().classes("metric-card p-4 mb-4"):
                        ui.label("Breakout & Patterns").classes("font-semibold mb-2 text-secondary")
                        ui.checkbox("20-Day High Breakouts", on_change=lambda e: self.update_filter('breakout_20_up', e.value))
                        ui.checkbox("Narrow Range Breakouts", on_change=lambda e: self.update_filter('narrow_range_breakout', e.value))
                        ui.checkbox("Currently in Narrow Range", on_change=lambda e: self.update_filter('narrow_range_current', e.value))
                        ui.number("Max Range % (for NR)", value=None, min=0.1, placeholder="Any",
                                 on_change=lambda e: self.update_filter('max_range_pct', e.value)).classes("w-full")
                    
                    # Price & Risk Filters
                    with ui.card().classes("metric-card p-4 mb-4"):
                        ui.label("Price & Risk Filters").classes("font-semibold mb-2 text-secondary")
                        ui.number("Min Price (₹)", value=None, min=1, placeholder="Any", 
                                 on_change=lambda e: self.update_filter('min_price', e.value)).classes("w-full mb-2")
                        ui.number("Max Price (₹)", value=None, min=1, placeholder="Any",
                                 on_change=lambda e: self.update_filter('max_price', e.value)).classes("w-full mb-2")
                        ui.number("Min ATR", value=None, min=0.1, placeholder="Any",
                                 on_change=lambda e: self.update_filter('min_atr', e.value)).classes("w-full mb-2")
                        ui.number("Max ATR", value=None, min=0.1, placeholder="Any",
                                 on_change=lambda e: self.update_filter('max_atr', e.value)).classes("w-full")
                    
                    ui.button("Run Advanced Scan", on_click=self.run_advanced_scan,
                             icon="search", color="primary").classes("w-full font-semibold")
            
            # Scanner Results
            with ui.column().classes("w-2/3"):
                with ui.card().classes("enhanced-card w-full flex-1"):
                    ui.label("Advanced Scanner Results").classes("text-xl font-bold mb-4 text-primary")
                    self.scanner_results_container = ui.column().classes("w-full")
                    with self.scanner_results_container:
                        ui.label("Configure filters and run advanced scan").classes("text-center text-secondary p-8")
    
    # Data Loading and Processing Methods
    async def load_signals_dashboard(self):
        """Load signals dashboard data"""
        try:
            result = await self._api_request("GET", "/api/scanner/signals/dashboard")
            if result:
                self.signals_dashboard_data = result
                self.display_signals_dashboard(result)
        except Exception as e:
            logger.error(f"Error loading signals dashboard: {e}")
    
    async def load_comprehensive_analytics(self, timeframe="daily"):
        """Load comprehensive analytics data"""
        try:
            result = await self._api_request("GET", f"/api/scanner/analytics/comprehensive?timeframe={timeframe}")
            if result:
                self.comprehensive_analytics = result
                self.display_comprehensive_analytics(result)
        except Exception as e:
            logger.error(f"Error loading comprehensive analytics: {e}")
    
    async def load_selected_data_table(self):
        """Load selected data table (AGG_DATA or EOD_Summary)"""
        try:
            # Get the actual values from the select widgets
            data_source_key = self.data_source_select.value
            data_source = self.data_source_select.options[data_source_key] if data_source_key in self.data_source_select.options else "agg"
            
            sort_by = self.sort_column_select.value
            sort_order_key = self.sort_order_select.value
            sort_order = self.sort_order_select.options[sort_order_key] if sort_order_key in self.sort_order_select.options else "asc"

            endpoint = f"/api/scanner/full-data/{data_source}?sort_by={sort_by}&sort_order={sort_order}&limit=100"
            result = await self._api_request("GET", endpoint)
            
            if result and 'data' in result:
                self.display_data_table(result['data'], data_source)
            else:
                # Clear the table and show no data message
                self.data_table_container.clear()
                with self.data_table_container:
                    ui.label("No data available").classes("text-center text-secondary p-8")
        except Exception as e:
            logger.error(f"Error loading data table: {e}")
            # Clear the table and show error message
            if hasattr(self, 'data_table_container'):
                self.data_table_container.clear()
                with self.data_table_container:
                    ui.label("Error loading data").classes("text-center text-error p-8")
    
    async def load_stock_list(self):
        """Load available stocks list"""
        try:
            result = await self._api_request("GET", "/api/scanner/full-data/agg?limit=500")
            if result and 'data' in result:
                stocks = result['data']
                self.stock_list_container.clear()
                with self.stock_list_container:
                    for stock in stocks[:100]:  # Show first 100 stocks
                        symbol = stock.get('Symbol', '')
                        price = stock.get('close', 0)
                        change = stock.get('Pct_Chg_D', 0)
                        
                        with ui.card().classes("metric-card p-2 mb-1 cursor-pointer hover:bg-gray-100").on('click', 
                                             lambda s=symbol: self.select_stock_for_analysis(s)):
                            with ui.row().classes("items-center justify-between w-full"):
                                ui.label(symbol).classes("font-semibold")
                                ui.label(f"₹{price:.2f}").classes("text-sm")
                                change_class = "text-success" if change >= 0 else "text-error"
                                ui.label(f"{change:+.2f}%").classes(f"text-sm {change_class}")
                ui.notify("Stock list loaded", type="positive")
        except Exception as e:
            logger.error(f"Error loading stock list: {e}")
            ui.notify("Failed to load stock list", type="negative")
    
    async def search_stocks(self, e):
        """Search for stocks based on input"""
        search_term = e.value.upper() if e.value else ""
        
        if len(search_term) >= 2:
            # Get stock list from backend
            result = await self._api_request("GET", f"/api/scanner/full-data/agg?limit=500")
            if result and 'data' in result:
                stocks = result['data']
                filtered_stocks = [stock for stock in stocks if search_term in stock.get('Symbol', '')]
                
                self.stock_list_container.clear()
                with self.stock_list_container:
                    for stock in filtered_stocks[:20]:  # Limit to 20 results
                        symbol = stock.get('Symbol', '')
                        price = stock.get('close', 0)
                        change = stock.get('Pct_Chg_D', 0)
                        
                        with ui.card().classes("metric-card p-2 mb-1 cursor-pointer hover:bg-gray-100").on('click', 
                                             lambda s=symbol: self.select_stock_for_analysis(s)):
                            with ui.row().classes("items-center justify-between w-full"):
                                ui.label(symbol).classes("font-semibold")
                                ui.label(f"₹{price:.2f}").classes("text-sm")
                                change_class = "text-success" if change >= 0 else "text-error"
                                ui.label(f"{change:+.2f}%").classes(f"text-sm {change_class}")
        else:
            self.stock_list_container.clear()
            with self.stock_list_container:
                ui.label("Enter at least 2 characters to search").classes("text-center text-secondary p-4")
    
    def select_stock_for_analysis(self, symbol):
        """Select a stock for analysis"""
        self.selected_stock = symbol
        self.stock_search.value = symbol
        ui.notify(f"Selected {symbol} for analysis", type="positive")
    
    async def analyze_selected_stock(self):
        """Analyze the selected stock"""
        if not hasattr(self, 'selected_stock') or not self.selected_stock:
            if hasattr(self, 'stock_search') and self.stock_search.value:
                self.selected_stock = self.stock_search.value.upper()
            else:
                ui.notify("Please select or enter a stock symbol", type="warning")
                return
        
        await self.load_stock_analysis(self.selected_stock)
    
    async def load_stock_analysis(self, symbol):
        """Load comprehensive stock analysis"""
        ui.notify(f"Loading analysis for {symbol}...", type="info")
        
        try:
            # Get stock data from both tables
            agg_endpoint = f"/api/scanner/full-data/agg?symbol={symbol}"
            eod_endpoint = f"/api/scanner/full-data/eod?symbol={symbol}"
            
            agg_result = await self._api_request("GET", agg_endpoint)
            eod_result = await self._api_request("GET", eod_endpoint)
            
            self.display_stock_analysis(symbol, agg_result, eod_result)
        except Exception as e:
            logger.error(f"Error loading stock analysis: {e}")
            ui.notify("Failed to load stock analysis", type="negative")
    
    def display_stock_analysis(self, symbol, agg_data, eod_data):
        """Display comprehensive stock analysis"""
        self.stock_analysis_container.clear()
        
        if not agg_data or not eod_data or 'data' not in agg_data or 'data' not in eod_data:
            with self.stock_analysis_container:
                ui.label(f"No data found for {symbol}").classes("text-center text-secondary p-4")
            return
        
        agg_stock = agg_data['data'][0] if agg_data['data'] else {}
        eod_stock = eod_data['data'][0] if eod_data['data'] else {}
        
        with self.stock_analysis_container:
            # Stock Header
            with ui.card().classes("metric-card p-4 mb-4"):
                with ui.row().classes("items-center justify-between"):
                    ui.label(symbol).classes("text-2xl font-bold text-primary")
                    ui.label(f"₹{agg_stock.get('close', 0):.2f}").classes("text-xl font-semibold")
                    change = agg_stock.get('Pct_Chg_D', 0)
                    change_class = "text-success" if change >= 0 else "text-error"
                    ui.label(f"{change:+.2f}%").classes(f"text-lg font-semibold {change_class}")
            
            # Key Metrics
            with ui.row().classes("gap-4 mb-4"):
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Volume Ratio").classes("text-secondary mb-1")
                    vol_ratio = eod_stock.get('Vol_Abv_Avg20', 0)
                    vol_class = "text-success" if vol_ratio > 1.5 else "text-warning" if vol_ratio > 1.2 else "text-secondary"
                    ui.label(f"{vol_ratio:.2f}x").classes(f"text-xl font-bold {vol_class}")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("ATR").classes("text-secondary mb-1")
                    ui.label(f"{eod_stock.get('ATR', 0):.2f}").classes("text-xl font-bold text-primary")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Range %").classes("text-secondary mb-1")
                    ui.label(f"{eod_stock.get('Range_Pct', 0):.1f}%").classes("text-xl font-bold text-primary")
            
            # Technical Signals
            with ui.card().classes("enhanced-card mb-4"):
                ui.label("Technical Signals").classes("text-lg font-bold mb-3 text-primary")
                
                with ui.row().classes("gap-4"):
                    # EMA Signals
                    with ui.column().classes("flex-1"):
                        ui.label("EMA Signals").classes("font-semibold mb-2 text-secondary")
                        ui.label(f"EMA 20: {eod_stock.get('EMA20_Sig', 'N/A')}").classes("text-sm mb-1")
                        ui.label(f"EMA 200: {eod_stock.get('EMA200_Sig', 'N/A')}").classes("text-sm mb-1")
                    
                    # Regression Signals
                    with ui.column().classes("flex-1"):
                        ui.label("Regression Signals").classes("font-semibold mb-2 text-secondary")
                        ui.label(f"Reg Cross: {eod_stock.get('Reg_Cross_Sig', 'N/A')}").classes("text-sm mb-1")
                        ui.label(f"Reg6 Sig: {eod_stock.get('Reg6_Sig', 'N/A')}").classes("text-sm mb-1")
                    
                    # Support/Resistance
                    with ui.column().classes("flex-1"):
                        ui.label("Support/Resistance").classes("font-semibold mb-2 text-secondary")
                        ui.label(f"Support: {eod_stock.get('Support', 'N/A')}").classes("text-sm mb-1")
                        ui.label(f"Resistance: {eod_stock.get('Resistance', 'N/A')}").classes("text-sm mb-1")
                        ui.label(f"Break S/R: {eod_stock.get('Break_Sup_Res', 'N/A')}").classes("text-sm mb-1")
            
            # Breakout Analysis
            if eod_stock.get('Breakout_20') or eod_stock.get('Narrow_Range_Breakout'):
                with ui.card().classes("enhanced-card"):
                    ui.label("Breakout Analysis").classes("text-lg font-bold mb-3 text-primary")
                    
                    if eod_stock.get('Breakout_20'):
                        ui.label(f"20-Day Breakout: {eod_stock.get('Breakout_20')}").classes("text-sm mb-1 text-success")
                    
                    if eod_stock.get('Narrow_Range_Breakout'):
                        ui.label("Narrow Range Breakout: Yes").classes("text-sm mb-1 text-warning")
                    
                    if eod_stock.get('Narrow_Range'):
                        ui.label("Currently in Narrow Range").classes("text-sm mb-1 text-info")
    
    # Display Methods
    def display_signals_dashboard(self, data):
        """Display signals dashboard with categorized stocks"""
        self.signals_content_container.clear()
        
        if not data or 'categories' not in data:
            with self.signals_content_container:
                ui.label("No signals data available").classes("text-center text-secondary p-4")
            return
        
        # Update category counts
        categories = data.get('categories', {})
        if 'bullish_momentum' in categories:
            self.bullish_count.text = str(categories['bullish_momentum'].get('count', 0))
        if 'volume_breakouts' in categories:
            self.volume_breakouts_count.text = str(categories['volume_breakouts'].get('count', 0))
        if 'support_resistance_signals' in categories:
            self.sr_breaks_count.text = str(categories['support_resistance_signals'].get('count', 0))
        if 'narrow_range_breakouts' in categories:
            self.narrow_range_count.text = str(categories['narrow_range_breakouts'].get('count', 0))
        if 'regression_signals' in categories:
            self.regression_count.text = str(categories['regression_signals'].get('count', 0))
        
        # Display categories in sidebar
        with self.signal_categories_container:
            for category_key, category_data in categories.items():
                with ui.card().classes("metric-card p-3 mb-2 cursor-pointer hover:bg-gray-100"):
                    ui.label(category_data['name']).classes("font-semibold text-primary")
                    ui.label(f"{category_data['count']} stocks").classes("text-sm text-secondary")
                    ui.label(category_data['description']).classes("text-xs text-secondary")
        
        # Display main signals content
        with self.signals_content_container:
            for category_key, category_data in categories.items():
                with ui.card().classes("metric-card p-4 mb-4"):
                    ui.label(category_data['name']).classes("text-lg font-bold text-primary mb-2")
                    ui.label(category_data['description']).classes("text-secondary mb-3")
                    
                    if category_data.get('stocks'):
                        # Create table for stocks in this category
                        with ui.row().classes("w-full font-bold text-secondary border-b pb-2 mb-2"):
                            ui.label("Symbol").classes("w-20")
                            ui.label("Price").classes("w-20")
                            ui.label("Change %").classes("w-20")
                            ui.label("Volume Ratio").classes("w-24")
                            ui.label("Signal").classes("flex-1")
                        
                        for stock in category_data['stocks'][:5]:  # Show top 5
                            with ui.row().classes("w-full py-1"):
                                ui.label(stock['symbol']).classes("w-20 font-medium")
                                ui.label(f"₹{stock['price']:.2f}").classes("w-20")
                                
                                change = stock['change']
                                change_class = "text-success" if change >= 0 else "text-error"
                                ui.label(f"{change:+.2f}%").classes(f"w-20 {change_class}")
                                
                                ui.label(f"{stock['volume_ratio']:.2f}x").classes("w-24")
                                ui.label(stock['signal']).classes("flex-1 text-sm")
                    else:
                        ui.label("No stocks found in this category").classes("text-secondary text-sm")
    
    def display_comprehensive_analytics(self, data):
        """Display comprehensive analytics with charts and metrics"""
        self.market_breadth_container.clear()
        self.technical_summary_container.clear()
        self.performance_charts_container.clear()
        
        if not data:
            return
        
        # Market Breadth
        with self.market_breadth_container:
            breadth = data.get('market_breadth', {})
            
            with ui.row().classes("gap-4 mb-4"):
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Total Stocks").classes("text-secondary mb-1")
                    ui.label(str(breadth.get('total_stocks', 0))).classes("text-xl font-bold text-primary")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Gainers").classes("text-secondary mb-1")
                    ui.label(f"{breadth.get('gainers', 0)} ({breadth.get('gainers_pct', 0)}%)").classes("text-xl font-bold text-success")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Losers").classes("text-secondary mb-1")
                    ui.label(f"{breadth.get('losers', 0)} ({breadth.get('losers_pct', 0)}%)").classes("text-xl font-bold text-error")
            
            # Advance/Decline Ratio
            ui.label(f"Advance/Decline Ratio: {breadth.get('advance_decline_ratio', 0):.2f}").classes("text-lg font-semibold")
        
        # Technical Summary
        with self.technical_summary_container:
            technical = data.get('technical_summary', {})
            
            with ui.row().classes("gap-4 mb-4"):
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Above EMA 200").classes("text-secondary mb-1")
                    ui.label(f"{technical.get('above_ema200', 0)} ({technical.get('above_ema200_pct', 0)}%)").classes("text-xl font-bold text-success")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("High Volume").classes("text-secondary mb-1")
                    ui.label(str(technical.get('high_volume_stocks', 0))).classes("text-xl font-bold text-warning")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("High Volatility").classes("text-secondary mb-1")
                    ui.label(str(technical.get('high_volatility_stocks', 0))).classes("text-xl font-bold text-info")
        
        # Performance Charts
        with self.performance_charts_container:
            performance = data.get('performance_metrics', {})
            signals = data.get('signal_summary', {})
            sentiment = data.get('market_sentiment', {})
            
            # Market Sentiment
            with ui.card().classes("metric-card p-4 mb-4"):
                ui.label("Market Sentiment Analysis").classes("text-lg font-bold mb-2")
                sentiment_color = ("text-success" if sentiment.get('score', 0) > 2 else 
                                 "text-warning" if sentiment.get('score', 0) > 0 else "text-error")
                ui.label(f"{sentiment.get('sentiment', 'Neutral')} (Score: {sentiment.get('score', 0)})").classes(f"text-xl font-bold {sentiment_color}")
                ui.label(f"Confidence: {sentiment.get('confidence', 'Medium')}").classes("text-secondary")
            
            # Performance Metrics
            with ui.row().classes("gap-4"):
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Avg Change").classes("text-secondary mb-1")
                    change = performance.get('avg_change', 0)
                    change_color = "text-success" if change >= 0 else "text-error"
                    ui.label(f"{change:+.2f}%").classes(f"text-lg font-bold {change_color}")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Market Volatility").classes("text-secondary mb-1")
                    ui.label(f"{performance.get('market_volatility', 0):.2f}%").classes("text-lg font-bold text-primary")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Net Breakouts").classes("text-secondary mb-1")
                    net_breakouts = signals.get('net_breakouts', 0)
                    breakout_color = "text-success" if net_breakouts > 0 else "text-error" if net_breakouts < 0 else "text-secondary"
                    ui.label(str(net_breakouts)).classes(f"text-lg font-bold {breakout_color}")
    
    def display_data_table(self, data, data_source):
        """Display data table with pagination and sorting"""
        self.data_table_container.clear()
        
        if not data:
            with self.data_table_container:
                ui.label("No data available").classes("text-center text-secondary p-4")
            return
        
        with self.data_table_container:
            ui.label(f"Data Source: {data_source.upper()}").classes("text-lg font-semibold mb-4")
            ui.label(f"Showing {len(data)} records").classes("text-secondary mb-4")
            
            # Create scrollable table
            with ui.scroll_area().classes("w-full h-96"):
                if data:
                    # Get column names from first row
                    columns = list(data[0].keys())
                    
                    # Header row
                    with ui.row().classes("w-full font-bold text-secondary border-b pb-2 mb-2"):
                        for col in columns[:10]:  # Show first 10 columns
                            ui.label(col).classes("min-w-24 text-xs")
                    
                    # Data rows
                    for i, row in enumerate(data[:50]):  # Show first 50 rows
                        bg_class = "bg-gray-50" if i % 2 == 0 else ""
                        with ui.row().classes(f"w-full py-1 {bg_class}"):
                            for col in columns[:10]:
                                value = row.get(col, '')
                                if isinstance(value, (int, float)) and value is not None:
                                    if col in ['close', 'high', 'low', 'open']:
                                        display_value = f"₹{value:.2f}"
                                    elif 'Pct_Chg' in col or 'Percent_Chg' in col:
                                        color_class = "text-success" if value >= 0 else "text-error"
                                        display_value = f"{value:+.2f}%"
                                        ui.label(display_value).classes(f"min-w-24 text-xs {color_class}")
                                        continue
                                    else:
                                        display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                                else:
                                    display_value = str(value) if value is not None else ''
                                
                                ui.label(display_value).classes("min-w-24 text-xs")
    
    def display_stock_list(self, stocks):
        """Display searchable stock list"""
        self.stock_list_container.clear()
        
        with self.stock_list_container:
            for stock in stocks[:20]:  # Show first 20 stocks
                with ui.card().classes("metric-card p-2 mb-1 cursor-pointer hover:bg-gray-100").on('click', 
                                     lambda s=stock['symbol']: self.select_stock(s)):
                    with ui.row().classes("items-center justify-between"):
                        with ui.column():
                            ui.label(stock['symbol']).classes("font-semibold")
                            ui.label(f"₹{stock['current_price']:.2f}").classes("text-sm text-secondary")
                        
                        change = stock['daily_change']
                        change_color = "text-success" if change >= 0 else "text-error"
                        ui.label(f"{change:+.2f}%").classes(f"text-sm {change_color}")
    
    # Event Handlers
    def update_filter(self, key: str, value):
        """Update filter criteria"""
        self.current_filters[key] = value
        logger.info(f"Updated filter: {key} = {value}")
    
    def select_stock(self, symbol):
        """Select stock for analysis"""
        self.selected_stock = symbol
        ui.notify(f"Selected {symbol} for analysis", type="positive")
    
    async def analyze_selected_stock(self):
        """Analyze the selected stock"""
        if not self.selected_stock:
            ui.notify("Please select a stock first", type="warning")
            return
        
        try:
            result = await self._api_request("GET", f"/api/scanner/stock/{self.selected_stock}/analysis")
            if result:
                self.display_stock_analysis(result)
        except Exception as e:
            logger.error(f"Error analyzing stock: {e}")
    
    def display_stock_analysis(self, analysis):
        """Display comprehensive stock analysis"""
        self.stock_analysis_container.clear()
        
        with self.stock_analysis_container:
            basic_info = analysis.get('basic_info', {})
            performance = analysis.get('performance', {})
            technical = analysis.get('technical_indicators', {})
            trading_summary = analysis.get('trading_summary', {})
            
            # Basic Info
            ui.label(f"Analysis for {basic_info.get('symbol', 'N/A')}").classes("text-2xl font-bold text-primary mb-4")
            
            # Key Metrics Row
            with ui.row().classes("gap-4 mb-4"):
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Current Price").classes("text-secondary mb-1")
                    ui.label(f"₹{basic_info.get('current_price', 0):.2f}").classes("text-xl font-bold text-primary")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Daily Change").classes("text-secondary mb-1")
                    daily_change = performance.get('daily_change', 0)
                    change_color = "text-success" if daily_change >= 0 else "text-error"
                    ui.label(f"{daily_change:+.2f}%").classes(f"text-xl font-bold {change_color}")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Signal Strength").classes("text-secondary mb-1")
                    signal_strength = trading_summary.get('signal_strength', 0)
                    strength_color = "text-success" if signal_strength > 2 else "text-warning" if signal_strength > 0 else "text-error"
                    ui.label(f"{signal_strength}/10").classes(f"text-xl font-bold {strength_color}")
                
                with ui.card().classes("metric-card-modern flex-1"):
                    ui.label("Recommendation").classes("text-secondary mb-1")
                    recommendation = trading_summary.get('recommendation', 'Hold')
                    rec_color = ("text-success" if recommendation in ["Strong Buy", "Buy"] else 
                               "text-error" if recommendation in ["Strong Sell", "Sell"] else "text-secondary")
                    ui.label(recommendation).classes(f"text-lg font-bold {rec_color}")
            
            # Trading Signals
            with ui.card().classes("metric-card p-4 mb-4"):
                ui.label("Active Trading Signals").classes("text-lg font-bold mb-2")
                signals = trading_summary.get('signals', [])
                if signals:
                    for signal in signals:
                        ui.label(f"• {signal}").classes("text-secondary mb-1")
                else:
                    ui.label("No active signals").classes("text-secondary")
            
            # Technical Indicators
            with ui.card().classes("metric-card p-4"):
                ui.label("Technical Indicators").classes("text-lg font-bold mb-2")
                with ui.row().classes("gap-4"):
                    ui.label(f"EMA 20: ₹{technical.get('ema_20', 0):.2f}").classes("text-secondary")
                    ui.label(f"EMA 60: ₹{technical.get('ema_60', 0):.2f}").classes("text-secondary")
                    ui.label(f"EMA 200: ₹{technical.get('ema_200', 0):.2f}").classes("text-secondary")
                    ui.label(f"ATR: {technical.get('atr', 0):.2f}").classes("text-secondary")
    
    async def run_advanced_scan(self):
        """Run advanced stock scan with current filters"""
        try:
            # Remove None values from filters
            scan_data = {k: v for k, v in self.current_filters.items() if v is not None}
            scan_data.update({
                "sort_by": "signal_strength",
                "sort_order": "desc",
                "limit": 50
            })
            
            result = await self._api_request("POST", "/api/scanner/advanced-scan", scan_data)
            if result:
                self.display_scanner_results(result)
        except Exception as e:
            logger.error(f"Error running advanced scan: {e}")
    
    def display_scanner_results(self, result):
        """Display advanced scanner results"""
        self.scanner_results_container.clear()
        
        with self.scanner_results_container:
            if not result or 'stocks' not in result:
                ui.label("No results found").classes("text-center text-secondary p-4")
                return
            
            stocks = result.get('stocks', [])
            analytics = result.get('analytics', {})
            
            # Summary
            ui.label(f"Found {len(stocks)} stocks matching criteria").classes("text-lg font-semibold mb-4")
            
            if analytics:
                with ui.row().classes("gap-4 mb-4"):
                    ui.label(f"Avg Signal Strength: {analytics.get('avg_signal_strength', 0):.1f}").classes("text-secondary")
                    ui.label(f"Strong Buy: {analytics.get('strong_buy_count', 0)}").classes("text-success")
                    ui.label(f"Buy: {analytics.get('buy_count', 0)}").classes("text-info")
            
            # Results table
            if stocks:
                with ui.scroll_area().classes("w-full h-96"):
                    # Header
                    with ui.row().classes("w-full font-bold text-secondary border-b pb-2 mb-2"):
                        ui.label("Symbol").classes("w-24")
                        ui.label("Price").classes("w-20")
                        ui.label("Change %").classes("w-20")
                        ui.label("Volume").classes("w-24")
                        ui.label("Signal").classes("w-20")
                        ui.label("Recommendation").classes("flex-1")
                    
                    # Data rows
                    for i, stock in enumerate(stocks[:30]):
                        bg_class = "bg-gray-50" if i % 2 == 0 else ""
                        with ui.row().classes(f"w-full py-1 {bg_class}"):
                            ui.label(stock.get('symbol', 'N/A')).classes("w-24 font-medium")
                            ui.label(f"₹{stock.get('current_price', 0):.2f}").classes("w-20")
                            
                            change = stock.get('daily_change', 0)
                            change_class = "text-success" if change >= 0 else "text-error"
                            ui.label(f"{change:+.2f}%").classes(f"w-20 {change_class}")
                            
                            volume = stock.get('volume_vs_avg', 0)
                            ui.label(f"{volume:.2f}x").classes("w-24")
                            
                            signal = stock.get('signal_strength', 0)
                            signal_class = "text-success" if signal >= 7 else "text-warning" if signal >= 5 else "text-secondary"
                            ui.label(f"{signal:.1f}").classes(f"w-20 {signal_class}")
                            
                            recommendation = stock.get('recommendation', 'Hold')
                            rec_class = ("text-success" if recommendation in ["Strong Buy", "Buy"] else 
                                       "text-error" if recommendation in ["Strong Sell", "Sell"] else "text-secondary")
                            ui.label(recommendation).classes(f"flex-1 {rec_class}")
    
    async def search_stocks(self):
        """Search stocks based on input"""
        search_term = self.stock_search.value
        if len(search_term) >= 2:
            await self.load_stock_list()
    
    async def update_comprehensive_analytics(self):
        """Update comprehensive analytics based on selected timeframe"""
        timeframe = self.timeframe_select.value
        await self.load_comprehensive_analytics(timeframe)
    
    async def refresh_all_data(self):
        """Refresh all data based on current view"""
        ui.notify("Refreshing data...", type="info")
        self.last_update.text = f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
        
        if self.current_view == "signals_dashboard":
            await self.load_signals_dashboard()
        elif self.current_view == "comprehensive_analytics":
            await self.load_comprehensive_analytics(self.timeframe_select.value if hasattr(self, 'timeframe_select') else "daily")
        elif self.current_view == "stock_analysis":
            await self.load_stock_list()
        
        await self.update_market_stats()
    
    async def export_data(self):
        """Export current data to CSV"""
        ui.notify("Export functionality coming soon", type="info")
    
    def show_enhanced_settings(self):
        """Show enhanced settings dialog"""
        with ui.dialog() as dialog:
            with ui.card().classes("p-6 min-w-96"):
                ui.label("Enhanced Scanner Settings").classes("text-xl font-bold mb-4")
                
                ui.number("Refresh Interval (seconds)", value=self.refresh_interval, min=10, max=300,
                         on_change=lambda e: setattr(self, 'refresh_interval', e.value)).classes("w-full mb-4")
                
                ui.checkbox("Enable notifications", value=True).classes("mb-4")
                ui.checkbox("Show advanced metrics", value=True).classes("mb-4")
                
                with ui.row().classes("gap-2 mt-4"):
                    ui.button("Save Settings", on_click=dialog.close, color="primary")
                    ui.button("Cancel", on_click=dialog.close)
        dialog.open()
    
    def toggle_auto_refresh(self, value: bool):
        """Toggle auto-refresh functionality"""
        self.auto_refresh = value
        if value:
            ui.notify("Auto-refresh enabled", type="positive")
            ui.timer(self.refresh_interval, self.refresh_all_data, active=True)
        else:
            ui.notify("Auto-refresh disabled", type="info")
    
    async def update_market_stats(self):
        """Update market statistics in header"""
        try:
            result = await self._api_request("GET", "/api/scanner/market-stats")
            if result:
                self.market_status.text = "Market Status: Open"
                # Update stat cards with real data
                for card in self.stat_cards:
                    if card['type'] == 'total_stocks':
                        card['label'].text = str(result.get('total_stocks', 0))
                    elif card['type'] == 'signals':
                        card['label'].text = str(result.get('active_signals', 0))
                    elif card['type'] == 'volume':
                        card['label'].text = str(result.get('volume_surge_count', 0))
                    elif card['type'] == 'breakouts':
                        card['label'].text = str(result.get('breakout_count', 0))
                    elif card['type'] == 'sentiment':
                        card['label'].text = "Bullish"  # This would come from comprehensive analytics
            else:
                self.market_status.text = "Market Status: Loading..."
        except Exception as e:
            logger.error(f"Error updating market stats: {e}")
            self.market_status.text = "Market Status: Error"
    
    async def initialize_data(self):
        """Initialize scanner data on page load"""
        logger.info("Initializing enhanced scanner data...")
        await self.update_market_stats()
        
        # Load data based on current view
        if self.current_view == "signals_dashboard":
            await self.load_signals_dashboard()
        elif self.current_view == "comprehensive_analytics":
            await self.load_comprehensive_analytics()
        elif self.current_view == "stock_analysis":
            await self.load_stock_list()


def create_enhanced_professional_scanner_page(auth_token: str):
    """Create the enhanced professional trading analytics page"""
    scanner = EnhancedProfessionalTradingAnalytics()
    scanner.set_auth_token(auth_token)
    
    scanner.create_main_interface()
    
    return scanner
