# frontend/theme_manager.py
"""
Enhanced Centralized Theme Manager for the Trading Application
Handles all styling, themes, and page-specific CSS from one place
"""

from nicegui import ui
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ThemeMode(Enum):
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"


class PageTheme(Enum):
    """Enum for different page theme configurations"""
    DEFAULT = "default"
    LOGIN = "login"
    DASHBOARD = "dashboard"
    TRADING = "trading"
    ANALYTICS = "analytics"
    BACKTESTING = "backtesting"
    SIP_STRATEGY = "sip_strategy"
    SETTINGS = "settings"
    ORDERBOOK = "orderbook"
    PORTFOLIO = "portfolio"
    POSITIONS = "positions"
    WATCHLIST = "watchlist"
    LIVE_TRADING = "live_trading"
    ORDER_MANAGEMENT = "order_management"


@dataclass
class ThemeConfig:
    """Configuration for a theme"""
    name: str
    mode: ThemeMode
    primary_color: str
    accent_color: str
    background: str
    surface_color: str
    text_primary: str
    text_secondary: str
    border_color: str
    hover_color: str
    glass_opacity: float = 0.08
    blur_amount: int = 20


class ThemeManager:
    """Centralized theme management system"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ThemeManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.current_theme: ThemeMode = ThemeMode.DARK
        self.themes: Dict[ThemeMode, ThemeConfig] = self._initialize_themes()
        self.page_styles: Dict[PageTheme, str] = self._initialize_page_styles()
        self.global_styles_applied = False

    def _initialize_themes(self) -> Dict[ThemeMode, ThemeConfig]:
        """Initialize available themes"""
        return {
            ThemeMode.DARK: ThemeConfig(
                name="Dark",
                mode=ThemeMode.DARK,
                primary_color="#0a0f23",
                accent_color="#22c5fc",
                background="linear-gradient(135deg, #0a0f23 0%, #1a1f3a 100%)",
                surface_color="rgba(255, 255, 255, 0.08)",
                text_primary="#ffffff",
                text_secondary="#94a3b8",
                border_color="rgba(255, 255, 255, 0.1)",
                hover_color="rgba(255, 255, 255, 0.12)",
                glass_opacity=0.08,
                blur_amount=20
            ),
            ThemeMode.LIGHT: ThemeConfig(
                name="Light",
                mode=ThemeMode.LIGHT,
                primary_color="#f8fafc",
                accent_color="#0ea5e9",
                background="linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)",
                surface_color="rgba(255, 255, 255, 0.8)",
                text_primary="#1e293b",
                text_secondary="#475569",
                border_color="rgba(0, 0, 0, 0.06)",
                hover_color="rgba(0, 0, 0, 0.08)",
                glass_opacity=0.8,
                blur_amount=10
            )
        }

    def _initialize_page_styles(self) -> Dict[PageTheme, str]:
        """Initialize page-specific styles"""
        return {
            PageTheme.LOGIN: self._get_login_styles(),
            PageTheme.DASHBOARD: self._get_dashboard_styles(),
            PageTheme.TRADING: self._get_trading_styles(),
            PageTheme.ANALYTICS: self._get_analytics_styles(),
            PageTheme.BACKTESTING: self._get_backtesting_styles(),
            PageTheme.SIP_STRATEGY: self._get_sip_strategy_styles(),
            PageTheme.SETTINGS: self._get_settings_styles(),
            PageTheme.ORDERBOOK: self._get_orderbook_styles(),
            PageTheme.PORTFOLIO: self._get_portfolio_styles(),
            PageTheme.POSITIONS: self._get_positions_styles(),
            PageTheme.WATCHLIST: self._get_watchlist_styles(),
            PageTheme.LIVE_TRADING: self._get_live_trading_styles(),
            PageTheme.ORDER_MANAGEMENT: self._get_order_management_styles(),
            PageTheme.DEFAULT: self._get_default_styles(),
        }

    def apply_theme(self, page_type: PageTheme = PageTheme.DEFAULT,
                    theme_mode: Optional[ThemeMode] = None,
                    storage: Optional[Dict] = None):
        """Apply theme to a specific page"""

        # Determine theme mode
        if theme_mode:
            self.current_theme = theme_mode
        elif storage and 'theme' in storage:
            self.current_theme = ThemeMode(storage.get('theme', 'dark').lower())

        # Apply global styles only once
        if not self.global_styles_applied:
            self._apply_global_styles()
            self.global_styles_applied = True

        # Apply theme-specific styles
        self._apply_theme_styles()

        # Apply page-specific styles
        if page_type in self.page_styles:
            ui.add_head_html(f'<style>{self.page_styles[page_type]}</style>')

        # Apply Quasar dark mode
        ui.dark_mode(self.current_theme == ThemeMode.DARK)

        logger.info(f"Applied {self.current_theme.value} theme for {page_type.value} page")

    def _apply_global_styles(self):
        """Apply global styles that are common across all pages"""
        theme = self.themes[self.current_theme]

        global_css = f'''
        /* Global Theme Variables */
        :root {{
            --primary-bg: {theme.primary_color};
            --accent-color: {theme.accent_color};
            --surface-color: {theme.surface_color};
            --text-primary: {theme.text_primary};
            --text-secondary: {theme.text_secondary};
            --border-color: {theme.border_color};
            --hover-color: {theme.hover_color};
            --glass-opacity: {theme.glass_opacity};
            --blur-amount: {theme.blur_amount}px;

            /* Additional colors */
            --success-color: #22c55e;
            --error-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #3b82f6;
            --purple-accent: #8b5cf6;

            /* Shadows */
            --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.15);
            --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.2);
            --shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.3);

            /* Border radius */
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 16px;
            --radius-full: 9999px;

            /* Transitions */
            --transition-fast: 0.15s ease;
            --transition-base: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 0.5s ease;

            /* Spacing */
            --spacing-xs: 0.25rem;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
        }}

        /* Base Styles */
        * {{
            box-sizing: border-box;
        }}

        body, .q-page {{
            background: {theme.background};
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            min-height: 100vh;
            margin: 0;
            padding: 0;
        }}

        /* Glassmorphism Card */
        .glass-card,
        .dashboard-card,
        .trading-card,
        .q-card {{
            background: var(--surface-color) !important;
            backdrop-filter: blur(var(--blur-amount));
            -webkit-backdrop-filter: blur(var(--blur-amount));
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
            transition: all var(--transition-base);
            overflow: hidden;
        }}

        .glass-card:hover,
        .dashboard-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl);
            border-color: rgba(34, 197, 252, 0.3);
        }}

        /* Compact Cards */
        .compact-card {{
            padding: var(--spacing-md) !important;
        }}

        .metric-card {{
            padding: var(--spacing-md) !important;
            text-align: center;
            min-height: auto !important;
        }}

        /* Input Fields - Universal Fix */
        .q-field__control,
        .q-input__inner,
        input, select, textarea {{
            background: var(--surface-color) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md);
            transition: all var(--transition-fast);
        }}

        .q-field__control:hover {{
            border-color: var(--hover-color) !important;
        }}

        .q-field__control:focus-within {{
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 3px rgba(34, 197, 252, 0.1);
        }}

        .q-field__label {{
            color: var(--text-secondary) !important;
        }}

        input::placeholder,
        textarea::placeholder {{
            color: var(--text-secondary) !important;
            opacity: 0.6;
        }}

        /* Buttons */
        .q-btn {{
            border-radius: var(--radius-md);
            font-weight: 500;
            transition: all var(--transition-fast);
            text-transform: none;
        }}

        .q-btn--standard {{
            background: var(--accent-color) !important;
            color: white !important;
        }}

        .q-btn--standard:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(34, 197, 252, 0.3);
        }}

        /* Enhanced Tables */
        .q-table {{
            background: var(--surface-color) !important;
            border-radius: var(--radius-lg);
            overflow: hidden;
        }}

        .q-table thead tr {{
            background: rgba(0, 0, 0, 0.2) !important;
        }}

        .q-table thead th {{
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.75rem;
            padding: 1rem;
            color: var(--text-secondary);
        }}

        .q-table tbody tr:hover {{
            background: var(--hover-color) !important;
        }}

        .q-table tbody td {{
            padding: 0.75rem 1rem;
            vertical-align: middle;
        }}

        /* Compact Table */
        .compact-table td {{
            padding: 0.5rem 0.75rem !important;
        }}

        /* Widget Alignment */
        .widget-container {{
            display: flex;
            flex-direction: column;
            gap: var(--spacing-md);
        }}

        .widget-row {{
            display: flex;
            align-items: center;
            gap: var(--spacing-md);
            flex-wrap: wrap;
        }}

        .widget-col {{
            flex: 1;
            min-width: 0;
        }}

        /* Text Utilities */
        .text-mono {{
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }}

        .text-value {{
            font-size: 1.5rem;
            font-weight: 700;
            line-height: 1.2;
        }}

        .text-label {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        /* Status Colors */
        .text-success {{ color: var(--success-color) !important; }}
        .text-error {{ color: var(--error-color) !important; }}
        .text-warning {{ color: var(--warning-color) !important; }}
        .text-info {{ color: var(--info-color) !important; }}

        .bg-success {{ background-color: var(--success-color) !important; }}
        .bg-error {{ background-color: var(--error-color) !important; }}
        .bg-warning {{ background-color: var(--warning-color) !important; }}
        .bg-info {{ background-color: var(--info-color) !important; }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: rgba(0, 0, 0, 0.1);
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: var(--hover-color);
        }}

        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-10px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}

        .fade-in {{
            animation: fadeIn 0.5s ease-out;
        }}

        .slide-in {{
            animation: slideIn 0.3s ease-out;
        }}

        /* Loading States */
        .loading-shimmer {{
            background: linear-gradient(90deg, 
                var(--surface-color) 25%, 
                var(--hover-color) 50%, 
                var(--surface-color) 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }}

        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}

        /* Enhanced Focus States */
        *:focus-visible {{
            outline: 2px solid var(--accent-color);
            outline-offset: 2px;
        }}

        /* Grid System */
        .grid-auto {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: var(--spacing-md);
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: var(--spacing-md);
        }}

        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: var(--spacing-md);
        }}

        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: var(--spacing-md);
        }}

        /* Responsive Grid */
        @media (max-width: 768px) {{
            .grid-2, .grid-3, .grid-4 {{
                grid-template-columns: 1fr;
            }}
        }}

        @media (max-width: 1024px) {{
            .grid-4 {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        '''

        ui.add_head_html(f'<style>{global_css}</style>')

        # Load external CSS file if exists
        ui.add_css('static/styles.css')

    def _apply_theme_styles(self):
        """Apply theme-specific styles"""
        theme = self.themes[self.current_theme]

        theme_css = f'''
        /* Theme-specific overrides */
        body {{
            background: {theme.background} !important;
        }}

        /* Theme-specific component styles */
        .q-dark {{
            --q-dark: {theme.background};
            --q-dark-page: {theme.primary_color};
        }}
        '''

        ui.add_head_html(f'<style>{theme_css}</style>')

    def _get_default_styles(self) -> str:
        """Default page styles"""
        return '''
        /* Default Page Styles */
        .page-container {
            padding: var(--spacing-lg);
            max-width: 1400px;
            margin: 0 auto;
        }

        .page-header {
            margin-bottom: var(--spacing-xl);
        }

        .content-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: var(--spacing-lg);
        }
        '''

    def _get_login_styles(self) -> str:
        """Login page specific styles"""
        return '''
        /* Login Page Styles */
        .login-container {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: var(--spacing-md);
        }

        .login-card {
            width: 100%;
            max-width: 400px;
            padding: var(--spacing-xl);
        }

        .login-header {
            text-align: center;
            margin-bottom: var(--spacing-xl);
        }

        .login-logo {
            font-size: 3rem;
            margin-bottom: var(--spacing-md);
        }

        .login-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: var(--spacing-sm);
        }

        .login-subtitle {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .login-form {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-md);
        }

        .login-button {
            margin-top: var(--spacing-md);
            padding: 0.75rem;
            font-weight: 600;
        }
        '''

    def _get_dashboard_styles(self) -> str:
        """Enhanced Dashboard page styles with beautiful metric cards"""
        return '''
        /* Enhanced Dashboard Styles */
        .dashboard-container {
            padding: var(--spacing-md);
        }

        .dashboard-header {
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
            border: 1px solid var(--border-color);
        }

        .dashboard-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-color) 0%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: var(--spacing-xs);
        }

        .dashboard-subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        /* Metric Cards Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--spacing-md);
            margin-bottom: var(--spacing-lg);
        }

        /* Beautiful Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, var(--surface-color) 0%, rgba(255, 255, 255, 0.05) 100%);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            position: relative;
            overflow: hidden;
            transition: all var(--transition-base);
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--accent-color);
            opacity: 0;
            transition: opacity var(--transition-base);
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        }

        .metric-card:hover::before {
            opacity: 1;
        }

        .metric-icon {
            width: 48px;
            height: 48px;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: var(--spacing-md);
            font-size: 1.5rem;
        }

        .metric-icon.success {
            background: rgba(34, 197, 94, 0.1);
            color: var(--success-color);
        }

        .metric-icon.info {
            background: rgba(59, 130, 246, 0.1);
            color: var(--info-color);
        }

        .metric-icon.warning {
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning-color);
        }

        .metric-icon.purple {
            background: rgba(139, 92, 246, 0.1);
            color: var(--purple-accent);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: var(--spacing-xs);
        }

        .metric-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .metric-change {
            display: flex;
            align-items: center;
            gap: var(--spacing-xs);
            margin-top: var(--spacing-sm);
            font-size: 0.875rem;
            font-weight: 500;
        }

        .metric-change.positive {
            color: var(--success-color);
        }

        .metric-change.negative {
            color: var(--error-color);
        }

        /* Dashboard Grid Layout */
        .dashboard-main-grid {
            display: grid;
            grid-template-columns: 1fr 3fr 1fr;
            gap: var(--spacing-lg);
        }

        /* Chart Section */
        .chart-section {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            min-height: 400px;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--spacing-lg);
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .dashboard-main-grid {
                grid-template-columns: 1fr 2fr;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 768px) {
            .dashboard-main-grid {
                grid-template-columns: 1fr;
            }

            .dashboard-title {
                font-size: 2rem;
            }

            .metric-value {
                font-size: 1.5rem;
            }
        }
        '''

    def _get_portfolio_styles(self) -> str:
        """Portfolio page specific styles with proper data alignment"""
        return '''
        /* Portfolio Page Styles */
        .portfolio-header {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-xl);
            margin-bottom: var(--spacing-lg);
        }

        .portfolio-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: var(--spacing-md);
            margin-bottom: var(--spacing-lg);
        }

        .portfolio-metric {
            text-align: center;
            padding: var(--spacing-md);
        }

        /* Portfolio Table - Fixed alignment */
        .portfolio-table {
            width: 100%;
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            overflow: hidden;
        }

        .portfolio-table thead th {
            background: rgba(0, 0, 0, 0.2);
            padding: var(--spacing-md);
            text-align: left;
            font-weight: 600;
            font-size: 0.875rem;
            color: var(--text-secondary);
            white-space: nowrap;
        }

        /* Proper column widths for alignment */
        .portfolio-table th:nth-child(1) { width: 15%; } /* Symbol */
        .portfolio-table th:nth-child(2) { width: 10%; } /* Quantity */
        .portfolio-table th:nth-child(3) { width: 12%; } /* Avg Cost */
        .portfolio-table th:nth-child(4) { width: 12%; } /* Current Price */
        .portfolio-table th:nth-child(5) { width: 15%; } /* Current Value */
        .portfolio-table th:nth-child(6) { width: 12%; } /* P&L */
        .portfolio-table th:nth-child(7) { width: 10%; } /* P&L % */
        .portfolio-table th:nth-child(8) { width: 14%; } /* Actions */

        .portfolio-table tbody td {
            padding: var(--spacing-md);
            border-bottom: 1px solid var(--border-color);
            vertical-align: middle;
        }

        .portfolio-table tbody tr:hover {
            background: var(--hover-color);
        }

        /* Symbol styling */
        .symbol-cell {
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Numeric values */
        .numeric-cell {
            font-family: var(--font-mono);
            text-align: right;
        }

        /* P&L Styling */
        .pnl-positive {
            color: var(--success-color);
            font-weight: 600;
        }

        .pnl-negative {
            color: var(--error-color);
            font-weight: 600;
        }

        /* Mutual Funds Section */
        .mf-card {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-md);
            border: 1px solid var(--border-color);
            transition: all var(--transition-base);
        }

        .mf-card:hover {
            border-color: var(--accent-color);
            transform: translateY(-2px);
        }

        .mf-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--spacing-md);
        }

        .mf-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: var(--spacing-md);
        }

        .mf-detail-item {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-xs);
        }

        .mf-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .mf-value {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        '''

    def _get_analytics_styles(self) -> str:
        """Enhanced Analytics page styles with proper layout"""
        return '''
        /* Analytics Page - Complete Revamp */
        .analytics-container {
            padding: var(--spacing-lg);
        }

        .analytics-header {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-xl);
            margin-bottom: var(--spacing-xl);
            text-align: center;
        }

        .analytics-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #22c55e 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: var(--spacing-sm);
        }

        .analytics-subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        /* Analytics Layout Grid */
        .analytics-layout {
            display: grid;
            grid-template-columns: 1fr;
            gap: var(--spacing-lg);
        }

        /* Chart Controls Section */
        .chart-controls-section {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
        }

        .chart-controls-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: var(--spacing-md);
            margin-bottom: var(--spacing-lg);
        }

        .symbol-selector-group {
            display: flex;
            align-items: center;
            gap: var(--spacing-md);
        }

        .symbol-selector {
            min-width: 200px;
        }

        .timeframe-selector {
            display: flex;
            gap: var(--spacing-xs);
            background: rgba(0, 0, 0, 0.2);
            padding: var(--spacing-xs);
            border-radius: var(--radius-md);
        }

        .timeframe-btn {
            padding: var(--spacing-sm) var(--spacing-md);
            border-radius: var(--radius-sm);
            background: transparent;
            color: var(--text-secondary);
            border: none;
            cursor: pointer;
            transition: all var(--transition-fast);
            font-weight: 500;
            font-size: 0.875rem;
        }

        .timeframe-btn:hover {
            background: var(--hover-color);
            color: var(--text-primary);
        }

        .timeframe-btn.active {
            background: var(--accent-color);
            color: white;
        }

        /* Main Chart Area */
        .chart-main-section {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: 0;
            overflow: hidden;
            min-height: 500px;
        }

        .chart-container {
            height: 500px;
            padding: var(--spacing-md);
        }

        /* Analytics Grid - 3 Column Layout */
        .analytics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: var(--spacing-lg);
            margin-top: var(--spacing-lg);
        }

        /* Technical Indicators Panel */
        .indicators-panel {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
        }

        .indicators-header {
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            margin-bottom: var(--spacing-lg);
            padding-bottom: var(--spacing-md);
            border-bottom: 1px solid var(--border-color);
        }

        .indicator-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: var(--spacing-sm) 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .indicator-name {
            font-size: 0.875rem;
            color: var(--text-primary);
        }

        .indicator-value {
            font-weight: 600;
            font-family: var(--font-mono);
        }

        /* Market Overview Panel */
        .market-overview-panel {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
        }

        .market-stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: var(--spacing-md) 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .market-stat-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .market-stat-value {
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Stock Screener Section */
        .screener-section {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-top: var(--spacing-lg);
        }

        .screener-filters {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--spacing-md);
            margin-bottom: var(--spacing-lg);
        }

        .screener-results {
            max-height: 400px;
            overflow-y: auto;
        }

        /* Responsive Analytics */
        @media (max-width: 1200px) {
            .analytics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 768px) {
            .analytics-grid {
                grid-template-columns: 1fr;
            }

            .chart-controls-header {
                flex-direction: column;
                align-items: stretch;
            }

            .analytics-title {
                font-size: 2rem;
            }
        }
        '''

    def _get_orderbook_styles(self) -> str:
        """Order Book page specific styles"""
        return '''
        /* Order Book Styles */
        .orderbook-container {
            padding: var(--spacing-lg);
            max-width: 1400px;
            margin: 0 auto;
        }

        .orderbook-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--spacing-lg);
        }

        .orderbook-filters {
            display: flex;
            gap: var(--spacing-md);
            align-items: center;
            flex-wrap: wrap;
        }

        .orderbook-table {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            overflow: hidden;
        }

        /* Use grid for both header and data rows, matching columns */
        .orders-header,
        .order-row {
            display: grid;
            grid-template-columns: 1.2fr 1.2fr 1.2fr 1fr 1.2fr 1.2fr 1.3fr 1fr 1.2fr;
            width: 100%;
            align-items: center;
            gap: 0;
        }
        .orders-header > *, .order-row > * {
            min-width: 80px;
        }

        .orders-header {
            background: rgba(0, 0, 0, 0.2);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
        }

        .order-row {
            background: none;
            border-bottom: 1px solid var(--border-color);
            transition: background var(--transition-fast);
            min-height: 48px;
        }
        .order-row:hover {
            background: var(--hover-color);
        }

        /* Cell alignment */
        .orders-header > *, .order-row > * {
            padding: 0.5rem 0.75rem;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }
        .orders-header > .text-right, .order-row > .text-right {
            text-align: right;
        }
        .orders-header > .text-center, .order-row > .text-center {
            text-align: center;
        }

        /* Status chip alignment */
        .order-row .order-status-chip {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        /* Actions column alignment */
        .order-row .order-actions {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        '''

    def _get_positions_styles(self) -> str:
        """Positions page specific styles"""
        return '''
        /* Positions Page Styles */
        .positions-container {
            padding: var(--spacing-lg);
        }

        .positions-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--spacing-md);
            margin-bottom: var(--spacing-lg);
        }

        .position-summary-card {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            text-align: center;
            border: 1px solid var(--border-color);
            transition: all var(--transition-base);
        }

        .position-summary-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent-color);
        }

        .positions-table {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            overflow: hidden;
        }

        .position-row {
            display: grid;
            grid-template-columns: 150px 100px 100px 120px 120px 150px 120px auto;
            padding: var(--spacing-md);
            border-bottom: 1px solid var(--border-color);
            align-items: center;
        }

        .position-symbol {
            font-weight: 600;
            color: var(--text-primary);
        }

        .position-pnl {
            font-weight: 600;
            font-family: var(--font-mono);
        }

        .position-actions {
            display: flex;
            gap: var(--spacing-sm);
            justify-content: flex-end;
        }
        '''

    def _get_watchlist_styles(self) -> str:
        """Watchlist page specific styles"""
        return '''
        /* Watchlist Styles */
        .watchlist-container {
            padding: var(--spacing-lg);
        }

        .watchlist-grid {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: var(--spacing-lg);
        }

        .watchlist-sidebar {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }

        .watchlist-item {
            padding: var(--spacing-md);
            border-radius: var(--radius-md);
            margin-bottom: var(--spacing-sm);
            cursor: pointer;
            transition: all var(--transition-fast);
            border-left: 3px solid transparent;
        }

        .watchlist-item:hover {
            background: var(--hover-color);
            transform: translateX(4px);
        }

        .watchlist-item.active {
            background: rgba(34, 197, 252, 0.1);
            border-left-color: var(--accent-color);
        }

        .watchlist-symbol {
            font-weight: 600;
            margin-bottom: var(--spacing-xs);
        }

        .watchlist-price {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .watchlist-main {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-lg);
        }

        .watchlist-detail {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
        }

        @media (max-width: 768px) {
            .watchlist-grid {
                grid-template-columns: 1fr;
            }

            .watchlist-sidebar {
                max-height: 300px;
            }
        }
        '''

    def _get_live_trading_styles(self) -> str:
        """Live Trading page specific styles"""
        return '''
        /* Live Trading Styles */
        .live-trading-container {
            padding: var(--spacing-lg);
        }

        .trading-status-bar {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-md);
            margin-bottom: var(--spacing-lg);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
        }

        .status-light {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-light.active {
            background: var(--success-color);
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .trading-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: var(--spacing-lg);
        }

        .active-strategies {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
        }

        .strategy-card {
            background: rgba(0, 0, 0, 0.2);
            border-radius: var(--radius-md);
            padding: var(--spacing-md);
            margin-bottom: var(--spacing-md);
            border: 1px solid var(--border-color);
            transition: all var(--transition-base);
        }

        .strategy-card.active {
            border-color: var(--success-color);
            background: rgba(34, 197, 94, 0.05);
        }

        .trade-log {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            max-height: 600px;
            overflow-y: auto;
        }

        .trade-entry {
            padding: var(--spacing-sm);
            border-bottom: 1px solid var(--border-color);
            font-size: 0.875rem;
        }

        .trade-time {
            color: var(--text-secondary);
            font-family: var(--font-mono);
        }
        '''

    def _get_order_management_styles(self) -> str:
        """Order Management page specific styles"""
        return '''
        /* Order Management Styles */
        .order-management-container {
            padding: var(--spacing-lg);
        }

        .order-form-section {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: var(--spacing-lg);
        }

        .order-form {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
        }

        .form-group {
            margin-bottom: var(--spacing-lg);
        }

        .form-label {
            display: block;
            margin-bottom: var(--spacing-sm);
            font-weight: 500;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .order-type-selector {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: var(--spacing-sm);
        }

        .order-type-btn {
            padding: var(--spacing-md);
            border: 2px solid var(--border-color);
            border-radius: var(--radius-md);
            background: transparent;
            cursor: pointer;
            transition: all var(--transition-fast);
        }

        .order-type-btn:hover {
            border-color: var(--accent-color);
        }

        .order-type-btn.selected {
            background: var(--accent-color);
            border-color: var(--accent-color);
            color: white;
        }

        .price-display-section {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
        }

        .current-price {
            font-size: 3rem;
            font-weight: 700;
            font-family: var(--font-mono);
            margin-bottom: var(--spacing-md);
        }

        .order-actions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--spacing-md);
            margin-top: var(--spacing-lg);
        }

        .buy-btn {
            background: linear-gradient(135deg, var(--success-color) 0%, #16a34a 100%);
            color: white;
            padding: var(--spacing-md) var(--spacing-lg);
            border: none;
            border-radius: var(--radius-md);
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-fast);
        }

        .sell-btn {
            background: linear-gradient(135deg, var(--error-color) 0%, #dc2626 100%);
            color: white;
            padding: var(--spacing-md) var(--spacing-lg);
            border: none;
            border-radius: var(--radius-md);
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-fast);
        }

        .buy-btn:hover, .sell-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }
        '''

    def _get_trading_styles(self) -> str:
        """Trading pages specific styles"""
        return '''
        /* Trading Pages Common Styles */
        .trading-panel {
            padding: var(--spacing-lg);
        }

        .trading-widget {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
        }

        .widget-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--spacing-lg);
            padding-bottom: var(--spacing-md);
            border-bottom: 1px solid var(--border-color);
        }

        .widget-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        '''

    def _get_backtesting_styles(self) -> str:
        """Backtesting page specific styles"""
        return '''
        /* Backtesting Page Styles */
        .backtest-container {
            padding: var(--spacing-lg);
        }

        .strategy-selector {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: var(--spacing-md);
            margin-bottom: var(--spacing-xl);
        }

        .strategy-card {
            background: var(--surface-color);
            border: 2px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            cursor: pointer;
            transition: all var(--transition-base);
        }

        .strategy-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent-color);
        }

        .strategy-card.selected {
            border-color: var(--accent-color);
            background: rgba(34, 197, 252, 0.1);
        }

        .strategy-icon {
            font-size: 2rem;
            margin-bottom: var(--spacing-md);
        }

        .parameter-section {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
        }

        .parameter-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--spacing-lg);
        }

        .backtest-actions {
            display: flex;
            gap: var(--spacing-md);
            justify-content: center;
            margin: var(--spacing-xl) 0;
        }

        .backtest-btn {
            padding: var(--spacing-md) var(--spacing-xl);
            font-weight: 600;
            border-radius: var(--radius-md);
            transition: all var(--transition-fast);
        }

        .results-section {
            margin-top: var(--spacing-xl);
        }

        .results-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: var(--spacing-md);
            margin-bottom: var(--spacing-xl);
        }

        .result-card {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            text-align: center;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: var(--spacing-xs);
        }

        .result-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        '''

    def _get_sip_strategy_styles(self) -> str:
        """SIP Strategy page specific styles"""
        return '''
        /* SIP Strategy Page Styles */
        .sip-container {
            padding: var(--spacing-lg);
        }

        .sip-header {
            text-align: center;
            margin-bottom: var(--spacing-xl);
        }

        .sip-tabs {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-sm);
            margin-bottom: var(--spacing-lg);
        }

        .portfolio-builder {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: var(--spacing-xl);
        }

        .symbol-selector-panel {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            max-height: 600px;
            overflow-y: auto;
        }

        .symbol-search {
            margin-bottom: var(--spacing-lg);
        }

        .symbol-list {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-sm);
        }

        .symbol-item {
            padding: var(--spacing-md);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all var(--transition-fast);
        }

        .symbol-item:hover {
            background: var(--hover-color);
            border-color: var(--accent-color);
        }

        .symbol-item.selected {
            background: rgba(34, 197, 252, 0.1);
            border-color: var(--accent-color);
        }

        .allocation-panel {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
        }

        .allocation-item {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr auto;
            gap: var(--spacing-md);
            align-items: center;
            padding: var(--spacing-md);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            margin-bottom: var(--spacing-md);
        }

        .allocation-chart {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-top: var(--spacing-lg);
        }

        .config-section {
            background: rgba(0, 0, 0, 0.2);
            border-radius: var(--radius-lg);
            padding: var(--spacing-xl);
            margin: var(--spacing-xl) 0;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: var(--spacing-lg);
        }

        .signal-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: var(--spacing-md);
        }

        .signal-card {
            padding: var(--spacing-lg);
            border-left: 4px solid transparent;
            border-radius: var(--radius-md);
            transition: all var(--transition-base);
        }

        .signal-card.strong-buy {
            border-left-color: var(--success-color);
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, transparent 100%);
        }

        .signal-card.buy {
            border-left-color: var(--info-color);
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%);
        }

        .signal-card.weak-buy {
            border-left-color: var(--warning-color);
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, transparent 100%);
        }
        '''

    def _get_settings_styles(self) -> str:
        """Settings page specific styles"""
        return '''
        /* Settings Page Styles */
        .settings-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: var(--spacing-xl);
        }

        .settings-header {
            text-align: center;
            margin-bottom: var(--spacing-xl);
        }

        .settings-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: var(--spacing-sm);
        }

        .settings-grid {
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: var(--spacing-xl);
        }

        .settings-nav {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            height: fit-content;
            position: sticky;
            top: var(--spacing-lg);
        }

        .settings-nav-item {
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            padding: var(--spacing-md);
            color: var(--text-secondary);
            text-decoration: none;
            border-radius: var(--radius-md);
            transition: all var(--transition-fast);
            margin-bottom: var(--spacing-xs);
        }

        .settings-nav-item:hover {
            background: var(--hover-color);
            color: var(--text-primary);
        }

        .settings-nav-item.active {
            background: rgba(34, 197, 252, 0.1);
            color: var(--accent-color);
            border-left: 3px solid var(--accent-color);
        }

        .settings-content {
            background: var(--surface-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-xl);
            min-height: 600px;
        }

        .settings-section {
            margin-bottom: var(--spacing-xl);
        }

        .settings-section-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: var(--spacing-lg);
            padding-bottom: var(--spacing-md);
            border-bottom: 1px solid var(--border-color);
        }

        .settings-group {
            background: rgba(0, 0, 0, 0.2);
            padding: var(--spacing-lg);
            border-radius: var(--radius-md);
            margin-bottom: var(--spacing-lg);
        }

        .settings-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: var(--spacing-md) 0;
        }

        .settings-label {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-xs);
        }

        .settings-label-text {
            font-weight: 500;
            color: var(--text-primary);
        }

        .settings-label-desc {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        @media (max-width: 768px) {
            .settings-grid {
                grid-template-columns: 1fr;
            }

            .settings-nav {
                position: static;
                margin-bottom: var(--spacing-lg);
            }
        }
        '''

    def switch_theme(self, new_theme: ThemeMode, storage: Optional[Dict] = None):
        """Switch to a new theme"""
        self.current_theme = new_theme

        # Save to storage if provided
        if storage:
            storage['theme'] = new_theme.value

        # Re-apply styles
        self.global_styles_applied = False
        self.apply_theme()

        logger.info(f"Switched to {new_theme.value} theme")

    def get_theme_toggle_button(self, storage: Optional[Dict] = None) -> ui.button:
        """Create a theme toggle button"""

        def toggle_theme():
            new_theme = ThemeMode.LIGHT if self.current_theme == ThemeMode.DARK else ThemeMode.DARK
            self.switch_theme(new_theme, storage)

        return ui.button(icon="brightness_6", on_click=toggle_theme).props(
            "flat round dense"
        ).classes("theme-toggle")


# Singleton instance
theme_manager = ThemeManager()


# Convenience functions
def apply_page_theme(page: PageTheme, storage: Optional[Dict] = None):
    """Apply theme for a specific page"""
    theme_manager.apply_theme(page, storage=storage)


def switch_theme(new_theme: ThemeMode, storage: Optional[Dict] = None):
    """Switch to a new theme"""
    theme_manager.switch_theme(new_theme, storage)


def get_current_theme() -> ThemeMode:
    """Get the current theme mode"""
    return theme_manager.current_theme