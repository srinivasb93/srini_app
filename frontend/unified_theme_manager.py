# frontend/unified_theme_manager.py
"""
UNIFIED Theme Manager - Single source of truth for all styling
Consolidates the best features from theme_manager.py, static CSS, and new enhancements
"""

from nicegui import ui
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ThemeMode(Enum):
    DARK = "dark"
    LIGHT = "light"


class PageTheme(Enum):
    DEFAULT = "default"
    LOGIN = "login"
    DASHBOARD = "dashboard"
    TRADING = "trading"
    ANALYTICS = "analytics"
    BACKTESTING = "backtesting"
    SIP_STRATEGY = "sip_strategy"
    STRATEGIES = "strategies"
    SETTINGS = "settings"
    ORDERBOOK = "orderbook"
    PORTFOLIO = "portfolio"
    POSITIONS = "positions"
    WATCHLIST = "watchlist"
    LIVE_TRADING = "live_trading"
    ORDER_MANAGEMENT = "order_management"
    ENHANCED_SCANNER = "enhanced_scanner"


@dataclass
class ThemeConfig:
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


class UnifiedThemeManager:
    """Single, unified theme manager combining all best features"""

    _instance = None
    _styles_applied = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UnifiedThemeManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.current_theme = ThemeMode.DARK
        self.themes = self._initialize_themes()

    def _initialize_themes(self) -> Dict[ThemeMode, ThemeConfig]:
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
                hover_color="rgba(255, 255, 255, 0.12)"
            ),
            ThemeMode.LIGHT: ThemeConfig(
                name="Light",
                mode=ThemeMode.LIGHT,
                primary_color="#f8fafc",
                accent_color="#0ea5e9",
                background="linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)",
                surface_color="rgba(0, 0, 0, 0.02)",
                text_primary="#1e293b",
                text_secondary="#475569",
                border_color="rgba(0, 0, 0, 0.08)",
                hover_color="rgba(0, 0, 0, 0.04)"
            )
        }

    def apply_unified_theme(self, page_type: PageTheme = PageTheme.DEFAULT, storage: Optional[Dict] = None):
        """Apply unified theme - ONLY call this once per page"""

        # Reset styles flag on new page load to ensure fresh application
        UnifiedThemeManager._styles_applied = False

        # Determine theme from storage
        if storage and 'theme' in storage:
            theme_value = storage.get('theme', 'dark').lower()
            if theme_value in ['dark', 'light']:
                self.current_theme = ThemeMode(theme_value)

        # Apply ALL styles in one comprehensive block
        self._apply_complete_styles(page_type)

        # Set Quasar dark mode
        ui.dark_mode(self.current_theme == ThemeMode.DARK)

        # Mark as applied
        UnifiedThemeManager._styles_applied = True

        logger.info(f"Applied unified {self.current_theme.value} theme for {page_type.value}")

    def _apply_complete_styles(self, page_type: PageTheme):
        """Apply ALL styles in a single comprehensive CSS block"""

        theme = self.themes[self.current_theme]

        # COMPLETE UNIFIED CSS - Everything in one place
        # Using multi-line string to avoid IDE parsing issues
        complete_css = '''
        
        /* UNIFIED THEME CSS - Consolidating all styling systems */
        /* UNIFIED THEME CSS - Consolidating all styling systems */
        
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        /* CSS Variables - Enhanced from all sources */
        :root {
            --primary-color: ''' + theme.primary_color + ''';
            --accent-color: ''' + theme.accent_color + ''';
            --surface-color: ''' + theme.surface_color + ''';
            --text-primary: ''' + theme.text_primary + ''';
            --text-secondary: ''' + theme.text_secondary + ''';
            --border-color: ''' + theme.border_color + ''';
            --hover-color: ''' + theme.hover_color + ''';
            
            --success-color: #22c55e;
            --error-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #3b82f6;
            --purple-accent: #8b5cf6;
            
            --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.15);
            --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.2);
            --shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.3);
            
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 16px;
            --radius-xl: 26px;
            --radius-full: 9999px;
            
            --transition-fast: 0.15s ease;
            --transition-base: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 0.5s ease;
            
            /* CONSISTENT SIZING SYSTEM */
            --size-xs: 0.25rem;    /* 4px */
            --size-sm: 0.5rem;     /* 8px */
            --size-md: 1rem;       /* 16px */
            --size-lg: 1.5rem;     /* 26px */
            --size-xl: 2rem;       /* 32px */
            --size-2xl: 2.5rem;    /* 40px */
            --size-3xl: 3rem;      /* 48px */
            
            /* FORM ELEMENT STANDARDS */
            --form-height-sm: 36px;
            --form-height-md: 44px;
            --form-height-lg: 52px;
            
            /* SPACING STANDARDS */
            --spacing-xs: 0.25rem;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
            
            --blur-amount: 20px;
            --glass-opacity: 0.08;
        }
        
        /* Global Reset */
        * { box-sizing: border-box; }
        
        /* Base Styles - Enhanced */
        html, body, .q-page, .q-page-container, .q-layout {
            background: ''' + theme.background + ''' !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            min-height: 100vh;
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }

        /* Theme-aware utility classes */
        .theme-text-primary { color: var(--text-primary) !important; }
        .theme-text-secondary { color: var(--text-secondary) !important; }
        .theme-text-accent { color: var(--accent-color) !important; }
        .theme-text-success { color: var(--success-color) !important; }
        .theme-text-error { color: var(--error-color) !important; }
        .theme-text-warning { color: var(--warning-color) !important; }
        .theme-text-info { color: var(--info-color) !important; }
        .theme-text-purple { color: var(--purple-accent) !important; }

        .theme-bg-success { background: var(--success-color) !important; }

        .theme-surface-card {
            background: var(--surface-color) !important;
            border: 1px solid var(--border-color) !important;
        }

        .positions-action-btn {
            background: transparent !important;
            color: var(--accent-color) !important;
            border: 1px solid var(--accent-color) !important;
            border-radius: var(--radius-md) !important;
            padding: 0.25rem 0.75rem !important;
            font-weight: 600 !important;
            transition: background var(--transition-fast), color var(--transition-fast) !important;
        }

        .positions-action-btn:hover,
        .positions-action-btn:focus {
            background: var(--accent-color) !important;
            color: #ffffff !important;
        }

        .positions-action-btn--exit {
            color: var(--error-color) !important;
            border-color: var(--error-color) !important;
        }

        .positions-action-btn--exit:hover,
        .positions-action-btn--exit:focus {
            background: var(--error-color) !important;
            color: #ffffff !important;
        }
        
        /* Enhanced App Container */
        .enhanced-app,
        .enhanced-dashboard {
            background: ''' + theme.background + ''' !important;
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        /* UNIFIED HEADER SYSTEM - Consistent across all pages */
        .q-header {
            background: rgba(10, 15, 35, 0.95) !important;
            backdrop-filter: blur(var(--blur-amount));
            border-bottom: 1px solid var(--border-color);
            padding: 0.75rem 1.5rem !important;
            min-height: 64px;
        }
        
        .q-header .q-toolbar {
            background: transparent !important;
            color: var(--text-primary) !important;
            min-height: 48px;
        }
        
        /* App title in header - THEME AWARE */
        .q-header .q-toolbar .text-xl {
            color: var(--text-primary) !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
        }
        
        /* Navigation Tabs - FIXED VISIBILITY & THEME AWARE */
        .nav-tabs-container {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        
        .nav-tab-btn {
            color: var(--text-primary) !important;
            background: var(--surface-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            transition: all var(--transition-fast) !important;
            text-transform: none !important;
            font-size: 0.875rem !important;
            min-height: var(--form-height-sm) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        .nav-tab-btn:hover {
            background: var(--hover-color) !important;
            border-color: var(--accent-color) !important;
            color: var(--text-primary) !important;
            transform: translateY(-1px);
        }
        
        /* Dropdown menus - THEME AWARE */
        .q-menu {
            background: var(--surface-color) !important;
            backdrop-filter: blur(var(--blur-amount)) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
        }
        
        .q-item {
            color: var(--text-primary) !important;
            transition: all var(--transition-fast) !important;
            min-height: var(--form-height-sm) !important;
            display: flex !important;
            align-items: center !important;
            padding: 0.5rem 1rem !important;
        }
        
        .q-item:hover {
            background: var(--hover-color) !important;
        }
        
        /* UNIFIED PAGE HEADER SYSTEM - Consistent sizing */
        .page-header-standard {
            background: var(--surface-color);
            backdrop-filter: blur(10px);
            border-radius: var(--radius-lg);
            padding: var(--spacing-sm);
            margin-bottom: var(--spacing-xs);
            border: 1px solid var(--border-color);
        }
        
        .page-title-standard {
            font-size: 1.75rem !important;  /* Reduced from 2.5rem to 1.75rem */
            font-weight: 600 !important;    /* Reduced from 700 to 600 */
            color: var(--text-primary) !important;
            margin-bottom: var(--spacing-xs);
            line-height: 1.2;
        }
        
        .page-subtitle-standard {
            color: var(--text-secondary) !important;
            font-size: 0.875rem;
            font-weight: 400;
        }
        
        /* Theme-aware header text with gradient effect */
        .theme-header-text {
            background: linear-gradient(135deg, var(--accent-color) 0%, var(--text-primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 1.75rem !important;
            font-weight: 600 !important;
        }
        
        /* Enhanced Card System - Combining all card types */
        .glass-card,
        .dashboard-card,
        .trading-card,
        .enhanced-card,
        .modern-card,
        .metric-card,
        .q-card {
            background: var(--surface-color) !important;
            backdrop-filter: blur(var(--blur-amount));
            -webkit-backdrop-filter: blur(var(--blur-amount));
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-lg) !important;
            box-shadow: var(--shadow-lg);
            transition: all var(--transition-base);
            overflow: hidden;
            padding: 0 !important;  /* Remove default padding */
        }
        
        .glass-card:hover,
        .dashboard-card:hover,
        .trading-card:hover,
        .enhanced-card:hover,
        .modern-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl);
            border-color: rgba(34, 197, 252, 0.3) !important;
        }
        
        /* Card Components - THEME AWARE */
        .card-header {
            padding: var(--spacing-md);
            border-bottom: 1px solid var(--border-color);
            background: rgba(0, 0, 0, 0.1);
        }
        
        .card-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary) !important;
        }
        
        /* UNIFIED FORM SYSTEM - COMPREHENSIVE FIX for all nested elements */
        
        /* CRITICAL: Override Quasar's default alignment classes */
        .q-field.row.no-wrap.items-start {
            align-items: center !important;
        }
        
        /* Base field container alignment */
        .q-field {
            min-height: 26px !important;
            margin-bottom: 0.5rem !important;
            display: flex !important;
            align-items: center !important;
        }
        
        /* CRITICAL: Fix for main field control container */
        .q-field .q-field__control {
            background: var(--surface-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            min-height: 26px !important;
            padding: 0 !important;
            transition: all var(--transition-fast) !important;
            position: relative !important;
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
        }
        
        /* CRITICAL: Fix the nested control container alignment */
        .q-field__control-container {
            min-height: 26px !important;
            padding: 0 0.75rem !important;
            display: flex !important;
            align-items: center !important;
            flex: 1 !important;
            width: 100% !important;
        }
        
        /* CRITICAL: Fix for Quasar's problematic row classes */
        .q-field__control-container.col.relative-position.row.no-wrap {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            flex-direction: row !important;
        }
        
        /* CRITICAL: Native input alignment */
        .q-field__native {
            min-height: 26px !important;
            padding: 0 !important;
            color: var(--text-primary) !important;
            font-size: 0.875rem !important;
            line-height: 1.4 !important;
            background: transparent !important;
            border: none !important;
            display: flex !important;
            align-items: center !important;
            flex: 1 !important;
            width: 100% !important;
        }
        
        /* CRITICAL: Actual input elements */
        .q-field input,
        .q-field textarea {
            background: transparent !important;
            border: none !important;
            color: var(--text-primary) !important;
            font-size: 0.875rem !important;
            min-height: 26px !important;
            padding: 0 !important;
            outline: none !important;
            box-shadow: none !important;
            line-height: 1.4 !important;
            text-align: left !important;
            vertical-align: middle !important;
            width: 100% !important;
            display: block !important;
        }
        
        /* CRITICAL: Fix label positioning */
        .q-field__label {
            color: var(--text-secondary) !important;
            font-size: 0.8rem !important;
            padding: 0 4px !important;
            top: -0.7rem !important;
            left: 0.5rem !important;
            background: var(--primary-color) !important;
            z-index: 10 !important;
        }
        
        /* CRITICAL: Fix field append (clear button, etc.) */
        .q-field__append {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 0.5rem !important;
            color: var(--text-secondary) !important;
        }
        
        /* CRITICAL: Modern card design preservation */
        .bg-gray-800\/50 {
            background: rgba(31, 41, 55, 0.5) !important;
            backdrop-filter: blur(12px) !important;
            border-radius: 12px !important;
        }
        
        .border-purple-500\/30 {
            border-color: rgba(168, 85, 247, 0.3) !important;
        }
        
        .border-green-500\/30 {
            border-color: rgba(34, 197, 94, 0.3) !important;
        }
        
        .border-cyan-500\/30 {
            border-color: rgba(6, 182, 212, 0.3) !important;
        }
        
        .border-yellow-500\/30 {
            border-color: rgba(245, 158, 11, 0.3) !important;
        }
        
        /* CRITICAL: Preserve modern grid layouts */
        .w-1\/2 {
            width: 50% !important;
            max-width: 50% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
        }
        
        .flex-1 {
            flex: 1 1 0% !important;
            min-width: 0 !important;
        }
        
        /* CRITICAL: Side-by-side layout preservation */
        .dashboard-grid-2 {
            display: grid !important;
            grid-template-columns: 1fr 1fr !important;
            gap: 1rem !important;
            width: 100% !important;
        }
        
        .dashboard-grid-3 {
            display: grid !important;
            grid-template-columns: 1fr 1fr 1fr !important;
            gap: 1rem !important;
            width: 100% !important;
        }
        
        /* CRITICAL: Fix responsive grids for settings/livetrading */
        .settings-layout {
            display: flex !important;
            flex-direction: row !important;
            gap: 1.5rem !important;
            width: 100% !important;
        }
        
        .settings-layout > * {
            flex: 1 1 0% !important;
            min-width: 0 !important;
            width: 0 !important; /* Let flex handle the sizing */
        }
        
        @media (max-width: 1026px) {
            .settings-layout {
                grid-template-columns: 1fr !important;
            }
        }
        
        /* CRITICAL: Live trading metrics grid */
        .live-trading-metrics {
            display: grid !important;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
            gap: 1rem !important;
            width: 100% !important;
            margin-bottom: 1.5rem !important;
        }
        
        /* CRITICAL: Modern metric card styling */
        .metric-card-modern {
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.9) 100%) !important;
            border: 1px solid rgba(59, 130, 246, 0.2) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            backdrop-filter: blur(12px) !important;
            position: relative !important;
            overflow: hidden !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .metric-card-modern::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981);
            opacity: 0.6;
        }
        
        .metric-card-modern:hover {
            transform: translateY(-4px) !important;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3) !important;
            border-color: rgba(59, 130, 246, 0.4) !important;
        }
        
        /* CRITICAL: Control panel modern styling */
        .control-panel-modern {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%) !important;
            border: 1px solid rgba(6, 182, 212, 0.3) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            backdrop-filter: blur(12px) !important;
        }
        
        /* CRITICAL: Advanced button styling */
        .btn-modern-primary {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }
        
        .btn-modern-primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4) !important;
        }
        
        .btn-modern-success {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
        }
        
        .btn-modern-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3) !important;
        }
        
        /* CRITICAL: Grid container fixes for different layouts */
        .grid-container-2col {
            display: grid !important;
            grid-template-columns: 1fr 1fr !important;
            gap: 1.5rem !important;
            width: 100% !important;
        }
        
        .grid-container-3col {
            display: grid !important;
            grid-template-columns: 1fr 1fr 1fr !important;
            gap: 1rem !important;
            width: 100% !important;
        }
        
        .grid-container-4col {
            display: grid !important;
            grid-template-columns: repeat(4, 1fr) !important;
            gap: 1rem !important;
            width: 100% !important;
        }
        
        /* CRITICAL: Responsive grid adjustments */
        @media (max-width: 1200px) {
            .grid-container-4col {
                grid-template-columns: repeat(2, 1fr) !important;
            }
        }
        
        @media (max-width: 768px) {
            .grid-container-2col,
            .grid-container-3col,
            .grid-container-4col {
                grid-template-columns: 1fr !important;
            }
        }
        
        /* CRITICAL: Fix for Tailwind/Quasar class conflicts */
        .gap-4 {
            gap: 1rem !important;
        }
        
        .gap-6 {
            gap: 1.5rem !important;
        }
        
        .p-4 {
            padding: .75rem !important;
        }
        
        .p-6 {
            padding: 1.5rem !important;
        }
        
        .mb-4 {
            margin-bottom: 1rem !important;
        }
        
        .mb-6 {
            margin-bottom: 1.5rem !important;
        }
        
        /* CRITICAL: Row layout preservation */
        .ui-row {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            gap: 1rem !important;
        }
        
        .ui-column {
            display: flex !important;
            flex-direction: column !important;
            gap: 0.5rem !important;
        }
        
        /* CRITICAL: Fix for status indicators and chips */
        .status-chip-modern {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            padding: 0.25rem 0.75rem !important;
            border-radius: 9999px !important;
            border: none !important;
            animation: pulse 2s infinite !important;
        }
        
        /* CRITICAL: Fix for enhanced dashboard components */
        .enhanced-dashboard {
            background: var(--background) !important;
            color: var(--text-primary) !important;
            min-height: 100vh !important;
            padding: 0 !important;
            width: 100% !important;
        }
        
        .dashboard-title-section {
            background: rgba(0, 0, 0, 0.1) !important;
            backdrop-filter: blur(10px) !important;
            border-bottom: 1px solid var(--border-color) !important;
            padding: 1rem !important;
            width: 100% !important;
        }
        
        .dashboard-title {
            color: var(--text-primary) !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }
        
        .dashboard-subtitle {
            color: var(--text-secondary) !important;
            font-size: 1rem !important;
            font-weight: 400 !important;
        }
        '''

        # Apply light theme overrides if needed
        if self.current_theme == ThemeMode.LIGHT:
            complete_css += self._get_light_theme_overrides()

        # Apply page-specific styles
        page_specific_css = self._get_page_specific_styles(page_type)
        if page_specific_css:
            complete_css += page_specific_css

        # Apply the complete CSS
        ui.add_head_html(f'<style>{complete_css}</style>')
        

    def _get_light_theme_overrides(self) -> str:
        """Light theme specific overrides"""
        return '''
        /* Light Theme Overrides */
        .q-header {
            background: rgba(248, 250, 252, 0.95) !important;
        }
        
        .q-header .q-toolbar {
            background: transparent !important;
            color: #1e293b !important;
        }
        
        /* CRITICAL: Light theme header text fixes */
        .q-header .q-toolbar .text-xl,
        .q-header .text-xl {
            color: var(--text-primary) !important;
        }
        
        /* Navigation buttons light theme */
        .nav-tab-btn {
            color: #1e293b !important;
            background: rgba(0, 0, 0, 0.05) !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
        }
        
        .nav-tab-btn:hover {
            background: rgba(0, 0, 0, 0.1) !important;
            color: #1e293b !important;
        }
        
        /* Menu styling for light theme */
        .q-menu {
            background: rgba(248, 250, 252, 0.95) !important;
            color: #1e293b !important;
        }
        
        .q-menu .q-item {
            color: #1e293b !important;
        }
        
        /* Profile dropdown light theme */
        .profile-dropdown-menu {
            background: rgba(248, 250, 252, 0.95) !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
        }
        
        .profile-dropdown-item {
            color: #1e293b !important;
        }
        
        .profile-dropdown-item:hover {
            background: rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Light theme form elements */
        .q-field .q-field__control {
            background: rgba(255, 255, 255, 0.8) !important;
            border-color: rgba(0, 0, 0, 0.2) !important;
            color: #1e293b !important;
        }
        
        .q-field input,
        .q-field textarea,
        .q-field .q-select__focus-target,
        .q-field .q-field__native input {
            color: #1e293b !important;
        }
        
        /* Light theme cards */
        .dashboard-card,
        .glass-card,
        .q-card {
            background: rgba(255, 255, 255, 0.8) !important;
            border-color: rgba(0, 0, 0, 0.1) !important;
            color: #1e293b !important;
        }
        
        .card-title {
            color: #1e293b !important;
        }
        
        /* Light theme buttons */
        .q-btn {
            color: #1e293b !important;
        }
        
        .q-btn--flat {
            color: #1e293b !important;
        }
        
        /* Light theme watchlist page headers */
        .dashboard-title-section {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            border-bottom: 2px solid #1e40af !important;
        }
        
        .dashboard-title {
            color: #ffffff !important;
        }
        
        .dashboard-subtitle {
            color: #e0e7ff !important;
        }
        
        /* Watchlist card headers */
        .card-header {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
            border-bottom: 1px solid #bae6fd !important;
        }
        
        .card-header .card-title {
            color: #0c4a6e !important;
            font-weight: 600 !important;
        }
        
        /* Watchlist header row */
        .watchlist-header {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
            border-bottom: 2px solid #cbd5e1 !important;
        }
        
        /* Status chips in light theme */
        .status-chip {
            background: #10b981 !important;
            color: white !important;
        }
        
        /* Order Management - Light Theme Enhancements */
        .theme-surface-elevated {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
            border: 1px solid #cbd5e1 !important;
        }
        
        .theme-surface-card {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        .theme-text-accent {
            color: #0891b2 !important;
        }
        
        .theme-text-primary {
            color: #0f172a !important;
        }
        
        .theme-text-muted {
            color: #64748b !important;
        }
        
        /* Order Management Header in Light Theme */
        .order-mgmt-header {
            background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%) !important;
            border-bottom: 2px solid #0284c7 !important;
            box-shadow: 0 2px 8px rgba(14, 165, 233, 0.15) !important;
        }
        
        .order-mgmt-header .theme-text-primary {
            color: #ffffff !important;
        }
        
        .order-mgmt-header .theme-text-accent {
            color: #ffffff !important;
        }
        
        /* Order Tabs in Light Theme */
        .order-tabs {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        }
        
        .order-tabs .q-tab {
            color: #475569 !important;
        }
        
        .order-tabs .q-tab--active {
            background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%) !important;
            color: #ffffff !important;
        }
        
        /* Order Form Header in Light Theme */
        .order-form-header {
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
            border-bottom: 1px solid #cbd5e1 !important;
        }
        
        .order-form-header .theme-text-primary {
            color: #0f172a !important;
        }
        
        .order-form-header .theme-text-accent {
            color: #0891b2 !important;
        }
        
        /* Icon Button in Light Theme */
        .theme-icon-button {
            color: #0891b2 !important;
        }
        
        .theme-icon-button:hover {
            background: rgba(14, 165, 233, 0.1) !important;
        }
        
        .theme-expansion {
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
        }
        
        .theme-expansion .q-expansion-item__label {
            color: #1e293b !important;
        }
        
        .theme-expansion .q-expansion-item__content {
            background: #ffffff !important;
            border-top: 1px solid #e2e8f0 !important;
        }
        
        .theme-divider {
            background: #cbd5e1 !important;
        }
        
        .theme-toggle {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
        }
        '''

    def _get_page_specific_styles(self, page_type: PageTheme) -> str:
        """Get comprehensive page-specific styles with context-aware colors and proper spacing"""

        styles = {
            PageTheme.DASHBOARD: '''
            /* Dashboard Specific - Enhanced layout with proper alignment */
            .dashboard-grid {
                display: grid;
                grid-template-columns: 300px 1fr 280px;
                gap: 1rem;
                padding: 1rem;
                max-height: calc(100vh - 80px);
                overflow: hidden;
            }
            
            .dashboard-left-panel, .dashboard-right-panel {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                max-height: 100%;
                overflow-y: auto;
            }
            
            .dashboard-center-panel {
                display: flex;
                flex-direction: column;
                gap: 1rem;
                max-height: 100%;
                overflow-y: auto;
            }
            
            /* Enhanced Quick Trade Form Layout */
            .quick-trade-form {
                display: flex !important;
                flex-direction: column !important;
                gap: 1rem !important;
                padding: 1rem !important;
                width: 100% !important;
            }
            
            .quick-trade-form .q-field,
            .quick-trade-form .q-select,
            .quick-trade-form .q-input {
                width: 100% !important;
                min-height: 44px !important;
            }
            
            .quick-trade-form .flex-1 {
                flex: 1 1 0% !important;
                min-width: 0 !important;
            }
            
            /* Trade Button Styling */
            .quick-trade-form .q-btn {
                min-height: 48px !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
                border-radius: 8px !important;
                transition: all 0.2s ease !important;
            }
            
            .quick-trade-form .q-btn:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
            }
            
            /* Recent Orders Widget Enhanced Alignment */
            .orders-container {
                width: 100% !important;
                overflow-x: hidden !important;
            }
            
            .order-row-grid {
                display: grid !important;
                grid-template-columns: 2fr 1fr 1fr 1fr 1fr !important;
                gap: 0.5rem !important;
                align-items: center !important;
                padding: 0.5rem !important;
                min-height: 32px !important;
                border-radius: 6px !important;
                transition: all 0.2s ease !important;
            }
            
            .order-row-grid:hover {
                background: rgba(255, 255, 255, 0.05) !important;
                transform: translateX(2px) !important;
            }
            
            /* Portfolio Summary Compact Layout */
            .portfolio-metrics-row {
                display: grid !important;
                grid-template-columns: repeat(5, 1fr) !important;
                gap: 1rem !important;
                padding: 1rem !important;
                text-align: center !important;
            }
            
            .portfolio-metric {
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                gap: 0.25rem !important;
            }
            
            /* Watchlist and Active Strategies 20% Width Enforcement */
            .dashboard-left-sidebar {
                flex: 0 0 20% !important;
                max-width: 20% !important;
                min-width: 20% !important;
                width: 20% !important;
            }
            
            /* Historical Chart and Market Data Proper Positioning */
            .dashboard-center-content {
                flex: 0 0 45% !important;
                max-width: 45% !important;
                min-width: 45% !important;
                width: 45% !important;
            }
            
            .dashboard-right-content {
                flex: 0 0 35% !important;
                max-width: 35% !important;
                min-width: 35% !important;
                width: 35% !important;
            }
            
            /* Improved Card Spacing and Alignment */
            .dashboard-card {
                width: 100% !important;
                margin-bottom: 1rem !important;
                border-radius: 12px !important;
                overflow: hidden !important;
            }
            
            .dashboard-card .card-header {
                padding: 1rem !important;
                border-bottom: 1px solid var(--border-color) !important;
                background: rgba(0, 0, 0, 0.1) !important;
            }
            
            .dashboard-card .card-title {
                font-size: 1rem !important;
                font-weight: 600 !important;
                color: var(--text-primary) !important;
            }
            
            /* Responsive Dashboard Layout */
            @media (max-width: 1400px) {
                .dashboard-left-sidebar {
                    flex: 0 0 22% !important;
                    max-width: 22% !important;
                    min-width: 22% !important;
                }
                
                .dashboard-center-content {
                    flex: 0 0 43% !important;
                    max-width: 43% !important;
                    min-width: 43% !important;
                }
                
                .dashboard-right-content {
                    flex: 0 0 35% !important;
                    max-width: 35% !important;
                    min-width: 35% !important;
                }
            }
            
            @media (max-width: 1200px) {
                .dashboard-left-sidebar,
                .dashboard-center-content,
                .dashboard-right-content {
                    flex: 1 1 100% !important;
                    max-width: 100% !important;
                    min-width: 100% !important;
                    width: 100% !important;
                }
            }
            ''',

            PageTheme.LIVE_TRADING: '''
            /* Live Trading Specific - Action-oriented colors */
            .live-trading-container {
                padding: 1rem;
                gap: 1rem;
            }
            
            .strategy-control-btn {
                background: linear-gradient(135deg, var(--info-color) 0%, #1e40af 100%) !important;
                color: white !important;
                min-height: 36px !important;
                padding: 0.5rem 1rem !important;
                border-radius: 6px !important;
                font-size: 0.8rem !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
            }
            
            .strategy-control-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
            }
            
            .emergency-btn {
                background: linear-gradient(135deg, var(--error-color) 0%, #b91c1c 100%) !important;
                color: white !important;
                animation: pulse-urgent 2s infinite !important;
            }
            
            .pause-btn {
                background: linear-gradient(135deg, var(--warning-color) 0%, #d97706 100%) !important;
                color: white !important;
            }
            
            .start-btn {
                background: linear-gradient(135deg, var(--success-color) 0%, #15803d 100%) !important;
                color: white !important;
            }
            
            .strategy-active {
                border-left: 4px solid var(--success-color) !important;
                background: linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, transparent 100%) !important;
            }
            
            .strategy-paused {
                border-left: 4px solid var(--warning-color) !important;
                background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, transparent 100%) !important;
            }
            
            .strategy-stopped {
                border-left: 4px solid var(--error-color) !important;
                background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, transparent 100%) !important;
            }
            
            @keyframes pulse-urgent {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.05); opacity: 0.8; }
            }
            ''',

            PageTheme.ORDER_MANAGEMENT: '''
            /* Order Management Specific - Professional trading colors and layouts */

            /* HTML class aliases scoped to order management page */
            .order-management-page .max-w-7xl { max-width: 80rem; width: 100%; margin-left: auto; margin-right: auto; }
            .order-management-page .mx-auto { margin-left: auto; margin-right: auto; }
            .order-management-page .px-6 { padding-left: 1.5rem; padding-right: 1.5rem; }
            .order-management-page .py-4 { padding-top: 1rem; padding-bottom: 1rem; }
            .order-management-page .mb-8 { margin-bottom: 2rem; }
            .order-management-page .rounded-xl { border-radius: var(--radius-xl); }
            .order-management-page .grid { display: grid; }
            .order-management-page .grid-cols-1 { grid-template-columns: 1fr; }
            .order-management-page .gap-6 { gap: 1.5rem; }
            @media (min-width: 768px) {
                .order-management-page .md\:grid-cols-4 { grid-template-columns: repeat(4, 1fr); }
            }
            
            /* ============================================================================ */
            /* CRITICAL FIX: Force consistent element sizing in Order Management */
            /* ============================================================================ */
            
            /* ULTRA HIGH SPECIFICITY: Force equal width for ALL flex-1 input columns */
            /* But preserve main layout flex-[4] and flex-1 ratios */
            .row.w-full.gap-3 > .column.flex-1.gap-2,
            .row.w-full.gap-3 > .column.flex-1.gap-1,
            .row.w-full.gap-3 > div.flex-1.gap-2,
            .row.w-full.gap-3 > div.flex-1.gap-1,
            .row.w-full.gap-2 > .column.flex-1.gap-2,
            .row.w-full.gap-2 > .column.flex-1.gap-1,
            .row.w-full.gap-4 > .column.flex-1.gap-2,
            .row.w-full.gap-4 > .column.flex-1.gap-1 {
                flex: 1 1 0px !important;
                min-width: 0 !important;
                max-width: 100% !important;
                width: auto !important;
                flex-basis: 0px !important;
            }
            
            /* Preserve main layout ratios */
            .flex-\[4\] {
                flex: 4 1 0% !important;  /* 80% width */
            }
            
            .flex-\[3\] {
                flex: 3 1 0% !important;  /* 75% width */
            }
            
            /* Additional catch-all for any flex-1 in gap rows that aren't special layouts */
            .gap-3 > .flex-1:not([class*="flex-["]),
            .gap-2 > .flex-1:not([class*="flex-["]) {
                flex: 1 1 0px !important;
                flex-basis: 0px !important;
            }
            
            /* Force all Quasar fields inside flex-1 to take full width */
            .flex-1 > .q-field,
            .flex-1 > .q-select,
            .flex-1 > .q-input,
            .flex-1 > .nicegui-select,
            .flex-1 > .nicegui-input,
            .flex-1 > .nicegui-number,
            .flex-1 .q-field,
            .flex-1 .q-select,
            .flex-1 .q-input,
            .flex-1 .nicegui-select,
            .flex-1 .nicegui-input,
            .flex-1 .nicegui-number {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                flex: 1 1 auto !important;
                box-sizing: border-box !important;
            }
            
            /* Override Quasar's intrinsic widths with MAXIMUM specificity */
            .flex-1 .q-field__control,
            .flex-1 .q-field__inner,
            .flex-1 .q-field__native,
            .q-field .q-field__control,
            .q-field .q-field__inner,
            .q-field .q-field__native {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
            }
            
            /* Nuclear option: Force all inputs in flex containers */
            .gap-3 .flex-1 input,
            .gap-3 .flex-1 select,
            .gap-2 .flex-1 input,
            .gap-2 .flex-1 select {
                width: 100% !important;
                box-sizing: border-box !important;
            }
            
            /* Force consistent row layouts */
            .w-full.gap-3,
            .w-full.gap-2,
            .w-full.gap-4 {
                display: flex !important;
            }
            
            /* Ensure all input fields have consistent height */
            .q-field__control,
            .q-field__inner {
                min-height: 40px !important;
                height: 40px !important;
            }
            
            /* Fix checkbox styling and alignment */
            .q-checkbox {
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                align-items: center !important;
                min-height: 40px !important;
                display: inline-flex !important;
            }
            
            .q-checkbox__inner {
                width: 24px !important;
                height: 24px !important;
                border-radius: 4px !important;
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
            }
            
            .q-checkbox__bg {
                border-radius: 4px !important;
            }
            
            /* Checkbox in flex container alignment */
            .flex-1 .q-checkbox,
            .gap-2 .q-checkbox,
            .gap-3 .q-checkbox {
                margin-top: auto !important;
                margin-bottom: auto !important;
            }
            
            /* CRITICAL FIX: Prevent border truncation on ALL cards */
            .q-card {
                overflow: visible !important;
                position: relative !important;  /* Establish stacking context */
            }
            
            .theme-surface-card {
                border: 1px solid var(--border-color) !important;
                border-radius: 8px !important;
                overflow: visible !important;
                box-sizing: border-box !important;
                position: relative !important;
            }
            
            /* Ensure market data panel doesn't clip borders */
            .flex-1.theme-surface-elevated {
                overflow: visible !important;
            }
            
            /* Fix nested card borders */
            .q-card .q-card {
                border: 1px solid var(--border-color) !important;
                margin: 0.5rem 0 !important;
            }
            
            /* CRITICAL: Prevent layout shift when dropdowns open */
            .q-menu,
            .q-dialog,
            .q-select__menu,
            .q-menu__container {
                position: fixed !important;
                z-index: 9000 !important;
            }
            
            /* Ensure dropdown doesn't push content */
            .q-field--with-bottom {
                padding-bottom: 0 !important;
            }
            
            /* Prevent card movement on interaction */
            .dashboard-card,
            .order-card,
            [class*="card"] {
                will-change: auto !important;
                transform: translateZ(0) !important;
            }
            
            /* Force all overlays to be above content without affecting layout */
            .q-menu,
            .q-tooltip,
            .q-dialog__backdrop {
                position: fixed !important;
            }
            
            /* Additional targeted fixes for specific input rows only */
            /* Don't affect main layout columns */
            .w-full > [class*="gap-"] > .flex-1:not(.flex-\[4\]):not(.flex-\[3\]) {
                flex-basis: 0 !important;
                flex-grow: 1 !important;
                flex-shrink: 1 !important;
            }
            
            /* ============================================================================ */
            /* Order Management - Theme-Aware Styling (Dark Theme) */
            /* ============================================================================ */
            
            /* Order Management Header in Dark Theme */
            .order-mgmt-header {
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
                border-bottom: 2px solid rgba(6, 182, 212, 0.3);
                box-shadow: 0 2px 8px rgba(6, 182, 212, 0.15);
            }
            
            /* Order Tabs in Dark Theme */
            .order-tabs {
                background: rgba(30, 41, 59, 0.5);
                border: 1px solid rgba(100, 116, 139, 0.3);
            }
            
            .order-tabs .q-tab {
                color: #cbd5e1;
            }
            
            .order-tabs .q-tab--active {
                background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
                color: #ffffff;
            }
            
            /* Order Form Header in Dark Theme */
            .order-form-header {
                background: rgba(30, 41, 59, 0.3);
                border-bottom-color: rgba(100, 116, 139, 0.3);
            }
            
            /* Icon Button in Dark Theme */
            .theme-icon-button {
                color: #06b6d4;
            }
            
            .theme-icon-button:hover {
                background: rgba(6, 182, 212, 0.1);
            }
            
            /* Dark Theme Text Colors */
            .theme-text-primary {
                color: #f1f5f9;
            }
            
            .theme-text-accent {
                color: #06b6d4;
            }
            
            .order-mgmt-logo-box {
                width: 2.5rem;
                height: 2.5rem;
                border-radius: var(--radius-md);
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, var(--accent-color) 0%, var(--purple-accent) 100%);
            }
            
            /* Metrics Dashboard Cards */
            .order-metrics-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 1.5rem;
                margin-bottom: 2rem;
            }
            
            @media (max-width: 1024px) {
                .order-metrics-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
            
            @media (max-width: 640px) {
                .order-metrics-grid {
                    grid-template-columns: 1fr;
                }
            }
            
            /* Metric card - also expose as .metric-card to match HTML */
            .order-metric-card,
            .metric-card {
                background: linear-gradient(135deg, rgba(248, 250, 252, 0.95) 0%, rgba(226, 232, 240, 0.9) 100%);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1.5rem;
                backdrop-filter: blur(10px);
                position: relative;
                overflow: hidden;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .order-metric-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, var(--info-color), var(--accent-color), var(--success-color));
                opacity: 0.6;
            }
            
            .order-metric-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                border-color: rgba(59, 130, 246, 0.4);
            }
            
            .order-metric-icon-box {
                width: 3rem;
                height: 3rem;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: transform 0.2s ease;
            }
            
            .order-metric-card:hover .order-metric-icon-box {
                transform: scale(1.1);
            }
            
            /* Order Card Styles (HTML lines 12-25) */
            .order-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid var(--border-color);
                border-left: 4px solid transparent;
                border-radius: var(--radius-md);
                padding: 1rem;
                margin-bottom: 1rem;
                transition: all var(--transition-base);
                cursor: pointer;
            }
            
            .order-card:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-xl);
            }
            
            /* Order type specific borders (HTML lines 22-25) */
            .regular-order { 
                border-left-color: var(--info-color);
            }
            
            .gtt-order { 
                border-left-color: var(--success-color);
            }
            
            .scheduled-order { 
                border-left-color: var(--warning-color);
            }
            
            .auto-order { 
                border-left-color: var(--purple-accent);
            }
            
            /* Status Badge Styles (HTML lines 27-30) */
            .status-active {
                background: linear-gradient(135deg, var(--success-color), #059669);
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: var(--radius-full);
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .status-pending {
                background: linear-gradient(135deg, var(--warning-color), #d97706);
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: var(--radius-full);
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .status-completed {
                background: linear-gradient(135deg, var(--text-secondary), #4b5563);
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: var(--radius-full);
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .status-cancelled {
                background: linear-gradient(135deg, var(--error-color), #dc2626);
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: var(--radius-full);
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            /* Transaction Type Badges */
            .transaction-buy-badge {
                background: rgba(16, 185, 129, 0.1);
                color: var(--success-color);
                padding: 0.25rem 0.75rem;
                border-radius: var(--radius-full);
                font-size: 0.75rem;
                font-weight: 600;
            }
            
            .transaction-sell-badge {
                background: rgba(239, 68, 68, 0.1);
                color: var(--error-color);
                padding: 0.25rem 0.75rem;
                border-radius: var(--radius-full);
                font-size: 0.75rem;
                font-weight: 600;
            }
            
            /* Tab Navigation Styles - Enhanced Modern Look */
            .order-tabs-container {
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
                border: 1px solid rgba(139, 92, 246, 0.2);
                border-radius: var(--radius-lg);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                margin-bottom: 2rem;
                padding: 0.5rem;
            }
            
            /* Modern tabs */
            .q-tab {
                background: transparent !important;
                border: 1px solid transparent !important;
                border-radius: 10px !important;
                color: var(--text-secondary) !important;
                transition: all 0.3s ease !important;
                font-weight: 500 !important;
                margin: 0 0.25rem !important;
            }
            
            .q-tab:hover {
                background: rgba(139, 92, 246, 0.1) !important;
                border-color: rgba(139, 92, 246, 0.3) !important;
                color: var(--text-primary) !important;
            }
            
            .q-tab--active,
            .q-tab.q-tab--active {
                background: linear-gradient(135deg, var(--purple-accent) 0%, #6366f1 100%) !important;
                color: white !important;
                border: none !important;
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4) !important;
            }
            
            /* Active tab - alias to match HTML's .tab-active */
            .order-tab-active,
            .tab-active {
                background: linear-gradient(135deg, var(--purple-accent), #6366f1) !important;
                color: white !important;
                border-radius: var(--radius-md) !important;
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4) !important;
            }
            
            /* Tab panels */
            .q-tab-panel {
                background: transparent !important;
                padding: 1.5rem !important;
            }
            
            /* Modal Styles */
            .order-modal-overlay {
                background: rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(4px);
            }
            
            .order-modal-content {
                background: var(--surface-color);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-xl);
                box-shadow: var(--shadow-xl);
                max-height: 90vh;
                overflow-y: auto;
            }
            
            .order-modal-header {
                padding: 1.5rem;
                border-bottom: 1px solid var(--border-color);
                background: rgba(0, 0, 0, 0.05);
            }
            
            .order-modal-body {
                padding: 1.5rem;
            }
            
            /* Position Sizing Modal Enhancements */
            .position-sizing-dialog .q-dialog__inner {
                width: min(900px, 94vw) !important;
                margin: 0 auto !important;
                padding: 0 !important;
                align-items: center !important;
            }
            
            .position-sizing-dialog .q-card.position-sizing-card {
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(8, 47, 73, 0.92) 100%) !important;
                border: 1px solid rgba(148, 163, 184, 0.28) !important;
                border-radius: var(--radius-xl) !important;
                box-shadow: 0 24px 48px rgba(2, 6, 23, 0.55) !important;
                padding: 1.75rem !important;
                gap: 1.5rem !important;
                max-height: calc(100vh - 4rem) !important;
                overflow-y: auto !important;
                scrollbar-width: thin;
            }
            
            .position-sizing-dialog .q-card.position-sizing-card::-webkit-scrollbar {
                width: 8px;
            }
            
            .position-sizing-dialog .q-card.position-sizing-card::-webkit-scrollbar-thumb {
                background: rgba(148, 163, 184, 0.35);
                border-radius: 999px;
            }
            
            .position-sizing-surface {
                backdrop-filter: blur(14px);
            }
            
            .position-sizing-summary {
                display: grid !important;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) !important;
                gap: 1rem !important;
                padding: 1.25rem 1.5rem !important;
                border-radius: var(--radius-lg) !important;
                border: 1px solid rgba(20, 184, 166, 0.35) !important;
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(45, 212, 191, 0.12) 100%) !important;
                box-shadow: inset 0 1px 0 rgba(45, 212, 191, 0.2) !important;
            }
            
            .position-sizing-summary .text-sm {
                margin: 0 !important;
                white-space: nowrap !important;
            }
            
            .position-sizing-grid {
                width: 100% !important;
                gap: 1.25rem !important;
                flex-wrap: wrap !important;
                align-items: stretch !important;
            }
            
            .position-sizing-grid > .column.flex-1 {
                flex: 1 1 220px !important;
                min-width: 220px !important;
            }
            
            .position-sizing-grid .q-field,
            .position-sizing-grid .q-select,
            .position-sizing-grid .q-input {
                width: 100% !important;
                background: rgba(15, 23, 42, 0.65) !important;
                border-radius: var(--radius-md) !important;
                border: 1px solid rgba(148, 163, 184, 0.32) !important;
                box-shadow: inset 0 1px 0 rgba(148, 163, 184, 0.12) !important;
            }
            
            .position-sizing-grid .q-field__control {
                min-height: 44px !important;
            }
            
            .position-sizing-grid .text-lg {
                text-align: right !important;
                color: rgba(56, 189, 248, 0.95) !important;
            }
            
            .position-sizing-summary-panel {
                background: linear-gradient(135deg, rgba(14, 116, 144, 0.18) 0%, rgba(15, 23, 42, 0.88) 100%) !important;
                border: 1px solid rgba(94, 234, 212, 0.2) !important;
                box-shadow: inset 0 1px 0 rgba(14, 165, 233, 0.2) !important;
                padding: 1.25rem !important;
                gap: 0.75rem !important;
            }
            
            .position-sizing-summary-panel .text-sm {
                color: rgba(226, 232, 240, 0.92) !important;
            }
            
            .position-sizing-summary-panel .text-xs {
                color: rgba(148, 163, 184, 0.85) !important;
            }
            
            .position-sizing-actions {
                flex-wrap: wrap !important;
                gap: 0.75rem !important;
                justify-content: flex-end !important;
            }
            
            .position-sizing-actions .q-btn {
                min-width: 160px !important;
                height: 44px !important;
                border-radius: var(--radius-md) !important;
            }
            
            @media (max-width: 768px) {
                .position-sizing-dialog .q-card.position-sizing-card {
                    padding: 1.25rem !important;
                    max-height: calc(100vh - 2rem) !important;
                }
                
                .position-sizing-summary {
                    grid-template-columns: 1fr !important;
                }
                
                .position-sizing-grid > .column.flex-1 {
                    min-width: 100% !important;
                }
                
                .position-sizing-grid .text-lg {
                    text-align: left !important;
                }
            }
            
            /* Basket Orders Modal Enhancements */
            .basket-orders-dialog .q-dialog__inner {
                width: min(820px, 94vw) !important;
                margin: 0 auto !important;
                padding: 0 !important;
            }
            
            .basket-orders-card {
                background: linear-gradient(135deg, rgba(17, 24, 39, 0.97) 0%, rgba(15, 23, 42, 0.92) 100%) !important;
                border: 1px solid rgba(59, 130, 246, 0.28) !important;
                border-radius: var(--radius-xl) !important;
                box-shadow: 0 24px 48px rgba(2, 6, 23, 0.5) !important;
                padding: 1.75rem !important;
                gap: 1.5rem !important;
                max-height: calc(100vh - 4rem) !important;
                display: flex !important;
                flex-direction: column !important;
            }
            
            .basket-orders-card .q-card__section {
                display: flex !important;
                flex-direction: column !important;
                gap: 1.25rem !important;
                padding: 0 !important;
            }
            
            .basket-orders-header {
                border-bottom: 1px solid rgba(148, 163, 184, 0.18) !important;
                padding-bottom: 1rem !important;
                gap: 1rem !important;
            }
            
            .basket-orders-summary {
                color: rgba(226, 232, 240, 0.8) !important;
            }
            
            .basket-orders-list {
                flex: 1 1 auto !important;
                max-height: 320px !important;
                overflow-y: auto !important;
                padding-right: 0.5rem !important;
                margin-right: -0.5rem !important;
            }
            
            .basket-orders-list::-webkit-scrollbar {
                width: 8px;
            }
            
            .basket-orders-list::-webkit-scrollbar-thumb {
                background: rgba(59, 130, 246, 0.35);
                border-radius: 999px;
            }
            
            .basket-orders-feedback {
                gap: 1rem !important;
            }
            
            .basket-orders-actions {
                border-top: 1px solid rgba(148, 163, 184, 0.18) !important;
                padding-top: 1rem !important;
                gap: 0.75rem !important;
                flex-wrap: wrap !important;
            }
            
            .basket-orders-card .row.justify-end,
            .basket-orders-card .row.items-end {
                flex-direction: column !important;
                align-items: stretch !important;
                gap: 0.75rem !important;
            }
            
            .basket-orders-card .row.justify-end .q-btn,
            .basket-orders-card .row.items-end .q-btn {
                width: 100% !important;
            }
            
            .basket-orders-actions .q-btn {
                min-width: 180px !important;
                height: 44px !important;
                border-radius: var(--radius-md) !important;
            }
            
            @media (max-width: 768px) {
                .basket-orders-card {
                    padding: 1.25rem !important;
                    max-height: calc(100vh - 2rem) !important;
                }
                
                .basket-orders-actions {
                    justify-content: flex-start !important;
                }
                
                .basket-orders-actions .q-btn {
                    flex: 1 1 100% !important;
                    min-width: 0 !important;
                }
            }
            
            .order-action-row {
                gap: 0.75rem !important;
                flex-wrap: wrap !important;
            }
            
            .order-action-row .q-btn {
                min-height: 48px !important;
                border-radius: 14px !important;
            }
            
            /* Buy/Sell Toggle Buttons */
            .buy-toggle-btn {
                background: linear-gradient(135deg, var(--success-color) 0%, #059669 100%);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: var(--radius-md);
                font-weight: 600;
                border: none;
                transition: all var(--transition-fast);
            }
            
            .buy-toggle-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            }
            
            .sell-toggle-btn {
                background: linear-gradient(135deg, var(--error-color) 0%, #dc2626 100%);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: var(--radius-md);
                font-weight: 600;
                border: none;
                transition: all var(--transition-fast);
            }
            
            .sell-toggle-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
            }
            
            .toggle-btn-inactive {
                background: var(--surface-color);
                color: var(--text-secondary);
                border: 1px solid var(--border-color);
            }
            
            /* Order Form Grid */
            .order-form-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 1rem;
            }
            
            .order-form-field {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            
            .order-form-label {
                color: var(--text-secondary);
                font-size: 0.875rem;
                font-weight: 500;
            }
            
            /* Order Type Selection Buttons */
            .order-type-btn {
                background: var(--surface-color);
                border: 2px solid var(--border-color);
                color: var(--text-secondary);
                padding: 0.75rem 1.25rem;
                border-radius: var(--radius-md);
                font-weight: 500;
                transition: all var(--transition-fast);
                cursor: pointer;
            }
            
            .order-type-btn:hover {
                border-color: var(--accent-color);
                background: rgba(34, 197, 252, 0.05);
            }
            
            .order-type-btn-active {
                border-color: var(--info-color);
                background: rgba(59, 130, 246, 0.1);
                color: var(--info-color);
            }
            
            /* Position Calculator Styles */
            .calculator-summary {
                background: linear-gradient(135deg, 
                    rgba(59, 130, 246, 0.1) 0%, 
                    rgba(34, 197, 252, 0.1) 100%);
                border: 1px solid var(--accent-color);
                border-radius: var(--radius-md);
                padding: 1rem;
                margin-top: 1rem;
            }
            
            .calculator-result-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1rem;
            }
            
            .calculator-result-item {
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
            }
            
            .calculator-result-label {
                color: var(--text-secondary);
                font-size: 0.875rem;
            }
            
            .calculator-result-value {
                color: var(--text-primary);
                font-size: 1rem;
                font-weight: 600;
            }
            
            .calculator-apply-btn {
                background: linear-gradient(135deg, var(--success-color) 0%, #059669 100%);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: var(--radius-md);
                font-weight: 600;
                border: none;
                transition: all var(--transition-base);
                width: 100%;
                margin-top: 1rem;
            }
            
            .calculator-apply-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            }
            
            /* Modern Basket Badge with Counter - Enhanced */
            .basket-badge {
                background: linear-gradient(135deg, var(--purple-accent) 0%, #6d28d9 100%);
                color: white;
                border-radius: var(--radius-xl);
                padding: 0.5rem 1.25rem;
                font-size: 0.875rem;
                font-weight: 600;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
                position: relative;
            }
            
            .basket-badge:hover {
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 8px 20px rgba(139, 92, 246, 0.5);
            }
            
            .basket-count {
                background: linear-gradient(135deg, #f97316 0%, #ef4444 100%);
                padding: 0.125rem 0.5rem;
                border-radius: var(--radius-full);
                font-size: 0.75rem;
                font-weight: 700;
                min-width: 1.5rem;
                text-align: center;
                animation: pulse-badge 2s infinite;
            }
            
            @keyframes pulse-badge {
                0%, 100% { 
                    transform: scale(1);
                    opacity: 1;
                }
                50% { 
                    transform: scale(1.1);
                    opacity: 0.9;
                }
            }
            
            /* Modern gradient buttons */
            .btn-gradient-primary {
                background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important;
                color: white !important;
                font-weight: 600 !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.5rem !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3) !important;
            }
            
            .btn-gradient-primary:hover {
                transform: translateY(-2px) scale(1.05) !important;
                box-shadow: 0 8px 20px rgba(6, 182, 212, 0.5) !important;
            }
            
            .btn-gradient-purple {
                background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
                color: white !important;
                font-weight: 600 !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.5rem !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3) !important;
            }
            
            .btn-gradient-purple:hover {
                transform: translateY(-2px) scale(1.05) !important;
                box-shadow: 0 8px 20px rgba(139, 92, 246, 0.5) !important;
            }
            
            /* Modern card gradients */
            .card-gradient-blue {
                background: linear-gradient(135deg, rgba(30, 58, 138, 0.3) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
                border: 1px solid rgba(6, 182, 212, 0.3) !important;
            }
            
            .card-gradient-purple {
                background: linear-gradient(135deg, rgba(109, 40, 217, 0.3) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
                border: 1px solid rgba(139, 92, 246, 0.3) !important;
            }
            
            .card-gradient-green {
                background: linear-gradient(135deg, rgba(5, 150, 105, 0.3) 0%, rgba(16, 185, 129, 0.2) 100%) !important;
                border: 1px solid rgba(16, 185, 129, 0.3) !important;
            }
            
            .card-gradient-red {
                background: linear-gradient(135deg, rgba(220, 38, 38, 0.3) 0%, rgba(239, 68, 68, 0.2) 100%) !important;
                border: 1px solid rgba(239, 68, 68, 0.3) !important;
            }
            
            .card-gradient-yellow {
                background: linear-gradient(135deg, rgba(217, 119, 6, 0.3) 0%, rgba(245, 158, 11, 0.2) 100%) !important;
                border: 1px solid rgba(245, 158, 11, 0.3) !important;
            }
            
            /* Action Buttons */
            .order-action-modify {
                color: var(--info-color);
                transition: all var(--transition-fast);
            }
            
            .order-action-modify:hover {
                color: #2563eb;
                transform: scale(1.05);
            }
            
            .order-action-cancel {
                color: var(--error-color);
                transition: all var(--transition-fast);
            }
            
            .order-action-cancel:hover {
                color: #dc2626;
                transform: scale(1.05);
            }
            
            /* Order Summary Section */
            .order-summary-box {
                background: rgba(0, 0, 0, 0.05);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-md);
                padding: 1rem;
                margin-top: 1rem;
            }
            
            .order-summary-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 0;
                border-bottom: 1px solid var(--border-color);
            }
            
            .order-summary-item:last-child {
                border-bottom: none;
            }
            
            /* Loading States */
            .order-loading-overlay {
                position: absolute;
                inset: 0;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: var(--radius-lg);
                z-index: 50;
            }
            
            /* Responsive Adjustments */
            @media (max-width: 768px) {
                .order-form-grid {
                    grid-template-columns: 1fr;
                }
                
                .calculator-result-grid {
                    grid-template-columns: 1fr;
                }
                
                .order-mgmt-header {
                    flex-direction: column;
                    gap: 1rem;
                }
            }
            
            /* =====================================================
               CRITICAL FIXES FOR ORDER MANAGEMENT
               ===================================================== */
            
            /* FIX 1: Element Sizing Consistency */
            /* Ensure all form elements in a row have matching heights */
            .q-field,
            .q-select,
            .q-input,
            
            .q-field .q-field__control,
            .q-select .q-field__control,
            .q-input .q-field__control {
                min-height: 40px !important;
                height: 40px !important;
            }
            
            /* FIX 2: Nested Card Padding and Border */
            /* Override default card padding removal for nested cards */
            .theme-surface-elevated .q-card,
            .theme-surface-card.q-card {
                padding: 0.5rem !important;  /* Restore padding for nested cards */
                overflow: visible !important;  /* Prevent border clipping */
            }
            
            /* Fix nested card border visibility */
            .theme-surface-elevated .theme-surface-card,
            .flex-1.theme-surface-elevated .q-card {
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-md) !important;
                box-sizing: border-box !important;  /* Include border in dimensions */
            }
            
            /* Specific fix for Market Data LTP card */
            .theme-surface-elevated .w-full.p-2.mb-2 {
                padding: 0.5rem !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-md) !important;
                margin-bottom: 0.5rem !important;
            }
            ''',

            PageTheme.PORTFOLIO: '''
            /* Portfolio Specific - Wealth management colors */
            .portfolio-header {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            
            .portfolio-table-container {
                background: var(--surface-color);
                border-radius: var(--radius-lg);
                border: 1px solid var(--border-color);
                overflow: hidden;
            }
            
            .portfolio-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
            }
            
            .portfolio-table th {
                background: rgba(0, 0, 0, 0.2) !important;
                padding: 1rem 0.75rem !important;
                text-align: left !important;
                font-weight: 600 !important;
                font-size: 0.75rem !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
                color: var(--text-secondary) !important;
                border-bottom: 1px solid var(--border-color) !important;
            }
            
            .portfolio-table td {
                padding: 0.75rem !important;
                border-bottom: 1px solid var(--border-color) !important;
                color: var(--text-primary) !important;
                font-size: 0.875rem !important;
            }
            
            .portfolio-table tr:hover {
                background: var(--hover-color) !important;
            }
            
            .portfolio-gain {
                color: var(--success-color) !important;
                font-weight: 600 !important;
            }
            
            .portfolio-loss {
                color: var(--error-color) !important;
                font-weight: 600 !important;
            }
            
            .portfolio-neutral {
                color: var(--text-secondary) !important;
            }
            ''',

            PageTheme.POSITIONS: '''
            /* Positions Specific - Trading position colors */
            .positions-header-wrapper {
                border-left: 4px solid transparent;
                border-radius: var(--radius-lg) var(--radius-lg) 0 0;
                overflow: hidden;
                background: var(--surface-color);
                border: 1px solid var(--border-color);
                border-bottom: none;
            }

            .position-row {
                background: var(--surface-color);
                border: 1px solid var(--border-color);
                margin-bottom: 0.5rem;
            }
            
            .positions-header-grid {
                display: grid;
                grid-template-columns: minmax(160px, 1.5fr) repeat(3, minmax(120px, 1fr)) minmax(140px, 1.2fr) minmax(120px, 1fr) minmax(120px, 1fr) minmax(120px, 1fr) minmax(140px, 1fr);
                column-gap: 0.75rem;
                align-items: center;
                padding: 0.75rem;
                box-sizing: border-box;
                background: transparent;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-size: 0.75rem;
                color: var(--text-secondary);
                border-bottom: 1px solid var(--border-color);
            }
            
            .positions-row-grid {
                display: grid;
                grid-template-columns: minmax(160px, 1.5fr) repeat(3, minmax(120px, 1fr)) minmax(140px, 1.2fr) minmax(120px, 1fr) minmax(120px, 1fr) minmax(120px, 1fr) minmax(140px, 1fr);
                column-gap: 0.75rem;
                align-items: center;
                padding: 0.75rem;
                box-sizing: border-box;
                border-bottom: 1px solid var(--border-color);
                transition: background 0.2s ease;
                background: transparent;
            }
            
            .positions-row-grid:hover {
                background: var(--hover-color);
            }
            
            .position-long {
                border-left: 4px solid var(--success-color);
                background: linear-gradient(90deg, rgba(34, 197, 94, 0.08) 0%, transparent 100%), var(--surface-color);
            }
            
            .position-short {
                border-left: 4px solid var(--error-color);
                background: linear-gradient(90deg, rgba(239, 68, 68, 0.08) 0%, transparent 100%), var(--surface-color);
            }
            ''',

            PageTheme.ANALYTICS: '''
            /* Analytics Specific - Data visualization colors */
            .analytics-controls {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr auto;
                gap: 1rem;
                padding: 1rem;
                background: var(--surface-color);
                border-radius: var(--radius-lg);
                margin-bottom: 1rem;
            }
            
            .chart-type-btn {
                background: var(--surface-color) !important;
                border: 2px solid var(--border-color) !important;
                color: var(--text-secondary) !important;
                min-height: 36px !important;
                transition: all 0.2s ease !important;
            }
            
            .chart-type-btn.active {
                border-color: var(--purple-accent) !important;
                background: rgba(139, 92, 246, 0.1) !important;
                color: var(--purple-accent) !important;
            }
            
            .indicator-btn {
                background: linear-gradient(135deg, var(--purple-accent) 0%, #7c3aed 100%) !important;
                color: white !important;
                min-height: 32px !important;
                padding: 0.5rem 1rem !important;
                font-size: 0.8rem !important;
            }
            
            .timeframe-grid {
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 0.25rem;
                margin-bottom: 1rem;
            }
            ''',

            PageTheme.BACKTESTING: '''
            /* Backtesting Specific - Analysis colors */
            .backtest-config {
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 1.5rem;
                padding: 1rem;
            }
            
            .config-section {
                background: var(--surface-color);
                border-radius: var(--radius-lg);
                padding: 1.5rem;
                border: 1px solid var(--border-color);
            }
            
            .config-group {
                margin-bottom: 1.5rem;
            }
            
            .config-group:last-child {
                margin-bottom: 0;
            }
            
            .run-backtest-btn {
                background: linear-gradient(135deg, var(--purple-accent) 0%, #6d28d9 100%) !important;
                color: white !important;
                min-height: 48px !important;
                padding: 1rem 2rem !important;
                font-size: 1rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 1px !important;
                width: 100% !important;
            }
            
            .results-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            ''',

            PageTheme.STRATEGIES: '''
            /* Strategies Specific - Strategy management colors and full width fixes */
            .strategy-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1rem;
                padding: 1rem;
            }
            
            .strategy-card {
                background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(31, 41, 55, 0.8) 100%) !important;
                border: 1px solid rgba(168, 85, 247, 0.2) !important;
                border-radius: 16px !important;
                padding: 1.5rem !important;
                backdrop-filter: blur(12px) !important;
                position: relative !important;
                overflow: hidden !important;
                transition: all 0.3s ease !important;
            }
            
            .strategy-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, #a855f7, #ec4899, #f59e0b);
                opacity: 0.6;
                transition: opacity 0.3s ease;
            }
            
            .strategy-card:hover::before {
                opacity: 1;
            }
            
            .create-strategy-btn {
                background: linear-gradient(135deg, var(--accent-color) 0%, #0284c7 100%) !important;
                color: white !important;
                min-height: 48px !important;
                padding: 1rem 2rem !important;
                font-weight: 600 !important;
                border-radius: var(--radius-lg) !important;
            }
            
            .strategy-actions {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 0.5rem;
                margin-top: 1rem;
            }
            
            .strategy-action-btn {
                min-height: 32px !important;
                padding: 0.5rem 1rem !important;
                font-size: 0.8rem !important;
                border-radius: 6px !important;
            }
            
            /* STRATEGIES PAGE SPECIFIC - Better card backgrounds without affecting global headers */
            .strategies-page .nicegui-card,
            .strategies-page .q-card {
                background: rgb(30, 41, 59) !important; /* slate-800 equivalent */
                border: 1px solid rgb(71, 85, 105) !important; /* slate-600 equivalent */
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04) !important;
                width: 100% !important;
            }
            
            .strategies-page .bg-slate-800 {
                background-color: rgb(30, 41, 59) !important;
            }
            
            .strategies-page .bg-slate-700 {
                background-color: rgb(51, 65, 85) !important;
            }
            
            .strategies-page .border-slate-600 {
                border-color: rgb(71, 85, 105) !important;
            }
            
            .strategies-page .border-slate-500 {
                border-color: rgb(100, 116, 139) !important;
            }
            
            .strategies-page .text-slate-200 {
                color: rgb(226, 232, 240) !important;
            }
            
            .strategies-page .text-slate-300 {
                color: rgb(203, 213, 225) !important;
            }
            
            /* STRATEGIES PAGE SPECIFIC - Full width utilization fixes */
            .strategies-page .q-tab-panels {
                width: 100% !important;
                max-width: none !important;
            }
            
            .strategies-page .q-tab-panel {
                width: 100% !important;
                max-width: none !important;
                padding: 1rem !important;
            }
            
            .strategies-page .q-card__section {
                width: 100% !important;
            }
            
            /* STRATEGIES PAGE SPECIFIC - Form elements full width */
            .strategies-page .nicegui-row.row {
                width: 100% !important;
            }
            
            .strategies-page .flex-1 {
                flex: 1 1 0% !important;
            }
            
            .strategies-page .w-full {
                width: 100% !important;
            }
            
            /* STRATEGIES PAGE SPECIFIC - Strategy form containers */
            .strategies-page .strategy-form-container {
                width: 100% !important;
                max-width: none !important;
            }
            
            .strategies-page .strategy-parameters-row {
                width: 100% !important;
                display: flex !important;
                gap: 1rem !important;
            }
            
            .strategies-page .strategy-parameters-row > * {
                flex: 1 !important;
            }
            
            /* STRATEGIES PAGE SPECIFIC - Header fixes for strategies page only */
            .strategies-page .dashboard-title-section {
                background: rgba(0, 0, 0, 0.1) !important;
                backdrop-filter: blur(10px) !important;
                border-bottom: 1px solid var(--border-color) !important;
                padding: 1rem !important;
                width: 100% !important;
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
            }
            
            .strategies-page .dashboard-title {
                color: var(--text-primary) !important;
                font-size: 2rem !important;
                font-weight: 600 !important;
                line-height: 1.2 !important;
            }
            ''',

            PageTheme.WATCHLIST: '''
            /* Watchlist Specific - Market monitoring colors */
            .watchlist-controls {
                display: grid;
                grid-template-columns: 1fr auto auto;
                gap: 1rem;
                align-items: center;
                padding: 1rem;
                background: var(--surface-color);
                border-radius: var(--radius-lg);
                margin-bottom: 1rem;
            }
            
            .add-symbol-btn {
                background: linear-gradient(135deg, var(--success-color) 0%, #15803d 100%) !important;
                color: white !important;
                min-height: 36px !important;
                padding: 0.5rem 1rem !important;
            }
            
            .refresh-btn {
                background: var(--surface-color) !important;
                border: 1px solid var(--border-color) !important;
                color: var(--text-secondary) !important;
                min-height: 36px !important;
            }
            
            .watchlist-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
            }
            
            .watchlist-item-row {
                transition: all 0.2s ease;
                cursor: pointer;
            }
            
            .watchlist-item-row:hover {
                background: var(--hover-color);
                transform: translateX(2px);
            }
            
            .price-up {
                color: var(--success-color) !important;
                background: rgba(34, 197, 94, 0.1) !important;
            }
            
            .price-down {
                color: var(--error-color) !important;
                background: rgba(239, 68, 68, 0.1) !important;
            }
            
            /* Watchlist expansion panels */
            .watchlist-expansion {
                background: var(--surface-color) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-md) !important;
            }
            
            .watchlist-expansion .q-expansion-item__label {
                color: var(--text-primary) !important;
            }
            
            .watchlist-expansion .q-expansion-item__content {
                background: var(--background) !important;
                border-top: 1px solid var(--border-color) !important;
                overflow: visible !important;
                padding: 0 !important;
            }
            
            .watchlist-expansion .q-expansion-item__container {
                width: 100% !important;
            }
            
            /* Grid layout for depth display */
            .depth-container {
                display: grid !important;
                grid-template-columns: 1fr auto 1fr !important;
                gap: 0.5rem !important;
                padding: 0.5rem !important;
                width: 100% !important;
            }
            
            .buy-column,
            .sell-column {
                display: flex !important;
                flex-direction: column !important;
                gap: 0.25rem !important;
            }
            
            /* Watchlist header with theme support */
            .watchlist-header {
                background: var(--surface-color) !important;
            }
            ''',

            PageTheme.SETTINGS: '''
            /* Settings Specific - Configuration colors */
            .settings-grid {
                display: grid;
                grid-template-columns: 250px 1fr;
                gap: 2rem;
                padding: 1rem;
                min-height: calc(100vh - 120px);
            }
            
            .settings-nav {
                background: var(--surface-color);
                border-radius: var(--radius-lg);
                padding: 1rem;
                border: 1px solid var(--border-color);
                height: fit-content;
            }
            
            .settings-content {
                background: var(--surface-color);
                border-radius: var(--radius-lg);
                padding: 2rem;
                border: 1px solid var(--border-color);
            }
            
            .settings-section {
                margin-bottom: 2rem;
                padding-bottom: 2rem;
                border-bottom: 1px solid var(--border-color);
            }
            
            .settings-section:last-child {
                margin-bottom: 0;
                padding-bottom: 0;
                border-bottom: none;
            }
            
            .save-settings-btn {
                background: linear-gradient(135deg, var(--success-color) 0%, #15803d 100%) !important;
                color: white !important;
                min-height: 44px !important;
                padding: 0.75rem 2rem !important;
                font-weight: 600 !important;
            }
            
            .reset-btn {
                background: var(--surface-color) !important;
                border: 1px solid var(--error-color) !important;
            }
            ''',
            
            PageTheme.SIP_STRATEGY: '''
            /* SIP Strategy Specific - Clean and theme-aware styling */
            
            /* Clear all nested element backgrounds to prevent layering */
            .sip-strategy-page * {
                background-color: transparent !important;
            }
            
            /* Main cards should have proper surface color only */
            .sip-strategy-page .q-card {
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-lg) !important;
                box-shadow: var(--shadow-md) !important;
                backdrop-filter: blur(var(--blur-amount)) !important;
                transition: all var(--transition-base) !important;
                max-width: 100% !important;
                margin: 0.25rem 0 !important;
            }
            
            .sip-strategy-page .q-card:hover {
                transform: translateY(-2px) !important;
                box-shadow: var(--shadow-lg) !important;
                border-color: var(--accent-color) !important;
            }
            
            /* Form controls with proper theme-aware styling */
            .sip-strategy-page .q-field__control {
                background: var(--surface-color) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-md) !important;
                min-height: var(--form-height-md) !important;
                transition: all var(--transition-fast) !important;
            }
            
            .sip-strategy-page .q-field__control:hover {
                border-color: var(--accent-color) !important;
                box-shadow: 0 0 0 2px rgba(34, 197, 252, 0.1) !important;
            }
            
            .sip-strategy-page .q-field__control:focus-within {
                border-color: var(--accent-color) !important;
                box-shadow: 0 0 0 3px rgba(34, 197, 252, 0.2) !important;
            }
            
            /* Input text styling */
            .sip-strategy-page .q-field input,
            .sip-strategy-page .q-field textarea,
            .sip-strategy-page .q-field .q-field__native {
                color: var(--text-primary) !important;
                background: transparent !important;
            }
            
            /* Labels */
            .sip-strategy-page .q-field__label {
                color: var(--text-secondary) !important;
                background: var(--primary-color) !important;
            }
            
            /* Buttons */
            .sip-strategy-page .q-btn {
                background: linear-gradient(135deg, var(--accent-color) 0%, #0284c7 100%) !important;
                color: white !important;
                border-radius: var(--radius-md) !important;
                min-height: var(--form-height-md) !important;
                font-weight: 500 !important;
                transition: all var(--transition-fast) !important;
            }
            
            .sip-strategy-page .q-btn:hover {
                transform: translateY(-1px) !important;
                box-shadow: var(--shadow-md) !important;
            }
            
            /* Layout fixes */
            .sip-strategy-page .q-row {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: wrap !important;
                align-items: flex-start !important;
                gap: var(--spacing-sm) !important;
                margin: var(--spacing-xs) 0 !important;
            }
            
            .sip-strategy-page .q-column {
                display: flex !important;
                flex-direction: column !important;
                align-items: stretch !important;
                gap: var(--spacing-xs) !important;
                margin: 0 !important;
            }
            
            .sip-strategy-page .q-field {
                margin: var(--spacing-xs) 0 !important;
            }
            ''',
            
            PageTheme.ENHANCED_SCANNER: '''
            /* Enhanced Scanner Specific - Professional trading scanner styling */
            
            /* Main scanner container */
            .enhanced-scanner-container {
                background: var(--background) !important;
                color: var(--text-primary) !important;
                min-height: 100vh !important;
                padding: 0 !important;
                width: 100% !important;
            }
            
            /* Scanner navigation tabs */
            .scanner-nav-container {
                display: flex !important;
                gap: 0.5rem !important;
                align-items: center !important;
                padding: 1rem !important;
                background: var(--surface-color) !important;
                border-radius: var(--radius-lg) !important;
                border: 1px solid var(--border-color) !important;
                margin-bottom: 1rem !important;
            }
            
            .scanner-nav-btn {
                background: var(--surface-color) !important;
                border: 2px solid var(--border-color) !important;
                color: var(--text-secondary) !important;
                min-height: 40px !important;
                padding: 0.5rem 1rem !important;
                border-radius: var(--radius-md) !important;
                font-weight: 500 !important;
                transition: all var(--transition-fast) !important;
                text-transform: none !important;
                font-size: 0.875rem !important;
            }
            
            .scanner-nav-btn:hover {
                background: var(--hover-color) !important;
                border-color: var(--accent-color) !important;
                color: var(--text-primary) !important;
                transform: translateY(-1px) !important;
            }
            
            .scanner-nav-btn.active {
                border-color: var(--accent-color) !important;
                background: rgba(34, 197, 252, 0.1) !important;
                color: var(--accent-color) !important;
            }
            
            /* Enhanced stat cards */
            .scanner-stat-card {
                background: linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.9) 100%) !important;
                border: 1px solid rgba(59, 130, 246, 0.2) !important;
                border-radius: 16px !important;
                padding: 1rem !important;
                backdrop-filter: blur(12px) !important;
                position: relative !important;
                overflow: hidden !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                min-width: 140px !important;
            }
            
            .scanner-stat-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981);
                opacity: 0.6;
            }
            
            .scanner-stat-card:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3) !important;
                border-color: rgba(59, 130, 246, 0.4) !important;
            }
            
            /* Scanner filter panels */
            .scanner-filter-panel {
                background: var(--surface-color) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-lg) !important;
                padding: 1.5rem !important;
                backdrop-filter: blur(var(--blur-amount)) !important;
                height: fit-content !important;
            }
            
            .scanner-filter-group {
                background: rgba(0, 0, 0, 0.1) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-md) !important;
                padding: 1rem !important;
                margin-bottom: 1rem !important;
            }
            
            .scanner-filter-group:last-child {
                margin-bottom: 0 !important;
            }
            
            /* Scanner results table */
            .scanner-results-table {
                background: var(--surface-color) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-lg) !important;
                overflow: hidden !important;
            }
            
            .scanner-table-header {
                background: rgba(0, 0, 0, 0.2) !important;
                padding: 0.75rem !important;
                border-bottom: 1px solid var(--border-color) !important;
                font-weight: 600 !important;
                font-size: 0.875rem !important;
                color: var(--text-secondary) !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
            }
            
            .scanner-table-row {
                padding: 0.75rem !important;
                border-bottom: 1px solid var(--border-color) !important;
                transition: all var(--transition-fast) !important;
                cursor: pointer !important;
            }
            
            .scanner-table-row:hover {
                background: var(--hover-color) !important;
                transform: translateX(2px) !important;
            }
            
            .scanner-table-row:last-child {
                border-bottom: none !important;
            }
            
            /* Signal strength indicators */
            .signal-strength-high {
                color: var(--success-color) !important;
                font-weight: 600 !important;
            }
            
            .signal-strength-medium {
                color: var(--warning-color) !important;
                font-weight: 500 !important;
            }
            
            .signal-strength-low {
                color: var(--text-secondary) !important;
            }
            
            /* Recommendation badges */
            .recommendation-strong-buy {
                background: linear-gradient(135deg, var(--success-color) 0%, #15803d 100%) !important;
                color: white !important;
                padding: 0.25rem 0.75rem !important;
                border-radius: var(--radius-full) !important;
                font-size: 0.75rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
            }
            
            .recommendation-buy {
                background: linear-gradient(135deg, var(--info-color) 0%, #1d4ed8 100%) !important;
                color: white !important;
                padding: 0.25rem 0.75rem !important;
                border-radius: var(--radius-full) !important;
                font-size: 0.75rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
            }
            
            .recommendation-hold {
                background: linear-gradient(135deg, var(--text-secondary) 0%, #6b7280 100%) !important;
                color: white !important;
                padding: 0.25rem 0.75rem !important;
                border-radius: var(--radius-full) !important;
                font-size: 0.75rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
            }
            
            .recommendation-sell {
                background: linear-gradient(135deg, var(--warning-color) 0%, #d97706 100%) !important;
                color: white !important;
                padding: 0.25rem 0.75rem !important;
                border-radius: var(--radius-full) !important;
                font-size: 0.75rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
            }
            
            .recommendation-strong-sell {
                background: linear-gradient(135deg, var(--error-color) 0%, #dc2626 100%) !important;
                color: white !important;
                padding: 0.25rem 0.75rem !important;
                border-radius: var(--radius-full) !important;
                font-size: 0.75rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
            }
            
            /* Scanner action buttons */
            .scanner-action-btn {
                background: linear-gradient(135deg, var(--accent-color) 0%, #0284c7 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: var(--radius-md) !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 600 !important;
                transition: all var(--transition-base) !important;
                box-shadow: 0 4px 12px rgba(34, 197, 252, 0.3) !important;
                min-height: 44px !important;
            }
            
            .scanner-action-btn:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 20px rgba(34, 197, 252, 0.4) !important;
            }
            
            /* Data explorer specific */
            .data-explorer-controls {
                display: flex !important;
                gap: 1rem !important;
                align-items: center !important;
                padding: 1rem !important;
                background: var(--surface-color) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-lg) !important;
                margin-bottom: 1rem !important;
            }
            
            .data-explorer-table {
                background: var(--surface-color) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--radius-lg) !important;
                overflow: hidden !important;
                max-height: 600px !important;
                overflow-y: auto !important;
            }
            
            /* Stock analysis specific */
            .stock-analysis-metrics {
                display: grid !important;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
                gap: 1rem !important;
                margin-bottom: 1.5rem !important;
            }
            
            .stock-metric-card {
                background: linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.9) 100%) !important;
                border: 1px solid rgba(59, 130, 246, 0.2) !important;
                border-radius: 12px !important;
                padding: 1.25rem !important;
                backdrop-filter: blur(12px) !important;
                text-align: center !important;
                transition: all 0.3s ease !important;
            }
            
            .stock-metric-card:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3) !important;
                border-color: rgba(59, 130, 246, 0.4) !important;
            }
            
            /* Responsive design */
            @media (max-width: 1200px) {
                .scanner-nav-container {
                    flex-wrap: wrap !important;
                }
                
                .scanner-filter-panel {
                    margin-bottom: 1rem !important;
                }
                
                .stock-analysis-metrics {
                    grid-template-columns: repeat(2, 1fr) !important;
                }
            }
            
            @media (max-width: 768px) {
                .scanner-nav-container {
                    flex-direction: column !important;
                    gap: 0.5rem !important;
                }
                
                .scanner-nav-btn {
                    width: 100% !important;
                    text-align: center !important;
                }
                
                .stock-analysis-metrics {
                    grid-template-columns: 1fr !important;
                }
                
                .data-explorer-controls {
                    flex-direction: column !important;
                    align-items: stretch !important;
                }
            }
            '''
        }

        return styles.get(page_type, '')

    def reset_styles(self):
        """Reset the styles flag - use when navigating between pages"""
        UnifiedThemeManager._styles_applied = False


# Singleton instance
unified_theme_manager = UnifiedThemeManager()


# Convenience functions
def apply_unified_theme(page: PageTheme, storage: Optional[Dict] = None):
    """Apply unified theme - call this ONCE per page load"""
    unified_theme_manager.apply_unified_theme(page, storage)


def reset_theme_styles():
    """Reset styles for navigation - call before switching pages"""
    unified_theme_manager.reset_styles()


def switch_unified_theme(new_theme: ThemeMode, storage: Optional[Dict] = None):
    """Switch theme and refresh"""
    unified_theme_manager.current_theme = new_theme
    if storage:
        storage['theme'] = new_theme.value
    unified_theme_manager.reset_styles()

    # Force page refresh to apply new theme
    def refresh_page():
        ui.navigate.reload()

    # Small delay to ensure storage is saved
    ui.timer(0.1, refresh_page, once=True)
