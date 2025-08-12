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
            --radius-xl: 24px;
            --radius-full: 9999px;
            
            --transition-fast: 0.15s ease;
            --transition-base: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 0.5s ease;
            
            /* CONSISTENT SIZING SYSTEM */
            --size-xs: 0.25rem;    /* 4px */
            --size-sm: 0.5rem;     /* 8px */
            --size-md: 1rem;       /* 16px */
            --size-lg: 1.5rem;     /* 24px */
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
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
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
            min-height: 40px !important;
            margin-bottom: 0.5rem !important;
            display: flex !important;
            align-items: center !important;
        }
        
        /* CRITICAL: Fix for main field control container */
        .q-field .q-field__control {
            background: var(--surface-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            min-height: 40px !important;
            padding: 0 !important;
            transition: all var(--transition-fast) !important;
            position: relative !important;
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
        }
        
        /* CRITICAL: Fix the nested control container alignment */
        .q-field__control-container {
            min-height: 38px !important;
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
            min-height: 36px !important;
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
            min-height: 36px !important;
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
        
        @media (max-width: 1024px) {
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
            padding: 1rem !important;
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
            font-size: 2.5rem !important;
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
        .q-header .text-xl,
        .theme-text-primary {
            color: #1e293b !important;
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
        
        /* Status indicators light theme */
        .theme-text-secondary {
            color: #475569 !important;
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
        '''

    def _get_page_specific_styles(self, page_type: PageTheme) -> str:
        """Get comprehensive page-specific styles with context-aware colors and proper spacing"""

        styles = {
            PageTheme.DASHBOARD: '''
            /* Dashboard Specific - Compact grid layout */
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
            
            .quick-trade-form {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.75rem;
                padding: 1rem;
            }
            
            .quick-trade-form .q-field,
            .quick-trade-form .q-select {
                min-height: 40px !important;
            }
            
            .trade-buttons {
                grid-column: 1 / -1;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.75rem;
                margin-top: 0.5rem;
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
            /* Order Management Specific - Professional trading colors */
            .order-form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 1rem;
                padding: 1rem;
                background: var(--surface-color);
                border-radius: var(--radius-lg);
                margin-bottom: 1rem;
            }
            
            .order-type-btn {
                background: var(--surface-color) !important;
                border: 2px solid var(--border-color) !important;
                color: var(--text-secondary) !important;
                min-height: 40px !important;
                transition: all 0.2s ease !important;
            }
            
            .order-type-btn.active {
                border-color: var(--accent-color) !important;
                background: rgba(34, 197, 252, 0.1) !important;
                color: var(--accent-color) !important;
            }
            
            .limit-order-btn.active {
                border-color: var(--info-color) !important;
                background: rgba(59, 130, 246, 0.1) !important;
                color: var(--info-color) !important;
            }
            
            .market-order-btn.active {
                border-color: var(--warning-color) !important;
                background: rgba(245, 158, 11, 0.1) !important;
                color: var(--warning-color) !important;
            }
            
            .stop-loss-btn.active {
                border-color: var(--error-color) !important;
                background: rgba(239, 68, 68, 0.1) !important;
                color: var(--error-color) !important;
            }
            
            .order-actions {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin-top: 1rem;
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
            .positions-header-grid {
                display: grid;
                grid-template-columns: 1.2fr 1fr 1fr 1fr 1.2fr 1fr 1fr 1fr 0.8fr;
                gap: 0.5rem;
                align-items: center;
                padding: 1rem 0.75rem;
                background: rgba(0, 0, 0, 0.2);
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-size: 0.75rem;
                color: var(--text-secondary);
                border-bottom: 1px solid var(--border-color);
            }
            
            .positions-row-grid {
                display: grid;
                grid-template-columns: 1.2fr 1fr 1fr 1fr 1.2fr 1fr 1fr 1fr 0.8fr;
                gap: 0.5rem;
                align-items: center;
                padding: 0.75rem;
                border-bottom: 1px solid var(--border-color);
                transition: background 0.2s ease;
            }
            
            .positions-row-grid:hover {
                background: var(--hover-color);
            }
            
            .position-long {
                border-left: 4px solid var(--success-color);
                background: linear-gradient(90deg, rgba(34, 197, 94, 0.05) 0%, transparent 100%);
            }
            
            .position-short {
                border-left: 4px solid var(--error-color);
                background: linear-gradient(90deg, rgba(239, 68, 68, 0.05) 0%, transparent 100%);
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
