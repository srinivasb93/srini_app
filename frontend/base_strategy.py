# frontend/base_strategy.py
"""
Base Strategy Class for Frontend Strategy Components
Provides common functionality and interface for all strategy UI components
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json

from nicegui import ui

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all strategy UI components.
    Provides common functionality and standardized interface.
    """

    def __init__(self):
        self.strategy_name = "Base Strategy"
        self.description = "Base strategy component"
        self.version = "1.0.0"
        self.author = "Trading System"
        self.created_at = datetime.now()

        # Common UI state
        self.is_loading = False
        self.last_error = None
        self.ui_refs = {}

        # Common configuration
        self.default_config = {}
        self.current_config = {}

        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    async def render(self, fetch_api: Callable, user_storage: Dict) -> None:
        """
        Main render method that each strategy must implement.

        Args:
            fetch_api: Function to make API calls
            user_storage: User-specific storage dictionary
        """
        pass

    # ============================================================================
    # COMMON UI HELPER METHODS
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

    # ============================================================================
    # COMMON DATA HANDLING METHODS
    # ============================================================================

    async def safe_api_call(self, fetch_api: Callable, endpoint: str,
                            method: str = "GET", data: Optional[Dict] = None,
                            error_message: str = "API call failed") -> Optional[Dict]:
        """
        Safely make API calls with error handling.

        Args:
            fetch_api: API function
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            error_message: Error message to show

        Returns:
            API response or None on error
        """
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

    def format_date(self, date_obj: datetime, format_str: str = "%Y-%m-%d") -> str:
        """Format date for display"""
        if date_obj is None:
            return "N/A"
        if isinstance(date_obj, str):
            try:
                date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
            except:
                return date_obj
        return date_obj.strftime(format_str)

    def calculate_return_color(self, return_value: float) -> str:
        """Get color class based on return value"""
        if return_value > 5:
            return "text-green-600"
        elif return_value > 0:
            return "text-green-500"
        elif return_value > -5:
            return "text-red-500"
        else:
            return "text-red-600"

    # ============================================================================
    # COMMON UI COMPONENTS
    # ============================================================================

    def create_metric_card(self, title: str, value: str, subtitle: Optional[str] = None,
                           icon: Optional[str] = None, color_class: str = "text-blue-600") -> None:
        """Create a metric display card"""
        with ui.card().classes("p-4 text-center"):
            if icon:
                ui.label(icon).classes("text-2xl mb-2")
            ui.label(title).classes("text-sm text-gray-600")
            ui.label(value).classes(f"text-xl font-bold {color_class}")
            if subtitle:
                ui.label(subtitle).classes("text-xs text-gray-500 mt-1")

    def create_status_badge(self, status: str) -> None:
        """Create a status badge with appropriate styling"""
        status_colors = {
            "active": "bg-green-100 text-green-800",
            "inactive": "bg-gray-100 text-gray-800",
            "cancelled": "bg-yellow-100 text-yellow-800",
            "deleted": "bg-red-100 text-red-800",
            "pending": "bg-blue-100 text-blue-800",
            "completed": "bg-green-100 text-green-800",
            "failed": "bg-red-100 text-red-800"
        }

        color_class = status_colors.get(status.lower(), "bg-gray-100 text-gray-800")
        ui.label(status.title()).classes(f"px-2 py-1 rounded text-xs {color_class}")

    def create_action_button(self, label: str, icon: str, callback: Callable,
                             color: str = "blue", size: str = "sm") -> None:
        """Create a standardized action button"""
        color_classes = {
            "blue": "bg-blue-500 hover:bg-blue-600",
            "green": "bg-green-500 hover:bg-green-600",
            "red": "bg-red-500 hover:bg-red-600",
            "yellow": "bg-yellow-500 hover:bg-yellow-600",
            "purple": "bg-purple-500 hover:bg-purple-600",
            "gray": "bg-gray-500 hover:bg-gray-600"
        }

        size_classes = {
            "xs": "text-xs px-2 py-1",
            "sm": "text-sm px-3 py-1",
            "md": "text-md px-4 py-2",
            "lg": "text-lg px-6 py-3"
        }

        color_class = color_classes.get(color, color_classes["blue"])
        size_class = size_classes.get(size, size_classes["sm"])

        ui.button(f"{icon} {label}", on_click=callback).classes(
            f"{color_class} text-white {size_class} rounded"
        )

    def create_confirmation_dialog(self, title: str, message: str,
                                   confirm_callback: Callable,
                                   confirm_text: str = "Confirm",
                                   cancel_text: str = "Cancel") -> None:
        """Create a confirmation dialog"""
        with ui.dialog() as dialog, ui.card():
            ui.label(title).classes("text-lg font-bold mb-4")
            ui.label(message).classes("text-gray-600 mb-4")

            with ui.row().classes("gap-2"):
                ui.button(cancel_text, on_click=dialog.close).classes("bg-gray-500 text-white")

                async def confirm_and_close():
                    await confirm_callback()
                    dialog.close()

                ui.button(confirm_text, on_click=confirm_and_close).classes("bg-red-500 text-white")

        dialog.open()

    # ============================================================================
    # CONFIGURATION MANAGEMENT
    # ============================================================================

    def save_config(self, user_storage: Dict, config: Dict) -> None:
        """Save configuration to user storage"""
        try:
            storage_key = f"{self.strategy_name.lower().replace(' ', '_')}_config"
            user_storage[storage_key] = config
            self.current_config = config.copy()
            self.show_success("Configuration saved")
        except Exception as e:
            self.show_error("Failed to save configuration", str(e))

    def load_config(self, user_storage: Dict) -> Dict:
        """Load configuration from user storage"""
        try:
            storage_key = f"{self.strategy_name.lower().replace(' ', '_')}_config"
            config = user_storage.get(storage_key, self.default_config.copy())
            self.current_config = config.copy()
            return config
        except Exception as e:
            self.show_error("Failed to load configuration", str(e))
            return self.default_config.copy()

    def reset_config(self, user_storage: Dict) -> Dict:
        """Reset configuration to defaults"""
        self.current_config = self.default_config.copy()
        self.save_config(user_storage, self.current_config)
        return self.current_config

    # ============================================================================
    # VALIDATION HELPERS
    # ============================================================================

    def validate_number_input(self, value: Any, min_value: Optional[float] = None,
                              max_value: Optional[float] = None,
                              field_name: str = "Value") -> bool:
        """Validate numeric input"""
        try:
            num_value = float(value)

            if min_value is not None and num_value < min_value:
                self.show_error(f"{field_name} must be at least {min_value}")
                return False

            if max_value is not None and num_value > max_value:
                self.show_error(f"{field_name} must be at most {max_value}")
                return False

            return True

        except (ValueError, TypeError):
            self.show_error(f"{field_name} must be a valid number")
            return False

    def validate_date_input(self, date_str: str, field_name: str = "Date") -> bool:
        """Validate date input"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            self.show_error(f"{field_name} must be in YYYY-MM-DD format")
            return False

    def validate_required_field(self, value: Any, field_name: str) -> bool:
        """Validate required field"""
        if not value or (isinstance(value, str) and not value.strip()):
            self.show_error(f"{field_name} is required")
            return False
        return True

    # ============================================================================
    # DATA EXPORT/IMPORT HELPERS
    # ============================================================================

    def export_data_as_json(self, data: Dict, filename: str) -> None:
        """Export data as JSON (placeholder for actual implementation)"""
        try:
            # In a real implementation, this would trigger a download
            json_data = json.dumps(data, indent=2, default=str)
            self.show_success(f"Data exported successfully")
            logger.info(f"Exported data for {self.strategy_name}: {len(json_data)} characters")
        except Exception as e:
            self.show_error("Failed to export data", str(e))

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            "name": self.strategy_name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "current_config": self.current_config,
            "default_config": self.default_config
        }

    # ============================================================================
    # COMMON CHART HELPERS (for strategies that need charts)
    # ============================================================================

    def create_performance_chart_placeholder(self, data: List[Dict]) -> None:
        """Create a placeholder for performance charts"""
        with ui.card().classes("w-full p-4"):
            ui.label("📊 Performance Chart").classes("text-lg font-bold mb-4")

            if not data:
                ui.label("No data available for charting").classes("text-gray-500 text-center py-8")
                return

            # Placeholder for actual chart implementation
            ui.label(f"Chart would display {len(data)} data points").classes("text-center py-8 text-gray-600")
            ui.label("📈 Chart implementation pending").classes("text-center text-blue-600")

    # ============================================================================
    # LOGGING AND DEBUGGING
    # ============================================================================

    def log_user_action(self, action: str, details: Optional[Dict] = None) -> None:
        """Log user actions for debugging and analytics"""
        log_entry = {
            "strategy": self.strategy_name,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        logger.info(f"User Action: {json.dumps(log_entry)}")

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for troubleshooting"""
        return {
            "strategy_name": self.strategy_name,
            "is_loading": self.is_loading,
            "last_error": self.last_error,
            "current_config": self.current_config,
            "ui_refs_count": len(self.ui_refs),
            "created_at": self.created_at.isoformat()
        }