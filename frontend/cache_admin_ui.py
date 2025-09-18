"""
Cache Administration UI Components
Provides UI components for cache monitoring and management
"""

from nicegui import ui
import logging
from cache_monitor import cache_monitor, get_cache_health_dashboard
from cache_manager import frontend_cache
from cache_invalidation import emergency_cache_clear

logger = logging.getLogger(__name__)

async def render_cache_admin_panel():
    """Render cache administration panel"""
    
    with ui.card().classes("dashboard-card w-full"):
        with ui.row().classes("card-header w-full justify-between items-center p-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("memory", size="1.5rem").classes("text-blue-400")
                ui.label("Cache Administration").classes("card-title")
            
            ui.button("Refresh Data", icon="refresh", on_click=lambda: refresh_cache_data()).props("flat").classes("text-cyan-400")
        
        ui.separator().classes("card-separator")
        
        # Cache health overview
        await render_cache_health_overview()
        
        ui.separator().classes("my-4")
        
        # Cache actions
        await render_cache_actions()
        
        ui.separator().classes("my-4")
        
        # Cache statistics
        await render_cache_statistics()

async def render_cache_health_overview():
    """Render cache health overview section"""
    
    with ui.column().classes("w-full p-4"):
        ui.label("Cache Health Overview").classes("text-lg font-semibold text-white mb-4")
        
        try:
            dashboard_data = get_cache_health_dashboard()
            health_report = dashboard_data.get('health_report', {})
            alerts = dashboard_data.get('alerts', [])
            
            # Health status
            overall_health = health_report.get('overall_health', 'UNKNOWN')
            health_color = get_health_color(overall_health)
            
            with ui.row().classes("w-full gap-4 mb-4"):
                # Overall health card
                with ui.card().classes("dashboard-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("health_and_safety", size="2rem").classes(f"{health_color} mb-2")
                        ui.label("Overall Health").classes("text-sm text-gray-400")
                        ui.label(overall_health).classes(f"text-xl font-bold {health_color}")
                
                # Hit rate card
                performance = health_report.get('performance', {})
                hit_rate = performance.get('hit_rate', 0)
                hit_rate_color = get_performance_color(hit_rate)
                
                with ui.card().classes("dashboard-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("track_changes", size="2rem").classes(f"{hit_rate_color} mb-2")
                        ui.label("Hit Rate").classes("text-sm text-gray-400")
                        ui.label(f"{hit_rate:.1f}%").classes(f"text-xl font-bold {hit_rate_color}")
                
                # Memory usage card
                memory = health_report.get('memory', {})
                usage_percent = memory.get('usage_percent', 0)
                memory_color = get_memory_color(usage_percent)
                
                with ui.card().classes("dashboard-card flex-1"):
                    with ui.column().classes("p-4 text-center"):
                        ui.icon("storage", size="2rem").classes(f"{memory_color} mb-2")
                        ui.label("Memory Usage").classes("text-sm text-gray-400")
                        ui.label(f"{usage_percent:.1f}%").classes(f"text-xl font-bold {memory_color}")
            
            # Alerts section
            if alerts:
                ui.label("Active Alerts").classes("text-lg font-semibold text-yellow-400 mb-2")
                for alert in alerts:
                    severity_color = get_severity_color(alert.get('severity', 'LOW'))
                    with ui.card().classes(f"w-full border-l-4 border-{severity_color.split('-')[1]}-500"):
                        with ui.row().classes("items-center p-3"):
                            ui.icon("warning", size="1.2rem").classes(severity_color)
                            with ui.column().classes("flex-1 ml-2"):
                                ui.label(alert.get('message', 'Unknown alert')).classes("text-white font-medium")
                                ui.label(alert.get('recommendation', '')).classes("text-gray-400 text-sm")
            else:
                with ui.row().classes("items-center text-green-400 mb-2"):
                    ui.icon("check_circle", size="1.2rem")
                    ui.label("No active alerts").classes("ml-2")
            
        except Exception as e:
            logger.error(f"Error rendering cache health overview: {e}")
            ui.label("Error loading cache health data").classes("text-red-400")

async def render_cache_actions():
    """Render cache management actions"""
    
    with ui.column().classes("w-full p-4"):
        ui.label("Cache Management Actions").classes("text-lg font-semibold text-white mb-4")
        
        with ui.row().classes("w-full gap-4"):
            # Cache invalidation actions
            with ui.card().classes("dashboard-card flex-1"):
                with ui.column().classes("p-4"):
                    ui.label("Cache Invalidation").classes("text-white font-medium mb-2")
                    
                    with ui.row().classes("gap-2 mb-2"):
                        ui.button("Clear Strategy Cache", icon="psychology", 
                                 on_click=lambda: clear_strategy_cache(),
                                 color="orange").classes("text-xs")
                        ui.button("Clear Position Cache", icon="donut_small",
                                 on_click=lambda: clear_position_cache(),
                                 color="blue").classes("text-xs")
                    
                    with ui.row().classes("gap-2"):
                        ui.button("Clear Market Data", icon="trending_up",
                                 on_click=lambda: clear_market_cache(),
                                 color="green").classes("text-xs")
                        ui.button("CLEAR ALL", icon="delete_forever",
                                 on_click=lambda: confirm_clear_all_cache(),
                                 color="red").classes("text-xs")
            
            # Cache monitoring actions
            with ui.card().classes("dashboard-card flex-1"):
                with ui.column().classes("p-4"):
                    ui.label("Monitoring Controls").classes("text-white font-medium mb-2")
                    
                    with ui.row().classes("gap-2 mb-2"):
                        ui.button("Enable Monitoring", icon="visibility",
                                 on_click=lambda: toggle_monitoring(True),
                                 color="green").classes("text-xs")
                        ui.button("Disable Monitoring", icon="visibility_off",
                                 on_click=lambda: toggle_monitoring(False),
                                 color="gray").classes("text-xs")
                    
                    with ui.row().classes("gap-2"):
                        ui.button("Log Performance", icon="assessment",
                                 on_click=lambda: log_cache_performance(),
                                 color="blue").classes("text-xs")
                        ui.button("Export Stats", icon="download",
                                 on_click=lambda: export_cache_stats(),
                                 color="purple").classes("text-xs")

async def render_cache_statistics():
    """Render detailed cache statistics"""
    
    with ui.column().classes("w-full p-4"):
        ui.label("Detailed Statistics").classes("text-lg font-semibold text-white mb-4")
        
        try:
            stats = frontend_cache.get_stats()
            
            # Statistics in a grid
            with ui.row().classes("w-full gap-4"):
                # Performance stats
                with ui.card().classes("dashboard-card flex-1"):
                    with ui.column().classes("p-4"):
                        ui.label("Performance Metrics").classes("text-white font-medium mb-2")
                        
                        stats_items = [
                            ("Total Requests", stats.get('total_requests', 0)),
                            ("Cache Hits", stats.get('hits', 0)),
                            ("Cache Misses", stats.get('misses', 0)),
                            ("Hit Rate", f"{stats.get('hit_rate', 0):.2f}%"),
                        ]
                        
                        for label, value in stats_items:
                            with ui.row().classes("justify-between items-center mb-1"):
                                ui.label(label).classes("text-gray-400 text-sm")
                                ui.label(str(value)).classes("text-white text-sm font-mono")
                
                # Memory stats
                with ui.card().classes("dashboard-card flex-1"):
                    with ui.column().classes("p-4"):
                        ui.label("Memory Metrics").classes("text-white font-medium mb-2")
                        
                        memory_stats = [
                            ("Current Entries", stats.get('total_entries', 0)),
                            ("Max Capacity", stats.get('max_size', 0)),
                            ("Evictions", stats.get('evictions', 0)),
                            ("Usage", f"{(stats.get('total_entries', 0) / max(stats.get('max_size', 1), 1) * 100):.1f}%"),
                        ]
                        
                        for label, value in memory_stats:
                            with ui.row().classes("justify-between items-center mb-1"):
                                ui.label(label).classes("text-gray-400 text-sm")
                                ui.label(str(value)).classes("text-white text-sm font-mono")
            
        except Exception as e:
            logger.error(f"Error rendering cache statistics: {e}")
            ui.label("Error loading cache statistics").classes("text-red-400")

# Helper functions
def get_health_color(health_status: str) -> str:
    """Get color class for health status"""
    color_map = {
        'EXCELLENT': 'text-green-400',
        'GOOD': 'text-blue-400', 
        'FAIR': 'text-yellow-400',
        'POOR': 'text-red-400',
        'UNKNOWN': 'text-gray-400'
    }
    return color_map.get(health_status, 'text-gray-400')

def get_performance_color(hit_rate: float) -> str:
    """Get color class for performance metrics"""
    if hit_rate >= 80:
        return 'text-green-400'
    elif hit_rate >= 60:
        return 'text-blue-400'
    elif hit_rate >= 40:
        return 'text-yellow-400'
    else:
        return 'text-red-400'

def get_memory_color(usage_percent: float) -> str:
    """Get color class for memory usage"""
    if usage_percent >= 90:
        return 'text-red-400'
    elif usage_percent >= 75:
        return 'text-yellow-400'
    elif usage_percent >= 50:
        return 'text-blue-400'
    else:
        return 'text-green-400'

def get_severity_color(severity: str) -> str:
    """Get color class for alert severity"""
    color_map = {
        'HIGH': 'text-red-400',
        'MEDIUM': 'text-yellow-400',
        'LOW': 'text-blue-400'
    }
    return color_map.get(severity, 'text-gray-400')

# Action handlers
def refresh_cache_data():
    """Refresh cache data display"""
    ui.notify("Cache data refreshed", type="positive")

def clear_strategy_cache():
    """Clear strategy-related caches"""
    try:
        frontend_cache.delete_pattern("fetch_api:*/strategies*")
        frontend_cache.delete_pattern("fetch_api:*/executions*")
        frontend_cache.delete_pattern("safe_api:*/strategies*")
        frontend_cache.delete_pattern("safe_api:*/executions*")
        ui.notify("Strategy cache cleared successfully", type="positive")
    except Exception as e:
        ui.notify(f"Failed to clear strategy cache: {e}", type="negative")

def clear_position_cache():
    """Clear position-related caches"""
    try:
        frontend_cache.delete_pattern("fetch_api:*/positions*")
        frontend_cache.delete_pattern("fetch_api:*/orders*")
        frontend_cache.delete_pattern("positions:*")
        frontend_cache.delete_pattern("orders:*")
        ui.notify("Position cache cleared successfully", type="positive")
    except Exception as e:
        ui.notify(f"Failed to clear position cache: {e}", type="negative")

def clear_market_cache():
    """Clear market data caches"""
    try:
        frontend_cache.delete_pattern("fetch_api:*/quotes*")
        frontend_cache.delete_pattern("fetch_api:*/ltp*")
        frontend_cache.delete_pattern("live_quotes*")
        frontend_cache.delete_pattern("market_data*")
        ui.notify("Market data cache cleared successfully", type="positive")
    except Exception as e:
        ui.notify(f"Failed to clear market cache: {e}", type="negative")

def confirm_clear_all_cache():
    """Confirm and clear all caches"""
    with ui.dialog() as dialog, ui.card():
        ui.label("Clear All Cache Data?").classes("text-h6 mb-4")
        ui.label("This will clear all cached data and may impact performance temporarily.").classes("text-body2 mb-4")
        
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("CLEAR ALL", color="red", on_click=lambda: execute_clear_all(dialog))
    
    dialog.open()

def execute_clear_all(dialog):
    """Execute clearing all caches"""
    try:
        emergency_cache_clear()
        ui.notify("All caches cleared successfully", type="warning")
        dialog.close()
    except Exception as e:
        ui.notify(f"Failed to clear all caches: {e}", type="negative")
        dialog.close()

def toggle_monitoring(enable: bool):
    """Toggle cache monitoring"""
    try:
        if enable:
            cache_monitor.enable_monitoring()
            ui.notify("Cache monitoring enabled", type="positive")
        else:
            cache_monitor.disable_monitoring()
            ui.notify("Cache monitoring disabled", type="info")
    except Exception as e:
        ui.notify(f"Failed to toggle monitoring: {e}", type="negative")

def log_cache_performance():
    """Log current cache performance"""
    try:
        cache_monitor.log_cache_performance()
        ui.notify("Performance logged successfully", type="positive")
    except Exception as e:
        ui.notify(f"Failed to log performance: {e}", type="negative")

def export_cache_stats():
    """Export cache statistics"""
    try:
        stats = frontend_cache.get_stats()
        # In a real implementation, this would generate a downloadable file
        ui.notify("Cache statistics exported (feature in development)", type="info")
    except Exception as e:
        ui.notify(f"Failed to export statistics: {e}", type="negative")