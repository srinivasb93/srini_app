# User Profile Module - profile.py
# Displays comprehensive user profile information and preferences

from nicegui import ui
import logging

logger = logging.getLogger(__name__)


async def render_profile_page(fetch_api, user_storage, broker):
    """
    Render comprehensive user profile page with all user data and preferences.
    
    Args:
        fetch_api: Function to call backend APIs
        user_storage: User storage dictionary
        broker: Current broker name
    """
    
    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):
        # Enhanced title section
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("person", size="2rem").classes("text-blue-400")
                    ui.label("User Profile").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("ACCOUNT", color="blue").classes("text-xs status-chip")
                
                ui.label("View and manage your account information and preferences").classes(
                    "text-gray-400 dashboard-subtitle")
            
            with ui.row().classes("items-center gap-4"):
                ui.button("Edit Settings", icon="settings", on_click=lambda: ui.navigate.to('/settings')).classes(
                    "text-cyan-400 border-cyan-400").props("outline")
        
        # Main content in responsive grid layout
        with ui.grid(columns=2).classes("w-full gap-6 p-6 grid-cols-1 lg:grid-cols-2"):
            # Left Column - User Information
            with ui.column().classes("gap-6"):
                await render_profile_info_card(fetch_api, broker)
                await render_api_config_card(fetch_api)
            
            # Right Column - User Preferences
            with ui.column().classes("gap-6"):
                await render_preferences_card(fetch_api)
                render_account_actions_card()


async def render_profile_info_card(fetch_api, broker):
    """Render basic profile information card"""
    with ui.card().classes("dashboard-card w-full"):
        with ui.row().classes("card-header w-full items-center p-6"):
            ui.icon("account_circle", size="1.5rem").classes("text-green-400")
            ui.label("Profile Information").classes("card-title")
        
        ui.separator().classes("card-separator")
        
        profile_container = ui.column().classes("w-full p-6 gap-4")
        
        async def load_profile_info():
            """Load user profile information from backend"""
            try:
                # Fetch profile from broker API
                profile = await fetch_api(f"/profile/{broker}")
                
                with profile_container:
                    if profile:
                        # User Info Grid
                        with ui.grid(columns=2).classes("w-full gap-4"):
                            # User ID
                            with ui.column().classes("gap-1"):
                                ui.label("User ID").classes("text-xs text-gray-400")
                                ui.label(profile.get("user_id", "N/A")).classes(
                                    "text-sm text-white font-mono")
                            
                            # Email
                            with ui.column().classes("gap-1"):
                                ui.label("Email").classes("text-xs text-gray-400")
                                ui.label(profile.get("email", "N/A")).classes("text-sm text-white")
                            
                            # Name
                            with ui.column().classes("gap-1"):
                                ui.label("Name").classes("text-xs text-gray-400")
                                ui.label(profile.get("name", "N/A")).classes("text-sm text-white")
                            
                            # Broker
                            with ui.column().classes("gap-1"):
                                ui.label("Current Broker").classes("text-xs text-gray-400")
                                ui.label(broker).classes("text-sm text-cyan-400 font-semibold")
                    else:
                        ui.label("Unable to load profile information").classes("text-red-400")
            except Exception as e:
                logger.error(f"Error loading profile: {e}")
                with profile_container:
                    ui.label(f"Error: {str(e)}").classes("text-red-400 text-sm")
        
        ui.timer(0.1, load_profile_info, once=True)


async def render_api_config_card(fetch_api):
    """Render API configuration card showing configured brokers"""
    with ui.card().classes("dashboard-card w-full"):
        with ui.row().classes("card-header w-full items-center p-6"):
            ui.icon("key", size="1.5rem").classes("text-yellow-400")
            ui.label("API Configuration").classes("card-title")
        
        ui.separator().classes("card-separator")
        
        api_keys_container = ui.column().classes("w-full p-6 gap-4")
        
        async def load_api_keys():
            """Load API keys information from backend"""
            try:
                api_keys = await fetch_api("/users/me/api-keys")
                
                with api_keys_container:
                    if api_keys:
                        # Show configured brokers
                        configured_brokers = api_keys.get("configured_brokers", [])
                        
                        if configured_brokers:
                            ui.label("Configured Brokers").classes("text-sm text-gray-400 mb-2")
                            for broker_name in configured_brokers:
                                with ui.row().classes("items-center gap-2 mb-2"):
                                    ui.icon("check_circle", size="1rem").classes("text-green-400")
                                    ui.label(broker_name).classes("text-white")
                        else:
                            ui.label("No brokers configured").classes("text-gray-400")
                        
                        # Show if multiple brokers available
                        if api_keys.get("has_multiple_brokers"):
                            with ui.card().classes("w-full bg-blue-900/20 border border-blue-500/30 mt-4"):
                                with ui.row().classes("p-3 items-center gap-2"):
                                    ui.icon("info", size="1rem").classes("text-blue-400")
                                    ui.label("Multiple brokers available - switch in settings").classes(
                                        "text-sm text-blue-300")
                    else:
                        ui.label("Unable to load API configuration").classes("text-red-400")
            except Exception as e:
                logger.error(f"Error loading API keys: {e}")
                with api_keys_container:
                    ui.label(f"Error: {str(e)}").classes("text-red-400 text-sm")
        
        ui.timer(0.1, load_api_keys, once=True)


async def render_preferences_card(fetch_api):
    """Render user preferences card with expandable sections"""
    with ui.card().classes("dashboard-card w-full"):
        with ui.row().classes("card-header w-full items-center p-6"):
            ui.icon("tune", size="1.5rem").classes("text-purple-400")
            ui.label("Current Preferences").classes("card-title")
        
        ui.separator().classes("card-separator")
        
        preferences_container = ui.column().classes("w-full p-6 gap-4")
        
        async def load_preferences():
            """Load user preferences from backend"""
            try:
                prefs_response = await fetch_api("/user/preferences")
                
                with preferences_container:
                    if prefs_response and prefs_response.get("status") == "success":
                        preferences = prefs_response.get("preferences", {})
                        
                        # Risk Management Section
                        with ui.expansion("Risk Management", icon="security").classes("w-full mb-3"):
                            with ui.column().classes("gap-2 p-2"):
                                ui.label(f"Daily Loss Limit: ₹{preferences.get('daily_loss_limit', 10000):,}").classes(
                                    "text-sm text-white")
                                ui.label(f"Position Size Limit: ₹{preferences.get('position_size_limit', 50000):,}").classes(
                                    "text-sm text-white")
                                ui.label(f"Max Open Positions: {preferences.get('max_open_positions', 10)}").classes(
                                    "text-sm text-white")
                                ui.label(f"Risk per Trade: {preferences.get('risk_per_trade', 2.0)}%").classes(
                                    "text-sm text-white")
                                auto_stop = "✓ Enabled" if preferences.get('auto_stop_trading', True) else "✗ Disabled"
                                auto_stop_color = "text-sm text-green-400" if preferences.get('auto_stop_trading', True) else "text-sm text-red-400"
                                ui.label(f"Auto-stop Trading: {auto_stop}").classes(auto_stop_color)
                        
                        # Trading Preferences Section
                        with ui.expansion("Trading Preferences", icon="trending_up").classes("w-full mb-3"):
                            with ui.column().classes("gap-2 p-2"):
                                ui.label(f"Default Order Type: {preferences.get('default_order_type', 'MARKET')}").classes(
                                    "text-sm text-white")
                                ui.label(f"Default Product Type: {preferences.get('default_product_type', 'CNC')}").classes(
                                    "text-sm text-white")
                                ui.label(f"Refresh Interval: {preferences.get('refresh_interval', 5)}s").classes(
                                    "text-sm text-white")
                        
                        # Notification Settings
                        with ui.expansion("Notifications", icon="notifications").classes("w-full mb-3"):
                            with ui.column().classes("gap-2 p-2"):
                                order_alerts = "✓ Enabled" if preferences.get('order_alerts', True) else "✗ Disabled"
                                ui.label(f"Order Alerts: {order_alerts}").classes("text-sm text-white")
                                
                                pnl_alerts = "✓ Enabled" if preferences.get('pnl_alerts', True) else "✗ Disabled"
                                ui.label(f"P&L Alerts: {pnl_alerts}").classes("text-sm text-white")
                                
                                strategy_alerts = "✓ Enabled" if preferences.get('strategy_alerts', True) else "✗ Disabled"
                                ui.label(f"Strategy Alerts: {strategy_alerts}").classes("text-sm text-white")
                        
                        # Edit button
                        ui.button("Edit Preferences", icon="edit", 
                                 on_click=lambda: ui.navigate.to('/settings')).classes(
                            "w-full mt-4 text-cyan-400 border-cyan-400").props("outline")
                    else:
                        ui.label("Unable to load preferences").classes("text-red-400")
            except Exception as e:
                logger.error(f"Error loading preferences: {e}")
                with preferences_container:
                    ui.label(f"Error: {str(e)}").classes("text-red-400 text-sm")
        
        ui.timer(0.1, load_preferences, once=True)


def render_account_actions_card():
    """Render account actions card with quick links"""
    with ui.card().classes("dashboard-card w-full"):
        with ui.row().classes("card-header w-full items-center p-6"):
            ui.icon("admin_panel_settings", size="1.5rem").classes("text-red-400")
            ui.label("Account Actions").classes("card-title")
        
        ui.separator().classes("card-separator")
        
        with ui.column().classes("w-full p-6 gap-3"):
            ui.button("Change Settings", icon="settings", 
                     on_click=lambda: ui.navigate.to('/settings')).classes(
                "w-full text-cyan-400 border-cyan-400").props("outline")
            
            ui.button("Manage Brokers", icon="link", 
                     on_click=lambda: ui.navigate.to('/settings')).classes(
                "w-full text-green-400 border-green-400").props("outline")
            
            ui.button("View Risk Metrics", icon="analytics", 
                     on_click=lambda: ui.navigate.to('/analytics')).classes(
                "w-full text-purple-400 border-purple-400").props("outline")

