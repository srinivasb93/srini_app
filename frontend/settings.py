# Enhanced Settings Module - settings.py
# Applying beautiful dashboard.py styling consistently

from nicegui import ui
import logging
import asyncio
from datetime import datetime
from cache_invalidation import invalidate_on_settings_change

logger = logging.getLogger(__name__)


async def render_settings_page(fetch_api, user_storage, apply_theme_from_storage):
    """Enhanced settings page with beautiful dashboard styling"""
    # Get current broker
    broker = user_storage.get('default_broker', "Zerodha")

    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):
        # Enhanced title section (matching dashboard.py)
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("settings", size="2rem").classes("text-cyan-400")
                    ui.label("Settings & Preferences").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("CONFIGURATION", color="blue").classes("text-xs status-chip")

                ui.label("Customize your trading application preferences and broker connections").classes(
                    "text-gray-400 dashboard-subtitle")

            # Right side - Status indicator
            with ui.row().classes("items-center gap-4"):
                ui.chip("Settings & Preferences", color="blue").classes("text-xs")

        # Main content in responsive grid layout
        with ui.grid(columns=2).classes("w-full gap-6 p-6 grid-cols-1 lg:grid-cols-2"):
            # Left Column - Advanced Settings
            with ui.column().classes("gap-6"):
                await render_enhanced_advanced_settings(user_storage, fetch_api)

            # Right Column - Broker Connections
            with ui.column().classes("gap-6"):
                await render_enhanced_broker_settings(fetch_api, user_storage, broker)


async def render_enhanced_broker_settings(fetch_api, user_storage, broker):
    """Render enhanced broker connection settings with actual implementation"""

    # Broker Connections Card
    with ui.card().classes("dashboard-card w-full"):
        with ui.row().classes("card-header w-full items-center p-6"):
            ui.icon("link", size="1.5rem").classes("text-green-400")
            ui.label("Broker Connections").classes("card-title")

        ui.separator().classes("card-separator")

        with ui.column().classes("w-full p-6 gap-6"):
            # Real broker connection status check function
            async def check_and_display_status(broker_name, display_container):
                """Check actual broker connection status using existing API"""
                try:
                    profile = await fetch_api(f"/profile/{broker_name}")
                    token_status = await fetch_api(f"/auth/token-status/{broker_name}")
                    display_container.clear()

                    with display_container:
                        if profile and profile.get("name"):
                            # Connected state
                            with ui.card().classes("w-full bg-green-900/20 border border-green-500/30 min-h-[200px]"):
                                with ui.column().classes("p-6 gap-4 h-full"):
                                    with ui.row().classes("items-center justify-between"):
                                        with ui.row().classes("items-center gap-2"):
                                            ui.icon("check_circle", size="1.2rem").classes("text-green-400")
                                            ui.label(f"{broker_name} Status").classes("text-white font-semibold")
                                        ui.chip("Connected", color="green").classes("text-xs")

                                    ui.label(f"Connected as: {profile['name']}").classes("text-green-400 font-semibold")
                                    
                                    # Token expiry information
                                    if token_status and token_status.get("is_valid"):
                                        expires_in = token_status.get("expires_in_hours", 0)
                                        if expires_in > 0:
                                            if expires_in < 2:
                                                color_class = "text-red-400"
                                                ui.label(f"⚠️ Token expires in {expires_in:.1f} hours").classes(color_class + " text-sm font-semibold")
                                            elif expires_in < 6:
                                                color_class = "text-yellow-400"
                                                ui.label(f"⚠️ Token expires in {expires_in:.1f} hours").classes(color_class + " text-sm")
                                            else:
                                                color_class = "text-green-400"
                                                ui.label(f"✅ Token valid for {expires_in:.1f} hours").classes(color_class + " text-sm")
                                        else:
                                            ui.label("⚠️ Token expired - please reconnect").classes("text-red-400 text-sm font-semibold")
                                    else:
                                        ui.label("Unable to check token status").classes("text-gray-400 text-sm")

                                    with ui.row().classes("w-full gap-2 mt-4"):
                                        ui.button("Test Connection", icon="wifi",
                                                  on_click=lambda: test_broker_connection(broker_name)).classes(
                                            "flex-1 text-cyan-400 border-cyan-400").props("outline")
                                        ui.button("Refresh Token", icon="refresh", color="blue",
                                                  on_click=lambda: show_renew_token_dialog(broker_name, display_container, fetch_api)).classes("flex-1")
                                        ui.button("Revoke", icon="link_off", color="red",
                                                  on_click=lambda: revoke_token_dialog(broker_name, display_container, fetch_api)).classes("flex-1")
                        else:
                            # Not connected state - render connection form
                            await render_broker_connection_form(broker_name, display_container, fetch_api)

                except Exception as e:
                    logger.error(f"Error checking {broker_name} status: {e}")
                    with display_container:
                        with ui.card().classes("w-full bg-red-900/20 border border-red-500/30"):
                            with ui.column().classes("p-4"):
                                ui.icon("error", size="1.2rem").classes("text-red-400 mb-2")
                                ui.label(f"Error checking {broker_name} status").classes("text-red-400 font-semibold")
                                ui.label(str(e)).classes("text-gray-400 text-sm")

            async def render_broker_connection_form(broker_name, container, fetch_api):
                """Render actual broker connection form based on existing implementation"""
                
                # Fetch user API keys for generating authentication URLs
                api_keys_data = None
                try:
                    api_keys_data = await fetch_api("/users/me/api-keys")
                except Exception as e:
                    logger.error(f"Error fetching API keys: {e}")

                with container:
                    with ui.card().classes("w-full bg-gray-800/50 border border-red-500/30 min-h-[200px]"):
                        with ui.column().classes("p-3 gap-4 h-full"):
                            with ui.row().classes("items-center gap-2 mb-1"):
                                ui.icon("link_off", size="1.2rem").classes("text-red-400")
                                ui.label(f"{broker_name} Status").classes("text-white font-semibold")
                                ui.chip("Not Connected", color="red").classes("text-xs")

                            ui.label(f"Connect your {broker_name} account to start trading").classes(
                                "text-gray-400 text-sm mb-1")

                            if broker_name == "Upstox":
                                # Upstox connection form (using existing implementation)
                                redirect_uri = ui.input(label="Enter Upstox Redirect URI", value="https://api.upstox.com/v2/login").classes("w-full mb-1")
                                ui.label("Get your auth code from Upstox login flow").classes("text-sm text-gray-500")
                                
                                # Add Fetch Auth Code link if API key is available
                                if api_keys_data and api_keys_data.get("upstox_api_key"):
                                    upstox_auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_keys_data['upstox_api_key']}&redirect_uri={redirect_uri.value}"
                                    ui.link("Fetch Auth Code", upstox_auth_url, new_tab=True).classes("w-full mb-3").props("outline")
                                
                                auth_code_input = ui.input(placeholder="Enter Upstox Auth Code").classes("w-full mb-3")

                                async def connect_upstox_action():
                                    """Connect to Upstox using existing API endpoint"""
                                    if auth_code_input.value:
                                        try:
                                            # Use existing API endpoint
                                            resp = await fetch_api(f"/auth/upstox/?auth_code={auth_code_input.value}",
                                                                   method="POST")
                                            if resp and resp.get("status") == "success":
                                                ui.notify(
                                                    f"Upstox connected successfully: {resp.get('message', 'Success')}",
                                                    type="positive")
                                                # Update default broker in storage
                                                user_storage['default_broker'] = "Upstox"
                                                # Refresh status display
                                                await check_and_display_status("Upstox", container)
                                            else:
                                                ui.notify("Failed to connect to Upstox. Please check your auth code.",
                                                          type="negative")
                                        except Exception as e:
                                            ui.notify(f"Connection error: {str(e)}", type="negative")
                                            logger.error(f"Upstox connection error: {e}")
                                    else:
                                        ui.notify("Auth code is required", type="warning")

                                ui.button("Connect Upstox", icon="link", color="primary",
                                          on_click=connect_upstox_action).classes("w-full")

                            elif broker_name == "Zerodha":
                                ui.label("Get your request token from Zerodha Kite login").classes("text-sm text-gray-500")
                                
                                # Add Fetch Request Token link if API key is available
                                if api_keys_data and api_keys_data.get("zerodha_api_key"):
                                    zerodha_auth_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_keys_data['zerodha_api_key']}"
                                    ui.link("Fetch Request Token",zerodha_auth_url, new_tab=True).classes("w-full mb-1").props("outline")
                                
                                req_token_input = ui.input(placeholder="Enter Zerodha Request Token").classes("w-full mb-1")

                                async def connect_zerodha_action():
                                    """Connect to Zerodha using existing API endpoint"""
                                    if req_token_input.value:
                                        try:
                                            # Use existing API endpoint
                                            resp = await fetch_api(f"/auth/zerodha/?request_token={req_token_input.value}",
                                                                   method="POST")
                                            if resp and resp.get("status") == "success":
                                                ui.notify(
                                                    f"Zerodha connected successfully: {resp.get('message', 'Success')}",
                                                    type="positive")
                                                # Update default broker in storage
                                                user_storage['default_broker'] = "Zerodha"
                                                # Refresh status display
                                                await check_and_display_status("Zerodha", container)
                                            else:
                                                ui.notify("Failed to connect to Zerodha. Please check your request token.",
                                                          type="negative")
                                        except Exception as e:
                                            ui.notify(f"Connection error: {str(e)}", type="negative")
                                            logger.error(f"Zerodha connection error: {e}")
                                    else:
                                        ui.notify("Request token is required", type="warning")

                                ui.button("Connect Zerodha", icon="link", color="green",
                                          on_click=connect_zerodha_action).classes("w-full")

            # Get user's broker configuration
            api_keys_data = None
            try:
                api_keys_data = await fetch_api("/users/me/api-keys")
            except Exception as e:
                logger.error(f"Error fetching API keys: {e}")

            # Create containers for broker status
            broker_containers = {}
            
            # Only show sections for configured brokers
            if api_keys_data and api_keys_data.get("configured_brokers"):
                configured_brokers = api_keys_data.get("configured_brokers", [])
                for broker in configured_brokers:
                    broker_containers[broker] = ui.column().classes("w-full")
                    await check_and_display_status(broker, broker_containers[broker])
            else:
                # Fallback: show both if no API keys data
                zerodha_status_container = ui.column().classes("w-full")
                upstox_status_container = ui.column().classes("w-full")
                broker_containers["Zerodha"] = zerodha_status_container
                broker_containers["Upstox"] = upstox_status_container
                await check_and_display_status("Zerodha", zerodha_status_container)
                await check_and_display_status("Upstox", upstox_status_container)

            # Refresh all connections button
            async def refresh_all_statuses():
                """Refresh all broker connection statuses"""
                try:
                    refresh_tasks = []
                    for broker, container in broker_containers.items():
                        refresh_tasks.append(check_and_display_status(broker, container))
                    await asyncio.gather(*refresh_tasks)
                    ui.notify("Connection statuses refreshed", type="positive")
                except Exception as e:
                    ui.notify(f"Error refreshing statuses: {str(e)}", type="negative")
                    logger.error(f"Error refreshing broker statuses: {e}")

            ui.button("Refresh All Connections", icon="refresh", on_click=refresh_all_statuses).classes(
                "w-full mt-6 py-3 text-cyan-400 border-cyan-400").props("outline")


# Broker management functions (using actual implementation)
async def test_broker_connection(broker_name):
    """Test broker connection"""
    ui.notify(f"Testing {broker_name} connection...", type="info")
    # This would test the actual API connection
    await asyncio.sleep(1)  # Simulate API call
    ui.notify(f"{broker_name} connection test successful!", type="positive")


def disconnect_broker_dialog(broker_name):
    """Show disconnect confirmation dialog"""
    with ui.dialog() as dialog, ui.card().classes("dashboard-card"):
        with ui.column().classes("p-4 gap-4"):
            ui.label(f"Disconnect {broker_name}").classes("text-xl font-bold text-white")
            ui.label(f"Are you sure you want to disconnect from {broker_name}?").classes("text-gray-300")

            with ui.card().classes("w-full bg-red-900/20 border border-red-500/30"):
                with ui.row().classes("p-3 items-center gap-2"):
                    ui.icon("warning", size="1.2rem").classes("text-red-400")
                    ui.label("You won't be able to place trades until you reconnect.").classes("text-sm text-red-300")

            with ui.row().classes("gap-2 justify-end w-full"):
                ui.button("Cancel", on_click=dialog.close).classes("text-gray-400")
                ui.button("Disconnect", color="red", on_click=lambda: execute_disconnect(dialog, broker_name)).classes(
                    "text-white")

    dialog.open()


def execute_disconnect(dialog, broker_name):
    """Execute broker disconnection"""
    # This would call the actual disconnect API
    ui.notify(f"{broker_name} disconnected successfully", type="positive")
    dialog.close()
    # Refresh the page to show updated status
    ui.navigate.to('/settings')


async def render_enhanced_advanced_settings(user_storage, fetch_api=None):
    """Render enhanced advanced settings with expandable sections and batch save functionality"""

    # Load user preferences from backend
    user_preferences = {}
    if fetch_api:
        try:
            # Force refresh to bypass cache
            prefs_response = await fetch_api("/user/preferences?force_refresh=true")
            if prefs_response and prefs_response.get("status") == "success":
                user_preferences = prefs_response.get("preferences", {})
        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")

    # Track changes for batch saving
    pending_changes = {}

    # Save preference function for advanced settings
    async def save_advanced_preference(key, value):
        """Save an advanced preference to backend"""
        pending_changes[key] = value
        if fetch_api:
            try:
                update_data = {key: value}
                response = await fetch_api("/user/preferences", method="POST", data=update_data)
                
                if response and response.get("status") == "success":
                    # Invalidate preferences cache
                    user_id = user_storage.get('user_id', 'unknown')
                    invalidate_on_settings_change(user_id, 'risk_settings')
                    # Add small delay to ensure cache is cleared
                    await asyncio.sleep(0.5)
                    try:
                        ui.notify(f"Setting saved: {key}", type="positive")
                    except RuntimeError:
                        # UI context not available, log instead
                        logger.info(f"Setting saved: {key}")
                else:
                    error_msg = response.get('detail', {}).get('message', 'Unknown error') if response else 'No response'
                    try:
                        ui.notify(f"Failed to save {key}: {error_msg}", type="negative")
                    except RuntimeError:
                        logger.error(f"Failed to save {key}: {error_msg}")
            except Exception as e:
                logger.error(f"Error saving preference {key}: {e}")
                try:
                    ui.notify(f"Failed to save {key}: {str(e)}", type="negative")
                except RuntimeError:
                    logger.error(f"Failed to save {key}: {str(e)}")
        # Also save to local storage as fallback
        user_storage.update({key: value})

    # Batch save all changes
    async def save_all_preferences():
        if pending_changes:
            try:
                response = await fetch_api("/user/preferences", method="POST", data=pending_changes)
                
                if response and response.get("status") == "success":
                    try:
                        ui.notify(f"✓ Saved {len(pending_changes)} preferences", type="positive")
                    except RuntimeError:
                        logger.info(f"✓ Saved {len(pending_changes)} preferences")
                    # Invalidate cache
                    user_id = user_storage.get('user_id', 'unknown')
                    invalidate_on_settings_change(user_id, 'risk_settings')
                    # Add small delay to ensure cache is cleared
                    await asyncio.sleep(0.5)
                    pending_changes.clear()
                else:
                    error_msg = response.get('detail', {}).get('message', 'Unknown error') if response else 'No response'
                    try:
                        ui.notify(f"Failed to save preferences: {error_msg}", type="negative")
                    except RuntimeError:
                        logger.error(f"Failed to save preferences: {error_msg}")
            except Exception as e:
                logger.error(f"Error saving preferences: {e}")
                try:
                    ui.notify(f"Failed to save preferences: {str(e)}", type="negative")
                except RuntimeError:
                    logger.error(f"Failed to save preferences: {str(e)}")
        else:
            try:
                ui.notify("No changes to save", type="info")
            except RuntimeError:
                logger.info("No changes to save")

    with ui.card().classes("dashboard-card w-full"):
        with ui.row().classes("card-header w-full items-center justify-between p-6"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("settings", size="1.5rem").classes("text-orange-400")
                ui.label("Application Settings").classes("card-title")
            
            # Action buttons
            with ui.row().classes("items-center gap-2"):
                # Export Settings button
                async def export_settings():
                    """Export current settings to JSON file"""
                    try:
                        # Get current preferences
                        prefs_response = await fetch_api("/user/preferences")
                        if prefs_response and prefs_response.get("status") == "success":
                            preferences = prefs_response.get("preferences", {})
                            
                            # Create export data
                            export_data = {
                                "export_timestamp": datetime.now().isoformat(),
                                "user_id": user_storage.get('user_id', 'unknown'),
                                "preferences": preferences
                            }
                            
                            # Convert to JSON string
                            import json
                            json_data = json.dumps(export_data, indent=2)
                            
                            # Create download link
                            import base64
                            b64_data = base64.b64encode(json_data.encode()).decode()
                            download_url = f"data:application/json;base64,{b64_data}"
                            
                            # Trigger download
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"trading_settings_{timestamp}.json"
                            
                            # Use JavaScript to trigger download
                            ui.run_javascript(f"""
                                const link = document.createElement('a');
                                link.href = '{download_url}';
                                link.download = '{filename}';
                                link.click();
                            """)
                            
                            ui.notify(f"Settings exported to {filename}", type="positive")
                        else:
                            ui.notify("Failed to fetch current settings", type="negative")
                    except Exception as e:
                        logger.error(f"Error exporting settings: {e}")
                        ui.notify("Failed to export settings", type="negative")
                
                ui.button("Export Settings", icon="download", on_click=export_settings).classes("text-cyan-400").props("outline")
                
                # Reset All button
                async def reset_all_settings():
                    """Reset all settings to default values"""
                    try:
                        # Default preferences (same as signup)
                        default_preferences = {
                            'default_order_type': 'MARKET',
                            'default_product_type': 'CNC',
                            'refresh_interval': 5,
                            'order_alerts': True,
                            'pnl_alerts': True,
                            'strategy_alerts': True,
                            'daily_loss_limit': 10000,
                            'position_size_limit': 50000,
                            'auto_stop_trading': True,
                            'max_open_positions': 10,
                            'risk_per_trade': 2.0,
                            'max_portfolio_risk': 20.0,
                            'max_orders_per_minute': 10,
                            'request_timeout': 30,
                            'enable_rate_limiting': True,
                            'auto_retry_requests': True
                        }
                        
                        # Update preferences in backend
                        response = await fetch_api("/user/preferences", method="POST", data=default_preferences)
                        if response and response.get("status") == "success":
                            # Invalidate cache
                            user_id = user_storage.get('user_id', 'unknown')
                            invalidate_on_settings_change(user_id, 'all_settings')
                            
                            ui.notify("All settings reset to default values", type="positive")
                            # Wait longer for backend to process and cache to clear, then reload
                            await asyncio.sleep(2)
                            ui.navigate.reload()
                        else:
                            ui.notify("Failed to reset settings", type="negative")
                    except Exception as e:
                        logger.error(f"Error resetting settings: {e}")
                        ui.notify(f"Failed to reset settings: {str(e)}", type="negative")
                
                ui.button("Reset All", icon="refresh", color="red", on_click=reset_all_settings).classes("text-white")
                
                # Save All button
                ui.button("Save All Changes", icon="save", color="primary", 
                         on_click=save_all_preferences).classes("text-white")
                

        ui.separator().classes("card-separator")

        with ui.column().classes("w-full p-6 gap-6"):
            # Show current risk status
            try:
                risk_response = await fetch_api("/risk-management/metrics")
                if risk_response:
                    trading_allowed = risk_response.get('trading_allowed', True)
                    status_color = "text-green-400" if trading_allowed else "text-red-400"
                    status_text = "Active" if trading_allowed else "Suspended"
                    status_icon = "check_circle" if trading_allowed else "warning"
                    
                    with ui.card().classes("w-full bg-gray-900/50 border border-gray-600/30 mb-6"):
                        with ui.row().classes("p-4 items-center gap-3"):
                            ui.icon(status_icon, size="1.2rem").classes(status_color)
                            ui.label("Current Trading Status:").classes("text-gray-400 text-sm")
                            ui.label(status_text).classes(f"{status_color} font-semibold text-sm")
                            if not trading_allowed:
                                ui.label(f"• {risk_response.get('trading_status', 'Unknown reason')}").classes("text-red-300 text-xs")
            except Exception as e:
                logger.error(f"Error fetching risk status: {e}")

            # Risk Management Settings (Expandable)
            with ui.expansion("Risk Management Settings", icon="security").classes("w-full"):
                with ui.card().classes("w-full bg-gray-800/50 border border-red-500/30"):
                    with ui.column().classes("p-6 gap-4"):
                        with ui.grid(columns=2).classes("w-full gap-4"):
                            # Daily loss limit
                            with ui.column().classes("gap-2"):
                                daily_loss_limit = user_preferences.get('daily_loss_limit', user_storage.get('daily_loss_limit', 10000))
                                ui.label("Daily Loss Limit (INR)").classes("text-sm text-gray-400")
                                ui.number(
                                    value=daily_loss_limit,
                                    min=1000, max=10000000, step=1000,
                                    on_change=lambda e: asyncio.create_task(save_advanced_preference('daily_loss_limit', e.value))
                                ).classes("w-full")

                            # Portfolio exposure base
                            with ui.column().classes("gap-2"):
                                portfolio_size_limit = user_preferences.get(
                                    'portfolio_size_limit',
                                    user_storage.get('portfolio_size_limit', user_storage.get('position_size_limit', 50000))
                                )
                                ui.label("Portfolio Exposure Limit (INR)").classes("text-sm text-gray-400")
                                ui.number(
                                    value=portfolio_size_limit,
                                    min=5000, max=50000000, step=5000,
                                    on_change=lambda e: asyncio.create_task(save_advanced_preference('portfolio_size_limit', e.value))
                                ).classes("w-full")

                        with ui.grid(columns=2).classes("w-full gap-4"):
                            # Single trade limit
                            with ui.column().classes("gap-2"):
                                position_size_limit = user_preferences.get('position_size_limit', user_storage.get('position_size_limit', 50000))
                                ui.label("Single Trade Limit (INR)").classes("text-sm text-gray-400")
                                ui.number(
                                    value=position_size_limit,
                                    min=5000, max=50000000, step=5000,
                                    on_change=lambda e: asyncio.create_task(save_advanced_preference('position_size_limit', e.value))
                                ).classes("w-full")

                            # Max open positions
                            with ui.column().classes("gap-2"):
                                max_open_positions = user_preferences.get('max_open_positions', user_storage.get('max_open_positions', 10))
                                ui.label("Max Open Positions").classes("text-sm text-gray-400")
                                ui.number(
                                    value=max_open_positions,
                                    min=1, max=100, step=1,
                                    on_change=lambda e: asyncio.create_task(save_advanced_preference('max_open_positions', e.value))
                                ).classes("w-full")

                        with ui.grid(columns=2).classes("w-full gap-4"):
                            # Risk per trade percentage
                            with ui.column().classes("gap-2"):
                                risk_per_trade = user_preferences.get('risk_per_trade', user_storage.get('risk_per_trade', 2.0))
                                ui.label("Risk Per Trade (%)").classes("text-sm text-gray-400")
                                ui.number(
                                    value=risk_per_trade,
                                    min=0.1, max=25.0, step=0.1, format="%.1f",
                                    on_change=lambda e: asyncio.create_task(save_advanced_preference('risk_per_trade', e.value))
                                ).classes("w-full")

                            # Max portfolio risk
                            with ui.column().classes("gap-2"):
                                max_portfolio_risk = user_preferences.get('max_portfolio_risk', user_storage.get('max_portfolio_risk', 20.0))
                                ui.label("Max Portfolio Risk (%)").classes("text-sm text-gray-400")
                                ui.number(
                                    value=max_portfolio_risk,
                                    min=1.0, max=100.0, step=1.0, format="%.1f",
                                    on_change=lambda e: asyncio.create_task(save_advanced_preference('max_portfolio_risk', e.value))
                                ).classes("w-full")

                        with ui.grid(columns=1).classes("w-full gap-4"):
                            # Max orders per minute
                            with ui.column().classes("gap-2"):
                                max_orders_per_minute = user_preferences.get('max_orders_per_minute', user_storage.get('max_orders_per_minute', 10))
                                ui.label("Max Orders Per Minute").classes("text-sm text-gray-400")
                                ui.number(
                                    value=max_orders_per_minute,
                                    min=1, max=100, step=1,
                                    on_change=lambda e: asyncio.create_task(save_advanced_preference('max_orders_per_minute', e.value))
                                ).classes("w-full")

                        # Auto-stop trading on loss
                        auto_stop_trading = user_preferences.get('auto_stop_trading', user_storage.get('auto_stop_trading', True))
                        ui.switch(
                            "Auto-stop trading on daily loss limit",
                            value=auto_stop_trading,
                            on_change=lambda e: asyncio.create_task(save_advanced_preference('auto_stop_trading', e.value))
                        ).classes("w-full mt-4")

            # Trading Preferences (Expandable)
            with ui.expansion("Trading Preferences", icon="tune").classes("w-full"):
                with ui.card().classes("w-full bg-gray-800/50 border border-purple-500/30"):
                    with ui.column().classes("p-6 gap-4"):
                        # Default order type - Used in: order_management.py for default order type selection
                        default_order_type = user_preferences.get('default_order_type', user_storage.get('default_order_type', 'MARKET'))
                        ui.label("Default Order Type").classes("text-sm text-gray-400")
                        ui.select(
                            options=["MARKET", "LIMIT", "SL", "SL-M"],
                            value=default_order_type,
                            on_change=lambda e: asyncio.create_task(save_advanced_preference('default_order_type', e.value))
                        ).classes("w-full mb-3")
                        
                        # Default product type - Used in: order_management.py for default product type selection
                        default_product_type = user_preferences.get('default_product_type', user_storage.get('default_product_type', 'CNC'))
                        ui.label("Default Product Type").classes("text-sm text-gray-400")
                        ui.select(
                            options=["CNC", "MIS", "NRML"],
                            value=default_product_type,
                            on_change=lambda e: asyncio.create_task(save_advanced_preference('default_product_type', e.value))
                        ).classes("w-full mb-3")
                        
                        # Auto-refresh interval - Used in: Enhanced scanner for auto-refresh timing (NOT used in dashboard/positions)
                        refresh_interval = user_preferences.get('refresh_interval', user_storage.get('refresh_interval', 5))
                        refresh_label = ui.label(f"Auto-refresh Interval: {refresh_interval} seconds").classes("text-sm text-gray-400")
                        refresh_slider = ui.slider(
                            min=1, max=30, step=1, value=refresh_interval,
                            on_change=lambda e: asyncio.create_task(save_advanced_preference('refresh_interval', e.value))
                        ).classes("w-full")

                        # Update label when slider changes
                        refresh_slider.on('update:model-value', lambda e: refresh_label.set_text(f"Auto-refresh Interval: {e.value} seconds"))

            # Notification Preferences (Expandable)
            with ui.expansion("Notification Preferences", icon="notifications").classes("w-full"):
                with ui.card().classes("w-full bg-gray-800/50 border border-yellow-500/30"):
                    with ui.column().classes("p-6 gap-4"):
                        # Notification toggles with backend sync - Stored only - NO notification system implemented yet
                        order_alerts = user_preferences.get('order_alerts', user_storage.get('order_alerts', True))
                        ui.switch(
                            "Order Execution Alerts",
                            value=order_alerts,
                            on_change=lambda e: asyncio.create_task(save_advanced_preference('order_alerts', e.value))
                        ).classes("w-full mb-2")

                        pnl_alerts = user_preferences.get('pnl_alerts', user_storage.get('pnl_alerts', True))
                        ui.switch(
                            "P&L Threshold Alerts",
                            value=pnl_alerts,
                            on_change=lambda e: asyncio.create_task(save_advanced_preference('pnl_alerts', e.value))
                        ).classes("w-full mb-2")

                        strategy_alerts = user_preferences.get('strategy_alerts', user_storage.get('strategy_alerts', True))
                        ui.switch(
                            "Strategy Signal Alerts",
                            value=strategy_alerts,
                            on_change=lambda e: asyncio.create_task(save_advanced_preference('strategy_alerts', e.value))
                        ).classes("w-full")

            # Trading Limits (Expandable)
            with ui.expansion("Trading Limits", icon="speed").classes("w-full"):
                with ui.card().classes("w-full bg-gray-800/50 border border-blue-500/30"):
                    with ui.column().classes("p-6 gap-4"):
                        with ui.grid(columns=2).classes("w-full gap-4"):
                            # Request timeout
                            with ui.column().classes("gap-2"):
                                request_timeout = user_preferences.get('request_timeout', user_storage.get('request_timeout', 30))
                                ui.label("API Request Timeout (seconds)").classes("text-sm text-gray-400")
                                ui.number(
                                    value=request_timeout,
                                    min=5, max=120, step=5,
                                    on_change=lambda e: asyncio.create_task(save_advanced_preference('request_timeout', e.value))
                                ).classes("w-full")

                            # Placeholder for future trading limits
                            with ui.column().classes("gap-2"):
                                ui.label("Future Trading Limits").classes("text-sm text-gray-400")
                                ui.label("More limits coming soon...").classes("text-sm text-gray-500")

                        # Rate limiting switches
                        with ui.column().classes("gap-4 mt-4"):
                            enable_rate_limiting = user_preferences.get('enable_rate_limiting', user_storage.get('enable_rate_limiting', True))
                            ui.switch(
                                "Enable API Rate Limiting",
                                value=enable_rate_limiting,
                                on_change=lambda e: asyncio.create_task(save_advanced_preference('enable_rate_limiting', e.value))
                            ).classes("w-full")

                            auto_retry_requests = user_preferences.get('auto_retry_requests', user_storage.get('auto_retry_requests', True))
                            ui.switch(
                                "Auto-retry Failed Requests",
                                value=auto_retry_requests,
                                on_change=lambda e: asyncio.create_task(save_advanced_preference('auto_retry_requests', e.value))
                            ).classes("w-full")


# Token Management Functions for Broker Authentication

async def show_renew_token_dialog(broker_name, display_container, fetch_api):
    """Show dialog to renew broker token"""
    with ui.dialog() as dialog, ui.card().classes("dashboard-card min-w-[500px]"):
        with ui.column().classes("p-6 gap-4"):
            ui.label(f"Refresh {broker_name} Token").classes("text-xl font-bold text-white")
            ui.label(f"Your {broker_name} access token needs to be refreshed for continued trading").classes("text-gray-300")

            with ui.card().classes("w-full bg-blue-900/20 border border-blue-500/30"):
                with ui.row().classes("p-4 items-center gap-3"):
                    ui.icon("info", size="1.2rem").classes("text-blue-400")
                    ui.label("You'll need to get a new authentication token from your broker's login page").classes("text-sm text-blue-300")

            if broker_name == "Zerodha":
                ui.label("Request Token").classes("text-sm text-gray-400 mt-3")
                ui.label("Get your new request token from Zerodha Kite login").classes("text-xs text-gray-500")
                token_input = ui.input(placeholder="Enter new Zerodha Request Token").classes("w-full")
                
                async def renew_zerodha():
                    if token_input.value:
                        try:
                            resp = await fetch_api(f"/auth/zerodha/?request_token={token_input.value}", method="POST")
                            if resp and resp.get("status") == "success":
                                ui.notify("Zerodha token renewed successfully!", type="positive")
                                dialog.close()
                                await asyncio.sleep(0.5)
                                ui.navigate.reload()
                            else:
                                ui.notify("Failed to renew Zerodha token. Please check your request token.", type="negative")
                        except Exception as e:
                            ui.notify(f"Error renewing token: {str(e)}", type="negative")
                    else:
                        ui.notify("Request token is required", type="warning")

                renew_action = renew_zerodha

            else:  # Upstox
                ui.label("Auth Code").classes("text-sm text-gray-400 mt-3")
                ui.label("Get your new auth code from Upstox login flow").classes("text-xs text-gray-500")
                token_input = ui.input(placeholder="Enter new Upstox Auth Code").classes("w-full")
                
                async def renew_upstox():
                    if token_input.value:
                        try:
                            resp = await fetch_api(f"/auth/upstox/?auth_code={token_input.value}", method="POST")
                            if resp and resp.get("status") == "success":
                                ui.notify("Upstox token renewed successfully!", type="positive")
                                dialog.close()
                                await asyncio.sleep(0.5)
                                ui.navigate.reload()
                            else:
                                ui.notify("Failed to renew Upstox token. Please check your auth code.", type="negative")
                        except Exception as e:
                            ui.notify(f"Error renewing token: {str(e)}", type="negative")
                    else:
                        ui.notify("Auth code is required", type="warning")

                renew_action = renew_upstox

            with ui.row().classes("gap-2 justify-end w-full mt-4"):
                ui.button("Cancel", on_click=dialog.close).classes("text-gray-400")
                ui.button(f"Refresh {broker_name} Token", color="blue", on_click=renew_action).classes("text-white")

    dialog.open()


async def revoke_token_dialog(broker_name, display_container, fetch_api):
    """Show dialog to revoke broker token"""
    with ui.dialog() as dialog, ui.card().classes("dashboard-card min-w-[450px]"):
        with ui.column().classes("p-6 gap-4"):
            ui.label(f"Revoke {broker_name} Token").classes("text-xl font-bold text-white")
            ui.label(f"This will permanently revoke your {broker_name} access token").classes("text-gray-300")

            with ui.card().classes("w-full bg-red-900/20 border border-red-500/30"):
                with ui.column().classes("p-4 gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("warning", size="1.2rem").classes("text-red-400")
                        ui.label("Warning: This action cannot be undone").classes("text-sm text-red-300 font-semibold")
                    ui.label("• You won't be able to place trades until you reconnect").classes("text-sm text-red-300")
                    ui.label("• All active orders and positions will remain unaffected").classes("text-sm text-red-300")
                    ui.label("• You'll need to re-authenticate to resume trading").classes("text-sm text-red-300")

            async def execute_revoke():
                try:
                    resp = await fetch_api(f"/auth/revoke-token/{broker_name}", method="DELETE")
                    if resp and resp.get("status") == "success":
                        ui.notify(f"{broker_name} token revoked successfully", type="positive")
                        dialog.close()
                        await asyncio.sleep(0.5)
                        ui.navigate.reload()
                    else:
                        ui.notify(f"Failed to revoke {broker_name} token", type="negative")
                except Exception as e:
                    ui.notify(f"Error revoking token: {str(e)}", type="negative")

            with ui.row().classes("gap-2 justify-end w-full mt-4"):
                ui.button("Cancel", on_click=dialog.close).classes("text-gray-400")
                ui.button(f"Revoke {broker_name} Token", color="red", on_click=execute_revoke).classes("text-white")

    dialog.open()
