# Enhanced Strategies Module - strategies.py
# Applying beautiful dashboard.py styling consistently

from nicegui import ui
import logging
import pandas as pd
from datetime import datetime
import json

logger = logging.getLogger(__name__)


async def render_strategies_page(fetch_api, user_storage, get_cached_instruments):
    """Enhanced strategies page with beautiful dashboard styling"""
    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):
        
        # Enhanced title section (matching dashboard.py)
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("psychology", size="2rem").classes("text-cyan-400")
                    ui.label("Trading Strategies").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("AUTOMATED", color="green").classes("text-xs status-chip")
                
                ui.label("Create, manage and monitor your algorithmic trading strategies").classes("text-gray-400 dashboard-subtitle")
            
            # Right side - Action buttons
            with ui.row().classes("items-center gap-4"):
                ui.button("New Strategy", icon="add", color="primary").classes("text-white")
                ui.button("Import Strategy", icon="upload").classes("text-cyan-400")
                ui.button("Export All", icon="download").classes("text-gray-400")
        
        # Strategy summary cards
        await render_enhanced_strategy_summary(fetch_api, user_storage)
        
        # Main content in grid layout
        with ui.row().classes("w-full gap-4 p-4"):
            
            # Strategy templates (left panel)
            with ui.card().classes("dashboard-card w-1/3"):
                with ui.row().classes("card-header w-full items-center p-4"):
                    ui.icon("library_books", size="1.5rem").classes("text-purple-400")
                    ui.label("Strategy Templates").classes("card-title")
                
                ui.separator().classes("card-separator")
                
                await render_strategy_templates()
            
            # Active strategies (right panel)
            with ui.card().classes("dashboard-card flex-1"):
                with ui.row().classes("card-header w-full justify-between items-center p-4"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("auto_awesome", size="1.5rem").classes("text-green-400")
                        ui.label("Active Strategies").classes("card-title")
                        
                        # Live update indicator
                        with ui.row().classes("items-center gap-1 ml-2"):
                            ui.element('div').classes("w-2 h-2 bg-green-400 rounded-full animate-pulse")
                            ui.label("Live").classes("text-xs text-green-400")
                    
                    with ui.row().classes("items-center gap-2"):
                        ui.button("Start All", icon="play_arrow").props("flat").classes("text-green-400")
                        ui.button("Stop All", icon="stop").props("flat").classes("text-red-400")
                        ui.button("Refresh", icon="refresh").props("flat").classes("text-gray-400")
                
                ui.separator().classes("card-separator")
                
                strategies_container = ui.column().classes("w-full p-4")
                await render_enhanced_strategies_list(fetch_api, user_storage, strategies_container)

async def render_enhanced_strategy_summary(fetch_api, user_storage):
    """Enhanced strategy summary metrics"""
    
    with ui.row().classes("w-full gap-4 p-4"):
        try:
            # Fetch strategies data - this would be from your API
            # strategies_data = await fetch_api("/strategies")
            
            # Sample data for demonstration
            total_strategies = 5
            active_strategies = 3
            profitable_strategies = 2
            total_pnl = 12750.50
            win_rate = 65.5
            
            # Total Strategies
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("psychology", size="2rem").classes("text-blue-400 mb-2")
                    ui.label("Total Strategies").classes("text-sm text-gray-400")
                    ui.label(str(total_strategies)).classes("text-2xl font-bold text-white")
            
            # Active Strategies
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("play_circle", size="2rem").classes("text-green-400 mb-2")
                    ui.label("Active").classes("text-sm text-gray-400")
                    ui.label(str(active_strategies)).classes("text-2xl font-bold text-green-400")
            
            # Profitable Strategies
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("trending_up", size="2rem").classes("text-purple-400 mb-2")
                    ui.label("Profitable").classes("text-sm text-gray-400")
                    ui.label(str(profitable_strategies)).classes("text-2xl font-bold text-purple-400")
            
            # Total P&L
            pnl_color = "text-green-400" if total_pnl >= 0 else "text-red-400"
            pnl_icon = "trending_up" if total_pnl >= 0 else "trending_down"
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon(pnl_icon, size="2rem").classes(f"{pnl_color} mb-2")
                    ui.label("Total P&L").classes("text-sm text-gray-400")
                    ui.label(f"₹{total_pnl:,.2f}").classes(f"text-2xl font-bold {pnl_color}")
            
            # Win Rate
            with ui.card().classes("dashboard-card metric-card flex-1"):
                with ui.column().classes("p-4 text-center"):
                    ui.icon("percent", size="2rem").classes("text-orange-400 mb-2")
                    ui.label("Win Rate").classes("text-sm text-gray-400")
                    ui.label(f"{win_rate:.1f}%").classes("text-2xl font-bold text-orange-400")
        
        except Exception as e:
            logger.error(f"Error fetching strategy summary: {e}")
            with ui.card().classes("dashboard-card w-full"):
                ui.label("Error loading strategy summary").classes("text-red-500 text-center p-4")

async def render_strategy_templates():
    """Render strategy templates section"""
    
    templates = [
        {
            "name": "RSI Mean Reversion",
            "description": "Buy oversold, sell overbought based on RSI levels",
            "category": "Technical",
            "icon": "show_chart",
            "color": "text-blue-400"
        },
        {
            "name": "Moving Average Crossover",
            "description": "Trade when fast MA crosses above/below slow MA",
            "category": "Trend Following",
            "icon": "trending_up",
            "color": "text-green-400"
        },
        {
            "name": "Bollinger Bands Squeeze",
            "description": "Trade breakouts from low volatility periods",
            "category": "Volatility",
            "icon": "compress",
            "color": "text-purple-400"
        },
        {
            "name": "MACD Momentum",
            "description": "Trade momentum shifts using MACD signals",
            "category": "Momentum",
            "icon": "speed",
            "color": "text-orange-400"
        },
        {
            "name": "Grid Trading",
            "description": "Place buy/sell orders at regular price intervals",
            "category": "Market Making",
            "icon": "grid_on",
            "color": "text-cyan-400"
        }
    ]
    
    with ui.column().classes("w-full p-4 gap-3"):
        for template in templates:
            with ui.card().classes("strategy-template-card w-full hover:bg-gray-800/30 transition-all cursor-pointer border border-gray-700/50"):
                with ui.column().classes("p-3 gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(template["icon"], size="1.2rem").classes(template["color"])
                        ui.label(template["name"]).classes("text-white font-semibold text-sm")
                    
                    ui.label(template["description"]).classes("text-gray-400 text-xs")
                    
                    with ui.row().classes("items-center justify-between mt-2"):
                        ui.chip(template["category"], color=None).classes("text-xs bg-gray-700/50 text-gray-300")
                        ui.button("Use", on_click=lambda t=template: create_strategy_from_template(t)).classes("text-cyan-400")

async def render_enhanced_strategies_list(fetch_api, user_storage, container):
    """Render enhanced strategies list"""
    
    try:
        # Fetch active strategies - this would be from your API
        # strategies_data = await fetch_api("/strategies/active")
        
        # Sample strategies data
        strategies_data = [
            {
                "id": "strat_001",
                "name": "RSI Mean Reversion",
                "status": "ACTIVE",
                "symbol": "RELIANCE",
                "pnl": 2450.75,
                "trades_today": 3,
                "win_rate": 67.5,
                "created": "2024-01-15",
                "last_signal": "2024-01-20 14:30:00"
            },
            {
                "id": "strat_002", 
                "name": "MA Crossover",
                "status": "PAUSED",
                "symbol": "TCS",
                "pnl": -125.50,
                "trades_today": 1,
                "win_rate": 55.2,
                "created": "2024-01-10",
                "last_signal": "2024-01-20 11:15:00"
            },
            {
                "id": "strat_003",
                "name": "Bollinger Bands",
                "status": "ACTIVE",
                "symbol": "HDFCBANK",
                "pnl": 1875.25,
                "trades_today": 2,
                "win_rate": 72.8,
                "created": "2024-01-12",
                "last_signal": "2024-01-20 15:45:00"
            }
        ]
        
        if not strategies_data:
            # Enhanced empty state
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("psychology", size="4rem").classes("text-gray-500 mb-4")
                    ui.label("No active strategies").classes("text-xl text-gray-400 mb-2")
                    ui.label("Create your first strategy to start automated trading").classes("text-sm text-gray-500")
                    ui.button("Create Strategy", icon="add", color="primary", on_click=create_new_strategy).classes("mt-4")
            return
        
        with container:
            for strategy in strategies_data:
                await render_enhanced_strategy_card(strategy, fetch_api)
    
    except Exception as e:
        logger.error(f"Error rendering strategies list: {e}")
        with container:
            with ui.column().classes("w-full text-center p-8"):
                ui.icon("error", size="3rem").classes("text-red-500 mb-4")
                ui.label("Error loading strategies").classes("text-xl text-red-400 mb-2")
                ui.label(str(e)).classes("text-sm text-gray-500")

async def render_enhanced_strategy_card(strategy, fetch_api):
    """Render individual enhanced strategy card"""
    
    try:
        strategy_id = strategy.get('id', 'N/A')
        name = strategy.get('name', 'N/A')
        status = strategy.get('status', 'UNKNOWN').upper()
        symbol = strategy.get('symbol', 'N/A')
        pnl = float(strategy.get('pnl', 0))
        trades_today = strategy.get('trades_today', 0)
        win_rate = strategy.get('win_rate', 0)
        
        # Determine status styling
        if status == 'ACTIVE':
            status_color = "text-green-400"
            status_bg = "bg-green-900/20"
            border_color = "border-green-500/30"
            status_icon = "play_circle"
        elif status == 'PAUSED':
            status_color = "text-yellow-400"
            status_bg = "bg-yellow-900/20"
            border_color = "border-yellow-500/30"
            status_icon = "pause_circle"
        else:
            status_color = "text-red-400"
            status_bg = "bg-red-900/20"
            border_color = "border-red-500/30"
            status_icon = "stop_circle"
        
        # Determine P&L styling
        pnl_color = "text-green-400" if pnl >= 0 else "text-red-400"
        pnl_icon = "trending_up" if pnl >= 0 else "trending_down"
        
        with ui.card().classes(f"strategy-card w-full mb-4 border {border_color} hover:bg-gray-800/30 transition-all"):
            with ui.column().classes("p-4"):
                # Header row
                with ui.row().classes("w-full justify-between items-center mb-3"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(status_icon, size="1.5rem").classes(status_color)
                        ui.label(name).classes("text-white font-bold text-lg")
                        ui.chip(status, color=None).classes(f"{status_color} {status_bg} text-xs")
                    
                    with ui.row().classes("items-center gap-2"):
                        # Action buttons
                        if status == 'ACTIVE':
                            ui.button(icon="pause", on_click=lambda sid=strategy_id: pause_strategy(sid)).props("flat round size=sm").classes("text-yellow-400")
                        else:
                            ui.button(icon="play_arrow", on_click=lambda sid=strategy_id: start_strategy(sid)).props("flat round size=sm").classes("text-green-400")
                        
                        ui.button(icon="stop", on_click=lambda sid=strategy_id: stop_strategy(sid)).props("flat round size=sm").classes("text-red-400")
                        ui.button(icon="edit", on_click=lambda sid=strategy_id: edit_strategy(sid)).props("flat round size=sm").classes("text-cyan-400")
                        ui.button(icon="delete", on_click=lambda sid=strategy_id: delete_strategy(sid)).props("flat round size=sm").classes("text-gray-400")
                
                # Metrics row
                with ui.row().classes("w-full gap-6"):
                    # Symbol
                    with ui.column().classes("gap-1"):
                        ui.label("Symbol").classes("text-xs text-gray-500")
                        ui.label(symbol).classes("text-white font-semibold")
                    
                    # P&L
                    with ui.column().classes("gap-1"):
                        ui.label("P&L").classes("text-xs text-gray-500")
                        with ui.row().classes("items-center gap-1"):
                            ui.icon(pnl_icon, size="1rem").classes(pnl_color)
                            ui.label(f"₹{pnl:,.2f}").classes(f"{pnl_color} font-semibold text-mono")
                    
                    # Trades Today
                    with ui.column().classes("gap-1"):
                        ui.label("Trades Today").classes("text-xs text-gray-500")
                        ui.label(str(trades_today)).classes("text-white font-semibold")
                    
                    # Win Rate
                    with ui.column().classes("gap-1"):
                        ui.label("Win Rate").classes("text-xs text-gray-500")
                        ui.label(f"{win_rate:.1f}%").classes("text-purple-400 font-semibold")
                
                # Progress bar for win rate
                with ui.row().classes("w-full mt-3"):
                    ui.linear_progress(win_rate / 100).classes("w-full").props(f"color={'positive' if win_rate > 60 else 'warning' if win_rate > 40 else 'negative'}")
    
    except Exception as e:
        logger.error(f"Error rendering strategy card: {e}")
        with ui.card().classes("strategy-card w-full mb-4 border border-red-500/30"):
            ui.label("Error loading strategy").classes("text-red-400 p-4")

def create_strategy_from_template(template):
    """Create strategy from template"""
    ui.notify(f"Creating strategy from template: {template['name']}", type="info")
    # This would open a strategy creation dialog
    show_strategy_creation_dialog(template)

def create_new_strategy():
    """Create new custom strategy"""
    ui.notify("Opening strategy builder...", type="info")
    show_strategy_creation_dialog()

def show_strategy_creation_dialog(template=None):
    """Show strategy creation dialog"""
    with ui.dialog() as dialog, ui.card().classes("dashboard-card min-w-96"):
        with ui.column().classes("p-6 gap-4"):
            if template:
                ui.label(f"Create Strategy: {template['name']}").classes("text-xl font-bold text-white")
                ui.label(template['description']).classes("text-gray-400 text-sm")
            else:
                ui.label("Create Custom Strategy").classes("text-xl font-bold text-white")
            
            # Strategy form
            strategy_name = ui.input("Strategy Name", placeholder="Enter strategy name").classes("w-full")
            
            symbol_select = ui.select(
                options=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"],
                label="Trading Symbol"
            ).classes("w-full")
            
            if template:
                # Pre-fill template-specific parameters
                if template['name'] == "RSI Mean Reversion":
                    rsi_period = ui.number("RSI Period", value=14, min=5, max=50).classes("w-full")
                    oversold_level = ui.number("Oversold Level", value=30, min=10, max=40).classes("w-full")
                    overbought_level = ui.number("Overbought Level", value=70, min=60, max=90).classes("w-full")
            
            quantity = ui.number("Quantity", value=1, min=1).classes("w-full")
            
            with ui.row().classes("gap-2 justify-end w-full mt-4"):
                ui.button("Cancel", on_click=dialog.close).classes("text-gray-400")
                ui.button("Create Strategy", color="primary", on_click=lambda: create_strategy(dialog, strategy_name.value, symbol_select.value, quantity.value)).classes("text-white")
    
    dialog.open()

def create_strategy(dialog, name, symbol, quantity):
    """Create the strategy"""
    if not name or not symbol:
        ui.notify("Please fill all required fields", type="warning")
        return
    
    ui.notify(f"Strategy '{name}' created successfully!", type="positive")
    dialog.close()
    ui.navigate.to('/strategies')

def start_strategy(strategy_id):
    """Start a strategy"""
    ui.notify(f"Starting strategy {strategy_id}", type="positive")
    # This would call your API to start the strategy

def pause_strategy(strategy_id):
    """Pause a strategy"""
    ui.notify(f"Pausing strategy {strategy_id}", type="warning")
    # This would call your API to pause the strategy

def stop_strategy(strategy_id):
    """Stop a strategy"""
    ui.notify(f"Stopping strategy {strategy_id}", type="info")
    # This would call your API to stop the strategy

def edit_strategy(strategy_id):
    """Edit a strategy"""
    ui.notify(f"Opening editor for strategy {strategy_id}", type="info")
    # This would open the strategy editor

def delete_strategy(strategy_id):
    """Delete a strategy"""
    ui.notify(f"Deleting strategy {strategy_id}", type="negative")
    # This would show confirmation and delete the strategy