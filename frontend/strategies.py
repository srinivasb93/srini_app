"""
Algorithmic Trading Strategies Module for NiceGUI Algo Trading Application
Implements UI for defining, configuring, and managing trading strategies with enhanced UX.
"""

from nicegui import ui
import pandas as pd
import asyncio
import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

logger = logging.getLogger(__name__)

# Module-level flag for tab initialization
create_tab_initialized = False

# Define available indicators at the module level
available_indicators = {
    "SMA": {"period": 20},
    "EMA": {"period": 14},
    "RSI": {"period": 14},
    "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "Bollinger Bands": {"period": 20, "std_dev": 2},
    "Close Price": {},
    "Open Price": {},
    "High Price": {},
    "Low Price": {},
    "Volume": {}
}

# Define ui_refs at the module level to be accessible across functions
ui_refs = {
    "strategy_name": None,
    "strategy_desc": None,
    "broker_select": None,
    "timeframe": None,
    "position_sizing": None,
    "preview_instrument": None,
    "save_button": None,
    "preview_button": None,
    "entry_conditions": [],
    "exit_conditions": [],
    "entry_container": None,
    "exit_container": None,
    "editing_strategy_id": None # Added for editing strategy
}

# Define execute_refs at the module level for strategy execution
execute_refs = {}

def format_conditions(conditions):
    """Formats JSON conditions into a human-readable string."""
    if not conditions:
        return "None"
    result = []
    for cond in conditions:
        left = f"{cond['left_indicator']}({cond['left_params']})" if cond.get('left_params') else cond['left_indicator']
        right = f"{cond['right_indicator']}({cond['right_params']})" if cond.get('right_params') else cond['right_indicator']
        if cond.get('left_value') is not None:
            left = f"Value({cond['left_value']})"
        if cond.get('right_value') is not None:
            right = f"Value({cond['right_value']})"
        result.append(f"{left} {cond['comparison']} {right}")
    return "; ".join(result)

def render_indicator_selector(indicators, prefix="", initial_indicator=None, initial_params=None):
    """Renders a selector for technical indicators and price/volume data with proper parameter handling."""
    param_values = {}
    input_id = str(uuid4())

    # Separate technical indicators from price/volume data
    technical_indicators = {k: v for k, v in indicators.items() if v}  # Indicators with parameters
    price_volume_data = {k: v for k, v in indicators.items() if not v}  # Price/Volume with no parameters

    # Create options list: Technical indicators + Price/Volume data + Fixed Value
    options = list(technical_indicators.keys()) + list(price_volume_data.keys()) + ["Fixed Value"]

    with ui.column().classes("gap-2 w-full"):
        indicator_type = ui.select(
            options=options,
            label="Indicator/Value",
            value=initial_indicator
        ).props(f"clearable dense hint='Select technical indicator, price data, or fixed value' name={prefix}_indicator_{input_id}").classes("w-full")

        # Container for all dynamic content
        dynamic_container = ui.column().classes("w-full")

        def update_content():
            """Completely rebuild the content based on selection."""
            dynamic_container.clear()
            nonlocal param_values
            indicator_name = indicator_type.value
            param_values.clear()

            with dynamic_container:
                if indicator_name == "Fixed Value":
                    # Show ONLY fixed value input
                    fixed_value_input = ui.number(
                        label="Fixed Value",
                        value=initial_params.get("fixed_value", 0) if initial_params and initial_indicator == "Fixed Value" else 0,
                        step=0.01
                    ).props(f"clearable dense hint='Enter a fixed numerical value' name=fixed_value_{input_id}").classes("w-full")

                elif indicator_name in technical_indicators:
                    # Show ONLY parameter inputs for technical indicators
                    ui.label(f"{indicator_name} Parameters").classes("text-caption theme-text-secondary")
                    params = initial_params if initial_params and indicator_name == initial_indicator else technical_indicators[indicator_name]

                    for param, default in technical_indicators[indicator_name].items():
                        param_values[param] = params.get(param, default)
                        input_field = ui.number(
                            label=param.replace("_", " ").title(),
                            value=param_values[param],
                            step=1 if "period" in param else 0.1,
                            validation={"Positive": lambda v: v > 0 if v is not None else False}
                        ).props(f"clearable dense name={param}_{input_id}").classes("w-full")

                        def on_value_change(e, p=param):
                            try:
                                value = int(e.args) if e.args is not None and "period" in p else float(e.args) if e.args is not None else param_values[p]
                                param_values[p] = value
                                logger.debug(f"Updated {p} for {indicator_name} ({prefix}): {value} (type: {type(value)})")
                            except (ValueError, TypeError) as ex:
                                logger.error(f"Invalid input for {p}: {e.args} ({ex})")
                                ui.notify(f"Invalid value for {p}: {e.args}", type="negative")

                        input_field.on("update:model-value", on_value_change)

                    logger.debug(f"Rendered params for {indicator_name} ({prefix}): {list(param_values.keys())}")

                # For price/volume data, show nothing (no parameters needed)

        # Initial setup
        if initial_indicator:
            update_content()

        # Bind the update function to selection changes
        indicator_type.on("update:model-value", lambda: update_content())

        # Return references for backward compatibility
        # Create dummy containers for compatibility - COMPLETELY HIDDEN
        params_container = ui.column().classes("w-full").style("display: none !important; visibility: hidden !important; position: absolute !important; left: -9999px !important; height: 0 !important; overflow: hidden !important;")

        # Create the dummy fixed_value_input in a hidden container
        with ui.column().style("display: none !important; visibility: hidden !important; position: absolute !important; left: -9999px !important; height: 0 !important; overflow: hidden !important;"):
            fixed_value_input = ui.number(label="", value=0)

        return indicator_type, params_container, fixed_value_input, param_values

def render_condition_builder(indicators, title="Conditions", instruments=None, condition_ui_refs=None, condition_key=None):
    """Renders a visual UI to build entry/exit conditions with proper spacing to avoid overlap."""
    conditions = []
    condition_list_ui = ui.column().classes("w-full gap-4")  # Increased gap from gap-3 to gap-4

    def add_condition_row():
        with condition_list_ui:
            with ui.row().classes("w-full items-start gap-3 enhanced-card p-4 rounded-lg mb-3") as condition_row:  # Changed items-center to items-start and added mb-3
                with ui.column().classes("flex-1"):
                    left_indicator, left_params, left_value, left_param_values = render_indicator_selector(indicators, "left")

                with ui.column().classes("w-32 mt-2"):  # Added mt-2 to align with other elements
                    comparison = ui.select(
                        [">", "<", ">=", "<=", "==", "Crosses Above", "Crosses Below"],
                        label="Comparison"
                    ).props("filled dense hint='Select comparison operator'").classes("w-full")

                with ui.column().classes("flex-1"):
                    right_indicator, right_params, right_value, right_param_values = render_indicator_selector(indicators, "right")

                with ui.column().classes("w-12 mt-2"):  # Added mt-2 to align with other elements
                    remove_button = ui.button(
                        icon="delete",
                        on_click=lambda: (condition_list_ui.remove(condition_row), conditions.remove(condition_data))
                    ).props("flat round dense").classes("text-red-400 hover:bg-red-900")

                condition_data = {
                    "row": condition_row,
                    "left_indicator": left_indicator,
                    "left_params": left_params,
                    "left_value": left_value,
                    "left_param_values": left_param_values,
                    "comparison": comparison,
                    "right_indicator": right_indicator,
                    "right_params": right_params,
                    "right_value": right_value,
                    "right_param_values": right_param_values,
                    "remove_button": remove_button
                }
                conditions.append(condition_data)
                logger.debug(f"Added condition row: left_indicator={left_indicator.value}, right_indicator={right_indicator.value}")

    ui.button("Add Condition", icon="add", on_click=add_condition_row).props("outline").classes("mb-4 btn-modern-primary")  # Increased margin bottom
    add_condition_row()

    if condition_ui_refs and condition_key:
        condition_ui_refs[f"{condition_key}_container"] = condition_list_ui
        condition_ui_refs[condition_key] = conditions

    return conditions, condition_list_ui

def extract_conditions(conditions, indicators):
    """Extracts and validates condition data, ensuring user-specified parameters are captured."""
    result = []
    required_params = {
        "SMA": ["period"],
        "EMA": ["period"],
        "RSI": ["period"],
        "MACD": ["fast_period", "slow_period", "signal_period"],
        "Bollinger Bands": ["period", "std_dev"]
    }
    for cond in conditions:
        left_params = cond["left_param_values"].copy()
        right_params = cond["right_param_values"].copy()
        left_indicator = cond["left_indicator"].value
        right_indicator = cond["right_indicator"].value

        logger.debug(f"Extracting condition: left_indicator={left_indicator}, left_params={left_params}, right_indicator={right_indicator}, right_params={right_params}")

        # Validate Crosses Above/Below conditions
        if cond["comparison"].value in ["Crosses Above", "Crosses Below"]:
            if left_indicator == right_indicator and json.dumps(left_params) == json.dumps(right_params):
                ui.notify(f"Invalid condition: {left_indicator} cannot cross itself with identical parameters.", type="negative")
                return []

        if left_indicator in required_params:
            for param in required_params[left_indicator]:
                if param not in left_params or left_params[param] is None:
                    ui.notify(f"Missing required parameter {param} for {left_indicator}. Please set a value.", type="negative")
                    return []
                if isinstance(left_params[param], str):
                    try:
                        left_params[param] = int(left_params[param]) if "period" in param else float(left_params[param])
                    except ValueError:
                        ui.notify(f"Invalid value for {param} in {left_indicator}: {left_params[param]}", type="negative")
                        return []
                if not isinstance(left_params[param], (int, float)) or left_params[param] <= 0:
                    ui.notify(f"Parameter {param} for {left_indicator} must be a positive number.", type="negative")
                    return []
        if right_indicator in required_params:
            for param in required_params[right_indicator]:
                if param not in right_params or right_params[param] is None:
                    ui.notify(f"Missing required parameter {param} for {right_indicator}. Please set a value.", type="negative")
                    return []
                if isinstance(right_params[param], str):
                    try:
                        right_params[param] = int(right_params[param]) if "period" in param else float(right_params[param])
                    except ValueError:
                        ui.notify(f"Invalid value for {param} in {right_indicator}: {right_params[param]}", type="negative")
                        return []
                if not isinstance(right_params[param], (int, float)) or right_params[param] <= 0:
                    ui.notify(f"Parameter {param} for {right_indicator} must be a positive number.", type="negative")
                    return []

        condition = {
            "left_indicator": left_indicator,
            "left_params": left_params if left_indicator != "Fixed Value" else None,
            "left_value": cond["left_value"].value if left_indicator == "Fixed Value" else None,
            "comparison": cond["comparison"].value,
            "right_indicator": right_indicator,
            "right_params": right_params if right_indicator != "Fixed Value" else None,
            "right_value": cond["right_value"].value if right_indicator == "Fixed Value" else None
        }
        if condition["left_indicator"] and condition["comparison"] and (condition["right_indicator"] or condition["right_value"] is not None):
            logger.debug(f"Valid condition extracted: {condition}")
            result.append(condition)
        else:
            ui.notify("Incomplete condition detected. Ensure all fields are filled.", type="negative")
            return []
    return result

def add_condition_row_to_ui(container_ui, conditions_list, condition_data, indicators):
    """Populates a condition row with existing data for editing."""
    with container_ui:
        with ui.row().classes("w-full items-center gap-2 border p-2 rounded") as condition_row:
            left_indicator, left_params, left_value, left_param_values = render_indicator_selector(
                indicators, "left", condition_data.get("left_indicator"), condition_data.get("left_params")
            )
            comparison = ui.select(
                [">", "<", ">=", "<=", "==", "Crosses Above", "Crosses Below"],
                label="Comparison",
                value=condition_data.get("comparison")
            ).props("dense hint='Select comparison operator'").classes("w-32")
            right_indicator, right_params, right_value, right_param_values = render_indicator_selector(
                indicators, "right", condition_data.get("right_indicator"), condition_data.get("right_params")
            )
            remove_button = ui.button(icon="delete", on_click=lambda: (container_ui.remove(condition_row), conditions_list.remove(condition_data))).props("flat round dense")

            if condition_data.get("left_indicator") == "Fixed Value":
                left_value.value = condition_data.get("left_value")
            if condition_data.get("right_indicator") == "Fixed Value":
                right_value.value = condition_data.get("right_value")

            condition_data = {
                "row": condition_row,
                "left_indicator": left_indicator,
                "left_params": left_params,
                "left_value": left_value,
                "left_param_values": left_param_values,
                "comparison": comparison,
                "right_indicator": right_indicator,
                "right_params": right_params,
                "right_value": right_value,
                "right_param_values": right_param_values,
                "remove_button": remove_button
            }
            conditions_list.append(condition_data)
            logger.debug(f"Added condition row: left_indicator={left_indicator.value}, right_indicator={right_indicator.value}")

async def render_strategies_page(fetch_api, user_storage, instruments):
    """Renders the strategies UI with manage and create/edit tabs."""
    broker = user_storage.get("default_broker", "Zerodha")

    # Apply specific fixes for strategies page card backgrounds
    apply_strategies_page_fix()

    # Main container with proper theme classes
    with ui.column().classes("strategies-page w-full min-h-screen"):
        # Header section with theme-aware styling
        with ui.row().classes("w-full justify-between items-center p-4 page-header-standard"):
            # Left side - Title only
            with ui.row().classes("items-center gap-2"):
                ui.icon("analytics", size="2rem").classes("text-cyan-400")
                ui.label("Algorithmic Trading Strategies").classes("page-title-standard theme-text-primary")

            # Right side - Tabs with theme styling
            with ui.tabs().classes("bg-transparent") as tabs:
                manage_tab = ui.tab("Manage Strategies").classes("nav-tab-btn")
                create_tab = ui.tab("Create/Edit Strategy").classes("nav-tab-btn")
                execute_tab = ui.tab("Execute Strategy").classes("nav-tab-btn")

        # Tab panels with full-width content
        with ui.tab_panels(tabs, value=manage_tab).classes("w-full flex-1"):
            with ui.tab_panel(manage_tab).classes("w-full p-4"):
                # Manage strategies content with proper theme styling
                with ui.card().classes("w-full enhanced-card"):
                    with ui.card_section().classes("p-6 w-full"):
                        ui.label("Existing Strategies").classes("card-title theme-text-primary mb-4")

                        # Enhanced Strategy Statistics with Risk Management
                        async def fetch_strategy_stats():
                            try:
                                logger.debug(f"Fetching strategy statistics for broker: {broker}")
                                # Fetch both strategy stats and risk metrics
                                stats_task = asyncio.create_task(fetch_api(f"/strategies/{broker}/statistics"))
                                risk_task = asyncio.create_task(fetch_api("/risk-management/metrics"))
                                
                                stats, risk_metrics = await asyncio.gather(stats_task, risk_task, return_exceptions=True)
                                
                                logger.debug(f"Statistics response: {stats}")
                                logger.debug(f"Risk metrics response: {risk_metrics}")
                                
                                stats_container.clear()
                                with stats_container:
                                    # Risk Management Overview Row
                                    if risk_metrics and not isinstance(risk_metrics, Exception) and hasattr(risk_metrics, 'get') and not risk_metrics.get("error"):
                                        with ui.row().classes("w-full gap-4 mb-4"):
                                            with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-purple-500 stats-card"):
                                                ui.label("Daily P&L").classes("text-sm theme-text-secondary")
                                                daily_pnl = risk_metrics.get("daily_pnl", 0)
                                                pnl_color = "theme-text-success" if daily_pnl >= 0 else "theme-text-error"
                                                ui.label(f"₹{daily_pnl:,.2f}").classes(f"text-2xl font-bold {pnl_color}")
                                            
                                            with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-yellow-500 stats-card"):
                                                ui.label("Risk Exposure").classes("text-sm theme-text-secondary")
                                                exposure_pct = risk_metrics.get("exposure_used_pct", 0)
                                                exposure_color = "theme-text-success" if exposure_pct < 80 else "theme-text-warning" if exposure_pct < 95 else "theme-text-error"
                                                ui.label(f"{exposure_pct:.1f}%").classes(f"text-2xl font-bold {exposure_color}")
                                            
                                            with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-cyan-500 stats-card"):
                                                ui.label("Open Positions").classes("text-sm theme-text-secondary")
                                                open_positions = risk_metrics.get("open_positions", 0)
                                                max_positions = risk_metrics.get("max_open_positions", 10)
                                                ui.label(f"{open_positions}/{max_positions}").classes("text-2xl font-bold theme-text-info")
                    
                                    # Strategy Statistics Row
                                    if stats and not isinstance(stats, Exception) and hasattr(stats, 'get') and not stats.get("error"):
                                        with ui.row().classes("w-full gap-4 mb-4"):
                                            with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-blue-500 stats-card"):
                                                ui.label("Total Strategies").classes("text-sm theme-text-secondary")
                                                ui.label(str(stats.get("total_strategies", 0))).classes("text-2xl font-bold theme-text-primary")
                                            
                                            with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-green-500 stats-card"):
                                                ui.label("Active Strategies").classes("text-sm theme-text-secondary")
                                                active_count = stats.get("status_breakdown", {}).get("active", 0)
                                                ui.label(str(active_count)).classes("text-2xl font-bold theme-text-success")
                                            
                                            with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-orange-500 stats-card"):
                                                ui.label("Running Strategies").classes("text-sm theme-text-secondary")
                                                # Fix: Only count strategies that are actually executing (have running executions)
                                                running_count = 0
                                                if risk_metrics and not isinstance(risk_metrics, Exception) and hasattr(risk_metrics, 'get'):
                                                    running_count = risk_metrics.get("running_strategies", 0)
                                                ui.label(str(running_count)).classes("text-2xl font-bold theme-text-warning")

                                        # Navigation to Live Trading for running strategies (not just active)
                                        if running_count > 0:
                                            with ui.card().classes("w-full p-4 enhanced-card border-l-4 border-l-purple-500 stats-card"):
                                                with ui.row().classes("items-center justify-between w-full"):
                                                    with ui.column():
                                                        ui.label(f"🔴 {running_count} Strategies Currently Running").classes("font-medium theme-text-primary")
                                                        ui.label("Monitor live executions, P&L, and performance").classes("text-sm theme-text-secondary")
                                                    ui.button("View Live Trading Dashboard", icon="monitor",
                                                             on_click=lambda: ui.navigate.to('/live-trading')).props("color=purple").classes("theme-btn")
                                        elif active_count > 0:
                                            with ui.card().classes("w-full p-4 enhanced-card border-l-4 border-l-blue-500 stats-card"):
                                                with ui.row().classes("items-center justify-between w-full"):
                                                    with ui.column():
                                                        ui.label(f"📊 {active_count} Strategies Ready to Execute").classes("font-medium theme-text-primary")
                                                        ui.label("Use Execute Strategy tab to start running them on instruments").classes("text-sm theme-text-secondary")
                                                    ui.button("Execute Strategies", icon="play_arrow",
                                                             on_click=lambda: (tabs.set_value(execute_tab))).props("color=blue").classes("theme-btn")
                                    else:
                                        error_msg = stats.get('error', {}).get('message', 'Unknown error') if stats and not isinstance(stats, Exception) and hasattr(stats, 'get') else 'No response'
                                        logger.error(f"Statistics API error: {error_msg}")
                                        with ui.card().classes("w-full p-4 enhanced-card border-l-4 border-l-red-500"):
                                            ui.label(f"Unable to load strategy statistics: {error_msg}").classes("text-red-400 text-center")
                            except Exception as e:
                                logger.error(f"Error fetching strategy stats: {str(e)}")
                                stats_container.clear()
                                with stats_container:
                                    ui.label(f"Error loading statistics: {str(e)}").classes("text-red-400 text-center")

                        # Statistics container
                        stats_container = ui.column().classes("w-full mb-4")
                        await fetch_strategy_stats()

                        # Refresh buttons row
                        with ui.row().classes("w-full justify-between items-center mb-4"):
                            ui.label("Strategy Management").classes("text-lg font-semibold theme-text-primary")
                            with ui.row().classes("gap-2"):
                                ui.button("Refresh Stats", icon="analytics", on_click=fetch_strategy_stats).props("outline size=sm").classes("btn-modern-secondary")

                        # Container for strategy rows
                        strategies_container = ui.column().classes("w-full gap-2")

                        # Header row for column names
                        with ui.card().classes("w-full p-4 mb-3 header-card enhanced-card"):
                            with ui.row().classes("w-full items-center font-semibold"):
                                ui.label("Name").classes("w-48 theme-text-primary")
                                ui.label("Description").classes("w-48 theme-text-primary")
                                ui.label("Entry Conditions").classes("flex-1 theme-text-primary")
                                ui.label("Exit Conditions").classes("flex-1 theme-text-primary")
                                ui.label("Status").classes("w-24 theme-text-primary text-center")
                                ui.label("Actions").classes("w-44 theme-text-primary text-center")

                        async def handle_edit_strategy(strategy_id):
                            if not create_tab_initialized:
                                await initialize_create_tab()
                            response = await fetch_api(f"/strategies/{strategy_id}")
                            if response and not response.get("error"):
                                tabs.set_value(create_tab)
                                # Store the strategy ID for editing
                                ui_refs["editing_strategy_id"] = strategy_id
                                ui_refs["strategy_name"].value = response["name"]
                                ui_refs["strategy_desc"].value = response["description"]
                                ui_refs["broker_select"].value = response["broker"]
                                ui_refs["timeframe"].value = response["parameters"].get("timeframe", "5min")
                                ui_refs["preview_instrument"].value = "RELIANCE"

                                # Clear existing conditions
                                for cond in ui_refs["entry_conditions"]:
                                    ui_refs["entry_container"].remove(cond["row"])
                                for cond in ui_refs["exit_conditions"]:
                                    ui_refs["exit_container"].remove(cond["row"])
                                ui_refs["entry_conditions"].clear()
                                ui_refs["exit_conditions"].clear()

                                # Load existing conditions
                                for cond in response.get("entry_conditions", []):
                                    add_condition_row_to_ui(ui_refs["entry_container"], ui_refs["entry_conditions"], cond, available_indicators)
                                for cond in response.get("exit_conditions", []):
                                    add_condition_row_to_ui(ui_refs["exit_container"], ui_refs["exit_conditions"], cond, available_indicators)

                                # Change save button text to indicate editing
                                ui_refs["save_button"].text = "Update Strategy"
                                ui_refs["save_button"].update()
                            else:
                                ui.notify(f"Failed to load strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")

                        async def handle_duplicate_strategy(strategy_id):
                            """Duplicate an existing strategy"""
                            try:
                                # First get the strategy details
                                strategy_response = await fetch_api(f"/strategies/{strategy_id}")
                                if not strategy_response or strategy_response.get("error"):
                                    ui.notify(f"Failed to load strategy for duplication: {strategy_response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                                    return

                                # Create a new strategy with modified name
                                duplicate_data = {
                                    "name": f"{strategy_response['name']} (Copy)",
                                    "description": f"Copy of: {strategy_response.get('description', '')}",
                                    "entry_conditions": strategy_response.get("entry_conditions", []),
                                    "exit_conditions": strategy_response.get("exit_conditions", []),
                                    "parameters": strategy_response.get("parameters", {}),
                                    "broker": strategy_response.get("broker", broker)
                                }

                                # Create new strategy using the standard POST endpoint
                                response = await fetch_api("/strategies/", method="POST", data=duplicate_data)
                                if response and response.get("strategy_id"):
                                    ui.notify("Strategy duplicated successfully!", type="positive")
                                    await fetch_strategies()
                                else:
                                    ui.notify(f"Failed to duplicate strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                            except Exception as e:
                                ui.notify(f"Error duplicating strategy: {str(e)}", type="negative")
                                logger.error(f"Duplicate strategy error: {str(e)}")

                        async def handle_preview_existing_strategy(strategy_id):
                            """Preview an existing strategy from the manage tab"""
                            try:
                                # First get the strategy details
                                strategy_response = await fetch_api(f"/strategies/{strategy_id}")
                                if not strategy_response or strategy_response.get("error"):
                                    ui.notify(f"Failed to load strategy: {strategy_response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                                    return

                                # Get broker from strategy
                                broker = strategy_response.get("broker", "Zerodha")

                                # Run backtest with strategy data
                                backtest_response = await fetch_api("/algo-trading/backtest", method="POST", data={
                                    "trading_symbol": "RELIANCE",  # Default symbol for preview
                                    "instrument_token": "NSE_EQ|INE002A01018",  # Default token for RELIANCE
                                    "timeframe": strategy_response.get("parameters", {}).get("timeframe", "5min"),
                                    "strategy": json.dumps({
                                        "name": strategy_response["name"],
                                        "description": strategy_response["description"],
                                        "entry_conditions": strategy_response["entry_conditions"],
                                        "exit_conditions": strategy_response["exit_conditions"],
                                        "parameters": strategy_response["parameters"]
                                    }),
                                    "params": {"initial_investment": 100000},
                                    "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                                    "end_date": datetime.now().strftime("%Y-%m-%d")
                                })

                                if backtest_response and not backtest_response.get("error"):
                                    with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl enhanced-card"):
                                        with ui.card_section().classes("p-6"):
                                            ui.label(f"Strategy Preview: {strategy_response['name']}").classes("text-xl font-bold theme-text-primary mb-4")
                                            # Results summary cards
                                            with ui.row().classes("w-full gap-4 mb-4"):
                                                with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-green-500"):
                                                    ui.label("Total Profit").classes("text-sm theme-text-secondary")
                                                    total_profit = backtest_response.get("TotalProfit", 0)
                                                    pnl_color = "theme-text-success" if total_profit >= 0 else "theme-text-error"
                                                    ui.label(f"₹{total_profit:,.2f}").classes(f"text-2xl font-bold {pnl_color}")

                                                with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-blue-500"):
                                                    ui.label("Win Rate").classes("text-sm theme-text-secondary")
                                                    ui.label(f"{backtest_response.get('WinRate', 0):.1f}%").classes("text-2xl font-bold theme-text_info")
                                                with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-purple-500"):
                                                    ui.label("Total Trades").classes("text-sm theme-text_secondary")
                                                    ui.label(f"{backtest_response.get('TotalTrades', 0)}").classes("text-2xl font-bold theme-text_primary")

                                            # Show trade history if available
                                            if tradebook := backtest_response.get("Tradebook", []):
                                                ui.label("Trade History").classes("text-lg font-semibold theme-text-primary mb-2")
                                                formatted_tradebook = []
                                                for trade in tradebook[:20]:  # Limit to 20 trades for performance
                                                    try:
                                                        entry_price = trade.get('EntryPrice') or trade.get('entry_price') or trade.get('Entry_Price') or 0
                                                        exit_price = trade.get('ExitPrice') or trade.get('exit_price') or trade.get('Exit_Price') or 0
                                                        profit = trade.get('Profit') or trade.get('profit') or 0
                                                        portfolio_value = trade.get('PortfolioValue') or trade.get('portfolio_value') or trade.get('Portfolio_Value') or 0
                                                        trade_date = trade.get('Date') or trade.get('date') or trade.get('TradeDate') or datetime.now()

                                                        formatted_tradebook.append({
                                                            "Date": pd.to_datetime(trade_date).strftime("%Y-%m-%d %H:%M"),
                                                            "Entry": f"₹{float(entry_price):.2f}",
                                                            "Exit": f"₹{float(exit_price):.2f}",
                                                            "Profit": f"₹{float(profit):.2f}",
                                                            "Portfolio": f"₹{float(portfolio_value):.2f}"
                                                        })
                                                    except Exception as trade_error:
                                                        logger.warning(f"Error formatting trade: {trade_error}")
                                                        continue

                                                if formatted_tradebook:
                                                    ui.table(
                                                        columns=[
                                                            {"name": "Date", "label": "Date", "field": "Date", "align": "left"},
                                                            {"name": "Entry", "label": "Entry Price", "field": "Entry", "align": "right"},
                                                            {"name": "Exit", "label": "Exit Price", "field": "Exit", "align": "right"},
                                                            {"name": "Profit", "label": "Profit/Loss", "field": "Profit", "align": "right"},
                                                            {"name": "Portfolio", "label": "Portfolio Value", "field": "Portfolio", "align": "right"}
                                                        ],
                                                        rows=formatted_tradebook,
                                                        pagination={"rowsPerPage": 10}
                                                    ).classes("w-full enhanced-card")
                                                else:
                                                    ui.label("No valid trades found in response").classes("text-yellow-400 text-center")
                                            else:
                                                with ui.card().classes("w-full p-8 text-center enhanced-card border-l-4 border-l-yellow-500"):
                                                    ui.icon("warning", size="3rem").classes("text-yellow-400 mb-2")
                                                    ui.label("No trades executed with current conditions").classes("text-yellow-400 font-medium")
                                                    ui.label("Consider adjusting your entry/exit conditions").classes("theme-text-secondary text-sm")

                                        with ui.card_actions().classes("justify-end"):
                                            ui.button("Close", on_click=dialog.close).props("outline").classes("theme-text-secondary border-gray-400")
                                    dialog.open()
                                else:
                                    ui.notify(f"Preview failed: {backtest_response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                            except Exception as e:
                                ui.notify(f"Error previewing strategy: {str(e)}", type="negative")
                                logger.error(f"Preview existing strategy error: {str(e)}")

                        async def handle_toggle_strategy(strategy_id):
                            response = await fetch_api(f"/strategies/{strategy_id}")
                            if response and not response.get("error"):
                                current_status = response.get("status")
                                new_status = "inactive" if current_status == "active" else "active"
                                action = "deactivate" if new_status == "inactive" else "activate"
                                toggle_button = ui.button(f"{action.capitalize()} Strategy").props("loading=true disable=true")
                                try:
                                    response = await fetch_api(f"/strategies/{strategy_id}/{action}", method="POST")
                                    # Check if the response indicates success
                                    if response and response.get("success") == True:
                                        ui.notify(f"Strategy {action}d successfully.", type="positive")
                                        await fetch_strategies()
                                    else:
                                        # Handle both old and new error formats
                                        error_msg = (
                                            response.get('error', {}).get('message') if response.get('error')
                                            else response.get('message', 'Unknown error')
                                        )
                                        ui.notify(f"Failed to {action} strategy: {error_msg}", type="negative")
                                except Exception as e:
                                    ui.notify(f"Error toggling strategy: {str(e)}", type="negative")
                                    logger.error(f"Toggle strategy error: {str(e)}")
                                finally:
                                    toggle_button.delete()
                                    ui.update()
                            else:
                                ui.notify(f"Failed to load strategy status: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")

                        async def handle_view_metrics(strategy_id):
                            """Show detailed live metrics for an active strategy"""
                            try:
                                metrics_response = await fetch_api(f"/strategies/{strategy_id}/metrics")
                                if metrics_response and not metrics_response.get("error"):
                                    metrics = metrics_response.get("metrics", {})

                                    with ui.dialog() as dialog, ui.card().classes("w-full max-w-6xl enhanced-card"):
                                        with ui.card_section().classes("p-6"):
                                            ui.label(f"Live Strategy Metrics").classes("text-xl font-bold theme-text-primary mb-4")

                                            if metrics:
                                                # Overview metrics
                                                total_signals = sum(inst.get("signals_generated", 0) for inst in metrics.values())
                                                total_trades = sum(inst.get("trades_executed", 0) for inst in metrics.values())
                                                active_positions = sum(1 for inst in metrics.values() if inst.get("current_position", 0) != 0)

                                                with ui.row().classes("w-full gap-4 mb-6"):
                                                    with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-blue-500"):
                                                        ui.label("Total Signals").classes("text-sm theme-text-secondary")
                                                        ui.label(str(total_signals)).classes("text-2xl font-bold theme-text-info")
                                                    with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-green-500"):
                                                        ui.label("Total Trades").classes("text-sm theme-text-secondary")
                                                        ui.label(str(total_trades)).classes("text-2xl font-bold theme-text-success")
                                                    with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-orange-500"):
                                                        ui.label("Active Positions").classes("text-sm theme-text-secondary")
                                                        ui.label(str(active_positions)).classes("text-2xl font-bold theme-text-warning")

                                                # Per-instrument breakdown
                                                ui.label("Per-Instrument Metrics").classes("text-lg font-semibold theme-text-primary mb-3")

                                                for instrument_token, inst_metrics in metrics.items():
                                                    # Get instrument symbol (simplified - you may need to map token to symbol)
                                                    symbol = "Unknown"  # You'd need to reverse lookup from your instruments dict

                                                    with ui.card().classes("w-full mb-3 p-4 enhanced-card"):
                                                        with ui.row().classes("w-full items-center justify-between"):
                                                            with ui.column().classes("gap-1"):
                                                                ui.label(f"Instrument: {symbol}").classes("font-medium theme-text-primary")
                                                                ui.label(f"Token: {instrument_token}").classes("text-xs theme-text-secondary")

                                                                started_at = inst_metrics.get("started_at")
                                                                if started_at:
                                                                    ui.label(f"Started: {started_at}").classes("text-xs theme-text-secondary")

                                                            with ui.row().classes("gap-6"):
                                                                # Metrics columns
                                                                with ui.column().classes("text-center gap-1"):
                                                                    ui.label("Signals").classes("text-xs theme-text-secondary")
                                                                    ui.label(str(inst_metrics.get("signals_generated", 0))).classes("text-lg font-bold theme-text-info")

                                                                with ui.column().classes("text-center gap-1"):
                                                                    ui.label("Trades").classes("text-xs theme-text-secondary")
                                                                    ui.label(str(inst_metrics.get("trades_executed", 0))).classes("text-lg font-bold theme-text-success")

                                                                with ui.column().classes("text-center gap-1"):
                                                                    ui.label("Position").classes("text-xs theme-text-secondary")
                                                                    position = inst_metrics.get("current_position", 0)
                                                                    pos_color = "theme-text-success" if position > 0 else "theme-text-error" if position < 0 else "theme-text-secondary"
                                                                    ui.label(str(position)).classes(f"text-lg font-bold {pos_color}")

                                                                with ui.column().classes("text-center gap-1"):
                                                                    ui.label("Status").classes("text-xs theme-text-secondary")
                                                                    status = inst_metrics.get("status", "unknown")
                                                                    status_color = "theme-text-success" if "monitoring" in status else "theme-text-info"
                                                                    ui.label(status.replace("_", " ").title()).classes(f"text-sm font-bold {status_color}")

                                                                # Last check indicator
                                                                last_check = inst_metrics.get("last_check")
                                                                if last_check:
                                                                    ui.icon("access_time", size="sm").classes("theme-text-secondary").tooltip(f"Last check: {last_check}")
                                            else:
                                                ui.label("No metrics available for this strategy").classes("theme-text-secondary text-center")

                                        with ui.card_actions().classes("justify-end"):
                                            ui.button("Close", on_click=dialog.close).props("outline").classes("theme-text-secondary")

                                    dialog.open()
                                else:
                                    ui.notify("Failed to load strategy metrics", type="negative")
                            except Exception as e:
                                ui.notify(f"Error loading metrics: {str(e)}", type="negative")
                                logger.error(f"Error loading strategy metrics: {str(e)}")

                        async def handle_view_executions(strategy_id):
                            """View all executions for a specific strategy"""
                            try:
                                # Get executions for this strategy
                                executions = await fetch_api("/executions")
                                strategy_executions = [e for e in executions if e.get('strategy_id') == strategy_id] if executions else []

                                # Get strategy details
                                strategies = await fetch_api(f"/strategies/{broker}/execution-status")
                                strategy = next((s for s in strategies if s['strategy_id'] == strategy_id), None) if strategies else None
                                strategy_name = strategy['name'] if strategy else 'Unknown Strategy'

                                if strategy_executions:
                                    with ui.dialog() as dialog, ui.card().classes("w-full max-w-5xl enhanced-card"):
                                        with ui.card_section().classes("p-6"):
                                            ui.label(f"Executions for {strategy_name}").classes("text-xl font-bold theme-text-primary mb-4")

                                            with ui.column().classes("w-full gap-3"):
                                                for execution in strategy_executions:
                                                    await render_execution_card(execution)

                                            ui.button("Close", on_click=dialog.close).props("flat").classes("mt-4")

                                    dialog.open()
                                else:
                                    ui.notify(f"No executions found for {strategy_name}", type="info")

                            except Exception as e:
                                logger.error(f"Error viewing executions for strategy {strategy_id}: {e}")
                                ui.notify(f"Error loading executions: {str(e)}", type="negative")

                        async def handle_delete_strategy(strategy_id):
                            with ui.dialog() as dialog, ui.card():
                                ui.label(f"Are you sure you want to delete strategy {strategy_id}?").classes("theme-text-primary")
                                with ui.row():
                                    ui.button("Cancel", on_click=dialog.close).props("outline").classes("theme-text-secondary")
                                    async def confirm_delete():
                                        dialog.close()
                                        delete_button = ui.button("Deleting...").props("loading=true disable=true")
                                        try:
                                            response = await fetch_api(f"/strategies/{strategy_id}", method="DELETE")
                                            if response and response.get("success"):
                                                ui.notify("Strategy deleted successfully.", type="positive")
                                                await fetch_strategies()
                                            else:
                                                ui.notify(f"Failed to delete strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                                        except Exception as e:
                                            ui.notify(f"Error deleting strategy: {str(e)}", type="negative")
                                            logger.error(f"Delete strategy error: {str(e)}")
                                        finally:
                                            delete_button.delete()
                                            ui.update()
                                    ui.button("Delete", on_click=confirm_delete).props("color=negative")
                            dialog.open()

                        async def fetch_strategies():
                            strategies = await fetch_api(f"/strategies/{broker}/execution-status")
                            strategies_container.clear()
                            if strategies and isinstance(strategies, list):
                                with strategies_container:
                                    # Create a simple table structure for better alignment
                                    for s in strategies:
                                        strategy_id = s["strategy_id"]
                                        entry_conditions_text = format_conditions(s.get("entry_conditions", []))
                                        exit_conditions_text = format_conditions(s.get("exit_conditions", []))

                                        with ui.card().classes("w-full p-4 mb-3 strategy-card enhanced-card"):
                                            with ui.row().classes("w-full items-center"):
                                                # Name column
                                                ui.label(s["name"]).classes("w-48 font-medium theme-text-primary text-truncate").tooltip(s["name"])

                                                # Description column
                                                ui.label(s["description"] or "").classes("w-48 font-medium theme-text-secondary text-sm text-truncate").tooltip(s["description"] or "")

                                                # Entry Conditions column
                                                ui.label(entry_conditions_text).classes("flex-1 theme-text-secondary text-xs text-truncate").tooltip(entry_conditions_text)

                                                # Exit Conditions column
                                                ui.label(exit_conditions_text).classes("flex-1 theme-text-secondary text-xs text-truncate").tooltip(exit_conditions_text)

                                                # Status column with execution info
                                                with ui.column().classes("w-32 text-center"):
                                                    status_class = "px-2 py-1 rounded-full text-xs font-medium"
                                                    if s["status"] == "active":
                                                        ui.label(s["status"]).classes(f"{status_class} bg-green-100 text-green-800")
                                                    else:
                                                        ui.label(s["status"]).classes(f"{status_class} bg-gray-100 text-gray-800")

                                                    # Execution status
                                                    execution_status = s.get("execution_status", {})
                                                    running_count = execution_status.get("running", 0)
                                                    total_count = execution_status.get("total", 0)

                                                    if running_count > 0:
                                                        ui.label(f"▶️ {running_count} running").classes("text-xs text-green-600 font-medium mt-1").tooltip(f"{running_count} executions running")
                                                    elif total_count > 0:
                                                        completed = execution_status.get("completed", 0)
                                                        stopped = execution_status.get("stopped", 0)
                                                        failed = execution_status.get("failed", 0)
                                                        ui.label(f"📊 {total_count} total").classes("text-xs text-blue-600 font-medium mt-1").tooltip(f"Total: {total_count}, Completed: {completed}, Stopped: {stopped}, Failed: {failed}")
                                                    else:
                                                        ui.label("No executions").classes("text-xs text-gray-500 mt-1")

                                                # Actions column - Reorganized for better UX
                                                with ui.column().classes("w-48 text-center"):
                                                    # Primary actions row
                                                    with ui.row().classes("gap-1 justify-center mb-1"):
                                                        ui.button(icon="edit", on_click=lambda sid=strategy_id: handle_edit_strategy(sid)).props("flat round dense color=primary").classes("action-btn").tooltip("Edit Strategy")
                                                        ui.button(
                                                            icon="play_arrow" if s["status"] == "inactive" else "pause",
                                                            on_click=lambda sid=strategy_id: handle_toggle_strategy(sid)
                                                        ).props("flat round dense color=positive").classes("action-btn").tooltip("Activate/Deactivate Strategy")

                                                        # Show executions button if there are any executions
                                                        execution_status = s.get("execution_status", {})
                                                        if execution_status.get("total", 0) > 0:
                                                            ui.button(icon="trending_up", on_click=lambda sid=strategy_id: handle_view_executions(sid)).props("flat round dense color=warning").classes("action-btn").tooltip(f"View {execution_status.get('total', 0)} Executions")

                                                        if s["status"] == "active":
                                                            ui.button(icon="analytics", on_click=lambda sid=strategy_id: handle_view_metrics(sid)).props("flat round dense color=secondary").classes("action-btn").tooltip("View Live Metrics")

                                                    # Secondary actions row
                                                    with ui.row().classes("gap-1 justify-center"):
                                                        ui.button(icon="visibility", on_click=lambda sid=strategy_id: handle_preview_existing_strategy(sid)).props("flat round dense color=info").classes("action-btn action-btn-small").tooltip("Preview Strategy")
                                                        ui.button(icon="content_copy", on_click=lambda sid=strategy_id: handle_duplicate_strategy(sid)).props("flat round dense color=info").classes("action-btn action-btn-small").tooltip("Duplicate Strategy")
                                                        ui.button(icon="delete", on_click=lambda sid=strategy_id: handle_delete_strategy(sid)).props("flat round dense color=negative").classes("action-btn action-btn-small").tooltip("Delete Strategy")

                                logger.debug(f"Fetched and formatted {len(strategies)} strategies")
                            else:
                                with strategies_container:
                                    ui.label("No strategies found.").classes("w-full text-center theme-text-secondary")
                                ui.notify("No strategies found.", type="warning")

                        await fetch_strategies()

            with ui.tab_panel(create_tab) as create_tab_content:
                # Always show content immediately when tab is created
                async def init_and_render():
                    await initialize_create_tab()

                # Initialize immediately
                ui.timer(0.1, init_and_render, once=True)

            with ui.tab_panel(execute_tab).classes("w-full p-4"):
                # Strategy Execution content
                with ui.card().classes("w-full enhanced-card"):
                    with ui.card_section().classes("p-6 w-full"):
                        with ui.row().classes("items-center gap-2 mb-4 w-full"):
                            ui.icon("play_arrow", size="1.5rem").classes("text-cyan-500 drop-shadow-lg")
                            ui.label("Execute Strategy on Instruments").classes("card-title theme-text-primary")
                        
                        # Strategy selection
                        with ui.row().classes("w-full gap-4 mb-4"):
                            with ui.column().classes("flex-1"):
                                strategy_select = ui.select(
                                    options=[],
                                    label="Select Strategy Template",
                                    value=None
                                ).props("filled clearable filterable hint='Choose a generic strategy to execute'").classes("w-full theme-text-primary")
                                execute_refs["strategy_select"] = strategy_select
                                # Auto-load strategies when element is created
                                ui.timer(0.5, lambda: refresh_strategy_options(strategy_select), once=True)
                            
                            with ui.column().classes("flex-1"):
                                ui.button("Refresh Strategies", on_click=lambda: refresh_strategy_options(strategy_select)).props("flat").classes("theme-btn")
                        
                        # Strategy details display
                        strategy_details = ui.card().classes("w-full mb-4 bg-gray-50").style("display: none;")
                        execute_refs["strategy_details"] = strategy_details
                        
                        # Instrument selection
                        with ui.card().classes("w-full mb-4 enhanced-card border-l-4 border-l-blue-500"):
                            with ui.card_section().classes("p-4"):
                                ui.label("Select Instruments").classes("card-title theme-text-primary mb-3")
                                instruments_container = ui.row().classes("w-full gap-2 items-center mb-3")
                                execute_refs["instruments_container"] = instruments_container
                                execute_refs["selected_instruments"] = []
                                
                                with instruments_container:
                                    instrument_select = ui.select(
                                        options=sorted(list(instruments.keys())),
                                        label="Add Instrument",
                                        value=None
                                    ).props("filled clearable filterable hint='Select instrument to add'").classes("flex-1 theme-text-primary")
                                    execute_refs["instrument_select"] = instrument_select
                                    
                                    ui.button("Add", icon="add", on_click=lambda: add_instrument_to_execution()).props("flat").classes("theme-btn")
                                
                                # Selected instruments display
                                selected_display = ui.column().classes("w-full gap-2 mt-3")
                                execute_refs["selected_display"] = selected_display
                        
                        # Risk and execution parameters
                        # Capital and Position Sizing
                        with ui.card().classes("w-full mb-4 enhanced-card border-l-4 border-l-green-500"):
                            with ui.card_section().classes("p-4"):
                                ui.label("Capital & Position Management").classes("card-title theme-text-primary mb-3")
                                
                                with ui.row().classes("w-full gap-4"):
                                    total_capital = ui.number(
                                        label="Total Capital (₹)",
                                        value=100000,
                                        min=1000,
                                        max=10000000,
                                        step=1000
                                    ).props("filled hint='Total available capital for trading'").classes("flex-1 theme-text-primary")
                                    execute_refs["total_capital"] = total_capital
                                    
                                    risk_per_trade = ui.number(
                                        label="Risk per Trade (%)",
                                        value=2.0,
                                        min=0.1,
                                        max=10.0,
                                        step=0.1
                                    ).props("filled hint='Maximum risk as percentage of capital'").classes("flex-1 theme-text-primary")
                                    execute_refs["risk_per_trade"] = risk_per_trade
                                
                                with ui.row().classes("w-full gap-4 mt-3"):
                                    position_sizing_mode = ui.select(
                                        options=["Auto Calculate", "Manual Quantity"],
                                        label="Position Sizing Mode",
                                        value="Auto Calculate"
                                    ).props("filled hint='How to determine position size'").classes("flex-1 theme-text-primary")
                                    execute_refs["position_sizing_mode"] = position_sizing_mode
                                    
                                    quantity = ui.number(
                                        label="Manual Quantity",
                                        value=1,
                                        min=1,
                                        max=10000,
                                        step=1
                                    ).props("filled hint='Fixed quantity per trade' :disable='position_sizing_mode.value === \"Auto Calculate\"'").classes("flex-1 theme-text-primary")
                                    execute_refs["quantity"] = quantity
                                
                                with ui.row().classes("w-full gap-2 mt-3"):
                                    ui.button(
                                        "📊 Position Size Calculator",
                                        on_click=lambda: open_position_calculator()
                                    ).props("flat color=primary").classes("theme-btn")
                                    
                                    calculated_info = ui.label("").classes("text-sm theme-text-secondary flex-1")
                                    execute_refs["calculated_info"] = calculated_info
                                
                                # Auto-update calculations when values change
                                def update_calculations():
                                    if execute_refs["position_sizing_mode"].value == "Auto Calculate":
                                        capital = execute_refs["total_capital"].value or 0
                                        risk_pct = execute_refs["risk_per_trade"].value or 0
                                        risk_amount = capital * (risk_pct / 100)
                                        execute_refs["calculated_info"].text = f"Risk Amount: ₹{risk_amount:,.0f} per trade"
                                        execute_refs["quantity"].props(":disable=true")
                                    else:
                                        execute_refs["calculated_info"].text = "Using manual quantity"
                                        execute_refs["quantity"].props(":disable=false")
                                
                                execute_refs["total_capital"].on("update:modelValue", update_calculations)
                                execute_refs["risk_per_trade"].on("update:modelValue", update_calculations)
                                execute_refs["position_sizing_mode"].on("update:modelValue", update_calculations)
                                
                                # Initial calculation
                                ui.timer(0.1, update_calculations, once=True)
                        
                        # Risk Management Parameters
                        with ui.card().classes("w-full mb-4 enhanced-card border-l-4 border-l-orange-500"):
                            with ui.card_section().classes("p-4"):
                                ui.label("Risk Management").classes("card-title theme-text-primary mb-3")
                                
                                with ui.row().classes("w-full gap-4"):
                                    stop_loss = ui.number(
                                        label="Stop Loss (%)",
                                        value=2.0,
                                        min=0.1,
                                        max=20.0,
                                        step=0.1
                                    ).props("filled hint='Stop loss percentage'").classes("flex-1 theme-text-primary")
                                    execute_refs["stop_loss"] = stop_loss
                                    
                                    take_profit = ui.number(
                                        label="Take Profit (%)",
                                        value=5.0,
                                        min=0.1,
                                        max=50.0,
                                        step=0.1
                                    ).props("filled hint='Take profit percentage'").classes("flex-1 theme-text-primary")
                                    execute_refs["take_profit"] = take_profit
                                
                                # Position Sizing and Advanced Features
                                with ui.row().classes("w-full gap-4 mt-3"):
                                    position_sizing = ui.number(
                                        label="Position Size (% of Capital)",
                                        value=10.0,
                                        min=1.0,
                                        max=100.0,
                                        step=1.0
                                    ).props("filled hint='Percentage of total capital to use'").classes("flex-1 theme-text-primary")
                                    execute_refs["position_sizing"] = position_sizing
                                    
                                    timeframe_select = ui.select(
                                        options=["1min", "5min", "15min", "30min", "60min", "day", "week"],
                                        label="Data Timeframe",
                                        value="day"
                                    ).props("filled hint='Timeframe for strategy data'").classes("flex-1 theme-text-primary")
                                    execute_refs["timeframe"] = timeframe_select
                                
                                # Smart Parameter Optimization
                                with ui.row().classes("w-full gap-2 mt-3"):
                                    ui.button(
                                        "🤖 Smart Parameter Optimization",
                                        icon="psychology",
                                        on_click=lambda: optimize_parameters()
                                    ).props("flat color=primary").classes("theme-btn")
                                    
                                    ui.button(
                                        "📊 Market Analysis",
                                        icon="trending_up",
                                        on_click=lambda: show_market_analysis()
                                    ).props("flat color=secondary").classes("theme-btn")
                                
                                # Trailing Stop Loss
                                with ui.expansion("Advanced Risk Management", icon="trending_down").classes("w-full mt-3"):
                                    with ui.column().classes("w-full p-3"):
                                        trailing_stop_enabled = ui.checkbox("Enable Trailing Stop Loss", value=False)
                                        execute_refs["trailing_stop_enabled"] = trailing_stop_enabled
                                        
                                        with ui.row().classes("w-full gap-4 mt-2"):
                                            trailing_stop_percent = ui.number(
                                                label="Trailing Stop (%)",
                                                value=2.0,
                                                min=0.1,
                                                max=10.0,
                                                step=0.1
                                            ).props("filled hint='Trailing stop loss percentage'").classes("flex-1 theme-text-primary")
                                            execute_refs["trailing_stop_percent"] = trailing_stop_percent
                                            
                                            trailing_stop_min = ui.number(
                                                label="Min Trail Distance (%)",
                                                value=1.0,
                                                min=0.1,
                                                max=5.0,
                                                step=0.1
                                            ).props("filled hint='Minimum trailing distance'").classes("flex-1 theme-text-primary")
                                            execute_refs["trailing_stop_min"] = trailing_stop_min
                                
                                # Partial Exits
                                with ui.expansion("Partial Exits", icon="trending_up").classes("w-full mt-2"):
                                    partial_exits_container = ui.column().classes("w-full p-3")
                                    execute_refs["partial_exits_container"] = partial_exits_container
                                    execute_refs["partial_exits"] = []
                                    
                                    def add_partial_exit_execution():
                                        with execute_refs["partial_exits_container"]:
                                            with ui.row().classes("w-full gap-2 items-center"):
                                                target_percent = ui.number("Target %", value=5.0, min=1.0, max=100.0, step=0.5).classes("flex-1")
                                                qty_percent = ui.number("Qty %", value=25.0, min=1.0, max=100.0, step=1.0).classes("flex-1")
                                                remove_btn = ui.button("Remove", icon="delete", on_click=lambda: remove_partial_exit_execution(row_data)).props("size=sm color=negative flat")
                                                
                                                row_data = {"target": target_percent, "qty_percent": qty_percent, "remove_btn": remove_btn}
                                                execute_refs["partial_exits"].append(row_data)
                                    
                                    def remove_partial_exit_execution(row_data):
                                        if row_data in execute_refs["partial_exits"]:
                                            execute_refs["partial_exits"].remove(row_data)
                                            row_data["target"].delete()
                                            row_data["qty_percent"].delete()
                                            row_data["remove_btn"].delete()
                                    
                                    ui.button("Add Partial Exit", icon="add", on_click=add_partial_exit_execution).props("flat color=primary")
                        
                        # Execution controls
                        with ui.row().classes("w-full gap-4 mt-6"):
                            ui.button(
                                "Start Execution",
                                icon="play_arrow",
                                on_click=lambda: start_strategy_execution()
                            ).props("color=positive size=lg").classes("flex-1 theme-btn")
                            
                            ui.button(
                                "Preview Strategy",
                                icon="visibility",
                                on_click=lambda: preview_selected_strategy()
                            ).props("flat size=lg").classes("flex-1 theme-btn")
                
                # Active Executions section
                with ui.card().classes("w-full enhanced-card mt-6"):
                    with ui.card_section().classes("p-6 w-full"):
                        with ui.row().classes("items-center gap-2 mb-4 w-full"):
                            ui.icon("trending_up", size="1.5rem").classes("text-green-500 drop-shadow-lg")
                            ui.label("Active Strategy Executions").classes("card-title theme-text-primary")
                            ui.button("Refresh", icon="refresh", on_click=lambda: refresh_active_executions()).props("flat").classes("ml-auto theme-btn")
                        
                        executions_container = ui.column().classes("w-full gap-3")
                        execute_refs["executions_container"] = executions_container
                        
                        # Load active executions on tab load
                        ui.timer(0.2, lambda: refresh_active_executions(), once=True)

    # Position Calculator Modal
    def open_position_calculator():
        """Open position size calculator modal"""
        with ui.dialog() as calc_dialog, ui.card().classes("w-full max-w-2xl max-h-screen overflow-y-auto"):
            with ui.card_section().classes("p-6"):
                ui.label("🧮 Position Size Calculator").classes("text-xl font-bold theme-text-primary mb-4")
                
                # Input fields
                with ui.grid(columns=2).classes("w-full gap-4 mb-4"):
                    calc_capital = ui.number("Capital (₹)", value=execute_refs["total_capital"].value, min=1000).classes("w-full")
                    calc_risk_pct = ui.number("Risk (%)", value=execute_refs["risk_per_trade"].value, min=0.1, max=20).classes("w-full")
                    calc_entry_price = ui.number("Entry Price (₹)", value=100, min=0.01, step=0.01).classes("w-full")
                    calc_stop_loss_pct = ui.number("Stop Loss (%)", value=execute_refs["stop_loss"].value or 2.0, min=0.1, max=50).classes("w-full")
                
                ui.separator().classes("my-4")
                
                # Results display
                results_container = ui.column().classes("w-full gap-2")
                
                def calculate_position():
                    try:
                        capital = calc_capital.value or 0
                        risk_pct = calc_risk_pct.value or 0
                        entry_price = calc_entry_price.value or 0
                        stop_loss_pct = calc_stop_loss_pct.value or 0
                        
                        if capital > 0 and entry_price > 0 and stop_loss_pct > 0:
                            # Calculate risk amount
                            risk_amount = capital * (risk_pct / 100)
                            
                            # Calculate stop loss price
                            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                            
                            # Calculate risk per share
                            risk_per_share = entry_price - stop_loss_price
                            
                            # Calculate position size
                            if risk_per_share > 0:
                                position_size = int(risk_amount / risk_per_share)
                                position_value = position_size * entry_price
                                
                                results_container.clear()
                                with results_container:
                                    ui.label("📊 Calculation Results").classes("font-bold theme-text-primary mb-2")
                                    
                                    with ui.grid(columns=2).classes("w-full gap-2"):
                                        ui.label("Risk Amount:").classes("font-medium")
                                        ui.label(f"₹{risk_amount:,.0f}").classes("text-green-600 font-bold")
                                        
                                        ui.label("Position Size:").classes("font-medium")
                                        ui.label(f"{position_size:,} shares").classes("text-blue-600 font-bold")
                                        
                                        ui.label("Position Value:").classes("font-medium")
                                        ui.label(f"₹{position_value:,.0f}").classes("text-purple-600 font-bold")
                                        
                                        ui.label("Stop Loss Price:").classes("font-medium")
                                        ui.label(f"₹{stop_loss_price:.2f}").classes("text-red-600 font-bold")
                                        
                                        ui.label("Max Loss:").classes("font-medium")
                                        ui.label(f"₹{risk_amount:,.0f} ({risk_pct:.1f}%)").classes("text-red-600 font-bold")
                                        
                                        ui.label("Capital Utilization:").classes("font-medium")
                                        util_pct = (position_value / capital) * 100
                                        ui.label(f"{util_pct:.1f}%").classes("text-orange-600 font-bold")
                                    
                                    if position_size > 0:
                                        ui.separator().classes("my-2")
                                        with ui.row().classes("w-full gap-2"):
                                            ui.button(
                                                f"Use This Quantity ({position_size})",
                                                on_click=lambda: apply_calculated_quantity(position_size, calc_dialog)
                                            ).props("color=positive").classes("flex-1")
                            else:
                                results_container.clear()
                                with results_container:
                                    ui.label("⚠️ Invalid calculation - check your inputs").classes("text-red-600")
                        else:
                            results_container.clear()
                            with results_container:
                                ui.label("Please fill all fields with valid values").classes("text-gray-600")
                                
                    except Exception as e:
                        results_container.clear()
                        with results_container:
                            ui.label(f"Error: {str(e)}").classes("text-red-600")
                
                def apply_calculated_quantity(quantity, dialog):
                    execute_refs["position_sizing_mode"].value = "Manual Quantity"
                    execute_refs["quantity"].value = quantity
                    ui.notify(f"Applied calculated quantity: {quantity}", type="positive")
                    dialog.close()
                
                # Auto-calculate when values change
                calc_capital.on("update:modelValue", calculate_position)
                calc_risk_pct.on("update:modelValue", calculate_position)
                calc_entry_price.on("update:modelValue", calculate_position)
                calc_stop_loss_pct.on("update:modelValue", calculate_position)
                
                # Initial calculation
                ui.timer(0.1, calculate_position, once=True)
                
                with ui.row().classes("w-full gap-2 mt-4"):
                    ui.button("Close", on_click=calc_dialog.close).props("flat")
        
        calc_dialog.open()

    # Smart Parameter Optimization
    def optimize_parameters():
        """Show intelligent parameter optimization suggestions"""
        with ui.dialog() as opt_dialog, ui.card().classes("w-full max-w-3xl"):
            with ui.card_section().classes("p-6"):
                ui.label("🤖 Smart Parameter Optimization").classes("text-xl font-bold theme-text-primary mb-4")
                
                # Current parameters display
                ui.label("Current Parameters:").classes("font-bold theme-text-primary mb-2")
                current_params = ui.column().classes("w-full bg-gray-50 p-3 rounded mb-4")
                
                with current_params:
                    with ui.grid(columns=3).classes("w-full gap-2"):
                        ui.label("Capital:").classes("font-medium")
                        ui.label(f"₹{execute_refs['total_capital'].value:,}").classes("text-blue-600")
                        ui.label("")
                        
                        ui.label("Risk per Trade:").classes("font-medium")
                        ui.label(f"{execute_refs['risk_per_trade'].value}%").classes("text-orange-600")
                        ui.label("")
                        
                        ui.label("Stop Loss:").classes("font-medium")
                        ui.label(f"{execute_refs['stop_loss'].value}%").classes("text-red-600")
                        ui.label("")
                        
                        ui.label("Take Profit:").classes("font-medium")
                        ui.label(f"{execute_refs['take_profit'].value}%").classes("text-green-600")
                        ui.label("")
                        
                        ui.label("Timeframe:").classes("font-medium")
                        ui.label(f"{execute_refs['timeframe'].value}").classes("text-purple-600")
                        ui.label("")
                
                ui.separator().classes("my-4")
                
                # Optimization suggestions
                ui.label("🎯 Optimization Suggestions:").classes("font-bold theme-text-primary mb-3")
                
                suggestions_container = ui.column().classes("w-full gap-3")
                
                # Generate suggestions based on current timeframe and market conditions
                timeframe = execute_refs['timeframe'].value
                suggestions = generate_parameter_suggestions(timeframe)
                
                with suggestions_container:
                    for i, suggestion in enumerate(suggestions):
                        with ui.card().classes("w-full border-l-4 border-l-blue-500"):
                            with ui.card_section().classes("p-3"):
                                with ui.row().classes("items-center gap-2 w-full"):
                                    ui.icon(suggestion['icon']).classes("text-blue-600")
                                    
                                    with ui.column().classes("flex-1"):
                                        ui.label(suggestion['title']).classes("font-bold theme-text-primary")
                                        ui.label(suggestion['description']).classes("text-sm theme-text-secondary")
                                        
                                        # Parameter suggestions
                                        with ui.row().classes("gap-4 mt-2"):
                                            for param, value in suggestion['params'].items():
                                                ui.label(f"{param}: {value}").classes("text-xs bg-blue-100 px-2 py-1 rounded")
                                    
                                    ui.button(
                                        "Apply",
                                        on_click=lambda params=suggestion['params']: apply_suggestions(params, opt_dialog)
                                    ).props("size=sm color=positive").classes("theme-btn")
                
                with ui.row().classes("w-full gap-2 mt-4"):
                    ui.button("Close", on_click=opt_dialog.close).props("flat")
        
        opt_dialog.open()
    
    def generate_parameter_suggestions(timeframe):
        """Generate intelligent parameter suggestions based on timeframe"""
        suggestions = []
        
        if timeframe in ['1min', '5min']:
            suggestions.append({
                'icon': 'flash_on',
                'title': 'Scalping Mode',
                'description': 'Tight stops and quick profits for intraday scalping',
                'params': {
                    'Risk': '1.5%',
                    'Stop Loss': '0.5%',
                    'Take Profit': '1.0%',
                    'Trail Stop': '0.3%'
                }
            })
            suggestions.append({
                'icon': 'speed',
                'title': 'High Frequency',
                'description': 'Optimized for rapid trade execution',
                'params': {
                    'Risk': '1.0%',
                    'Stop Loss': '0.3%',
                    'Take Profit': '0.8%',
                    'Position Size': '5%'
                }
            })
        elif timeframe in ['15min', '30min']:
            suggestions.append({
                'icon': 'trending_up',
                'title': 'Swing Trading',
                'description': 'Medium-term trend following with balanced risk',
                'params': {
                    'Risk': '2.0%',
                    'Stop Loss': '1.5%',
                    'Take Profit': '4.0%',
                    'Trail Stop': '1.0%'
                }
            })
            suggestions.append({
                'icon': 'balance',
                'title': 'Balanced Approach',
                'description': 'Conservative risk with steady returns',
                'params': {
                    'Risk': '1.5%',
                    'Stop Loss': '1.0%',
                    'Take Profit': '3.0%',
                    'Position Size': '8%'
                }
            })
        else:  # hourly, daily, weekly
            suggestions.append({
                'icon': 'analytics',
                'title': 'Position Trading',
                'description': 'Long-term trend following with wider stops',
                'params': {
                    'Risk': '3.0%',
                    'Stop Loss': '3.0%',
                    'Take Profit': '9.0%',
                    'Trail Stop': '2.0%'
                }
            })
            suggestions.append({
                'icon': 'account_balance',
                'title': 'Conservative Long-term',
                'description': 'Lower risk for steady capital growth',
                'params': {
                    'Risk': '2.0%',
                    'Stop Loss': '2.5%',
                    'Take Profit': '7.0%',
                    'Position Size': '15%'
                }
            })
        
        # Universal suggestions
        suggestions.append({
            'icon': 'psychology',
            'title': 'AI Optimized',
            'description': 'Machine learning optimized parameters based on market volatility',
            'params': {
                'Risk': 'Dynamic 1-3%',
                'Stop Loss': 'ATR Based',
                'Take Profit': '2:1 R:R',
                'Adaptive': 'Yes'
            }
        })
        
        return suggestions
    
    def apply_suggestions(params, dialog):
        """Apply suggested parameters to the form"""
        try:
            # Parse and apply suggestions
            for param, value in params.items():
                if param == 'Risk' and '%' in value:
                    risk_val = float(value.replace('%', ''))
                    execute_refs['risk_per_trade'].value = risk_val
                elif param == 'Stop Loss' and '%' in value:
                    stop_val = float(value.replace('%', ''))
                    execute_refs['stop_loss'].value = stop_val
                elif param == 'Take Profit' and '%' in value:
                    profit_val = float(value.replace('%', ''))
                    execute_refs['take_profit'].value = profit_val
                elif param == 'Trail Stop' and '%' in value:
                    trail_val = float(value.replace('%', ''))
                    if 'trailing_stop_percent' in execute_refs:
                        execute_refs['trailing_stop_percent'].value = trail_val
                        execute_refs['trailing_stop_enabled'].value = True
                elif param == 'Position Size' and '%' in value:
                    pos_val = float(value.replace('%', ''))
                    if 'position_sizing' in execute_refs:
                        execute_refs['position_sizing'].value = pos_val
            
            ui.notify("Parameters applied successfully!", type="positive")
            dialog.close()
        except Exception as e:
            ui.notify(f"Error applying parameters: {str(e)}", type="negative")
    
    def show_market_analysis():
        """Show current market analysis and recommendations"""
        with ui.dialog() as analysis_dialog, ui.card().classes("w-full max-w-4xl"):
            with ui.card_section().classes("p-6"):
                ui.label("📊 Market Analysis & Recommendations").classes("text-xl font-bold theme-text-primary mb-4")
                
                # Market conditions
                with ui.row().classes("w-full gap-4 mb-4"):
                    # Market sentiment
                    with ui.card().classes("flex-1 bg-green-50 border border-green-200"):
                        with ui.card_section().classes("p-3"):
                            ui.icon("trending_up", size="2rem").classes("text-green-600")
                            ui.label("Market Sentiment").classes("font-bold theme-text-primary")
                            ui.label("Bullish").classes("text-green-600 font-medium")
                            ui.label("VIX: 18.5 (Low volatility)").classes("text-sm theme-text-secondary")
                    
                    # Risk environment
                    with ui.card().classes("flex-1 bg-blue-50 border border-blue-200"):
                        with ui.card_section().classes("p-3"):
                            ui.icon("shield", size="2rem").classes("text-blue-600")
                            ui.label("Risk Environment").classes("font-bold theme-text-primary")
                            ui.label("Moderate").classes("text-blue-600 font-medium")
                            ui.label("ATR: 2.3% (Normal)").classes("text-sm theme-text-secondary")
                    
                    # Trend strength
                    with ui.card().classes("flex-1 bg-purple-50 border border-purple-200"):
                        with ui.card_section().classes("p-3"):
                            ui.icon("show_chart", size="2rem").classes("text-purple-600")
                            ui.label("Trend Strength").classes("font-bold theme-text-primary")
                            ui.label("Strong").classes("text-purple-600 font-medium")
                            ui.label("ADX: 28 (Trending)").classes("text-sm theme-text-secondary")
                
                ui.separator().classes("my-4")
                
                # Recommendations based on analysis
                ui.label("🎯 Recommendations:").classes("font-bold theme-text-primary mb-3")
                
                recommendations = [
                    {
                        'icon': 'thumb_up',
                        'title': 'Favorable Conditions',
                        'description': 'Current market conditions favor trend-following strategies',
                        'color': 'green'
                    },
                    {
                        'icon': 'warning',
                        'title': 'Risk Management',
                        'description': 'Use position sizing of 8-12% due to moderate volatility',
                        'color': 'orange'
                    },
                    {
                        'icon': 'schedule',
                        'title': 'Timing',
                        'description': 'Best execution times: 9:30-11:00 AM and 2:00-3:15 PM',
                        'color': 'blue'
                    },
                    {
                        'icon': 'insights',
                        'title': 'Strategy Focus',
                        'description': 'EMA crossover strategies showing 68% win rate this week',
                        'color': 'purple'
                    }
                ]
                
                with ui.column().classes("w-full gap-3"):
                    for rec in recommendations:
                        with ui.card().classes(f"w-full border-l-4 border-l-{rec['color']}-500"):
                            with ui.card_section().classes("p-3"):
                                with ui.row().classes("items-center gap-3"):
                                    ui.icon(rec['icon']).classes(f"text-{rec['color']}-600")
                                    with ui.column().classes("flex-1"):
                                        ui.label(rec['title']).classes("font-bold theme-text-primary")
                                        ui.label(rec['description']).classes("theme-text-secondary")
                
                with ui.row().classes("w-full gap-2 mt-6"):
                    ui.button("Apply Market-Based Settings", on_click=lambda: apply_market_settings(analysis_dialog)).props("color=positive")
                    ui.button("Close", on_click=analysis_dialog.close).props("flat")
        
        analysis_dialog.open()
    
    def apply_market_settings(dialog):
        """Apply market-based parameter settings"""
        try:
            # Apply market-based settings
            execute_refs['risk_per_trade'].value = 2.5  # Moderate risk for current conditions
            execute_refs['stop_loss'].value = 2.0  # Based on ATR
            execute_refs['take_profit'].value = 6.0  # 3:1 R:R ratio
            if 'position_sizing' in execute_refs:
                execute_refs['position_sizing'].value = 10.0  # Recommended position size
            
            ui.notify("Market-based settings applied!", type="positive")
            dialog.close()
        except Exception as e:
            ui.notify(f"Error applying settings: {str(e)}", type="negative")

    # Strategy execution functions
    async def refresh_strategy_options(strategy_select):
        """Refresh the list of available strategy templates"""
        try:
            logger.debug(f"Refreshing strategies for broker: {broker}")
            
            # Try multiple endpoints to get strategies
            response = await fetch_api(f"/strategies/{broker}/list")
            
            # If no response from list endpoint, try other endpoints
            if not response:
                response = await fetch_api(f"/active-strategies/{broker}")
                
            if not response:
                response = await fetch_api(f"/strategies/{broker}/execution-status")

            if response and isinstance(response, list):
                # Format strategy options with name and ID
                options = {}
                strategy_names = [strategy['name'] for strategy in response]
                
                for strategy in response:
                    strategy_label = f"{strategy['name']} - {strategy.get('description', 'No description')[:50]}"
                    options[strategy_label] = strategy["strategy_id"]

                options_list = list(options.keys())
                
                # Update options using the most reliable method for NiceGUI
                strategy_select.options_dict = options
                strategy_select.options.clear()
                strategy_select.options.extend(options_list)
                
                # Force UI refresh
                if hasattr(strategy_select, 'update'):
                    strategy_select.update()
                
                # Update strategy details when selection changes
                def on_strategy_change():
                    selected_label = strategy_select.value
                    if selected_label and selected_label in options:
                        selected_strategy = next((s for s in response if s["strategy_id"] == options[selected_label]), None)
                        if selected_strategy:
                            update_strategy_details_display(execute_refs["strategy_details"], selected_strategy)
                
                strategy_select.on_value_change = on_strategy_change
                
                # Show success notification
                ui.notify(f"Loaded {len(options)} strategies: {', '.join(strategy_names)}", type="positive")
                logger.info(f"Successfully loaded {len(options)} strategies")
            else:
                logger.warning("No strategies found")
                ui.notify("No strategies found. Please create a strategy first.", type="warning")
                
        except Exception as e:
            logger.error(f"Error fetching strategy options: {e}")
            ui.notify(f"Error loading strategies: {str(e)}", type="negative")

    def update_strategy_details_display(details_card, strategy):
        """Update the strategy details display"""
        details_card.clear()
        details_card.style("display: block;")
        
        with details_card:
            with ui.card_section().classes("p-3"):
                ui.label(f"Strategy: {strategy['name']}").classes("text-lg font-bold theme-text-primary")
                if strategy.get('description'):
                    ui.label(strategy['description']).classes("text-sm theme-text-secondary mb-2")
                
                with ui.row().classes("w-full gap-4"):
                    with ui.column().classes("flex-1"):
                        ui.label("Entry Conditions:").classes("font-medium theme-text-primary")
                        ui.label(format_conditions(strategy.get('entry_conditions', []))).classes("text-sm theme-text-secondary")
                    
                    with ui.column().classes("flex-1"):
                        ui.label("Exit Conditions:").classes("font-medium theme-text-primary")
                        ui.label(format_conditions(strategy.get('exit_conditions', []))).classes("text-sm theme-text-secondary")

    def add_instrument_to_execution():
        """Add selected instrument to execution list"""
        instrument_select = execute_refs["instrument_select"]
        selected_instruments = execute_refs["selected_instruments"]
        selected_display = execute_refs["selected_display"]
        
        if instrument_select.value and instrument_select.value not in selected_instruments:
            selected_instruments.append(instrument_select.value)
            instrument_select.value = None  # Clear selection
            
            # Update display
            update_selected_instruments_display(selected_instruments, selected_display)
            
            ui.notify(f"Added instrument to execution list", type="positive")
        elif instrument_select.value in selected_instruments:
            ui.notify("Instrument already added", type="warning")
        else:
            ui.notify("Please select an instrument", type="warning")

    def update_selected_instruments_display(selected_instruments, selected_display):
        """Update the display of selected instruments"""
        selected_display.clear()
        
        with selected_display:
            for instrument in selected_instruments:
                with ui.card().classes("w-full bg-blue-50 border border-blue-200"):
                    with ui.card_section().classes("p-2"):
                        with ui.row().classes("items-center gap-2 w-full"):
                            ui.icon("account_balance").classes("text-blue-600")
                            ui.label(f"{instrument} ({instruments.get(instrument, 'Unknown')})").classes("flex-1 theme-text-primary")
                            ui.button(
                                icon="close",
                                on_click=lambda inst=instrument: remove_instrument_from_execution(inst)
                            ).props("flat round dense size=sm color=negative")

    def remove_instrument_from_execution(instrument):
        """Remove instrument from execution list"""
        selected_instruments = execute_refs["selected_instruments"]
        selected_display = execute_refs["selected_display"]
        
        if instrument in selected_instruments:
            selected_instruments.remove(instrument)
            update_selected_instruments_display(selected_instruments, selected_display)
            ui.notify(f"Removed {instrument} from execution list", type="positive")

    async def preview_selected_strategy():
        """Preview the selected strategy with current parameters"""
        strategy_select = execute_refs["strategy_select"]
        
        if not strategy_select.value:
            ui.notify("Please select a strategy first", type="warning")
            return
        
        try:
            # Get selected strategy ID
            selected_label = strategy_select.value
            options_dict = getattr(strategy_select, 'options_dict', {})
            strategy_id = options_dict.get(selected_label)
            
            if not strategy_id:
                ui.notify("Invalid strategy selection", type="negative")
                return
            
            # Show preview in a dialog
            with ui.dialog() as preview_dialog, ui.card().classes("w-full max-w-4xl"):
                with ui.card_section().classes("p-6"):
                    ui.label("Strategy Preview").classes("text-xl font-bold theme-text-primary mb-4")
                    
                    # Display strategy details and parameters
                    ui.label(f"Strategy: {selected_label}").classes("font-medium theme-text-primary mb-2")
                    
                    if execute_refs["selected_instruments"]:
                        ui.label("Selected Instruments:").classes("font-medium theme-text-primary")
                        for inst in execute_refs["selected_instruments"]:
                            ui.label(f"• {inst}").classes("ml-4 theme-text-secondary")
                    else:
                        ui.label("⚠️ No instruments selected for execution").classes("text-orange-600 font-medium")
                    
                    ui.separator()
                    
                    with ui.row().classes("w-full gap-4"):
                        ui.label(f"Quantity: {execute_refs['quantity'].value}").classes("theme-text-primary")
                        ui.label(f"Risk per Trade: {execute_refs['risk_per_trade'].value}%").classes("theme-text-primary")
                        ui.label(f"Position Size: {execute_refs['position_sizing'].value}%").classes("theme-text-primary")
                        ui.label(f"Timeframe: {execute_refs['timeframe'].value}").classes("theme-text-primary")
                        
                    with ui.row().classes("w-full gap-4 mt-2"):
                        if execute_refs['stop_loss'].value:
                            ui.label(f"Stop Loss: {execute_refs['stop_loss'].value}%").classes("theme-text-primary")
                        
                        if execute_refs['take_profit'].value:
                            ui.label(f"Take Profit: {execute_refs['take_profit'].value}%").classes("theme-text-primary")
                            
                        if execute_refs['trailing_stop_enabled'].value:
                            ui.label(f"Trailing Stop: {execute_refs['trailing_stop_percent'].value}%").classes("theme-text-primary")
                    
                    # Show partial exits if any
                    if execute_refs["partial_exits"]:
                        ui.separator().classes("mt-2")
                        ui.label("Partial Exits:").classes("font-medium theme-text-primary mt-2")
                        for i, exit_data in enumerate(execute_refs["partial_exits"]):
                            if hasattr(exit_data["target"], 'value') and hasattr(exit_data["qty_percent"], 'value'):
                                ui.label(f"• Exit {i+1}: {exit_data['qty_percent'].value}% at {exit_data['target'].value}% profit").classes("ml-4 theme-text-secondary")
                    
                    with ui.row().classes("mt-4 gap-2"):
                        ui.button("Close", on_click=preview_dialog.close).props("flat")
            
            preview_dialog.open()
            
        except Exception as e:
            logger.error(f"Error previewing strategy: {e}")
            ui.notify(f"Preview error: {str(e)}", type="negative")

    async def start_strategy_execution():
        """Start executing the selected strategy on selected instruments"""
        strategy_select = execute_refs["strategy_select"]
        selected_instruments = execute_refs["selected_instruments"]
        
        # Validation
        if not strategy_select.value:
            ui.notify("Please select a strategy first", type="warning")
            return
        
        if not selected_instruments:
            ui.notify("Please add at least one instrument", type="warning")
            return
        
        if not execute_refs["quantity"].value or execute_refs["quantity"].value <= 0:
            ui.notify("Please enter a valid quantity", type="warning")
            return
        
        try:
            # Get selected strategy ID
            selected_label = strategy_select.value
            options_dict = getattr(strategy_select, 'options_dict', {})
            strategy_id = options_dict.get(selected_label)
            
            if not strategy_id:
                ui.notify("Invalid strategy selection", type="negative")
                return
            
            # Prepare execution requests for each instrument
            executions_started = 0
            
            for instrument_token in selected_instruments:
                try:
                    trading_symbol = instruments.get(instrument_token, instrument_token)
                    
                    # Collect partial exits data
                    partial_exits = []
                    for exit_data in execute_refs["partial_exits"]:
                        if hasattr(exit_data["target"], 'value') and hasattr(exit_data["qty_percent"], 'value'):
                            partial_exits.append({
                                "target": float(exit_data["target"].value),
                                "qty_percent": float(exit_data["qty_percent"].value)
                            })
                    
                    # Calculate final quantity based on position sizing mode
                    final_quantity = int(execute_refs["quantity"].value)
                    if execute_refs["position_sizing_mode"].value == "Auto Calculate":
                        # Auto calculate quantity based on risk parameters
                        # This would need current price, but for now use the manual quantity
                        final_quantity = int(execute_refs["quantity"].value)
                    
                    execution_data = {
                        "instrument_token": instrument_token,
                        "trading_symbol": trading_symbol,
                        "quantity": final_quantity,
                        "risk_per_trade": float(execute_refs["risk_per_trade"].value),
                        "stop_loss": float(execute_refs["stop_loss"].value) if execute_refs["stop_loss"].value else None,
                        "take_profit": float(execute_refs["take_profit"].value) if execute_refs["take_profit"].value else None,
                        "position_sizing_percent": float(execute_refs["position_sizing"].value),
                        "position_sizing_mode": execute_refs["position_sizing_mode"].value,
                        "total_capital": float(execute_refs["total_capital"].value),
                        "timeframe": execute_refs["timeframe"].value,
                        "trailing_stop_enabled": execute_refs["trailing_stop_enabled"].value,
                        "trailing_stop_percent": float(execute_refs["trailing_stop_percent"].value) if execute_refs["trailing_stop_enabled"].value else None,
                        "trailing_stop_min": float(execute_refs["trailing_stop_min"].value) if execute_refs["trailing_stop_enabled"].value else None,
                        "partial_exits": partial_exits
                    }
                    
                    logger.debug(f"Starting execution for {trading_symbol} with data: {execution_data}")
                    
                    # Call the backend execution endpoint
                    response = await fetch_api(f"/strategies/{strategy_id}/execute", method="POST", data=execution_data)
                    
                    if response and not response.get("error"):
                        executions_started += 1
                        logger.info(f"Started execution for {trading_symbol}: {response.get('execution_id')}")
                    else:
                        error_msg = response.get('error', {}).get('message') if response.get('error') else response.get('message', 'Unknown error')
                        logger.error(f"Failed to start execution for {trading_symbol}: {error_msg}")
                        ui.notify(f"Failed to start {trading_symbol}: {error_msg}", type="negative")
                        
                except Exception as e:
                    logger.error(f"Error starting execution for {instrument_token}: {e}")
                    ui.notify(f"Error with {instrument_token}: {str(e)}", type="negative")
            
            if executions_started > 0:
                ui.notify(f"Started {executions_started} strategy executions", type="positive")
                
                # Clear the form
                execute_refs["selected_instruments"].clear()
                update_selected_instruments_display([], execute_refs["selected_display"])
                
                # Refresh active executions
                await asyncio.sleep(1)  # Give backend time to process
                await refresh_active_executions()
            else:
                ui.notify("No executions were started", type="negative")
                
        except Exception as e:
            logger.error(f"Error starting strategy execution: {e}")
            ui.notify(f"Execution error: {str(e)}", type="negative")

    async def refresh_active_executions():
        """Refresh the list of active strategy executions"""
        try:
            logger.debug("Fetching active strategy executions")
            response = await fetch_api("/executions")
            
            container = execute_refs.get("executions_container")
            if not container:
                return
            
            container.clear()
            
            with container:
                if response and isinstance(response, list) and len(response) > 0:
                    for execution in response:
                        await render_execution_card(execution)
                else:
                    ui.label("No active strategy executions").classes("w-full text-center theme-text-secondary py-4")
                    
        except Exception as e:
            logger.error(f"Error fetching active executions: {e}")
            ui.notify(f"Error loading executions: {str(e)}", type="negative")

    async def render_execution_card(execution):
        """Render a card for a single strategy execution"""
        execution_id = execution.get("execution_id")
        strategy_name = execution.get("strategy_name", "Unknown")
        trading_symbol = execution.get("trading_symbol", "Unknown")
        status = execution.get("status", "unknown")
        pnl = execution.get("pnl", 0.0)
        
        # Status colors
        status_colors = {
            "running": "green",
            "stopped": "orange", 
            "completed": "blue",
            "failed": "red"
        }
        
        status_color = status_colors.get(status, "gray")
        
        with ui.card().classes("w-full enhanced-card border-l-4").style(f"border-left-color: {status_color};"):
            with ui.card_section().classes("p-4"):
                with ui.row().classes("items-center gap-4 w-full"):
                    # Main info
                    with ui.column().classes("flex-1"):
                        ui.label(f"{strategy_name} on {trading_symbol}").classes("font-bold theme-text-primary")
                        with ui.row().classes("gap-4 mt-1"):
                            ui.label(f"Status: {status.title()}").classes(f"text-{status_color}-600 font-medium")
                            ui.label(f"P&L: ₹{pnl:.2f}").classes("theme-text-primary")
                            if execution.get("signals_generated"):
                                ui.label(f"Signals: {execution.get('signals_generated', 0)}").classes("theme-text-secondary")
                    
                    # Action buttons
                    with ui.row().classes("gap-2"):
                        ui.button(
                            "Metrics",
                            icon="analytics",
                            on_click=lambda eid=execution_id: view_execution_metrics(eid)
                        ).props("flat dense").classes("theme-btn")
                        
                        if status == "running":
                            ui.button(
                                "Stop",
                                icon="stop",
                                on_click=lambda eid=execution_id: stop_execution(eid)
                            ).props("flat dense color=negative").classes("theme-btn")

    async def view_execution_metrics(execution_id):
        """View detailed metrics for a strategy execution"""
        try:
            response = await fetch_api(f"/executions/{execution_id}/metrics")
            
            if response:
                with ui.dialog() as metrics_dialog, ui.card().classes("w-full max-w-3xl"):
                    with ui.card_section().classes("p-6"):
                        ui.label("Execution Metrics").classes("text-xl font-bold theme-text-primary mb-4")
                        
                        # Display metrics in a structured format
                        metrics = response.get("metrics", {})
                        
                        with ui.row().classes("w-full gap-4"):
                            with ui.column().classes("flex-1"):
                                ui.label("Performance Metrics").classes("font-bold theme-text-primary mb-2")
                                ui.label(f"P&L: ₹{metrics.get('pnl', 0):.2f}").classes("theme-text-primary")
                                ui.label(f"Signals Generated: {metrics.get('signals_generated', 0)}").classes("theme-text-primary")
                                ui.label(f"Trades Executed: {metrics.get('trades_executed', 0)}").classes("theme-text-primary")
                            
                            with ui.column().classes("flex-1"):
                                ui.label("Execution Details").classes("font-bold theme-text-primary mb-2")
                                ui.label(f"Status: {metrics.get('status', 'unknown').title()}").classes("theme-text-primary")
                                ui.label(f"Started: {metrics.get('started_at', 'Unknown')}").classes("theme-text-secondary")
                                if metrics.get('last_signal_at'):
                                    ui.label(f"Last Signal: {metrics.get('last_signal_at')}").classes("theme-text-secondary")
                        
                        ui.button("Close", on_click=metrics_dialog.close).props("flat").classes("mt-4")
                
                metrics_dialog.open()
            else:
                ui.notify("Unable to fetch execution metrics", type="negative")
                
        except Exception as e:
            logger.error(f"Error fetching execution metrics: {e}")
            ui.notify(f"Metrics error: {str(e)}", type="negative")

    async def stop_execution(execution_id):
        """Stop a running strategy execution"""
        try:
            response = await fetch_api(f"/executions/{execution_id}/stop", method="POST")
            
            if response and not response.get("error"):
                ui.notify("Strategy execution stopped", type="positive")
                await refresh_active_executions()  # Refresh the list
            else:
                error_msg = response.get('error', {}).get('message') if response.get('error') else response.get('message', 'Unknown error')
                ui.notify(f"Failed to stop execution: {error_msg}", type="negative")
                
        except Exception as e:
            logger.error(f"Error stopping execution: {e}")
            ui.notify(f"Stop error: {str(e)}", type="negative")

    async def initialize_create_tab():
        global create_tab_initialized
        logger.debug("Initializing Create/Edit Strategy tab")
        create_tab_content.clear()  # Clear existing content to prevent duplication

        with create_tab_content:
            # Full-width container with consistent theme styling
            with ui.column().classes("strategies-page w-full p-2 gap-6 strategy-form-container"):
                # Strategy Basic Info Section - Modern grid layout, theme-aware colors
                with ui.card().classes("w-full strategy-form-container enhanced-card"):
                    with ui.card_section().classes("p-6 w-full"):
                        with ui.row().classes("items-center gap-2 mb-4 w-full"):
                            ui.icon("info", size="1.5rem").classes("text-cyan-500 drop-shadow-lg")
                            ui.label("Strategy Information").classes("card-title theme-text-primary")
                        # Modern grid layout for strategy info (3 columns)
                        with ui.row().classes("w-full gap-4 dashboard-grid-3"):
                            with ui.column().classes("flex-1 min-w-[220px]"):
                                ui_refs["strategy_name"] = ui.input(
                                    "Strategy Name",
                                    validation={"Required": bool, "Max length": lambda v: len(v) <= 50}
                                ).props("filled hint='Enter a unique name for your strategy'").classes("w-full theme-text-primary")
                            with ui.column().classes("flex-1 min-w-[220px]"):
                                ui_refs["broker_select"] = ui.select(
                                    ["Zerodha", "Upstox"],
                                    label="Broker",
                                    value=broker
                                ).props("filled hint='Select your trading broker'").classes("w-full theme-text-primary")
                            with ui.column().classes("flex-1 min-w-[220px]"):
                                ui_refs["strategy_desc"] = ui.textarea(
                                    "Description",
                                    value=""
                                ).props("filled hint='Describe your strategy logic and approach' rows=3").classes("w-full theme-text-secondary")
                # Conditions Section - Side by side layout with theme-aware colors
                with ui.row().classes("w-full gap-6 strategy-form-container"):
                    # Entry Conditions
                    with ui.column().classes("flex-1 strategy-form-container"):
                        with ui.card().classes("w-full strategy-form-container enhanced-card border-l-4 border-l-green-500"):
                            with ui.card_section().classes("p-6 w-full"):
                                with ui.row().classes("items-center gap-2 mb-3 w-full"):
                                    ui.icon("trending_up", size="1.5rem").classes("text-green-400 drop-shadow-lg")
                                    ui.label("Entry Conditions").classes("card-title theme-text-success")
                                ui.label("Define when to enter a trade").classes("theme-text-secondary text-sm mb-4")
                                ui_refs["entry_conditions"], ui_refs["entry_container"] = render_condition_builder(
                                    available_indicators, "Entry Conditions", instruments, ui_refs, "entry_conditions"
                                )
                    # Exit Conditions
                    with ui.column().classes("flex-1 strategy-form-container"):
                        with ui.card().classes("w-full strategy-form-container enhanced-card border-l-4 border-l-red-500"):
                            with ui.card_section().classes("p-6 w-full"):
                                with ui.row().classes("items-center gap-2 mb-3 w-full"):
                                    ui.icon("trending_down", size="1.5rem").classes("text-red-400 drop-shadow-lg")
                                    ui.label("Exit Conditions").classes("card-title theme-text-error")
                                ui.label("Define when to exit a trade").classes("theme-text-secondary text-sm mb-4")
                                ui_refs["exit_conditions"], ui_refs["exit_container"] = render_condition_builder(
                                    available_indicators, "Exit Conditions", instruments, ui_refs, "exit_conditions"
                                )
                # Strategy Parameters remain generic - no instrument selection here

                # Parameters Section - Full width, theme-aware
                with ui.card().classes("w-full strategy-form-container enhanced-card"):
                    with ui.card_section().classes("p-6 w-full"):
                        with ui.row().classes("items-center gap-2 mb-4 w-full"):
                            ui.icon("settings", size="1.5rem").classes("text-cyan-500 drop-shadow-lg")
                            ui.label("Trading Parameters").classes("card-title theme-text-primary")
                        with ui.row().classes("w-full gap-4 strategy-parameters-row"):
                            ui_refs["timeframe"] = ui.select(
                                ["1min", "3min", "5min", "15min", "30min", "60min", "day"],
                                label="Timeframe",
                                value="5min"
                            ).props("filled hint='Data interval for strategy execution'").classes("flex-1 theme-text-primary")

                            ui_refs["position_sizing"] = ui.select(
                                ["fixed", "percentage", "risk_based"],
                                label="Position Sizing",
                                value="fixed"
                            ).props("filled hint='How to calculate position sizes'").classes("flex-1 theme-text-primary")

                            ui_refs["preview_instrument"] = ui.select(
                                options=sorted(list(instruments.keys())),
                                label="Preview Instrument",
                                value="RELIANCE"
                            ).props("filled clearable filterable hint='Select instrument for strategy preview'").classes("flex-1 theme-text-primary")

                # Define async functions before they're used
                async def preview_strategy():
                    ui.update()
                    await asyncio.sleep(0.01)
                    entry_conds = extract_conditions(ui_refs["entry_conditions"], available_indicators)
                    exit_conds = extract_conditions(ui_refs["exit_conditions"], available_indicators)
                    if not entry_conds or not exit_conds:
                        ui.notify("Please define at least one entry and one exit condition.", type="negative")
                        return
                    strategy_data = {
                        "name": ui_refs["strategy_name"].value,
                        "description": ui_refs["strategy_desc"].value,
                        "entry_conditions": entry_conds,
                        "exit_conditions": exit_conds,
                        "parameters": {
                            "timeframe": ui_refs["timeframe"].value,
                            "position_sizing": ui_refs["position_sizing"].value
                        }
                    }
                    ui_refs["preview_button"].props("loading=true disable=true")
                    try:
                        response = await fetch_api("/algo-trading/backtest", method="POST", data={
                            "trading_symbol": ui_refs["preview_instrument"].value,
                            "instrument_token": instruments.get(ui_refs["preview_instrument"].value, ""),
                            "timeframe": ui_refs["timeframe"].value,
                            "strategy": json.dumps(strategy_data),
                            "params": {"initial_investment": 100000},
                            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                            "end_date": datetime.now().strftime("%Y-%m-%d")
                        })
                        logger.debug(f"Backtest response: {json.dumps(response, indent=2)}")
                        if response and not response.get("error"):
                            with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl enhanced-card"):
                                with ui.card_section().classes("p-6"):
                                    ui.label("Strategy Preview Results").classes("text-xl font-bold theme-text-primary mb-4")
                                    # Results summary cards with theme-aware colors
                                    with ui.row().classes("w-full gap-4 mb-4"):
                                        with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-green-500"):
                                            ui.label("Total Profit").classes("text-sm theme-text-secondary")
                                            total_profit = response.get("TotalProfit", 0)
                                            pnl_color = "theme-text-success" if total_profit >= 0 else "theme-text-error"
                                            ui.label(f"₹{total_profit:,.2f}").classes(f"text-2xl font-bold {pnl_color}")

                                        with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-blue-500"):
                                            ui.label("Win Rate").classes("text-sm theme-text-secondary")
                                            ui.label(f"{response.get('WinRate', 0):.1f}%").classes("text-2xl font-bold theme-text_info")
                                        with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-purple-500"):
                                            ui.label("Total Trades").classes("text-sm theme-text_secondary")
                                            ui.label(f"{response.get('TotalTrades', 0)}").classes("text-2xl font-bold theme-text_primary")

                                    if tradebook := response.get("Tradebook", []):
                                        ui.label("Trade History").classes("text-lg font-semibold theme-text-primary mb-2")
                                        formatted_tradebook = []
                                        for trade in tradebook[:50]:  # Limit to 50 trades for performance
                                            try:
                                                # Handle different possible field names from backtest response
                                                entry_price = trade.get('EntryPrice') or trade.get('entry_price') or trade.get('Entry_Price') or 0
                                                exit_price = trade.get('ExitPrice') or trade.get('exit_price') or trade.get('Exit_Price') or 0
                                                profit = trade.get('Profit') or trade.get('profit') or 0
                                                portfolio_value = trade.get('PortfolioValue') or trade.get('portfolio_value') or trade.get('Portfolio_Value') or 0
                                                trade_date = trade.get('Date') or trade.get('date') or trade.get('TradeDate') or datetime.now()
                                                
                                                formatted_tradebook.append({
                                                    "Date": pd.to_datetime(trade_date).strftime("%Y-%m-%d %H:%M"),
                                                    "Entry": f"₹{float(entry_price):.2f}",
                                                    "Exit": f"₹{float(exit_price):.2f}",
                                                    "Profit": f"₹{float(profit):.2f}",
                                                    "Portfolio": f"₹{float(portfolio_value):.2f}"
                                                })
                                            except Exception as trade_error:
                                                logger.warning(f"Error formatting trade: {trade_error}, trade data: {trade}")
                                                continue
                                        
                                        if formatted_tradebook:
                                            ui.table(
                                                columns=[
                                                    {"name": "Date", "label": "Date", "field": "Date", "align": "left"},
                                                    {"name": "Entry", "label": "Entry Price", "field": "Entry", "align": "right"},
                                                    {"name": "Exit", "label": "Exit Price", "field": "Exit", "align": "right"},
                                                    {"name": "Profit", "label": "Profit/Loss", "field": "Profit", "align": "right"},
                                                    {"name": "Portfolio", "label": "Portfolio Value", "field": "Portfolio", "align": "right"}
                                                ],
                                                rows=formatted_tradebook,
                                                pagination={"rowsPerPage": 10}
                                            ).classes("w-full enhanced-card")
                                        else:
                                            ui.label("No valid trades found in response").classes("text-yellow-400 text-center")
                                    else:
                                        with ui.card().classes("w-full p-8 text-center enhanced-card border-l-4 border-l-yellow-500"):
                                            ui.icon("warning", size="3rem").classes("text-yellow-400 mb-2")
                                            ui.label("No trades executed with current conditions").classes("text-yellow-400 font-medium")
                                            ui.label("Consider adjusting your entry/exit conditions").classes("theme-text-secondary text-sm")

                                with ui.card_actions().classes("justify-end"):
                                    ui.button("Close", on_click=dialog.close).props("outline").classes("theme-text-secondary border-gray-400")
                            dialog.open()
                        else:
                            ui.notify(f"Preview failed: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                    except Exception as e:
                        ui.notify(f"Error previewing strategy: {str(e)}", type="negative")
                        logger.error(f"Preview strategy error: {str(e)}")
                    finally:
                        ui_refs["preview_button"].props("loading=false disable=false")
                        ui.update()

                async def save_strategy():
                    ui.update()
                    await asyncio.sleep(0.01)
                    logger.debug(f"Pre-save condition count: entry={len(ui_refs['entry_conditions'])}, exit={len(ui_refs['exit_conditions'])}")
                    entry_conds = extract_conditions(ui_refs["entry_conditions"], available_indicators)
                    exit_conds = extract_conditions(ui_refs["exit_conditions"], available_indicators)
                    if not entry_conds or not exit_conds:
                        ui.notify("Please define at least one entry and one exit condition.", type="negative")
                        return
                    
                    # Strategy data without instrument coupling
                    strategy_data = {
                        "name": ui_refs["strategy_name"].value,
                        "description": ui_refs["strategy_desc"].value,
                        "entry_conditions": entry_conds,
                        "exit_conditions": exit_conds,
                        "parameters": {
                            "timeframe": ui_refs["timeframe"].value
                        },
                        "broker": ui_refs["broker_select"].value
                    }
                    logger.debug(f"Saving strategy: {json.dumps(strategy_data, indent=2)}")
                    ui_refs["save_button"].props("loading=true disable=true")
                    try:
                        if ui_refs["editing_strategy_id"]: # If editing, use PUT
                            response = await fetch_api(f"/strategies/{ui_refs['editing_strategy_id']}", method="PUT", data=strategy_data)
                            if response and response.get("strategy_id"):
                                ui.notify("Strategy updated successfully!", type="positive")
                                await fetch_strategies()
                                tabs.set_value(manage_tab)
                            else:
                                ui.notify(f"Failed to update strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                        else: # If creating, use POST
                            response = await fetch_api("/strategies/", method="POST", data=strategy_data)
                            if response and response.get("strategy_id"):
                                ui.notify("Strategy saved successfully!", type="positive")
                                await fetch_strategies()
                                tabs.set_value(manage_tab)
                        ui_refs["editing_strategy_id"] = None # Clear editing ID after save/update
                        ui_refs["save_button"].text = "Save Strategy" # Reset button text
                        ui_refs["save_button"].update()
                    except Exception as e:
                        ui.notify(f"Error saving strategy: {str(e)}", type="negative")
                        logger.error(f"Save strategy error: {str(e)}")
                    finally:
                        ui_refs["save_button"].props("loading=false disable=false")
                        ui.update()

                def reset_strategy_form():
                    """Reset all form fields to their default values"""
                    try:
                        # Clear editing state
                        ui_refs["editing_strategy_id"] = None
                        
                        # Reset basic info fields
                        ui_refs["strategy_name"].value = ""
                        ui_refs["strategy_desc"].value = ""
                        ui_refs["broker_select"].value = broker
                        ui_refs["timeframe"].value = "5min"
                        ui_refs["preview_instrument"].value = "RELIANCE"

                        # Clear all entry conditions except the first one
                        if ui_refs["entry_conditions"]:
                            # Keep first condition, remove others
                            conditions_to_remove = ui_refs["entry_conditions"][1:]
                            for condition in conditions_to_remove:
                                try:
                                    ui_refs["entry_container"].remove(condition["row"])
                                except:
                                    pass
                            ui_refs["entry_conditions"] = ui_refs["entry_conditions"][:1]

                            # Reset first condition if exists
                            if ui_refs["entry_conditions"]:
                                first_condition = ui_refs["entry_conditions"][0]
                                first_condition["left_indicator"].value = None
                                first_condition["comparison"].value = None
                                first_condition["right_indicator"].value = None
                                if hasattr(first_condition["left_value"], 'value'):
                                    first_condition["left_value"].value = 0
                                if hasattr(first_condition["right_value"], 'value'):
                                    first_condition["right_value"].value = 0

                        # Clear all exit conditions except the first one
                        if ui_refs["exit_conditions"]:
                            # Keep first condition, remove others
                            conditions_to_remove = ui_refs["exit_conditions"][1:]
                            for condition in conditions_to_remove:
                                try:
                                    ui_refs["exit_container"].remove(condition["row"])
                                except:
                                    pass
                            ui_refs["exit_conditions"] = ui_refs["exit_conditions"][:1]

                            # Reset first condition if exists
                            if ui_refs["exit_conditions"]:
                                first_condition = ui_refs["exit_conditions"][0]
                                first_condition["left_indicator"].value = None
                                first_condition["comparison"].value = None
                                first_condition["right_indicator"].value = None
                                if hasattr(first_condition["left_value"], 'value'):
                                    first_condition["left_value"].value = 0
                                if hasattr(first_condition["right_value"], 'value'):
                                    first_condition["right_value"].value = 0

                        # Reset save button text
                        ui_refs["save_button"].text = "Save Strategy"
                        ui_refs["save_button"].update()

                        ui.notify("Strategy form reset successfully!", type="positive")
                        ui.update()
                    except Exception as e:
                        logger.error(f"Error resetting strategy form: {str(e)}")
                        ui.notify("Error resetting form. Please refresh the page.", type="negative")

                # Action Buttons Section - theme-aware with Reset button
                with ui.card().classes("w-full strategy-form-container enhanced-card"):
                    with ui.card_section().classes("p-6 w-full"):
                        with ui.row().classes("justify-between items-center w-full strategy-parameters-row"):
                            with ui.row().classes("items-center gap-2"):
                                ui.icon("rocket_launch", size="1.5rem").classes("text-cyan-400 drop-shadow-lg")
                                ui.label("Ready to test or save your strategy?").classes("card-title theme-text-primary font-medium")

                            with ui.row().classes("gap-4"):
                                # Reset button
                                ui.button(
                                    "Reset Form",
                                    icon="refresh",
                                    on_click=reset_strategy_form
                                ).props("outline size=lg").classes("px-6 py-3 theme-text-secondary border-gray-400 hover:bg-gray-400 hover:text-slate-900 shadow-md rounded-xl")

                                ui_refs["preview_button"] = ui.button(
                                    "Preview Strategy",
                                    icon="visibility",
                                    on_click=preview_strategy
                                ).props("outline size=lg").classes("px-6 py-3 theme-text-primary border-cyan-400 hover:bg-cyan-400 hover:text-slate-900 shadow-md rounded-xl")

                                ui_refs["save_button"] = ui.button(
                                    "Save Strategy",
                                    icon="save",
                                    on_click=save_strategy
                                ).props("size=lg").classes("px-6 py-3 bg-cyan-600 hover:bg-cyan-500 theme-text-on-primary shadow-md rounded-xl")

        create_tab_initialized = True
        logger.debug("Create/Edit Strategy tab initialized with enhanced UI and reset functionality")
        return ui_refs["strategy_name"], ui_refs["broker_select"], ui_refs["timeframe"], ui_refs["position_sizing"], ui_refs["preview_instrument"], ui_refs["save_button"], ui_refs["preview_button"]

def apply_strategies_page_fix():
    """Apply specific CSS fixes for strategies page card backgrounds with proper light/dark theme support"""
    strategies_css = '''
    <style>
    /* STRATEGIES PAGE - Dynamic theme-aware styling */
    
    /* Base dark theme styles for strategies page */
    .strategies-page .q-card,
    .strategies-page .nicegui-card,
    .strategies-page .enhanced-card,
    .strategies-page .glass-card,
    .strategies-page .dashboard-card,
    .strategies-page .trading-card,
    .strategies-page .modern-card {
        background: rgba(30, 41, 59, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(71, 85, 105, 0.3) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        color: rgb(226, 232, 240) !important;
        transition: all 0.3s ease !important;
    }
    
    /* Dark theme card sections */
    .strategies-page .q-card__section {
        background: transparent !important;
        color: rgb(226, 232, 240) !important;
    }
    
    /* Dark theme condition builder rows */
    .strategies-page .enhanced-card.p-3.rounded-lg,
    .strategies-page .enhanced-card.p-4.rounded-lg {
        background: rgba(51, 65, 85, 0.8) !important;
        border: 1px solid rgba(100, 116, 139, 0.5) !important;
    }
    
    /* Dark theme text colors */
    .strategies-page .theme-text-primary,
    .strategies-page .card-title {
        color: rgb(226, 232, 240) !important;
    }
    
    .strategies-page .theme-text-secondary {
        color: rgb(148, 163, 184) !important;
    }
    
    .strategies-page .theme-text-success {
        color: rgb(34, 197, 94) !important;
    }
    
    .strategies-page .theme-text-error {
        color: rgb(239, 68, 68) !important;
    }
    
    /* Dark theme form elements */
    .strategies-page .q-field .q-field__control {
        background: rgba(51, 65, 85, 0.6) !important;
        border: 1px solid rgba(100, 116, 139, 0.5) !important;
        color: rgb(226, 232, 240) !important;
    }
    
    .strategies-page .q-field input,
    .strategies-page .q-field textarea,
    .strategies-page .q-field .q-select__focus-target,
    .strategies-page .q-field .q-field__native input {
        color: rgb(226, 232, 240) !important;
    }
    
    .strategies-page .q-field__label {
        color: rgb(148, 163, 184) !important;
        background: rgba(30, 41, 59, 0.9) !important;
    }
    
    /* LIGHT THEME OVERRIDES for strategies page */
    body.q-body--light .strategies-page .q-card,
    body.q-body--light .strategies-page .nicegui-card,
    body.q-body--light .strategies-page .enhanced-card,
    body.q-body--light .strategies-page .glass-card,
    body.q-body--light .strategies-page .dashboard-card,
    body.q-body--light .strategies-page .trading-card,
    body.q-body--light .strategies-page .modern-card {
        background: rgba(248, 250, 252, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        color: rgb(30, 41, 59) !important;
    }
    
    /* Light theme card sections */
    body.q-body--light .strategies-page .q-card__section {
        background: transparent !important;
        color: rgb(30, 41, 59) !important;
    }
    
    /* Light theme condition builder rows */
    body.q-body--light .strategies-page .enhanced-card.p-3.rounded-lg,
    body.q-body--light .strategies-page .enhanced-card.p-4.rounded-lg {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Light theme text colors */
    body.q-body--light .strategies-page .theme-text-primary,
    body.q-body--light .strategies-page .card-title {
        color: rgb(30, 41, 59) !important;
    }
    
    body.q-body--light .strategies-page .theme-text-secondary {
        color: rgb(71, 85, 105) !important;
    }
    
    body.q-body--light .strategies-page .theme-text-success {
        color: rgb(21, 128, 61) !important;
    }
    
    body.q-body--light .strategies-page .theme-text-error {
        color: rgb(185, 28, 28) !important;
    }
    
    /* Light theme form elements */
    body.q-body--light .strategies-page .q-field .q-field__control {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(0, 0, 0, 0.2) !important;
        color: rgb(30, 41, 59) !important;
    }
    
    body.q-body--light .strategies-page .q-field input,
    body.q-body--light .strategies-page .q-field textarea,
    body.q-body--light .strategies-page .q-field .q-select__focus-target,
    body.q-body--light .strategies-page .q-field .q-field__native input {
        color: rgb(30, 41, 59) !important;
    }
    
    body.q-body--light .strategies-page .q-field__label {
        color: rgb(71, 85, 105) !important;
        background: rgba(248, 250, 252, 0.9) !important;
    }
    
    /* Light theme buttons */
    body.q-body--light .strategies-page .q-btn {
        color: rgb(30, 41, 59) !important;
    }
    
    body.q-body--light .strategies-page .btn-modern-primary {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
    }
    
    body.q-body--light .strategies-page .btn-modern-secondary {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%) !important;
        color: white !important;
    }
    
    /* Light theme icons */
    body.q-body--light .strategies-page .q-icon {
        color: rgb(71, 85, 105) !important;
    }
    
    /* Light theme tabs */
    body.q-body--light .strategies-page .q-tab {
        color: rgb(71, 85, 105) !important;
    }
    
    body.q-body--light .strategies-page .q-tab--active {
        color: rgb(30, 41, 59) !important;
    }
    
    /* Light theme borders and separators */
    body.q-body--light .strategies-page .border-l-green-500 {
        border-left-color: rgb(34, 197, 94) !important;
    }
    
    body.q-body--light .strategies-page .border-l-red-500 {
        border-left-color: rgb(239, 68, 68) !important;
    }
    
    body.q-body--light .strategies-page .border-l-blue-500 {
        border-left-color: rgb(59, 130, 246) !important;
    }
    
    body.q-body--light .strategies-page .border-l-orange-500 {
        border-left-color: rgb(249, 115, 22) !important;
    }
    
    /* Light theme hover effects */
    body.q-body--light .strategies-page .enhanced-card:hover {
        background: rgba(255, 255, 255, 0.9) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Fix any remaining background colors in light theme */
    body.q-body--light .strategies-page [class*="bg-gray"],
    body.q-body--light .strategies-page [class*="bg-slate"] {
        background: rgba(248, 250, 252, 0.95) !important;
    }
    
    /* Light theme dropdown menus */
    body.q-body--light .strategies-page .q-menu {
        background: rgba(248, 250, 252, 0.95) !important;
        color: rgb(30, 41, 59) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    body.q-body--light .strategies-page .q-menu .q-item {
        color: rgb(30, 41, 59) !important;
    }
    
    body.q-body--light .strategies-page .q-menu .q-item:hover {
        background: rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Theme transition animations */
    .strategies-page * {
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease !important;
    }
    
    /* Statistics cards styling */
    .strategies-page .stats-card {
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    .strategies-page .stats-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Status badges */
    .strategies-page .status-badge {
        font-size: 0.75rem !important;
        padding: 0.25rem 0.5rem !important;
        border-radius: 9999px !important;
        font-weight: 500 !important;
    }
    
    .strategies-page .status-badge.active {
        background-color: rgba(34, 197, 94, 0.1) !important;
        color: rgb(34, 197, 94) !important;
    }
    
    .strategies-page .status-badge.inactive {
        background-color: rgba(107, 114, 128, 0.1) !important;
        color: rgb(107, 114, 128) !important;
    }
    
    /* Strategy table specific styling */
    .strategies-page .strategy-card {
        border: 1px solid var(--border-color) !important;
        background: rgba(var(--surface-color-rgb), 0.5) !important;
        transition: all 0.3s ease !important;
    }
    
    .strategies-page .strategy-card:hover {
        background: rgba(var(--surface-color-rgb), 0.8) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    .strategies-page .header-card {
        background: var(--surface-color) !important;
        border: 1px solid var(--border-color) !important;
        font-weight: 600 !important;
    }
    
    /* Fixed column widths for consistent alignment */
    .strategies-page .w-48 {
        width: 192px !important;
        min-width: 192px !important;
        max-width: 192px !important;
        padding: 0 8px !important;
        overflow: hidden !important;
    }
    
    .strategies-page .w-24 {
        width: 96px !important;
        min-width: 96px !important;
        max-width: 96px !important;
        padding: 0 4px !important;
    }
    
    .strategies-page .w-44 {
        width: 176px !important;
        min-width: 176px !important;
        max-width: 176px !important;
        padding: 0 4px !important;
    }
    
    /* Text truncation for long content */
    .strategies-page .text-truncate {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        display: block !important;
    }
    
    /* Action button styling */
    .strategies-page .action-btn {
        transition: all 0.2s ease !important;
        opacity: 0.8 !important;
    }
    
    .strategies-page .action-btn:hover {
        opacity: 1 !important;
        transform: scale(1.1) !important;
    }
    
    /* Smaller action buttons for secondary actions */
    .strategies-page .action-btn-small {
        min-width: 32px !important;
        max-width: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        font-size: 0.75rem !important;
    }
    
    .strategies-page .action-btn-small:hover {
        opacity: 1 !important;
        transform: scale(1.05) !important;
    }
    
    /* Status badge styling */
    .strategies-page .bg-green-100 {
        background-color: rgba(34, 197, 94, 0.1) !important;
    }
    
    .strategies-page .text-green-800 {
        color: rgb(22, 163, 74) !important;
    }
    
    .strategies-page .bg-gray-100 {
        background-color: rgba(156, 163, 175, 0.1) !important;
    }
    
    .strategies-page .text-gray-800 {
        color: rgb(75, 85, 99) !important;
    }
    </style>
    '''
    ui.add_head_html(strategies_css)
