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
    "exit_container": None
}

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
    """Renders a selector for technical indicators with persistent parameter bindings."""
    param_values = {}
    input_id = str(uuid4())
    with ui.column().classes("gap-2 w-full"):
        indicator_type = ui.select(
            options=list(indicators.keys()) + ["Fixed Value"],
            label="Indicator/Value",
            value=initial_indicator
        ).props(f"clearable dense hint='Select indicator or fixed value' name={prefix}_indicator_{input_id}").classes("w-full")
        params_container = ui.column().classes("w-full").bind_visibility_from(indicator_type, "value", lambda v: v != "Fixed Value")
        fixed_value_input = ui.number(
            label="Fixed Value",
            value=0,
            step=0.01
        ).props(f"clearable dense hint='Enter a fixed numerical value' name=fixed_value_{input_id}").classes("w-full").bind_visibility_from(indicator_type, "value", lambda v: v == "Fixed Value")

        def update_params():
            params_container.clear()
            nonlocal param_values
            indicator_name = indicator_type.value
            if indicator_name and indicator_name in indicators:
                with params_container:
                    ui.label(f"{indicator_name} Parameters").classes("text-caption")
                    params = initial_params if initial_params and indicator_name == initial_indicator else indicators[indicator_name]
                    for param, default in indicators[indicator_name].items():
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
            else:
                param_values.clear()

        if initial_indicator:
            update_params()
        indicator_type.on("update:model-value", update_params)

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

        # Tab panels with full-width content
        with ui.tab_panels(tabs, value=manage_tab).classes("w-full flex-1"):
            with ui.tab_panel(manage_tab).classes("w-full p-4"):
                # Manage strategies content with proper theme styling
                with ui.card().classes("w-full enhanced-card"):
                    with ui.card_section().classes("p-6 w-full"):
                        ui.label("Existing Strategies").classes("card-title theme-text-primary mb-4")

                        # Header row with theme-aware styling
                        with ui.row().classes("w-full strategy-parameters-row p-3 font-semibold mb-4 rounded-lg").style("background: var(--surface-color); border: 1px solid var(--border-color);"):
                            ui.label("Name").classes("flex-1 theme-text-primary")
                            ui.label("Description").classes("flex-1 theme-text-primary")
                            ui.label("Entry Conditions").classes("flex-2 theme-text-primary")
                            ui.label("Exit Conditions").classes("flex-2 theme-text-primary")
                            ui.label("Status").classes("w-24 text-center theme-text-primary")
                            ui.label("Actions").classes("w-32 text-center theme-text-primary")

                        # Container for strategy rows
                        strategies_container = ui.column().classes("w-full gap-2")

                        async def handle_edit_strategy(strategy_id):
                            if not create_tab_initialized:
                                await initialize_create_tab()
                            response = await fetch_api(f"/strategies/{strategy_id}")
                            if response and not response.get("error"):
                                tabs.set_value(create_tab)
                                ui_refs["strategy_name"].value = response["name"]
                                ui_refs["strategy_desc"].value = response["description"]
                                ui_refs["broker_select"].value = response["broker"]
                                ui_refs["timeframe"].value = response["parameters"].get("timeframe", "5min")
                                ui_refs["position_sizing"].value = response["parameters"].get("position_sizing", 100)
                                ui_refs["preview_instrument"].value = "RELIANCE"
                                for cond in ui_refs["entry_conditions"]:
                                    ui_refs["entry_container"].remove(cond["row"])
                                for cond in ui_refs["exit_conditions"]:
                                    ui_refs["exit_container"].remove(cond["row"])
                                ui_refs["entry_conditions"].clear()
                                ui_refs["exit_conditions"].clear()
                                for cond in response.get("entry_conditions", []):
                                    add_condition_row_to_ui(ui_refs["entry_container"], ui_refs["entry_conditions"], cond, available_indicators)
                                for cond in response.get("exit_conditions", []):
                                    add_condition_row_to_ui(ui_refs["exit_container"], ui_refs["exit_conditions"], cond, available_indicators)
                            else:
                                ui.notify(f"Failed to load strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")

                        async def handle_toggle_strategy(strategy_id):
                            response = await fetch_api(f"/strategies/{strategy_id}")
                            if response and not response.get("error"):
                                current_status = response.get("status")
                                new_status = "inactive" if current_status == "active" else "active"
                                action = "deactivate" if new_status == "inactive" else "activate"
                                toggle_button = ui.button(f"{action.capitalize()} Strategy").props("loading=true disable=true")
                                try:
                                    response = await fetch_api(f"/strategies/{strategy_id}/{action}", method="POST")
                                    if response and response.get("status") == new_status:
                                        ui.notify(f"Strategy {action}d successfully.", type="positive")
                                        await fetch_strategies()
                                    else:
                                        ui.notify(f"Failed to {action} strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                                except Exception as e:
                                    ui.notify(f"Error toggling strategy: {str(e)}", type="negative")
                                    logger.error(f"Toggle strategy error: {str(e)}")
                                finally:
                                    toggle_button.delete()
                                    ui.update()
                            else:
                                ui.notify(f"Failed to load strategy status: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")

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
                            strategies = await fetch_api(f"/strategies/all/{broker}")
                            strategies_container.clear()
                            if strategies and isinstance(strategies, list):
                                with strategies_container:
                                    for s in strategies:
                                        strategy_id = s["strategy_id"]
                                        with ui.row().classes("w-full p-2 items-center").style("border-bottom: 1px solid var(--border-color);"):
                                            ui.label(s["name"]).classes("flex-1 theme-text-primary")
                                            ui.label(s["description"]).classes("flex-1 theme-text-secondary")
                                            ui.label(format_conditions(s.get("entry_conditions", []))).classes("flex-2 theme-text-secondary")
                                            ui.label(format_conditions(s.get("exit_conditions", []))).classes("flex-2 theme-text-secondary")
                                            ui.label(s["status"]).classes("w-24 text-center theme-text-primary")
                                            with ui.row().classes("w-32 gap-1 justify-center"):
                                                ui.button(icon="edit", on_click=lambda sid=strategy_id: handle_edit_strategy(sid)).props("flat round dense color=primary")
                                                ui.button(
                                                    icon="play_arrow" if s["status"] == "inactive" else "pause",
                                                    on_click=lambda sid=strategy_id: handle_toggle_strategy(sid)
                                                ).props("flat round dense color=positive")
                                                ui.button(icon="delete", on_click=lambda sid=strategy_id: handle_delete_strategy(sid)).props("flat round dense color=negative")
                                logger.debug(f"Fetched and formatted {len(strategies)} strategies")
                            else:
                                with strategies_container:
                                    ui.label("No strategies found.").classes("w-full text-center theme-text-secondary")
                                ui.notify("No strategies found.", type="warning")

                        await fetch_strategies()

                        ui.button("Refresh List", icon="refresh", on_click=fetch_strategies).props("outline").classes("mt-4 btn-modern-primary")

            with ui.tab_panel(create_tab) as create_tab_content:
                # Always show content immediately when tab is created
                async def init_and_render():
                    await initialize_create_tab()

                # Initialize immediately
                ui.timer(0.1, init_and_render, once=True)

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
                            ui_refs["position_sizing"] = ui.number(
                                label="Position Size",
                                value=100,
                                validation={"Positive": lambda v: v > 0}
                            ).props("filled hint='Size of each trade (units or % of capital)' suffix='units'").classes("flex-1 theme-text-primary")
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
                                            ui.label(f"₹{response.get('TotalProfit', 0):.2f}").classes("text-2xl font-bold theme-text-success")
                                        with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-blue-500"):
                                            ui.label("Win Rate").classes("text-sm theme-text-secondary")
                                            ui.label(f"{response.get('WinRate', 0):.1f}%").classes("text-2xl font-bold theme-text-info")
                                        with ui.card().classes("flex-1 p-4 enhanced-card border-l-4 border-l-purple-500"):
                                            ui.label("Total Trades").classes("text-sm theme-text-secondary")
                                            ui.label(f"{response.get('TotalTrades', 0)}").classes("text-2xl font-bold theme-text-primary")

                                    if tradebook := response.get("Tradebook", []):
                                        ui.label("Trade History").classes("text-lg font-semibold theme-text-primary mb-2")
                                        formatted_tradebook = [
                                            {
                                                "Date": pd.to_datetime(trade["Date"]).strftime("%Y-%m-%d %H:%M"),
                                                "Entry": f"₹{trade['EntryPrice']:.2f}",
                                                "Exit": f"₹{trade['ExitPrice']:.2f}",
                                                "Profit": f"₹{trade['Profit']:.2f}",
                                                "Portfolio": f"₹{trade['PortfolioValue']:.2f}"
                                            }
                                            for trade in tradebook[:50]  # Limit to 50 trades for performance
                                        ]
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
                    strategy_data = {
                        "name": ui_refs["strategy_name"].value,
                        "description": ui_refs["strategy_desc"].value,
                        "entry_conditions": entry_conds,
                        "exit_conditions": exit_conds,
                        "parameters": {
                            "timeframe": ui_refs["timeframe"].value,
                            "position_sizing": ui_refs["position_sizing"].value
                        },
                        "broker": ui_refs["broker_select"].value
                    }
                    logger.debug(f"Saving strategy: {json.dumps(strategy_data, indent=2)}")
                    ui_refs["save_button"].props("loading=true disable=true")
                    try:
                        response = await fetch_api("/strategies/", method="POST", data=strategy_data)
                        if response and response.get("strategy_id"):
                            ui.notify("Strategy saved successfully!", type="positive")
                            await fetch_strategies()
                            tabs.set_value(manage_tab)
                        else:
                            ui.notify(f"Failed to save strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                    except Exception as e:
                        ui.notify(f"Error saving strategy: {str(e)}", type="negative")
                        logger.error(f"Save strategy error: {str(e)}")
                    finally:
                        ui_refs["save_button"].props("loading=false disable=false")
                        ui.update()

                def reset_strategy_form():
                    """Reset all form fields to their default values"""
                    try:
                        # Reset basic info fields
                        ui_refs["strategy_name"].value = ""
                        ui_refs["strategy_desc"].value = ""
                        ui_refs["broker_select"].value = broker
                        ui_refs["timeframe"].value = "5min"
                        ui_refs["position_sizing"].value = 100
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
    </style>
    '''
    ui.add_head_html(strategies_css)
