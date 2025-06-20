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
    """Renders a visual UI to build entry/exit conditions."""
    conditions = []
    with ui.card().classes("w-full p-4"):
        ui.label(title).classes("text-subtitle1 mb-2")
        condition_list_ui = ui.column().classes("w-full gap-2")

        def add_condition_row():
            with condition_list_ui:
                with ui.row().classes("w-full items-center gap-2 border p-2 rounded") as condition_row:
                    left_indicator, left_params, left_value, left_param_values = render_indicator_selector(indicators, "left")
                    comparison = ui.select(
                        [">", "<", ">=", "<=", "==", "Crosses Above", "Crosses Below"],
                        label="Comparison"
                    ).props("dense hint='Select comparison operator'").classes("w-32")
                    right_indicator, right_params, right_value, right_param_values = render_indicator_selector(indicators, "right")
                    remove_button = ui.button(icon="delete", on_click=lambda: (condition_list_ui.remove(condition_row), conditions.remove(condition_data))).props("flat round dense")

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

        ui.button("Add Condition", icon="add", on_click=add_condition_row).props("outline size=sm")
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
    ui.label("Algorithmic Trading Strategies").classes("text-h5 q-pa-md")

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

    with ui.tabs().classes("w-full") as tabs:
        manage_tab = ui.tab("Manage Strategies")
        create_tab = ui.tab("Create/Edit Strategy")

    with ui.tab_panels(tabs, value=manage_tab).classes("w-full") as tab_panels:
        with ui.tab_panel(manage_tab):
            ui.label("Existing Strategies").classes("text-h6 q-mb-md")
            # Create a grid for table-like display
            strategies_grid = ui.grid().classes("w-full mt-4")
            # Header row
            with strategies_grid:
                with ui.row().classes("w-full bg-primary text-white p-2 font-bold"):
                    ui.label("Name").classes("flex-1")
                    ui.label("Description").classes("flex-1")
                    ui.label("Entry Conditions").classes("flex-2")
                    ui.label("Exit Conditions").classes("flex-2")
                    ui.label("Status").classes("w-24 text-center")
                    ui.label("Actions").classes("w-32 text-center")

            # Container for strategy rows
            strategies_container = ui.column().classes("w-full")

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
                    ui.label(f"Are you sure you want to delete strategy {strategy_id}?")
                    with ui.row():
                        ui.button("Cancel", on_click=dialog.close).props("outline")
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
                strategies = await fetch_api(f"/strategies/broker/{broker}")
                strategies_container.clear()  # Clear existing rows
                if strategies and isinstance(strategies, list):
                    with strategies_container:
                        for s in strategies:
                            strategy_id = s["strategy_id"]
                            with ui.row().classes("w-full border-b p-2 items-center"):
                                ui.label(s["name"]).classes("flex-1")
                                ui.label(s["description"]).classes("flex-1")
                                ui.label(format_conditions(s.get("entry_conditions", []))).classes("flex-2")
                                ui.label(format_conditions(s.get("exit_conditions", []))).classes("flex-2")
                                ui.label(s["status"]).classes("w-24 text-center")
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
                        ui.label("No strategies found.").classes("w-full text-center text-warning")
                    ui.notify("No strategies found.", type="warning")

            await fetch_strategies()

            ui.button("Refresh List", on_click=fetch_strategies).props("outline").classes("mt-4")

        with ui.tab_panel(create_tab) as create_tab_content:
            if not create_tab_initialized:
                ui.label("Loading...").classes("text-center")
                async def init_on_mount():
                    await initialize_create_tab()
                ui.timer(0.1, init_on_mount, once=True)

        async def initialize_create_tab():
            global create_tab_initialized
            logger.debug("Initializing Create/Edit Strategy tab")
            create_tab_content.clear()  # Clear existing content to prevent duplication
            with create_tab_content:
                with ui.card().classes("w-full p-4"):
                    ui.label("Define Strategy").classes("text-h6")
                    with ui.column().classes("w-full gap-4"):
                        ui_refs["broker_select"] = ui.select(
                            ["Zerodha", "Upstox"],
                            label="Broker",
                            value=broker
                        ).props("hint='Select broker for this strategy'").classes("w-full")
                        ui_refs["strategy_name"] = ui.input(
                            "Strategy Name",
                            validation={"Required": bool, "Max length": lambda v: len(v) <= 50}
                        ).props("hint='Unique name for the strategy'").classes("w-full")
                        ui_refs["strategy_desc"] = ui.textarea("Description").props("hint='Brief description of the strategy'").classes("w-full")

                        ui.separator()
                        ui.label("Entry Conditions").classes("text-subtitle1")
                        ui_refs["entry_conditions"], ui_refs["entry_container"] = render_condition_builder(
                            available_indicators, "Entry Conditions", instruments, ui_refs, "entry_conditions"
                        )

                        ui.separator()
                        ui.label("Exit Conditions").classes("text-subtitle1")
                        ui_refs["exit_conditions"], ui_refs["exit_container"] = render_condition_builder(
                            available_indicators, "Exit Conditions", instruments, ui_refs, "exit_conditions"
                        )

                        ui.separator()
                        ui.label("Parameters").classes("text-subtitle1")
                        with ui.row().classes("w-full gap-2"):
                            ui_refs["timeframe"] = ui.select(
                                ["1min", "3min", "5min", "15min", "30min", "60min", "day"],
                                label="Timeframe",
                                value="5min"
                            ).props("hint='Data interval for strategy'").classes("w-1/3")
                            ui_refs["position_sizing"] = ui.number(
                                label="Position Size (Units or % Capital)",
                                value=100,
                                validation={"Positive": lambda v: v > 0}
                            ).props("hint='Size of each trade'").classes("w-1/3")
                            ui_refs["preview_instrument"] = ui.select(
                                options=sorted(list(instruments.keys())),
                                label="Preview Instrument",
                                value="RELIANCE"
                            ).props("clearable filter hint='Instrument for preview backtest'").classes("w-1/3")

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
                            ui_refs["preview_button"].props("loading=true disable=true text='Previewing...'")
                            try:
                                response = await fetch_api("/algo-trading/backtest", method="POST", data={
                                    "instrument_token": ui_refs["preview_instrument"].value,
                                    "timeframe": ui_refs["timeframe"].value,
                                    "strategy": json.dumps(strategy_data),
                                    "params": {"initial_investment": 100000},
                                    "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                                    "end_date": datetime.now().strftime("%Y-%m-%d")
                                })
                                logger.debug(f"Backtest response: {json.dumps(response, indent=2)}")
                                if response and not response.get("error"):
                                    with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl"):
                                        ui.label("Preview Results").classes("text-h6")
                                        with ui.column():
                                            ui.label(f"Total Profit: ₹{response.get('TotalProfit', 0):.2f}")
                                            ui.label(f"Win Rate: {response.get('WinRate', 0):.2f}%")
                                            ui.label(f"Total Trades: {response.get('TotalTrades', 0)}")
                                            if tradebook := response.get("Tradebook", []):
                                                ui.label("Trade Details").classes("text-subtitle1 mt-4")
                                                formatted_tradebook = [
                                                    {
                                                        "Date": pd.to_datetime(trade["Date"]).strftime("%Y-%m-%d %H:%M:%S"),
                                                        "EntryPrice": f"{trade['EntryPrice']:.2f}",
                                                        "ExitPrice": f"{trade['ExitPrice']:.2f}",
                                                        "Profit": f"{trade['Profit']:.2f}",
                                                        "PortfolioValue": f"{trade['PortfolioValue']:.2f}"
                                                    }
                                                    for trade in tradebook
                                                ]
                                                logger.debug(f"Formatted tradebook: {len(formatted_tradebook)} trades")
                                                ui.table(
                                                    columns=[
                                                        {"name": "Date", "label": "Date", "field": "Date"},
                                                        {"name": "EntryPrice", "label": "Entry Price", "field": "EntryPrice"},
                                                        {"name": "ExitPrice", "label": "Exit Price", "field": "ExitPrice"},
                                                        {"name": "Profit", "label": "Profit", "field": "Profit"},
                                                        {"name": "PortfolioValue", "label": "Portfolio Value", "field": "PortfolioValue"}
                                                    ],
                                                    rows=formatted_tradebook,
                                                    pagination=10
                                                ).classes("w-full")
                                            else:
                                                ui.label("No trades executed.").classes("text-warning")
                                        ui.button("Close", on_click=dialog.close).props("outline")
                                    dialog.open()
                                else:
                                    ui.notify(f"Preview failed: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                            except Exception as e:
                                ui.notify(f"Error previewing strategy: {str(e)}", type="negative")
                                logger.error(f"Preview strategy error: {str(e)}")
                            finally:
                                ui_refs["preview_button"].props("loading=false disable=false text='Preview Strategy'")
                                ui.update()

                        async def save_strategy():
                            ui.update()
                            await asyncio.sleep(0.01)
                            logger.debug(f"Pre-save condition count: entry={len(ui_refs['entry_conditions'])}, exit={len(ui_refs['exit_conditions'])}")
                            for cond in ui_refs["entry_conditions"] + ui_refs["exit_conditions"]:
                                logger.debug(f"Condition state: left_indicator={cond['left_indicator'].value}, left_params={cond['left_param_values']}, right_indicator={cond['right_indicator'].value}, right_params={cond['right_param_values']}")
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
                            ui_refs["save_button"].props("loading=true disable=true text='Saving...'")
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
                                ui_refs["save_button"].props("loading=false disable=false text='Save Strategy'")
                                ui.update()

                        ui_refs["save_button"] = ui.button("Save Strategy").props("color=primary").classes("mt-4")
                        ui_refs["preview_button"] = ui.button("Preview Strategy").props("outline").classes("mt-4")

                        ui_refs["save_button"].on_click(save_strategy)
                        ui_refs["preview_button"].on_click(preview_strategy)

                        with ui.row():
                            ui_refs["preview_button"]
                            ui_refs["save_button"]

            create_tab_initialized = True
            logger.debug("Create/Edit Strategy tab initialized")
            return ui_refs["strategy_name"], ui_refs["broker_select"], ui_refs["timeframe"], ui_refs["position_sizing"], ui_refs["preview_instrument"], ui_refs["save_button"], ui_refs["preview_button"]

        async def on_tab_change():
            global create_tab_initialized
            logger.debug(f"Tab changed to: {tabs.value}, initialized: {create_tab_initialized}")
            if tabs.value == create_tab and not create_tab_initialized:
                ui_refs["strategy_name"], ui_refs["broker_select"], ui_refs["timeframe"], ui_refs["position_sizing"], ui_refs["preview_instrument"], ui_refs["save_button"], ui_refs["preview_button"] = await initialize_create_tab()

        tab_panels.on("update:model-value", on_tab_change)