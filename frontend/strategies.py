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

logger = logging.getLogger(__name__)

def render_indicator_selector(indicators, prefix=""):
    """Renders a selector for technical indicators with parameters."""
    param_inputs = {}  # Local dictionary to store parameter inputs
    with ui.column().classes("gap-2 w-full"):
        indicator_type = ui.select(
            options=list(indicators.keys()) + ["Fixed Value"],
            label="Indicator/Value",
            value=None
        ).props(f"clearable dense hint='Select indicator or fixed value' name={prefix}_indicator").classes("w-full")
        params_container = ui.column().classes("w-full").bind_visibility_from(indicator_type, "value", lambda v: v != "Fixed Value")
        fixed_value_input = ui.number(
            label="Fixed Value",
            value=0,
            step=0.01
        ).props("clearable dense hint='Enter a fixed numerical value'").classes("w-full").bind_visibility_from(indicator_type, "value", lambda v: v == "Fixed Value")

        def update_params():
            params_container.clear()
            nonlocal param_inputs
            param_inputs = {}  # Reset dictionary
            indicator_name = indicator_type.value
            if indicator_name and indicator_name in indicators:
                with params_container:
                    ui.label(f"{indicator_name} Parameters").classes("text-caption")
                    for param, default in indicators[indicator_name].items():
                        param_inputs[param] = ui.number(
                            label=param.replace("_", " ").title(),
                            value=default,
                            step=1 if "period" in param else 0.1
                        ).props("clearable dense").classes("w-full")

        indicator_type.on("update:model-value", update_params)
        return indicator_type, params_container, fixed_value_input, param_inputs

def render_condition_builder(indicators, title="Conditions"):
    """Renders a visual UI to build entry/exit conditions."""
    conditions = []
    with ui.card().classes("w-full p-4"):
        ui.label(title).classes("text-subtitle1 mb-2")
        condition_list_ui = ui.column().classes("w-full gap-2")

        def add_condition_row():
            with condition_list_ui:
                with ui.row().classes("w-full items-center gap-2 border p-2 rounded") as condition_row:
                    left_indicator, left_params, left_value, left_param_inputs = render_indicator_selector(indicators, "left")
                    comparison = ui.select(
                        [">", "<", ">=", "<=", "==", "Crosses Above", "Crosses Below"],
                        label="Comparison"
                    ).props("dense hint='Select comparison operator'").classes("w-32")
                    right_indicator, right_params, right_value, right_param_inputs = render_indicator_selector(indicators, "right")
                    remove_button = ui.button(icon="delete", on_click=lambda: (condition_list_ui.remove(condition_row), conditions.remove(condition_data))).props("flat round dense")

                    condition_data = {
                        "row": condition_row,
                        "left_indicator": left_indicator,
                        "left_params": left_params,
                        "left_value": left_value,
                        "left_param_inputs": left_param_inputs,
                        "comparison": comparison,
                        "right_indicator": right_indicator,
                        "right_params": right_params,
                        "right_value": right_value,
                        "right_param_inputs": right_param_inputs,
                        "remove_button": remove_button
                    }
                    conditions.append(condition_data)

        ui.button("Add Condition", icon="add", on_click=add_condition_row).props("outline size=sm")
        add_condition_row()

    return conditions, condition_list_ui

def extract_conditions(conditions):
    """Extract condition data from UI components."""
    result = []
    for cond in conditions:
        left_params = {k: v.value for k, v in cond["left_param_inputs"].items()}
        right_params = {k: v.value for k, v in cond["right_param_inputs"].items()}
        condition = {
            "left_indicator": cond["left_indicator"].value,
            "left_params": left_params if cond["left_indicator"].value != "Fixed Value" else None,
            "left_value": cond["left_value"].value if cond["left_indicator"].value == "Fixed Value" else None,
            "comparison": cond["comparison"].value,
            "right_indicator": cond["right_indicator"].value,
            "right_params": right_params if cond["right_indicator"].value != "Fixed Value" else None,
            "right_value": cond["right_value"].value if cond["right_indicator"].value == "Fixed Value" else None
        }
        if condition["left_indicator"] and condition["comparison"] and (condition["right_indicator"] or condition["right_value"] is not None):
            result.append(condition)
    return result

async def render_strategies_page(fetch_api, user_storage):
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

    # Store condition list UI references
    condition_ui_refs = {"entry": None, "exit": None}

    with ui.tabs().classes("w-full") as tabs:
        manage_tab = ui.tab("Manage Strategies")
        create_tab = ui.tab("Create/Edit Strategy")

    with ui.tab_panels(tabs, value=manage_tab).classes("w-full"):
        with ui.tab_panel(manage_tab):
            ui.label("Existing Strategies").classes("text-h6 q-mb-md")
            strategies_grid = ui.aggrid({
                "columnDefs": [
                    {"headerName": "ID", "field": "strategy_id", "maxWidth": 100},
                    {"headerName": "Name", "field": "name"},
                    {"headerName": "Description", "field": "description"},
                    {"headerName": "Status", "field": "status"},
                    {"headerName": "Actions", "cellRenderer": "actionButtons"}
                ],
                "rowData": [],
                "rowSelection": "single",
                "pagination": True,
                "paginationPageSize": 10,
                "components": {
                    "actionButtons": """
                        class ActionButtons {
                            init(params) {
                                this.eGui = document.createElement("div");
                                this.eGui.innerHTML = `
                                    <button class="q-btn q-btn-item q-btn--flat q-btn--round bg-primary text-white" title="Edit">
                                        <i class="q-icon material-icons">edit</i>
                                    </button>
                                    <button class="q-btn q-btn-item q-btn--flat q-btn--round bg-positive text-white" title="Activate/Deactivate">
                                        <i class="q-icon material-icons">play_arrow</i>
                                    </button>
                                    <button class="q-btn q-btn-item q-btn--flat q-btn--round bg-negative text-white" title="Delete">
                                        <i class="q-icon material-icons">delete</i>
                                    </button>
                                `;
                                this.params = params;
                                this.eGui.children[0].addEventListener("click", () => params.api.emitEvent('edit_strategy', this.params.data.strategy_id));
                                this.eGui.children[1].addEventListener("click", () => params.api.emitEvent('toggle_strategy', this.params.data.strategy_id));
                                this.eGui.children[2].addEventListener("click", () => params.api.emitEvent('delete_strategy', this.params.data.strategy_id));
                                this.updateToggleButton();
                            }
                            getGui() { return this.eGui; }
                            refresh(params) {
                                this.params = params;
                                this.updateToggleButton();
                                return true;
                            }
                            updateToggleButton() {
                                const button = this.eGui.children[1];
                                const icon = button.querySelector("i");
                                if (this.params.data.status === "active") {
                                    icon.textContent = "pause";
                                    button.title = "Deactivate";
                                } else {
                                    icon.textContent = "play_arrow";
                                    button.title = "Activate";
                                }
                            }
                        }
                    """
                }
            }).classes("w-full mt-4")

            async def fetch_strategies():
                strategies = await fetch_api(f"/strategies/{broker}")
                if strategies and isinstance(strategies, list):
                    strategies_grid.options["rowData"] = strategies
                    strategies_grid.update()
                else:
                    strategies_grid.options["rowData"] = []
                    strategies_grid.update()
                    ui.notify("No strategies found.", type="warning")

            await fetch_strategies()

            async def handle_edit_strategy(strategy_id):
                response = await fetch_api(f"/strategies/{strategy_id}")
                if response and not response.get("error"):
                    tabs.set_value(create_tab)
                    strategy_name.value = response["name"]
                    strategy_desc.value = response["description"]
                    timeframe.value = response["parameters"].get("timeframe", "5min")
                    position_sizing.value = response["parameters"].get("position_sizing", 100)
                    # Clear existing conditions
                    for cond in entry_conditions:
                        condition_ui_refs["entry"].remove(cond["row"])
                    for cond in exit_conditions:
                        condition_ui_refs["exit"].remove(cond["row"])
                    entry_conditions.clear()
                    exit_conditions.clear()
                    # Rebuild conditions (simplified, requires custom logic to populate UI)
                    ui.notify("Strategy loaded for editing. Conditions need to be manually re-added.", type="info")
                else:
                    ui.notify(f"Failed to load strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")

            async def handle_toggle_strategy(strategy_id):
                strategy = next((s for s in strategies_grid.options["rowData"] if s.get("strategy_id") == strategy_id), None)
                if not strategy:
                    ui.notify("Strategy not found.", type="negative")
                    return
                new_status = "inactive" if strategy.get("status") == "active" else "active"
                action = "deactivate" if new_status == "inactive" else "activate"
                toggle_button = ui.button(f"{action.capitalize()} Strategy").props("loading=true disable=true")
                try:
                    response = await fetch_api(f"/strategies/{strategy_id}/{action}", method="POST")
                    if response and response.get("status") == new_status:
                        ui.notify(f"Strategy {action}d successfully.", type="positive")
                        await fetch_strategies()
                    else:
                        ui.notify(f"Failed to {action} strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                finally:
                    toggle_button.props("loading=false disable=false")

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
                            finally:
                                delete_button.props("loading=false disable=false")
                        ui.button("Delete", on_click=confirm_delete).props("color=negative")
                dialog.open()

            strategies_grid.on("edit_strategy", lambda e: handle_edit_strategy(e.args))
            strategies_grid.on("toggle_strategy", lambda e: handle_toggle_strategy(e.args))
            strategies_grid.on("delete_strategy", lambda e: handle_delete_strategy(e.args))

            ui.button("Refresh List", on_click=fetch_strategies).props("outline").classes("mt-4")

        with ui.tab_panel(create_tab):
            with ui.card().classes("w-full p-4"):
                ui.label("Define Strategy").classes("text-h6")
                with ui.column().classes("w-full gap-4"):
                    strategy_name = ui.input(
                        "Strategy Name",
                        validation={"Required": bool, "Max length": lambda v: len(v) <= 50}
                    ).props("hint='Unique name for the strategy'").classes("w-full")
                    strategy_desc = ui.textarea("Description").props("hint='Brief description of the strategy'").classes("w-full")

                    ui.separator()
                    ui.label("Entry Conditions").classes("text-subtitle1")
                    entry_conditions, condition_ui_refs["entry"] = render_condition_builder(available_indicators, "Entry Conditions")

                    ui.separator()
                    ui.label("Exit Conditions").classes("text-subtitle1")
                    exit_conditions, condition_ui_refs["exit"] = render_condition_builder(available_indicators, "Exit Conditions")

                    ui.separator()
                    ui.label("Parameters").classes("text-subtitle1")
                    with ui.row().classes("w-full gap-2"):
                        timeframe = ui.select(
                            ["1min", "3min", "5min", "15min", "30min", "60min", "day"],
                            label="Timeframe",
                            value="5min"
                        ).props("hint='Data interval for strategy'").classes("w-1/2")
                        position_sizing = ui.number(
                            label="Position Size (Units or % Capital)",
                            value=100,
                            validation={"Positive": lambda v: v > 0}
                        ).props("hint='Size of each trade'").classes("w-1/2")

                    async def preview_strategy():
                        entry_conds = extract_conditions(entry_conditions)
                        exit_conds = extract_conditions(exit_conditions)
                        if not entry_conds or not exit_conds:
                            ui.notify("Please define at least one entry and one exit condition.", type="negative")
                            return
                        strategy_data = {
                            "name": strategy_name.value,
                            "description": strategy_desc.value,
                            "entry_conditions": entry_conds,
                            "exit_conditions": exit_conds,
                            "parameters": {"timeframe": timeframe.value, "position_sizing": position_sizing.value}
                        }
                        preview_button = ui.button("Previewing...").props("loading=true disable=true")
                        try:
                            response = await fetch_api("/algo-trading/backtest", method="POST", data={
                                "instrument_token": "RELIANCE",  # Default for preview
                                "timeframe": timeframe.value,
                                "strategy": json.dumps(strategy_data),
                                "params": {"initial_investment": 100000},
                                "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                                "end_date": datetime.now().strftime("%Y-%m-%d")
                            })
                            if response and not response.get("error"):
                                with ui.dialog() as dialog, ui.card():
                                    ui.label(f"Preview Results:")
                                    ui.label(f"Total Profit: ₹{response['TotalProfit']:.2f}")
                                    ui.label(f"Win Rate: {response['WinRate']:.2f}%")
                                    ui.label(f"Total Trades: {response['TotalTrades']}")
                                    ui.button("Close", on_click=dialog.close).props("outline")
                                dialog.open()
                            else:
                                ui.notify(f"Preview failed: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                        finally:
                            preview_button.props("loading=false disable=false")

                    async def save_strategy():
                        if not strategy_name.value:
                            ui.notify("Strategy name is required.", type="negative")
                            return
                        entry_conds = extract_conditions(entry_conditions)
                        exit_conds = extract_conditions(exit_conditions)
                        if not entry_conds or not exit_conds:
                            ui.notify("Please define at least one entry and one exit condition.", type="negative")
                            return
                        strategy_data = {
                            "name": strategy_name.value,
                            "description": strategy_desc.value,
                            "entry_conditions": entry_conds,
                            "exit_conditions": exit_conds,
                            "parameters": {
                                "timeframe": timeframe.value,
                                "position_sizing": position_sizing.value
                            },
                            "broker": broker
                        }
                        save_button = ui.button("Saving...").props("loading=true disable=true")
                        try:
                            response = await fetch_api("/strategies/", method="POST", data=strategy_data)
                            if response and response.get("strategy_id"):
                                ui.notify("Strategy saved successfully!", type="positive")
                                await fetch_strategies()
                                tabs.set_value(manage_tab)
                            else:
                                ui.notify(f"Failed to save strategy: {response.get('error', {}).get('message', 'Unknown error')}", type="negative")
                        finally:
                            save_button.props("loading=false disable=false")

                    with ui.row():
                        ui.button("Preview Strategy", on_click=preview_strategy).props("outline").classes("mt-4")
                        ui.button("Save Strategy", on_click=save_strategy).props("color=primary").classes("mt-4")