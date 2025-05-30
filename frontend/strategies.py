"""
Algorithmic Trading Strategies Module for NiceGUI Algo Trading Application
Implements UI for defining, configuring, and managing trading strategies.
"""

from nicegui import ui
import pandas as pd
import asyncio
import json
import logging
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# --- Strategy Definition Components --- #

def render_indicator_selector(indicators):
    """Renders a selector for technical indicators."""
    with ui.column().classes("gap-2"):
        indicator_type = ui.select(list(indicators.keys()), label="Indicator Type").classes("w-full")
        params_container = ui.column().classes("w-full")

        def update_params():
            params_container.clear()
            indicator_name = indicator_type.value
            if indicator_name and indicator_name in indicators:
                with params_container:
                    ui.label(f"{indicator_name} Parameters").classes("text-caption")
                    param_inputs = {}
                    for param, default in indicators[indicator_name].items():
                        param_inputs[param] = ui.number(label=param.replace("_", " ").title(), value=default).classes("w-full")
                    # Store param inputs for later retrieval
                    params_container.storage["param_inputs"] = param_inputs

        indicator_type.on("update:model-value", update_params)
        update_params() # Initial call

    return indicator_type, params_container

def render_condition_builder(indicators):
    """Renders a UI to build entry/exit conditions."""
    with ui.column().classes("gap-4 border p-4 rounded"):
        ui.label("Condition Builder").classes("text-subtitle1")
        conditions = []

        condition_list_ui = ui.column().classes("w-full gap-2")

        def add_condition_row():
            with condition_list_ui:
                with ui.row().classes("w-full items-center gap-2") as condition_row:
                    left_indicator_type, left_params_container = render_indicator_selector(indicators)
                    comparison = ui.select([">", "<", ">=", "<=", "==", "Crosses Above", "Crosses Below"], label="Comparison").classes("w-32")
                    right_indicator_type, right_params_container = render_indicator_selector(indicators)
                    # Add option for fixed value comparison
                    right_value_input = ui.number(label="Fixed Value", visible=False).classes("w-32")

                    def toggle_right_value(e):
                        is_fixed = e.value == "Fixed Value"
                        right_indicator_type.visible = not is_fixed
                        right_params_container.visible = not is_fixed
                        right_value_input.visible = is_fixed

                    # Allow selecting "Fixed Value" instead of a second indicator
                    right_options = list(indicators.keys()) + ["Fixed Value"]
                    right_indicator_type.options = right_options
                    right_indicator_type.on("update:model-value", toggle_right_value)

                    remove_button = ui.button(icon="delete", on_click=lambda row=condition_row: (condition_list_ui.remove(row), conditions.remove(condition_data)))

                    condition_data = {
                        "left_indicator": left_indicator_type,
                        "left_params": left_params_container,
                        "comparison": comparison,
                        "right_indicator": right_indicator_type,
                        "right_params": right_params_container,
                        "right_value": right_value_input
                    }
                    conditions.append(condition_data)

        ui.button("Add Condition", icon="add", on_click=add_condition_row).props("outline size=sm")
        add_condition_row() # Add the first row by default

    return conditions

# --- Strategies Page --- #

async def render_strategies_page(fetch_api, user_storage):
    """Render the main page for managing trading strategies."""
    broker = user_storage.get("broker", "Zerodha")
    ui.label("Algorithmic Trading Strategies").classes("text-h5 q-pa-md")

    # --- Available Indicators (Example) --- #
    # In a real app, this might come from the backend or a config file
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

    with ui.tabs().classes("w-full") as tabs:
        manage_tab = ui.tab("Manage Strategies")
        create_tab = ui.tab("Create New Strategy")

    with ui.tab_panels(tabs, value=manage_tab).classes("w-full"):
        # --- Manage Strategies Tab --- #
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
                                    <button class="q-btn q-btn-item non-selectable no-outline q-btn--flat q-btn--round q-btn--actionable q-focusable q-hoverable q-btn--dense bg-primary text-white" title="Edit">
                                        <span class="q-focus-helper"></span>
                                        <span class="q-btn__content text-center col items-center q-anchor--skip justify-center row">
                                            <i class="q-icon notranslate material-icons" aria-hidden="true" role="img">edit</i>
                                        </span>
                                    </button>
                                    <button class="q-btn q-btn-item non-selectable no-outline q-btn--flat q-btn--round q-btn--actionable q-focusable q-hoverable q-btn--dense bg-positive text-white" title="Activate/Deactivate">
                                        <span class="q-focus-helper"></span>
                                        <span class="q-btn__content text-center col items-center q-anchor--skip justify-center row">
                                            <i class="q-icon notranslate material-icons" aria-hidden="true" role="img">play_arrow</i>
                                        </span>
                                    </button>
                                    <button class="q-btn q-btn-item non-selectable no-outline q-btn--flat q-btn--round q-btn--actionable q-focusable q-hoverable q-btn--dense bg-negative text-white" title="Delete">
                                        <span class="q-focus-helper"></span>
                                        <span class="q-btn__content text-center col items-center q-anchor--skip justify-center row">
                                            <i class="q-icon notranslate material-icons" aria-hidden="true" role="img">delete</i>
                                        </span>
                                    </button>
                                `;
                                this.params = params;
                                this.eGui.children[0].addEventListener("click", () => this.editStrategy());
                                this.eGui.children[1].addEventListener("click", () => this.toggleStrategy());
                                this.eGui.children[2].addEventListener("click", () => this.deleteStrategy());
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
                            async editStrategy() {
                                // Emit event to Python to handle editing
                                await this.params.context.emitEvent("edit_strategy", this.params.data.strategy_id);
                            }
                            async toggleStrategy() {
                                // Emit event to Python to handle activation/deactivation
                                await this.params.context.emitEvent("toggle_strategy", this.params.data.strategy_id);
                            }
                            async deleteStrategy() {
                                // Emit event to Python to handle deletion
                                await this.params.context.emitEvent("delete_strategy", this.params.data.strategy_id);
                            }
                        }
                    """
                }
            }).classes("w-full mt-4")

            async def fetch_strategies():
                strategies = await fetch_api(f"/strategies/{broker}")
                if strategies and isinstance(strategies, list):
                    await strategies_grid.update_grid_options({"rowData": strategies})
                else:
                    await strategies_grid.update_grid_options({"rowData": []})

            await fetch_strategies()

            # Handle grid events
            async def handle_edit_strategy(e):
                strategy_id = e.args
                ui.notify(f"Editing strategy {strategy_id} - Functionality to be implemented.")
                # TODO: Fetch strategy details and populate the create/edit form
                # tabs.set_value(create_tab)

            async def handle_toggle_strategy(e):
                strategy_id = e.args
                # Find current status
                rows = await strategies_grid.get_rows()
                strategy = next((s for s in rows if s.get("strategy_id") == strategy_id), None)
                if not strategy:
                    return
                
                new_status = "inactive" if strategy.get("status") == "active" else "active"
                action = "deactivate" if new_status == "inactive" else "activate"
                
                with ui.loading(text=f"{action.capitalize()}ing strategy..."):
                    response = await fetch_api(f"/strategies/{strategy_id}/{action}", method="POST")
                    if response and response.get("status") == new_status:
                        ui.notify(f"Strategy {action}d successfully.", type="positive")
                        await fetch_strategies() # Refresh grid
                    else:
                        ui.notify(f"Failed to {action} strategy.", type="negative")

            async def handle_delete_strategy(e):
                strategy_id = e.args
                with ui.dialog() as dialog, ui.card():
                    ui.label(f"Are you sure you want to delete strategy {strategy_id}?")
                    with ui.row():
                        ui.button("Cancel", on_click=dialog.close).props("outline")
                        async def confirm_delete():
                            dialog.close()
                            with ui.loading(text="Deleting strategy..."):
                                response = await fetch_api(f"/strategies/{strategy_id}", method="DELETE")
                                if response and response.get("success"):
                                    ui.notify("Strategy deleted successfully.", type="positive")
                                    await fetch_strategies() # Refresh grid
                                else:
                                    ui.notify("Failed to delete strategy.", type="negative")
                        ui.button("Delete", on_click=confirm_delete).props("color=negative")
                dialog.open()

            strategies_grid.context.on("edit_strategy", handle_edit_strategy)
            strategies_grid.context.on("toggle_strategy", handle_toggle_strategy)
            strategies_grid.context.on("delete_strategy", handle_delete_strategy)

            ui.button("Refresh List", on_click=fetch_strategies).props("outline").classes("mt-4")

        # --- Create New Strategy Tab --- #
        with ui.tab_panel(create_tab):
            with ui.card().classes("w-full"):
                ui.label("Define New Strategy").classes("text-h6")

                with ui.form().classes("w-full gap-4") as form:
                    strategy_name = ui.input("Strategy Name").classes("w-full")
                    strategy_desc = ui.textarea("Description").classes("w-full")

                    ui.separator()
                    ui.label("Entry Conditions").classes("text-subtitle1")
                    entry_conditions = render_condition_builder(available_indicators)

                    ui.separator()
                    ui.label("Exit Conditions").classes("text-subtitle1")
                    exit_conditions = render_condition_builder(available_indicators)

                    ui.separator()
                    ui.label("Parameters").classes("text-subtitle1")
                    # Add strategy-level parameters if needed (e.g., timeframe, position sizing)
                    timeframe = ui.select(["1min", "3min", "5min", "15min", "30min", "60min", "day"], label="Timeframe", value="5min").classes("w-1/2")
                    position_sizing = ui.number(label="Position Size (Units or % Capital)", value=100).classes("w-1/2")

                    # Function to extract condition data
                    def extract_conditions(condition_ui_list):
                        extracted = []
                        for cond_data in condition_ui_list:
                            left_params = {}
                            if cond_data["left_params"].storage.get("param_inputs"):
                                left_params = {k: v.value for k, v in cond_data["left_params"].storage["param_inputs"].items()}
                            
                            right_params = {}
                            if cond_data["right_indicator"].value != "Fixed Value" and cond_data["right_params"].storage.get("param_inputs"):
                                right_params = {k: v.value for k, v in cond_data["right_params"].storage["param_inputs"].items()}
                            
                            extracted.append({
                                "left_indicator": cond_data["left_indicator"].value,
                                "left_params": left_params,
                                "comparison": cond_data["comparison"].value,
                                "right_indicator": cond_data["right_indicator"].value,
                                "right_params": right_params,
                                "right_value": cond_data["right_value"].value if cond_data["right_indicator"].value == "Fixed Value" else None
                            })
                        return extracted

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
                        
                        # logger.info(f"Saving strategy data: {json.dumps(strategy_data, indent=2)}")
                        
                        with ui.loading(text="Saving strategy..."):
                            # TODO: Add endpoint for updating existing strategy if an ID is present
                            response = await fetch_api("/strategies/", method="POST", data=strategy_data)
                            if response and response.get("strategy_id"):
                                ui.notify("Strategy saved successfully!", type="positive")
                                await fetch_strategies() # Refresh list on manage tab
                                tabs.set_value(manage_tab) # Switch back to manage tab
                                # Optionally clear the form here
                            else:
                                ui.notify("Failed to save strategy.", type="negative")

                    ui.button("Save Strategy", on_click=save_strategy).props("color=primary").classes("mt-4")
