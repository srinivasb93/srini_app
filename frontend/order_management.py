# Enhanced Order Management Module - order_management.py
# Complete implementation with beautiful dashboard styling
# Preserves all existing functionality: Regular, Scheduled, GTT, and Auto Orders

from nicegui import ui
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


async def render_order_management(fetch_api, user_storage, instruments):
    """Render the complete enhanced order management page with all order types"""
    broker = user_storage.get('default_broker', 'Zerodha')

    # Fetch instruments if not already available
    if not instruments:
        instruments_data = await fetch_api(f"/instruments/{broker}/?exchange=NSE")
        if instruments_data:
            # Filter for equity instruments
            equity_instruments = [i for i in instruments_data if i.get('segment') == 'NSE' and i.get('instrument_type') == 'EQ']
            instruments.update({i['trading_symbol']: i['instrument_token'] for i in equity_instruments})

    # Main container with dashboard styling
    with ui.column().classes("enhanced-dashboard w-full min-h-screen"):

        # Enhanced title section (matching dashboard.py)
        with ui.row().classes("dashboard-title-section w-full justify-between items-center p-4"):
            # Left side - Title and subtitle
            with ui.column().classes("gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("shopping_cart", size="2rem").classes("text-cyan-400")
                    ui.label(f"Order Management - {broker}").classes("text-3xl font-bold text-white dashboard-title")
                    ui.chip("LIVE", color="green").classes("text-xs status-chip")

                ui.label("Place, schedule and manage all your trading orders").classes("text-gray-400 dashboard-subtitle")

            # Right side - Action buttons
            with ui.row().classes("items-center gap-4"):
                ui.button("Quick Buy", icon="trending_up", color="green").classes("text-white")
                ui.button("Quick Sell", icon="trending_down", color="red").classes("text-white")
                ui.button("Basket Orders", icon="shopping_basket").classes("text-cyan-400")

        # Enhanced tabs for different order types
        with ui.card().classes("dashboard-card w-full m-4"):
            with ui.tabs().props("dense indicator-color=cyan").classes('w-full') as tabs:
                regular_tab = ui.tab('regular', label='Regular Orders', icon='flash_on')
                scheduled_tab = ui.tab('scheduled', label='Scheduled Orders', icon='schedule')
                gtt_tab = ui.tab('gtt', label='GTT Orders', icon='notification_important')
                auto_tab = ui.tab('auto', label='Auto Orders', icon='auto_awesome')

            with ui.tab_panels(tabs, value='regular').classes('w-full'):
                with ui.tab_panel('regular'):
                    await render_enhanced_regular_orders(fetch_api, user_storage, instruments)

                with ui.tab_panel('scheduled'):
                    await render_enhanced_scheduled_orders(fetch_api, user_storage, instruments)

                with ui.tab_panel('gtt'):
                    await render_enhanced_gtt_orders(fetch_api, user_storage, instruments)

                with ui.tab_panel('auto'):
                    await render_enhanced_auto_orders(fetch_api, user_storage, instruments)

async def render_enhanced_regular_orders(fetch_api, user_storage, instruments):
    """Enhanced regular order placement form"""
    broker = user_storage.get('default_broker', 'Zerodha')

    with ui.row().classes("w-full gap-4 p-4"):

        # Order placement form (left side)
        with ui.card().classes("dashboard-card w-1/2"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("add_shopping_cart", size="1.5rem").classes("text-green-400")
                ui.label("Place Regular Order").classes("card-title")

            ui.separator().classes("card-separator")

            # Track validation state for required fields
            validation_state = {'symbol': False, 'quantity': False, 'price': True, 'trigger_price': True}

            with ui.column().classes('w-full p-4 gap-4'):
                # Symbol selection
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Symbol").classes("text-sm text-gray-400 w-1/4")
                    symbol_options = sorted(list(instruments.keys())[:100]) if instruments else []
                    symbol_select = ui.select(
                        options=symbol_options,
                        label="Select Symbol",
                        with_input=True,
                        new_value_mode="add-unique"
                    ).classes('w-3/4').props("use-input input-debounce=300")
                    symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

                # Transaction type (Buy/Sell buttons)
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Action").classes("text-sm text-gray-400 w-1/4")
                    with ui.row().classes('w-3/4 gap-2'):
                        transaction_type = ui.toggle(['BUY', 'SELL'], value='BUY').classes('w-full')
                        transaction_type.props('toggle-color=green toggle-text-color=white')

                # Quantity
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Quantity").classes("text-sm text-gray-400 w-1/4")
                    quantity = ui.number(
                        label="Quantity",
                        value=1,
                        min=1,
                        format='%d'
                    ).classes('w-3/4')
                    quantity.on_value_change(lambda e: validation_state.update({'quantity': e.value > 0}))

                # Order type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Order Type").classes("text-sm text-gray-400 w-1/4")
                    order_type = ui.select(
                        options=['MARKET', 'LIMIT', 'SL', 'SL-M'],
                        value='MARKET',
                        label="Order Type"
                    ).classes('w-3/4')

                # Price field (conditional)
                with ui.row().classes('w-full items-center gap-4') as price_row:
                    ui.label("Price").classes("text-sm text-gray-400 w-1/4")
                    price_field = ui.number(
                        label="Price",
                        value=0,
                        min=0,
                        step=0.05,
                        format='%.2f'
                    ).classes('w-3/4')
                    price_field.on_value_change(lambda e: validation_state.update(
                        {'price': e.value > 0 if order_type.value in ['LIMIT', 'SL'] else True}))

                # Trigger price field (conditional)
                with ui.row().classes('w-full items-center gap-4') as trigger_row:
                    ui.label("Trigger Price").classes("text-sm text-gray-400 w-1/4")
                    trigger_price_field = ui.number(
                        label="Trigger Price",
                        value=0,
                        min=0,
                        step=0.05,
                        format='%.2f'
                    ).classes('w-3/4')
                    trigger_price_field.on_value_change(lambda e: validation_state.update(
                        {'trigger_price': e.value > 0 if order_type.value in ['SL', 'SL-M'] else True}))

                # Product type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Product").classes("text-sm text-gray-400 w-1/4")
                    product_type = ui.select(
                        options=['CNC', 'MIS', 'NRML'],
                        value='CNC',
                        label="Product Type"
                    ).classes('w-3/4')

                def update_price_fields():
                    """Update visibility of price fields based on order type"""
                    price_row.set_visibility(order_type.value in ['LIMIT', 'SL'])
                    trigger_row.set_visibility(order_type.value in ['SL', 'SL-M'])

                order_type.on_value_change(lambda: update_price_fields())
                update_price_fields()

                # Order summary card
                with ui.card().classes("w-full bg-gray-800/50 border border-cyan-500/30 mt-4"):
                    with ui.column().classes("p-3"):
                        ui.label("Order Summary").classes("text-sm text-gray-400 mb-2")
                        summary_label = ui.label("Select symbol and quantity").classes("text-white text-sm")

                # Place order button
                loading_container = ui.element('div')

                async def place_regular_order():
                    """Place regular order"""
                    if not all(validation_state.values()):
                        ui.notify("Please fill all required fields", type="warning")
                        return

                    order_data = {
                        "trading_symbol": symbol_select.value,
                        "transaction_type": transaction_type.value,
                        "quantity": int(quantity.value),
                        "order_type": order_type.value,
                        "product_type": product_type.value,
                        "broker": broker
                    }

                    if order_type.value in ['LIMIT', 'SL']:
                        order_data["price"] = float(price_field.value)
                    if order_type.value in ['SL', 'SL-M']:
                        order_data["trigger_price"] = float(trigger_price_field.value)

                    with loading_container:
                        loading_container.clear()
                        with ui.row().classes("items-center gap-2"):
                            ui.spinner(size="sm")
                            ui.label("Placing order...").classes("text-sm text-gray-400")

                    try:
                        response = await fetch_api(f"/orders/{broker}/place", method="POST", data=order_data)
                        if response and response.get('order_id'):
                            ui.notify(f"Order placed successfully: {response['order_id']}", type='positive')
                            # Reset form
                            quantity.value = 1
                            price_field.value = 0
                            trigger_price_field.value = 0
                        else:
                            ui.notify("Failed to place order", type='negative')
                    except Exception as e:
                        ui.notify(f"Error: {str(e)}", type='negative')
                    finally:
                        loading_container.clear()

                ui.button('Place Order',
                         on_click=lambda: asyncio.create_task(place_regular_order()),
                         icon="send",
                         color="primary").props('size=lg').classes('w-full mt-4')

        # Recent orders display (right side)
        with ui.card().classes("dashboard-card flex-1"):
            with ui.row().classes("card-header w-full justify-between items-center p-4"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("receipt_long", size="1.5rem").classes("text-orange-400")
                    ui.label("Recent Orders").classes("card-title")

                ui.button("View All", icon="open_in_new", on_click=lambda: ui.navigate.to('/order-book')).props("flat").classes("text-cyan-400")

            ui.separator().classes("card-separator")

            # Recent orders table
            recent_orders_container = ui.column().classes("w-full p-4")
            await render_recent_orders_table(fetch_api, broker, recent_orders_container)

async def render_enhanced_scheduled_orders(fetch_api, user_storage, instruments):
    """Enhanced scheduled order placement form"""
    broker = user_storage.get('default_broker', 'Zerodha')

    with ui.row().classes("w-full gap-4 p-4"):

        # Scheduled order form (left side)
        with ui.card().classes("dashboard-card w-1/2"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("schedule", size="1.5rem").classes("text-yellow-400")
                ui.label("Schedule Order").classes("card-title")

            ui.separator().classes("card-separator")

            validation_state = {'symbol': False, 'quantity': False, 'schedule_date': True, 'schedule_time': True}

            with ui.column().classes('w-full p-4 gap-4'):
                # Symbol selection
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Symbol").classes("text-sm text-gray-400 w-1/4")
                    symbol_options = sorted(list(instruments.keys())[:100]) if instruments else []
                    symbol_select = ui.select(
                        options=symbol_options,
                        label="Select Symbol",
                        with_input=True
                    ).classes('w-3/4')
                    symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

                # Transaction type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Action").classes("text-sm text-gray-400 w-1/4")
                    transaction_type = ui.toggle(['BUY', 'SELL'], value='BUY').classes('w-3/4')

                # Quantity
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Quantity").classes("text-sm text-gray-400 w-1/4")
                    quantity = ui.number("Quantity", value=1, min=1, format='%d').classes('w-3/4')
                    quantity.on_value_change(lambda e: validation_state.update({'quantity': e.value > 0}))

                # Order type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Order Type").classes("text-sm text-gray-400 w-1/4")
                    order_type = ui.select(['MARKET', 'LIMIT', 'SL', 'SL-M'], value='MARKET').classes('w-3/4')

                # Price fields (conditional based on order type)
                with ui.row().classes('w-full items-center gap-4') as price_row:
                    ui.label("Price").classes("text-sm text-gray-400 w-1/4")
                    price_field = ui.number("Price", value=0, min=0, step=0.05, format='%.2f').classes('w-3/4')

                with ui.row().classes('w-full items-center gap-4') as trigger_row:
                    ui.label("Trigger Price").classes("text-sm text-gray-400 w-1/4")
                    trigger_price_field = ui.number("Trigger Price", value=0, min=0, step=0.05, format='%.2f').classes('w-3/4')

                def update_price_fields():
                    price_row.set_visibility(order_type.value in ['LIMIT', 'SL'])
                    trigger_row.set_visibility(order_type.value in ['SL', 'SL-M'])

                order_type.on_value_change(lambda: update_price_fields())
                update_price_fields()

                # Schedule settings
                ui.label('Schedule Settings').classes('text-white font-semibold mt-4')

                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Schedule Date").classes("text-sm text-gray-400 w-1/4")
                    schedule_date = ui.date(
                        value=(datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
                    ).classes('w-3/4')

                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Schedule Time").classes("text-sm text-gray-400 w-1/4")
                    schedule_time = ui.time(value='09:30').classes('w-3/4')

                # Product type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Product").classes("text-sm text-gray-400 w-1/4")
                    product_type = ui.select(['CNC', 'MIS', 'NRML'], value='CNC').classes('w-3/4')

                loading_container = ui.element('div')

                async def schedule_order():
                    """Schedule the order"""
                    if not all(validation_state.values()):
                        ui.notify("Please fill all required fields", type="warning")
                        return

                    # Combine date and time
                    schedule_datetime = f"{schedule_date.value} {schedule_time.value}:00"

                    order_data = {
                        "trading_symbol": symbol_select.value,
                        "transaction_type": transaction_type.value,
                        "quantity": int(quantity.value),
                        "order_type": order_type.value,
                        "product_type": product_type.value,
                        "schedule_datetime": schedule_datetime,
                        "broker": broker
                    }

                    if order_type.value in ['LIMIT', 'SL']:
                        order_data["price"] = float(price_field.value)
                    if order_type.value in ['SL', 'SL-M']:
                        order_data["trigger_price"] = float(trigger_price_field.value)

                    with loading_container:
                        loading_container.clear()
                        with ui.row().classes("items-center gap-2"):
                            ui.spinner(size="sm")
                            ui.label("Scheduling order...").classes("text-sm text-gray-400")

                    try:
                        response = await fetch_api("/scheduled-orders/", method="POST", data=order_data)
                        if response and response.get('order_id'):
                            ui.notify(f"Order scheduled successfully: {response['order_id']}", type='positive')
                        else:
                            ui.notify("Failed to schedule order", type='negative')
                    except Exception as e:
                        ui.notify(f"Error: {str(e)}", type='negative')
                    finally:
                        loading_container.clear()

                ui.button('Schedule Order',
                         on_click=lambda: asyncio.create_task(schedule_order()),
                         icon="schedule_send",
                         color="warning").props('size=lg').classes('w-full mt-4')

        # Scheduled orders list (right side)
        with ui.card().classes("dashboard-card flex-1"):
            with ui.row().classes("card-header w-full justify-between items-center p-4"):
                ui.icon("schedule", size="1.5rem").classes("text-yellow-400")
                ui.label("Scheduled Orders").classes("card-title")

            ui.separator().classes("card-separator")

            scheduled_orders_container = ui.column().classes("w-full p-4")
            await render_scheduled_orders_table(fetch_api, broker, scheduled_orders_container)

async def render_enhanced_gtt_orders(fetch_api, user_storage, instruments):
    """Enhanced GTT (Good Till Triggered) order placement form"""
    broker = user_storage.get('default_broker', 'Zerodha')

    with ui.row().classes("w-full gap-4 p-4"):

        # GTT order form (left side)
        with ui.card().classes("dashboard-card w-1/2"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("notification_important", size="1.5rem").classes("text-red-400")
                ui.label("Create GTT Order").classes("card-title")

            ui.separator().classes("card-separator")

            validation_state = {'symbol': False, 'quantity': False, 'trigger_price': False, 'limit_price': False}

            with ui.column().classes('w-full p-4 gap-4'):
                # Symbol selection
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Symbol").classes("text-sm text-gray-400 w-1/4")
                    symbol_options = sorted(list(instruments.keys())[:100]) if instruments else []
                    symbol_select = ui.select(
                        options=symbol_options,
                        label="Select Symbol",
                        with_input=True
                    ).classes('w-3/4')
                    symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

                # Transaction type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Action").classes("text-sm text-gray-400 w-1/4")
                    transaction_type = ui.toggle(['BUY', 'SELL'], value='BUY').classes('w-3/4')

                # Quantity
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Quantity").classes("text-sm text-gray-400 w-1/4")
                    quantity = ui.number("Quantity", value=1, min=1, format='%d').classes('w-3/4')
                    quantity.on_value_change(lambda e: validation_state.update({'quantity': e.value > 0}))

                # Trigger type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Trigger Type").classes("text-sm text-gray-400 w-1/4")
                    trigger_type = ui.select(['single', 'two_leg'], value='single', label="Trigger Type").classes('w-3/4')

                # Trigger price
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Trigger Price").classes("text-sm text-gray-400 w-1/4")
                    trigger_price = ui.number("Trigger Price", value=0, min=0, step=0.05, format='%.2f').classes('w-3/4')
                    trigger_price.on_value_change(lambda e: validation_state.update({'trigger_price': e.value > 0}))

                # Limit price
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Limit Price").classes("text-sm text-gray-400 w-1/4")
                    limit_price = ui.number("Limit Price", value=0, min=0, step=0.05, format='%.2f').classes('w-3/4')
                    limit_price.on_value_change(lambda e: validation_state.update({'limit_price': e.value > 0}))

                # Two-leg GTT fields (conditional)
                with ui.column().classes('w-full gap-4') as two_leg_container:
                    ui.label('Second Leg Settings').classes('text-white font-semibold')

                    with ui.row().classes('w-full items-center gap-4'):
                        ui.label("Second Trigger").classes("text-sm text-gray-400 w-1/4")
                        second_trigger_price = ui.number("Second Trigger Price", value=0, min=0, step=0.05, format='%.2f').classes('w-3/4')

                    with ui.row().classes('w-full items-center gap-4'):
                        ui.label("Second Limit").classes("text-sm text-gray-400 w-1/4")
                        second_limit_price = ui.number("Second Limit Price", value=0, min=0, step=0.05, format='%.2f').classes('w-3/4')

                def update_two_leg_visibility():
                    two_leg_container.set_visibility(trigger_type.value == 'two_leg')

                trigger_type.on_value_change(lambda: update_two_leg_visibility())
                update_two_leg_visibility()

                # Product type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Product").classes("text-sm text-gray-400 w-1/4")
                    product_type = ui.select(['CNC', 'MIS', 'NRML'], value='CNC').classes('w-3/4')

                loading_container = ui.element('div')

                async def create_gtt_order():
                    """Create GTT order"""
                    if not all(validation_state.values()):
                        ui.notify("Please fill all required fields", type="warning")
                        return

                    order_data = {
                        "trading_symbol": symbol_select.value,
                        "transaction_type": transaction_type.value,
                        "quantity": int(quantity.value),
                        "trigger_type": trigger_type.value,
                        "trigger_price": float(trigger_price.value),
                        "limit_price": float(limit_price.value),
                        "product_type": product_type.value,
                        "broker": broker
                    }

                    if trigger_type.value == 'two_leg':
                        order_data.update({
                            "second_trigger_price": float(second_trigger_price.value),
                            "second_limit_price": float(second_limit_price.value)
                        })

                    with loading_container:
                        loading_container.clear()
                        with ui.row().classes("items-center gap-2"):
                            ui.spinner(size="sm")
                            ui.label("Creating GTT order...").classes("text-sm text-gray-400")

                    try:
                        response = await fetch_api("/gtt-orders/", method="POST", data=order_data)
                        if response and response.get('gtt_id'):
                            ui.notify(f"GTT order created: {response['gtt_id']}", type='positive')
                        else:
                            ui.notify("Failed to create GTT order", type='negative')
                    except Exception as e:
                        ui.notify(f"Error: {str(e)}", type='negative')
                    finally:
                        loading_container.clear()

                ui.button('Create GTT Order',
                         on_click=lambda: asyncio.create_task(create_gtt_order()),
                         icon="notification_add",
                         color="negative").props('size=lg').classes('w-full mt-4')

        # GTT orders list (right side)
        with ui.card().classes("dashboard-card flex-1"):
            with ui.row().classes("card-header w-full justify-between items-center p-4"):
                ui.icon("notification_important", size="1.5rem").classes("text-red-400")
                ui.label("Active GTT Orders").classes("card-title")

            ui.separator().classes("card-separator")

            gtt_orders_container = ui.column().classes("w-full p-4")
            await render_gtt_orders_table(fetch_api, broker, gtt_orders_container)

async def render_enhanced_auto_orders(fetch_api, user_storage, instruments):
    """Enhanced auto order creation form"""
    broker = user_storage.get('default_broker', 'Zerodha')

    with ui.row().classes("w-full gap-4 p-4"):

        # Auto order form (left side)
        with ui.card().classes("dashboard-card w-1/2"):
            with ui.row().classes("card-header w-full items-center p-4"):
                ui.icon("auto_awesome", size="1.5rem").classes("text-purple-400")
                ui.label("Create Auto Order").classes("card-title")

            ui.separator().classes("card-separator")

            with ui.column().classes('w-full p-4 gap-4'):
                # Symbol selection
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Symbol").classes("text-sm text-gray-400 w-1/4")
                    symbol_options = sorted(list(instruments.keys())[:100]) if instruments else []
                    symbol_select = ui.select(options=symbol_options, label="Select Symbol", with_input=True).classes('w-3/4')

                # Transaction type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Action").classes("text-sm text-gray-400 w-1/4")
                    transaction_type = ui.toggle(['BUY', 'SELL'], value='BUY').classes('w-3/4')

                # Risk management
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Risk Per Trade (%)").classes("text-sm text-gray-400 w-1/4")
                    risk_per_trade = ui.number("Risk %", value=2.0, min=0.1, max=10.0, step=0.1, format='%.1f').classes('w-3/4')

                # Stop loss settings
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Stop Loss Type").classes("text-sm text-gray-400 w-1/4")
                    stop_loss_type = ui.select(['percentage', 'atr', 'fixed'], value='percentage').classes('w-3/4')

                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Stop Loss Value").classes("text-sm text-gray-400 w-1/4")
                    stop_loss_value = ui.number("SL Value", value=2.0, min=0.1, step=0.1, format='%.1f').classes('w-3/4')

                # Target settings
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Target Type").classes("text-sm text-gray-400 w-1/4")
                    target_type = ui.select(['percentage', 'atr', 'fixed'], value='percentage').classes('w-3/4')

                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Target Value").classes("text-sm text-gray-400 w-1/4")
                    target_value = ui.number("Target Value", value=4.0, min=0.1, step=0.1, format='%.1f').classes('w-3/4')

                # Product type
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Product").classes("text-sm text-gray-400 w-1/4")
                    product_type = ui.select(['CNC', 'MIS', 'NRML'], value='MIS').classes('w-3/4')

                loading_container = ui.element('div')

                async def create_auto_order():
                    """Create auto order"""
                    if not symbol_select.value:
                        ui.notify("Please select a symbol", type="warning")
                        return

                    order_data = {
                        "trading_symbol": symbol_select.value,
                        "transaction_type": transaction_type.value,
                        "risk_per_trade": float(risk_per_trade.value),
                        "stop_loss_type": stop_loss_type.value,
                        "stop_loss_value": float(stop_loss_value.value),
                        "target_type": target_type.value,
                        "target_value": float(target_value.value),
                        "product_type": product_type.value,
                        "broker": broker
                    }

                    with loading_container:
                        loading_container.clear()
                        with ui.row().classes("items-center gap-2"):
                            ui.spinner(size="sm")
                            ui.label("Creating auto order...").classes("text-sm text-gray-400")

                    try:
                        response = await fetch_api("/auto-orders/", method="POST", data=order_data)
                        if response and response.get('auto_order_id'):
                            ui.notify(f"Auto order created: {response['auto_order_id']}", type='positive')
                        else:
                            ui.notify("Failed to create auto order", type='negative')
                    except Exception as e:
                        ui.notify(f"Error: {str(e)}", type='negative')
                    finally:
                        loading_container.clear()

                ui.button('Create Auto Order',
                         on_click=lambda: asyncio.create_task(create_auto_order()),
                         icon="auto_awesome",
                         color="purple").props('size=lg').classes('w-full mt-4')

        # Auto orders list (right side)
        with ui.card().classes("dashboard-card flex-1"):
            with ui.row().classes("card-header w-full justify-between items-center p-4"):
                ui.icon("auto_awesome", size="1.5rem").classes("text-purple-400")
                ui.label("Active Auto Orders").classes("card-title")

            ui.separator().classes("card-separator")

            auto_orders_container = ui.column().classes("w-full p-4")
            await render_auto_orders_table(fetch_api, broker, auto_orders_container)

# Helper functions for rendering order tables

async def render_recent_orders_table(fetch_api, broker, container):
    """Render recent orders table"""
    try:
        orders_data = await fetch_api(f"/orderbook/{broker}")

        if not orders_data:
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("inbox", size="3rem").classes("text-gray-500 mb-4")
                    ui.label("No recent orders").classes("text-lg text-gray-400 mb-2")
                    ui.label("Your orders will appear here").classes("text-sm text-gray-500")
            return

        # Take only the 5 most recent orders
        recent_orders = orders_data[:5] if len(orders_data) > 5 else orders_data

        with container:
            # Table header
            with ui.row().classes("orders-header w-full p-2 text-xs font-semibold text-gray-400 border-b border-gray-700 mb-2"):
                ui.label("Symbol").classes("w-24")
                ui.label("Side").classes("w-16")
                ui.label("Qty").classes("w-16 text-right")
                ui.label("Price").classes("w-20 text-right")
                ui.label("Status").classes("w-20")
                ui.label("Time").classes("flex-1")

            # Order rows
            for order in recent_orders:
                await render_order_row(order, fetch_api, broker)

    except Exception as e:
        logger.error(f"Error rendering recent orders: {e}")
        with container:
            ui.label("Error loading orders").classes("text-red-500 text-center p-4")

async def render_order_row(order, fetch_api, broker):
    """Render individual order row"""
    try:
        symbol = order.get('trading_symbol', 'N/A')
        transaction_type = order.get('transaction_type', 'N/A')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0)
        status = order.get('status', 'UNKNOWN').upper()
        order_time = order.get('order_timestamp', 'N/A')

        # Format time
        if order_time != 'N/A':
            try:
                if isinstance(order_time, str):
                    dt = datetime.fromisoformat(order_time.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%H:%M:%S')
                else:
                    formatted_time = str(order_time)
            except:
                formatted_time = 'N/A'
        else:
            formatted_time = 'N/A'

        # Status styling
        if status in ['COMPLETE', 'EXECUTED']:
            status_color = "text-green-400"
            border_color = "border-green-500/20"
        elif status in ['OPEN', 'PENDING']:
            status_color = "text-yellow-400"
            border_color = "border-yellow-500/20"
        else:
            status_color = "text-red-400"
            border_color = "border-red-500/20"

        # Side styling
        side_color = "text-green-400" if transaction_type.upper() == "BUY" else "text-red-400"

        with ui.row().classes(f"order-row w-full p-2 hover:bg-gray-800/50 transition-all border-l-2 {border_color} mb-1 rounded-r-lg"):
            ui.label(symbol).classes("w-24 text-white text-xs font-semibold")
            ui.label(transaction_type).classes(f"w-16 {side_color} text-xs font-semibold")
            ui.label(f"{quantity:,}").classes("w-16 text-right text-white text-xs")
            ui.label(f"₹{price:,.2f}").classes("w-20 text-right text-white text-xs text-mono")
            ui.label(status).classes(f"w-20 {status_color} text-xs")
            ui.label(formatted_time).classes("flex-1 text-gray-400 text-xs")

    except Exception as e:
        logger.error(f"Error rendering order row: {e}")

async def render_scheduled_orders_table(fetch_api, broker, container):
    """Render scheduled orders table"""
    try:
        orders_data = await fetch_api(f"/scheduled-orders/{broker}")

        if not orders_data:
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("schedule", size="3rem").classes("text-gray-500 mb-4")
                    ui.label("No scheduled orders").classes("text-lg text-gray-400 mb-2")
                    ui.label("Create scheduled orders to automate your trading").classes("text-sm text-gray-500")
            return

        with container:
            # Enhanced table with action buttons
            for order in orders_data:
                await render_scheduled_order_card(order, fetch_api, broker)

    except Exception as e:
        logger.error(f"Error rendering scheduled orders: {e}")
        with container:
            ui.label("Error loading scheduled orders").classes("text-red-500 text-center p-4")

async def render_scheduled_order_card(order, fetch_api, broker):
    """Render individual scheduled order card"""
    try:
        order_id = order.get('scheduled_order_id', 'N/A')
        symbol = order.get('trading_symbol', 'N/A')
        transaction_type = order.get('transaction_type', 'N/A')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0)
        status = order.get('status', 'UNKNOWN').upper()
        schedule_datetime = order.get('schedule_datetime', 'N/A')

        # Status styling
        status_color = "text-yellow-400" if status == "PENDING" else "text-green-400" if status == "EXECUTED" else "text-red-400"

        with ui.card().classes(f"w-full mb-2 bg-gray-800/50 border border-gray-700/50 hover:bg-gray-800/70 transition-all"):
            with ui.row().classes("p-3 items-center justify-between"):
                # Order info
                with ui.column().classes("gap-1"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"{transaction_type} {symbol}").classes("text-white font-semibold")
                        ui.chip(status, color=None).classes(f"{status_color} bg-gray-700 text-xs")
                    ui.label(f"Qty: {quantity:,} | Price: ₹{price:,.2f} | Time: {schedule_datetime}").classes("text-gray-400 text-sm")

                # Action buttons
                with ui.row().classes("gap-1"):
                    if status == "PENDING":
                        ui.button(icon="edit", on_click=lambda oid=order_id: modify_scheduled_order(oid, fetch_api, broker)).props("flat round size=sm").classes("text-cyan-400")
                        ui.button(icon="cancel", on_click=lambda oid=order_id: cancel_scheduled_order(oid, fetch_api, broker)).props("flat round size=sm").classes("text-red-400")

    except Exception as e:
        logger.error(f"Error rendering scheduled order card: {e}")

async def render_gtt_orders_table(fetch_api, broker, container):
    """Render GTT orders table"""
    try:
        orders_data = await fetch_api(f"/gtt-orders/{broker}")

        if not orders_data:
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("notification_important", size="3rem").classes("text-gray-500 mb-4")
                    ui.label("No GTT orders").classes("text-lg text-gray-400 mb-2")
                    ui.label("Create GTT orders for automated trigger-based trading").classes("text-sm text-gray-500")
            return

        with container:
            for order in orders_data:
                await render_gtt_order_card(order, fetch_api, broker)

    except Exception as e:
        logger.error(f"Error rendering GTT orders: {e}")
        with container:
            ui.label("Error loading GTT orders").classes("text-red-500 text-center p-4")

async def render_gtt_order_card(order, fetch_api, broker):
    """Render individual GTT order card"""
    try:
        gtt_id = order.get('gtt_order_id', 'N/A')
        symbol = order.get('trading_symbol', 'N/A')
        transaction_type = order.get('transaction_type', 'N/A')
        quantity = order.get('quantity', 0)
        trigger_price = order.get('trigger_price', 0)
        limit_price = order.get('limit_price', 0)
        status = order.get('status', 'UNKNOWN').upper()

        status_color = "text-yellow-400" if status == "PENDING" else "text-green-400" if status == "TRIGGERED" else "text-red-400"

        with ui.card().classes(f"w-full mb-2 bg-gray-800/50 border border-gray-700/50 hover:bg-gray-800/70 transition-all"):
            with ui.row().classes("p-3 items-center justify-between"):
                with ui.column().classes("gap-1"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"{transaction_type} {symbol}").classes("text-white font-semibold")
                        ui.chip(status, color=None).classes(f"{status_color} bg-gray-700 text-xs")
                    ui.label(f"Qty: {quantity:,} | Trigger: ₹{trigger_price:,.2f} | Limit: ₹{limit_price:,.2f}").classes("text-gray-400 text-sm")

                with ui.row().classes("gap-1"):
                    if status == "PENDING":
                        ui.button(icon="edit", on_click=lambda gid=gtt_id: modify_gtt_order(gid, fetch_api, broker)).props("flat round size=sm").classes("text-cyan-400")
                        ui.button(icon="cancel", on_click=lambda gid=gtt_id: cancel_gtt_order(gid, fetch_api, broker)).props("flat round size=sm").classes("text-red-400")

    except Exception as e:
        logger.error(f"Error rendering GTT order card: {e}")

async def render_auto_orders_table(fetch_api, broker, container):
    """Render auto orders table"""
    try:
        orders_data = await fetch_api(f"/auto-orders/{broker}")

        if not orders_data:
            with container:
                with ui.column().classes("w-full text-center p-8"):
                    ui.icon("auto_awesome", size="3rem").classes("text-gray-500 mb-4")
                    ui.label("No auto orders").classes("text-lg text-gray-400 mb-2")
                    ui.label("Create auto orders for intelligent risk-managed trading").classes("text-sm text-gray-500")
            return

        with container:
            for order in orders_data:
                await render_auto_order_card(order, fetch_api, broker)

    except Exception as e:
        logger.error(f"Error rendering auto orders: {e}")
        with container:
            ui.label("Error loading auto orders").classes("text-red-500 text-center p-4")

async def render_auto_order_card(order, fetch_api, broker):
    """Render individual auto order card"""
    try:
        auto_id = order.get('auto_order_id', 'N/A')
        symbol = order.get('trading_symbol', 'N/A')
        transaction_type = order.get('transaction_type', 'N/A')
        risk_per_trade = order.get('risk_per_trade', 0)
        stop_loss_value = order.get('stop_loss_value', 0)
        target_value = order.get('target_value', 0)
        status = order.get('status', 'UNKNOWN').upper()

        status_color = "text-green-400" if status == "ACTIVE" else "text-yellow-400" if status == "PAUSED" else "text-red-400"

        with ui.card().classes(f"w-full mb-2 bg-gray-800/50 border border-gray-700/50 hover:bg-gray-800/70 transition-all"):
            with ui.row().classes("p-3 items-center justify-between"):
                with ui.column().classes("gap-1"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"{transaction_type} {symbol}").classes("text-white font-semibold")
                        ui.chip(status, color=None).classes(f"{status_color} bg-gray-700 text-xs")
                    ui.label(f"Risk: {risk_per_trade}% | SL: {stop_loss_value}% | Target: {target_value}%").classes("text-gray-400 text-sm")

                with ui.row().classes("gap-1"):
                    if status == "ACTIVE":
                        ui.button(icon="pause", on_click=lambda aid=auto_id: pause_auto_order(aid, fetch_api, broker)).props("flat round size=sm").classes("text-yellow-400")
                    elif status == "PAUSED":
                        ui.button(icon="play_arrow", on_click=lambda aid=auto_id: resume_auto_order(aid, fetch_api, broker)).props("flat round size=sm").classes("text-green-400")

                    ui.button(icon="delete", on_click=lambda aid=auto_id: delete_auto_order(aid, fetch_api, broker)).props("flat round size=sm").classes("text-red-400")

    except Exception as e:
        logger.error(f"Error rendering auto order card: {e}")

# Action functions for order management

async def modify_scheduled_order(order_id, fetch_api, broker):
    """Modify scheduled order"""
    ui.notify(f"Opening modification for scheduled order {order_id}", type="info")
    # This would open a modification dialog

async def cancel_scheduled_order(order_id, fetch_api, broker):
    """Cancel scheduled order"""
    try:
        response = await fetch_api(f"/scheduled-orders/{order_id}", method="DELETE")
        if response and response.get("status") == "success":
            ui.notify(f"Scheduled order {order_id} cancelled", type="positive")
        else:
            ui.notify("Failed to cancel scheduled order", type="negative")
    except Exception as e:
        ui.notify(f"Error: {str(e)}", type="negative")

async def modify_gtt_order(gtt_id, fetch_api, broker):
    """Modify GTT order"""
    ui.notify(f"Opening modification for GTT order {gtt_id}", type="info")
    # This would open a modification dialog

async def cancel_gtt_order(gtt_id, fetch_api, broker):
    """Cancel GTT order"""
    try:
        response = await fetch_api(f"/gtt-orders/{gtt_id}", method="DELETE")
        if response and response.get("status") == "success":
            ui.notify(f"GTT order {gtt_id} cancelled", type="positive")
        else:
            ui.notify("Failed to cancel GTT order", type="negative")
    except Exception as e:
        ui.notify(f"Error: {str(e)}", type="negative")

async def pause_auto_order(auto_id, fetch_api, broker):
    """Pause auto order"""
    try:
        response = await fetch_api(f"/auto-orders/{auto_id}/pause", method="POST")
        if response and response.get("status") == "success":
            ui.notify(f"Auto order {auto_id} paused", type="warning")
        else:
            ui.notify("Failed to pause auto order", type="negative")
    except Exception as e:
        ui.notify(f"Error: {str(e)}", type="negative")

async def resume_auto_order(auto_id, fetch_api, broker):
    """Resume auto order"""
    try:
        response = await fetch_api(f"/auto-orders/{auto_id}/resume", method="POST")
        if response and response.get("status") == "success":
            ui.notify(f"Auto order {auto_id} resumed", type="positive")
        else:
            ui.notify("Failed to resume auto order", type="negative")
    except Exception as e:
        ui.notify(f"Error: {str(e)}", type="negative")

async def delete_auto_order(auto_id, fetch_api, broker):
    """Delete auto order"""
    try:
        response = await fetch_api(f"/auto-orders/{auto_id}", method="DELETE")
        if response and response.get("status") == "success":
            ui.notify(f"Auto order {auto_id} deleted", type="negative")
        else:
            ui.notify("Failed to delete auto order", type="negative")
    except Exception as e:
        ui.notify(f"Error: {str(e)}", type="negative")

# Compatibility function for existing imports
async def render_regular_orders(fetch_api, user_storage, instruments):
    """Compatibility function - redirects to enhanced regular orders"""
    await render_enhanced_regular_orders(fetch_api, user_storage, instruments)