"""
Order Management Module for NiceGUI Algo Trading Application
Implements regular, scheduled, and GTT orders functionality
"""

from nicegui import ui
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

# Order Management Page
async def render_order_management(fetch_api, user_storage, instruments):
    """Render the complete order management page with all order types"""
    
    broker = user_storage.get('broker', 'Zerodha')
    
    # Fetch instruments if not already available
    if not instruments:
        instruments_data = await fetch_api(f"/instruments/{broker}/?exchange=NSE")
        if instruments_data:
            # Filter for equity instruments
            equity_instruments = [i for i in instruments_data if i.get('segment') == 'NSE' and i.get('instrument_type') == 'EQ']
            instruments.update({i['trading_symbol']: i['instrument_token'] for i in equity_instruments})
    
    # Create tabs for different order types
    with ui.tabs().classes('w-full') as tabs:
        regular_tab = ui.tab('Regular Orders')
        scheduled_tab = ui.tab('Scheduled Orders')
        gtt_tab = ui.tab('GTT Orders')
        auto_tab = ui.tab('Auto Orders')
    
    with ui.tab_panels(tabs).classes('w-full'):
        with ui.tab_panel(regular_tab):
            await render_regular_orders(fetch_api, user_storage, instruments)
        
        with ui.tab_panel(scheduled_tab):
            await render_scheduled_orders(fetch_api, user_storage, instruments)
        
        with ui.tab_panel(gtt_tab):
            await render_gtt_orders(fetch_api, user_storage, instruments)
        
        with ui.tab_panel(auto_tab):
            await render_auto_orders(fetch_api, user_storage, instruments)

async def render_regular_orders(fetch_api, user_storage, instruments):
    broker = user_storage.get('broker', 'Zerodha')
    with ui.card().classes('card'):
        ui.label('Place Regular Order').classes('text-h6')

        validation_state = {'symbol': True, 'quantity': True, 'price': True, 'trigger_price': True}

        with ui.column().classes('w-full space-y-4'):
            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Index Filter").classes("text-subtitle1 w-1/4")
                index_select = ui.select(
                    options=['NIFTY 50', 'NIFTY NEXT 50', 'All Instruments'],
                    value='NIFTY 50',
                ).classes('input w-3/4')

                # Prepare the initial options for symbol_select (first 20 symbols)
                symbol_options = sorted(list(instruments.keys())[:20]) if instruments else []
                # Set the initial value to the first option, or None if no options
                initial_symbol = symbol_options[0] if symbol_options else None

                ui.label("Symbol").classes("text-subtitle1 w-1/4")
                symbol_select = ui.select(
                    options=symbol_options,
                    with_input=True,
                    value=initial_symbol
                ).classes('input w-3/4')
                symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

                async def update_symbol_options():
                    if index_select.value == 'All Instruments':
                        symbol_select.options = sorted(list(instruments.keys()))
                    else:
                        # Simplified filter for demo; in a real app, filter by index membership
                        filtered_symbols = [s for s in instruments.keys() if len(s) < 10]
                        symbol_select.options = sorted(filtered_symbols)
                    # Ensure the current value is valid; reset to first option if not
                    if symbol_select.value not in symbol_select.options:
                        symbol_select.value = symbol_select.options[0] if symbol_select.options else None
                    symbol_select.update()

                index_select.on_value_change(lambda: asyncio.create_task(update_symbol_options()))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Transaction Type").classes("text-subtitle1 w-1/4")
                transaction_type = ui.select(
                    options=['BUY', 'SELL'],
                    value='BUY',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Product Type").classes("text-subtitle1 w-1/4")
                product_type = ui.select(
                    options=['MIS', 'CNC'] if broker == 'Zerodha' else ['I', 'D'],
                    value='CNC' if broker == 'Zerodha' else 'D',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Quantity").classes("text-subtitle1 w-1/4")
                quantity = ui.number(
                    value=1,
                    min=1,
                    format='%d'
                ).classes('input w-3/4')
                quantity.on_value_change(lambda e: validation_state.update({'quantity': e.value > 0}))

            with ui.expansion('Advanced Options', icon="settings").classes("w-full"):
                with ui.column().classes('space-y-4'):
                    with ui.row().classes('w-full items-center gap-4'):
                        ui.label("Order Type").classes("text-subtitle1 w-1/4")
                        order_type = ui.select(
                            options=['MARKET', 'LIMIT', 'SL', 'SL-M'],
                            value='MARKET',
                        ).classes('input w-3/4')

                    with ui.row().classes('w-full items-center gap-4'):
                        ui.label("Price").classes("text-subtitle1 w-1/4")
                        price_field = ui.number(
                            value=0,
                            min=0,
                            step=0.05,
                            format='%.2f'
                        ).classes('input w-3/4')
                        price_field.on_value_change(lambda e: validation_state.update(
                            {'price': e.value > 0 if order_type.value in ['LIMIT', 'SL'] else True}))

                    with ui.row().classes('w-full items-center gap-4'):
                        ui.label("Trigger Price").classes("text-subtitle1 w-1/4")
                        trigger_price_field = ui.number(
                            value=0,
                            min=0,
                            step=0.05,
                            format='%.2f'
                        ).classes('input w-3/4')
                        trigger_price_field.on_value_change(lambda e: validation_state.update(
                            {'trigger_price': e.value > 0 if order_type.value in ['SL', 'SL-M'] else True}))

                    def update_price_fields():
                        price_field.visible = order_type.value in ['LIMIT', 'SL']
                        trigger_price_field.visible = order_type.value in ['SL', 'SL-M']

                    order_type.on_value_change(update_price_fields)
                    update_price_fields()

                    with ui.row().classes('w-full items-center gap-4'):
                        ui.label("Validity").classes("text-subtitle1 w-1/4")
                        validity = ui.select(
                            options=['DAY', 'IOC'],
                            value='DAY',
                        ).classes('input w-3/4')

                    with ui.row().classes('w-full items-center gap-4'):
                        ui.label("Disclosed Qty").classes("text-subtitle1 w-1/4")
                        disclosed_quantity = ui.number(
                            value=0,
                            min=0,
                            format='%d'
                        ).classes('input w-3/4')

                    with ui.row().classes('w-full items-center gap-4'):
                        ui.label("After Market Order").classes("text-subtitle1 w-1/4")
                        is_amo = ui.checkbox(text="Enable AMO").classes('w-3/4')

                    with ui.row().classes('w-full items-center gap-4'):
                        market_price_container = ui.column().classes('w-full')
                        async def fetch_market_price():
                            if symbol_select.value and symbol_select.value in instruments:
                                instrument_token = instruments[symbol_select.value]
                                ltp_data = await fetch_api(f"/ltp/Upstox?instruments={instrument_token}")
                                if ltp_data and isinstance(ltp_data, list) and ltp_data:
                                    price = ltp_data[0].get('last_price', 0)
                                    price_field.value = price
                                    trigger_price_field.value = price
                                    with market_price_container:
                                        market_price_container.clear()
                                        ui.notify(f"Market price: ₹{price:.2f}", type="info", position="top-right")
                                else:
                                    with market_price_container:
                                        market_price_container.clear()
                                        ui.notify("Failed to fetch market price.", type="warning", position="top-right")

                        ui.button('Get Market Price', on_click=lambda: asyncio.create_task(fetch_market_price())).classes(
                            'button-outline w-full')

            ui.label('Risk Management').classes('text-subtitle1 mt-4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Stop Loss (₹)").classes("text-subtitle1 w-1/4")
                stop_loss = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Target (₹)").classes("text-subtitle1 w-1/4")
                target = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')

            # Container for loading state
            loading_container = ui.column().classes('w-full')

            async def place_regular_order():
                if not all(validation_state.values()):
                    ui.notify('Please fix form errors', type='negative', position="top-right")
                    return

                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative', position="top-right")
                    return

                order_data = {
                    "trading_symbol": symbol_select.value,
                    "instrument_token": instruments[symbol_select.value],
                    "quantity": int(quantity.value),
                    "transaction_type": transaction_type.value,
                    "order_type": order_type.value,
                    "product_type": product_type.value,
                    "price": float(price_field.value) if order_type.value in ['LIMIT', 'SL'] else 0,
                    "trigger_price": float(trigger_price_field.value) if order_type.value in ['SL', 'SL-M'] else 0,
                    "validity": validity.value,
                    "disclosed_quantity": int(disclosed_quantity.value) if disclosed_quantity.value > 0 else 0,
                    "is_amo": bool(is_amo.value),
                    "stop_loss": float(stop_loss.value) if stop_loss.value > 0 else None,
                    "target": float(target.value) if target.value > 0 else None,
                    "broker": broker
                }

                with ui.dialog() as dialog, ui.card().classes("card"):
                    ui.label('Confirm Order').classes('text-h6')
                    for key, value in order_data.items():
                        if value and key not in ['instrument_token', 'broker']:
                            ui.label(f"{key.replace('_', ' ').title()}: {value}").classes("text-subtitle1")
                    with ui.row().classes("space-x-4 mt-4"):
                        ui.button('Cancel', on_click=dialog.close).classes('button-outline')

                        async def confirm_order():
                            dialog.close()
                            with loading_container:
                                loading_container.clear()
                                with ui.element().classes("relative"):
                                    ui.spinner(size="lg")
                                    ui.label("Placing order...").classes("text-subtitle1 text-gray-400 ml-2")
                                response = await fetch_api("/orders/", method="POST", data=order_data)
                                if isinstance(response, dict) and "error" in response:
                                    ui.notify(f"Failed to place order: {response['error']}", type="negative",
                                              position="top-right")
                                elif response and response.get('order_id'):
                                    ui.notify(f"Order placed successfully: {response['order_id']}", type="positive",
                                              position="top-right")
                                else:
                                    ui.notify("Failed to place order", type="negative", position="top-right")
                                loading_container.clear()

                        ui.button('Confirm', on_click=lambda: asyncio.create_task(confirm_order())).classes('button-primary')
                    dialog.open()

            ui.button('Place Order', on_click=place_regular_order).classes('button-primary')

async def render_scheduled_orders(fetch_api, user_storage, instruments):
    """Render the scheduled order placement form"""
    broker = user_storage.get('broker', 'Zerodha')

    with ui.card().classes('card'):
        ui.label('Schedule Order').classes('text-h6')

        # Track validation state for required fields
        validation_state = {'symbol': True, 'quantity': True, 'price': True, 'trigger_price': True,
                            'schedule_datetime': True}

        with ui.column().classes('w-full space-y-4'):
            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Symbol").classes("text-subtitle1 w-1/4")
                # Prepare initial options for symbol_select (first 20 symbols)
                symbol_options = sorted(list(instruments.keys())[:20]) if instruments else []
                initial_symbol = symbol_options[0] if symbol_options else None
                symbol_select = ui.select(
                    options=symbol_options,
                    with_input=True,
                    value=initial_symbol
                ).classes('input w-3/4')
                symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Transaction Type").classes("text-subtitle1 w-1/4")
                transaction_type = ui.select(
                    options=['BUY', 'SELL'],
                    value='BUY',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Product Type").classes("text-subtitle1 w-1/4")
                product_type = ui.select(
                    options=['MIS', 'CNC'] if broker == 'Zerodha' else ['I', 'D'],
                    value='CNC' if broker == 'Zerodha' else 'D',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Quantity").classes("text-subtitle1 w-1/4")
                quantity = ui.number(
                    value=1,
                    min=1,
                    format='%d'
                ).classes('input w-3/4')
                quantity.on_value_change(lambda e: validation_state.update({'quantity': e.value > 0}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Order Type").classes("text-subtitle1 w-1/4")
                order_type = ui.select(
                    options=['MARKET', 'LIMIT', 'SL', 'SL-M'],
                    value='MARKET',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Price").classes("text-subtitle1 w-1/4")
                price_field = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')
                price_field.on_value_change(lambda e: validation_state.update(
                    {'price': e.value > 0 if order_type.value in ['LIMIT', 'SL'] else True}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Trigger Price").classes("text-subtitle1 w-1/4")
                trigger_price_field = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')
                trigger_price_field.on_value_change(lambda e: validation_state.update(
                    {'trigger_price': e.value > 0 if order_type.value in ['SL', 'SL-M'] else True}))

            def update_price_fields():
                price_field.visible = order_type.value in ['LIMIT', 'SL']
                trigger_price_field.visible = order_type.value in ['SL', 'SL-M']

            order_type.on_value_change(update_price_fields)
            update_price_fields()

            ui.label('Schedule Settings').classes('text-subtitle1 mt-4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Schedule Date").classes("text-subtitle1 w-1/4")
                schedule_date = ui.date(
                    value=(datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Schedule Time").classes("text-subtitle1 w-1/4")
                schedule_time = ui.time(
                    value='09:15'
                ).classes('input w-3/4')

            ui.label('Risk Management').classes('text-subtitle1')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Stop Loss (₹)").classes("text-subtitle1 w-1/4")
                stop_loss = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Target (₹)").classes("text-subtitle1 w-1/4")
                target = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')

            # Container for market price notifications
            market_price_container = ui.column().classes('w-full')

            with ui.row().classes('w-full items-center gap-4'):
                async def fetch_market_price():
                    if symbol_select.value and symbol_select.value in instruments:
                        instrument_token = instruments[symbol_select.value]
                        ltp_data = await fetch_api(f"/ltp/Upstox?instruments={instrument_token}")
                        if ltp_data and isinstance(ltp_data, list) and ltp_data:
                            price = ltp_data[0].get('last_price', 0)
                            price_field.value = price
                            trigger_price_field.value = price
                            with market_price_container:
                                market_price_container.clear()
                                ui.notify(f"Market price: ₹{price:.2f}", type="info", position="top-right")
                        else:
                            with market_price_container:
                                market_price_container.clear()
                                ui.notify("Failed to fetch market price.", type="warning", position="top-right")

                ui.button('Get Market Price', on_click=lambda: asyncio.create_task(fetch_market_price())).classes('button-outline w-full')

            # Container for loading state
            loading_container = ui.column().classes('w-full')

            async def schedule_order():
                if not all(validation_state.values()):
                    ui.notify(f'Please fix form errors - {validation_state.values()}', type='negative', position="top-right")
                    return

                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative', position="top-right")
                    return

                # Validate inputs
                if quantity.value <= 0:
                    ui.notify('Quantity must be greater than 0', type='negative', position="top-right")
                    return

                if order_type.value in ['LIMIT', 'SL'] and price_field.value <= 0:
                    ui.notify('Price must be greater than 0 for LIMIT and SL orders', type='negative',
                              position="top-right")
                    return

                if order_type.value in ['SL', 'SL-M'] and trigger_price_field.value <= 0:
                    ui.notify('Trigger price must be greater than 0 for SL and SL-M orders', type='negative',
                              position="top-right")
                    return

                # Prepare schedule datetime
                try:
                    schedule_datetime = datetime.combine(
                        datetime.strptime(schedule_date.value, '%Y-%m-%d').date(),
                        datetime.strptime(schedule_time.value, '%H:%M').time()
                    )

                    # Validate schedule time is in the future
                    if schedule_datetime <= datetime.now():
                        ui.notify('Schedule time must be in the future', type='negative', position="top-right")
                        validation_state['schedule_datetime'] = False
                        return
                    validation_state['schedule_datetime'] = True
                except Exception as e:
                    ui.notify(f'Invalid schedule time: {str(e)}', type='negative', position="top-right")
                    validation_state['schedule_datetime'] = False
                    return

                # Prepare order data
                order_data = {
                    "trading_symbol": symbol_select.value,
                    "instrument_token": instruments[symbol_select.value],
                    "quantity": int(quantity.value),
                    "transaction_type": transaction_type.value,
                    "order_type": order_type.value,
                    "product_type": product_type.value,
                    "price": float(price_field.value) if order_type.value in ['LIMIT', 'SL'] else 0,
                    "trigger_price": float(trigger_price_field.value) if order_type.value in ['SL', 'SL-M'] else 0,
                    "schedule_datetime": schedule_datetime.isoformat(),
                    "stop_loss": float(stop_loss.value) if stop_loss.value > 0 else None,
                    "target": float(target.value) if target.value > 0 else None,
                    "broker": broker
                }

                # Show confirmation dialog
                with ui.dialog() as dialog, ui.card().classes("card"):
                    ui.label('Confirm Scheduled Order').classes('text-h6')
                    ui.label(f"Symbol: {order_data['trading_symbol']}").classes("text-subtitle1")
                    ui.label(f"Type: {order_data['transaction_type']} {order_data['order_type']}").classes(
                        "text-subtitle1")
                    ui.label(f"Quantity: {order_data['quantity']}").classes("text-subtitle1")
                    ui.label(f"Schedule: {schedule_datetime.strftime('%Y-%m-%d %H:%M')}").classes("text-subtitle1")

                    if order_data['price'] > 0:
                        ui.label(f"Price: ₹{order_data['price']:.2f}").classes("text-subtitle1")

                    if order_data['trigger_price'] > 0:
                        ui.label(f"Trigger Price: ₹{order_data['trigger_price']:.2f}").classes("text-subtitle1")

                    if order_data['stop_loss']:
                        ui.label(f"Stop Loss: ₹{order_data['stop_loss']:.2f}").classes("text-subtitle1")

                    if order_data['target']:
                        ui.label(f"Target: ₹{order_data['target']:.2f}").classes("text-subtitle1")

                    with ui.row().classes("space-x-4 mt-4"):
                        ui.button('Cancel', on_click=dialog.close).classes('button-outline')

                        async def confirm_scheduled_order():
                            dialog.close()
                            with loading_container:
                                loading_container.clear()
                                with ui.element().classes("relative"):
                                    ui.spinner(size="lg")
                                    ui.label("Scheduling order...").classes("text-subtitle1 text-gray-400 ml-2")
                                response = await fetch_api("/scheduled-orders/", method="POST", data=order_data)
                                if response and response.get('order_id'):
                                    ui.notify(f"Order scheduled successfully: {response['order_id']}", type='positive',
                                              position="top-right")
                                else:
                                    ui.notify("Failed to schedule order", type='negative', position="top-right")
                                loading_container.clear()

                        ui.button('Confirm', on_click=lambda: asyncio.create_task(confirm_scheduled_order())).classes(
                            'button-primary')

                dialog.open()

            ui.button('Schedule Order', on_click=schedule_order).classes('button-primary')

async def render_gtt_orders(fetch_api, user_storage, instruments):
    """Render the GTT (Good Till Triggered) order placement form"""
    broker = user_storage.get('broker', 'Zerodha')

    with ui.card().classes('card'):
        ui.label('Place GTT Order').classes('text-h6')

        # Track validation state for required fields
        validation_state = {'symbol': True, 'quantity': True, 'trigger_price': True, 'limit_price': True}

        with ui.column().classes('w-full space-y-4'):
            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Symbol").classes("text-subtitle1 w-1/4")
                # Prepare initial options for symbol_select (first 20 symbols)
                symbol_options = sorted(list(instruments.keys())[:20]) if instruments else []
                initial_symbol = symbol_options[0] if symbol_options else None
                symbol_select = ui.select(
                    options=symbol_options,
                    with_input=True,
                    value=initial_symbol
                ).classes('input w-3/4')
                symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Transaction Type").classes("text-subtitle1 w-1/4")
                transaction_type = ui.select(
                    options=['BUY', 'SELL'],
                    value='BUY',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Quantity").classes("text-subtitle1 w-1/4")
                quantity = ui.number(
                    value=1,
                    min=1,
                    format='%d'
                ).classes('input w-3/4')
                quantity.on_value_change(lambda e: validation_state.update({'quantity': e.value > 0}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Trigger Type").classes("text-subtitle1 w-1/4")
                trigger_type = ui.select(
                    options=['SINGLE', 'OCO'],  # SINGLE for single trigger, OCO for Order Cancels Other
                    value='SINGLE',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Trigger Price").classes("text-subtitle1 w-1/4")
                trigger_price = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')
                trigger_price.on_value_change(lambda e: validation_state.update({'trigger_price': e.value > 0}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Limit Price").classes("text-subtitle1 w-1/4")
                limit_price = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')
                limit_price.on_value_change(lambda e: validation_state.update({'limit_price': e.value > 0}))

            # OCO-specific fields (visible only if trigger_type is OCO)
            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Second Trigger Price").classes("text-subtitle1 w-1/4")
                second_trigger_price = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Second Limit Price").classes("text-subtitle1 w-1/4")
                second_limit_price = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')

            def update_trigger_fields():
                is_oco = trigger_type.value == 'OCO'
                second_trigger_price.visible = is_oco
                second_limit_price.visible = is_oco

            trigger_type.on_value_change(update_trigger_fields)
            update_trigger_fields()

            # Container for market price notifications
            market_price_container = ui.column().classes('w-full')

            with ui.row().classes('w-full items-center gap-4'):
                async def fetch_market_price():
                    if symbol_select.value and symbol_select.value in instruments:
                        instrument_token = instruments[symbol_select.value]
                        ltp_data = await fetch_api(f"/ltp/Upstox?instruments={instrument_token}")
                        if ltp_data and isinstance(ltp_data, list) and ltp_data:
                            price = ltp_data[0].get('last_price', 0)
                            trigger_price.value = price
                            limit_price.value = price
                            if trigger_type.value == 'OCO':
                                second_trigger_price.value = price
                                second_limit_price.value = price
                            with market_price_container:
                                market_price_container.clear()
                                ui.notify(f"Market price: ₹{price:.2f}", type="info", position="top-right")
                        else:
                            with market_price_container:
                                market_price_container.clear()
                                ui.notify("Failed to fetch market price.", type="warning", position="top-right")

                ui.button('Get Market Price', on_click=lambda: asyncio.create_task(fetch_market_price())).classes(
                    'button-outline w-full')

            # Container for loading state
            loading_container = ui.column().classes('w-full')

            async def place_gtt_order():
                if not all(validation_state.values()):
                    ui.notify('Please fix form errors', type='negative', position="top-right")
                    return

                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative', position="top-right")
                    return

                # Validate inputs
                if quantity.value <= 0:
                    ui.notify('Quantity must be greater than 0', type='negative', position="top-right")
                    return

                if trigger_price.value <= 0:
                    ui.notify('Trigger price must be greater than 0', type='negative', position="top-right")
                    return

                if limit_price.value <= 0:
                    ui.notify('Limit price must be greater than 0', type='negative', position="top-right")
                    return

                if trigger_type.value == 'OCO':
                    if second_trigger_price.value <= 0:
                        ui.notify('Second trigger price must be greater than 0 for OCO orders', type='negative',
                                  position="top-right")
                        return
                    if second_limit_price.value <= 0:
                        ui.notify('Second limit price must be greater than 0 for OCO orders', type='negative',
                                  position="top-right")
                        return

                # Prepare GTT order data (excluding last_price for now; will fetch it in confirm_gtt_order)
                order_data = {
                    "instrument_token": instruments[symbol_select.value],
                    "trading_symbol": symbol_select.value,
                    "transaction_type": transaction_type.value,
                    "quantity": int(quantity.value),
                    "trigger_type": "single" if trigger_type.value == "SINGLE" else "two-leg",
                    "trigger_price": float(trigger_price.value),
                    "limit_price": float(limit_price.value),
                    "second_trigger_price": float(second_trigger_price.value) if trigger_type.value == 'OCO' else None,
                    "second_limit_price": float(second_limit_price.value) if trigger_type.value == 'OCO' else None,
                    "broker": broker
                }

                # Show confirmation dialog
                with ui.dialog() as dialog, ui.card().classes("card"):
                    ui.label('Confirm GTT Order').classes('text-h6')
                    ui.label(f"Symbol: {order_data['trading_symbol']}").classes("text-subtitle1")
                    ui.label(f"Type: {order_data['transaction_type']} {order_data['trigger_type']}").classes(
                        "text-subtitle1")
                    ui.label(f"Quantity: {order_data['quantity']}").classes("text-subtitle1")
                    ui.label(f"Trigger Price: ₹{order_data['trigger_price']:.2f}").classes("text-subtitle1")
                    ui.label(f"Limit Price: ₹{order_data['limit_price']:.2f}").classes("text-subtitle1")

                    if order_data['trigger_type'] == 'OCO':
                        ui.label(f"Second Trigger Price: ₹{order_data['second_trigger_price']:.2f}").classes(
                            "text-subtitle1")
                        ui.label(f"Second Limit Price: ₹{order_data['second_limit_price']:.2f}").classes(
                            "text-subtitle1")

                    with ui.row().classes("space-x-4 mt-4"):
                        ui.button('Cancel', on_click=dialog.close).classes('button-outline')

                        async def confirm_gtt_order():
                            dialog.close()
                            with loading_container:
                                loading_container.clear()
                                with ui.element().classes("relative"):
                                    ui.spinner(size="lg")
                                    ui.label("Placing GTT order...").classes("text-subtitle1 text-gray-400 ml-2")

                                # Fetch the latest last_price before placing the order
                                last_price = 0
                                if symbol_select.value and symbol_select.value in instruments:
                                    instrument_token = instruments[symbol_select.value]
                                    ltp_data = await fetch_api(f"/ltp/Upstox?instruments={instrument_token}")
                                    if ltp_data and isinstance(ltp_data, list) and ltp_data:
                                        last_price = ltp_data[0].get('last_price', 0)
                                    else:
                                        ui.notify("Failed to fetch last price. Using default value of 0.",
                                                  type="warning", position="top-right")

                                # Update order_data with the latest last_price
                                order_data["last_price"] = float(last_price)

                                response = await fetch_api("/gtt-orders/", method="POST", data=order_data)
                                print(response)
                                if response and response.get('gtt_id'):
                                    ui.notify(f"GTT order placed successfully: {response['gtt_id']}",
                                              type='positive', position="top-right")
                                else:
                                    ui.notify("Failed to place GTT order", type='negative', position="top-right")
                                loading_container.clear()

                        ui.button('Confirm', on_click=lambda: asyncio.create_task(confirm_gtt_order())).classes(
                            'button-primary')

                dialog.open()

            ui.button('Place GTT Order', on_click=place_gtt_order).classes('button-primary')

async def render_auto_orders(fetch_api, user_storage, instruments):
    """Render the auto order placement form for algorithmic trading"""
    broker = user_storage.get('broker', 'Zerodha')

    with ui.card().classes('card'):
        ui.label('Auto Orders').classes('text-h6')
        ui.label('Set up automated orders based on risk parameters').classes('text-subtitle1 text-gray-400 mb-4')

        # Track validation state for required fields
        validation_state = {'symbol': True, 'risk_per_trade': True, 'stop_loss_value': True, 'target_value': True,
                            'limit_price': True, 'atr_period': True, 'check_interval': True}

        with ui.column().classes('w-full space-y-4'):
            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Symbol").classes("text-subtitle1 w-1/4")
                # Prepare initial options for symbol_select (first 20 symbols)
                symbol_options = sorted(list(instruments.keys())[:20]) if instruments else []
                initial_symbol = symbol_options[0] if symbol_options else None
                symbol_select = ui.select(
                    options=symbol_options,
                    with_input=True,
                    value=initial_symbol
                ).classes('input w-3/4')
                symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Transaction Type").classes("text-subtitle1 w-1/4")
                transaction_type = ui.select(
                    options=['BUY', 'SELL'],
                    value='BUY',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Order Type").classes("text-subtitle1 w-1/4")
                order_type = ui.select(
                    options=['MARKET', 'LIMIT'],
                    value='MARKET',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Product Type").classes("text-subtitle1 w-1/4")
                product_type = ui.select(
                    options=['MIS', 'CNC'] if broker == 'Zerodha' else ['I', 'D'],
                    value='CNC' if broker == 'Zerodha' else 'D',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Limit Price").classes("text-subtitle1 w-1/4")
                limit_price = ui.number(
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('input w-3/4')
                limit_price.on_value_change(lambda e: validation_state.update(
                    {'limit_price': e.value > 0 if order_type.value == 'LIMIT' else True}))

                def update_limit_price():
                    limit_price.visible = order_type.value == 'LIMIT'

                order_type.on_value_change(update_limit_price)
                update_limit_price()

                # Container for market price notifications
                market_price_container = ui.column().classes('w-full')

                async def fetch_market_price():
                    if symbol_select.value and symbol_select.value in instruments:
                        instrument_token = instruments[symbol_select.value]
                        ltp_data = await fetch_api(f"/ltp/Upstox?instruments={instrument_token}")
                        if ltp_data and isinstance(ltp_data, list) and ltp_data:
                            price = ltp_data[0].get('last_price', 0)
                            limit_price.value = price
                            with market_price_container:
                                market_price_container.clear()
                                ui.notify(f"Market price: ₹{price:.2f}", type="info", position="top-right")
                            return price
                        else:
                            with market_price_container:
                                market_price_container.clear()
                                ui.notify("Failed to fetch market price.", type="warning", position="top-right")
                            return 0
                    return None

                ui.button('Get Market Price', on_click=lambda: asyncio.create_task(fetch_market_price())).classes('button-outline w-full')

            ui.label('Risk Management').classes('text-subtitle1 mt-4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Risk per Trade (%)").classes("text-subtitle1 w-1/4")
                risk_per_trade = ui.number(
                    value=1.0,
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    format='%.1f'
                ).classes('input w-3/4')
                risk_per_trade.on_value_change(lambda e: validation_state.update({'risk_per_trade': e.value > 0}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Stop Loss Type").classes("text-subtitle1 w-1/4")
                stop_loss_type = ui.select(
                    options=['Fixed Amount', 'Percentage of Entry', 'ATR Based'],
                    value='Fixed Amount',
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Stop Loss Value (₹)").classes("text-subtitle1 w-1/4")
                stop_loss_value = ui.number(
                    value=1.0,
                    min=0.1,
                    step=0.1,
                    format='%.1f'
                ).classes('input w-3/4')
                stop_loss_value.on_value_change(lambda e: validation_state.update({'stop_loss_value': e.value > 0}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Target Value (₹)").classes("text-subtitle1 w-1/4")
                target_value = ui.number(
                    value=2.0,
                    min=0.1,
                    step=0.1,
                    format='%.1f'
                ).classes('input w-3/4')
                target_value.on_value_change(lambda e: validation_state.update({'target_value': e.value > 0}))

                def update_stop_loss_labels():
                    if stop_loss_type.value == 'Fixed Amount':
                        stop_loss_value.label = 'Stop Loss Value (₹)'
                        target_value.label = 'Target Value (₹)'
                    elif stop_loss_type.value == 'Percentage of Entry':
                        stop_loss_value.label = 'Stop Loss (%)'
                        target_value.label = 'Target (%)'
                    else:  # ATR Based
                        stop_loss_value.label = 'Stop Loss (ATR Multiple)'
                        target_value.label = 'Target (ATR Multiple)'

                stop_loss_type.on_value_change(update_stop_loss_labels)
                update_stop_loss_labels()

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("ATR Period").classes("text-subtitle1 w-1/4")
                atr_period = ui.number(
                    value=14,
                    min=5,
                    max=50,
                    format='%d'
                ).classes('input w-3/4')
                atr_period.on_value_change(lambda e: validation_state.update({'atr_period': e.value >= 5}))

                def update_atr_field():
                    atr_period.visible = stop_loss_type.value == 'ATR Based'

                stop_loss_type.on_value_change(update_atr_field)
                update_atr_field()

            ui.label('Execution Settings').classes('text-subtitle1 mt-4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Execution Type").classes("text-subtitle1 w-1/4")
                execution_type = ui.radio(['Manual', 'Automatic'], value='Manual').classes('w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Check Interval (minutes)").classes("text-subtitle1 w-1/4")
                check_interval = ui.number(
                    value=5,
                    min=1,
                    max=60,
                    format='%d'
                ).classes('input w-3/4')
                check_interval.on_value_change(lambda e: validation_state.update({'check_interval': e.value >= 1}))

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Market Start Time").classes("text-subtitle1 w-1/4")
                market_start_time = ui.time(
                    value='09:15'
                ).classes('input w-3/4')

            with ui.row().classes('w-full items-center gap-4'):
                ui.label("Market End Time").classes("text-subtitle1 w-1/4")
                market_end_time = ui.time(
                    value='15:30'
                ).classes('input w-3/4')

                def update_execution_fields():
                    is_automatic = execution_type.value == 'Automatic'
                    check_interval.visible = is_automatic
                    market_start_time.visible = is_automatic
                    market_end_time.visible = is_automatic

                execution_type.on_value_change(update_execution_fields)
                update_execution_fields()

            # Container for loading state
            loading_container = ui.column().classes('w-full')

            async def place_auto_order():
                if not all(validation_state.values()):
                    ui.notify('Please fix form errors', type='negative', position="top-right")
                    return

                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative', position="top-right")
                    return

                # Validate inputs
                if risk_per_trade.value <= 0:
                    ui.notify('Risk per trade must be greater than 0', type='negative', position="top-right")
                    return

                if stop_loss_value.value <= 0 or target_value.value <= 0:
                    ui.notify('Stop loss and target values must be greater than 0', type='negative', position="top-right")
                    return

                if order_type.value == 'LIMIT' and limit_price.value <= 0:
                    ui.notify('Limit price must be greater than 0', type='negative', position="top-right")
                    return

                if stop_loss_type.value == 'ATR Based' and atr_period.value < 5:
                    ui.notify('ATR period must be at least 5', type='negative', position="top-right")
                    return

                if execution_type.value == 'Automatic' and check_interval.value < 1:
                    ui.notify('Check interval must be at least 1 minute', type='negative', position="top-right")
                    return

                # Prepare order data
                order_data = {
                    "trading_symbol": symbol_select.value,
                    "instrument_token": instruments[symbol_select.value],
                    "transaction_type": transaction_type.value,
                    "order_type": order_type.value,
                    "product_type": product_type.value,
                    "limit_price": float(limit_price.value) if order_type.value == 'LIMIT' else 0,
                    "risk_per_trade": float(risk_per_trade.value),
                    "stop_loss_type": stop_loss_type.value,
                    "stop_loss_value": float(stop_loss_value.value),
                    "target_value": float(target_value.value),
                    "execution_type": execution_type.value,
                    "broker": broker
                }

                if stop_loss_type.value == 'ATR Based':
                    order_data["atr_period"] = int(atr_period.value)

                if execution_type.value == 'Automatic':
                    order_data["check_interval"] = int(check_interval.value)
                    order_data["market_start_time"] = market_start_time.value
                    order_data["market_end_time"] = market_end_time.value

                # Show confirmation dialog
                with ui.dialog() as dialog, ui.card().classes("card"):
                    ui.label('Confirm Auto Order').classes('text-h6')
                    ui.label(f"Symbol: {order_data['trading_symbol']}").classes("text-subtitle1")
                    ui.label(f"Type: {order_data['transaction_type']} {order_data['order_type']}").classes("text-subtitle1")
                    ui.label(f"Risk per Trade: {order_data['risk_per_trade']}%").classes("text-subtitle1")
                    ui.label(f"Stop Loss Type: {order_data['stop_loss_type']}").classes("text-subtitle1")
                    ui.label(f"Stop Loss Value: {order_data['stop_loss_value']}").classes("text-subtitle1")
                    ui.label(f"Target Value: {order_data['target_value']}").classes("text-subtitle1")
                    ui.label(f"Execution: {order_data['execution_type']}").classes("text-subtitle1")

                    if order_data['order_type'] == 'LIMIT':
                        ui.label(f"Limit Price: ₹{order_data['limit_price']:.2f}").classes("text-subtitle1")

                    if order_data['execution_type'] == 'Automatic':
                        ui.label(f"Check Interval: {order_data['check_interval']} minutes").classes("text-subtitle1")
                        ui.label(f"Market Hours: {order_data['market_start_time']} - {order_data['market_end_time']}").classes("text-subtitle1")

                    with ui.row().classes("space-x-4 mt-4"):
                        ui.button('Cancel', on_click=dialog.close).classes('button-outline')

                        async def confirm_auto_order():
                            dialog.close()
                            with loading_container:
                                loading_container.clear()
                                with ui.element().classes("relative"):
                                    ui.spinner(size="lg")
                                    ui.label("Setting up auto order...").classes("text-subtitle1 text-gray-400 ml-2")
                                response = await fetch_api("/auto-orders/", method="POST", data=order_data)
                                if response and response.get('auto_order_id'):
                                    ui.notify(f"Auto order set up successfully: {response['auto_order_id']}", type='positive', position="top-right")
                                else:
                                    ui.notify("Failed to set up auto order", type='negative', position="top-right")
                                loading_container.clear()

                        ui.button('Confirm', on_click=lambda: asyncio.create_task(confirm_auto_order())).classes('button-primary')

                dialog.open()

            ui.button('Set Up Auto Order', on_click=place_auto_order).classes('button-primary mt-4')

        # Display active auto orders
        ui.separator()
        ui.label('Active Auto Orders').classes('text-subtitle1 mt-4')

        auto_orders_grid = ui.aggrid({
            'columnDefs': [
                {'headerName': 'Auto Order ID', 'field': 'auto_order_id'},
                {'headerName': 'Symbol', 'field': 'trading_symbol'},
                {'headerName': 'Type', 'field': 'transaction_type'},
                {'headerName': 'Risk (%)', 'field': 'risk_per_trade'},
                {'headerName': 'Stop Loss Type', 'field': 'stop_loss_type'},
                {'headerName': 'Execution', 'field': 'execution_type'},
                {'headerName': 'Status', 'field': 'status'}
            ],
            'rowData': [],
            'rowSelection': 'single',
            'pagination': True,
            'paginationPageSize': 10
        }).classes('w-full mt-4')

        # Container for loading state in cancel_auto_order
        cancel_loading_container = ui.column().classes('w-full')

        async def fetch_auto_orders():
            auto_orders = await fetch_api(f"/auto-orders/{broker}")
            if auto_orders and isinstance(auto_orders, list):
                auto_orders_grid.options['rowData'] = auto_orders
                auto_orders_grid.update()
            else:
                auto_orders_grid.options['rowData'] = []
                auto_orders_grid.update()

        await fetch_auto_orders()

        with ui.row().classes('w-full mt-4'):
            ui.button('Refresh', on_click=lambda: asyncio.create_task(fetch_auto_orders())).classes('button-outline')

            async def cancel_auto_order():
                selected_rows = await auto_orders_grid.get_selected_rows()
                if not selected_rows:
                    ui.notify('Please select an auto order to cancel', type='warning', position="top-right")
                    return

                auto_order_id = selected_rows[0].get('auto_order_id')
                if not auto_order_id:
                    ui.notify('Invalid order selection', type='negative', position="top-right")
                    return

                with cancel_loading_container:
                    cancel_loading_container.clear()
                    with ui.element().classes("relative"):
                        ui.spinner(size="lg")
                        ui.label("Cancelling auto order...").classes("text-subtitle1 text-gray-400 ml-2")
                    response = await fetch_api(f"/auto-orders/{auto_order_id}", method="DELETE")
                    if response and response.get('success'):
                        ui.notify('Auto order cancelled successfully', type='positive', position="top-right")
                        await fetch_auto_orders()  # Refresh the list
                    else:
                        ui.notify('Failed to cancel auto order', type='negative', position="top-right")
                    cancel_loading_container.clear()

            ui.button('Cancel Selected', on_click=lambda: asyncio.create_task(cancel_auto_order())).classes('button-danger')