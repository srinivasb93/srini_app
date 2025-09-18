"""
Enhanced Order Management Module for NiceGUI Algo Trading Application
Complete implementation with all references properly resolved
"""

from nicegui import ui
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import json
import logging
import os, sys
from sqlalchemy.sql import text

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.database import get_db, get_nsedata_db
from common_utils.db_utils import async_fetch_query

logger = logging.getLogger(__name__)

async def get_symbol_options(index_value):
    if 'NIFTY' in str(index_value):
        try:
            nsedata_db_gen = get_nsedata_db()
            nsedata_db = await nsedata_db_gen.__anext__()
            query = f'SELECT * FROM "{str(index_value)}_REF"'
            symbols_list = await async_fetch_query(nsedata_db, text(query), {})
            return sorted([symbol['Symbol'] for symbol in symbols_list[1:]])
        except Exception:
            return []
    else:
        return []

async def render_order_management(fetch_api, user_storage, instruments):
    """Enhanced order management page with compact header and full-width forms"""

    broker = user_storage.get('default_broker', 'Zerodha')  # Changed from 'broker' to 'default_broker'

    # Fetch instruments if not already available
    if not instruments:
        instruments_data = await fetch_api(f"/instruments/{broker}/?exchange=NSE")
        if instruments_data:
            equity_instruments = [i for i in instruments_data if i.get('segment') == 'NSE' and i.get('instrument_type') == 'EQ']
            instruments.update({i['trading_symbol']: i['instrument_token'] for i in equity_instruments})

    # Compact header
    with ui.row().classes("w-full items-center justify-between p-2 bg-gray-900/50"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("shopping_cart", size="1.5rem").classes("text-cyan-400")
            ui.label("Order Management").classes("text-2xl font-bold text-white")
            ui.chip("TRADING", color="cyan").classes("text-xs")

        with ui.row().classes("items-center gap-2"):
            with ui.tabs().classes('bg-gray-800/50 rounded-lg') as tabs:
                regular_tab = ui.tab('Regular Orders', icon='flash_on').classes('px-4 py-2')
                scheduled_tab = ui.tab('Scheduled Orders', icon='schedule').classes('px-4 py-2')
                gtt_tab = ui.tab('GTT Orders', icon='compare_arrows').classes('px-4 py-2')
                auto_tab = ui.tab('Auto Orders', icon='smart_toy').classes('px-4 py-2')

        # Tab panels with lazy loading
        with ui.tab_panels(tabs).classes('w-full p-.5'):
            with ui.tab_panel(regular_tab):
                await render_regular_orders(fetch_api, user_storage, instruments, broker)

            with ui.tab_panel(scheduled_tab):
                scheduled_container = ui.column().classes('w-full')
                scheduled_loaded = False

                async def load_scheduled_orders():
                    nonlocal scheduled_loaded
                    if not scheduled_loaded:
                        scheduled_container.clear()
                        with scheduled_container:
                            await render_scheduled_orders(fetch_api, user_storage, instruments, broker)
                        scheduled_loaded = True

                scheduled_tab.on('click', lambda: asyncio.create_task(load_scheduled_orders()))

            with ui.tab_panel(gtt_tab):
                gtt_container = ui.column().classes('w-full')
                gtt_loaded = False

                async def load_gtt_orders():
                    nonlocal gtt_loaded
                    if not gtt_loaded:
                        gtt_container.clear()
                        with gtt_container:
                            await render_gtt_orders(fetch_api, user_storage, instruments, broker)
                        gtt_loaded = True

                gtt_tab.on('click', lambda: asyncio.create_task(load_gtt_orders()))

            with ui.tab_panel(auto_tab):
                auto_container = ui.column().classes('w-full')
                auto_loaded = False

                async def load_auto_orders():
                    nonlocal auto_loaded
                    if not auto_loaded:
                        auto_container.clear()
                        with auto_container:
                            await render_auto_orders(fetch_api, user_storage, instruments, broker)
                        auto_loaded = True

                auto_tab.on('click', lambda: asyncio.create_task(load_auto_orders()))

async def render_regular_orders(fetch_api, user_storage, instruments, broker):
    """Regular orders form with stop_loss and target fields"""

    with ui.card().classes('w-full enhanced-card'):
        with ui.row().classes("w-full items-center justify-between p-2 border-b border-gray-700"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("add_circle", size="1.2rem").classes("text-cyan-400")
                ui.label("Place Regular Order").classes("text-lg font-semibold text-white")
            ui.chip("LIVE", color="green").classes("text-xs")

        with ui.column().classes("p-2 gap-3 w-full"):
            validation_state = {'symbol': True, 'quantity': True, 'price': True, 'trigger_price': True}

            # Index Filter and Symbol Selection
            with ui.row().classes('w-full gap-3'):
                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Index Filter").classes("text-sm font-medium text-gray-300")
                    index_select = ui.select(
                        options=['NIFTY_50', 'NIFTY_NEXT_50', 'All Instruments'],
                        value='NIFTY_50'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Trading Symbol").classes("text-sm font-medium text-gray-300")

                    # Prepare initial symbol options
                    symbol_options = await get_symbol_options(index_select.value)
                    initial_symbol = symbol_options[0] if symbol_options else None

                    symbol_select = ui.select(
                        options=symbol_options,
                        with_input=True,
                        value=initial_symbol
                    ).classes('w-full')
                    symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

                    async def update_symbol_options():
                        if index_select.value == 'All Instruments':
                            symbol_select.options = sorted(list(instruments.keys()))
                        else:
                            symbol_select.options = await get_symbol_options(index_select.value)
                        symbol_select.update()

                    index_select.on_value_change(update_symbol_options)

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Transaction Type").classes("text-sm font-medium text-gray-300")
                    transaction_type = ui.select(
                        options=['BUY', 'SELL'],
                        value='BUY'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Product Type").classes("text-sm font-medium text-gray-300")
                    product_type = ui.select(
                        options=['MIS', 'CNC'] if broker == 'Zerodha' else ['I', 'D'],
                        value='CNC' if broker == 'Zerodha' else 'D'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Quantity").classes("text-sm font-medium text-gray-300")
                    quantity = ui.number(
                        value=1,
                        min=1,
                        format='%d'
                    ).classes('w-full')
                    quantity.on_value_change(lambda e: validation_state.update({'quantity': e.value > 0}))

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Disclosed Qty").classes("text-sm font-medium text-gray-300")
                    disclosed_quantity = ui.number(
                        value=0,
                        min=0,
                        format='%d'
                    ).classes('w-full')

            # Price Configuration Row
            with ui.row().classes('w-full gap-3'):
                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Validity").classes("text-sm font-medium text-gray-300")
                    validity = ui.select(
                        options=['DAY', 'IOC'],
                        value='DAY'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Order Type").classes("text-sm font-medium text-gray-300")
                    order_type = ui.select(
                        options=['MARKET', 'LIMIT', 'SL', 'SL-M'],
                        value='MARKET'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Price").classes("text-sm font-medium text-gray-300")
                    price_field = ui.number(
                        value=0,
                        min=0,
                        step=0.05,
                        format='%.2f'
                    ).classes('w-full')
                    price_field.on_value_change(lambda e: validation_state.update(
                        {'price': e.value > 0 if order_type.value in ['LIMIT', 'SL'] else True}))

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Trigger Price").classes("text-sm font-medium text-gray-300")
                    trigger_price_field = ui.number(
                        value=0,
                        min=0,
                        step=0.05,
                        format='%.2f'
                    ).classes('w-full')
                    trigger_price_field.on_value_change(lambda e: validation_state.update(
                        {'trigger_price': e.value > 0 if order_type.value in ['SL', 'SL-M'] else True}))

                with ui.column().classes("flex-1"):
                    ui.label("Market Price").classes("text-sm font-medium text-gray-300")
                    market_price_label = ui.badge("").classes("text-sm font-medium text-gray-300")

                with ui.column().classes('flex-1'):
                    ui.label("AMO").classes("text-sm font-medium text-gray-300")
                    is_amo_checkbox = ui.checkbox('After Market Order (AMO)').classes('text-white')
                    is_amo_checkbox.value = False


            # Risk Management Section
            ui.label('Risk Management').classes('text-lg font-medium text-white mt-1 mb-1')

            # Risk Management Type Toggle
            with ui.row().classes('w-full gap-3 mb-2'):
                risk_management_type = ui.toggle({
                    'regular': 'Regular SL/Target',
                    'trailing': 'Trailing Stop Loss'
                }, value='regular').classes('bg-gray-700')

            # Regular Risk Management Section
            regular_risk_section = ui.column().classes('w-full')
            with regular_risk_section:
                with ui.row().classes('w-half gap-3'):
                    with ui.column().classes("flex-1 gap-2"):
                        ui.label("Stop Loss (₹)").classes("text-sm font-medium text-gray-300")
                        regular_stop_loss = ui.number(
                            value=0,
                            min=0,
                            step=0.05,
                            format='%.2f'
                        ).classes('w-full')

                    with ui.column().classes("flex-1 gap-2"):
                        ui.label("Target (₹)").classes("text-sm font-medium text-gray-300")
                        regular_target = ui.number(
                            value=0,
                            min=0,
                            step=0.05,
                            format='%.2f'
                        ).classes('w-full')

            # Trailing Stop Loss Section
            trailing_risk_section = ui.column().classes('w-full')
            trailing_risk_section.visible = False
            with trailing_risk_section:
                with ui.row().classes('w-full gap-3'):
                    with ui.column().classes("flex-1 gap-2"):
                        ui.label("Stop Loss (₹)").classes("text-sm font-medium text-gray-300")
                        ui.label("Initial stop loss when order is placed").classes("text-xs text-gray-400")
                        trailing_stop_loss = ui.number(
                            value=0,
                            min=0,
                            step=0.05,
                            format='%.2f'
                        ).classes('w-full')

                    with ui.column().classes("flex-1 gap-2"):
                        ui.label("Target (₹)").classes("text-sm font-medium text-gray-300")
                        ui.label("Fixed Target if required else leave it at 0.0").classes("text-xs text-gray-400")
                        trailing_target = ui.number(
                            value=0,
                            min=0,
                            step=0.05,
                            format='%.2f'
                        ).classes('w-full')

                    with ui.column().classes("flex-1 gap-2"):
                        ui.label("Trail Start Target (%)").classes("text-sm font-medium text-gray-300")
                        ui.label("Percentage gain to activate trailing").classes("text-xs text-gray-400")
                        trail_start_target_percent = ui.number(
                            value=5.0,
                            min=0.1,
                            max=50.0,
                            step=0.1,
                            format='%.1f'
                        ).classes('w-full')

                    with ui.column().classes("flex-1 gap-2"):
                        ui.label("Trailing Stop Loss (%)").classes("text-sm font-medium text-gray-300")
                        ui.label("Percentage below highest price").classes("text-xs text-gray-400")
                        trailing_stop_loss_percent = ui.number(
                            value=2.0,
                            min=0.1,
                            max=20.0,
                            step=0.1,
                            format='%.1f'
                        ).classes('w-full')

                # Info section for trailing stop loss
                with ui.card().classes('bg-blue-900 border-blue-600 p-2 mt-2'):
                    ui.icon('info').classes('text-blue-400')
                    ui.label('Trailing Stop Loss Info:').classes('font-semibold text-blue-100')
                    ui.label('• Activates when price reaches the trail start target percentage').classes('text-sm text-blue-200')
                    ui.label('• Automatically adjusts stop loss as price moves in your favor').classes('text-sm text-blue-200')
                    ui.label('• Triggers exit when price drops by the trailing stop loss percentage').classes('text-sm text-blue-200')
                    ui.label('• Checked every 5 minutes during market hours').classes('text-sm text-blue-200')

            # Market price update functions
            async def update_market_price(symbol):
                instrument_token = instruments.get(symbol)
                if instrument_token:
                    market_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
                    if market_data:
                        last_price = market_data[0]['last_price']
                        market_price_label.text = str(last_price)
                    else:
                        market_price_label.text = "0"
                else:
                    market_price_label.text = "0"
                market_price_label.update()

            # Toggle between risk management types
            def toggle_risk_management():
                if risk_management_type.value == 'regular':
                    regular_risk_section.visible = True
                    trailing_risk_section.visible = False
                else:
                    regular_risk_section.visible = False
                    trailing_risk_section.visible = True

            risk_management_type.on_value_change(toggle_risk_management)

            # Initial fetch
            await update_market_price(symbol_select.value)
            symbol_select.on_value_change(lambda e: asyncio.create_task(update_market_price(e.value)))

            def update_price_fields():
                price_field.visible = order_type.value in ['LIMIT', 'SL']
                trigger_price_field.visible = order_type.value in ['SL', 'SL-M']

            order_type.on_value_change(update_price_fields)
            update_price_fields()

            # Loading container
            loading_container = ui.column().classes('w-full mt-4')

            # Place Order Action
            async def place_regular_order():
                if not all(validation_state.values()):
                    ui.notify('Please fix form errors', type='negative')
                    return

                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative')
                    return

                # Validation
                if quantity.value <= 0:
                    ui.notify('Quantity must be greater than 0', type='negative')
                    return

                if order_type.value in ['LIMIT', 'SL'] and price_field.value <= 0:
                    ui.notify('Price must be greater than 0 for LIMIT and SL orders', type='negative')
                    return

                if order_type.value in ['SL', 'SL-M'] and trigger_price_field.value <= 0:
                    ui.notify('Trigger price must be greater than 0 for SL and SL-M orders', type='negative')
                    return

                # Validate trailing stop loss parameters
                if risk_management_type.value == 'trailing':
                    if trail_start_target_percent.value <= 0:
                        ui.notify('Trail start target percentage must be greater than 0', type='negative')
                        return
                    if trailing_stop_loss_percent.value <= 0:
                        ui.notify('Trailing stop loss percentage must be greater than 0', type='negative')
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
                    "is_amo": is_amo_checkbox.value,
                    "broker": broker
                }

                # Add risk management parameters based on type
                if risk_management_type.value == 'regular':
                    order_data.update({
                        "stop_loss": float(regular_stop_loss.value) if regular_stop_loss.value > 0 else None,
                        "target": float(regular_target.value) if regular_target.value > 0 else None,
                        "is_trailing_stop_loss": False
                    })
                else:  # trailing stop loss
                    order_data.update({
                        "stop_loss": float(trailing_stop_loss.value) if trailing_stop_loss.value > 0 else None,
                        "target": float(trailing_target.value) if trailing_target.value > 0 else None,
                        "is_trailing_stop_loss": True,
                        "trailing_stop_loss_percent": float(trailing_stop_loss_percent.value),
                        "trail_start_target_percent": float(trail_start_target_percent.value)
                    })

                # Confirmation dialog
                with ui.dialog() as dialog, ui.card().classes('p-6 min-w-96'):
                    ui.label('Confirm Order Placement').classes('text-xl font-bold mb-4')

                    with ui.column().classes('gap-2 mb-4'):
                        ui.label(f"Symbol: {order_data['trading_symbol']}").classes('text-white')
                        ui.label(f"Type: {order_data['transaction_type']} {order_data['quantity']} shares").classes('text-white')
                        ui.label(f"Order Type: {order_data['order_type']}").classes('text-white')
                        ui.label(f"Order details - {order_data}").classes('text-white')
                        if order_data['price'] > 0:
                            ui.label(f"Price: ₹{order_data['price']:.2f}").classes('text-white')
                        if order_data['trigger_price'] > 0:
                            ui.label(f"Trigger: ₹{order_data['trigger_price']:.2f}").classes('text-white')

                        if order_data.get('is_trailing_stop_loss'):
                            ui.label("Risk Management: Trailing Stop Loss").classes('text-green-400 font-semibold')
                            ui.label(f"Trail Start Target: {order_data['trail_start_target_percent']}%").classes('text-white')
                            ui.label(f"Trailing Stop Loss: {order_data['trailing_stop_loss_percent']}%").classes('text-white')
                        else:
                            if order_data.get('stop_loss'):
                                ui.label(f"Stop Loss: ₹{order_data['stop_loss']:.2f}").classes('text-white')
                            if order_data.get('target'):
                                ui.label(f"Target: ₹{order_data['target']:.2f}").classes('text-white')

                    # Upstox rules summary (GTT)
                    if broker == 'Upstox':
                        ui.label(f"GTT Type: {order_data['trigger_type']}").classes('text-white')
                        for r in order_data.get('rules', [])[:3]:
                            s = f"{r.get('strategy')} {r.get('trigger_type')} @ {r.get('trigger_price')}"
                            ui.label(s).classes('text-white')

                    with ui.row().classes('gap-3'):
                        ui.button('Cancel', on_click=dialog.close).classes('bg-gray-600 text-white px-4 py-2 rounded')

                        async def confirm_order():
                            dialog.close()
                            with loading_container:
                                loading_container.clear()
                                with ui.row().classes("items-center gap-3"):
                                    ui.spinner(size="lg")
                                    ui.label("Placing order...").classes("text-white")

                                response = await fetch_api("/orders", method="POST", data=order_data)
                                if response and response.get('order_id'):
                                    order_type_msg = "Trailing Stop Loss" if order_data.get('is_trailing_stop_loss') else "Regular"
                                    ui.notify(f"{order_type_msg} Order placed: {response['order_id']}", type='positive')
                                else:
                                    ui.notify("Failed to place order", type='negative')
                                loading_container.clear()

                        ui.button('Place Order', on_click=lambda: asyncio.create_task(confirm_order())).classes('bg-green-600 text-white px-4 py-2 rounded')

                dialog.open()

            ui.button('Place Order', icon="send", on_click=place_regular_order).classes('w-full bg-cyan-600 hover:bg-cyan-700 text-white px-6 py-3 rounded-lg font-medium text-lg mt-6')

async def render_scheduled_orders(fetch_api, user_storage, instruments, broker):
    """Scheduled orders form"""

    with ui.card().classes('w-full enhanced-card'):
        with ui.row().classes("w-full items-center justify-between p-4 border-b border-gray-700"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("schedule", size="1.2rem").classes("text-purple-400")
                ui.label("Schedule Order").classes("text-lg font-semibold text-white")
            ui.chip("SCHEDULED", color="purple").classes("text-xs")

        with ui.column().classes("p-6 gap-4 w-full"):
            validation_state = {'symbol': True, 'quantity': True, 'price': True, 'trigger_price': True, 'schedule_datetime': True}

            # Symbol Selection
            with ui.row().classes('w-full gap-6'):
                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Trading Symbol").classes("text-sm font-medium text-gray-300")
                    symbol_options = sorted(list(instruments.keys())[:20]) if instruments else []
                    initial_symbol = symbol_options[0] if symbol_options else None

                    symbol_select = ui.select(
                        options=symbol_options,
                        with_input=True,
                        value=initial_symbol
                    ).classes('w-full')
                    symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Transaction Type").classes("text-sm font-medium text-gray-300")
                    transaction_type = ui.select(
                        options=['BUY', 'SELL'],
                        value='BUY'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Product Type").classes("text-sm font-medium text-gray-300")
                    product_type = ui.select(
                        options=['MIS', 'CNC'] if broker == 'Zerodha' else ['I', 'D'],
                        value='CNC' if broker == 'Zerodha' else 'D'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Quantity").classes("text-sm font-medium text-gray-300")
                    quantity = ui.number(
                        value=1,
                        min=1,
                        format='%d'
                    ).classes('w-full')
                    quantity.on_value_change(lambda e: validation_state.update({'quantity': e.value > 0}))

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Order Type").classes("text-sm font-medium text-gray-300")
                    order_type = ui.select(
                        options=['MARKET', 'LIMIT', 'SL', 'SL-M'],
                        value='MARKET'
                    ).classes('w-full')

            # Price and Schedule Configuration
            with ui.row().classes('w-full gap-6'):
                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Price").classes("text-sm font-medium text-gray-300")
                    price_field = ui.number(
                        value=0,
                        min=0,
                        step=0.05,
                        format='%.2f'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Trigger Price").classes("text-sm font-medium text-gray-300")
                    trigger_price_field = ui.number(
                        value=0,
                        min=0,
                        step=0.05,
                        format='%.2f'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Schedule Date").classes("text-sm font-medium text-gray-300")
                    schedule_date = ui.input(
                                          value=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")).props(
                        "dense type=date").classes("w-full")

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Schedule Time").classes("text-sm font-medium text-gray-300")
                    schedule_time = ui.input(
                                          value=(datetime.now() + timedelta(minutes=15)).strftime("%H:%M")).props(
                        "dense type=time").classes("w-full")

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("AMO").classes("text-sm font-medium text-gray-300")
                    is_amo_checkbox = ui.checkbox('After Market Order (AMO)').classes('text-white')
                    is_amo_checkbox.value = False

            def update_price_fields():
                price_field.visible = order_type.value in ['LIMIT', 'SL']
                trigger_price_field.visible = order_type.value in ['SL', 'SL-M']

            order_type.on_value_change(update_price_fields)
            update_price_fields()

            # Market Price Section
            ui.label("Last Price").classes("text-sm font-medium text-gray-300")
            market_price_label = ui.label("").classes("text-sm font-medium text-gray-300")
            async def update_market_price(symbol):
                instrument_token = instruments.get(symbol)
                if instrument_token:
                    market_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
                    if market_data:
                        last_price = market_data[0]['last_price']
                        market_price_label.text = str(last_price)
                    else:
                        market_price_label.text = "0"
                else:
                    market_price_label.text = "0"
                market_price_label.update()
            await update_market_price(symbol_select.value)
            symbol_select.on_value_change(lambda e: asyncio.create_task(update_market_price(e.value)))

            # Loading container
            loading_container = ui.column().classes('w-full')

            # Schedule Order Action
            async def schedule_order():
                if not all(validation_state.values()):
                    ui.notify('Please fix form errors', type='negative')
                    return

                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative')
                    return

                if quantity.value <= 0:
                    ui.notify('Quantity must be greater than 0', type='negative')
                    return

                # Prepare schedule datetime
                try:
                    schedule_datetime = datetime.combine(
                        datetime.strptime(schedule_date.value, '%Y-%m-%d').date(),
                        datetime.strptime(schedule_time.value, '%H:%M').time()
                    )

                    if schedule_datetime <= datetime.now():
                        ui.notify('Schedule time must be in the future', type='negative')
                        return
                except Exception as e:
                    ui.notify(f'Invalid schedule time: {str(e)}', type='negative')
                    return

                order_data = {
                    "broker": broker,  # Add missing broker field
                    "trading_symbol": symbol_select.value,
                    "instrument_token": instruments[symbol_select.value],
                    "transaction_type": transaction_type.value,
                    "quantity": int(quantity.value),
                    "order_type": order_type.value,
                    "product_type": product_type.value,
                    "price": float(price_field.value) if order_type.value in ['LIMIT', 'SL'] else 0,
                    "trigger_price": float(trigger_price_field.value) if order_type.value in ['SL', 'SL-M'] else 0,
                    "validity": "DAY",
                    "disclosed_quantity": 0,
                    "schedule_datetime": schedule_datetime.isoformat(),
                    "is_amo": is_amo_checkbox.value
                }

                # Confirmation dialog
                with ui.dialog() as dialog, ui.card().classes('p-6 min-w-96'):
                    ui.label('Confirm Scheduled Order').classes('text-xl font-bold mb-4')

                    with ui.column().classes('gap-2 mb-4'):
                        ui.label(f"Symbol: {order_data['trading_symbol']}").classes('text-white')
                        ui.label(f"Schedule: {schedule_datetime.strftime('%Y-%m-%d %H:%M')}").classes('text-white')

                    with ui.row().classes('gap-3'):
                        ui.button('Cancel', on_click=dialog.close).classes('bg-gray-600 text-white px-4 py-2 rounded')

                        async def confirm_scheduled_order():
                            dialog.close()
                            with loading_container:
                                loading_container.clear()
                                with ui.row().classes("items-center gap-3"):
                                    ui.spinner(size="lg")
                                    ui.label("Scheduling order...").classes("text-white")

                                response = await fetch_api("/scheduled-orders/", method="POST", data=order_data)
                                if response and response.get('scheduled_order_id'):
                                    ui.notify(f"Order scheduled: {response['scheduled_order_id']}", type='positive')
                                else:
                                    ui.notify("Failed to schedule order", type='negative')
                                loading_container.clear()

                        ui.button('Schedule Order', on_click=lambda: asyncio.create_task(confirm_scheduled_order())).classes('bg-purple-600 text-white px-4 py-2 rounded')

                dialog.open()

            ui.button('Schedule Order', icon="schedule", on_click=schedule_order).classes('w-full bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-medium text-lg mt-4')

async def render_gtt_orders(fetch_api, user_storage, instruments, broker):
    """GTT orders form"""

    with ui.card().classes('w-full enhanced-card'):
        with ui.row().classes("w-full items-center justify-between p-4 border-b border-gray-700"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("compare_arrows", size="1.2rem").classes("text-green-400")
                ui.label("GTT Orders").classes("text-lg font-semibold text-white")
            ui.chip("TRIGGERED", color="green").classes("text-xs")

        with ui.column().classes("p-6 gap-4 w-full"):
            validation_state = {'symbol': True, 'quantity': True, 'trigger_price': True, 'limit_price': True}

            # Symbol and GTT Configuration
            with ui.row().classes('w-full gap-6'):
                # Index filter and symbol selection similar to regular orders
                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Index Filter").classes("text-sm font-medium text-gray-300")
                    index_select = ui.select(
                        options=['NIFTY_50', 'NIFTY_NEXT_50', 'All Instruments'],
                        value='NIFTY_50'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Trading Symbol").classes("text-sm font-medium text-gray-300")
                    # Prepare initial symbol options
                    symbol_options = await get_symbol_options(index_select.value)
                    if not symbol_options and instruments:
                        symbol_options = sorted(list(instruments.keys())[:50])
                    initial_symbol = symbol_options[0] if symbol_options else None

                    symbol_select = ui.select(
                        options=symbol_options,
                        with_input=True,
                        value=initial_symbol
                    ).classes('w-full')

                    async def on_index_change():
                        try:
                            new_options = await get_symbol_options(index_select.value)
                            if not new_options and instruments:
                                new_options = sorted(list(instruments.keys())[:50])
                            symbol_select.options = new_options
                            if new_options:
                                symbol_select.value = new_options[0]
                            symbol_select.update()
                        except Exception:
                            pass
                    index_select.on('update:model-value', lambda e: asyncio.create_task(on_index_change()))

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("GTT Type").classes("text-sm font-medium text-gray-300")
                    trigger_type = ui.select(
                        options=['single', 'OCO'] if broker == 'Zerodha' else ['SINGLE', 'OCO'],
                        value='single' if broker == 'Zerodha' else 'SINGLE'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Transaction Type").classes("text-sm font-medium text-gray-300")
                    transaction_type = ui.select(
                        options=['BUY', 'SELL'],
                        value='BUY'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Quantity").classes("text-sm font-medium text-gray-300")
                    quantity = ui.number(
                        value=1,
                        min=1,
                        format='%d'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Product Type").classes("text-sm font-medium text-gray-300")
                    product_type = ui.select(
                        options=['MIS', 'CNC'] if broker == 'Zerodha' else ['I', 'D'],
                        value='CNC' if broker == 'Zerodha' else 'D'
                    ).classes('w-full')

            # Price Configuration
            with ui.row().classes('w-full gap-6'):
                if broker == 'Zerodha':
                    with ui.column().classes("flex-1 gap-2"):
                        ui.label("Trigger Price").classes("text-sm font-medium text-gray-300")
                        trigger_price = ui.number(value=0, min=0, step=0.05, format='%.2f').classes('w-full')

                    with ui.column().classes("flex-1 gap-2"):
                        ui.label("Limit Price").classes("text-sm font-medium text-gray-300")
                        limit_price = ui.number(value=0, min=0, step=0.05, format='%.2f').classes('w-full')                

            # Upstox rules editor (ENTRY / TARGET / STOP_LOSS)
            if broker == 'Upstox':
                with ui.column().classes('gap-3'):
                    ui.label('Upstox GTT Rules').classes('text-sm font-medium text-gray-300')
                    with ui.row().classes('w-full gap-6'):
                        with ui.column().classes('flex-1 gap-2') as up_entry_container:
                            ui.label('ENTRY').classes('text-xs text-gray-400')
                            up_entry_enabled = ui.checkbox('Include ENTRY', value=True)
                            up_entry_trigger_type = ui.select(options=['BELOW','ABOVE','IMMEDIATE'], value='ABOVE').classes('w-full')
                            up_entry_trigger_price = ui.number(value=0, min=0, step=0.05, format='%.2f').classes('w-full')
                        with ui.column().classes('flex-1 gap-2') as up_target_container:
                            ui.label('TARGET').classes('text-xs text-gray-400')
                            up_target_enabled = ui.checkbox('Include TARGET', value=True)
                            up_target_trigger_type = ui.select(options=['IMMEDIATE'], value='IMMEDIATE').classes('w-full')
                            up_target_trigger_price = ui.number(value=0, min=0, step=0.05, format='%.2f').classes('w-full')
                        with ui.column().classes('flex-1 gap-2') as up_sl_container:
                            ui.label('STOP_LOSS').classes('text-xs text-gray-400')
                            up_sl_enabled = ui.checkbox('Include STOP_LOSS', value=True)
                            up_sl_trigger_type = ui.select(options=['IMMEDIATE'], value='IMMEDIATE').classes('w-full')
                            up_sl_trigger_price = ui.number(value=0, min=0, step=0.05, format='%.2f').classes('w-full')

                    # Show only ENTRY for SINGLE; show TARGET/STOP_LOSS for OCO
                    def update_upstox_rule_visibility():
                        is_single = str(trigger_type.value).upper() == 'SINGLE'
                        up_entry_container.visible = True
                        up_target_container.visible = not is_single
                        up_sl_container.visible = not is_single
                        up_target_enabled.value = not is_single
                        up_sl_enabled.value = not is_single
                        up_entry_container.update(); up_target_container.update(); up_sl_container.update()
                    trigger_type.on_value_change(lambda e: update_upstox_rule_visibility())
                    update_upstox_rule_visibility()

            # OCO specific fields (shown conditionally) for Zerodha only
            if broker == 'Zerodha':
                oco_section = ui.column().classes("gap-4")
                with oco_section:
                    ui.label("Second Trigger (OCO)").classes("text-lg font-medium text-orange-400")

                    with ui.row().classes('w-full gap-6'):
                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Second Trigger Price").classes("text-sm font-medium text-gray-300")
                            second_trigger_price = ui.number(
                                value=0,
                                min=0,
                                step=0.05,
                                format='%.2f'
                            ).classes('w-full')

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Second Limit Price").classes("text-sm font-medium text-gray-300")
                            second_limit_price = ui.number(
                                value=0,
                                min=0,
                                step=0.05,
                                format='%.2f'
                            ).classes('w-full')

                def update_oco_fields():
                    oco_section.visible = trigger_type.value == 'OCO'

                trigger_type.on_value_change(update_oco_fields)
                update_oco_fields()

            # Market Price Section for GTT Orders
            market_price_label = ui.label("").classes("text-sm font-medium text-gray-300")
            async def update_market_price(symbol):
                instrument_token = instruments.get(symbol)
                if instrument_token:
                    market_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
                    if market_data:
                        last_price = market_data[0]['last_price']
                        market_price_label.text = str(last_price)
                    else:
                        market_price_label.text = "0"
                else:
                    market_price_label.text = "0"
                market_price_label.update()
            await update_market_price(symbol_select.value)
            symbol_select.on_value_change(lambda e: asyncio.create_task(update_market_price(e.value)))

            # Loading container
            loading_container = ui.column().classes('w-full')

            # Place GTT Order Action
            async def place_gtt_order():
                if not all(validation_state.values()):
                    ui.notify('Please fix form errors', type='negative')
                    return

                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative')
                    return

                if quantity.value <= 0:
                    ui.notify('Quantity must be greater than 0', type='negative')
                    return

                if broker == 'Zerodha':
                    if trigger_price.value <= 0:
                        ui.notify('Trigger price must be greater than 0', type='negative')
                        return
                    if limit_price.value <= 0:
                        ui.notify('Limit price must be greater than 0', type='negative')
                        return
                    if trigger_type.value == 'OCO':
                        if second_trigger_price.value <= 0:
                            ui.notify('Second trigger price must be greater than 0 for OCO orders', type='negative')
                            return
                        if second_limit_price.value <= 0:
                            ui.notify('Second limit price must be greater than 0 for OCO orders', type='negative')
                            return

                # Fetch current market price for reference
                try:
                    instrument_token = instruments[symbol_select.value]
                    market_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
                    if market_data:
                        last_price = market_data[0].get('last_price', 0)
                    else:
                        last_price = 0
                except Exception as e:
                    last_price = 0

                order_data = {
                    "broker": broker,
                    "trading_symbol": symbol_select.value,
                    "instrument_token": instruments[symbol_select.value],
                    "transaction_type": transaction_type.value,
                    "quantity": int(quantity.value),
                    "product_type": product_type.value,
                    "last_price": float(last_price)
                }

                if broker == 'Zerodha':
                    order_data.update({
                        "trigger_type": trigger_type.value,
                        "trigger_price": float(trigger_price.value),
                        "limit_price": float(limit_price.value)
                    })
                    if trigger_type.value == 'OCO':
                        order_data.update({
                            "second_trigger_price": float(second_trigger_price.value),
                            "second_limit_price": float(second_limit_price.value)
                        })
                else:
                    # Upstox rules array
                    up_rules = []
                    try:
                        if up_entry_enabled.value:
                            rule = {
                                "strategy": "ENTRY",
                                "trigger_type": up_entry_trigger_type.value,
                                "trigger_price": float(up_entry_trigger_price.value)
                            }
                            up_rules.append(rule)
                        if up_target_enabled.value:
                            rule = {
                                "strategy": "TARGET",
                                "trigger_type": up_target_trigger_type.value,
                                "trigger_price": float(up_target_trigger_price.value)
                            }
                            up_rules.append(rule)
                        if up_sl_enabled.value:
                            rule = {
                                "strategy": "STOPLOSS",
                                "trigger_type": up_sl_trigger_type.value,
                                "trigger_price": float(up_sl_trigger_price.value)
                            }
                            up_rules.append(rule)
                    except Exception:
                        up_rules = []
                    if not up_rules:
                        ui.notify('Select at least one rule for Upstox GTT', type='negative')
                        return
                    order_data.update({
                        "trigger_type": str(trigger_type.value).upper(),  # SINGLE or OCO
                        "rules": up_rules
                    })

                # Confirmation dialog
                with ui.dialog() as dialog, ui.card().classes('p-6 min-w-96'):
                    ui.label('Confirm GTT Order').classes('text-xl font-bold mb-4')

                    with ui.column().classes('gap-2 mb-4'):
                        ui.label(f"Symbol: {order_data['trading_symbol']}").classes('text-white')
                        ui.label(f"Type: {order_data['transaction_type']} {order_data['quantity']} shares").classes('text-white')

                        if broker == 'Upstox':
                            # Upstox: show rules summary only
                            ui.label(f"GTT Type: {order_data['trigger_type']}").classes('text-white')
                            for r in order_data.get('rules', [])[:3]:
                                s = f"{r.get('strategy')} {r.get('trigger_type')} @ {r.get('trigger_price')}"
                                ui.label(s).classes('text-white')
                        else:
                            ui.label(f"Trigger Type: {order_data['trigger_type']}").classes('text-white')
                            ui.label(f"Trigger Price: ₹{order_data['trigger_price']:.2f}").classes('text-white')
                            ui.label(f"Limit Price: ₹{order_data['limit_price']:.2f}").classes('text-white')
                            if trigger_type.value == 'OCO':
                                ui.label(f"Second Trigger: ₹{order_data['second_trigger_price']:.2f}").classes('text-white')

                    with ui.row().classes('gap-3'):
                        ui.button('Cancel', on_click=dialog.close).classes('bg-gray-600 text-white px-4 py-2 rounded')

                        async def confirm_gtt_order():
                            dialog.close()
                            with loading_container:
                                loading_container.clear()
                                with ui.row().classes("items-center gap-3"):
                                    ui.spinner(size="lg")
                                    ui.label("Placing GTT order...").classes("text-white")

                                response = await fetch_api("/gtt-orders/", method="POST", data=order_data)
                                if response and response.get('gtt_id'):
                                    ui.notify(f"GTT order placed: {response['gtt_id']}", type='positive')
                                else:
                                    ui.notify("Failed to place GTT order", type='negative')
                                loading_container.clear()

                        ui.button('Place GTT Order', on_click=lambda: asyncio.create_task(confirm_gtt_order())).classes('bg-green-600 text-white px-4 py-2 rounded')

                dialog.open()

            ui.button('Place GTT Order', icon="compare_arrows", on_click=place_gtt_order).classes('w-full bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg font-medium text-lg mt-4')

async def render_auto_orders(fetch_api, user_storage, instruments, broker):
    """Auto orders form"""

    with ui.card().classes('w-full enhanced-card'):
        with ui.row().classes("w-full items-center justify-between p-4 border-b border-gray-700"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("smart_toy", size="1.2rem").classes("text-orange-400")
                ui.label("Auto Orders").classes("text-lg font-semibold text-white")
            ui.chip("ALGORITHMIC", color="orange").classes("text-xs")

        with ui.column().classes("p-6 gap-4 w-full"):
            ui.label('Set up automated orders based on risk parameters').classes('text-gray-400 mb-4')

            validation_state = {
                'symbol': True, 'risk_per_trade': True, 'stop_loss_value': True,
                'target_value': True, 'limit_price': True, 'atr_period': True,
                'check_interval': True
            }

            # Symbol and Basic Configuration
            with ui.row().classes('w-full gap-6'):
                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Trading Symbol").classes("text-sm font-medium text-gray-300")
                    symbol_options = sorted(list(instruments.keys())[:20]) if instruments else []
                    initial_symbol = symbol_options[0] if symbol_options else None

                    symbol_select = ui.select(
                        options=symbol_options,
                        with_input=True,
                        value=initial_symbol
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Transaction Type").classes("text-sm font-medium text-gray-300")
                    transaction_type = ui.select(
                        options=['BUY', 'SELL'],
                        value='BUY'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Product Type").classes("text-sm font-medium text-gray-300")
                    product_type = ui.select(
                        options=['MIS', 'CNC'] if broker == 'Zerodha' else ['I', 'D'],
                        value='CNC' if broker == 'Zerodha' else 'D'
                    ).classes('w-full')

            # Risk Management Configuration
            with ui.row().classes('w-full gap-6'):
                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Risk Per Trade (₹)").classes("text-sm font-medium text-gray-300")
                    risk_per_trade = ui.number(
                        value=1000,
                        min=100,
                        format='%.0f'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Stop Loss Type").classes("text-sm font-medium text-gray-300")
                    stop_loss_type = ui.select(
                        options=['Fixed Amount', 'Percentage of Entry', 'ATR Based'],
                        value='Fixed Amount'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Stop Loss Value").classes("text-sm font-medium text-gray-300")
                    stop_loss_value = ui.number(
                        value=2.0,
                        min=0.1,
                        step=0.1,
                        format='%.1f'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Target Value").classes("text-sm font-medium text-gray-300")
                    target_value = ui.number(
                        value=3.0,
                        min=0.1,
                        step=0.1,
                        format='%.1f'
                    ).classes('w-full')

            # Additional Configuration
            with ui.row().classes('w-full gap-6'):
                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Limit Price").classes("text-sm font-medium text-gray-300")
                    limit_price = ui.number(
                        value=0,
                        min=0,
                        step=0.05,
                        format='%.2f'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("ATR Period").classes("text-sm font-medium text-gray-300")
                    atr_period = ui.number(
                        value=14,
                        min=5,
                        max=50,
                        format='%.0f'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Check Interval (seconds)").classes("text-sm font-medium text-gray-300")
                    check_interval = ui.number(
                        value=60,
                        min=30,
                        format='%.0f'
                    ).classes('w-full')

                with ui.column().classes("flex-1 gap-2"):
                    ui.label("Square Off Time").classes("text-sm font-medium text-gray-300")
                    square_off_time = ui.input(
                        value="15:20"
                    ).props("dense type=time").classes("w-full")

                    # square_off_time = ui.time(
                    #     value="15:20"
                    # ).classes('w-full')

            # Additional Settings
            with ui.row().classes('w-full gap-6 mt-4'):
                trailing_stop = ui.switch('Enable Trailing Stop Loss').classes('text-white')

            def update_atr_field():
                atr_period.visible = stop_loss_type.value == 'ATR Based'

            stop_loss_type.on_value_change(update_atr_field)
            update_atr_field()

            # Loading container
            loading_container = ui.column().classes('w-full')

            # Place Auto Order Action
            async def place_auto_order():
                if not all(validation_state.values()):
                    ui.notify('Please fix form errors', type='negative')
                    return

                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative')
                    return

                order_data = {
                    "trading_symbol": symbol_select.value,
                    "instrument_token": instruments[symbol_select.value],
                    "transaction_type": transaction_type.value,
                    "product_type": product_type.value,
                    "risk_per_trade": float(risk_per_trade.value),
                    "stop_loss_type": stop_loss_type.value,
                    "stop_loss_value": float(stop_loss_value.value),
                    "target_value": float(target_value.value),
                    "limit_price": float(limit_price.value) if limit_price.value > 0 else None,
                    "atr_period": int(atr_period.value),
                    "check_interval": int(check_interval.value),
                    "trailing_stop_loss": trailing_stop.value,
                    "square_off_time": square_off_time.value
                }

                # Confirmation dialog
                with ui.dialog() as dialog, ui.card().classes('p-6 min-w-96'):
                    ui.label('Confirm Auto Order').classes('text-xl font-bold mb-4')

                    with ui.column().classes('gap-2 mb-4'):
                        ui.label(f"Symbol: {order_data['trading_symbol']}").classes('text-white')
                        ui.label(f"Type: {order_data['transaction_type']}").classes('text-white')
                        ui.label(f"Risk per Trade: ₹{order_data['risk_per_trade']:.0f}").classes('text-white')
                        ui.label(f"Stop Loss: {order_data['stop_loss_value']:.1f} ({order_data['stop_loss_type']})").classes('text-white')
                        ui.label(f"Target: {order_data['target_value']:.1f}").classes('text-white')

                    with ui.row().classes('gap-3'):
                        ui.button('Cancel', on_click=dialog.close).classes('bg-gray-600 text-white px-4 py-2 rounded')

                        async def confirm_auto_order():
                            dialog.close()
                            with loading_container:
                                loading_container.clear()
                                with ui.row().classes("items-center gap-3"):
                                    ui.spinner(size="lg")
                                    ui.label("Setting up auto order...").classes("text-white")

                                response = await fetch_api("/auto-orders/", method="POST", data=order_data)
                                if response and response.get('auto_order_id'):
                                    ui.notify(f"Auto order created: {response['auto_order_id']}", type='positive')
                                else:
                                    ui.notify("Failed to create auto order", type='negative')
                                loading_container.clear()

                        ui.button('Create Auto Order', on_click=lambda: asyncio.create_task(confirm_auto_order())).classes('bg-orange-600 text-white px-4 py-2 rounded')

                dialog.open()

            ui.button('Create Auto Order', icon="smart_toy", on_click=place_auto_order).classes('w-full bg-orange-600 hover:bg-orange-700 text-white px-6 py-3 rounded-lg font-medium text-lg mt-4')
