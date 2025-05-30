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
    """Render the regular order placement form"""
    broker = user_storage.get('broker', 'Zerodha')
    
    with ui.card().classes('w-full'):
        ui.label('Place Regular Order').classes('text-h6')
        
        with ui.row().classes('w-full items-center'):
            ui.label('Symbol Selection').classes('text-subtitle1 col-12')
        
        with ui.row().classes('w-full items-center gap-2'):
            index_select = ui.select(
                ['NIFTY 50', 'NIFTY NEXT 50', 'All Instruments'], 
                value='NIFTY 50',
                label='Index Filter'
            ).classes('col-4')
            
            # Filter instruments based on selected index
            async def update_symbol_options():
                if index_select.value == 'All Instruments':
                    symbol_select.options = sorted(list(instruments.keys()))
                else:
                    # In a real implementation, you would filter by index membership
                    # This is a simplified version
                    filtered_symbols = [s for s in instruments.keys() if len(s) < 10]  # Simple filter for demo
                    symbol_select.options = sorted(filtered_symbols)
            
            index_select.on('update:model-value', lambda _: asyncio.create_task(update_symbol_options()))
            
            symbol_select = ui.select(
                options=sorted(list(instruments.keys())[:20]),  # Start with first 20 for performance
                label='Symbol',
                with_input=True,
                value=sorted(list(instruments.keys()))[0] if instruments else None
            ).classes('col-8')
        
        with ui.form().classes('w-full').on('submit', lambda: place_regular_order()):
            with ui.row().classes('w-full items-center gap-2'):
                transaction_type = ui.select(
                    ['BUY', 'SELL'],
                    value='BUY',
                    label='Transaction Type'
                ).classes('col-4')
                
                product_type = ui.select(
                    ['INTRADAY', 'DELIVERY'] if broker == 'Zerodha' else ['I', 'D'],
                    value='INTRADAY' if broker == 'Zerodha' else 'I',
                    label='Product Type'
                ).classes('col-4')
                
                quantity = ui.number(
                    label='Quantity',
                    value=1,
                    min=1,
                    format='%d'
                ).classes('col-4')
            
            with ui.row().classes('w-full items-center gap-2'):
                order_type = ui.select(
                    ['MARKET', 'LIMIT', 'SL', 'SL-M'],
                    value='MARKET',
                    label='Order Type'
                ).classes('col-4')
                
                # Dynamic fields based on order type
                price_field = ui.number(
                    label='Price',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f',
                    visible=False
                ).classes('col-4')
                
                trigger_price_field = ui.number(
                    label='Trigger Price',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f',
                    visible=False
                ).classes('col-4')
                
                # Show/hide price and trigger fields based on order type
                def update_price_fields():
                    price_field.visible = order_type.value in ['LIMIT', 'SL']
                    trigger_price_field.visible = order_type.value in ['SL', 'SL-M']
                
                order_type.on('update:model-value', lambda _: update_price_fields())
                update_price_fields()  # Initial update
            
            with ui.row().classes('w-full items-center gap-2'):
                validity = ui.select(
                    ['DAY', 'IOC'],
                    value='DAY',
                    label='Validity'
                ).classes('col-4')
                
                disclosed_quantity = ui.number(
                    label='Disclosed Qty (Optional)',
                    value=0,
                    min=0,
                    format='%d'
                ).classes('col-4')
                
                # Fetch current market price for the selected symbol
                async def fetch_market_price():
                    if symbol_select.value and symbol_select.value in instruments:
                        instrument_token = instruments[symbol_select.value]
                        ltp_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
                        if ltp_data and isinstance(ltp_data, list) and len(ltp_data) > 0:
                            price = ltp_data[0].get('last_price', 0)
                            if price > 0:
                                price_field.value = price
                                trigger_price_field.value = price
                                return price
                    return 0
                
                ui.button('Get Market Price', on_click=lambda: asyncio.create_task(fetch_market_price())).classes('col-4')
            
            # Order placement function
            async def place_regular_order():
                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative')
                    return
                
                # Validate inputs
                if quantity.value <= 0:
                    ui.notify('Quantity must be greater than 0', type='negative')
                    return
                
                if order_type.value in ['LIMIT', 'SL'] and price_field.value <= 0:
                    ui.notify('Price must be greater than 0 for LIMIT and SL orders', type='negative')
                    return
                
                if order_type.value in ['SL', 'SL-M'] and trigger_price_field.value <= 0:
                    ui.notify('Trigger price must be greater than 0 for SL and SL-M orders', type='negative')
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
                    "validity": validity.value,
                    "disclosed_quantity": int(disclosed_quantity.value) if disclosed_quantity.value > 0 else 0,
                    "broker": broker
                }
                
                # Show confirmation dialog
                with ui.dialog() as dialog, ui.card():
                    ui.label('Confirm Order').classes('text-h6')
                    ui.label(f"Symbol: {order_data['trading_symbol']}")
                    ui.label(f"Type: {order_data['transaction_type']} {order_data['order_type']}")
                    ui.label(f"Quantity: {order_data['quantity']}")
                    
                    if order_data['price'] > 0:
                        ui.label(f"Price: ₹{order_data['price']:.2f}")
                    
                    if order_data['trigger_price'] > 0:
                        ui.label(f"Trigger Price: ₹{order_data['trigger_price']:.2f}")
                    
                    with ui.row():
                        ui.button('Cancel', on_click=dialog.close).props('outline')
                        
                        async def confirm_order():
                            dialog.close()
                            with ui.loading(text='Placing order...'):
                                response = await fetch_api("/orders/", method="POST", data=order_data)
                                if response and response.get('order_id'):
                                    ui.notify(f"Order placed successfully: {response['order_id']}", type='positive')
                                else:
                                    ui.notify("Failed to place order", type='negative')
                        
                        ui.button('Confirm', on_click=lambda: asyncio.create_task(confirm_order())).props('color=primary')
                
                dialog.open()
            
            ui.button('Place Order', on_click=place_regular_order).props('color=primary').classes('mt-4')

async def render_scheduled_orders(fetch_api, user_storage, instruments):
    """Render the scheduled order placement form"""
    broker = user_storage.get('broker', 'Zerodha')
    
    with ui.card().classes('w-full'):
        ui.label('Schedule Order').classes('text-h6')
        
        with ui.form().classes('w-full').on('submit', lambda: schedule_order()):
            with ui.row().classes('w-full items-center gap-2'):
                symbol_select = ui.select(
                    options=sorted(list(instruments.keys())),
                    label='Symbol',
                    with_input=True,
                    value=sorted(list(instruments.keys()))[0] if instruments else None
                ).classes('col-12')
            
            with ui.row().classes('w-full items-center gap-2'):
                transaction_type = ui.select(
                    ['BUY', 'SELL'],
                    value='BUY',
                    label='Transaction Type'
                ).classes('col-4')
                
                product_type = ui.select(
                    ['INTRADAY', 'DELIVERY'] if broker == 'Zerodha' else ['I', 'D'],
                    value='INTRADAY' if broker == 'Zerodha' else 'I',
                    label='Product Type'
                ).classes('col-4')
                
                quantity = ui.number(
                    label='Quantity',
                    value=1,
                    min=1,
                    format='%d'
                ).classes('col-4')
            
            with ui.row().classes('w-full items-center gap-2'):
                order_type = ui.select(
                    ['MARKET', 'LIMIT', 'SL', 'SL-M'],
                    value='MARKET',
                    label='Order Type'
                ).classes('col-4')
                
                price_field = ui.number(
                    label='Price',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f',
                    visible=False
                ).classes('col-4')
                
                trigger_price_field = ui.number(
                    label='Trigger Price',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f',
                    visible=False
                ).classes('col-4')
                
                def update_price_fields():
                    price_field.visible = order_type.value in ['LIMIT', 'SL']
                    trigger_price_field.visible = order_type.value in ['SL', 'SL-M']
                
                order_type.on('update:model-value', lambda _: update_price_fields())
                update_price_fields()
            
            ui.label('Schedule Settings').classes('text-subtitle1 col-12 mt-4')
            
            with ui.row().classes('w-full items-center gap-2'):
                # Date picker for schedule date
                schedule_date = ui.date(
                    label='Schedule Date',
                    value=datetime.now().date() + timedelta(days=1)
                ).classes('col-6')
                
                # Time picker for schedule time
                schedule_time = ui.time(
                    label='Schedule Time',
                    value='09:15'
                ).classes('col-6')
            
            with ui.row().classes('w-full items-center gap-2'):
                ui.label('Risk Management').classes('text-subtitle1 col-12')
            
            with ui.row().classes('w-full items-center gap-2'):
                stop_loss = ui.number(
                    label='Stop Loss (₹)',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('col-6')
                
                target = ui.number(
                    label='Target (₹)',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('col-6')
            
            # Schedule order function
            async def schedule_order():
                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative')
                    return
                
                # Validate inputs
                if quantity.value <= 0:
                    ui.notify('Quantity must be greater than 0', type='negative')
                    return
                
                if order_type.value in ['LIMIT', 'SL'] and price_field.value <= 0:
                    ui.notify('Price must be greater than 0 for LIMIT and SL orders', type='negative')
                    return
                
                if order_type.value in ['SL', 'SL-M'] and trigger_price_field.value <= 0:
                    ui.notify('Trigger price must be greater than 0 for SL and SL-M orders', type='negative')
                    return
                
                # Prepare schedule datetime
                try:
                    schedule_datetime = datetime.combine(
                        schedule_date.value,
                        datetime.strptime(schedule_time.value, '%H:%M').time()
                    )
                    
                    # Validate schedule time is in the future
                    if schedule_datetime <= datetime.now():
                        ui.notify('Schedule time must be in the future', type='negative')
                        return
                    
                except Exception as e:
                    ui.notify(f'Invalid schedule time: {str(e)}', type='negative')
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
                    "stop_loss": float(stop_loss.value),
                    "target": float(target.value),
                    "broker": broker
                }
                
                # Show confirmation dialog
                with ui.dialog() as dialog, ui.card():
                    ui.label('Confirm Scheduled Order').classes('text-h6')
                    ui.label(f"Symbol: {order_data['trading_symbol']}")
                    ui.label(f"Type: {order_data['transaction_type']} {order_data['order_type']}")
                    ui.label(f"Quantity: {order_data['quantity']}")
                    ui.label(f"Schedule: {schedule_datetime.strftime('%Y-%m-%d %H:%M')}")
                    
                    if order_data['price'] > 0:
                        ui.label(f"Price: ₹{order_data['price']:.2f}")
                    
                    if order_data['trigger_price'] > 0:
                        ui.label(f"Trigger Price: ₹{order_data['trigger_price']:.2f}")
                    
                    if order_data['stop_loss'] > 0:
                        ui.label(f"Stop Loss: ₹{order_data['stop_loss']:.2f}")
                    
                    if order_data['target'] > 0:
                        ui.label(f"Target: ₹{order_data['target']:.2f}")
                    
                    with ui.row():
                        ui.button('Cancel', on_click=dialog.close).props('outline')
                        
                        async def confirm_scheduled_order():
                            dialog.close()
                            with ui.loading(text='Scheduling order...'):
                                response = await fetch_api("/scheduled-orders/", method="POST", data=order_data)
                                if response and response.get('order_id'):
                                    ui.notify(f"Order scheduled successfully: {response['order_id']}", type='positive')
                                else:
                                    ui.notify("Failed to schedule order", type='negative')
                        
                        ui.button('Confirm', on_click=lambda: asyncio.create_task(confirm_scheduled_order())).props('color=primary')
                
                dialog.open()
            
            ui.button('Schedule Order', on_click=schedule_order).props('color=primary').classes('mt-4')
        
        # Display scheduled orders
        ui.separator()
        ui.label('Scheduled Orders').classes('text-subtitle1 mt-4')
        
        scheduled_orders_grid = ui.aggrid({
            'columnDefs': [
                {'headerName': 'Order ID', 'field': 'order_id'},
                {'headerName': 'Symbol', 'field': 'trading_symbol'},
                {'headerName': 'Type', 'field': 'transaction_type'},
                {'headerName': 'Quantity', 'field': 'quantity'},
                {'headerName': 'Schedule Time', 'field': 'schedule_datetime'},
                {'headerName': 'Status', 'field': 'status'}
            ],
            'rowData': [],
            'rowSelection': 'single',
            'pagination': True,
            'paginationPageSize': 10
        }).classes('w-full mt-4')
        
        async def fetch_scheduled_orders():
            scheduled_orders = await fetch_api(f"/scheduled-orders/{broker}")
            if scheduled_orders and isinstance(scheduled_orders, list):
                # Format datetime for display
                for order in scheduled_orders:
                    if 'schedule_datetime' in order:
                        try:
                            dt = datetime.fromisoformat(order['schedule_datetime'].replace('Z', '+00:00'))
                            order['schedule_datetime'] = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            pass
                
                await scheduled_orders_grid.update_grid_options({'rowData': scheduled_orders})
            else:
                await scheduled_orders_grid.update_grid_options({'rowData': []})
        
        await fetch_scheduled_orders()
        
        with ui.row().classes('w-full mt-4'):
            ui.button('Refresh', on_click=lambda: asyncio.create_task(fetch_scheduled_orders())).props('outline')
            
            async def cancel_scheduled_order():
                selected_rows = await scheduled_orders_grid.get_selected_rows()
                if not selected_rows:
                    ui.notify('Please select an order to cancel', type='warning')
                    return
                
                order_id = selected_rows[0].get('order_id')
                if not order_id:
                    ui.notify('Invalid order selection', type='negative')
                    return
                
                with ui.loading(text='Cancelling order...'):
                    response = await fetch_api(f"/scheduled-orders/{order_id}", method="DELETE")
                    if response and response.get('success'):
                        ui.notify('Order cancelled successfully', type='positive')
                        await fetch_scheduled_orders()  # Refresh the list
                    else:
                        ui.notify('Failed to cancel order', type='negative')
            
            ui.button('Cancel Selected', on_click=lambda: asyncio.create_task(cancel_scheduled_order())).props('color=negative')

async def render_gtt_orders(fetch_api, user_storage, instruments):
    """Render the GTT (Good Till Triggered) order placement form"""
    broker = user_storage.get('broker', 'Zerodha')
    
    with ui.card().classes('w-full'):
        ui.label('Place GTT Order').classes('text-h6')
        ui.label('Good Till Triggered orders remain active until the trigger condition is met').classes('text-caption')
        
        with ui.form().classes('w-full').on('submit', lambda: place_gtt_order()):
            with ui.row().classes('w-full items-center gap-2'):
                symbol_select = ui.select(
                    options=sorted(list(instruments.keys())),
                    label='Symbol',
                    with_input=True,
                    value=sorted(list(instruments.keys()))[0] if instruments else None
                ).classes('col-12')
            
            with ui.row().classes('w-full items-center gap-2'):
                transaction_type = ui.select(
                    ['BUY', 'SELL'],
                    value='BUY',
                    label='Transaction Type'
                ).classes('col-4')
                
                product_type = ui.select(
                    ['DELIVERY'] if broker == 'Zerodha' else ['D'],
                    value='DELIVERY' if broker == 'Zerodha' else 'D',
                    label='Product Type'
                ).classes('col-4')
                
                quantity = ui.number(
                    label='Quantity',
                    value=1,
                    min=1,
                    format='%d'
                ).classes('col-4')
            
            ui.label('Trigger Settings').classes('text-subtitle1 col-12 mt-4')
            
            with ui.row().classes('w-full items-center gap-2'):
                trigger_type = ui.select(
                    ['SINGLE', 'OCO'],  # One-Cancels-Other (OCO) for both upper and lower triggers
                    value='SINGLE',
                    label='Trigger Type'
                ).classes('col-4')
                
                trigger_condition = ui.select(
                    ['LTP', 'BID', 'ASK'],
                    value='LTP',
                    label='Trigger Condition'
                ).classes('col-4')
                
                # Fetch current market price for the selected symbol
                async def fetch_market_price():
                    if symbol_select.value and symbol_select.value in instruments:
                        instrument_token = instruments[symbol_select.value]
                        ltp_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
                        if ltp_data and isinstance(ltp_data, list) and len(ltp_data) > 0:
                            price = ltp_data[0].get('last_price', 0)
                            if price > 0:
                                trigger_price.value = price
                                upper_trigger_price.value = price * 1.05  # 5% above
                                lower_trigger_price.value = price * 0.95  # 5% below
                                return price
                    return 0
                
                ui.button('Get Market Price', on_click=lambda: asyncio.create_task(fetch_market_price())).classes('col-4')
            
            with ui.row().classes('w-full items-center gap-2'):
                trigger_price = ui.number(
                    label='Trigger Price',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('col-4')
                
                upper_trigger_price = ui.number(
                    label='Upper Trigger Price',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f',
                    visible=False
                ).classes('col-4')
                
                lower_trigger_price = ui.number(
                    label='Lower Trigger Price',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f',
                    visible=False
                ).classes('col-4')
                
                def update_trigger_fields():
                    upper_trigger_price.visible = trigger_type.value == 'OCO'
                    lower_trigger_price.visible = trigger_type.value == 'OCO'
                    trigger_price.visible = trigger_type.value == 'SINGLE'
                
                trigger_type.on('update:model-value', lambda _: update_trigger_fields())
                update_trigger_fields()
            
            ui.label('Order Settings').classes('text-subtitle1 col-12 mt-4')
            
            with ui.row().classes('w-full items-center gap-2'):
                order_type = ui.select(
                    ['LIMIT', 'MARKET'],
                    value='LIMIT',
                    label='Order Type'
                ).classes('col-4')
                
                price_field = ui.number(
                    label='Price',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f'
                ).classes('col-4')
                
                # Show/hide price field based on order type
                def update_price_field():
                    price_field.visible = order_type.value == 'LIMIT'
                
                order_type.on('update:model-value', lambda _: update_price_field())
                update_price_field()
                
                # GTT expiry date (optional)
                expiry_date = ui.date(
                    label='Expiry Date (Optional)',
                    value=None
                ).classes('col-4')
            
            # Place GTT order function
            async def place_gtt_order():
                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative')
                    return
                
                # Validate inputs
                if quantity.value <= 0:
                    ui.notify('Quantity must be greater than 0', type='negative')
                    return
                
                if trigger_type.value == 'SINGLE' and trigger_price.value <= 0:
                    ui.notify('Trigger price must be greater than 0', type='negative')
                    return
                
                if trigger_type.value == 'OCO':
                    if upper_trigger_price.value <= 0 or lower_trigger_price.value <= 0:
                        ui.notify('Both trigger prices must be greater than 0 for OCO orders', type='negative')
                        return
                    if upper_trigger_price.value <= lower_trigger_price.value:
                        ui.notify('Upper trigger price must be greater than lower trigger price', type='negative')
                        return
                
                if order_type.value == 'LIMIT' and price_field.value <= 0:
                    ui.notify('Price must be greater than 0 for LIMIT orders', type='negative')
                    return
                
                # Prepare order data
                order_data = {
                    "trading_symbol": symbol_select.value,
                    "instrument_token": instruments[symbol_select.value],
                    "exchange": "NSE",
                    "transaction_type": transaction_type.value,
                    "quantity": int(quantity.value),
                    "order_type": order_type.value,
                    "product_type": product_type.value,
                    "price": float(price_field.value) if order_type.value == 'LIMIT' else 0,
                    "trigger_type": trigger_type.value,
                    "trigger_condition": trigger_condition.value,
                    "broker": broker
                }
                
                if trigger_type.value == 'SINGLE':
                    order_data["trigger_price"] = float(trigger_price.value)
                else:
                    order_data["upper_trigger_price"] = float(upper_trigger_price.value)
                    order_data["lower_trigger_price"] = float(lower_trigger_price.value)
                
                if expiry_date.value:
                    order_data["expiry_date"] = expiry_date.value.isoformat()
                
                # Show confirmation dialog
                with ui.dialog() as dialog, ui.card():
                    ui.label('Confirm GTT Order').classes('text-h6')
                    ui.label(f"Symbol: {order_data['trading_symbol']}")
                    ui.label(f"Type: {order_data['transaction_type']} {order_data['order_type']}")
                    ui.label(f"Quantity: {order_data['quantity']}")
                    
                    if trigger_type.value == 'SINGLE':
                        ui.label(f"Trigger Price: ₹{order_data['trigger_price']:.2f}")
                    else:
                        ui.label(f"Upper Trigger: ₹{order_data['upper_trigger_price']:.2f}")
                        ui.label(f"Lower Trigger: ₹{order_data['lower_trigger_price']:.2f}")
                    
                    if order_data['price'] > 0:
                        ui.label(f"Price: ₹{order_data['price']:.2f}")
                    
                    if 'expiry_date' in order_data:
                        ui.label(f"Expiry: {order_data['expiry_date']}")
                    
                    with ui.row():
                        ui.button('Cancel', on_click=dialog.close).props('outline')
                        
                        async def confirm_gtt_order():
                            dialog.close()
                            with ui.loading(text='Placing GTT order...'):
                                response = await fetch_api("/gtt-orders/", method="POST", data=order_data)
                                if response and response.get('gtt_order_id'):
                                    ui.notify(f"GTT order placed successfully: {response['gtt_order_id']}", type='positive')
                                else:
                                    ui.notify("Failed to place GTT order", type='negative')
                        
                        ui.button('Confirm', on_click=lambda: asyncio.create_task(confirm_gtt_order())).props('color=primary')
                
                dialog.open()
            
            ui.button('Place GTT Order', on_click=place_gtt_order).props('color=primary').classes('mt-4')
        
        # Display active GTT orders
        ui.separator()
        ui.label('Active GTT Orders').classes('text-subtitle1 mt-4')
        
        gtt_orders_grid = ui.aggrid({
            'columnDefs': [
                {'headerName': 'GTT ID', 'field': 'gtt_order_id'},
                {'headerName': 'Symbol', 'field': 'trading_symbol'},
                {'headerName': 'Type', 'field': 'transaction_type'},
                {'headerName': 'Trigger Type', 'field': 'trigger_type'},
                {'headerName': 'Trigger Price', 'field': 'trigger_price', 
                 'valueFormatter': "params.data.trigger_type === 'SINGLE' ? '₹' + params.value.toFixed(2) : '—'"},
                {'headerName': 'Upper Trigger', 'field': 'upper_trigger_price',
                 'valueFormatter': "params.data.trigger_type === 'OCO' ? '₹' + params.value.toFixed(2) : '—'"},
                {'headerName': 'Lower Trigger', 'field': 'lower_trigger_price',
                 'valueFormatter': "params.data.trigger_type === 'OCO' ? '₹' + params.value.toFixed(2) : '—'"},
                {'headerName': 'Status', 'field': 'status'}
            ],
            'rowData': [],
            'rowSelection': 'single',
            'pagination': True,
            'paginationPageSize': 10
        }).classes('w-full mt-4')
        
        async def fetch_gtt_orders():
            gtt_orders = await fetch_api(f"/gtt-orders/{broker}")
            if gtt_orders and isinstance(gtt_orders, list):
                await gtt_orders_grid.update_grid_options({'rowData': gtt_orders})
            else:
                await gtt_orders_grid.update_grid_options({'rowData': []})
        
        await fetch_gtt_orders()
        
        with ui.row().classes('w-full mt-4'):
            ui.button('Refresh', on_click=lambda: asyncio.create_task(fetch_gtt_orders())).props('outline')
            
            async def cancel_gtt_order():
                selected_rows = await gtt_orders_grid.get_selected_rows()
                if not selected_rows:
                    ui.notify('Please select an order to cancel', type='warning')
                    return
                
                gtt_order_id = selected_rows[0].get('gtt_order_id')
                if not gtt_order_id:
                    ui.notify('Invalid order selection', type='negative')
                    return
                
                with ui.loading(text='Cancelling GTT order...'):
                    response = await fetch_api(f"/gtt-orders/{gtt_order_id}", method="DELETE")
                    if response and response.get('success'):
                        ui.notify('GTT order cancelled successfully', type='positive')
                        await fetch_gtt_orders()  # Refresh the list
                    else:
                        ui.notify('Failed to cancel GTT order', type='negative')
            
            ui.button('Cancel Selected', on_click=lambda: asyncio.create_task(cancel_gtt_order())).props('color=negative')

async def render_auto_orders(fetch_api, user_storage, instruments):
    """Render the auto order placement form for algorithmic trading"""
    broker = user_storage.get('broker', 'Zerodha')
    
    with ui.card().classes('w-full'):
        ui.label('Auto Orders').classes('text-h6')
        ui.label('Set up automated orders based on risk parameters').classes('text-caption')
        
        with ui.form().classes('w-full').on('submit', lambda: place_auto_order()):
            with ui.row().classes('w-full items-center gap-2'):
                symbol_select = ui.select(
                    options=sorted(list(instruments.keys())),
                    label='Symbol',
                    with_input=True,
                    value=sorted(list(instruments.keys()))[0] if instruments else None
                ).classes('col-12')
            
            with ui.row().classes('w-full items-center gap-2'):
                transaction_type = ui.select(
                    ['BUY', 'SELL'],
                    value='BUY',
                    label='Transaction Type'
                ).classes('col-4')
                
                order_type = ui.select(
                    ['MARKET', 'LIMIT'],
                    value='MARKET',
                    label='Order Type'
                ).classes('col-4')
                
                product_type = ui.select(
                    ['INTRADAY', 'DELIVERY'] if broker == 'Zerodha' else ['I', 'D'],
                    value='INTRADAY' if broker == 'Zerodha' else 'I',
                    label='Product Type'
                ).classes('col-4')
            
            with ui.row().classes('w-full items-center gap-2'):
                limit_price = ui.number(
                    label='Limit Price',
                    value=0,
                    min=0,
                    step=0.05,
                    format='%.2f',
                    visible=False
                ).classes('col-4')
                
                def update_limit_price():
                    limit_price.visible = order_type.value == 'LIMIT'
                
                order_type.on('update:model-value', lambda _: update_limit_price())
                update_limit_price()
                
                # Fetch current market price for the selected symbol
                async def fetch_market_price():
                    if symbol_select.value and symbol_select.value in instruments:
                        instrument_token = instruments[symbol_select.value]
                        ltp_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
                        if ltp_data and isinstance(ltp_data, list) and len(ltp_data) > 0:
                            price = ltp_data[0].get('last_price', 0)
                            if price > 0:
                                limit_price.value = price
                                return price
                    return 0
                
                ui.button('Get Market Price', on_click=lambda: asyncio.create_task(fetch_market_price())).classes('col-4')
            
            ui.label('Risk Management').classes('text-subtitle1 col-12 mt-4')
            
            with ui.row().classes('w-full items-center gap-2'):
                risk_per_trade = ui.number(
                    label='Risk per Trade (%)',
                    value=1.0,
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    format='%.1f'
                ).classes('col-4')
                
                stop_loss_type = ui.select(
                    ['Fixed Amount', 'Percentage of Entry', 'ATR Based'],
                    value='Fixed Amount',
                    label='Stop Loss Type'
                ).classes('col-4')
                
                # Dynamic fields based on stop loss type
                stop_loss_value = ui.number(
                    label='Stop Loss Value (₹)',
                    value=1.0,
                    min=0.1,
                    step=0.1,
                    format='%.1f'
                ).classes('col-4')
                
                target_value = ui.number(
                    label='Target Value (₹)',
                    value=2.0,
                    min=0.1,
                    step=0.1,
                    format='%.1f'
                ).classes('col-4 mt-2')
                
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
                
                stop_loss_type.on('update:model-value', lambda _: update_stop_loss_labels())
                
                # ATR Period field (only visible for ATR Based stop loss)
                atr_period = ui.number(
                    label='ATR Period',
                    value=14,
                    min=5,
                    max=50,
                    format='%d',
                    visible=False
                ).classes('col-4 mt-2')
                
                def update_atr_field():
                    atr_period.visible = stop_loss_type.value == 'ATR Based'
                
                stop_loss_type.on('update:model-value', lambda _: update_atr_field())
                update_atr_field()
            
            ui.label('Execution Settings').classes('text-subtitle1 col-12 mt-4')
            
            with ui.row().classes('w-full items-center gap-2'):
                execution_type = ui.radio(['Manual', 'Automatic'], value='Manual').classes('col-12')
                
                # Automatic execution settings
                check_interval = ui.number(
                    label='Check Interval (minutes)',
                    value=5,
                    min=1,
                    max=60,
                    format='%d',
                    visible=False
                ).classes('col-4')
                
                market_start_time = ui.time(
                    label='Market Start Time',
                    value='09:15',
                    visible=False
                ).classes('col-4')
                
                market_end_time = ui.time(
                    label='Market End Time',
                    value='15:30',
                    visible=False
                ).classes('col-4')
                
                def update_execution_fields():
                    check_interval.visible = execution_type.value == 'Automatic'
                    market_start_time.visible = execution_type.value == 'Automatic'
                    market_end_time.visible = execution_type.value == 'Automatic'
                
                execution_type.on('update:model-value', lambda _: update_execution_fields())
                update_execution_fields()
            
            # Place auto order function
            async def place_auto_order():
                if not symbol_select.value or symbol_select.value not in instruments:
                    ui.notify('Please select a valid symbol', type='negative')
                    return
                
                # Validate inputs
                if risk_per_trade.value <= 0:
                    ui.notify('Risk per trade must be greater than 0', type='negative')
                    return
                
                if stop_loss_value.value <= 0 or target_value.value <= 0:
                    ui.notify('Stop loss and target values must be greater than 0', type='negative')
                    return
                
                if order_type.value == 'LIMIT' and limit_price.value <= 0:
                    ui.notify('Limit price must be greater than 0', type='negative')
                    return
                
                if stop_loss_type.value == 'ATR Based' and atr_period.value < 5:
                    ui.notify('ATR period must be at least 5', type='negative')
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
                with ui.dialog() as dialog, ui.card():
                    ui.label('Confirm Auto Order').classes('text-h6')
                    ui.label(f"Symbol: {order_data['trading_symbol']}")
                    ui.label(f"Type: {order_data['transaction_type']} {order_data['order_type']}")
                    ui.label(f"Risk per Trade: {order_data['risk_per_trade']}%")
                    ui.label(f"Stop Loss Type: {order_data['stop_loss_type']}")
                    ui.label(f"Stop Loss Value: {order_data['stop_loss_value']}")
                    ui.label(f"Target Value: {order_data['target_value']}")
                    ui.label(f"Execution: {order_data['execution_type']}")
                    
                    if order_data['order_type'] == 'LIMIT':
                        ui.label(f"Limit Price: ₹{order_data['limit_price']:.2f}")
                    
                    if order_data['execution_type'] == 'Automatic':
                        ui.label(f"Check Interval: {order_data['check_interval']} minutes")
                        ui.label(f"Market Hours: {order_data['market_start_time']} - {order_data['market_end_time']}")
                    
                    with ui.row():
                        ui.button('Cancel', on_click=dialog.close).props('outline')
                        
                        async def confirm_auto_order():
                            dialog.close()
                            with ui.loading(text='Setting up auto order...'):
                                response = await fetch_api("/auto-orders/", method="POST", data=order_data)
                                if response and response.get('auto_order_id'):
                                    ui.notify(f"Auto order set up successfully: {response['auto_order_id']}", type='positive')
                                else:
                                    ui.notify("Failed to set up auto order", type='negative')
                        
                        ui.button('Confirm', on_click=lambda: asyncio.create_task(confirm_auto_order())).props('color=primary')
                
                dialog.open()
            
            ui.button('Set Up Auto Order', on_click=place_auto_order).props('color=primary').classes('mt-4')
        
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
        
        async def fetch_auto_orders():
            auto_orders = await fetch_api(f"/auto-orders/{broker}")
            if auto_orders and isinstance(auto_orders, list):
                await auto_orders_grid.update_grid_options({'rowData': auto_orders})
            else:
                await auto_orders_grid.update_grid_options({'rowData': []})
        
        await fetch_auto_orders()
        
        with ui.row().classes('w-full mt-4'):
            ui.button('Refresh', on_click=lambda: asyncio.create_task(fetch_auto_orders())).props('outline')
            
            async def cancel_auto_order():
                selected_rows = await auto_orders_grid.get_selected_rows()
                if not selected_rows:
                    ui.notify('Please select an auto order to cancel', type='warning')
                    return
                
                auto_order_id = selected_rows[0].get('auto_order_id')
                if not auto_order_id:
                    ui.notify('Invalid order selection', type='negative')
                    return
                
                with ui.loading(text='Cancelling auto order...'):
                    response = await fetch_api(f"/auto-orders/{auto_order_id}", method="DELETE")
                    if response and response.get('success'):
                        ui.notify('Auto order cancelled successfully', type='positive')
                        await fetch_auto_orders()  # Refresh the list
                    else:
                        ui.notify('Failed to cancel auto order', type='negative')
            
            ui.button('Cancel Selected', on_click=lambda: asyncio.create_task(cancel_auto_order())).props('color=negative')
