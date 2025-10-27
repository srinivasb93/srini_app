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
import math
import os, sys
import uuid
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, List
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def compute_trade_summary(
    transaction_type: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    quantity: int,
    brokerage_per_side: float,
    other_charges_pct: float
) -> Dict[str, float]:
    """Return derived trade metrics including brokerage-adjusted risk and profit figures."""
    summary = {
        'entry_charges': 0.0,
        'stop_exit_charges': 0.0,
        'target_exit_charges': 0.0,
        'total_investment': 0.0,
        'risk_move': 0.0,
        'risk_amount': 0.0,
        'profit_move': 0.0,
        'profit_amount': 0.0,
    }

    if quantity <= 0 or entry_price <= 0:
        return summary

    direction = 1 if str(transaction_type).upper() == 'BUY' else -1
    other_pct = max(other_charges_pct, 0.0)
    broker_fee = max(brokerage_per_side, 0.0)

    entry_value = entry_price * quantity
    summary['entry_charges'] = broker_fee + abs(entry_value) * (other_pct / 100.0)
    summary['total_investment'] = abs(entry_value) + summary['entry_charges']

    if stop_price > 0:
        stop_value = stop_price * quantity
        summary['stop_exit_charges'] = broker_fee + abs(stop_value) * (other_pct / 100.0)
        risk_move_per_share = (entry_price - stop_price) * direction
        if risk_move_per_share > 0:
            summary['risk_move'] = risk_move_per_share * quantity
            summary['risk_amount'] = summary['risk_move'] + summary['entry_charges'] + summary['stop_exit_charges']

    if target_price > 0:
        target_value = target_price * quantity
        summary['target_exit_charges'] = broker_fee + abs(target_value) * (other_pct / 100.0)
        profit_move_per_share = (target_price - entry_price) * direction
        if profit_move_per_share > 0:
            summary['profit_move'] = profit_move_per_share * quantity
            summary['profit_amount'] = summary['profit_move'] - (summary['entry_charges'] + summary['target_exit_charges'])

    summary = {k: round(v, 2) for k, v in summary.items()}
    return summary


def calculate_position_size(
    transaction_type: str,
    risk_amount: float,
    entry_price: float,
    stop_price: float
) -> int:
    """Return optimal quantity based on risk budget and price levels."""
    direction = 1 if str(transaction_type).upper() == 'BUY' else -1
    risk_amount = max(risk_amount, 0.0)
    entry_price = max(entry_price, 0.0)
    stop_price = max(stop_price, 0.0)

    risk_per_share = (entry_price - stop_price) * direction
    if risk_per_share <= 0 or risk_amount <= 0:
        return 0

    quantity = math.floor(risk_amount / risk_per_share)
    return max(int(quantity), 0)


def apply_button_style(button: ui.button, style: str) -> None:
    """Apply inline style with !important to override global overrides."""
    rules = []
    for rule in style.split(';'):
        rule = rule.strip()
        if not rule:
            continue
        if not rule.endswith('!important'):
            rule = f"{rule} !important"
        rules.append(rule)
    if rules:
        button.style('; '.join(rules))



def create_position_calculator_button(
    *,
    label: str,
    defaults: Dict[str, float],
    get_context: Callable[[], Dict[str, Any]],
    apply_callback: Callable[[Dict[str, Any]], None],
    button_classes: Optional[str] = None,
    button_style: Optional[str] = None,
    get_default_context: Optional[Callable[[], Dict[str, Any]]] = None,
    funds_fetcher: Optional[Callable[[], Awaitable[Any]]] = None,
    basket_totals_provider: Optional[Callable[[], Dict[str, float]]] = None,
    register_basket_listener: Optional[Callable[[Callable[[], None]], None]] = None,
    on_reset: Optional[Callable[[], Any]] = None,
) -> ui.button:
    """Attach a reusable position sizing calculator modal controlled by the returned button."""

    dialog_state = {'open': False}
    baseline_snapshot: Dict[str, Any] = {}
    context_snapshot: Dict[str, Any] = {}

    with ui.dialog().props('max-width=900px full-width').classes('position-sizing-dialog w-full max-w-[900px]') as dialog, ui.card().classes('position-sizing-card position-sizing-surface p-4 bg-gray-900 text-white gap-4 rounded-xl shadow-xl w-full'):
        dialog.style('width: min(50vw, 900px);')
        ui.label('Position Sizing Calculator').classes('text-xl font-semibold')

        with ui.row().classes('position-sizing-summary w-full items-center justify-between gap-2 bg-emerald-900/25 border border-emerald-500/30 px-3 py-5 rounded-lg'):
            funds_label = ui.label('Funds Available: --').classes('text-sm font-medium text-emerald-200 grow')
            basket_label = ui.label('Basket Allocation: --').classes('text-sm text-emerald-200 grow')
            net_funds_label = ui.label('After Basket: --').classes('text-sm font-semibold text-emerald-100 grow')

        with ui.row().classes('position-sizing-grid w-full gap-4'):
            with ui.column().classes('flex-1 gap-1'):
                ui.label('Capital').classes('text-sm font-medium text-gray-300')
                capital_input = ui.number(
                    value=defaults.get('capital', 100000.0),
                    min=0,
                    format='%.2f'
                ).classes('flex-1')

            with ui.column().classes('flex-1 gap-1'):
                ui.label('Risk % per Trade').classes('text-sm font-medium text-gray-300')
                risk_percent_input = ui.number(
                    value=defaults.get('risk_percent', 0.1),
                    min=0,
                    max=100,
                    step=0.1,
                    format='%.2f'
                ).classes('flex-1')

            with ui.column().classes('flex-1 gap-1'):
                ui.label('Risk Amount').classes('text-sm font-medium text-gray-300')
                risk_amount_input = ui.number(
                        value=defaults.get('risk_amount', 500.0),
                        min=0,
                        format='%.2f'
                    ).classes('flex-1')

        with ui.row().classes('position-sizing-grid w-full gap-4'):
            with ui.column().classes('flex-1 gap-1'):
                ui.label('Entry Price').classes('text-sm font-medium text-gray-300')
                entry_input = ui.number(
                    value=defaults.get('entry_price', 0.0),
                    min=0,
                    format='%.2f'
                ).classes('w-full')

            with ui.column().classes('flex-1 gap-1'):
                ui.label('Stop Loss').classes('text-sm font-medium text-gray-300')
                with ui.row().classes('items-center gap-2'):
                    stop_mode = ui.select(
                        options=['Absolute', 'Percentage'],
                        value=defaults.get('stop_mode', 'Percentage')
                    ).classes('w-32')
                    stop_input = ui.number(
                        value=defaults.get('stop_input', 2.0),
                        min=0,
                        step=0.1,
                        format='%.2f'
                    ).classes('flex-1')

            with ui.column().classes('flex-1 gap-1'):
                ui.label('Target').classes('text-sm font-medium text-gray-300')
                with ui.row().classes('items-center gap-2'):
                    target_mode = ui.select(
                        options=['Absolute', 'Percentage'],
                        value=defaults.get('target_mode', 'Percentage')
                    ).classes('w-32')
                    target_input = ui.number(
                        value=defaults.get('target_input', 4.0),
                        min=0,
                        step=0.1,
                        format='%.2f'
                    ).classes('flex-1')

        with ui.row().classes('position-sizing-grid w-full gap-4'):
            with ui.column().classes('flex-1 gap-1'):
                ui.label('Brokerage / Side').classes('text-sm font-medium text-gray-300')
                brokerage_input = ui.number(
                    value=defaults.get('brokerage', 20.0),
                    min=0,
                    format='%.2f'
                ).classes('flex-1')

            with ui.column().classes('flex-1 gap-1'):
                ui.label('Other Charges % / Side').classes('text-sm font-medium text-gray-300')
                charges_input = ui.number(
                        value=defaults.get('charges_pct', 0.03),
                        min=0,
                        format='%.4f'
                    ).classes('flex-1')

            quantity_preview = ui.label('Quantity: 0').classes('flex-1 text-lg font-semibold')

        with ui.column().classes('position-sizing-summary-panel w-full bg-gray-800/80 rounded-lg p-3 gap-2'):
            summary_title = ui.label('Trade Summary').classes('text-sm font-semibold text-cyan-300')
            summary_entry = ui.label('Entry: -').classes('text-sm text-gray-100')
            summary_levels = ui.label('Stop / Target: -').classes('text-sm text-gray-200')
            summary_risk = ui.label('Risk & Reward: -').classes('text-sm text-gray-200')
            summary_ratio = ui.label('Risk/Reward Ratio: -').classes('text-sm text-gray-200')
            summary_charges = ui.label('Charges: -').classes('text-xs text-gray-400')

        with ui.row().classes('position-sizing-actions w-full gap-3 justify-end mt-3'):
            reset_button = ui.button('Reset', icon='restart_alt').classes('bg-slate-700 text-white rounded-lg px-4')
            close_button = ui.button('Close').classes('bg-gray-600 text-white rounded-lg px-4')
            apply_button = ui.button('Apply to Order', icon='playlist_add_check').classes('bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg px-4')

    dialog.on('hide', lambda *_: dialog_state.update({'open': False}))

    state: Dict[str, Any] = {
        'transaction_type': 'BUY',
        'stop_mode': 'Absolute',
        'target_mode': 'Absolute',
    }
    update_lock = {'active': False}

    def format_currency(value: float) -> str:
        return f"Rs {value:,.2f}"

    def format_level(price: Optional[float], mode_value: str, raw_value: float) -> str:
        if price is None or price <= 0:
            return '-'
        if mode_value == 'Percentage':
            return f"{format_currency(price)} ({raw_value:.2f}%)"
        return format_currency(price)

    def resolve_level(mode_value: str, raw_value: float, entry_price: float, is_stop: bool) -> Optional[float]:
        entry_price = max(entry_price, 0.0)
        value = _safe_float(raw_value)
        if entry_price <= 0 or value <= 0:
            return None
        direction = 1 if str(state['transaction_type']).upper() == 'BUY' else -1
        if mode_value == 'Percentage':
            offset = entry_price * (value / 100.0)
            if is_stop:
                price = entry_price - offset if direction == 1 else entry_price + offset
            else:
                price = entry_price + offset if direction == 1 else entry_price - offset
            return max(price, 0.0)
        return value

    def sync_defaults_from_inputs() -> None:
        defaults.update({
            'capital': _safe_float(capital_input.value),
            'risk_percent': _safe_float(risk_percent_input.value),
            'risk_amount': _safe_float(risk_amount_input.value),
            'brokerage': _safe_float(brokerage_input.value),
            'charges_pct': _safe_float(charges_input.value),
            'entry_price': _safe_float(entry_input.value),
            'stop_mode': stop_mode.value,
            'target_mode': target_mode.value,
            'stop_input': _safe_float(stop_input.value),
            'target_input': _safe_float(target_input.value),
        })

    def get_basket_reserved() -> float:
        if not basket_totals_provider:
            return 0.0
        try:
            totals = basket_totals_provider() or {}
            return _safe_float(totals.get('required'), 0.0)
        except Exception as exc:
            logger.warning(f"Unable to compute basket totals: {exc}")
            return 0.0

    async def refresh_funds_info() -> None:
        available_value = None
        if funds_fetcher:
            try:
                raw_funds = await funds_fetcher()
                if isinstance(raw_funds, dict):
                    equity = raw_funds.get('equity') or {}
                    candidates = [
                        equity.get('available'),
                        equity.get('available_margin'),
                        raw_funds.get('available'),
                        raw_funds.get('available_margin'),
                    ]
                    available_dict = equity.get('available')
                    if isinstance(available_dict, dict):
                        candidates.append(available_dict.get('live_balance'))
                        candidates.append(available_dict.get('cash'))
                    for candidate in candidates:
                        value = _safe_float(candidate, None)
                        if value is not None and value > 0:
                            available_value = value
                            break
                    if available_value is None:
                        available_value = _safe_float(raw_funds.get('available'), None)
                else:
                    available_value = _safe_float(raw_funds, None)
            except Exception as exc:
                logger.warning(f"Unable to refresh funds: {exc}")

        reserved_value = get_basket_reserved()
        if available_value is None:
            funds_label.text = "Funds Available: --"
            net_funds_label.text = "After Basket: --"
        else:
            funds_label.text = f"Funds Available: {format_currency(available_value)}"
            net_amount = available_value - reserved_value
            net_funds_label.text = f"After Basket: {format_currency(net_amount)}"
        basket_label.text = f"Basket Allocation: {format_currency(reserved_value)}"
        funds_label.update()
        basket_label.update()
        net_funds_label.update()

    def set_inputs_from_data(data: Dict[str, Any], *, sync_defaults: bool = False) -> None:
        if not isinstance(data, dict):
            return
        update_lock['active'] = True
        capital_input.value = data.get('capital', capital_input.value)
        capital_input.update()
        risk_percent_input.value = data.get('risk_percent', risk_percent_input.value)
        risk_percent_input.update()
        risk_amount_input.value = data.get('risk_amount', risk_amount_input.value)
        risk_amount_input.update()
        update_lock['active'] = False

        entry_input.value = data.get('entry_price', entry_input.value)
        entry_input.update()
        stop_mode.value = data.get('stop_mode', stop_mode.value)
        stop_mode.update()
        stop_input.value = data.get('stop_input', stop_input.value)
        stop_input.update()
        target_mode.value = data.get('target_mode', target_mode.value)
        target_mode.update()
        target_input.value = data.get('target_input', target_input.value)
        target_input.update()
        brokerage_input.value = data.get('brokerage', brokerage_input.value)
        brokerage_input.update()
        charges_input.value = data.get('charges_pct', charges_input.value)
        charges_input.update()

        state['transaction_type'] = data.get('transaction_type', state.get('transaction_type', 'BUY'))
        update_summary()
        if sync_defaults:
            sync_defaults_from_inputs()

    def update_risk_amount_from_percent() -> None:
        if update_lock['active']:
            return
        capital = _safe_float(capital_input.value)
        percent = _safe_float(risk_percent_input.value)
        update_lock['active'] = True
        try:
            risk_amount_input.value = (capital * percent) / 100.0
            risk_amount_input.update()
        finally:
            update_lock['active'] = False
        update_summary()

    def update_risk_percent_from_amount() -> None:
        if update_lock['active']:
            return
        capital = _safe_float(capital_input.value)
        amount = _safe_float(risk_amount_input.value)
        update_lock['active'] = True
        try:
            percent = (amount / capital * 100.0) if capital > 0 else 0.0
            risk_percent_input.value = round(percent, 4)
            risk_percent_input.update()
        finally:
            update_lock['active'] = False
        update_summary()

    def update_summary() -> None:
        entry_price = _safe_float(entry_input.value)
        stop_raw = _safe_float(stop_input.value)
        target_raw = _safe_float(target_input.value)
        risk_amount = _safe_float(risk_amount_input.value)
        brokerage = _safe_float(brokerage_input.value)
        other_pct = _safe_float(charges_input.value)

        stop_price = resolve_level(stop_mode.value, stop_raw, entry_price, is_stop=True)
        target_price = resolve_level(target_mode.value, target_raw, entry_price, is_stop=False)
        stop_for_calc = stop_price if stop_price is not None else entry_price

        quantity = calculate_position_size(
            state['transaction_type'],
            risk_amount,
            entry_price,
            stop_for_calc
        )

        quantity_text = str(quantity) if quantity > 0 else '-'
        quantity_preview.text = f"Quantity: {quantity_text}"
        quantity_preview.update()

        summary = compute_trade_summary(
            state['transaction_type'],
            entry_price,
            stop_price or 0.0,
            target_price or 0.0,
            quantity,
            brokerage,
            other_pct
        )

        risk_per_share = (summary['risk_move'] / quantity) if quantity > 0 and summary['risk_move'] else 0.0
        reward_per_share = (summary['profit_move'] / quantity) if quantity > 0 and summary['profit_move'] else 0.0
        risk_reward_ratio = None
        if summary['risk_amount'] > 0 and summary['profit_amount'] > 0:
            risk_reward_ratio = summary['profit_amount'] / summary['risk_amount']

        summary.update({
            'stop_price': stop_price,
            'target_price': target_price,
            'stop_mode': stop_mode.value,
            'target_mode': target_mode.value,
            'stop_input': stop_raw,
            'target_input': target_raw,
            'risk_per_share': round(risk_per_share, 2) if risk_per_share else 0.0,
            'reward_per_share': round(reward_per_share, 2) if reward_per_share else 0.0,
            'risk_reward_ratio': round(risk_reward_ratio, 2) if risk_reward_ratio else None,
        })

        invest_text = format_currency(summary['total_investment']) if summary['total_investment'] else '-'
        summary_entry.text = (
            f"Entry: {format_currency(entry_price) if entry_price > 0 else '-'} | "
            f"Qty: {quantity_text} | Invest: {invest_text} | Risk Budget: {format_currency(risk_amount)}"
        )
        summary_entry.update()

        summary_levels.text = (
            f"Stop: {format_level(stop_price, stop_mode.value, stop_raw)} | "
            f"Target: {format_level(target_price, target_mode.value, target_raw)}"
        )
        summary_levels.update()

        risk_text = format_currency(summary['risk_amount']) if summary['risk_amount'] else '-'
        reward_text = format_currency(summary['profit_amount']) if summary['profit_amount'] else '-'
        risk_ps_text = f"{format_currency(summary['risk_per_share'])}/sh" if summary['risk_per_share'] else '-'
        reward_ps_text = f"{format_currency(summary['reward_per_share'])}/sh" if summary['reward_per_share'] else '-'
        summary_risk.text = f"Risk: {risk_text} ({risk_ps_text}) | Reward: {reward_text} ({reward_ps_text})"
        summary_risk.update()

        ratio_text = f"{summary['risk_reward_ratio']:.2f}" if summary.get('risk_reward_ratio') else '-'
        summary_ratio.text = f"Risk/Reward Ratio: {ratio_text}"
        summary_ratio.update()

        summary_charges.text = (
            "Charges: "
            f"Entry {format_currency(summary['entry_charges'])} | "
            f"Stop Exit {format_currency(summary['stop_exit_charges'])} | "
            f"Target Exit {format_currency(summary['target_exit_charges'])}"
        )
        summary_charges.update()

        state['quantity'] = quantity
        state['summary'] = summary
        state['resolved_stop'] = stop_price
        state['resolved_target'] = target_price
        state['stop_mode'] = stop_mode.value
        state['target_mode'] = target_mode.value
        state['stop_raw'] = stop_raw
        state['target_raw'] = target_raw

    capital_input.on_value_change(lambda _: update_risk_amount_from_percent())
    risk_percent_input.on_value_change(lambda _: update_risk_amount_from_percent())
    risk_amount_input.on_value_change(lambda _: update_risk_percent_from_amount())

    for field in (entry_input, stop_input, target_input, brokerage_input, charges_input):
        field.on_value_change(lambda _: update_summary())

    stop_mode.on_value_change(lambda _: update_summary())
    target_mode.on_value_change(lambda _: update_summary())

    async def handle_reset() -> None:
        baseline_data = get_default_context() if callable(get_default_context) else None
        if isinstance(baseline_data, dict) and not baseline_data:
            baseline_data = None
        if baseline_data is None:
            baseline_data = dict(baseline_snapshot)
        if on_reset:
            try:
                maybe_result = on_reset()
                if inspect.isawaitable(maybe_result):
                    await maybe_result
            except Exception as exc:
                logger.warning(f"Reset callback failed: {exc}")
        if callable(get_default_context):
            refreshed_default = get_default_context() or {}
            if refreshed_default:
                baseline_data = refreshed_default
        set_inputs_from_data(baseline_data or {}, sync_defaults=True)
        await refresh_funds_info()

    async def open_dialog_async() -> None:
        context = get_context() or {}
        context_snapshot.clear()
        context_snapshot.update(context)
        if callable(get_default_context):
            baseline_snapshot.clear()
            baseline_snapshot.update(get_default_context() or {})
        else:
            baseline_snapshot.clear()
            baseline_snapshot.update(context)

        set_inputs_from_data(context)
        state['transaction_type'] = context.get('transaction_type', state.get('transaction_type', 'BUY'))
        state['context'] = context
        dialog_state['open'] = True
        await refresh_funds_info()
        dialog.open()

    def close_dialog() -> None:
        dialog_state['open'] = False
        dialog.close()

    def on_apply() -> None:
        quantity = state.get('quantity', 0)
        if quantity <= 0 and not context_has_risk_only():
            ui.notify('Unable to derive quantity. Please adjust risk inputs.', type='warning')
            return

        stop_price = state.get('resolved_stop')
        target_price = state.get('resolved_target')

        result = {
            'quantity': quantity if not context_has_risk_only() else None,
            'stop_loss': stop_price,
            'target': target_price,
            'risk_amount': _safe_float(risk_amount_input.value),
            'entry_price': _safe_float(entry_input.value),
            'summary': state.get('summary', {}),
            'transaction_type': state.get('transaction_type', 'BUY'),
            'risk_only': context_has_risk_only(),
            'stop_mode': stop_mode.value,
            'target_mode': target_mode.value,
            'stop_input': state.get('stop_raw'),
            'target_input': state.get('target_raw'),
        }

        if context_has_risk_only():
            result['risk_per_trade'] = _safe_float(risk_amount_input.value)

        sync_defaults_from_inputs()
        close_dialog()
        apply_callback(result)

    def context_has_risk_only() -> bool:
        context = state.get('context', {}) or {}
        return context.get('risk_only', False)

    close_button.on_click(close_dialog)
    apply_button.on_click(on_apply)
    reset_button.on_click(lambda: asyncio.create_task(handle_reset()))

    def basket_listener() -> None:
        if dialog_state['open']:
            try:
                asyncio.create_task(refresh_funds_info())
            except RuntimeError:
                pass

    if register_basket_listener:
        register_basket_listener(basket_listener)

    button = ui.button(label, icon='calculate', on_click=lambda: asyncio.create_task(open_dialog_async()))
    button.classes(button_classes or 'bg-slate-700 text-white')
    if button_style:
        apply_button_style(button, button_style)
    return button

async def render_order_management(fetch_api, user_storage, instruments):
    """Enhanced order management page with compact header and full-width forms"""

    broker = user_storage.get('default_broker', 'Zerodha')  # Changed from 'broker' to 'default_broker'

    position_calc_defaults = user_storage.get('position_calc_defaults')
    if not position_calc_defaults:
        position_calc_defaults = {
            'capital': 100000.0,
            'risk_percent': 1.0,
            'risk_amount': 1000.0,
            'brokerage': 20.0,
            'charges_pct': 0.03,
            'entry_price': 0.0,
            'stop_mode': 'Percentage',
            'target_mode': 'Percentage',
            'stop_input': 2.0,
            'target_input': 4.0,
        }
        user_storage['position_calc_defaults'] = position_calc_defaults
    else:
        position_calc_defaults.setdefault('stop_mode', 'Percentage')
        position_calc_defaults.setdefault('target_mode', 'Percentage')
        position_calc_defaults.setdefault('stop_input', 2.0)
        position_calc_defaults.setdefault('target_input', 4.0)
        position_calc_defaults.setdefault('entry_price', 0.0)

    calc_template = {
        'capital': _safe_float(position_calc_defaults.get('capital'), 0.0),
        'risk_percent': _safe_float(position_calc_defaults.get('risk_percent'), 0.1),
        'risk_amount': _safe_float(position_calc_defaults.get('risk_amount'), 500.0),
        'brokerage': _safe_float(position_calc_defaults.get('brokerage'), 20.0),
        'charges_pct': _safe_float(position_calc_defaults.get('charges_pct'), 0.03),
    }
    if calc_template['risk_amount'] <= 0 and calc_template['capital'] > 0:
        calc_template['risk_amount'] = calc_template['capital'] * (calc_template['risk_percent'] / 100.0)
    position_calc_defaults['risk_amount'] = calc_template['risk_amount']
    user_storage['_position_calc_template'] = calc_template

    user_preferences: Dict[str, Any] = {}
    if fetch_api:
        try:
            prefs_response = await fetch_api("/user/preferences")
            if prefs_response and prefs_response.get('status') == 'success':
                user_preferences = prefs_response.get('preferences') or {}
        except Exception as pref_error:
            logger.warning(f"Unable to load user preferences for position calculator: {pref_error}")

    if user_preferences:
        capital_pref = user_preferences.get('portfolio_size_limit') or user_preferences.get('position_size_limit')
        if capital_pref is not None:
            position_calc_defaults['capital'] = _safe_float(capital_pref, position_calc_defaults['capital'])

        risk_percent_pref = user_preferences.get('risk_per_trade')
        if risk_percent_pref is not None:
            position_calc_defaults['risk_percent'] = _safe_float(risk_percent_pref, position_calc_defaults['risk_percent'])

        risk_amount_pref = user_preferences.get('risk_amount')
        if risk_amount_pref is not None:
            position_calc_defaults['risk_amount'] = _safe_float(risk_amount_pref, position_calc_defaults['risk_amount'])
        else:
            position_calc_defaults['risk_amount'] = (
                position_calc_defaults['capital'] * position_calc_defaults['risk_percent'] / 100.0
            )

        brokerage_pref = (
            user_preferences.get('brokerage_per_side') or
            user_preferences.get('default_brokerage')
        )
        if brokerage_pref is not None:
            position_calc_defaults['brokerage'] = _safe_float(brokerage_pref, position_calc_defaults['brokerage'])

        charges_pref = (
            user_preferences.get('charges_pct') or
            user_preferences.get('default_charges_pct')
        )
        if charges_pref is not None:
            position_calc_defaults['charges_pct'] = _safe_float(charges_pref, position_calc_defaults['charges_pct'])

    user_storage['position_calc_defaults'] = position_calc_defaults

    # Fetch instruments if not already available
    if not instruments:
        instruments_data = await fetch_api(f"/instruments/{broker}/?exchange=NSE")
        if instruments_data:
            equity_instruments = [i for i in instruments_data if i.get('segment') == 'NSE' and i.get('instrument_type') == 'EQ']
            instruments.update({i['trading_symbol']: i['instrument_token'] for i in equity_instruments})

    # Compact header with theme-aware styling
    with ui.row().classes("w-full items-center justify-between p-2 order-mgmt-header theme-surface-elevated"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("shopping_cart", size="1.5rem").classes("theme-text-accent")
            ui.label("Order Management").classes("text-2xl font-bold theme-text-primary")
            ui.chip("TRADING", color="cyan").classes("text-xs")

        with ui.row().classes("items-center gap-4"):
            with ui.tabs().classes('order-tabs theme-surface-elevated rounded-lg') as tabs:
                regular_tab = ui.tab('Regular Orders', icon='flash_on').classes('px-4 py-2')
                # scheduled_tab = ui.tab('Scheduled Orders', icon='schedule').classes('px-4 py-2')  # Merged into Regular Orders
                gtt_tab = ui.tab('GTT Orders', icon='compare_arrows').classes('px-4 py-2')
                auto_tab = ui.tab('Auto Orders', icon='smart_toy').classes('px-4 py-2')
            basket_button_container = ui.row().classes("items-center gap-2 ml-4")
    
    # Set default tab to Regular Orders
    tabs.set_value(regular_tab)

    basket_state = user_storage.get('order_basket')
    if basket_state is None:
        basket_state = []
        user_storage['order_basket'] = basket_state

    basket_processing = {'active': False}
    basket_observers: List[Callable[[], None]] = []

    def register_basket_observer(callback: Callable[[], None]) -> None:
        if callable(callback) and callback not in basket_observers:
            basket_observers.append(callback)

    def notify_basket_observers() -> None:
        for observer in list(basket_observers):
            try:
                observer()
            except Exception as exc:
                logger.warning(f"Basket observer error: {exc}")

    with basket_button_container:
        with ui.row().classes('items-center gap-3'):
            # Basket Orders Button
            with ui.row().classes('items-center gap-2 bg-gradient-to-r from-orange-500 to-pink-500 px-3 py-1.5 rounded-full shadow-lg'):
                basket_button = ui.button('Basket Orders', icon='shopping_basket').props('flat dense').classes('text-white bg-transparent hover:bg-white/10 px-2 min-w-0 rounded-full')
                basket_count_badge = ui.badge(str(len(basket_state))).classes('bg-white text-orange-600 text-xs font-semibold px-2 rounded-full')
            
            # Orderbook Button  
            ui.button('Orderbook', icon='library_books', on_click=lambda: ui.navigate.to('/order-book'))\
                .props('flat').classes('bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-4 py-1.5 rounded-full shadow-lg hover:shadow-xl transition-all')

    with ui.dialog().classes('basket-orders-dialog max-w-3xl') as basket_dialog:
        with ui.card().classes('basket-orders-card p-4 bg-gray-900 text-white gap-3 w-full'):
            with ui.row().classes('basket-orders-header w-full items-center justify-between'):
                ui.label('Order Basket').classes('text-xl font-semibold')
                ui.button(icon='close', on_click=basket_dialog.close).props('flat round dense').classes('text-white hover:bg-gray-700/70')
            basket_summary_label = ui.label('Basket is empty').classes('basket-orders-summary text-sm text-gray-300')
            basket_list_column = ui.column().classes('basket-orders-list w-full gap-3 py-2')
            basket_feedback_column = ui.column().classes('basket-orders-feedback w-full gap-3')
            with ui.row().classes('basket-orders-actions w-full justify-between gap-2'):
                clear_basket_button = ui.button('Clear Basket', icon='delete_sweep').classes('bg-gray-700 text-white')
                place_all_button = ui.button('Place Basket Orders', icon='playlist_play').classes('bg-green-600 text-white')

    def format_currency(value: Optional[float]) -> str:
        if value is None:
            return "-"
        return f"Rs {value:,.2f}"

    def describe_basket_entry(entry: Dict[str, Any]) -> List[str]:
        payload = entry.get('payload', {})
        entry_type = entry.get('type', 'Order')
        symbol = payload.get('trading_symbol') or payload.get('symbol') or '-'
        quantity = payload.get('quantity')
        qty_text = str(quantity) if quantity is not None else ''
        lines = [f"{entry_type}: {payload.get('transaction_type', '').upper()} {qty_text} {symbol}"]

        if entry_type == 'Regular':
            lines.append(f"{payload.get('order_type', '')} @ {format_currency(payload.get('price'))} ({payload.get('product_type', '-')})")
            if payload.get('stop_loss'):
                lines.append(f"SL {format_currency(payload.get('stop_loss'))}")
            if payload.get('target'):
                lines.append(f"Target {format_currency(payload.get('target'))}")
        elif entry_type == 'Scheduled':
            if payload.get('schedule_datetime'):
                lines.append(f"Schedule {payload['schedule_datetime']}")
            if payload.get('price'):
                lines.append(f"Price {format_currency(payload.get('price'))}")
            if payload.get('stop_loss'):
                lines.append(f"SL {format_currency(payload.get('stop_loss'))}")
            if payload.get('target'):
                lines.append(f"Target {format_currency(payload.get('target'))}")
        elif entry_type == 'GTT':
            if payload.get('trigger_type'):
                lines.append(f"Type {payload['trigger_type']}")
            if payload.get('trigger_price'):
                lines.append(f"Trigger {format_currency(payload.get('trigger_price'))}")
            if payload.get('limit_price'):
                lines.append(f"Limit {format_currency(payload.get('limit_price'))}")
        elif entry_type == 'Auto':
            lines.append(f"Risk {format_currency(payload.get('risk_per_trade'))}")
            lines.append(f"SL {payload.get('stop_loss_type')} {payload.get('stop_loss_value')}")
            lines.append(f"Target {payload.get('target_value')}")

        return [line for line in lines if line]

    def compute_basket_metrics(entry: Dict[str, Any]) -> Tuple[float, float, float]:
        payload = entry.get('payload', {}) or {}
        entry_type = entry.get('type', '')

        def safe_amount(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        summary_info = payload.get('summary') or {}
        total_required = 0.0
        total_risk = 0.0
        total_reward = 0.0

        quantity = safe_amount(payload.get('quantity'))
        transaction_type = str(payload.get('transaction_type', 'BUY')).upper()
        direction = 1 if transaction_type != 'SELL' else -1

        price_candidates = [
            payload.get('estimated_price'),
            payload.get('price'),
            payload.get('limit_price'),
            payload.get('entry_price'),
            payload.get('last_price'),
        ]
        entry_price = 0.0
        for candidate in price_candidates:
            value = safe_amount(candidate)
            if value > 0:
                entry_price = value
                break

        if entry_type == 'Auto':
            total_risk += safe_amount(payload.get('risk_per_trade'))
            target_val = safe_amount(payload.get('target_value'))
            if target_val > 0:
                total_reward += target_val
        else:
            if quantity > 0 and entry_price > 0:
                total_required += abs(quantity * entry_price)

            stop_price = payload.get('stop_loss')
            if stop_price is None:
                stop_price = payload.get('stop_loss_value')
            stop_price = safe_amount(stop_price)

            target_price = payload.get('target')
            if target_price is None:
                target_price = payload.get('target_value')
            target_price = safe_amount(target_price)

            if entry_price > 0 and stop_price > 0 and quantity > 0:
                move = (entry_price - stop_price) * direction * quantity
                if move > 0:
                    total_risk += move

            if entry_price > 0 and target_price > 0 and quantity > 0:
                move = (target_price - entry_price) * direction * quantity
                if move > 0:
                    total_reward += move

        if total_required == 0 and quantity > 0 and summary_info.get('total_investment'):
            total_required += safe_amount(summary_info.get('total_investment'))
        if total_risk == 0 and summary_info.get('risk_amount'):
            total_risk += safe_amount(summary_info.get('risk_amount'))
        if total_reward == 0 and summary_info.get('profit_amount'):
            total_reward += safe_amount(summary_info.get('profit_amount'))

        return total_required, total_risk, total_reward

    def calculate_basket_totals() -> Dict[str, float]:
        total_required = 0.0
        total_risk = 0.0
        total_reward = 0.0
        for entry in basket_state:
            value, risk, reward = compute_basket_metrics(entry)
            total_required += value
            total_risk += risk
            total_reward += reward
        return {
            'required': total_required,
            'risk': total_risk,
            'reward': total_reward,
        }

    def update_basket_badge() -> None:
        count = len(basket_state)
        basket_count_badge.text = str(count)
        basket_count_badge.visible = count > 0
        basket_count_badge.update()
        if count == 0:
            basket_summary_label.text = "Basket is empty"
            basket_summary_label.update()

    def refresh_basket_view() -> None:
        basket_list_column.clear()
        total_required = 0.0
        total_risk = 0.0
        total_reward = 0.0

        if not basket_state:
            with basket_list_column:
                ui.label('Basket is empty. Add orders to execute them together.').classes('text-gray-400 text-sm')
            basket_summary_label.text = "Basket is empty"
            basket_summary_label.update()
            basket_feedback_column.clear()
            basket_feedback_column.update()
            basket_list_column.update()
            return

        for entry in basket_state:
            value, risk, reward = compute_basket_metrics(entry)
            total_required += value
            total_risk += risk
            total_reward += reward
            with basket_list_column:
                with ui.card().props('flat bordered').classes('bg-gray-900/85 border border-gray-700 px-4 py-3 rounded-xl w-full text-white shadow-inner basket-order-card').style('max-width:100%; overflow:visible; border-radius:16px !important;'):
                    for line in describe_basket_entry(entry):
                        ui.label(line).classes('text-sm text-gray-100 break-words whitespace-normal')
                    with ui.row().classes('justify-end'):
                        ui.button(
                            'Remove',
                            icon='remove_circle',
                            on_click=lambda _, entry_id=entry['id']: remove_from_basket(entry_id)
                        ).classes('text-xs bg-red-500/80 hover:bg-red-500 text-white px-3 py-1 rounded-full')

        basket_summary_label.text = (
            f"{len(basket_state)} order(s) | "
            f"Capital: {format_currency(total_required)} | "
            f"Risk: {format_currency(total_risk)} | "
            f"Target P/L: {format_currency(total_reward)}"
        )
        basket_summary_label.update()
        basket_list_column.update()

    def remove_from_basket(entry_id: str) -> None:
        original_len = len(basket_state)
        basket_state[:] = [item for item in basket_state if item.get('id') != entry_id]
        if len(basket_state) != original_len:
            ui.notify('Removed order from basket', type='info')
        update_basket_badge()
        refresh_basket_view()
        notify_basket_observers()

    async def place_basket_orders(event=None) -> None:
        if basket_processing['active']:
            return
        if not basket_state:
            ui.notify('Basket is empty', type='warning')
            return

        basket_processing['active'] = True
        place_all_button.props('loading')
        basket_feedback_column.clear()
        basket_feedback_column.update()

        results = []
        for entry in list(basket_state):
            payload = entry.get('payload', {})
            try:
                response = await fetch_api(entry['endpoint'], method="POST", data=payload)
                success_key = entry.get('success_key')
                success = bool(response and success_key and response.get(success_key))
                results.append((entry, success, response))
                if success:
                    basket_state.remove(entry)
            except Exception as exc:
                results.append((entry, False, {'error': str(exc)}))

        update_basket_badge()
        refresh_basket_view()
        notify_basket_observers()

        success_count = sum(1 for _, success, _ in results if success)
        failure_items = [(entry, resp) for entry, success, resp in results if not success]

        if success_count:
            ui.notify(f"Placed {success_count} basket order(s)", type='positive')

        basket_feedback_column.clear()
        if failure_items:
            with basket_feedback_column:
                ui.label('Failed Orders').classes('text-red-400 text-sm font-semibold')
                for entry, resp in failure_items:
                    payload = entry.get('payload', {})
                    symbol = payload.get('trading_symbol') or '-'
                    ui.label(f"{entry.get('type', 'Order')} - {symbol}").classes('text-sm text-red-300')
                    ui.label(str(resp)).classes('text-xs text-gray-400 break-all')
            ui.notify(f"{len(failure_items)} order(s) failed. They remain in the basket for retry.", type='warning')

        place_all_button.props(remove='loading')
        basket_processing['active'] = False

    def clear_basket(event=None) -> None:
        if not basket_state:
            ui.notify('Basket already empty', type='info')
            return
        basket_state.clear()
        update_basket_badge()
        refresh_basket_view()
        notify_basket_observers()
        ui.notify('Cleared basket', type='info')

    def add_order_to_basket(entry: Dict[str, Any]) -> None:
        entry = dict(entry)
        entry.setdefault('id', str(uuid.uuid4()))
        basket_state.append(entry)
        update_basket_badge()
        notify_basket_observers()
        ui.notify('Order added to basket', type='positive')

    def open_basket(event=None) -> None:
        refresh_basket_view()
        basket_dialog.open()

    basket_button.on_click(open_basket)
    clear_basket_button.on_click(clear_basket)
    place_all_button.on_click(place_basket_orders)

    update_basket_badge()

    basket_controller = {
        'add': add_order_to_basket,
        'refresh': refresh_basket_view,
        'state': basket_state,
        'compute_metrics': compute_basket_metrics,
        'totals': calculate_basket_totals,
        'register_observer': register_basket_observer,
    }

    # Tab panels with lazy loading
    with ui.tab_panels(tabs).classes('w-full p-.5'):
        with ui.tab_panel(regular_tab):
            await render_regular_orders(fetch_api, user_storage, instruments, broker, position_calc_defaults, basket_controller)

        # Scheduled Orders tab merged into Regular Orders with Schedule toggle
        # with ui.tab_panel(scheduled_tab):
        #     scheduled_container = ui.column().classes('w-full')
        #     scheduled_loaded = False

        #     async def load_scheduled_orders():
        #         nonlocal scheduled_loaded
        #         if not scheduled_loaded:
        #             scheduled_container.clear()
        #             with scheduled_container:
        #                 await render_scheduled_orders(fetch_api, user_storage, instruments, broker, position_calc_defaults, basket_controller)
        #             scheduled_loaded = True

        #     scheduled_tab.on('click', lambda: asyncio.create_task(load_scheduled_orders()))

        with ui.tab_panel(gtt_tab):
            gtt_container = ui.column().classes('w-full')
            gtt_loaded = False

            async def load_gtt_orders():
                nonlocal gtt_loaded
                if not gtt_loaded:
                    gtt_container.clear()
                    with gtt_container:
                        await render_gtt_orders(fetch_api, user_storage, instruments, broker, position_calc_defaults, basket_controller)
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
                        await render_auto_orders(fetch_api, user_storage, instruments, broker, position_calc_defaults, basket_controller)
                    auto_loaded = True

            auto_tab.on('click', lambda: asyncio.create_task(load_auto_orders()))

async def render_regular_orders(fetch_api, user_storage, instruments, broker, position_calc_defaults, basket_controller):
    """Regular orders form with stop_loss and target fields"""

    market_price_state = {'last_price': 0.0}
    template_defaults = user_storage.get('_position_calc_template', {})

    def apply_symbol_defaults(last_price: float) -> None:
        template = user_storage.get('_position_calc_template', template_defaults) or {}
        position_calc_defaults['capital'] = template.get('capital', position_calc_defaults.get('capital', 0.0))
        position_calc_defaults['risk_percent'] = template.get('risk_percent', position_calc_defaults.get('risk_percent', 0.0))
        position_calc_defaults['brokerage'] = template.get('brokerage', position_calc_defaults.get('brokerage', 0.0))
        position_calc_defaults['charges_pct'] = template.get('charges_pct', position_calc_defaults.get('charges_pct', 0.0))
        position_calc_defaults['entry_price'] = _safe_float(last_price, 0.0)
        position_calc_defaults['stop_input'] = 0.0
        position_calc_defaults['target_input'] = 0.0
        position_calc_defaults['stop_mode'] = 'Absolute'
        position_calc_defaults['target_mode'] = 'Absolute'
        capital_val = position_calc_defaults.get('capital', 0.0)
        risk_percent_val = position_calc_defaults.get('risk_percent', 0.0)
        position_calc_defaults['risk_amount'] = capital_val * (risk_percent_val / 100.0) if capital_val and risk_percent_val else 0.0

    with ui.card().classes('w-full enhanced-card'):
        with ui.row().classes("w-full items-center justify-between p-2 border-b order-form-header theme-surface-elevated"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("add_circle", size="1.2rem").classes("theme-text-accent")
                ui.label("Place Regular Order").classes("text-lg font-semibold theme-text-primary")
            ui.chip("LIVE", color="green").classes("text-xs")

        with ui.column().classes("p-2 gap-3 w-full"):
            validation_state = {'symbol': True, 'quantity': True, 'price': True, 'trigger_price': True}

            # Main Layout: Two Columns (Order Entry + Market Data) - 80% : 20%
            with ui.row().classes('w-full gap-4 mb-2'):
                # LEFT COLUMN: Order Entry Fields (80%)
                with ui.column().classes('flex-[4] gap-3'):
                    # Index Filter and Symbol Selection
                    with ui.row().classes('w-full gap-3'):
                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Index Filter").classes("text-sm font-medium theme-text-primary")
                            index_select = ui.select(
                                options=['NIFTY_50', 'NIFTY_NEXT_50', 'All Instruments'],
                                value='NIFTY_50'
                            ).classes('w-full')

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Trading Symbol").classes("text-sm font-medium theme-text-primary")

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
                                if symbol_select.value:
                                    await update_market_price(symbol_select.value, reset_defaults=True)

                            index_select.on_value_change(update_symbol_options)

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Transaction Type").classes("text-sm font-medium theme-text-primary")
                            transaction_type = ui.select(
                                options=['BUY', 'SELL'],
                                value='BUY'
                            ).classes('w-full')

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Product Type").classes("text-sm font-medium theme-text-primary")
                            product_type = ui.select(
                                options=['MIS', 'CNC'] if broker == 'Zerodha' else ['I', 'D'],
                                value='CNC' if broker == 'Zerodha' else 'D'
                            ).classes('w-full')

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Quantity").classes("text-sm font-medium theme-text-primary")
                            quantity = ui.number(
                                value=1,
                                min=1,
                                format='%d'
                            ).classes('w-full')
                            quantity.on_value_change(lambda e: validation_state.update({'quantity': _safe_int(e.value) > 0}))

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Disclosed Qty").classes("text-sm font-medium theme-text-primary")
                            disclosed_quantity = ui.number(
                                value=0,
                                min=0,
                                format='%d'
                            ).classes('w-full')
                    
                    # Price Configuration Row
                    with ui.row().classes('w-full gap-3'):
                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Order Type").classes("text-sm font-medium theme-text-primary")
                            order_type = ui.select(
                                options=['MARKET', 'LIMIT', 'SL', 'SL-M'],
                                value='LIMIT'
                            ).classes('w-full')

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Price").classes("text-sm font-medium theme-text-primary")
                            price_field = ui.number(
                                value=0,
                                min=0,
                                step=0.05,
                                format='%.2f'
                            ).classes('w-full')
                            price_field.on_value_change(lambda e: validation_state.update(
                                {'price': (_safe_float(e.value) > 0) if order_type.value in ['LIMIT', 'SL'] else True}))

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Trigger Price").classes("text-sm font-medium theme-text-primary")
                            trigger_price_field = ui.number(
                                value=0,
                                min=0,
                                step=0.05,
                                format='%.2f'
                            ).classes('w-full')
                            trigger_price_field.on_value_change(lambda e: validation_state.update(
                                {'trigger_price': (_safe_float(e.value) > 0) if order_type.value in ['SL', 'SL-M'] else True}))

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Validity").classes("text-sm font-medium theme-text-primary")
                            validity = ui.select(
                                options=['DAY', 'IOC'],
                                value='DAY'
                            ).classes('w-full')

                        with ui.column().classes('flex-1 gap-2'):
                            ui.label("AMO").classes("text-sm font-medium theme-text-primary")
                            is_amo_checkbox = ui.checkbox('After Market Order (AMO)').classes('theme-text-primary')
                            is_amo_checkbox.value = False
                    
                    # Schedule Order and Risk Management in same row (25% : 75%)
                    with ui.row().classes('w-full gap-4 mt-2'):
                        # LEFT: Schedule Order Section (25%)
                        with ui.column().classes('flex-1 gap-2 bg-purple-500/10 border border-purple-500/30 p-3 rounded-lg'):
                            schedule_order_toggle = ui.checkbox('Schedule Order').classes('theme-text-primary font-semibold')
                            schedule_order_toggle.value = False
                            
                            schedule_fields_col = ui.column().classes('w-full gap-2 mt-2')
                            with schedule_fields_col:
                                with ui.row().classes('w-full gap-2'):
                                    with ui.column().classes("flex-1 gap-1"):
                                        ui.label("Date").classes("text-xs font-medium theme-text-secondary")
                                        schedule_date = ui.input(
                                            value=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                                        ).props("dense type=date").classes("w-full")
                                    
                                    with ui.column().classes("flex-1 gap-1"):
                                        ui.label("Time").classes("text-xs font-medium theme-text-secondary")
                                        schedule_time = ui.input(
                                            value=(datetime.now() + timedelta(minutes=15)).strftime("%H:%M")
                                        ).props("dense type=time").classes("w-full")
                            
                            # Initially hide schedule fields
                            schedule_fields_col.visible = False
                            
                            def toggle_schedule_fields():
                                schedule_fields_col.visible = schedule_order_toggle.value
                                validation_state['schedule_datetime'] = not schedule_order_toggle.value or bool(schedule_date.value and schedule_time.value)
                                schedule_fields_col.update()
                            
                            schedule_order_toggle.on_value_change(toggle_schedule_fields)
                        
                        # RIGHT: Risk Management Section (75%)
                        with ui.column().classes('flex-[3] gap-2 theme-surface-elevated p-3 rounded-lg'):
                            ui.label('Risk Management').classes('text-sm font-semibold theme-text-accent mb-1')
                            
                            # Risk Management Type Toggle
                            risk_management_type = ui.toggle({
                                'regular': 'Regular SL/Target',
                                'trailing': 'Trailing Stop Loss'
                            }, value='regular').classes('theme-toggle')
                            
                            # Regular Risk Management Section
                            regular_risk_section = ui.column().classes('w-full mt-2')
                            with regular_risk_section:
                                with ui.row().classes('w-full gap-3'):
                                    with ui.column().classes("flex-1 gap-2"):
                                        ui.label("Stop Loss (₹)").classes("text-sm font-medium theme-text-secondary")
                                        regular_stop_loss = ui.number(
                                            value=0,
                                            min=0,
                                            step=0.05,
                                            format='%.2f'
                                        ).classes('w-full')

                                    with ui.column().classes("flex-1 gap-2"):
                                        ui.label("Target (₹)").classes("text-sm font-medium theme-text-secondary")
                                        regular_target = ui.number(
                                            value=0,
                                            min=0,
                                            step=0.05,
                                            format='%.2f'
                                        ).classes('w-full')

                            # Trailing Stop Loss Section
                            trailing_risk_section = ui.column().classes('w-full mt-2')
                            trailing_risk_section.visible = False
                            with trailing_risk_section:
                                with ui.row().classes('w-full gap-3 mb-1'):
                                    with ui.column().classes("flex-1 gap-2"):
                                        ui.label("Stop Loss (₹)").classes("text-sm font-medium theme-text-secondary")
                                        ui.label("Initial stop loss").classes("text-xs theme-text-muted")
                                        trailing_stop_loss = ui.number(
                                            value=0,
                                            min=0,
                                            step=0.05,
                                            format='%.2f'
                                        ).classes('w-full')

                                    with ui.column().classes("flex-1 gap-2"):
                                        ui.label("Target (₹)").classes("text-sm font-medium theme-text-secondary")
                                        ui.label("Fixed target or 0").classes("text-xs theme-text-muted")
                                        trailing_target = ui.number(
                                            value=0,
                                            min=0,
                                            step=0.05,
                                            format='%.2f'
                                        ).classes('w-full')

                                    with ui.column().classes("flex-1 gap-2"):
                                        ui.label("Trail Start (%)").classes("text-sm font-medium theme-text-secondary")
                                        ui.label("% gain to activate").classes("text-xs theme-text-muted")
                                        trail_start_target_percent = ui.number(
                                            value=5.0,
                                            min=0.1,
                                            max=50.0,
                                            step=0.1,
                                            format='%.1f'
                                        ).classes('w-full')

                                    with ui.column().classes("flex-1 gap-2"):
                                        ui.label("Trailing SL (%)").classes("text-sm font-medium theme-text-secondary")
                                        ui.label("% below highest").classes("text-xs theme-text-muted")
                                        trailing_stop_loss_percent = ui.number(
                                            value=2.0,
                                            min=0.1,
                                            max=20.0,
                                            step=0.1,
                                            format='%.1f'
                                        ).classes('w-full')

                                # Info section for trailing stop loss (compact)
                                with ui.card().classes('bg-blue-900/30 border-blue-600/50 p-1 mt-1'):
                                    with ui.row().classes('gap-1 items-start'):
                                        ui.icon('info').classes('text-blue-400 text-xs')
                                        ui.label('Activates at trail start | Adjusts as price rises | Triggers on drop').classes('text-xs text-blue-200')

                    def reset_regular_inputs(last_price: Optional[float] = None) -> None:
                        if last_price is not None:
                            market_price_state['last_price'] = _safe_float(last_price, 0.0)
                        apply_symbol_defaults(market_price_state['last_price'])
                        regular_stop_loss.value = 0.0
                        regular_stop_loss.update()
                        regular_target.value = 0.0
                        regular_target.update()
                        trailing_stop_loss.value = 0.0
                        trailing_stop_loss.update()
                        trailing_target.value = 0.0
                        trailing_target.update()
                        trail_start_target_percent.value = 5.0
                        trail_start_target_percent.update()
                        trailing_stop_loss_percent.value = 2.0
                        trailing_stop_loss_percent.update()
                        risk_management_type.value = 'regular'
                        risk_management_type.update()

                    # Market price update functions
                    def reset_gtt_defaults(last_price: Optional[float] = None) -> None:
                        if last_price is not None:
                            market_price_state['last_price'] = _safe_float(last_price, 0.0)
                        apply_symbol_defaults(market_price_state['last_price'])

                    async def update_market_price(symbol, reset_defaults: bool = False):
                        market_price_state['last_price'] = 0.0
                        instrument_token = instruments.get(symbol)
                        if instrument_token:
                            market_data = await fetch_api(f"/quotes/{broker}?instruments={instrument_token}")
                            if market_data:
                                quote = market_data[0]
                                last_price = _safe_float(quote.get('last_price', 0))
                                market_price_state['last_price'] = last_price
                                
                                # Get additional price data
                                avg_price = _safe_float(quote.get('average_price', 0)) or last_price
                                net_change = _safe_float(quote.get('net_change', 0))
                                pct_change = _safe_float(quote.get('pct_change', 0))
                                
                                # Update price labels
                                market_price_label.text = f"₹{last_price:,.2f}"
                                avg_price_label.text = f"₹{avg_price:,.2f}"
                                
                                # Update change labels with color coding (green if positive, red if negative)
                                if net_change >= 0:
                                    net_change_label.text = f"+₹{net_change:,.2f}"
                                    net_change_label.classes('text-sm font-semibold text-green-400')
                                else:
                                    net_change_label.text = f"₹{net_change:,.2f}"
                                    net_change_label.classes('text-sm font-semibold text-red-400')
                                
                                if pct_change >= 0:
                                    pct_change_label.text = f"+{pct_change:.2f}%"
                                    pct_change_label.classes('text-sm font-semibold text-green-400')
                                else:
                                    pct_change_label.text = f"{pct_change:.2f}%"
                                    pct_change_label.classes('text-sm font-semibold text-red-400')
                                
                                # Update OHLC data
                                open_price = _safe_float(quote.get('ohlc', {}).get('open', 0))
                                high_price = _safe_float(quote.get('ohlc', {}).get('high', 0))
                                low_price = _safe_float(quote.get('ohlc', {}).get('low', 0))
                                close_price = _safe_float(quote.get('ohlc', {}).get('close', 0))
                                volume = _safe_int(quote.get('volume', 0))
                                
                                ohlc_open_label.text = f"Open: ₹{open_price:,.2f}"
                                ohlc_high_label.text = f"High: ₹{high_price:,.2f}"
                                ohlc_low_label.text = f"Low: ₹{low_price:,.2f}"
                                ohlc_close_label.text = f"Close: ₹{close_price:,.2f}"
                                ohlc_volume_label.text = f"Volume: {volume:,}"
                                
                                # Update Market Depth data
                                depth_data = quote.get("depth", {})
                                buy_orders = depth_data.get("buy", []) if depth_data else []
                                sell_orders = depth_data.get("sell", []) if depth_data else []
                                
                                # Clear previous depth data
                                depth_buy_container.clear()
                                depth_sell_container.clear()
                                
                                # Populate buy orders
                                with depth_buy_container:
                                    if buy_orders:
                                        for order in buy_orders[:5]:  # Show top 5
                                            with ui.row().classes("justify-between w-full gap-2"):
                                                ui.label(f"₹{order.get('price', 0):,.2f}").classes("text-green-400 font-mono text-xs")
                                                ui.label(f"{order.get('quantity', 0):,}").classes("text-gray-300 text-xs")
                                    else:
                                        ui.label("No orders").classes("text-gray-500 text-xs italic text-center")
                                
                                # Populate sell orders
                                with depth_sell_container:
                                    if sell_orders:
                                        for order in sell_orders[:5]:  # Show top 5
                                            with ui.row().classes("justify-between w-full gap-2"):
                                                ui.label(f"₹{order.get('price', 0):,.2f}").classes("text-red-400 font-mono text-xs")
                                                ui.label(f"{order.get('quantity', 0):,}").classes("text-gray-300 text-xs")
                                    else:
                                        ui.label("No orders").classes("text-gray-500 text-xs italic text-center")
                            else:
                                market_price_label.text = "--"
                                avg_price_label.text = "--"
                                net_change_label.text = "--"
                                pct_change_label.text = "--"
                                ohlc_open_label.text = "Open: --"
                                ohlc_high_label.text = "High: --"
                                ohlc_low_label.text = "Low: --"
                                ohlc_close_label.text = "Close: --"
                                ohlc_volume_label.text = "Volume: --"
                        else:
                            market_price_label.text = "--"
                            avg_price_label.text = "--"
                            net_change_label.text = "--"
                            pct_change_label.text = "--"
                            ohlc_open_label.text = "Open: --"
                            ohlc_high_label.text = "High: --"
                            ohlc_low_label.text = "Low: --"
                            ohlc_close_label.text = "Close: --"
                            ohlc_volume_label.text = "Volume: --"
                        market_price_label.update()
                        if reset_defaults:
                            reset_gtt_defaults(market_price_state['last_price'])
                            reset_regular_inputs(market_price_state['last_price'])

                    # Toggle between risk management types
                    def toggle_risk_management():
                        if risk_management_type.value == 'regular':
                            regular_risk_section.visible = True
                            trailing_risk_section.visible = False
                        else:
                            regular_risk_section.visible = False
                            trailing_risk_section.visible = True

                    risk_management_type.on_value_change(toggle_risk_management)

                    # NOTE: Initial fetch moved after Market Data panel creation
                    symbol_select.on_value_change(lambda e: asyncio.create_task(update_market_price(e.value, reset_defaults=True)))

                    def update_price_fields():
                        price_field.visible = order_type.value in ['LIMIT', 'SL']
                        trigger_price_field.visible = order_type.value in ['SL', 'SL-M']

                    order_type.on_value_change(update_price_fields)
                    update_price_fields()

                    def regular_calc_context() -> Dict[str, Any]:
                        stop_loss_val = 0.0
                        target_val = 0.0

                        if risk_management_type.value == 'regular':
                            stop_loss_val = _safe_float(regular_stop_loss.value)
                            target_val = _safe_float(regular_target.value)

                        return {
                            'transaction_type': transaction_type.value,
                            'entry_price': market_price_state['last_price'],
                            'stop_loss': stop_loss_val,
                            'target_price': target_val,
                            'stop_mode': position_calc_defaults.get('stop_mode', 'Absolute'),
                            'target_mode': position_calc_defaults.get('target_mode', 'Absolute'),
                            'stop_input': stop_loss_val or 0.0,
                            'target_input': target_val or 0.0,
                            'capital': position_calc_defaults.get('capital'),
                            'risk_percent': position_calc_defaults.get('risk_percent'),
                            'risk_amount': position_calc_defaults.get('risk_amount'),
                            'brokerage': position_calc_defaults.get('brokerage'),
                            'charges_pct': position_calc_defaults.get('charges_pct'),
                        }

                    def regular_default_calc() -> Dict[str, Any]:
                        template = user_storage.get('_position_calc_template', template_defaults) or {}
                        capital_val = template.get('capital', position_calc_defaults.get('capital', 0.0))
                        risk_percent_val = template.get('risk_percent', position_calc_defaults.get('risk_percent', 0.0))
                        risk_amount_val = capital_val * (risk_percent_val / 100.0) if capital_val and risk_percent_val else 0.0
                        return {
                            'transaction_type': transaction_type.value,
                            'entry_price': market_price_state['last_price'],
                            'stop_loss': 0.0,
                            'target_price': 0.0,
                            'stop_mode': 'Absolute',
                            'target_mode': 'Absolute',
                            'stop_input': 0.0,
                            'target_input': 0.0,
                            'capital': capital_val,
                            'risk_percent': risk_percent_val,
                            'risk_amount': risk_amount_val,
                            'brokerage': template.get('brokerage', position_calc_defaults.get('brokerage', 0.0)),
                            'charges_pct': template.get('charges_pct', position_calc_defaults.get('charges_pct', 0.0)),
                        }

                    async def fetch_equity_funds() -> Dict[str, Any]:
                        try:
                            response = await fetch_api(f"/funds/{broker}")
                            return response or {}
                        except Exception as exc:
                            logger.warning(f"Unable to fetch funds for calculator: {exc}")
                            return {}

                    def get_basket_totals() -> Dict[str, float]:
                        totals_fn = basket_controller.get('totals')
                        if callable(totals_fn):
                            try:
                                return totals_fn() or {}
                            except Exception as exc:
                                logger.warning(f"Unable to compute basket totals: {exc}")
                        return {'required': 0.0, 'risk': 0.0, 'reward': 0.0}

                    def apply_regular_calc(result: Dict[str, Any]) -> None:
                        quantity_val = result.get('quantity')
                        if quantity_val:
                            quantity.value = max(_safe_int(quantity_val), 1)
                            quantity.update()

                        if result.get('stop_loss') is not None:
                            risk_management_type.value = 'regular'
                            risk_management_type.update()
                            regular_stop_loss.value = result['stop_loss']
                            regular_stop_loss.update()

                        if result.get('target') is not None:
                            risk_management_type.value = 'regular'
                            risk_management_type.update()
                            regular_target.value = result['target']
                            regular_target.update()

                        entry_price_val = result.get('entry_price')
                        if entry_price_val and order_type.value in ['LIMIT', 'SL']:
                            price_field.value = entry_price_val
                            price_field.update()

                        summary = result.get('summary', {})
                        risk_text = summary.get('risk_amount')
                        invest_text = summary.get('total_investment')
                        if quantity_val:
                            message_parts = [f"Qty {quantity.value}"]
                            if risk_text is not None:
                                message_parts.append(f"Risk {risk_text:.2f}")
                            if invest_text is not None:
                                message_parts.append(f"Invest {invest_text:.2f}")
                            ui.notify(' | '.join(message_parts), type='positive')

                    # Loading container
                    loading_container = ui.column().classes('w-full mt-4')
                    regular_confirm_dialog = ui.dialog()

                    async def show_regular_confirm(order_data: Dict[str, Any]) -> None:
                        regular_confirm_dialog.clear()
                        is_scheduled = 'schedule_datetime' in order_data
                        with regular_confirm_dialog, ui.card().classes('p-6 min-w-96'):
                            ui.label('Confirm Order Placement').classes('text-xl font-bold mb-4')

                            with ui.column().classes('gap-2 mb-4'):
                                ui.label(f"Symbol: {order_data['trading_symbol']}").classes('text-white')
                                ui.label(f"Type: {order_data['transaction_type']} {order_data['quantity']} shares").classes('text-white')
                                ui.label(f"Order Type: {order_data['order_type']}").classes('text-white')
                                ui.label(f"Order details - {order_data}").classes('text-white')
                                
                                if is_scheduled:
                                    schedule_dt = datetime.fromisoformat(order_data['schedule_datetime'])
                                    ui.label(f"📅 Scheduled for: {schedule_dt.strftime('%Y-%m-%d %H:%M')}").classes('text-purple-400 font-semibold')
                                
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

                            if broker == 'Upstox':
                                ui.label("Order Type: Regular Order").classes('text-white')

                            with ui.row().classes('gap-3'):
                                ui.button('Cancel', on_click=regular_confirm_dialog.close).classes('bg-gray-600 text-white px-4 py-2 rounded')

                                async def confirm_order():
                                    regular_confirm_dialog.close()
                                    with loading_container:
                                        loading_container.clear()
                                        with ui.row().classes("items-center gap-3"):
                                            ui.spinner(size="lg")
                                            ui.label("Scheduling order..." if is_scheduled else "Placing order...").classes("text-white")

                                        endpoint = "/scheduled-orders/" if is_scheduled else "/orders"
                                        response = await fetch_api(endpoint, method="POST", data=order_data)
                                        
                                        if is_scheduled:
                                            if response and response.get('scheduled_order_id'):
                                                ui.notify(f"Order scheduled successfully: {response['scheduled_order_id']}", type='positive')
                                            else:
                                                ui.notify("Failed to schedule order", type='negative')
                                        else:
                                            if response and response.get('order_id'):
                                                order_type_msg = "Trailing Stop Loss" if order_data.get('is_trailing_stop_loss') else "Regular"
                                                ui.notify(f"{order_type_msg} Order placed: {response['order_id']}", type='positive')
                                            else:
                                                ui.notify("Failed to place order", type='negative')
                                        loading_container.clear()

                                button_text = 'Schedule Order' if is_scheduled else 'Place Order'
                                ui.button(button_text, on_click=confirm_order).classes('bg-green-600 text-white px-4 py-2 rounded')

                        regular_confirm_dialog.open()

                    def build_regular_order_payload() -> tuple[Optional[Dict[str, Any]], Optional[str]]:
                        """
                        Build order payload with validation.
                        Returns: (order_data, error_message)
                        """
                        if not all(validation_state.values()):
                            return None, 'Please fix form errors'

                        if not symbol_select.value or symbol_select.value not in instruments:
                            return None, 'Please select a valid symbol'

                        if quantity.value <= 0:
                            return None, 'Quantity must be greater than 0'

                        if order_type.value in ['LIMIT', 'SL'] and price_field.value <= 0:
                            return None, 'Price must be greater than 0 for LIMIT and SL orders'

                        if order_type.value in ['SL', 'SL-M'] and trigger_price_field.value <= 0:
                            return None, 'Trigger price must be greater than 0 for SL and SL-M orders'

                        # Validate schedule fields if scheduling is enabled
                        if schedule_order_toggle.value:
                            if not schedule_date.value or not schedule_time.value:
                                return None, 'Please provide schedule date and time'
                            try:
                                schedule_datetime = datetime.combine(
                                    datetime.strptime(schedule_date.value, '%Y-%m-%d').date(),
                                    datetime.strptime(schedule_time.value, '%H:%M').time()
                                )
                                if schedule_datetime <= datetime.now():
                                    return None, 'Schedule time must be in the future'
                            except Exception as e:
                                return None, f'Invalid schedule time: {str(e)}'

                        # Validate trailing stop loss parameters
                        if risk_management_type.value == 'trailing':
                            if trail_start_target_percent.value <= 0:
                                return None, 'Trail start target percentage must be greater than 0'
                            if trailing_stop_loss_percent.value <= 0:
                                return None, 'Trailing stop loss percentage must be greater than 0'

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

                        # Add schedule datetime if scheduling is enabled
                        if schedule_order_toggle.value:
                            schedule_datetime = datetime.combine(
                                datetime.strptime(schedule_date.value, '%Y-%m-%d').date(),
                                datetime.strptime(schedule_time.value, '%H:%M').time()
                            )
                            order_data["schedule_datetime"] = schedule_datetime.isoformat()

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

                        estimated_price = _safe_float(price_field.value) if order_type.value in ['LIMIT', 'SL'] and price_field.value else _safe_float(market_price_label.text)
                        order_data["estimated_price"] = estimated_price if estimated_price > 0 else 0.0

                        return order_data, None

                    async def place_regular_order():
                        order_data, error = build_regular_order_payload()
                        if error:
                            ui.notify(error, type='negative')
                            return
                        if not order_data:
                            return

                        await show_regular_confirm(order_data)

                    def add_regular_to_basket() -> None:
                        order_data, error = build_regular_order_payload()
                        if error:
                            ui.notify(error, type='negative')
                            return
                        if not order_data:
                            return
                        add_fn = basket_controller.get('add')
                        if callable(add_fn):
                            is_scheduled = 'schedule_datetime' in order_data
                            basket_entry = {
                                "type": "Scheduled" if is_scheduled else "Regular",
                                "endpoint": "/scheduled-orders/" if is_scheduled else "/orders",
                                "success_key": "scheduled_order_id" if is_scheduled else "order_id",
                                "payload": order_data,
                            }
                            add_fn(basket_entry)

                    basket_observer_register = basket_controller.get('register_observer')

                    with ui.row().classes('order-action-row w-full gap-3 mt-2'):
                        create_position_calculator_button(
                            label='Position Sizing',
                            defaults=position_calc_defaults,
                            get_context=regular_calc_context,
                            apply_callback=apply_regular_calc,
                            button_classes='flex-1 px-6 py-3 rounded-lg font-medium text-lg shadow',
                            button_style='background: linear-gradient(135deg, #06b6d4, #2563eb); color: #ffffff; border: none; box-shadow: 0 8px 16px rgba(37,99,235,0.45);',
                            get_default_context=regular_default_calc,
                            funds_fetcher=fetch_equity_funds,
                            basket_totals_provider=get_basket_totals,
                            register_basket_listener=basket_observer_register if callable(basket_observer_register) else None,
                            on_reset=lambda: reset_regular_inputs(market_price_state['last_price'])
                        )
                        regular_basket_btn = ui.button('Add to Basket', icon='add_shopping_cart', on_click=add_regular_to_basket)\
                            .classes('flex-1 px-6 py-3 rounded-lg font-medium text-lg shadow-inner')
                        apply_button_style(regular_basket_btn, 'background: linear-gradient(135deg, #334155, #1e293b); color: #f8fafc; border: 1px solid rgba(148,163,184,0.35);')
                        regular_place_btn = ui.button('Place Order', icon="send", on_click=place_regular_order)\
                            .classes('flex-1 px-6 py-3 rounded-lg font-semibold text-lg shadow')
                        apply_button_style(regular_place_btn, 'background: linear-gradient(135deg, #22d3ee, #38bdf8); color: #0f172a; border: none; box-shadow: 0 8px 16px rgba(56,189,248,0.45);')
                        
                        # Update button text and icon when schedule toggle changes
                        def update_place_button():
                            if schedule_order_toggle.value:
                                regular_place_btn.props('icon=schedule')
                                regular_place_btn.text = 'Schedule Order'
                                apply_button_style(regular_place_btn, 'background: linear-gradient(135deg, #a855f7, #ec4899); color:#fff; border:none; box-shadow:0 8px 16px rgba(236,72,153,0.45);')
                            else:
                                regular_place_btn.props('icon=send')
                                regular_place_btn.text = 'Place Order'
                                apply_button_style(regular_place_btn, 'background: linear-gradient(135deg, #22d3ee, #38bdf8); color: #0f172a; border: none; box-shadow: 0 8px 16px rgba(56,189,248,0.45);')
                        
                        schedule_order_toggle.on_value_change(update_place_button)
                
                # RIGHT COLUMN: Market Data Panel (20%)
                with ui.card().classes('flex-1 theme-surface-elevated p-3'):
                    # Market Data header with refresh icon
                    with ui.row().classes('w-full items-center justify-between mb-2'):
                        ui.label("Market Data").classes("text-sm font-semibold theme-text-accent")
                        ui.button(icon='refresh', on_click=lambda: asyncio.create_task(update_market_price(symbol_select.value, reset_defaults=False)))\
                            .props('flat dense round').classes('theme-icon-button').tooltip('Refresh Market Data')
                    
                    # Market Price (Enhanced with LTP, Avg Price, Change)
                    with ui.card().classes('w-full p-2 mb-2 theme-surface-card'):
                        # First row: LTP and Avg Price
                        with ui.row().classes('w-full items-center justify-between mb-1'):
                            with ui.column().classes('gap-0'):
                                ui.label("LTP").classes("text-xs theme-text-secondary")
                                market_price_label = ui.label("--").classes("text-base font-bold theme-text-primary")
                            with ui.column().classes('gap-0'):
                                ui.label("Avg Price").classes("text-xs theme-text-secondary")
                                avg_price_label = ui.label("--").classes("text-base font-bold theme-text-primary")
                        
                        # Second row: Net Change and % Change (color coded)
                        with ui.row().classes('w-full items-center justify-between'):
                            with ui.column().classes('gap-0'):
                                ui.label("Change").classes("text-xs theme-text-secondary")
                                net_change_label = ui.label("--").classes("text-sm font-semibold")
                            with ui.column().classes('gap-0'):
                                ui.label("Change %").classes("text-xs theme-text-secondary")
                                pct_change_label = ui.label("--").classes("text-sm font-semibold")
                    
                    # OHLC Data (horizontal layout, expanded by default)
                    with ui.expansion("OHLC", icon="candlestick_chart", value=True).classes("w-full text-xs theme-expansion mb-2").props("dense"):
                        with ui.row().classes("gap-3 p-2 items-center justify-between w-full"):
                            ohlc_open_label = ui.label("Open: --").classes("text-xs theme-text-primary")
                            ohlc_high_label = ui.label("High: --").classes("text-xs theme-text-success")
                            ohlc_low_label = ui.label("Low: --").classes("text-xs theme-text-error")
                            ohlc_close_label = ui.label("Close: --").classes("text-xs theme-text-primary")
                            ohlc_volume_label = ui.label("Volume: --").classes("text-xs theme-text-info")
                    
                    # Market Depth (expanded by default)
                    with ui.expansion("Market Depth", icon="layers", value=True).classes("w-full text-xs theme-expansion").props("dense"):
                        with ui.element('div').style("display: grid; grid-template-columns: 1fr auto 1fr; gap: 0.5rem; padding: 0.5rem; width: 100%;"):
                            # Buy orders column
                            with ui.element('div').style("min-width: 0;"):
                                ui.label("Buy Orders").classes("theme-text-success font-semibold text-xs mb-1 text-center")
                                depth_buy_container = ui.column().classes("gap-1")
                            
                            # Vertical separator
                            with ui.element('div').classes("theme-divider"):
                                pass
                            
                            # Sell orders column
                            with ui.element('div').style("min-width: 0;"):
                                ui.label("Sell Orders").classes("theme-text-error font-semibold text-xs mb-1 text-center")
                                depth_sell_container = ui.column().classes("gap-1")
            
            # Initial market data fetch (after UI elements are created)
            if symbol_select.value:
                await update_market_price(symbol_select.value, reset_defaults=True)

# render_scheduled_orders function removed - scheduled orders are now part of regular orders

async def render_gtt_orders(fetch_api, user_storage, instruments, broker, position_calc_defaults, basket_controller):
    """GTT orders form"""

    market_price_state = {'last_price': 0.0}
    template_defaults = user_storage.get('_position_calc_template', {})

    def apply_symbol_defaults(last_price: float) -> None:
        template = user_storage.get('_position_calc_template', template_defaults) or {}
        position_calc_defaults['capital'] = template.get('capital', position_calc_defaults.get('capital', 0.0))
        position_calc_defaults['risk_percent'] = template.get('risk_percent', position_calc_defaults.get('risk_percent', 0.0))
        position_calc_defaults['brokerage'] = template.get('brokerage', position_calc_defaults.get('brokerage', 0.0))
        position_calc_defaults['charges_pct'] = template.get('charges_pct', position_calc_defaults.get('charges_pct', 0.0))
        position_calc_defaults['entry_price'] = _safe_float(last_price, 0.0)
        position_calc_defaults['stop_input'] = 0.0
        position_calc_defaults['target_input'] = 0.0
        position_calc_defaults['stop_mode'] = 'Absolute'
        position_calc_defaults['target_mode'] = 'Absolute'
        capital_val = position_calc_defaults.get('capital', 0.0)
        risk_percent_val = position_calc_defaults.get('risk_percent', 0.0)
        position_calc_defaults['risk_amount'] = capital_val * (risk_percent_val / 100.0) if capital_val and risk_percent_val else 0.0

    trigger_price = None
    limit_price = None
    second_trigger_price = None
    second_limit_price = None
    up_entry_enabled = None
    up_entry_trigger_price = None
    up_target_enabled = None
    up_target_trigger_price = None
    up_sl_enabled = None
    up_sl_trigger_price = None
    upstox_visibility_callback: Optional[Callable[[], None]] = None
    oco_visibility_callback: Optional[Callable[[], None]] = None
    market_price_label = None
    gtt_avg_price_label = None
    gtt_net_change_label = None
    gtt_pct_change_label = None
    gtt_ohlc_open_label = None
    gtt_ohlc_high_label = None
    gtt_ohlc_low_label = None
    gtt_ohlc_close_label = None
    gtt_ohlc_volume_label = None
    gtt_depth_buy_container = None
    gtt_depth_sell_container = None

    with ui.card().classes('w-full enhanced-card'):
        with ui.row().classes("w-full items-center justify-between p-2 border-b order-form-header theme-surface-elevated"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("compare_arrows", size="1.2rem").classes("theme-text-accent")
                ui.label("Place GTT Order").classes("text-lg font-semibold theme-text-primary")
            ui.chip("TRIGGERED", color="green").classes("text-xs")

        with ui.column().classes("p-2 gap-3 w-full gtt-card-body").style(
            "display: flex; flex-direction: column !important; gap: 0.75rem; width: 100%;"
        ):
            validation_state = {'symbol': True, 'quantity': True, 'trigger_price': True, 'limit_price': True}

            # Main Layout: Order Form (80%) + Market Data (20%)
            with ui.element('div').classes('gtt-layout-grid').style(
                'display: grid; width: 100%; gap: 1rem; grid-template-columns: minmax(0, 4fr) minmax(260px, 360px); align-items: start;'
            ):
                # LEFT COLUMN: GTT Order Form
                gtt_form_column = ui.column().classes('gap-3').style('min-width: 0; display: flex; flex-direction: column; gap: 0.75rem;')
                with gtt_form_column:
                    # Row 1 - Core Inputs
                    with ui.row().classes('w-full gap-3'):
                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Index Filter").classes("text-sm font-medium theme-text-primary")
                            index_select = ui.select(
                                options=['NIFTY_50', 'NIFTY_NEXT_50', 'All Instruments'],
                                value='NIFTY_50'
                            ).classes('w-full')

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Trading Symbol").classes("text-sm font-medium theme-text-primary")

                            symbol_options = await get_symbol_options(index_select.value)
                            initial_symbol = symbol_options[0] if symbol_options else None

                            symbol_select = ui.select(
                                options=symbol_options,
                                with_input=True,
                                value=initial_symbol
                            ).classes('w-full')
                            symbol_select.on_value_change(lambda e: validation_state.update({'symbol': bool(e.value)}))

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("GTT Type").classes("text-sm font-medium theme-text-primary")
                            trigger_type = ui.select(
                                options=['single', 'OCO'] if broker == 'Zerodha' else ['SINGLE', 'OCO'],
                                value='single' if broker == 'Zerodha' else 'SINGLE'
                            ).classes('w-full')

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Transaction Type").classes("text-sm font-medium theme-text-primary")
                            transaction_type = ui.select(
                                options=['BUY', 'SELL'],
                                value='BUY'
                            ).classes('w-full')

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Quantity").classes("text-sm font-medium theme-text-primary")
                            quantity = ui.number(
                                value=1,
                                min=1,
                                format='%d'
                            ).classes('w-full')
                            quantity.on_value_change(lambda e: validation_state.update({'quantity': _safe_int(e.value) > 0}))

                        with ui.column().classes("flex-1 gap-2"):
                            ui.label("Product Type").classes("text-sm font-medium theme-text-primary")
                            product_type = ui.select(
                                options=['MIS', 'CNC'] if broker == 'Zerodha' else ['I', 'D'],
                                value='CNC' if broker == 'Zerodha' else 'D'
                            ).classes('w-full')

                    async def update_symbol_options():
                        if index_select.value == 'All Instruments':
                            symbol_select.options = sorted(list(instruments.keys()))
                        else:
                            symbol_select.options = await get_symbol_options(index_select.value)
                        symbol_select.update()
                        if symbol_select.value:
                            await update_market_price(symbol_select.value, reset_defaults=True)

                    index_select.on_value_change(update_symbol_options)

                    # Row 2 - Broker-specific fields
                    if broker == 'Zerodha':
                        with ui.row().classes('w-full gap-3'):
                            with ui.column().classes("flex-1 gap-2"):
                                ui.label("Trigger Price").classes("text-sm font-medium theme-text-primary")
                                trigger_price = ui.number(
                                    value=0,
                                    min=0,
                                    step=0.05,
                                    format='%.2f'
                                ).classes('w-full')
                                trigger_price.on_value_change(lambda e: validation_state.update({'trigger_price': _safe_float(e.value) > 0}))

                            with ui.column().classes("flex-1 gap-2"):
                                ui.label("Limit Price").classes("text-sm font-medium theme-text-primary")
                                limit_price = ui.number(
                                    value=0,
                                    min=0,
                                    step=0.05,
                                    format='%.2f'
                                ).classes('w-full')
                                limit_price.on_value_change(lambda e: validation_state.update({'limit_price': _safe_float(e.value) > 0}))

                        ui.label("Trigger Price should be 0.25% above or below market price").classes("text-xs theme-text-info")

                        with ui.row().classes('w-full gap-3') as oco_price_row:
                            with ui.column().classes("flex-1 gap-2"):
                                ui.label("Second Trigger Price").classes("text-sm font-medium theme-text-primary")
                                second_trigger_price = ui.number(
                                    value=0,
                                    min=0,
                                    step=0.05,
                                    format='%.2f'
                                ).classes('w-full')

                            with ui.column().classes("flex-1 gap-2"):
                                ui.label("Second Limit Price").classes("text-sm font-medium theme-text-primary")
                                second_limit_price = ui.number(
                                    value=0,
                                    min=0,
                                    step=0.05,
                                    format='%.2f'
                                ).classes('w-full')

                        def update_oco_fields():
                            is_oco = str(trigger_type.value).upper() == 'OCO'
                            oco_price_row.visible = is_oco
                            oco_price_row.update()

                        oco_visibility_callback = update_oco_fields
                        trigger_type.on_value_change(lambda e: update_oco_fields())
                        update_oco_fields()

                    if broker == 'Upstox':
                        with ui.column().classes('w-full gap-3 theme-surface-elevated p-3 rounded-lg'):
                            ui.label('Upstox GTT Rules').classes('text-sm font-semibold theme-text-accent')
                            with ui.row().classes('w-full gap-3'):
                                with ui.column().classes('flex-1 gap-2') as up_entry_container:
                                    ui.label('ENTRY').classes('text-xs theme-text-secondary')
                                    up_entry_enabled = ui.checkbox('Include ENTRY', value=True).classes('theme-text-primary')
                                    up_entry_trigger_type = ui.select(options=['BELOW', 'ABOVE', 'IMMEDIATE'], value='ABOVE').classes('w-full')
                                    up_entry_trigger_price = ui.number(value=0, min=0, step=0.05, format='%.2f').classes('w-full')

                                with ui.column().classes('flex-1 gap-2') as up_target_container:
                                    ui.label('TARGET').classes('text-xs theme-text-secondary')
                                    up_target_enabled = ui.checkbox('Include TARGET', value=True).classes('theme-text-primary')
                                    up_target_trigger_type = ui.select(options=['IMMEDIATE'], value='IMMEDIATE').classes('w-full')
                                    up_target_trigger_price = ui.number(value=0, min=0, step=0.05, format='%.2f').classes('w-full')

                                with ui.column().classes('flex-1 gap-2') as up_sl_container:
                                    ui.label('STOP_LOSS').classes('text-xs theme-text-secondary')
                                    up_sl_enabled = ui.checkbox('Include STOP_LOSS', value=True).classes('theme-text-primary')
                                    up_sl_trigger_type = ui.select(options=['IMMEDIATE'], value='IMMEDIATE').classes('w-full')
                                    up_sl_trigger_price = ui.number(value=0, min=0, step=0.05, format='%.2f').classes('w-full')

                            def update_upstox_rule_visibility():
                                is_single = str(trigger_type.value).upper() == 'SINGLE'
                                up_entry_container.visible = True
                                up_target_container.visible = not is_single
                                up_sl_container.visible = not is_single
                                up_target_enabled.value = not is_single
                                up_sl_enabled.value = not is_single
                                up_entry_container.update(); up_target_container.update(); up_sl_container.update()

                            upstox_visibility_callback = update_upstox_rule_visibility
                            trigger_type.on_value_change(lambda e: update_upstox_rule_visibility())
                            update_upstox_rule_visibility()

                # RIGHT COLUMN: Market Data Panel
                with ui.card().classes('theme-surface-elevated p-3').style('min-width: 260px; max-width: 360px;'):
                    with ui.row().classes('w-full items-center justify-between mb-2'):
                        ui.label("Market Data").classes("text-sm font-semibold theme-text-accent")
                        ui.button(
                            icon='refresh',
                            on_click=lambda: asyncio.create_task(update_market_price(symbol_select.value, reset_defaults=False))
                        ).props('flat dense round').classes('theme-icon-button').tooltip('Refresh Market Data')

                    with ui.card().classes('w-full p-2 mb-2 theme-surface-card'):
                        with ui.row().classes('w-full items-center justify-between mb-1'):
                            with ui.column().classes('gap-0'):
                                ui.label("LTP").classes("text-xs theme-text-secondary")
                                market_price_label = ui.label("--").classes("text-base font-bold theme-text-primary")
                            with ui.column().classes('gap-0'):
                                ui.label("Avg Price").classes("text-xs theme-text-secondary")
                                gtt_avg_price_label = ui.label("--").classes("text-base font-bold theme-text-primary")

                        with ui.row().classes('w-full items-center justify-between'):
                            with ui.column().classes('gap-0'):
                                ui.label("Change").classes("text-xs theme-text-secondary")
                                gtt_net_change_label = ui.label("--").classes("text-sm font-semibold theme-text-primary")
                            with ui.column().classes('gap-0'):
                                ui.label("Change %").classes("text-xs theme-text-secondary")
                                gtt_pct_change_label = ui.label("--").classes("text-sm font-semibold theme-text-primary")

                    with ui.expansion("OHLC", icon="candlestick_chart", value=True).classes("w-full text-xs theme-expansion mb-2").props("dense"):
                        with ui.row().classes("gap-3 p-2 items-center justify-between w-full"):
                            gtt_ohlc_open_label = ui.label("Open: --").classes("text-xs theme-text-primary")
                            gtt_ohlc_high_label = ui.label("High: --").classes("text-xs theme-text-success")
                            gtt_ohlc_low_label = ui.label("Low: --").classes("text-xs theme-text-error")
                            gtt_ohlc_close_label = ui.label("Close: --").classes("text-xs theme-text-primary")
                            gtt_ohlc_volume_label = ui.label("Volume: --").classes("text-xs theme-text-info")

                    with ui.expansion("Market Depth", icon="layers", value=True).classes("w-full text-xs theme-expansion").props("dense"):
                        with ui.element('div').style("display: grid; grid-template-columns: 1fr auto 1fr; gap: 0.5rem; padding: 0.5rem; width: 100%;"):
                            with ui.element('div').style("min-width: 0;"):
                                ui.label("Buy Orders").classes("theme-text-success font-semibold text-xs mb-1 text-center")
                                gtt_depth_buy_container = ui.column().classes("gap-1")

                            with ui.element('div').classes("theme-divider"):
                                pass

                            with ui.element('div').style("min-width: 0;"):
                                ui.label("Sell Orders").classes("theme-text-error font-semibold text-xs mb-1 text-center")
                                gtt_depth_sell_container = ui.column().classes("gap-1")

            async def update_market_price(symbol, reset_defaults: bool = False):
                market_price_state['last_price'] = 0.0

                def set_default_panels() -> None:
                    if market_price_label:
                        market_price_label.text = "--"
                    if gtt_avg_price_label:
                        gtt_avg_price_label.text = "--"
                    if gtt_net_change_label:
                        gtt_net_change_label.text = "--"
                        gtt_net_change_label.classes("text-sm font-semibold theme-text-primary")
                    if gtt_pct_change_label:
                        gtt_pct_change_label.text = "--"
                        gtt_pct_change_label.classes("text-sm font-semibold theme-text-primary")
                    if gtt_ohlc_open_label:
                        gtt_ohlc_open_label.text = "Open: --"
                    if gtt_ohlc_high_label:
                        gtt_ohlc_high_label.text = "High: --"
                    if gtt_ohlc_low_label:
                        gtt_ohlc_low_label.text = "Low: --"
                    if gtt_ohlc_close_label:
                        gtt_ohlc_close_label.text = "Close: --"
                    if gtt_ohlc_volume_label:
                        gtt_ohlc_volume_label.text = "Volume: --"
                    if gtt_depth_buy_container:
                        gtt_depth_buy_container.clear()
                        with gtt_depth_buy_container:
                            ui.label("No orders").classes("theme-text-secondary text-xs italic text-center")
                    if gtt_depth_sell_container:
                        gtt_depth_sell_container.clear()
                        with gtt_depth_sell_container:
                            ui.label("No orders").classes("theme-text-secondary text-xs italic text-center")

                instrument_token = instruments.get(symbol)
                if instrument_token:
                    market_data = await fetch_api(f"/quotes/{broker}?instruments={instrument_token}")
                    if market_data:
                        quote = market_data[0]
                        last_price = _safe_float(quote.get('last_price', 0))
                        avg_price = _safe_float(quote.get('average_price', 0)) or last_price
                        net_change = _safe_float(quote.get('net_change', 0))
                        pct_change = _safe_float(quote.get('pct_change', 0))
                        market_price_state['last_price'] = last_price

                        market_price_label.text = f"₹{last_price:,.2f}"
                        gtt_avg_price_label.text = f"₹{avg_price:,.2f}"

                        if net_change >= 0:
                            gtt_net_change_label.text = f"+₹{net_change:,.2f}"
                            gtt_net_change_label.classes('text-sm font-semibold text-green-400')
                        else:
                            gtt_net_change_label.text = f"₹{net_change:,.2f}"
                            gtt_net_change_label.classes('text-sm font-semibold text-red-400')

                        if pct_change >= 0:
                            gtt_pct_change_label.text = f"+{pct_change:.2f}%"
                            gtt_pct_change_label.classes('text-sm font-semibold text-green-400')
                        else:
                            gtt_pct_change_label.text = f"{pct_change:.2f}%"
                            gtt_pct_change_label.classes('text-sm font-semibold text-red-400')

                        ohlc_data = quote.get('ohlc', {}) or {}
                        open_price = _safe_float(ohlc_data.get('open', 0))
                        high_price = _safe_float(ohlc_data.get('high', 0))
                        low_price = _safe_float(ohlc_data.get('low', 0))
                        close_price = _safe_float(ohlc_data.get('close', 0))
                        volume = _safe_int(quote.get('volume', 0))

                        gtt_ohlc_open_label.text = f"Open: ₹{open_price:,.2f}"
                        gtt_ohlc_high_label.text = f"High: ₹{high_price:,.2f}"
                        gtt_ohlc_low_label.text = f"Low: ₹{low_price:,.2f}"
                        gtt_ohlc_close_label.text = f"Close: ₹{close_price:,.2f}"
                        gtt_ohlc_volume_label.text = f"Volume: {volume:,}"

                        depth_data = quote.get("depth", {}) or {}
                        buy_orders = depth_data.get("buy", [])
                        sell_orders = depth_data.get("sell", [])

                        if gtt_depth_buy_container:
                            gtt_depth_buy_container.clear()
                            with gtt_depth_buy_container:
                                if buy_orders:
                                    for order in buy_orders[:5]:
                                        with ui.row().classes("justify-between w-full gap-2"):
                                            ui.label(f"₹{order.get('price', 0):,.2f}").classes("theme-text-success font-mono text-xs")
                                            ui.label(f"{order.get('quantity', 0):,}").classes("theme-text-secondary text-xs")
                                else:
                                    ui.label("No orders").classes("theme-text-secondary text-xs italic text-center")

                        if gtt_depth_sell_container:
                            gtt_depth_sell_container.clear()
                            with gtt_depth_sell_container:
                                if sell_orders:
                                    for order in sell_orders[:5]:
                                        with ui.row().classes("justify-between w-full gap-2"):
                                            ui.label(f"₹{order.get('price', 0):,.2f}").classes("theme-text-error font-mono text-xs")
                                            ui.label(f"{order.get('quantity', 0):,}").classes("theme-text-secondary text-xs")
                                else:
                                    ui.label("No orders").classes("theme-text-secondary text-xs italic text-center")
                    else:
                        set_default_panels()
                else:
                    set_default_panels()

                if market_price_label:
                    market_price_label.update()
                if gtt_avg_price_label:
                    gtt_avg_price_label.update()
                if gtt_net_change_label:
                    gtt_net_change_label.update()
                if gtt_pct_change_label:
                    gtt_pct_change_label.update()

                if reset_defaults:
                    apply_symbol_defaults(market_price_state['last_price'])

            if symbol_select.value:
                await update_market_price(symbol_select.value, reset_defaults=True)
            symbol_select.on_value_change(lambda e: asyncio.create_task(update_market_price(e.value, reset_defaults=True)))
            def reset_gtt_defaults(last_price: Optional[float] = None) -> None:
                if last_price is not None:
                    market_price_state['last_price'] = _safe_float(last_price, 0.0)
                apply_symbol_defaults(market_price_state['last_price'])
                if trigger_price:
                    trigger_price.value = 0.0
                    trigger_price.update()
                if limit_price:
                    limit_price.value = 0.0
                    limit_price.update()
                if second_trigger_price:
                    second_trigger_price.value = 0.0
                    second_trigger_price.update()
                if second_limit_price:
                    second_limit_price.value = 0.0
                    second_limit_price.update()

            # Loading container
            loading_container = ui.column().classes('w-full')

            def build_gtt_order_payload() -> tuple[Optional[Dict[str, Any]], Optional[str]]:
                """
                Build GTT order payload with validation.
                Returns: (order_data, error_message)
                """
                if not all(validation_state.values()):
                    return None, 'Please fix form errors'

                if not symbol_select.value or symbol_select.value not in instruments:
                    return None, 'Please select a valid symbol'

                if quantity.value <= 0:
                    return None, 'Quantity must be greater than 0'

                # Get current market price for the symbol
                last_price = market_price_state.get('last_price', 0.0)
                if last_price <= 0:
                    # Try to extract price from market price label if available
                    if market_price_label and market_price_label.text != "--":
                        try:
                            price_text = market_price_label.text.split()[0]
                            clean_price = price_text.replace('₹', '').replace(',', '')
                            last_price = _safe_float(clean_price, 0.0)
                        except Exception:
                            last_price = 0.0

                order_data = {
                    "broker": broker,
                    "trading_symbol": symbol_select.value,
                    "instrument_token": instruments[symbol_select.value],
                    "transaction_type": transaction_type.value,
                    "quantity": int(quantity.value),
                    "product_type": product_type.value,
                    "last_price": last_price
                }
                estimated_price = order_data.get('last_price', 0.0)

                if broker == 'Zerodha':
                    if trigger_price is None or limit_price is None:
                        return None, 'Trigger/limit price fields are required'

                    trigger_val = _safe_float(trigger_price.value)
                    limit_val = _safe_float(limit_price.value)

                    if trigger_val <= 0:
                        return None, 'Trigger price must be greater than 0'
                    if limit_val <= 0:
                        return None, 'Limit price must be greater than 0'

                    order_data.update({
                        "trigger_type": trigger_type.value,
                        "trigger_price": trigger_val,
                        "limit_price": limit_val
                    })

                    if trigger_type.value == 'OCO':
                        second_trigger_val = _safe_float(second_trigger_price.value) if second_trigger_price else 0.0
                        second_limit_val = _safe_float(second_limit_price.value) if second_limit_price else 0.0
                        if second_trigger_val <= 0 or second_limit_val <= 0:
                            return None, 'OCO trigger/limit must be greater than 0'
                        order_data.update({
                            "second_trigger_price": second_trigger_val,
                            "second_limit_price": second_limit_val
                        })
                    estimated_price = limit_val
                else:
                    up_rules = []
                    try:
                        if up_entry_enabled and up_entry_enabled.value:
                            entry_price_val = _safe_float(up_entry_trigger_price.value)
                            if entry_price_val <= 0:
                                return None, 'Entry trigger price must be greater than 0'
                            up_rules.append({
                                "strategy": "ENTRY",
                                "trigger_type": up_entry_trigger_type.value,
                                "trigger_price": entry_price_val
                            })
                        if up_target_enabled and up_target_enabled.value:
                            target_price_val = _safe_float(up_target_trigger_price.value)
                            if target_price_val <= 0:
                                return None, 'Target trigger price must be greater than 0'
                            up_rules.append({
                                "strategy": "TARGET",
                                "trigger_type": up_target_trigger_type.value,
                                "trigger_price": target_price_val
                            })
                        if up_sl_enabled and up_sl_enabled.value:
                            sl_price_val = _safe_float(up_sl_trigger_price.value)
                            if sl_price_val <= 0:
                                return None, 'Stop loss trigger price must be greater than 0'
                            up_rules.append({
                                "strategy": "STOPLOSS",
                                "trigger_type": up_sl_trigger_type.value,
                                "trigger_price": sl_price_val
                            })
                    except Exception:
                        up_rules = []

                    if not up_rules:
                        return None, 'Select at least one rule for Upstox GTT'

                    order_data.update({
                        "trigger_type": str(trigger_type.value).upper(),
                        "rules": up_rules
                    })
                    estimated_price = order_data.get('last_price', 0.0)
                    if (not estimated_price or estimated_price <= 0) and up_rules:
                        entry_rule = next((r for r in up_rules if r.get('strategy') == 'ENTRY'), None)
                        if entry_rule:
                            estimated_price = _safe_float(entry_rule.get('trigger_price'))

                order_data["estimated_price"] = estimated_price if estimated_price and estimated_price > 0 else 0.0

                return order_data, None

            # GTT Confirmation Dialog (created outside async function)
            gtt_confirm_dialog = ui.dialog()
            
            # Place GTT Order Action
            async def place_gtt_order():
                order_data, error = build_gtt_order_payload()
                if error:
                    ui.notify(error, type='negative')
                    return
                if not order_data:
                    return

                # Fetch current market price for reference
                try:
                    instrument_token = instruments[symbol_select.value]
                    market_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
                    if market_data:
                        order_data['last_price'] = float(market_data[0].get('last_price', 0))
                except Exception:
                    pass

                # Clear and populate confirmation dialog
                gtt_confirm_dialog.clear()
                with gtt_confirm_dialog, ui.card().classes('p-6 min-w-96'):
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
                        ui.button('Cancel', on_click=gtt_confirm_dialog.close).classes('bg-gray-600 text-white px-4 py-2 rounded')

                        async def confirm_gtt_order():
                            gtt_confirm_dialog.close()
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

                        ui.button('Place GTT Order', on_click=confirm_gtt_order).classes('bg-green-600 text-white px-4 py-2 rounded')

                gtt_confirm_dialog.open()

            async def fetch_current_market_price(symbol: str) -> float:
                """Fetch current market price for the given symbol"""
                try:
                    instrument_token = instruments.get(symbol)
                    if instrument_token:
                        market_data = await fetch_api(f"/ltp/{broker}?instruments={instrument_token}")
                        if market_data:
                            return _safe_float(market_data[0]['last_price'], 0.0)
                except Exception:
                    pass
                return 0.0

            async def add_gtt_to_basket() -> None:
                order_data, error = build_gtt_order_payload()
                if error:
                    ui.notify(error, type='negative')
                    return
                if not order_data:
                    return
                
                # Ensure we have a valid last_price for GTT orders
                if order_data.get('last_price', 0.0) <= 0:
                    current_price = await fetch_current_market_price(order_data['trading_symbol'])
                    if current_price > 0:
                        order_data['last_price'] = current_price
                        market_price_state['last_price'] = current_price
                    else:
                        ui.notify('Unable to fetch current market price for GTT order', type='warning')
                        return
                
                add_fn = basket_controller.get('add')
                if callable(add_fn):
                    basket_entry = {
                        "type": "GTT",
                        "endpoint": "/gtt-orders/",
                        "success_key": "gtt_id",
                        "payload": order_data,
                    }
                    add_fn(basket_entry)

            def gtt_calc_context() -> Dict[str, Any]:
                entry_price = market_price_state['last_price']
                stop_val = 0.0
                target_val = 0.0

                if broker == 'Zerodha':
                    if limit_price:
                        entry_val = _safe_float(limit_price.value)
                        entry_price = entry_val if entry_val > 0 else entry_price
                    if trigger_price:
                        stop_val = _safe_float(trigger_price.value)
                    if trigger_type.value == 'OCO' and second_limit_price:
                        target_val = _safe_float(second_limit_price.value)
                    elif limit_price:
                        target_val = _safe_float(limit_price.value)
                else:
                    if up_entry_trigger_price and up_entry_enabled and up_entry_enabled.value:
                        entry_price = _safe_float(up_entry_trigger_price.value)
                    if up_sl_trigger_price and up_sl_enabled and up_sl_enabled.value:
                        stop_val = _safe_float(up_sl_trigger_price.value)
                    if up_target_trigger_price and up_target_enabled and up_target_enabled.value:
                        target_val = _safe_float(up_target_trigger_price.value)

                return {
                    'transaction_type': transaction_type.value,
                    'entry_price': entry_price,
                    'stop_loss': stop_val,
                    'target_price': target_val,
                    'stop_mode': position_calc_defaults.get('stop_mode', 'Absolute'),
                    'target_mode': position_calc_defaults.get('target_mode', 'Absolute'),
                    'stop_input': stop_val or 0.0,
                    'target_input': target_val or 0.0,
                    'capital': position_calc_defaults.get('capital'),
                    'risk_percent': position_calc_defaults.get('risk_percent'),
                    'risk_amount': position_calc_defaults.get('risk_amount'),
                    'brokerage': position_calc_defaults.get('brokerage'),
                    'charges_pct': position_calc_defaults.get('charges_pct'),
                }

            def gtt_default_calc() -> Dict[str, Any]:
                template = user_storage.get('_position_calc_template', template_defaults) or {}
                capital_val = template.get('capital', position_calc_defaults.get('capital', 0.0))
                risk_percent_val = template.get('risk_percent', position_calc_defaults.get('risk_percent', 0.0))
                risk_amount_val = capital_val * (risk_percent_val / 100.0) if capital_val and risk_percent_val else 0.0
                return {
                    'transaction_type': transaction_type.value,
                    'entry_price': market_price_state['last_price'],
                    'stop_loss': 0.0,
                    'target_price': 0.0,
                    'stop_mode': 'Absolute',
                    'target_mode': 'Absolute',
                    'stop_input': 0.0,
                    'target_input': 0.0,
                    'capital': capital_val,
                    'risk_percent': risk_percent_val,
                    'risk_amount': risk_amount_val,
                    'brokerage': template.get('brokerage', position_calc_defaults.get('brokerage', 0.0)),
                    'charges_pct': template.get('charges_pct', position_calc_defaults.get('charges_pct', 0.0)),
                }

            async def fetch_equity_funds() -> Dict[str, Any]:
                try:
                    response = await fetch_api(f"/funds/{broker}")
                    return response or {}
                except Exception as exc:
                    logger.warning(f"Unable to fetch funds for calculator: {exc}")
                    return {}

            def get_basket_totals() -> Dict[str, float]:
                totals_fn = basket_controller.get('totals')
                if callable(totals_fn):
                    try:
                        return totals_fn() or {}
                    except Exception as exc:
                        logger.warning(f"Unable to compute basket totals: {exc}")
                return {'required': 0.0, 'risk': 0.0, 'reward': 0.0}

            def apply_gtt_calc(result: Dict[str, Any]) -> None:
                quantity_val = result.get('quantity')
                if quantity_val:
                    quantity.value = max(_safe_int(quantity_val), 1)
                    quantity.update()

                entry_val = result.get('entry_price')
                if entry_val:
                    if broker == 'Zerodha' and limit_price:
                        limit_price.value = entry_val
                        limit_price.update()
                    elif up_entry_trigger_price:
                        up_entry_enabled.value = True
                        up_entry_trigger_price.value = entry_val
                        up_entry_trigger_price.update()
                        if upstox_visibility_callback:
                            upstox_visibility_callback()

                stop_val = result.get('stop_loss')
                if stop_val is not None:
                    if broker == 'Zerodha' and trigger_price:
                        trigger_price.value = stop_val
                        trigger_price.update()
                    elif up_sl_trigger_price:
                        up_sl_enabled.value = True
                        up_sl_trigger_price.value = stop_val
                        up_sl_trigger_price.update()
                        if upstox_visibility_callback:
                            upstox_visibility_callback()

                target_val = result.get('target')
                if target_val is not None:
                    if broker == 'Zerodha':
                        if trigger_type.value == 'OCO' and second_limit_price:
                            second_limit_price.value = target_val
                            second_limit_price.update()
                            if oco_visibility_callback:
                                oco_visibility_callback()
                        elif limit_price:
                            limit_price.value = target_val
                            limit_price.update()
                    elif up_target_trigger_price:
                        up_target_enabled.value = True
                        up_target_trigger_price.value = target_val
                        up_target_trigger_price.update()
                        if upstox_visibility_callback:
                            upstox_visibility_callback()

                summary = result.get('summary', {})
                details = []
                if quantity_val:
                    details.append(f"Qty {quantity.value}")
                if summary.get('risk_amount') is not None:
                    details.append(f"Risk {summary['risk_amount']:.2f}")
                if summary.get('profit_amount') is not None:
                    details.append(f"P/L {summary['profit_amount']:.2f}")
                if details:
                    ui.notify('GTT sizing applied: ' + ' | '.join(details), type='positive')
                    
            basket_observer_register = basket_controller.get('register_observer')

            with gtt_form_column:
                with ui.row().classes('order-action-row w-full gap-3 mt-4'):
                    create_position_calculator_button(
                        label='Position Sizing',
                        defaults=position_calc_defaults,
                        get_context=gtt_calc_context,
                        apply_callback=apply_gtt_calc,
                        button_classes='flex-1 px-6 py-3 rounded-lg font-medium text-lg shadow',
                        button_style='background: linear-gradient(135deg, #34d399, #14b8a6); color:#052e16; border:none; box-shadow:0 8px 16px rgba(52,211,153,0.45);',
                        get_default_context=gtt_default_calc,
                        funds_fetcher=fetch_equity_funds,
                        basket_totals_provider=get_basket_totals,
                        register_basket_listener=basket_observer_register if callable(basket_observer_register) else None,
                        on_reset=lambda: reset_gtt_defaults(market_price_state['last_price'])
                    )
                    gtt_basket_btn = ui.button('Add to Basket', icon='add_shopping_cart', on_click=add_gtt_to_basket)\
                        .classes('flex-1 px-6 py-3 rounded-lg font-medium text-lg shadow-inner')
                    apply_button_style(gtt_basket_btn, 'background: linear-gradient(135deg, #1f2937, #0f172a); color:#e2e8f0; border:1px solid rgba(148,163,184,0.35);')
                    gtt_place_btn = ui.button('Place GTT Order', icon="compare_arrows", on_click=place_gtt_order)\
                        .classes('flex-1 px-6 py-3 rounded-lg font-semibold text-lg shadow')
                    apply_button_style(gtt_place_btn, 'background: linear-gradient(135deg, #10b981, #2dd4bf); color:#022c22; border:none; box-shadow:0 8px 16px rgba(16,185,129,0.45);')

async def render_auto_orders(fetch_api, user_storage, instruments, broker, position_calc_defaults, basket_controller):
    """Auto orders form"""

    market_price_state = {'last_price': 0.0}
    template_defaults = user_storage.get('_position_calc_template', {})

    def apply_symbol_defaults(last_price: float) -> None:
        template = user_storage.get('_position_calc_template', template_defaults) or {}
        position_calc_defaults['capital'] = template.get('capital', position_calc_defaults.get('capital', 0.0))
        position_calc_defaults['risk_percent'] = template.get('risk_percent', position_calc_defaults.get('risk_percent', 0.0))
        position_calc_defaults['brokerage'] = template.get('brokerage', position_calc_defaults.get('brokerage', 0.0))
        position_calc_defaults['charges_pct'] = template.get('charges_pct', position_calc_defaults.get('charges_pct', 0.0))
        position_calc_defaults['entry_price'] = _safe_float(last_price, 0.0)
        position_calc_defaults['stop_input'] = 0.0
        position_calc_defaults['target_input'] = 0.0
        position_calc_defaults['stop_mode'] = 'Absolute'
        position_calc_defaults['target_mode'] = 'Absolute'
        capital_val = position_calc_defaults.get('capital', 0.0)
        risk_percent_val = position_calc_defaults.get('risk_percent', 0.0)
        position_calc_defaults['risk_amount'] = capital_val * (risk_percent_val / 100.0) if capital_val and risk_percent_val else 0.0

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

            ui.label("Reference Market Price").classes("text-sm font-medium text-gray-300")
            with ui.row().classes('items-center gap-2 bg-emerald-500/15 border border-emerald-500/30 px-3 py-1 rounded-lg w-fit'):
                ui.icon('show_chart').classes('text-emerald-400 text-sm')
                market_price_label = ui.label("--").classes("text-sm font-semibold text-emerald-200")
            
            # OHLC Data
            with ui.expansion("OHLC", icon="candlestick_chart").classes("w-full text-xs bg-gray-700/30 mt-2").props("dense"):
                auto_ohlc_container = ui.column().classes("gap-1 p-2")
                with auto_ohlc_container:
                    auto_ohlc_open_label = ui.label("Open: --").classes("text-xs text-gray-300")
                    auto_ohlc_high_label = ui.label("High: --").classes("text-xs text-green-400")
                    auto_ohlc_low_label = ui.label("Low: --").classes("text-xs text-red-400")
                    auto_ohlc_close_label = ui.label("Close: --").classes("text-xs text-gray-300")
                    auto_ohlc_volume_label = ui.label("Volume: --").classes("text-xs text-blue-400")
            
            # Market Depth
            with ui.expansion("Depth", icon="layers").classes("w-full text-xs bg-gray-700/30 mt-2").props("dense"):
                with ui.element('div').style("display: grid; grid-template-columns: 1fr auto 1fr; gap: 0.5rem; padding: 0.5rem; width: 100%;"):
                    # Buy orders column
                    with ui.element('div').style("min-width: 0;"):
                        ui.label("Buy Orders").classes("text-green-400 font-semibold text-xs mb-1 text-center")
                        auto_depth_buy_container = ui.column().classes("gap-1")
                    
                    # Vertical separator
                    with ui.element('div').style("width: 1px; background: #4B5563; height: auto; align-self: stretch;"):
                        pass
                    
                    # Sell orders column
                    with ui.element('div').style("min-width: 0;"):
                        ui.label("Sell Orders").classes("text-red-400 font-semibold text-xs mb-1 text-center")
                        auto_depth_sell_container = ui.column().classes("gap-1")

            async def update_market_price(symbol, reset_defaults: bool = False):
                market_price_state['last_price'] = 0.0
                instrument_token = instruments.get(symbol)
                if instrument_token:
                    market_data = await fetch_api(f"/quotes/{broker}?instruments={instrument_token}")
                    if market_data:
                        quote = market_data[0]
                        last_price = _safe_float(quote.get('last_price', 0))
                        prev_close = _safe_float(quote.get('previous_close', 0))
                        market_price_state['last_price'] = last_price
                        pct_change = ((last_price - prev_close) / prev_close * 100.0) if prev_close else 0.0
                        market_price_label.text = f"{last_price:.2f} ({pct_change:.2f}%)"
                        
                        # Update OHLC data
                        open_price = _safe_float(quote.get('ohlc', {}).get('open', 0))
                        high_price = _safe_float(quote.get('ohlc', {}).get('high', 0))
                        low_price = _safe_float(quote.get('ohlc', {}).get('low', 0))
                        close_price = _safe_float(quote.get('ohlc', {}).get('close', 0))
                        volume = _safe_int(quote.get('volume', 0))
                        
                        auto_ohlc_open_label.text = f"Open: ₹{open_price:,.2f}"
                        auto_ohlc_high_label.text = f"High: ₹{high_price:,.2f}"
                        auto_ohlc_low_label.text = f"Low: ₹{low_price:,.2f}"
                        auto_ohlc_close_label.text = f"Close: ₹{close_price:,.2f}"
                        auto_ohlc_volume_label.text = f"Volume: {volume:,}"
                        
                        # Update Market Depth data
                        depth_data = quote.get("depth", {})
                        buy_orders = depth_data.get("buy", []) if depth_data else []
                        sell_orders = depth_data.get("sell", []) if depth_data else []
                        
                        # Clear previous depth data
                        auto_depth_buy_container.clear()
                        auto_depth_sell_container.clear()
                        
                        # Populate buy orders
                        with auto_depth_buy_container:
                            if buy_orders:
                                for order in buy_orders[:5]:  # Show top 5
                                    with ui.row().classes("justify-between w-full gap-2"):
                                        ui.label(f"₹{order.get('price', 0):,.2f}").classes("text-green-400 font-mono text-xs")
                                        ui.label(f"{order.get('quantity', 0):,}").classes("text-gray-300 text-xs")
                            else:
                                ui.label("No orders").classes("text-gray-500 text-xs italic text-center")
                        
                        # Populate sell orders
                        with auto_depth_sell_container:
                            if sell_orders:
                                for order in sell_orders[:5]:  # Show top 5
                                    with ui.row().classes("justify-between w-full gap-2"):
                                        ui.label(f"₹{order.get('price', 0):,.2f}").classes("text-red-400 font-mono text-xs")
                                        ui.label(f"{order.get('quantity', 0):,}").classes("text-gray-300 text-xs")
                            else:
                                ui.label("No orders").classes("text-gray-500 text-xs italic text-center")
                    else:
                        market_price_label.text = "--"
                        auto_ohlc_open_label.text = "Open: --"
                        auto_ohlc_high_label.text = "High: --"
                        auto_ohlc_low_label.text = "Low: --"
                        auto_ohlc_close_label.text = "Close: --"
                        auto_ohlc_volume_label.text = "Volume: --"
                else:
                    market_price_label.text = "--"
                    auto_ohlc_open_label.text = "Open: --"
                    auto_ohlc_high_label.text = "High: --"
                    auto_ohlc_low_label.text = "Low: --"
                    auto_ohlc_close_label.text = "Close: --"
                    auto_ohlc_volume_label.text = "Volume: --"
                market_price_label.update()
                if reset_defaults:
                    reset_auto_defaults(market_price_state['last_price'])

            def reset_auto_defaults(last_price: Optional[float] = None) -> None:
                if last_price is not None:
                    market_price_state['last_price'] = _safe_float(last_price, 0.0)
                apply_symbol_defaults(market_price_state['last_price'])
                limit_price.value = 0.0
                limit_price.update()
                stop_loss_value.value = 2.0
                stop_loss_value.update()
                target_value.value = 3.0
                target_value.update()

            if symbol_select.value:
                await update_market_price(symbol_select.value, reset_defaults=True)
            symbol_select.on_value_change(lambda e: asyncio.create_task(update_market_price(e.value, reset_defaults=True)))

            # Additional Settings
            with ui.row().classes('w-full gap-6 mt-4'):
                trailing_stop = ui.switch('Enable Trailing Stop Loss').classes('text-white')

            def update_atr_field():
                atr_period.visible = stop_loss_type.value == 'ATR Based'

            stop_loss_type.on_value_change(update_atr_field)
            update_atr_field()

            def auto_calc_context() -> Dict[str, Any]:
                stop_mode_value = 'Percentage' if str(stop_loss_type.value).lower().startswith('percentage') else 'Absolute'
                entry_price = market_price_state['last_price']
                limit_val = _safe_float(limit_price.value)
                if limit_val > 0:
                    entry_price = limit_val
                return {
                    'transaction_type': transaction_type.value,
                    'entry_price': entry_price,
                    'stop_loss': _safe_float(stop_loss_value.value),
                    'target_price': _safe_float(target_value.value),
                    'risk_amount': _safe_float(risk_per_trade.value),
                    'stop_mode': stop_mode_value,
                    'target_mode': 'Absolute',
                    'stop_input': _safe_float(stop_loss_value.value),
                    'target_input': _safe_float(target_value.value),
                    'capital': position_calc_defaults.get('capital'),
                    'risk_percent': position_calc_defaults.get('risk_percent'),
                    'brokerage': position_calc_defaults.get('brokerage'),
                    'charges_pct': position_calc_defaults.get('charges_pct'),
                    'risk_only': True,
                }

            def auto_default_calc() -> Dict[str, Any]:
                template = user_storage.get('_position_calc_template', template_defaults) or {}
                capital_val = template.get('capital', position_calc_defaults.get('capital', 0.0))
                risk_percent_val = template.get('risk_percent', position_calc_defaults.get('risk_percent', 0.0))
                risk_amount_val = capital_val * (risk_percent_val / 100.0) if capital_val and risk_percent_val else 0.0
                return {
                    'transaction_type': transaction_type.value,
                    'entry_price': market_price_state['last_price'],
                    'stop_loss': 0.0,
                    'target_price': 0.0,
                    'risk_amount': risk_amount_val,
                    'stop_mode': 'Absolute',
                    'target_mode': 'Absolute',
                    'stop_input': 0.0,
                    'target_input': 0.0,
                    'capital': capital_val,
                    'risk_percent': risk_percent_val,
                    'brokerage': template.get('brokerage', position_calc_defaults.get('brokerage', 0.0)),
                    'charges_pct': template.get('charges_pct', position_calc_defaults.get('charges_pct', 0.0)),
                    'risk_only': True,
                }

            async def fetch_equity_funds() -> Dict[str, Any]:
                try:
                    response = await fetch_api(f"/funds/{broker}")
                    return response or {}
                except Exception as exc:
                    logger.warning(f"Unable to fetch funds for calculator: {exc}")
                    return {}

            def get_basket_totals() -> Dict[str, float]:
                totals_fn = basket_controller.get('totals')
                if callable(totals_fn):
                    try:
                        return totals_fn() or {}
                    except Exception as exc:
                        logger.warning(f"Unable to compute basket totals: {exc}")
                return {'required': 0.0, 'risk': 0.0, 'reward': 0.0}

            def apply_auto_calc(result: Dict[str, Any]) -> None:
                if result.get('risk_per_trade') is not None:
                    risk_per_trade.value = result['risk_per_trade']
                    risk_per_trade.update()

                if result.get('stop_loss') is not None:
                    if result.get('stop_mode') == 'Percentage' and result.get('stop_input') is not None:
                        stop_loss_value.value = result['stop_input']
                    else:
                        stop_loss_value.value = result['stop_loss']
                    stop_loss_value.update()

                if result.get('target') is not None:
                    if result.get('target_mode') == 'Percentage' and result.get('target_input') is not None:
                        target_value.value = result['target_input']
                    else:
                        target_value.value = result['target']
                    target_value.update()

                entry_val = result.get('entry_price')
                if entry_val is not None:
                    limit_price.value = entry_val
                    limit_price.update()

                summary = result.get('summary', {})
                details = []
                if result.get('risk_per_trade') is not None:
                    details.append(f"Risk {result['risk_per_trade']:.2f}")
                if summary.get('profit_amount') is not None:
                    details.append(f"Potential {summary['profit_amount']:.2f}")
                if details:
                    ui.notify('Auto order sizing applied: ' + ' | '.join(details), type='positive')

            # Loading container
            loading_container = ui.column().classes('w-full')

            def build_auto_order_payload() -> tuple[Optional[Dict[str, Any]], Optional[str]]:
                """
                Build auto order payload with validation.
                Returns: (order_data, error_message)
                """
                if not all(validation_state.values()):
                    return None, 'Please fix form errors'

                if not symbol_select.value or symbol_select.value not in instruments:
                    return None, 'Please select a valid symbol'

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
                return order_data, None

            # Place Auto Order Action
            async def place_auto_order():
                order_data, error = build_auto_order_payload()
                if error:
                    ui.notify(error, type='negative')
                    return
                if not order_data:
                    return

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

                        ui.button('Create Auto Order', on_click=confirm_auto_order).classes('bg-orange-600 text-white px-4 py-2 rounded')

                dialog.open()

            def add_auto_to_basket() -> None:
                order_data, error = build_auto_order_payload()
                if error:
                    ui.notify(error, type='negative')
                    return
                if not order_data:
                    return
                add_fn = basket_controller.get('add')
                if callable(add_fn):
                    basket_entry = {
                        "type": "Auto",
                        "endpoint": "/auto-orders/",
                        "success_key": "auto_order_id",
                        "payload": order_data,
                    }
                    add_fn(basket_entry)

            basket_observer_register = basket_controller.get('register_observer')

            with ui.row().classes('order-action-row w-full gap-3 mt-4'):
                create_position_calculator_button(
                    label='Position Sizing',
                    defaults=position_calc_defaults,
                    get_context=auto_calc_context,
                    apply_callback=apply_auto_calc,
                    button_classes='flex-1 px-6 py-3 rounded-lg font-medium text-lg shadow',
                    button_style='background: linear-gradient(135deg, #f59e0b, #fb923c); color:#431407; border:none; box-shadow:0 8px 16px rgba(251,146,60,0.45);',
                    get_default_context=auto_default_calc,
                    funds_fetcher=fetch_equity_funds,
                    basket_totals_provider=get_basket_totals,
                    register_basket_listener=basket_observer_register if callable(basket_observer_register) else None,
                    on_reset=lambda: reset_auto_defaults(market_price_state['last_price'])
                )
                auto_basket_btn = ui.button('Add to Basket', icon='add_shopping_cart', on_click=add_auto_to_basket)\
                    .classes('flex-1 px-6 py-3 rounded-lg font-medium text-lg shadow-inner')
                apply_button_style(auto_basket_btn, 'background: linear-gradient(135deg, #1f2937, #111827); color:#f3f4f6; border:1px solid rgba(148,163,184,0.35);')
                auto_place_btn = ui.button('Create Auto Order', icon="smart_toy", on_click=place_auto_order)\
                    .classes('flex-1 px-6 py-3 rounded-lg font-semibold text-lg shadow')
                apply_button_style(auto_place_btn, 'background: linear-gradient(135deg, #fb923c, #f97316); color:#431407; border:none; box-shadow:0 8px 16px rgba(251,146,60,0.45);')
