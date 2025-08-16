# Market Data Utilities - market_utils.py
# Generic functions for market data operations that can be used across modules

import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


async def fetch_batch_ltp_data(symbols, get_cached_instruments, broker, fetch_api):
    """Fetch live data for a batch of symbols"""
    ltp_data_map = {}

    try:
        # Get instruments mapping
        all_instruments_map = await get_cached_instruments(broker)

        # Get instrument tokens
        instrument_tokens = []
        symbol_to_token_map = {}

        for symbol in symbols:
            instrument_token = None
            if isinstance(all_instruments_map, dict):
                instrument_token = all_instruments_map.get(symbol)
            else:
                for inst in all_instruments_map:
                    if hasattr(inst, "trading_symbol") and inst.trading_symbol == symbol:
                        instrument_token = inst.instrument_token
                        break

            if instrument_token:
                instrument_tokens.append(instrument_token)
                symbol_to_token_map[symbol] = instrument_token

        # Fetch batch LTP data
        if instrument_tokens:
            instruments_param = ",".join(instrument_tokens)
            ltp_response = await fetch_api(f"/ltp/{broker}", params={"instruments": instruments_param})

            if ltp_response and isinstance(ltp_response, list):
                for item in ltp_response:
                    token = item.get("instrument_token")
                    if token:
                        # Map back to symbol
                        for symbol, sym_token in symbol_to_token_map.items():
                            if sym_token == token:
                                ltp_data_map[symbol] = item
                                break

    except Exception as e:
        logger.error(f"Error fetching batch LTP data: {e}")

    return ltp_data_map


async def fetch_indices_sectors(fetch_api):
    """Fetch unique indices and sectors from ALL_STOCKS table"""
    try:
        response = await fetch_api("/api/data/fetch-table-data/", params={
            "table_name": "ALL_STOCKS",
            "columns": ["STK_INDEX"],
            "required_db": "nsedata"
        })

        if response and "data" in response:
            indices_set = set()
            for row in response["data"]:
                index_value = row.get("STK_INDEX", "").strip()
                if index_value and index_value != "":
                    indices_set.add(index_value)

            return sorted(list(indices_set))
        return ["NIFTY 50", "NIFTY BANK", "NIFTY IT"]  # Fallback
    except Exception as e:
        logger.error(f"Error fetching indices/sectors: {e}")
        return ["NIFTY 50", "NIFTY BANK", "NIFTY IT"]  # Fallback


async def fetch_stocks_by_index(fetch_api, selected_index):
    """Fetch stocks for a specific index/sector"""
    try:
        response = await fetch_api("/api/data/fetch-table-data/", params={
            "table_name": "ALL_STOCKS",
            "columns": ["SYMBOL"],
            "filters": f'"STK_INDEX" = \'{selected_index}\'',
            "required_db": "nsedata"
        })

        if response and "data" in response:
            symbols = [row.get("SYMBOL", "") for row in response["data"]]
            return sorted([s for s in symbols if s.strip()])
        return []
    except Exception as e:
        logger.error(f"Error fetching stocks for index {selected_index}: {e}")
        return []


def format_currency(value):
    """Format currency values consistently"""
    if isinstance(value, (int, float)):
        return f"Rs.{value:,.2f}"
    return "Rs.0.00"


def format_percentage(value):
    """Format percentage values consistently"""
    if isinstance(value, (int, float)):
        return f"{value:+.2f}%"
    return "0.00%"


def get_change_class(value):
    """Get CSS class based on value change"""
    if value > 0:
        return "positive-change"
    elif value < 0:
        return "negative-change"
    else:
        return "neutral-change"


def get_trend_icon(value):
    """Get trend icon based on value"""
    if value > 0:
        return "trending_up"
    elif value < 0:
        return "trending_down"
    else:
        return "trending_flat"


def get_change_color_class(change):
    """Get CSS color class based on change value"""
    if change > 0:
        return "text-green-400"
    elif change < 0:
        return "text-red-400"
    else:
        return "text-gray-400"


def get_trend_symbol(change):
    """Get trend symbol based on change value"""
    if change > 0:
        return "▲"
    elif change < 0:
        return "▼"
    else:
        return "▬"