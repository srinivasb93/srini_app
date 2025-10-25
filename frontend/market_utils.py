# Market Data Utilities - market_utils.py
# Generic functions for market data operations that can be used across modules

import logging

logger = logging.getLogger(__name__)


async def fetch_batch_ltp_data(symbols, get_cached_instruments, broker, fetch_api):
    """Fetch live data for a batch of symbols using quotes endpoint"""
    ltp_data_map = {}

    try:
        # Get instruments mapping
        all_instruments_map = await get_cached_instruments(broker)

        # Build trading_symbols parameter (comma-separated symbols)
        valid_symbols = []
        for symbol in symbols:
            # Try to find the symbol in instruments
            symbol_found = False
            if isinstance(all_instruments_map, dict):
                if symbol in all_instruments_map:
                    valid_symbols.append(symbol)
                    symbol_found = True
            else:
                for inst in all_instruments_map:
                    if hasattr(inst, "trading_symbol") and inst.trading_symbol == symbol:
                        valid_symbols.append(symbol)
                        symbol_found = True
                        break
            
            # If not found in instruments, still try with the symbol directly
            if not symbol_found:
                valid_symbols.append(symbol)

        # Fetch batch quotes data using get_market_quotes endpoint
        if valid_symbols:
            symbols_param = ",".join(valid_symbols)
            logger.info(f"Fetching quotes for symbols: {symbols_param}")
            
            quotes_response = await fetch_api(f"/quotes/{broker}", params={"trading_symbols": symbols_param})

            if quotes_response and isinstance(quotes_response, list):
                logger.info(f"Received {len(quotes_response)} quote responses")
                for item in quotes_response:
                    # Extract symbol from the response
                    symbol = item.get("trading_symbol", "")
                    
                    # Extract OHLC data
                    ohlc_data = item.get("ohlc", {})
                    open_price = ohlc_data.get("open", 0.0) if isinstance(ohlc_data, dict) else 0.0
                    high_price = ohlc_data.get("high", 0.0) if isinstance(ohlc_data, dict) else 0.0
                    low_price = ohlc_data.get("low", 0.0) if isinstance(ohlc_data, dict) else 0.0
                    close_price = ohlc_data.get("close", 0.0) if isinstance(ohlc_data, dict) else 0.0
                    
                    # Get last price and calculate previous close
                    last_price = item.get("last_price", 0.0)
                    net_change = item.get("net_change", 0.0)
                    pct_change = item.get("pct_change", 0.0)
                    
                    # Calculate previous_close from net_change if available
                    previous_close = 0.0
                    if last_price > 0 and net_change != 0:
                        previous_close = last_price - net_change
                    elif close_price > 0:
                        previous_close = close_price
                    
                    # Build the data map with comprehensive market data
                    ltp_data_map[symbol] = {
                        "last_price": last_price,
                        "volume": item.get("volume", 0),
                        "previous_close": previous_close,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "change": net_change,
                        "change_percent": pct_change,
                        "depth": item.get("depth", {}),  # Include market depth data
                    }
                    logger.debug(f"Mapped data for {symbol}: {ltp_data_map[symbol]}")
            else:
                logger.warning(f"Invalid quotes response format: {quotes_response}")

    except Exception as e:
        logger.error(f"Error fetching batch LTP data: {e}", exc_info=True)

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