# Market Data Utilities - market_utils.py
# Generic functions for market data operations that can be used across modules

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# MARKET STATUS UTILITIES
# ============================================================================

def calculate_next_market_event_seconds():
    """
    Calculate seconds until next critical market event.
    
    Market Events:
    - 9:00 AM - Pre-Open starts
    - 9:15 AM - Normal session starts
    - 3:30 PM (15:30) - Closing session starts
    - 4:00 PM (16:00) - Market closes
    
    Returns:
        int: Seconds until next market event (minimum 1 second)
    """
    now = datetime.now()
    
    # Market event times (24-hour format)
    events = [
        (9, 0),    # Pre-Open starts
        (9, 15),   # Normal session starts
        (15, 30),  # Closing session starts
        (16, 0),   # Market closes
    ]
    
    # Find next event today
    for hour, minute in events:
        event_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now < event_time:
            seconds_until = (event_time - now).total_seconds()
            return max(1, seconds_until)  # At least 1 second
    
    # All events passed today, next event is 9:00 AM tomorrow
    next_day = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
    seconds_until = (next_day - now).total_seconds()
    return max(1, seconds_until)


def calculate_next_check_time(is_trading_day, error_retry=False):
    """
    Calculate seconds until next market status check based on current state.
    
    Args:
        is_trading_day (bool): Whether today is a trading day
        error_retry (bool): Whether this is an error retry
        
    Returns:
        tuple: (seconds_until_next_check, description)
    """
    now = datetime.now()
    
    if error_retry:
        # Retry in 5 minutes on error
        return 300, "Error occurred, retrying in 5 minutes"
    
    elif not is_trading_day:
        # Not a trading day (weekend/holiday)
        # Check again at 9:00 AM tomorrow
        next_check = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if now.hour >= 9:
            next_check += timedelta(days=1)
        seconds_until = (next_check - now).total_seconds()
        hours_until = seconds_until / 3600
        return seconds_until, f"Non-trading day, next check in {hours_until:.1f} hours"
    
    else:
        # Trading day - schedule for next event
        seconds_until = calculate_next_market_event_seconds()
        minutes_until = seconds_until / 60
        return seconds_until, f"Trading day, next update in {minutes_until:.1f} minutes"


def get_market_status_color_class(status_color):
    """
    Map backend status color to Tailwind CSS class.
    
    Args:
        status_color (str): Color name from backend ("green", "orange", "yellow", "red", "gray")
        
    Returns:
        str: Tailwind CSS class for the color
    """
    color_map = {
        "green": "text-green-500",
        "orange": "text-orange-500",
        "yellow": "text-yellow-500",
        "red": "text-red-500",
        "gray": "text-gray-500"
    }
    return color_map.get(status_color, "text-gray-500")


def format_market_status_tooltip(message, next_event=None):
    """
    Format tooltip text for market status display.
    
    Args:
        message (str): Main status message
        next_event (str, optional): Next market event description
        
    Returns:
        str: Formatted tooltip text
    """
    tooltip = message
    if next_event:
        tooltip += f"\n{next_event}"
    return tooltip


class MarketStatusManager:
    """
    Manages market status updates with event-driven scheduling.
    
    This class encapsulates the logic for fetching market status from the backend
    and scheduling updates at optimal times (only at market events).
    """
    
    def __init__(self, fetch_api_func):
        """
        Initialize the market status manager.
        
        Args:
            fetch_api_func: Async function to call backend API
        """
        self.fetch_api = fetch_api_func
        self.current_timer = None
        self.logger = logging.getLogger(f"{__name__}.MarketStatusManager")
    
    async def fetch_market_status(self):
        """
        Fetch current market status from backend.
        
        Returns:
            dict: Market status response or None on error
        """
        try:
            response = await self.fetch_api("/market-status", method="GET")
            return response
        except Exception as e:
            self.logger.error(f"Error fetching market status: {e}")
            return None
    
    def parse_market_status_response(self, response):
        """
        Parse market status response and extract key information.
        
        Args:
            response (dict): API response
            
        Returns:
            dict: Parsed status information with keys:
                - status (str): Market status text
                - color (str): Status color
                - message (str): Status message
                - tooltip (str): Formatted tooltip text
                - is_trading_day (bool): Whether it's a trading day
                - css_class (str): CSS class for icon color
        """
        if not response:
            return {
                "status": "Unknown",
                "color": "gray",
                "message": "Unable to fetch market status",
                "tooltip": "Market status unknown",
                "is_trading_day": False,
                "css_class": "text-gray-500"
            }
        
        status = response.get("market_status", "Unknown")
        color = response.get("status_color", "gray")
        message = response.get("status_message", "Market status unknown")
        next_event = response.get("next_event")
        is_trading_day = response.get("is_trading_day", False)
        
        return {
            "status": status,
            "color": color,
            "message": message,
            "tooltip": format_market_status_tooltip(message, next_event),
            "is_trading_day": is_trading_day,
            "css_class": get_market_status_color_class(color)
        }
    
    def calculate_next_update_time(self, is_trading_day, error_retry=False):
        """
        Calculate when the next status update should occur.
        
        Args:
            is_trading_day (bool): Whether today is a trading day
            error_retry (bool): Whether this is an error retry
            
        Returns:
            tuple: (seconds, description)
        """
        seconds, description = calculate_next_check_time(is_trading_day, error_retry)
        self.logger.info(f"Market status: {description}")
        return seconds, description
    
    def cancel_current_timer(self):
        """Cancel the current timer if it exists."""
        if self.current_timer is not None:
            try:
                self.current_timer.cancel()
                self.logger.debug("Cancelled existing market status timer")
            except Exception as e:
                self.logger.warning(f"Error cancelling timer: {e}")
            self.current_timer = None


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