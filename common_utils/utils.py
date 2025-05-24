import locale
import time
import logging
from functools import wraps
import requests

logger = logging.getLogger(__name__)


def add_comma(num):
    """API to add comma to a number"""
    locale.setlocale(locale.LC_ALL, 'en_IN')
    price = locale.currency(num, grouping=True)
    return price[1:]


def insert_commas(num):
    num_str = str(num)
    result = ""
    for i, digit in enumerate(num_str[::-1]):
        if i > 0:
            if i <= 3 and i % 3 == 0:
                result = "," + result
            if i == 4:
                pass
            elif i > 4 and i % 2 != 0:
                result = "," + result
        result = digit + result
    return result


def format_currency(value):
    if isinstance(value, (int, float)):
        return f"₹{value:,.2f}"
    return value


def notify(title, message, type='success'):
    logger.info(f"Notification: {title} - {message} ({type})")


def calculate_atr(df, period):
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr']


def get_historical_data(instrument_token):
    # Placeholder: Implement API call to fetch historical data
    return []
