import locale
import time
import logging
from functools import wraps
import requests
import math

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

def calculate_atr(df, period):
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr']

def sanitize_floats(obj):
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj
    
def fetch_indicies_sectors_list(required='all'):
    indices = ["NIFTY 50", "NIFTY NEXT 50", "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY MIDCAP 150",
               "NIFTY SMALLCAP 250", "NIFTY SMALLCAP 50", "NIFTY SMALLCAP 100", "NIFTY 100", "NIFTY 200",
               "NIFTY 500", "NIFTY MIDSMALLCAP 400", "NIFTY MIDCAP SELECT", "NIFTY LARGEMIDCAP 250"]

    sector_indices_list = ["NIFTY BANK", "NIFTY FINANCIAL SERVICES", "NIFTY IT", "NIFTY MEDIA", "NIFTY METAL",
                           "NIFTY PHARMA", "NIFTY PSU BANK", "NIFTY REALTY", "NIFTY AUTO", "NIFTY FMCG",
                           "NIFTY HEALTHCARE INDEX", "NIFTY PRIVATE BANK", "NIFTY CONSUMER DURABLES",
                           "NIFTY OIL AND GAS"]

    thematic_indices_list = ["NIFTY ENERGY", "NIFTY CPSE", "NIFTY INFRASTRUCTURE", "NIFTY100 LIQUID 15",
                             "NIFTY PSE", "NIFTY COMMODITIES", "NIFTY MNC", "NIFTY INDIA CONSUMPTION",
                             "NIFTY MIDCAP LIQUID 15", "NIFTY SERVICES SECTOR", "NIFTY INDIA DIGITAL",
                             "NIFTY EV"]
    
    core_indices_sectors = ["NIFTY 50", "NIFTY NEXT 50", "NIFTY MIDCAP 50",  "NIFTY SMALLCAP 50", "NIFTY MIDCAP SELECT",
                            "NIFTY BANK", "NIFTY FINANCIAL SERVICES", "NIFTY IT", "NIFTY MEDIA", "NIFTY METAL",
                           "NIFTY PHARMA", "NIFTY PSU BANK", "NIFTY REALTY", "NIFTY AUTO", "NIFTY FMCG",
                           "NIFTY HEALTHCARE INDEX", "NIFTY PRIVATE BANK", "NIFTY CONSUMER DURABLES",
                           "NIFTY OIL AND GAS", "NIFTY ENERGY", "NIFTY CPSE",]

    sectors = sector_indices_list + thematic_indices_list
    all_symbols = indices + sectors

    if required == 'indices':
        return indices
    elif required == 'sectors':
        return sectors
    elif required == 'core_indices_sectors':
        return indices + sector_indices_list

    return all_symbols
