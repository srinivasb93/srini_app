import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime, date, timedelta
import asyncio
import aiohttp
from openchart import NSEData
import nsepython as nsep
from nsetools import Nse
from backend.app.database import get_nsedata_db, get_db
from backend.app.routes.sip_routes import get_current_user
from backend.app.api_manager import get_cached_user_token
from common_utils.upstox_utils import UpstoxAsyncClient, UpstoxUnit, MarketQuote
import redis.asyncio as redis
import json
import hashlib
from functools import wraps
import time

logger = logging.getLogger(__name__)

equity_router = APIRouter(prefix="/equity", tags=["equity-data"])

# Global instances
nse_openchart = NSEData()
nse_openchart.download()
nse_tools = Nse()

# Redis configuration
REDIS_URL = "redis://localhost:6379"
CACHE_TTL = {
    "quote": 60,  # 1 minute for live quotes
    "historical_1d": 3600,  # 1 hour for daily data
    "historical_intraday": 300,  # 5 minutes for intraday
    "static_data": 86400,  # 24 hours for static data like lot sizes
    "market_data": 180,  # 3 minutes for market data
}


class RedisCache:
    """Async Redis cache manager"""

    def __init__(self):
        self.redis_client = None

    async def get_client(self):
        if not self.redis_client:
            self.redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
        return self.redis_client

    async def get(self, key: str) -> Optional[Any]:
        try:
            client = await self.get_client()
            value = await client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        try:
            client = await self.get_client()
            await client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")

    async def delete(self, pattern: str):
        try:
            client = await self.get_client()
            keys = await client.keys(pattern)
            if keys:
                await client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")


cache = RedisCache()


def cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key from function arguments"""
    key_parts = []
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool, list)):
            key_parts.append(f"{k}:{v}")

    key_str = ":".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def async_cache(ttl_key: str):
    """Decorator for caching async functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Skip caching if explicitly disabled
            if kwargs.get('skip_cache', False):
                kwargs.pop('skip_cache', None)
                return await func(*args, **kwargs)

            # Generate cache key
            cache_key = f"{func.__name__}:{cache_key_generator(*args, **kwargs)}"

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value

            # Execute function and cache result
            result = await func(*args, **kwargs)
            ttl = CACHE_TTL.get(ttl_key, 3600)
            await cache.set(cache_key, result, ttl)
            logger.debug(f"Cached {cache_key} with TTL {ttl}")

            return result

        return wrapper

    return decorator


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.call_times = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            # Remove old calls outside the period
            self.call_times = [t for t in self.call_times if now - t < self.period]

            if len(self.call_times) >= self.calls:
                # Wait until the oldest call expires
                sleep_time = self.period - (now - self.call_times[0]) + 0.1
                await asyncio.sleep(sleep_time)
                await self.acquire()  # Retry
            else:
                self.call_times.append(now)



@equity_router.get("/quote/{symbol}")
@async_cache("quote")
async def get_quote(
        symbol: str,
        source: str = Query(default="nsepython"),
        fallback_sources: List[str] = Query(default=[]),
        skip_cache: bool = Query(default=False),
        user_id: Optional[str] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Enhanced quote endpoint with caching"""
    sources = [source] + fallback_sources
    data = None
    used_source = None

    for src in sources:
        try:
            if src == "nsepython":
                data = nsep.nsetools_get_quote(symbol)

            elif src == "upstox":
                if user_id is None:
                    raise HTTPException(status_code=401, detail="Authentication required")

                access_token = await get_cached_user_token(user_id, db)
                if not access_token:
                    raise ValueError("Upstox access token not found")

                async with UpstoxAsyncClient(access_token) as client:
                    # Get instrument key
                    instrument_key = await client.get_instrument_key(symbol, "NSE")
                    if not instrument_key:
                        instrument_key = f"NSE_EQ|{symbol}"

                    # Get market quotes
                    quotes = await client.get_market_quotes([instrument_key], mode="full")

                    if instrument_key in quotes:
                        quote: MarketQuote = quotes[instrument_key]
                        data = {
                            'last_price': quote.last_price,
                            'open': quote.open,
                            'high': quote.high,
                            'low': quote.low,
                            'close': quote.close,
                            'volume': quote.volume,
                            'bid': quote.bid_price,
                            'ask': quote.ask_price,
                            'bid_quantity': quote.bid_quantity,
                            'ask_quantity': quote.ask_quantity,
                            'avg_price': quote.avg_price,
                            'pct_change': quote.pct_change,
                            'depth': quote.depth
                        }

            elif src == "openchart":
                today = datetime.today()
                df = nse_openchart.historical(
                    symbol=symbol,
                    exchange='NSE',
                    start=today,
                    end=today,
                    interval="1d"
                )
                if not df.empty:
                    latest = df.iloc[-1]
                    data = {
                        'last_price': latest['close'],
                        'open': latest['open'],
                        'high': latest['high'],
                        'low': latest['low'],
                        'volume': latest['volume'],
                    }

            elif src == "nsetools":
                data = nse_tools.get_quote(symbol)

            else:
                raise ValueError(f"Unknown source: {src}")

            used_source = src
            break

        except Exception as e:
            logger.warning(f"Source {src} failed for {symbol}: {str(e)}")
            continue

    if data is None:
        raise HTTPException(status_code=503, detail="All data sources failed")

    return {
        "data": data,
        "source": used_source,
        "cached": False
    }


@equity_router.get("/ltp/{symbols}")
@async_cache("quote")
async def get_ltp(
        symbols: str,
        source: str = Query(default="upstox"),
        skip_cache: bool = Query(default=False),
        user_id: Optional[str] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get last traded price for multiple symbols"""
    symbol_list = symbols.split(",")

    if source == "upstox":
        if user_id is None:
            raise HTTPException(status_code=401, detail="Authentication required")

        access_token = await get_cached_user_token(user_id, db)
        if not access_token:
            raise HTTPException(status_code=401, detail="Upstox access token not found")

        async with UpstoxAsyncClient(access_token) as client:
            # Get instrument keys for symbols
            instrument_keys = []
            for symbol in symbol_list:
                instrument_key = await client.get_instrument_key(symbol.strip(), "NSE")
                if instrument_key:
                    instrument_keys.append(instrument_key)
                else:
                    instrument_keys.append(f"NSE_EQ|{symbol.strip()}")

            # Get LTP data
            ltp_data = await client.get_ltp(instrument_keys)

            # Map back to symbols
            result = {}
            for symbol, key in zip(symbol_list, instrument_keys):
                result[symbol.strip()] = ltp_data.get(key, 0)

            return {
                "data": result,
                "source": "upstox",
                "cached": False
            }
    else:
        # Fallback to individual quote fetching
        result = {}
        for symbol in symbol_list:
            try:
                quote_response = await get_quote(
                    symbol=symbol.strip(),
                    source=source,
                    skip_cache=skip_cache,
                    user_id=user_id,
                    db=db
                )
                result[symbol.strip()] = quote_response["data"].get("last_price", 0)
            except:
                result[symbol.strip()] = 0

        return {
            "data": result,
            "source": source,
            "cached": False
        }


@equity_router.get("/ohlc/{symbols}")
@async_cache("quote")
async def get_ohlc_quotes(
        symbols: str,
        skip_cache: bool = Query(default=False),
        user_id: Optional[str] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get OHLC quotes for multiple symbols"""
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    symbol_list = symbols.split(",")
    access_token = await get_cached_user_token(user_id, db)
    if not access_token:
        raise HTTPException(status_code=401, detail="Upstox access token not found")

    async with UpstoxAsyncClient(access_token) as client:
        # Get instrument keys
        instrument_keys = []
        for symbol in symbol_list:
            instrument_key = await client.get_instrument_key(symbol.strip(), "NSE")
            if instrument_key:
                instrument_keys.append(instrument_key)
            else:
                instrument_keys.append(f"NSE_EQ|{symbol.strip()}")

        # Get OHLC quotes
        ohlc_data = await client.get_ohlc_quotes(instrument_keys)

        # Map back to symbols
        result = {}
        for symbol, key in zip(symbol_list, instrument_keys):
            result[symbol.strip()] = ohlc_data.get(key, {})

        return {
            "data": result,
            "source": "upstox",
            "cached": False
        }


@equity_router.delete("/cache/clear")
async def clear_cache(
        pattern: Optional[str] = Query(default="*", description="Cache key pattern to clear"),
        user_id: Optional[str] = Depends(get_current_user)
):
    """Clear cache entries matching pattern"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    await cache.delete(pattern)
    return {"message": f"Cache cleared for pattern: {pattern}"}


@equity_router.get("/instruments")
@async_cache("static_data")
async def get_instruments(
        exchange: str = Query(default="NSE", description="Exchange (NSE, NFO, BSE)"),
        skip_cache: bool = Query(default=False),
        user_id: Optional[str] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Fetch all instruments with caching"""
    try:
        if user_id:
            access_token = await get_cached_user_token(user_id, db)

            if access_token:
                async with UpstoxAsyncClient(access_token) as client:
                    df = await client.get_instruments()

                    # Filter by exchange
                    if exchange:
                        df = df[df['exchange'] == exchange]

                    instruments = df.to_dict(orient='records')
                    return {
                        "data": instruments,
                        "count": len(instruments),
                        "cached": False,
                        "source": "upstox"
                    }

        # Fallback to direct URL fetch
        url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    df = pd.read_json(await response.read())

                    # Filter by exchange
                    if exchange:
                        df = df[df['exchange'] == exchange]

                    instruments = df.to_dict(orient='records')
                    return {
                        "data": instruments,
                        "count": len(instruments),
                        "cached": False,
                        "source": "direct"
                    }
                else:
                    raise HTTPException(
                        status_code=response.status,
                        detail="Failed to fetch instruments"
                    )

    except Exception as e:
        logger.error(f"Error fetching instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/search_instruments/{query}")
@async_cache("static_data")
async def search_instruments(
        query: str,
        exchange: Optional[str] = Query(default=None),
        segment: Optional[str] = Query(default=None),
        skip_cache: bool = Query(default=False),
        user_id: Optional[str] = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Search instruments by name or symbol"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    access_token = await get_cached_user_token(user_id, db)
    if not access_token:
        raise HTTPException(status_code=401, detail="Upstox access token not found")

    async with UpstoxAsyncClient(access_token) as client:
        results = await client.search_instruments(query, exchange, segment)

        return {
            "data": results.to_dict(orient='records'),
            "count": len(results),
            "cached": False
        }


@equity_router.get("/fno_lot_sizes")
@async_cache("static_data")
async def get_fno_lot_sizes(
        source: str = Query(default="nsepython"),
        fallback_sources: List[str] = Query(default=[]),
        skip_cache: bool = Query(default=False)
):
    """Fetch FNO lot sizes with source selection"""
    sources = [source] + fallback_sources
    data = None
    used_source = None

    for src in sources:
        try:
            if src == "nsepython":
                data = nsep.nse_get_fno_lot_sizes()
            elif src == "nsetools":
                data = nse_tools.get_fno_lot_sizes()
            else:
                raise ValueError(f"Unknown source: {src}")

            used_source = src
            break
        except Exception as e:
            logger.warning(f"Source {src} failed: {str(e)}")
            continue

    if data is None:
        raise HTTPException(status_code=503, detail="All data sources failed")

    return {"data": data, "source": used_source, "cached": False}


@equity_router.get("/index_list")
@async_cache("static_data")
async def get_index_list(
        source: str = Query(default="nsepython"),
        fallback_sources: List[str] = Query(default=[]),
        skip_cache: bool = Query(default=False)
):
    """Fetch list of NSE indices with source selection"""
    sources = [source] + fallback_sources
    data = None
    used_source = None

    for src in sources:
        try:
            if src == "nsepython":
                data = nsep.nse_get_index_list()
            elif src == "nsetools":
                data = nse_tools.get_index_list()
            else:
                raise ValueError(f"Unknown source: {src}")

            used_source = src
            break
        except Exception as e:
            logger.warning(f"Source {src} failed: {str(e)}")
            continue

    if data is None:
        raise HTTPException(status_code=503, detail="All data sources failed")

    return {"data": data, "source": used_source, "cached": False}


@equity_router.get("/advances_declines")
@async_cache("market_data")
async def get_advances_declines(
        source: str = Query(default="nsepython"),
        fallback_sources: List[str] = Query(default=[]),
        skip_cache: bool = Query(default=False)
):
    """Fetch market advances/declines with source selection"""
    sources = [source] + fallback_sources
    data = None
    used_source = None

    for src in sources:
        try:
            if src == "nsepython":
                data = nsep.nse_get_advances_declines()
            elif src == "nsetools":
                data = nse_tools.get_advances_declines()
            else:
                raise ValueError(f"Unknown source: {src}")

            used_source = src
            break
        except Exception as e:
            logger.warning(f"Source {src} failed: {str(e)}")
            continue

    if data is None:
        raise HTTPException(status_code=503, detail="All data sources failed")

    return {"data": data, "source": used_source, "cached": False}


@equity_router.get("/top_gainers")
@async_cache("market_data")
async def get_top_gainers(
        source: str = Query(default="nsepython"),
        fallback_sources: List[str] = Query(default=[]),
        skip_cache: bool = Query(default=False)
):
    """Fetch top gainers with source selection"""
    sources = [source] + fallback_sources
    data = None
    used_source = None

    for src in sources:
        try:
            if src == "nsepython":
                data = nsep.nse_get_top_gainers()
            elif src == "nsetools":
                data = nse_tools.get_top_gainers(index="NIFTY 50")
            else:
                raise ValueError(f"Unknown source: {src}")

            used_source = src
            break
        except Exception as e:
            logger.warning(f"Source {src} failed: {str(e)}")
            continue

    if data is None:
        raise HTTPException(status_code=503, detail="All data sources failed")

    return {"data": data, "source": used_source, "cached": False}


@equity_router.get("/top_losers")
@async_cache("market_data")
async def get_top_losers(
        source: str = Query(default="nsepython"),
        fallback_sources: List[str] = Query(default=[]),
        skip_cache: bool = Query(default=False)
):
    """Fetch top losers with source selection"""
    sources = [source] + fallback_sources
    data = None
    used_source = None

    for src in sources:
        try:
            if src == "nsepython":
                data = nsep.nse_get_top_losers()
            elif src == "nsetools":
                data = nse_tools.get_top_losers()
            else:
                raise ValueError(f"Unknown source: {src}")

            used_source = src
            break
        except Exception as e:
            logger.warning(f"Source {src} failed: {str(e)}")
            continue

    if data is None:
        raise HTTPException(status_code=503, detail="All data sources failed")

    return {"data": data, "source": used_source, "cached": False}


@equity_router.get("/stock_codes")
@async_cache("static_data")
async def get_stock_codes(skip_cache: bool = Query(default=False)):
    """Fetch list of stock codes (nsetools only)"""
    try:
        data = nse_tools.get_stock_codes()
        return {"data": data, "source": "nsetools", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/is_valid_code/{code}")
async def is_valid_code(code: str):
    """Validate stock code (nsetools only)"""
    try:
        valid = nse_tools.is_valid_code(code)
        return {"valid": valid, "source": "nsetools"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/52_week_high")
@async_cache("market_data")
async def get_52_week_high(skip_cache: bool = Query(default=False)):
    """Fetch stocks at 52-week high (nsetools only)"""
    try:
        data = nse_tools.get_52_week_high()
        return {"data": data, "source": "nsetools", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/52_week_low")
@async_cache("market_data")
async def get_52_week_low(skip_cache: bool = Query(default=False)):
    """Fetch stocks at 52-week low (nsetools only)"""
    try:
        data = nse_tools.get_52_week_low()
        return {"data": data, "source": "nsetools", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/index_quote/{index}")
@async_cache("quote")
async def get_index_quote(index: str, skip_cache: bool = Query(default=False)):
    """Fetch quote for an index (nsetools only)"""
    try:
        data = nse_tools.get_index_quote(index)
        return {"data": data, "source": "nsetools", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/pre_open_market")
@async_cache("market_data")
async def get_pre_open_market(skip_cache: bool = Query(default=False)):
    """Fetch pre-open market data"""
    try:
        data = nsep.nse_preopen_movers(key="NIFTY")
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/bhavcopy/{bhav_date}")
async def get_bhavcopy(
        bhav_date: str,
        source: str = Query(default="nsepython"),
        fallback_sources: List[str] = Query(default=[]),
        skip_cache: bool = Query(default=False)
):
    """Fetch bhavcopy data for a specific date with source selection"""
    sources = [source] + fallback_sources
    data = None
    used_source = None

    for src in sources:
        try:
            if src == "nsepython":
                data = nsep.get_bhavcopy(date=bhav_date)
            elif src == "nsetools":
                data = nse_tools.get_bhavcopy(bhav_date)
                data = pd.DataFrame(data).to_dict(orient='records')
            else:
                raise ValueError(f"Unknown source: {src}")

            used_source = src
            break
        except Exception as e:
            logger.warning(f"Source {src} failed for bhavcopy on {bhav_date}: {str(e)}")
            continue

    if data is None:
        raise HTTPException(status_code=503, detail="All data sources failed")

    return {"data": data, "source": used_source, "cached": False}


@equity_router.get("/holiday_list")
@async_cache("static_data")
async def get_holiday_list(
        source: str = Query(default="nsepython"),
        fallback_sources: List[str] = Query(default=[]),
        skip_cache: bool = Query(default=False)
):
    """Fetch trading holiday list with source selection"""
    sources = [source] + fallback_sources
    data = None
    used_source = None

    for src in sources:
        try:
            if src == "nsepython":
                data = nsep.holiday_master()
            else:
                raise ValueError(f"Unknown source: {src}")

            used_source = src
            break
        except Exception as e:
            logger.warning(f"Source {src} failed: {str(e)}")
            continue

    if data is None:
        raise HTTPException(status_code=503, detail="All data sources failed")

    return {"data": data, "source": used_source, "cached": False}


@equity_router.get("/search/{symbol}")
@async_cache("static_data")
async def search_symbols(
        symbol: str,
        exchange: str = Query(default="NSE"),
        exact_match: bool = Query(default=False),
        skip_cache: bool = Query(default=False)
):
    """Search for symbols using openchart"""
    try:
        df = nse_openchart.search(symbol, exchange=exchange, exact_match=exact_match)
        data = df.to_dict(orient='records')
        return {"data": data, "source": "openchart", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/market_status")
@async_cache("market_data")
async def get_market_status(skip_cache: bool = Query(default=False)):
    """Fetch current market status using nsepython"""
    try:
        data = nsep.is_market_open()
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/fiidii")
@async_cache("market_data")
async def get_fiidii(mode: str, skip_cache: bool = Query(default=False)):
    """Fetch FII/DII trading data using nsepython"""
    try:
        if mode == "pandas":
            data = nsep.nse_fiidii(mode='pandas')
        else:
            data = nsep.nse_fiidii(mode='other')
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/events")
@async_cache("market_data")
async def get_events(skip_cache: bool = Query(default=False)):
    """Fetch market event calendar using nsepython"""
    try:
        data = nsep.nse_events()
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/results/{symbol}")
@async_cache("static_data")
async def get_past_results(symbol: str, skip_cache: bool = Query(default=False)):
    """Fetch past financial results for a symbol using nsepython"""
    try:
        data = nsep.nse_past_results(symbol)
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/index_history/{index}")
@async_cache("historical_1d")
async def get_index_history(
        index: str,
        start_date: str,
        end_date: Optional[str] = Query(default=None),
        skip_cache: bool = Query(default=False)
):
    """Fetch historical data for an index using nsepython"""
    if not end_date:
        end_date = datetime.now().strftime("%d-%m-%Y")

    try:
        data = nsep.index_history(index, start_date, end_date)
        data = data.rename(columns={
            'HistoricalDate': 'date',
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close',
            'VOLUME': 'volume'
        })
        data = data[['date', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/index_pe_pb_div/{index}")
@async_cache("market_data")
async def get_index_pe_pb_div(
        index: str,
        start_date: str,
        end_date: Optional[str] = Query(default=None),
        skip_cache: bool = Query(default=False)
):
    """Fetch PE, PB, and dividend yield data for an index using nsepython"""
    if not end_date:
        end_date = datetime.now().strftime("%d-%m-%Y")

    try:
        data = nsep.index_pe_pb_div(index, start_date, end_date).to_dict(orient='records')
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/index_total_returns/{index}")
@async_cache("historical_1d")
async def get_index_total_returns(
        index: str,
        start_date: str,
        end_date: Optional[str] = Query(default=None),
        skip_cache: bool = Query(default=False)
):
    """Fetch total returns data for an index using nsepython"""
    if not end_date:
        end_date = datetime.now().strftime("%d-%m-%Y")

    try:
        data = nsep.index_total_returns(index, start_date, end_date).to_dict(orient='records')
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/large_deals")
@async_cache("market_data")
async def get_large_deals(
        mode: str = Query(default="bulk_deals"),
        skip_cache: bool = Query(default=False)
):
    """Fetch large deals (bulk, short, or block) using nsepython"""
    try:
        data = nsep.nse_largedeals(mode=mode).to_dict(orient='records')
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/large_deals_historical")
@async_cache("historical_1d")
async def get_large_deals_historical(
        from_date: str,
        to_date: Optional[str] = Query(default=None),
        mode: str = Query(default="bulk_deals"),
        skip_cache: bool = Query(default=False)
):
    """Fetch historical large deals data using nsepython"""
    if not to_date:
        to_date = datetime.now().strftime("%d-%m-%Y")

    try:
        data = nsep.nse_largedeals_historical(from_date, to_date, mode).to_dict(orient='records')
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@equity_router.get("/price_band_hitters/{bandtype}")
@async_cache("market_data")
async def get_price_band_hitters(
        bandtype: str = Path(),
        view: str = Query(default="AllSec"),
        skip_cache: bool = Query(default=False)
):
    """Fetch price band hitters using nsepython"""
    try:
        data = nsep.nse_price_band_hitters(bandtype, view).to_dict(orient='records')
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/beta/{symbol}")
@async_cache("market_data")
async def get_beta(
        symbol: str,
        days: int = Query(default=365),
        symbol2: str = Query(default="NIFTY 50"),
        skip_cache: bool = Query(default=False)
):
    """Calculate beta for a stock relative to a benchmark index using nsepython"""
    try:
        data = nsep.get_beta(symbol, days, symbol2)
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@equity_router.get("/security_wise_archive/{symbol}")
@async_cache("historical_1d")
async def get_security_wise_archive(
        symbol: str,
        from_date: str,
        to_date: Optional[str] = Query(default=None),
        series: str = Query(default="ALL"),
        skip_cache: bool = Query(default=False)
):
    """Fetch security-wise archive data using nsepython"""
    if not to_date:
        to_date = datetime.now().strftime("%d-%m-%Y")

    try:
        data = nsep.security_wise_archive(from_date, to_date, symbol, series).to_dict(orient='records')
        return {"data": data, "source": "nsepython", "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def main():
    """Main function to test the endpoints in this module."""
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)

    # Example test for get_quote endpoint
    response = client.get("/equity_data/quote/SBIN")
    print(response.json())

    # Example test for get_ltp endpoint
    response = client.get("/equity_data/ltp/SBIN")
    print(response.json())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

