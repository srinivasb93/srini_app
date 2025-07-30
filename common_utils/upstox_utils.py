# upstox_utils_enhanced.py
import pandas as pd
from datetime import datetime, timedelta
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union
import gzip
import io
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class UpstoxUnit(Enum):
    """Upstox supported intervals"""
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"


class UpstoxExchange(Enum):
    """Upstox exchanges"""
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"
    MCX = "MCX"


@dataclass
class MarketQuote:
    """Market quote data structure"""
    symbol: str
    last_price: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    avg_price: Optional[float] = None
    pct_change: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_quantity: Optional[int] = None
    ask_quantity: Optional[int] = None
    oi: Optional[int] = None
    depth: Optional[dict] = None
    timestamp: Optional[datetime] = None


class UpstoxAsyncClient:
    """Async Upstox client with enhanced features"""

    def __init__(self, access_token: str, rate_limit: int = 100):
        self.access_token = access_token
        self.base_url = "https://api.upstox.com"
        self.session = None
        self.rate_limiter = AsyncRateLimiter(rate_limit, 60)  # 100 requests per minute
        self._instruments_cache = None
        self._instruments_last_update = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict:
        """Make rate-limited HTTP request"""
        await self.rate_limiter.acquire()

        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Upstox API error: {response.status} - {error_text}")
                    raise Exception(f"API error: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for URL: {url}")
            raise
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            raise

    async def get_historical_candle_data(
            self,
            instrument_key: str,
            unit: Union[str, UpstoxUnit],
            interval: int,
            to_date: Union[str, datetime],
            from_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Fetch historical candle data from Upstox V3 API

        Args:
            instrument_key: Instrument key (e.g., "NSE_EQ|INE669E01016")
            unit: Time unit (e.g., "days", "weeks", "months")
            interval: Time interval in minutes (e.g., 1, 5, 15, 30, 60)
            to_date: End date (YYYY-MM-DD format or datetime)
            from_date: Start date (YYYY-MM-DD format or datetime)

        Returns:
            DataFrame with OHLCV data
        """
        # Convert dates to string format
        if isinstance(to_date, datetime):
            to_date = to_date.strftime("%Y-%m-%d")
        if isinstance(from_date, datetime):
            from_date = from_date.strftime("%Y-%m-%d")

        # Convert interval enum to string
        if isinstance(interval, UpstoxUnit):
            interval = interval.value

        if isinstance(unit, UpstoxUnit):
            unit = unit.value

        # V3 API endpoint format: /v3/historical-candle/{instrumentKey}/{interval}/{to_date}/{from_date}
        # Note: instrumentKey should be URL encoded
        encoded_instrument_key = instrument_key.replace("|", "%7C")
        url = f"{self.base_url}/v3/historical-candle/{encoded_instrument_key}/{unit}/{interval}/{to_date}/{from_date}"

        data = await self._make_request("GET", url)

        # V3 API response structure
        if "data" in data:
            candles = data.get("data", {}).get("candles", [])

            # Convert to DataFrame
            # V3 API returns: [timestamp, open, high, low, close, volume, oi]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            df.set_index("timestamp", inplace=True)

            return df
        else:
            raise Exception(f"Failed to fetch historical data: {data}")

    async def get_intraday_candle_data(
            self,
            instrument_key: str,
            unit: Union[str, UpstoxUnit] = UpstoxUnit.MINUTES,
            interval: int = 1
    ) -> pd.DataFrame:
        """
        Fetch intraday candle data from V3 API

        Args:
            instrument_key: Instrument key
            unit: Time unit (default: "minutes")
            interval: Time interval (default: 1minute)

        Returns:
            DataFrame with intraday OHLCV data
        """
        if isinstance(interval, UpstoxUnit):
            interval = interval.value

        # V3 API endpoint for intraday data
        url = f"{self.base_url}/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"

        data = await self._make_request("GET", url)

        if "data" in data:
            # V3 API returns data directly in 'data' field
            candles = data.get("data", [])

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            df.set_index("timestamp", inplace=True)

            return df
        else:
            raise Exception(f"Failed to fetch intraday data: {data}")

    async def get_market_quotes(
            self,
            symbols: List[str],
            mode: str = "full"
    ) -> Dict[str, MarketQuote]:
        """
        Fetch market quotes from V3 API

        Args:
            symbols: List of instrument keys
            mode: Quote mode ("full", "ohlc", "ltp")

        Returns:
            Dictionary of symbol -> MarketQuote
        """
        symbol_param = ",".join(symbols)

        url = f"{self.base_url}/v2/market-quote/quotes?instrument_key={symbol_param}"

        data = await self._make_request("GET", url)

        quotes = {}
        if "data" in data:
            for symbol, quote_data in data.get("data", {}).items():
                if mode == "full":
                    quote = MarketQuote(
                        symbol=symbol,
                        last_price=quote_data.get("last_price", 0),
                        avg_price=quote_data.get("avg_price", 0),
                        pct_change=quote_data.get("net_change", 0),
                        open=quote_data.get("ohlc", {}).get("open", 0),
                        high=quote_data.get("ohlc", {}).get("high", 0),
                        low=quote_data.get("ohlc", {}).get("low", 0),
                        close=quote_data.get("ohlc", {}).get("close", 0),
                        volume=quote_data.get("volume", 0),
                        oi=quote_data.get("oi", 0),
                        depth= quote_data.get("depth", {}),
                        timestamp=datetime.now()
                    )

                    # Add bid/ask if available
                    depth = quote_data.get("depth", {})
                    if depth:
                        buy_depth = depth.get("buy", [])
                        sell_depth = depth.get("sell", [])
                        if buy_depth:
                            quote.bid_price = buy_depth[0].get("price")
                            quote.bid_quantity = buy_depth[0].get("quantity")
                        if sell_depth:
                            quote.ask_price = sell_depth[0].get("price")
                            quote.ask_quantity = sell_depth[0].get("quantity")

                elif mode == "ohlc":
                    quote = MarketQuote(
                        symbol=symbol,
                        last_price=quote_data.get("last_price", 0),
                        open=quote_data.get("ohlc", {}).get("open", 0),
                        high=quote_data.get("ohlc", {}).get("high", 0),
                        low=quote_data.get("ohlc", {}).get("low", 0),
                        close=quote_data.get("ohlc", {}).get("close", 0),
                        volume=quote_data.get("volume", 0),
                        timestamp=datetime.now()
                    )

                else:  # ltp
                    quote = MarketQuote(
                        symbol=symbol,
                        last_price=quote_data.get("last_price", 0),
                        open=0, high=0, low=0, close=0, volume=0,
                        timestamp=datetime.now()
                    )

                quotes[symbol] = quote

        return quotes

    async def get_ltp(self,
                      symbols: List[str]
                      ):
        """
        Fetch last traded price (LTP) for all instruments

        Returns:
            Dictionary of symbol -> last traded price
        """
        url = f"{self.base_url}/v3/market-quote/ltp?instrument_key={','.join(symbols)}"

        data = await self._make_request("GET", url)

        ltp_data = {}
        if "data" in data:
            for symbol, quote_data in data.get("data", {}).items():
                ltp_data[symbol] = quote_data.get("last_price", 0)

        return ltp_data

    async def get_ohlc_quotes(
            self,
            symbols: List[str]
    ) -> Dict[str, MarketQuote]:
        """
        Fetch OHLC quotes for all instruments

        Args:
            symbols: List of instrument keys

        Returns:
            Dictionary of symbol -> MarketQuote with OHLC data
        """
        url = f"{self.base_url}/v2/market-quote/ohlc"
        params = {
            "instrument_key": ",".join(symbols),
            "interval": "1d"  # Daily OHLC
        }


        data = await self._make_request("GET", url, params=params)

        ohlc_quotes = {}
        if "data" in data:
            for symbol, quote_data in data.get("data", {}).items():
                ohlc_quotes[symbol] = quote_data

        return ohlc_quotes

    async def get_instruments(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch all instruments with caching

        Args:
            force_refresh: Force refresh of instruments cache

        Returns:
            DataFrame with instrument details
        """
        # Check cache
        if not force_refresh and self._instruments_cache is not None:
            cache_age = datetime.now() - self._instruments_last_update
            if cache_age.total_seconds() < 86400:  # 24 hours
                return self._instruments_cache

        url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"

        async with self.session.get(url) as response:
            if response.status == 200:
                # Read compressed data
                data = await response.read()

                # Decompress and parse
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
                    df = pd.read_json(f)

                # Cache the result
                self._instruments_cache = df
                self._instruments_last_update = datetime.now()

                return df
            else:
                raise Exception(f"Failed to fetch instruments: {response.status}")

    async def search_instruments(
            self,
            query: str,
            exchange: Optional[str] = None,
            segment: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Search instruments by name or symbol

        Args:
            query: Search query
            exchange: Filter by exchange (NSE, BSE, etc.)
            segment: Filter by segment (EQ, FO, etc.)

        Returns:
            DataFrame with matching instruments
        """
        df = await self.get_instruments()

        # Apply filters
        mask = (
                df['name'].str.contains(query, case=False, na=False) |
                df['trading_symbol'].str.contains(query, case=False, na=False)
        )

        if exchange:
            mask &= (df['exchange'] == exchange)

        if segment:
            mask &= (df['segment'] == segment)

        return df[mask]

    async def get_instrument_key(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """
        Get instrument key for a symbol

        Args:
            symbol: Trading symbol
            exchange: Exchange (default: NSE)

        Returns:
            Instrument key or None if not found
        """
        df = await self.get_instruments()

        mask = (df['trading_symbol'] == symbol) & (df['exchange'] == exchange)
        matches = df[mask]

        if not matches.empty:
            return matches.iloc[0]['instrument_key']

        return None

    async def get_user_profile(self) -> Dict:
        """Fetch user profile information"""
        url = f"{self.base_url}/v2/user/profile"
        return await self._make_request("GET", url)

    async def get_funds_and_margin(self) -> Dict:
        """Fetch funds and margin information"""
        url = f"{self.base_url}/v2/user/get-funds-and-margin"
        return await self._make_request("GET", url)

    async def get_holdings(self) -> List[Dict]:
        """Fetch user holdings"""
        url = f"{self.base_url}/v2/portfolio/long-term-holdings"
        data = await self._make_request("GET", url)
        return data.get("data", [])

    async def get_positions(self) -> List[Dict]:
        """Fetch user positions"""
        url = f"{self.base_url}/v2/portfolio/short-term-positions"
        data = await self._make_request("GET", url)
        return data.get("data", [])


class AsyncRateLimiter:
    """Async rate limiter for API calls"""

    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.call_times = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make a call"""
        async with self.lock:
            now = asyncio.get_event_loop().time()

            # Remove old calls
            cutoff = now - self.period
            self.call_times = [t for t in self.call_times if t > cutoff]

            # Check if we can make a call
            if len(self.call_times) >= self.calls:
                # Calculate wait time
                sleep_time = self.call_times[0] + self.period - now + 0.1
                await asyncio.sleep(sleep_time)
                await self.acquire()  # Retry
            else:
                self.call_times.append(now)


# Helper functions for data conversion
def normalize_candle_data(candles: List[List], interval: str) -> pd.DataFrame:
    """Normalize candle data to standard format"""
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Add additional columns
    df["interval"] = interval
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # Calculate technical indicators
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    return df


def calculate_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pivot points for the given data"""
    df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
    df["r1"] = 2 * df["pivot"] - df["low"]
    df["s1"] = 2 * df["pivot"] - df["high"]
    df["r2"] = df["pivot"] + (df["high"] - df["low"])
    df["s2"] = df["pivot"] - (df["high"] - df["low"])
    df["r3"] = df["high"] + 2 * (df["pivot"] - df["low"])
    df["s3"] = df["low"] - 2 * (df["high"] - df["pivot"])

    return df

async def get_symbol_for_instrument(instrument_token: str) -> str:
    """Fetch instruments and reverse the key value pair to get the symbol for a given instrument token"""
    async def fetch_instruments():
        async with UpstoxAsyncClient("") as client:
            instruments_df = await client.get_instruments()
            return instruments_df

    instruments_df = asyncio.run(fetch_instruments())

    # Reverse the key-value pair to get symbol for instrument token
    symbol = instruments_df[instruments_df['instrument_token'] == instrument_token]['trading_symbol'].values
    return symbol[0] if symbol else None


# Example usage
async def example_usage():
    """Example of using the enhanced Upstox client"""
    access_token = ""

    async with UpstoxAsyncClient(access_token) as client:
        # Get historical data
        # df = await client.get_historical_candle_data(
        #     instrument_key="NSE_EQ|INE669E01016",
        #     unit=UpstoxUnit.DAYS,
        #     interval=1,
        #     to_date=datetime.now(),
        #     from_date=datetime.now() - timedelta(days=30)
        # )
        # print(f"Historical data shape: {df.shape}")
        # print(df.head())

        # Get market quotes
        quotes = await client.get_ohlc_quotes(
            ["NSE_EQ|INE669E01016", "NSE_EQ|INE002A01018"]
        )
        print(quotes)
        # for symbol, quote in quotes.items():
        #     print(f"{symbol}: LTP={quote.last_price}, Volume={quote.volume}")
        #     print(quote)

        # Search instruments
        # results = await client.search_instruments("RELIANCE", exchange="NSE")
        # print(f"Found {len(results)} instruments")


if __name__ == "__main__":
    asyncio.run(example_usage())