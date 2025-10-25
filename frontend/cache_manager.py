"""
Frontend Cache Manager for Trading Application
Handles client-side caching with intelligent invalidation
"""

import json
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Callable
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

class FrontendCacheConfig:
    """Frontend cache configuration with TTL settings"""
    
    # Real-time data (very short TTL)
    LIVE_QUOTES = 15          # 15 seconds
    PORTFOLIO_DATA = 30       # 30 seconds  
    ORDER_STATUS = 20         # 20 seconds
    POSITION_DATA = 45        # 45 seconds
    
    # Semi-static data
    USER_PREFERENCES = 900    # 15 minutes
    RISK_METRICS = 60         # 1 minute
    STRATEGY_LIST = 180       # 3 minutes
    WATCHLIST = 300           # 5 minutes
    
    # Static data (long TTL)
    INSTRUMENTS = 21600       # 6 hours (refresh twice daily)
    MARKET_DATA = 300         # 5 minutes
    ANALYTICS_DATA = 600      # 10 minutes

class FrontendCacheManager:
    """
    Lightweight frontend cache manager optimized for NiceGUI
    Features:
    - In-memory storage with automatic cleanup
    - Size-based eviction (LRU)
    - Trading-specific cache keys
    - Automatic invalidation
    """
    
    def __init__(self, max_size: int = 500):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time > entry['expires']:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _remove_entry(self, key: str):
        """Remove cache entry and its access time"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_lru(self):
        """Evict least recently used entries when cache is full"""
        if len(self.cache) >= self.max_size:
            # Sort by access time and remove oldest
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            evict_count = max(1, len(self.cache) - self.max_size + 50)  # Evict in batches
            
            for key, _ in sorted_keys[:evict_count]:
                self._remove_entry(key)
                self.stats['evictions'] += 1
    
    def generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key"""
        key_parts = [prefix]
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
        
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.append(hashlib.md5(str(sorted_kwargs).encode()).hexdigest()[:8])
        
        return ":".join(key_parts)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        self.stats['total_requests'] += 1
        
        # Periodic cleanup
        if self.stats['total_requests'] % 100 == 0:
            self._cleanup_expired()
        
        if key not in self.cache:
            self.stats['misses'] += 1
            return None
        
        entry = self.cache[key]
        current_time = time.time()
        
        # Check expiration
        if current_time > entry['expires']:
            self._remove_entry(key)
            self.stats['misses'] += 1
            return None
        
        # Update access time for LRU
        self.access_times[key] = current_time
        self.stats['hits'] += 1
        
        return entry['data']
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value"""
        current_time = time.time()
        
        # Ensure we don't exceed max size
        self._evict_lru()
        
        self.cache[key] = {
            'data': value,
            'expires': current_time + ttl,
            'created': current_time
        }
        self.access_times[key] = current_time
    
    def delete(self, key: str):
        """Delete specific cache entry"""
        self._remove_entry(key)
    
    def delete_pattern(self, pattern: str):
        """Delete entries matching pattern"""
        import re
        regex_pattern = pattern.replace('*', '.*')
        keys_to_delete = [
            key for key in self.cache.keys()
            if re.match(regex_pattern, key)
        ]
        
        for key in keys_to_delete:
            self._remove_entry(key)
        
        logger.debug(f"Deleted {len(keys_to_delete)} cache entries matching pattern: {pattern}")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.stats['hits'] / max(1, self.stats['total_requests']) * 100
        
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': round(hit_rate, 2),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'total_requests': self.stats['total_requests']
        }
    
    def cache_api_call(self, ttl: int, key_prefix: str = None):
        """Decorator to cache API call results"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                prefix = key_prefix or f"api_call:{func.__name__}"
                cache_key = self.generate_cache_key(prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function and cache result
                logger.debug(f"Cache miss for {func.__name__}, executing API call")
                result = await func(*args, **kwargs)
                
                # Only cache successful results
                if result and not (isinstance(result, dict) and result.get('error')):
                    self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator

# Global frontend cache manager
frontend_cache = FrontendCacheManager()

def cached_fetch_api(ttl: int = FrontendCacheConfig.LIVE_QUOTES):
    """Decorator for caching fetch_api calls"""
    return frontend_cache.cache_api_call(ttl, "fetch_api")

class TradingDataCache:
    """Specialized cache methods for trading data"""
    
    @staticmethod
    def cache_positions(user_id: str, broker: str, positions_data: Any):
        """Cache positions data"""
        key = frontend_cache.generate_cache_key("positions", user_id, broker)
        frontend_cache.set(key, positions_data, FrontendCacheConfig.POSITION_DATA)
    
    @staticmethod
    def get_positions(user_id: str, broker: str) -> Optional[Any]:
        """Get cached positions data"""
        key = frontend_cache.generate_cache_key("positions", user_id, broker)
        return frontend_cache.get(key)
    
    @staticmethod
    def cache_orders(user_id: str, broker: str, orders_data: Any):
        """Cache orders data"""
        key = frontend_cache.generate_cache_key("orders", user_id, broker)
        frontend_cache.set(key, orders_data, FrontendCacheConfig.ORDER_STATUS)
    
    @staticmethod
    def get_orders(user_id: str, broker: str) -> Optional[Any]:
        """Get cached orders data"""
        key = frontend_cache.generate_cache_key("orders", user_id, broker)
        return frontend_cache.get(key)
    
    @staticmethod
    def cache_risk_metrics(user_id: str, metrics_data: Any):
        """Cache risk metrics data"""
        key = frontend_cache.generate_cache_key("risk_metrics", user_id)
        frontend_cache.set(key, metrics_data, FrontendCacheConfig.RISK_METRICS)
    
    @staticmethod
    def get_risk_metrics(user_id: str) -> Optional[Any]:
        """Get cached risk metrics data"""
        key = frontend_cache.generate_cache_key("risk_metrics", user_id)
        return frontend_cache.get(key)
    
    @staticmethod
    def invalidate_user_data(user_id: str):
        """Invalidate all cached data for a user"""
        patterns = [
            f"*{user_id}*",
            f"positions:*{user_id}*",
            f"orders:*{user_id}*",
            f"risk_metrics:*{user_id}*",
            f"fetch_api:*/positions/{user_id}*",
            f"fetch_api:*/orders/{user_id}*"
        ]
        
        for pattern in patterns:
            frontend_cache.delete_pattern(pattern)
        
        logger.info(f"Invalidated cache for user {user_id}")
    
    @staticmethod
    def invalidate_market_data():
        """Invalidate market-related cache entries"""
        patterns = [
            "fetch_api:*/quotes*",
            "fetch_api:*/ltp*",
            "fetch_api:*/market*",
            "live_quotes:*"
        ]
        
        for pattern in patterns:
            frontend_cache.delete_pattern(pattern)
    
    @staticmethod
    def invalidate_broker_data(broker: str):
        """Invalidate broker-specific cache entries"""
        patterns = [
            f"fetch_api:*/profile/{broker}*",
            f"fetch_api:*/positions/{broker}*",
            f"fetch_api:*/orders/{broker}*",
            f"fetch_api:*/portfolio/{broker}*",
            f"fetch_api:*/margins/{broker}*",
            f"positions:*{broker}*",
            f"orders:*{broker}*"
        ]
        
        for pattern in patterns:
            frontend_cache.delete_pattern(pattern)
        
        logger.info(f"Invalidated cache for broker: {broker}")

def get_cache_stats():
    """Get cache statistics for monitoring"""
    return frontend_cache.get_stats()

def clear_cache():
    """Clear all cache (emergency function)"""
    frontend_cache.clear()

# Context manager for cache operations
class CacheContext:
    """Context manager for bulk cache operations"""
    
    def __init__(self, user_id: str = None):
        self.user_id = user_id
        self.operations = []
    
    def cache_positions(self, broker: str, data: Any):
        if self.user_id:
            TradingDataCache.cache_positions(self.user_id, broker, data)
    
    def cache_orders(self, broker: str, data: Any):
        if self.user_id:
            TradingDataCache.cache_orders(self.user_id, broker, data)
    
    def cache_risk_metrics(self, data: Any):
        if self.user_id:
            TradingDataCache.cache_risk_metrics(self.user_id, data)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # If there was an error, invalidate cache to ensure fresh data
            if self.user_id:
                TradingDataCache.invalidate_user_data(self.user_id)