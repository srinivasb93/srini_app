"""
Comprehensive Cache Manager for Algorithmic Trading Application
Handles Redis caching with intelligent invalidation strategies
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Callable
from functools import wraps
import asyncio

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class CacheConfig:
    """Cache configuration with TTL settings for different data types"""
    
    # Real-time data (short TTL for trading accuracy)
    LIVE_QUOTES = 30          # 30 seconds - real-time market data
    POSITIONS = 60            # 1 minute - user positions
    ORDERS = 30               # 30 seconds - order status
    RISK_METRICS = 120        # 2 minutes - risk calculations
    
    # Semi-static data (medium TTL)
    USER_PREFERENCES = 1800   # 30 minutes - user settings
    STRATEGIES = 300          # 5 minutes - strategy configurations
    EXECUTIONS = 180          # 3 minutes - execution status
    PORTFOLIO_METRICS = 600   # 10 minutes - portfolio analytics
    
    # Static data (long TTL)
    INSTRUMENTS = 86400       # 24 hours - instrument lists
    HISTORICAL_DATA = 3600    # 1 hour - historical prices
    MARKET_HOLIDAYS = 86400   # 24 hours - trading calendar
    
    # Database query results
    DAILY_PNL = 300           # 5 minutes - daily P&L calculations
    POSITION_AGGREGATES = 120 # 2 minutes - position summaries
    ORDER_HISTORY = 1800      # 30 minutes - historical orders

class TradingCacheManager:
    """
    Intelligent cache manager optimized for trading applications
    Features:
    - Redis backend with fallback to in-memory
    - Automatic cache invalidation
    - Trading-specific cache keys
    - Circuit breaker for cache failures
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", fallback_to_memory: bool = True):
        self.redis_url = redis_url
        self.fallback_to_memory = fallback_to_memory
        self.redis_client = None
        self.memory_cache = {}
        self.cache_failures = 0
        self.max_failures = 5
        self.circuit_open = False
        
    async def get_redis_client(self):
        """Get Redis client with connection handling"""
        if not REDIS_AVAILABLE:
            return None
            
        if self.circuit_open:
            return None
            
        try:
            if not self.redis_client:
                self.redis_client = await redis.from_url(
                    self.redis_url, 
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connection
                await self.redis_client.ping()
                self.cache_failures = 0
                self.circuit_open = False
                
            return self.redis_client
            
        except Exception as e:
            self.cache_failures += 1
            logger.warning(f"Redis connection failed ({self.cache_failures}/{self.max_failures}): {e}")
            
            if self.cache_failures >= self.max_failures:
                self.circuit_open = True
                logger.error("Redis circuit breaker opened - falling back to memory cache")
                
            return None
    
    def generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key from arguments"""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, dict):
                # Sort dict for consistent keys
                sorted_items = sorted(arg.items())
                key_parts.append(hashlib.md5(str(sorted_items).encode()).hexdigest()[:8])
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
        
        # Add keyword arguments
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.append(hashlib.md5(str(sorted_kwargs).encode()).hexdigest()[:8])
        
        return ":".join(key_parts)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value with Redis primary, memory fallback"""
        try:
            # Try Redis first
            redis_client = await self.get_redis_client()
            if redis_client:
                try:
                    value = await redis_client.get(key)
                    if value:
                        return json.loads(value)
                except Exception as e:
                    logger.debug(f"Redis get failed for key {key}: {e}")
            
            # Fall back to memory cache
            if self.fallback_to_memory and key in self.memory_cache:
                cached_item = self.memory_cache[key]
                if cached_item['expires'] > datetime.now():
                    return cached_item['data']
                else:
                    # Remove expired item
                    del self.memory_cache[key]
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value in both Redis and memory"""
        try:
            serialized_value = json.dumps(value, default=str)
            
            # Try Redis first
            redis_client = await self.get_redis_client()
            if redis_client:
                try:
                    await redis_client.setex(key, ttl, serialized_value)
                except Exception as e:
                    logger.debug(f"Redis set failed for key {key}: {e}")
            
            # Always store in memory as fallback
            if self.fallback_to_memory:
                self.memory_cache[key] = {
                    'data': value,
                    'expires': datetime.now() + timedelta(seconds=ttl)
                }
                
                # Clean up expired memory cache entries periodically
                if len(self.memory_cache) > 1000:
                    await self._cleanup_memory_cache()
                    
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
    
    async def delete(self, key: str):
        """Delete cached value from both Redis and memory"""
        try:
            # Delete from Redis
            redis_client = await self.get_redis_client()
            if redis_client:
                try:
                    await redis_client.delete(key)
                except Exception as e:
                    logger.debug(f"Redis delete failed for key {key}: {e}")
            
            # Delete from memory
            if key in self.memory_cache:
                del self.memory_cache[key]
                
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
    
    async def delete_pattern(self, pattern: str):
        """Delete multiple keys matching pattern"""
        try:
            # Delete from Redis
            redis_client = await self.get_redis_client()
            if redis_client:
                try:
                    keys = await redis_client.keys(pattern)
                    if keys:
                        await redis_client.delete(*keys)
                except Exception as e:
                    logger.debug(f"Redis pattern delete failed for {pattern}: {e}")
            
            # Delete from memory cache
            if self.fallback_to_memory:
                import re
                regex_pattern = pattern.replace('*', '.*')
                keys_to_delete = [
                    key for key in self.memory_cache.keys() 
                    if re.match(regex_pattern, key)
                ]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    
        except Exception as e:
            logger.error(f"Cache pattern delete error for {pattern}: {e}")
    
    async def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        try:
            now = datetime.now()
            expired_keys = [
                key for key, item in self.memory_cache.items()
                if item['expires'] <= now
            ]
            for key in expired_keys:
                del self.memory_cache[key]
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Memory cache cleanup error: {e}")
    
    def cache_result(self, ttl: int, key_prefix: str = None):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                prefix = key_prefix or f"func:{func.__name__}"
                cache_key = self.generate_cache_key(prefix, *args, **kwargs)
                
                # Try to get cached result
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                if result is not None:
                    await self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator

# Global cache manager instance
cache_manager = TradingCacheManager()

# Convenience decorators for common cache patterns
def cache_user_data(ttl: int = CacheConfig.USER_PREFERENCES):
    """Cache user-specific data"""
    return cache_manager.cache_result(ttl, "user_data")

def cache_market_data(ttl: int = CacheConfig.LIVE_QUOTES):
    """Cache market data with short TTL"""
    return cache_manager.cache_result(ttl, "market_data")

def cache_portfolio_data(ttl: int = CacheConfig.PORTFOLIO_METRICS):
    """Cache portfolio/position data"""
    return cache_manager.cache_result(ttl, "portfolio")

def cache_strategy_data(ttl: int = CacheConfig.STRATEGIES):
    """Cache strategy-related data"""
    return cache_manager.cache_result(ttl, "strategies")

async def invalidate_user_cache(user_id: str):
    """Invalidate all cached data for a specific user"""
    patterns = [
        f"user_data:*{user_id}*",
        f"portfolio:*{user_id}*", 
        f"strategies:*{user_id}*"
    ]
    for pattern in patterns:
        await cache_manager.delete_pattern(pattern)

async def invalidate_market_cache(symbol: str = None):
    """Invalidate market data cache"""
    if symbol:
        await cache_manager.delete_pattern(f"market_data:*{symbol}*")
    else:
        await cache_manager.delete_pattern("market_data:*")