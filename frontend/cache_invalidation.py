"""
Cache Invalidation Strategies for Trading Application
Handles intelligent cache invalidation based on data dependencies and trading operations
"""

import logging
from typing import List, Optional
from cache_manager import frontend_cache, TradingDataCache

logger = logging.getLogger(__name__)

class CacheInvalidationManager:
    """
    Centralized cache invalidation manager
    Handles complex dependencies between different data types
    """
    
    def __init__(self):
        self.invalidation_strategies = {
            'order_placed': self._invalidate_order_placed,
            'position_updated': self._invalidate_position_updated, 
            'strategy_started': self._invalidate_strategy_started,
            'strategy_stopped': self._invalidate_strategy_stopped,
            'user_settings_changed': self._invalidate_user_settings,
            'market_data_updated': self._invalidate_market_data,
            'execution_completed': self._invalidate_execution_completed,
            'portfolio_rebalanced': self._invalidate_portfolio_rebalanced,
            'risk_limits_changed': self._invalidate_risk_limits_changed,
            'broker_changed': self._invalidate_broker_changed
        }
    
    def invalidate(self, event_type: str, context: dict = None):
        """
        Invalidate caches based on event type and context
        """
        if event_type in self.invalidation_strategies:
            try:
                self.invalidation_strategies[event_type](context or {})
                logger.debug(f"Cache invalidation completed for event: {event_type}")
            except Exception as e:
                logger.error(f"Cache invalidation failed for {event_type}: {e}")
        else:
            logger.warning(f"Unknown cache invalidation event: {event_type}")
    
    def _invalidate_order_placed(self, context: dict):
        """Invalidate caches when an order is placed"""
        broker = context.get('broker')
        user_id = context.get('user_id')
        
        # Invalidate order-related caches
        if broker:
            frontend_cache.delete_pattern(f"fetch_api:*/orders/{broker}*")
            frontend_cache.delete_pattern(f"safe_api:*/orders/{broker}*")
            frontend_cache.delete_pattern(f"orders:{user_id}:{broker}*")
        
        # Invalidate position caches (orders affect positions)
        if broker:
            frontend_cache.delete_pattern(f"fetch_api:*/positions/{broker}*")
            frontend_cache.delete_pattern(f"positions:{user_id}:{broker}*")
        
        # Invalidate portfolio and risk metrics (orders affect portfolio)
        if user_id:
            frontend_cache.delete_pattern(f"risk_metrics:{user_id}*")
            frontend_cache.delete_pattern(f"fetch_api:*/portfolio*")
        
        logger.info(f"Invalidated caches for order placement (broker: {broker}, user: {user_id})")
    
    def _invalidate_position_updated(self, context: dict):
        """Invalidate caches when positions are updated"""
        broker = context.get('broker')
        user_id = context.get('user_id')
        symbol = context.get('symbol')
        
        # Invalidate position caches
        if broker:
            frontend_cache.delete_pattern(f"fetch_api:*/positions/{broker}*")
            frontend_cache.delete_pattern(f"positions:{user_id}:{broker}*")
        
        # Invalidate specific symbol data if provided
        if symbol:
            frontend_cache.delete_pattern(f"*{symbol}*")
        
        # Invalidate portfolio and analytics
        frontend_cache.delete_pattern(f"fetch_api:*/portfolio*")
        frontend_cache.delete_pattern(f"fetch_api:*/analytics*")
        
        logger.info(f"Invalidated caches for position update (symbol: {symbol}, broker: {broker})")
    
    def _invalidate_strategy_started(self, context: dict):
        """Invalidate caches when a strategy is started"""
        broker = context.get('broker')
        strategy_id = context.get('strategy_id')
        
        # Invalidate strategy execution caches
        frontend_cache.delete_pattern("fetch_api:*/executions*")
        frontend_cache.delete_pattern("safe_api:*/executions*")
        frontend_cache.delete_pattern(f"fetch_api:*/strategies/{broker}*")
        frontend_cache.delete_pattern(f"safe_api:*/strategies/{broker}*")
        
        if strategy_id:
            frontend_cache.delete_pattern(f"*{strategy_id}*")
        
        logger.info(f"Invalidated caches for strategy start (strategy: {strategy_id}, broker: {broker})")
    
    def _invalidate_strategy_stopped(self, context: dict):
        """Invalidate caches when a strategy is stopped"""
        broker = context.get('broker')
        strategy_id = context.get('strategy_id')
        
        # Similar to strategy started, but also invalidate performance data
        frontend_cache.delete_pattern("fetch_api:*/executions*")
        frontend_cache.delete_pattern("safe_api:*/executions*")
        frontend_cache.delete_pattern(f"fetch_api:*/strategies/{broker}*")
        frontend_cache.delete_pattern(f"safe_api:*/strategies/{broker}*")
        frontend_cache.delete_pattern("fetch_api:*/performance*")
        
        if strategy_id:
            frontend_cache.delete_pattern(f"*{strategy_id}*")
        
        logger.info(f"Invalidated caches for strategy stop (strategy: {strategy_id}, broker: {broker})")
    
    def _invalidate_user_settings(self, context: dict):
        """Invalidate caches when user settings are changed"""
        user_id = context.get('user_id')
        setting_type = context.get('setting_type')
        
        if user_id:
            # Invalidate user-specific caches
            TradingDataCache.invalidate_user_data(user_id)
            
            # Invalidate settings-related caches
            frontend_cache.delete_pattern(f"fetch_api:*/preferences/{user_id}*")
            frontend_cache.delete_pattern(f"user_preferences:{user_id}*")
        
        # If risk settings changed, invalidate risk-related caches
        if setting_type == 'risk':
            frontend_cache.delete_pattern("risk_metrics*")
            frontend_cache.delete_pattern("fetch_api:*/risk*")
        
        logger.info(f"Invalidated caches for user settings change (user: {user_id}, type: {setting_type})")
    
    def _invalidate_market_data(self, context: dict):
        """Invalidate caches when market data is updated"""
        symbols = context.get('symbols', [])
        
        # Invalidate market data caches
        TradingDataCache.invalidate_market_data()
        
        # Invalidate specific symbol caches if provided
        for symbol in symbols:
            frontend_cache.delete_pattern(f"*{symbol}*")
        
        # Invalidate quote and LTP caches
        frontend_cache.delete_pattern("fetch_api:*/quotes*")
        frontend_cache.delete_pattern("fetch_api:*/ltp*")
        frontend_cache.delete_pattern("live_quotes*")
        
        logger.info(f"Invalidated market data caches for symbols: {symbols}")
    
    def _invalidate_execution_completed(self, context: dict):
        """Invalidate caches when a strategy execution completes"""
        broker = context.get('broker')
        strategy_id = context.get('strategy_id')
        user_id = context.get('user_id')
        
        # Invalidate execution and performance caches
        frontend_cache.delete_pattern("fetch_api:*/executions*")
        frontend_cache.delete_pattern("safe_api:*/executions*")
        frontend_cache.delete_pattern("fetch_api:*/performance*")
        
        # Invalidate positions and portfolio (execution affects these)
        if broker:
            frontend_cache.delete_pattern(f"fetch_api:*/positions/{broker}*")
            frontend_cache.delete_pattern(f"positions:{user_id}:{broker}*")
        
        frontend_cache.delete_pattern("fetch_api:*/portfolio*")
        
        logger.info(f"Invalidated caches for execution completion (strategy: {strategy_id}, broker: {broker})")
    
    def _invalidate_portfolio_rebalanced(self, context: dict):
        """Invalidate caches when portfolio is rebalanced"""
        user_id = context.get('user_id')
        broker = context.get('broker')
        
        # Invalidate all portfolio-related data
        frontend_cache.delete_pattern("fetch_api:*/portfolio*")
        frontend_cache.delete_pattern("fetch_api:*/analytics*")
        
        if broker:
            frontend_cache.delete_pattern(f"fetch_api:*/positions/{broker}*")
            frontend_cache.delete_pattern(f"positions:{user_id}:{broker}*")
        
        # Invalidate risk metrics
        if user_id:
            frontend_cache.delete_pattern(f"risk_metrics:{user_id}*")
        
        logger.info(f"Invalidated caches for portfolio rebalancing (user: {user_id}, broker: {broker})")
    
    def _invalidate_risk_limits_changed(self, context: dict):
        """Invalidate caches when risk limits are changed"""
        user_id = context.get('user_id')
        
        # Invalidate all risk-related caches
        frontend_cache.delete_pattern("risk_metrics*")
        frontend_cache.delete_pattern("fetch_api:*/risk*")
        
        if user_id:
            frontend_cache.delete_pattern(f"risk_metrics:{user_id}*")
            frontend_cache.delete_pattern(f"fetch_api:*/preferences/{user_id}*")
    
    def _invalidate_broker_changed(self, context: dict):
        """Invalidate caches when broker is changed"""
        new_broker = context.get('new_broker')
        
        # Invalidate broker-specific caches
        frontend_cache.delete_pattern("fetch_api:*/profile*")
        frontend_cache.delete_pattern("fetch_api:*/positions*")
        frontend_cache.delete_pattern("fetch_api:*/orders*")
        frontend_cache.delete_pattern("fetch_api:*/portfolio*")
        frontend_cache.delete_pattern("fetch_api:*/margins*")
        
        # Clear trading data cache for the new broker
        if new_broker:
            TradingDataCache.invalidate_broker_data(new_broker)
        
        logger.info(f"Cache invalidated for broker change to: {new_broker}")


# Global cache invalidation manager
cache_invalidation_manager = CacheInvalidationManager()

# Convenience functions for common invalidation scenarios
def invalidate_on_order_placed(broker: str, user_id: str, order_data: dict = None):
    """Invalidate caches when an order is placed"""
    context = {
        'broker': broker,
        'user_id': user_id,
        'symbol': order_data.get('trading_symbol') if order_data else None
    }
    cache_invalidation_manager.invalidate('order_placed', context)

def invalidate_on_strategy_action(action: str, broker: str, strategy_id: str = None):
    """Invalidate caches for strategy start/stop actions"""
    context = {
        'broker': broker,
        'strategy_id': strategy_id
    }
    if action in ['start', 'started']:
        cache_invalidation_manager.invalidate('strategy_started', context)
    elif action in ['stop', 'stopped']:
        cache_invalidation_manager.invalidate('strategy_stopped', context)

def invalidate_on_position_change(broker: str, user_id: str, symbol: str = None):
    """Invalidate caches when positions change"""
    context = {
        'broker': broker,
        'user_id': user_id,
        'symbol': symbol
    }
    cache_invalidation_manager.invalidate('position_updated', context)

def invalidate_on_settings_change(user_id: str, setting_type: str = None):
    """Invalidate caches when user settings change"""
    context = {
        'user_id': user_id,
        'setting_type': setting_type
    }
    cache_invalidation_manager.invalidate('user_settings_changed', context)

def invalidate_on_market_update(symbols: List[str] = None):
    """Invalidate caches when market data is updated"""
    context = {
        'symbols': symbols or []
    }
    cache_invalidation_manager.invalidate('market_data_updated', context)

def invalidate_on_broker_change(new_broker: str):
    """Invalidate caches when broker is changed"""
    context = {
        'new_broker': new_broker
    }
    cache_invalidation_manager.invalidate('broker_changed', context)

def emergency_cache_clear():
    """Emergency function to clear all caches"""
    try:
        frontend_cache.clear()
        logger.warning("Emergency cache clear executed - all caches cleared")
    except Exception as e:
        logger.error(f"Emergency cache clear failed: {e}")