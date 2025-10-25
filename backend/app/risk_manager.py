"""
Risk Management System for Algorithmic Trading
Handles position sizing, loss limits, and trading restrictions
"""

import logging
from datetime import datetime, date, timedelta, time
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from .models import User, Order, Position, Strategy
from .database import get_db

logger = logging.getLogger(__name__)

class RiskViolation(Exception):
    """Exception raised when a trade violates risk management rules"""
    pass

class RiskManager:
    def __init__(self):
        self.logger = logger
        self.emergency_stop_active = False
        self.circuit_breaker_triggered = False
        
    async def get_user_risk_settings(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Fetch user risk management settings from database"""
        try:
            result = await db.execute(
                select(User).filter(User.user_id == user_id)
            )
            user = result.scalars().first()
            
            if not user:
                # Default risk settings if user not found
                return self._get_default_risk_settings()
            
            # Extract risk settings from user preferences or use defaults
            preferences = user.preferences or {}

            raw_position_limit = preferences.get('position_size_limit')
            raw_portfolio_limit = preferences.get('portfolio_size_limit', raw_position_limit)

            if raw_position_limit is None and raw_portfolio_limit is None:
                raw_position_limit = raw_portfolio_limit = 50000.0

            if raw_position_limit is None:
                raw_position_limit = raw_portfolio_limit
            if raw_portfolio_limit is None:
                raw_portfolio_limit = raw_position_limit

            try:
                position_limit = float(raw_position_limit)
            except (TypeError, ValueError):
                position_limit = 50000.0

            try:
                portfolio_limit = float(raw_portfolio_limit)
            except (TypeError, ValueError):
                portfolio_limit = position_limit

            # Ensure both limits stay in sync if one ends up invalid
            if portfolio_limit < position_limit:
                portfolio_limit = position_limit

            return {
                'daily_loss_limit': preferences.get('daily_loss_limit', 10000),
                'position_size_limit': position_limit,
                'portfolio_size_limit': portfolio_limit,
                'auto_stop_trading': preferences.get('auto_stop_trading', True),
                'max_open_positions': preferences.get('max_open_positions', 10),
                'risk_per_trade': preferences.get('risk_per_trade', 2.0),
                'max_portfolio_risk': preferences.get('max_portfolio_risk', 20.0),
            }
        except Exception as e:
            self.logger.error(f"Error fetching risk settings for user {user_id}: {e}")
            return self._get_default_risk_settings()
    
    def _get_default_risk_settings(self) -> Dict[str, Any]:
        """Default risk management settings"""
        return {
            'daily_loss_limit': 10000,
            'position_size_limit': 50000,
            'portfolio_size_limit': 50000,
            'auto_stop_trading': True,
            'max_open_positions': 10,
            'risk_per_trade': 2.0,
            'max_portfolio_risk': 20.0,
            'max_orders_per_minute': 10
        }
    
    async def get_daily_pnl(self, user_id: str, db: AsyncSession) -> float:
        """Calculate user's profit/loss for the current trading day"""
        try:
            today = date.today()
            
            # Get all orders executed today
            result = await db.execute(
                select(Order).filter(
                    and_(
                        Order.user_id == user_id,
                        func.date(Order.order_timestamp) == today,
                        Order.status.in_(['COMPLETE', 'FILLED'])
                    )
                )
            )
            orders = result.scalars().all()
            
            total_pnl = 0.0
            for order in orders:
                if order.transaction_type == 'SELL':
                    total_pnl += (order.filled_quantity or 0) * (order.average_price or order.price or 0)
                else:  # BUY
                    total_pnl -= (order.filled_quantity or 0) * (order.average_price or order.price or 0)
            
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating daily P&L for user {user_id}: {e}")
            return 0.0
    
    async def get_current_positions(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Get current position metrics"""
        try:
            result = await db.execute(
                select(Position).filter(
                    and_(
                        Position.user_id == user_id,
                        Position.quantity != 0
                    )
                )
            )
            positions = result.scalars().all()
            
            total_exposure = 0.0
            position_count = len(positions)
            
            for position in positions:
                total_exposure += abs(position.quantity * (position.average_price or 0))
            
            return {
                'total_exposure': total_exposure,
                'position_count': position_count,
                'positions': positions
            }
            
        except Exception as e:
            self.logger.error(f"Error getting positions for user {user_id}: {e}")
            return {'total_exposure': 0.0, 'position_count': 0, 'positions': []}
    
    async def validate_trade(self, user_id: str, instrument_token: str, quantity: int, 
                           price: float, transaction_type: str, db: AsyncSession) -> Tuple[bool, str]:
        """
        Validate if a trade can be executed based on risk management rules with race condition protection
        Returns (is_valid, reason)
        """
        try:
            # CRITICAL SECURITY FIX: Use row-level locking to prevent race conditions
            # Lock user row to prevent concurrent modifications
            result = await db.execute(
                select(User).filter(User.user_id == user_id).with_for_update()
            )
            user = result.scalars().first()
            
            if not user:
                return False, "User not found"
            
            risk_settings = await self.get_user_risk_settings(user_id, db)
            
            # Check if trading is stopped due to daily loss limit
            daily_pnl = await self.get_daily_pnl(user_id, db)
            if daily_pnl <= -risk_settings['daily_loss_limit']:
                if risk_settings['auto_stop_trading']:
                    return False, f"Trading stopped: Daily loss limit of INR {risk_settings['daily_loss_limit']:,.2f} exceeded"
            
            position_limit = float(risk_settings.get('position_size_limit', 0.0) or 0.0)
            portfolio_capital = float(risk_settings.get('portfolio_size_limit', position_limit) or position_limit)

            # Check position size limit
            trade_value = abs(quantity * price)
            if position_limit > 0 and trade_value > position_limit:
                return False, f"Trade rejected: Position size INR {trade_value:,.2f} exceeds limit of INR {position_limit:,.2f}"
            
            # Check maximum open positions with current data (race condition safe)
            position_info = await self.get_current_positions(user_id, db)
            if position_info['position_count'] >= risk_settings['max_open_positions']:
                return False, f"Trade rejected: Maximum {risk_settings['max_open_positions']} open positions reached"
            
            # Check portfolio risk exposure - critical to prevent over-exposure
            current_exposure = position_info['total_exposure']
            
            # Get pending orders to include in exposure calculation
            pending_orders_result = await db.execute(
                select(Order).filter(
                    and_(
                        Order.user_id == user_id,
                        Order.status.in_(['PENDING', 'OPEN', 'TRIGGER_PENDING'])
                    )
                )
            )
            pending_orders = pending_orders_result.scalars().all()
            
            # Calculate pending exposure
            pending_exposure = 0.0
            for order in pending_orders:
                if order.quantity and order.price:
                    pending_exposure += abs(order.quantity * order.price)
                elif order.quantity and order.trigger_price:
                    pending_exposure += abs(order.quantity * order.trigger_price)
            
            # Total exposure including this new trade
            new_total_exposure = current_exposure + pending_exposure + trade_value
            portfolio_limit = portfolio_capital * (risk_settings['max_portfolio_risk'] / 100)

            if portfolio_limit > 0 and new_total_exposure > portfolio_limit:
                return False, f"Trade rejected: Total exposure (including pending orders) INR {new_total_exposure:,.2f} would exceed limit of INR {portfolio_limit:,.2f}"
            
            # Check if user has reached trade frequency limits (prevent spam trading)
            recent_orders_result = await db.execute(
                select(func.count(Order.order_id)).filter(
                    and_(
                        Order.user_id == user_id,
                        Order.order_timestamp >= datetime.now() - timedelta(minutes=1)
                    )
                )
            )
            recent_orders_count = recent_orders_result.scalar()
            
            # Prevent more than 10 orders per minute (configurable)
            max_orders_per_minute = risk_settings.get('max_orders_per_minute', 10)
            if recent_orders_count >= max_orders_per_minute:
                return False, f"Trade rejected: Maximum {max_orders_per_minute} orders per minute exceeded"
            
            self.logger.info(f"Trade validation passed for user {user_id}: INR {trade_value:,.2f}, total exposure: INR {new_total_exposure:,.2f}")
            return True, "Trade approved"
            
        except Exception as e:
            self.logger.error(f"Error validating trade for user {user_id}: {e}")
            return False, f"Risk validation error: {str(e)}"
    
    async def calculate_position_size(self, user_id: str, risk_per_trade: float, 
                                    entry_price: float, stop_loss: float, 
                                    db: AsyncSession) -> int:
        """
        Calculate optimal position size based on risk per trade
        """
        try:
            risk_settings = await self.get_user_risk_settings(user_id, db)
            
            # Get available capital (assume from portfolio limits for now)
            available_capital = float(risk_settings.get('portfolio_size_limit', risk_settings.get('position_size_limit', 0)) or 0)
            
            # Calculate risk amount
            risk_amount = available_capital * (risk_per_trade / 100)
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                return 1  # Minimum quantity
            
            # Calculate position size
            position_size = int(risk_amount / risk_per_share)
            
            # Ensure minimum quantity
            position_size = max(1, position_size)
            
            # Ensure doesn't exceed position limit
            if entry_price > 0:
                max_position_value = risk_settings.get('position_size_limit', 0)
                max_quantity = int(max_position_value / entry_price) if max_position_value > 0 else position_size
                if max_quantity > 0:
                    position_size = min(position_size, max_quantity)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for user {user_id}: {e}")
            return 1
    
    async def check_trading_allowed(self, user_id: str, db: AsyncSession) -> Tuple[bool, str]:
        """Check if trading is allowed for the user"""
        try:
            risk_settings = await self.get_user_risk_settings(user_id, db)
            
            if not risk_settings['auto_stop_trading']:
                return True, "Trading allowed"
            
            # Check daily loss limit
            daily_pnl = await self.get_daily_pnl(user_id, db)
            if daily_pnl <= -risk_settings['daily_loss_limit']:
                return False, "Trading suspended due to daily loss limit"
            
            return True, "Trading allowed"
            
        except Exception as e:
            self.logger.error(f"Error checking trading status for user {user_id}: {e}")
            return False, f"Error checking trading status: {str(e)}"
    
    async def get_risk_metrics(self, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive risk metrics for dashboard display"""
        try:
            risk_settings = await self.get_user_risk_settings(user_id, db)
            daily_pnl = await self.get_daily_pnl(user_id, db)
            position_info = await self.get_current_positions(user_id, db)
            trading_allowed, trading_status = await self.check_trading_allowed(user_id, db)
            
            return {
                'daily_pnl': daily_pnl,
                'daily_loss_limit': risk_settings['daily_loss_limit'],
                'daily_loss_used_pct': abs(daily_pnl / risk_settings['daily_loss_limit']) * 100 if daily_pnl < 0 else 0,
                'total_exposure': position_info['total_exposure'],
                'position_size_limit': risk_settings['position_size_limit'],
                'portfolio_size_limit': risk_settings['portfolio_size_limit'],
                'exposure_used_pct': (
                    (position_info['total_exposure'] / risk_settings['portfolio_size_limit']) * 100
                    if risk_settings['portfolio_size_limit'] else 0
                ),
                'open_positions': position_info['position_count'],
                'max_open_positions': risk_settings['max_open_positions'],
                'trading_allowed': trading_allowed,
                'trading_status': trading_status,
                'risk_settings': risk_settings
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics for user {user_id}: {e}")
            return {}
    
    def is_market_hours(self) -> Tuple[bool, str]:
        """
        Check if current time is within market trading hours (Indian market)
        Returns: (is_open, reason)
        """
        try:
            now = datetime.now()
            current_time = now.time()
            current_weekday = now.weekday()  # 0=Monday, 6=Sunday
            
            # Indian market is closed on weekends
            if current_weekday >= 5:  # Saturday (5) or Sunday (6)
                return False, "Market closed: Weekend"
            
            # Indian market hours: 9:15 AM to 3:30 PM IST
            market_open = time(9, 15)  # 9:15 AM
            market_close = time(15, 30)  # 3:30 PM
            
            # Pre-market session: 9:00 AM to 9:15 AM (limited trading)
            pre_market_open = time(9, 0)
            
            # After-market orders (AMO): After 3:30 PM until 9:00 AM next day
            if current_time >= market_open and current_time <= market_close:
                return True, "Regular market hours"
            elif current_time >= pre_market_open and current_time < market_open:
                return True, "Pre-market session"
            else:
                return False, "Market closed: Outside trading hours"
                
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            return False, f"Market hours check failed: {str(e)}"
    
    def activate_emergency_stop(self, reason: str = "Manual activation") -> bool:
        """
        Activate emergency stop - halts all trading immediately
        """
        try:
            self.emergency_stop_active = True
            self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
            # In production, you might want to:
            # 1. Send alerts to all administrators
            # 2. Cancel all pending orders
            # 3. Log to audit trail
            # 4. Notify users
            return True
        except Exception as e:
            self.logger.error(f"Error activating emergency stop: {e}")
            return False
    
    def deactivate_emergency_stop(self, authorized_by: str = "System") -> bool:
        """
        Deactivate emergency stop - requires authorization
        """
        try:
            self.emergency_stop_active = False
            self.logger.warning(f"Emergency stop deactivated by: {authorized_by}")
            return True
        except Exception as e:
            self.logger.error(f"Error deactivating emergency stop: {e}")
            return False
    
    async def check_system_health(self, db: AsyncSession) -> Tuple[bool, str]:
        """
        Check overall system health and trigger circuit breaker if needed
        """
        try:
            # Check database connectivity
            try:
                await db.execute(select(1))
            except Exception as db_error:
                self.circuit_breaker_triggered = True
                return False, f"Database connectivity failed: {str(db_error)}"
            
            # Check for excessive failures in recent orders
            recent_time = datetime.now() - timedelta(minutes=10)
            failed_orders_result = await db.execute(
                select(func.count(Order.order_id)).filter(
                    and_(
                        Order.status.in_(['REJECTED', 'CANCELLED']),
                        Order.order_timestamp >= recent_time
                    )
                )
            )
            failed_orders = failed_orders_result.scalar()
            
            total_orders_result = await db.execute(
                select(func.count(Order.order_id)).filter(
                    Order.order_timestamp >= recent_time
                )
            )
            total_orders = total_orders_result.scalar()
            
            # If more than 50% of recent orders failed, trigger circuit breaker
            if total_orders > 10 and failed_orders / total_orders > 0.5:
                self.circuit_breaker_triggered = True
                return False, f"High failure rate: {failed_orders}/{total_orders} orders failed in last 10 minutes"
            
            # Check for system overload (too many orders per minute)
            recent_orders_result = await db.execute(
                select(func.count(Order.order_id)).filter(
                    Order.order_timestamp >= datetime.now() - timedelta(minutes=1)
                )
            )
            recent_orders = recent_orders_result.scalar()
            
            # If more than 100 orders per minute system-wide, trigger circuit breaker
            if recent_orders > 100:
                self.circuit_breaker_triggered = True
                return False, f"System overload: {recent_orders} orders in last minute"
            
            self.circuit_breaker_triggered = False
            return True, "System health OK"
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            self.circuit_breaker_triggered = True
            return False, f"Health check failed: {str(e)}"
    
    async def comprehensive_trading_check(self, user_id: str, db: AsyncSession) -> Tuple[bool, str]:
        """
        Comprehensive check before allowing any trading activity
        """
        try:
            # 1. Check emergency stop
            if self.emergency_stop_active:
                return False, "Emergency stop is active - all trading suspended"
            
            # 2. Check circuit breaker
            system_healthy, health_reason = await self.check_system_health(db)
            if not system_healthy:
                return False, f"Circuit breaker triggered: {health_reason}"
            
            # 3. Check market hours
            market_open, market_reason = self.is_market_hours()
            if not market_open:
                # Allow AMO orders outside market hours
                return True, f"AMO allowed: {market_reason}"
            
            # 4. Check user-specific trading status
            user_trading_allowed, user_reason = await self.check_trading_allowed(user_id, db)
            if not user_trading_allowed:
                return False, f"User trading suspended: {user_reason}"
            
            return True, "All checks passed"
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive trading check: {e}")
            return False, f"Trading check failed: {str(e)}"
