"""
Market Hours Manager - Handles automatic strategy stop and position exit
"""

import asyncio
from datetime import datetime, time, timedelta
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.sql import text
from .models import StrategyExecution, Order
from .database import db_manager

logger = logging.getLogger(__name__)

class MarketHoursManager:
    def __init__(self, strategy_engine=None):
        self.strategy_engine = strategy_engine
        self.market_close_time = time(15, 30)  # 3:30 PM
        self.position_exit_time = time(15, 10)  # 3:10 PM
        self.is_running = False
        self.monitor_task = None
        self._last_holiday_check = None
        self._cached_holidays = set()
        
    async def start_monitoring(self):
        """Start monitoring market hours"""
        if self.is_running:
            return
            
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_market_hours())
        logger.info("Market hours monitoring started")
        
    async def stop_monitoring(self):
        """Stop monitoring market hours"""
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Market hours monitoring stopped")
        
    async def _monitor_market_hours(self):
        """Monitor market hours and take actions at specific times"""
        # Refresh holiday cache on startup
        await self._check_if_holiday(datetime.now().date())
        
        while self.is_running:
            try:
                now = datetime.now()
                current_time = now.time()
                current_day = now.weekday()  # Monday = 0, Sunday = 6
                today_date = now.date()
                
                # Refresh holiday cache if needed (once per day)
                await self._check_if_holiday(today_date)
                
                # Only monitor on weekdays (Monday to Friday) that are not holidays
                is_trading_day = current_day < 5 and today_date not in self._cached_holidays
                
                if is_trading_day:
                    # Check if it's market close time (3:30 PM)
                    if (current_time >= self.market_close_time and 
                        current_time < self.position_exit_time):
                        logger.info("Market close time reached, stopping all strategies")
                        await self._stop_all_strategies()
                    
                    # Check if it's position exit time (3:10 PM)
                    elif current_time >= self.position_exit_time:
                        logger.info("Position exit time reached, exiting all positions")
                        await self._exit_all_positions()
                else:
                    if current_day >= 5:
                        logger.debug("Weekend - skipping market hours monitoring")
                    elif today_date in self._cached_holidays:
                        logger.debug("Market holiday - skipping market hours monitoring")
                
                # Check every 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in market hours monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _stop_all_strategies(self):
        """Stop all running strategies at market close (3:30 PM)"""
        try:
            session_factory = db_manager.get_session_factory('trading_db')
            if not session_factory:
                logger.error("Database session factory not available")
                return
                
            async with session_factory() as db:
                result = await db.execute(
                    select(StrategyExecution).filter(
                        StrategyExecution.status == "running"
                    )
                )
                active_executions = result.scalars().all()
                
                stopped_count = 0
                for execution in active_executions:
                    try:
                        execution.status = "stopped"
                        execution.stopped_at = datetime.now()
                        stopped_count += 1
                        
                        # Stop monitoring task if exists
                        if self.strategy_engine:
                            task_key = f"{execution.execution_id}_{execution.user_id}"
                            if task_key in self.strategy_engine.active_strategies:
                                task = self.strategy_engine.active_strategies.pop(task_key)
                                task.cancel()
                        
                        logger.info(f"Stopped strategy execution {execution.execution_id} for {execution.trading_symbol} at market close")
                        
                    except Exception as e:
                        logger.error(f"Error stopping execution {execution.execution_id}: {str(e)}")
                        
                await db.commit()
                
                if stopped_count > 0:
                    logger.info(f"Market close: Stopped {stopped_count} strategy executions at {datetime.now()}")
                    
        except Exception as e:
            logger.error(f"Error stopping strategies at market close: {str(e)}")
            
    async def _exit_all_positions(self):
        """Exit all open positions by 3:10 PM"""
        try:
            session_factory = db_manager.get_session_factory('trading_db')
            if not session_factory:
                logger.error("Database session factory not available")
                return
                
            async with session_factory() as db:
                # Find all open positions (buy orders that haven't been sold)
                result = await db.execute(
                    select(Order).filter(
                        and_(
                            Order.status == "executed",
                            Order.transaction_type == "BUY",
                            Order.exit_order_id.is_(None)  # No exit order placed yet
                        )
                    )
                )
                open_positions = result.scalars().all()
                
                exit_count = 0
                for position in open_positions:
                    try:
                        # Create market exit order
                        await self._place_market_exit_order(position, db)
                        exit_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error exiting position {position.order_id}: {str(e)}")
                        
                if exit_count > 0:
                    logger.info(f"Position exit: Placed {exit_count} market exit orders at {datetime.now()}")
                    
        except Exception as e:
            logger.error(f"Error exiting positions at 3:10 PM: {str(e)}")
            
    async def _place_market_exit_order(self, position: Order, db: AsyncSession):
        """Place a market exit order for an open position"""
        try:
            # Create exit order
            exit_order = Order(
                user_id=position.user_id,
                broker=position.broker,
                instrument_token=position.instrument_token,
                trading_symbol=position.trading_symbol,
                quantity=position.quantity,
                order_type="MARKET",
                transaction_type="SELL",
                price=0.0,  # Market price
                trigger_price=None,
                status="pending",
                order_tag=f"AUTO_EXIT_{position.order_id}",
                parent_order_id=position.order_id
            )
            
            db.add(exit_order)
            
            # Update original position to link exit order
            position.exit_order_id = exit_order.order_id
            
            await db.commit()
            await db.refresh(exit_order)
            
            logger.info(f"Created market exit order {exit_order.order_id} for position {position.order_id} ({position.trading_symbol})")
            
            # TODO: Execute the order through broker API
            # This would require access to the user's broker API
            
        except Exception as e:
            logger.error(f"Error creating exit order for position {position.order_id}: {str(e)}")
            raise
            
    async def _check_if_holiday(self, check_date) -> bool:
        """
        Check if a given date is a market holiday using NSE_HOLIDAYS table.
        Caches results for performance.
        """
        # Refresh cache once per day
        now = datetime.now()
        if self._last_holiday_check is None or (now.date() != self._last_holiday_check):
            logger.info("Refreshing holiday cache from NSE_HOLIDAYS table")
            self._last_holiday_check = now.date()
            self._cached_holidays = set()
            
            try:
                # Get NSE database session
                nse_session_factory = db_manager.get_session_factory('nse_db')
                if nse_session_factory:
                    async with nse_session_factory() as nse_db:
                        # Fetch holidays for current month and next month
                        start_date = now.replace(day=1).date()
                        end_date = (now + timedelta(days=60)).date()
                        
                        result = await nse_db.execute(
                            text("""
                                SELECT DATE(trading_date) FROM "NSE_HOLIDAYS" 
                                WHERE DATE(trading_date) BETWEEN :start_date AND :end_date
                            """),
                            {"start_date": start_date, "end_date": end_date}
                        )
                        
                        holidays = result.fetchall()
                        self._cached_holidays = {row[0] for row in holidays}
                        logger.info(f"Loaded {len(self._cached_holidays)} holidays from database")
            except Exception as e:
                logger.warning(f"Could not load holidays from database: {e}")
        
        return check_date in self._cached_holidays
    
    def is_market_hours(self) -> bool:
        """
        Check if current time is within market hours.
        Enhanced to check:
        1. Weekday (Monday-Friday)
        2. Market holidays from NSE_HOLIDAYS table (with caching)
        3. Market hours (9:15 AM to 3:30 PM)
        
        Note: Holiday checking is done synchronously using cached data to avoid
        async issues in sync contexts. Cache is refreshed periodically.
        """
        now = datetime.now()
        today_date = now.date()
        current_time = now.time()
        current_day = now.weekday()
        
        # Check if it's a weekday (Monday=0, Friday=4)
        if current_day >= 5:  # Saturday or Sunday
            logger.debug("Market closed: Weekend")
            return False
        
        # Check if today is a holiday (using cached data)
        # Note: This uses cached holiday data to avoid async calls
        if today_date in self._cached_holidays:
            logger.debug("Market closed: Holiday")
            return False
        
        # Check if within market hours (9:15 AM to 3:30 PM)
        market_start = time(9, 15)
        market_end = time(15, 30)
        
        is_open = market_start <= current_time <= market_end
        
        if not is_open:
            logger.debug(f"Market closed: Outside trading hours (current: {current_time})")
        
        return is_open
        
    def should_allow_trading(self) -> bool:
        """Check if trading should be allowed (before 3:00 PM)"""
        current_time = datetime.now().time()
        current_day = datetime.now().weekday()
        
        if current_day < 5:  # Weekday
            return current_time < self.market_close_time
        return False