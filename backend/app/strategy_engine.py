"""
Strategy Execution Engine for Real-time Algorithmic Trading
Handles strategy monitoring, condition evaluation, and trade execution
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, time as date_time
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from fastapi import HTTPException

from .models import Strategy, User, Order, Position, StrategyExecution
from .risk_manager import RiskManager, RiskViolation
from .database import get_db
from .services import get_historical_data, place_order
from common_utils.technical_indicators import moving_average, exponential_moving_average, relative_strength_index, macd, bollinger_bands

logger = logging.getLogger(__name__)

class StrategyExecutionEngine:
    def __init__(self):
        self.active_strategies: Dict[str, asyncio.Task] = {}
        self.risk_manager = RiskManager()
        self.logger = logger
        self.strategy_metrics: Dict[str, Dict] = {}
        
    async def execute_strategy(self, strategy_id: str, user_id: str, instrument_token: str, 
                             trading_symbol: str, quantity: int, risk_per_trade: float,
                             stop_loss: float = None, take_profit: float = None,
                             position_sizing_percent: float = None, position_sizing_mode: str = None,
                             total_capital: float = None, timeframe: str = None,
                             trailing_stop_enabled: bool = False, trailing_stop_percent: float = None,
                             trailing_stop_min: float = None, partial_exits: list = None,
                             db: AsyncSession = None, api=None) -> Dict[str, Any]:
        """Execute a strategy on a specific instrument"""
        try:
            # Check market hours - don't allow new executions after 3:00 PM
            from datetime import datetime, time
            current_time = datetime.now().time()
            market_close_time = time(22, 0)  # 3:00 PM
            
            if current_time >= market_close_time:
                return {
                    "success": False,
                    "message": "Strategy execution not allowed after 3:00 PM market close"
                }
            
            # Load strategy from database
            result = await db.execute(
                select(Strategy).filter(
                    and_(
                        Strategy.strategy_id == strategy_id,
                        Strategy.user_id == user_id
                    )
                )
            )
            strategy = result.scalars().first()
            
            if not strategy:
                return {
                    "success": False,
                    "message": "Strategy not found"
                }
            
            # Check if user trading is allowed
            trading_allowed, reason = await self.risk_manager.check_trading_allowed(user_id, db)
            if not trading_allowed:
                return {
                    "success": False,
                    "message": f"Trading not allowed: {reason}"
                }
            
            # Validate trade parameters
            is_valid, validation_msg = await self.risk_manager.validate_trade(
                user_id, instrument_token, quantity, 100.0, 'BUY', db  # Placeholder price for validation
            )
            
            if not is_valid:
                return {
                    "success": False,
                    "message": validation_msg
                }
            
            # Check for existing active execution for this strategy and instrument
            existing_result = await db.execute(
                select(StrategyExecution).filter(
                    and_(
                        StrategyExecution.strategy_id == strategy_id,
                        StrategyExecution.user_id == user_id,
                        StrategyExecution.trading_symbol == trading_symbol,
                        StrategyExecution.status == "running"
                    )
                )
            )
            execution = existing_result.scalars().first()
            
            if execution:
                # Update existing execution with new parameters
                execution.quantity = quantity
                execution.risk_per_trade = risk_per_trade
                execution.stop_loss = stop_loss
                execution.take_profit = take_profit
                execution.execution_config.update({
                    "strategy_name": strategy.name,
                    "entry_conditions": strategy.entry_conditions,
                    "exit_conditions": strategy.exit_conditions,
                    "parameters": strategy.parameters,
                    # Add execution-specific parameters
                    "execution_params": {
                        "position_sizing_percent": position_sizing_percent,
                        "position_sizing_mode": position_sizing_mode,
                        "total_capital": total_capital,
                        "timeframe": timeframe,
                        "trailing_stop_enabled": trailing_stop_enabled,
                        "trailing_stop_percent": trailing_stop_percent,
                        "trailing_stop_min": trailing_stop_min,
                        "partial_exits": partial_exits or []
                    }
                })
                self.logger.info(f"Updated existing execution {execution.execution_id} for {trading_symbol}")
            else:
                # Create new strategy execution record
                execution = StrategyExecution(
                    strategy_id=strategy_id,
                    user_id=user_id,
                    broker=strategy.broker,
                    instrument_token=instrument_token,
                    trading_symbol=trading_symbol,
                    quantity=quantity,
                    risk_per_trade=risk_per_trade,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    status="running",
                    execution_config={
                        "strategy_name": strategy.name,
                        "entry_conditions": strategy.entry_conditions,
                        "exit_conditions": strategy.exit_conditions,
                        "parameters": strategy.parameters,
                        # Add execution-specific parameters
                        "execution_params": {
                            "position_sizing_percent": position_sizing_percent,
                            "position_sizing_mode": position_sizing_mode,
                            "total_capital": total_capital,
                            "timeframe": timeframe,
                            "trailing_stop_enabled": trailing_stop_enabled,
                            "trailing_stop_percent": trailing_stop_percent,
                            "trailing_stop_min": trailing_stop_min,
                            "partial_exits": partial_exits or []
                        }
                    }
                )
                
                db.add(execution)
                self.logger.info(f"Created new execution record for {trading_symbol}")
            
            # Update strategy status to active when execution starts
            strategy.status = "active"
            strategy.updated_at = datetime.now()
            
            await db.commit()
            await db.refresh(execution)
            
            # Start monitoring task
            task_key = f"{execution.execution_id}_{user_id}"
            
            if task_key not in self.active_strategies:
                task = asyncio.create_task(
                    self._monitor_strategy_execution(execution, db, api)
                )
                self.active_strategies[task_key] = task
                
                # Initialize metrics
                self.strategy_metrics[task_key] = {
                    'execution_id': execution.execution_id,
                    'strategy_id': strategy_id,
                    'trading_symbol': trading_symbol,
                    'started_at': datetime.now(),
                    'signals_generated': 0,
                    'trades_executed': 0,
                    'last_check': None,
                    'current_position': 0,
                    'pnl': 0.0,
                    'status': 'monitoring'
                }
            
            self.logger.info(f"Strategy execution started: {execution.execution_id} for {trading_symbol}")
            
            return {
                "success": True,
                "message": f"Strategy '{strategy.name}' execution started on {trading_symbol}",
                "execution_id": execution.execution_id
            }
            
        except Exception as e:
            self.logger.error(f"Error executing strategy {strategy_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to execute strategy: {str(e)}"
            }

    async def activate_strategy(self, strategy_id: str, user_id: str, db: AsyncSession, api=None) -> Dict[str, Any]:
        """Activate a strategy template - marks it as ready for execution"""
        try:
            self.logger.info(f"Activating strategy template {strategy_id} for user {user_id}")
            
            # Find the strategy
            result = await db.execute(
                select(Strategy).filter(
                    and_(
                        Strategy.strategy_id == strategy_id,
                        Strategy.user_id == user_id
                    )
                )
            )
            strategy = result.scalars().first()
            
            if not strategy:
                return {
                    "success": False,
                    "message": "Strategy not found or access denied"
                }
            
            # Validate strategy has required conditions
            if not strategy.entry_conditions or not strategy.exit_conditions:
                return {
                    "success": False, 
                    "message": "Strategy must have both entry and exit conditions to be activated"
                }
            
            # Update strategy status
            strategy.status = "active"
            strategy.updated_at = datetime.now()
            await db.commit()
            
            self.logger.info(f"Strategy {strategy_id} activated successfully as template")
            
            return {
                "success": True,
                "message": "Strategy template activated successfully. You can now execute it on specific instruments.",
                "strategy_id": strategy_id,
                "status": "active"
            }
            
        except Exception as e:
            await db.rollback()
            self.logger.error(f"Error activating strategy {strategy_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to activate strategy: {str(e)}"
            }
    
    async def stop_strategy_execution(self, execution_id: str, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Stop a running strategy execution"""
        try:
            # Find and update the execution record
            result = await db.execute(
                select(StrategyExecution).filter(
                    and_(
                        StrategyExecution.execution_id == execution_id,
                        StrategyExecution.user_id == user_id,
                        StrategyExecution.status == "running"
                    )
                )
            )
            execution = result.scalars().first()
            
            if not execution:
                return {
                    "success": False,
                    "message": "Running execution not found"
                }
            
            # Stop the monitoring task
            task_key = f"{execution_id}_{user_id}"
            if task_key in self.active_strategies:
                task = self.active_strategies[task_key]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                # Clean up metrics
                if task_key in self.strategy_metrics:
                    del self.strategy_metrics[task_key]
                
                del self.active_strategies[task_key]
            
            # Update execution record
            execution.status = "stopped"
            execution.stopped_at = datetime.now()
            
            # Check if there are any other running executions for this strategy
            other_executions_result = await db.execute(
                select(StrategyExecution).filter(
                    and_(
                        StrategyExecution.strategy_id == execution.strategy_id,
                        StrategyExecution.user_id == user_id,
                        StrategyExecution.status == "running",
                        StrategyExecution.execution_id != execution_id
                    )
                )
            )
            other_running_executions = other_executions_result.scalars().all()
            
            # If no other executions are running, set strategy status to inactive
            if not other_running_executions:
                strategy_result = await db.execute(
                    select(Strategy).filter(Strategy.strategy_id == execution.strategy_id)
                )
                strategy = strategy_result.scalars().first()
                if strategy:
                    strategy.status = "inactive"
                    strategy.updated_at = datetime.now()
                    self.logger.info(f"Strategy {execution.strategy_id} status updated to inactive - no more running executions")
            
            await db.commit()
            
            self.logger.info(f"Strategy execution stopped: {execution_id}")
            
            return {
                "success": True,
                "message": f"Strategy execution stopped for {execution.trading_symbol}",
                "execution_id": execution_id
            }
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy execution {execution_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to stop execution: {str(e)}"
            }
    
    async def _monitor_strategy_execution(self, execution: StrategyExecution, db: AsyncSession, api=None):
        """Monitor a single strategy execution"""
        task_key = f"{execution.execution_id}_{execution.user_id}"
        
        self.logger.info(f"Started monitoring execution {execution.execution_id} for {execution.trading_symbol}")
        
        try:
            while True:
                try:
                    # Check market hours - stop monitoring after 3:10 PM
                    from datetime import time
                    current_time = datetime.now().time()
                    position_exit_time = time(15, 10)  # 3:10 PM
                    
                    if current_time >= position_exit_time:
                        self.logger.info(f"Stopping monitoring for {execution.trading_symbol} - market closed at {current_time}")
                        # Update execution status to stopped
                        current_execution_result = await db.execute(
                            select(StrategyExecution).filter(StrategyExecution.execution_id == execution.execution_id)
                        )
                        current_exec = current_execution_result.scalars().first()
                        if current_exec and current_exec.status == "running":
                            current_exec.status = "stopped"
                            current_exec.stopped_at = datetime.now()
                            await db.commit()
                        break
                    
                    # Update last check time
                    if task_key in self.strategy_metrics:
                        self.strategy_metrics[task_key]['last_check'] = datetime.now()
                    
                    # Check if execution is still running
                    result = await db.execute(
                        select(StrategyExecution).filter(StrategyExecution.execution_id == execution.execution_id)
                    )
                    current_execution = result.scalars().first()
                    if not current_execution or current_execution.status != "running":
                        break
                    
                    # Use the fresh execution object
                    execution = current_execution
                    
                    # Get strategy conditions from execution config
                    config = execution.execution_config
                    entry_conditions = config.get('entry_conditions', [])
                    exit_conditions = config.get('exit_conditions', [])
                    
                    # Get current market data
                    historical_data = await self._get_market_data(execution.trading_symbol, api)
                    if not historical_data or len(historical_data) < 50:
                        await asyncio.sleep(60)
                        continue
                    
                    # Get current position for this execution
                    current_position = await self._get_execution_position(execution, db)
                    
                    # Evaluate entry conditions if no position
                    if current_position == 0:
                        entry_signal = await self._evaluate_conditions(entry_conditions, historical_data)
                        
                        if entry_signal:
                            await self._handle_execution_entry_signal(execution, historical_data, db, api)
                    
                    # Evaluate exit conditions if we have a position
                    elif current_position != 0:
                        # Check trailing stop first (highest priority)
                        trailing_stop_triggered = await self._check_trailing_stop(execution, current_position, historical_data)
                        
                        # Check partial exits (handles its own order placement)
                        partial_exit_triggered = await self._check_partial_exits(execution, current_position, historical_data, db, api)
                        
                        # Check regular exit conditions (only if no partial exit was triggered)
                        exit_signal = await self._evaluate_conditions(exit_conditions, historical_data)
                        
                        # Full exit if trailing stop triggered or regular exit signal (but not if partial exit just happened)
                        if (trailing_stop_triggered or exit_signal) and not partial_exit_triggered:
                            await self._handle_execution_exit_signal(execution, current_position, historical_data, db, api)
                    
                    # Update metrics
                    if task_key in self.strategy_metrics:
                        self.strategy_metrics[task_key]['current_position'] = current_position
                    
                    # Sleep based on execution timeframe
                    exec_params = config.get('execution_params', {})
                    timeframe = exec_params.get('timeframe') or config.get('parameters', {}).get('timeframe', '5min')
                    sleep_seconds = self._get_sleep_duration(timeframe)
                    await asyncio.sleep(sleep_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Error in execution monitoring for {execution.trading_symbol}: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            self.logger.info(f"Strategy execution monitoring cancelled for {execution.trading_symbol}")
        except Exception as e:
            self.logger.error(f"Fatal error in execution monitoring: {e}")
            # Update execution status to failed
            execution.status = "failed"
            execution.stopped_at = datetime.now()
            await db.commit()
    
    async def _get_execution_position(self, execution: StrategyExecution, db: AsyncSession) -> int:
        """Get current position for a strategy execution"""
        try:
            result = await db.execute(
                select(Position).filter(
                    and_(
                        Position.user_id == execution.user_id,
                        Position.instrument_token == execution.instrument_token
                    )
                )
            )
            position = result.scalars().first()
            return position.quantity if position else 0
            
        except Exception as e:
            self.logger.error(f"Error getting execution position: {e}")
            return 0
    
    async def _handle_execution_entry_signal(self, execution: StrategyExecution, data: pd.DataFrame, db: AsyncSession, api=None):
        """Handle entry signal for a strategy execution"""
        try:
            task_key = f"{execution.execution_id}_{execution.user_id}"
            
            # Update signals count
            execution.signals_generated += 1
            if task_key in self.strategy_metrics:
                self.strategy_metrics[task_key]['signals_generated'] += 1
            
            current_price = data['close'].iloc[-1]
            
            # Get execution parameters
            exec_params = execution.execution_config.get('execution_params', {})
            position_sizing_percent = exec_params.get('position_sizing_percent')
            position_sizing_mode = exec_params.get('position_sizing_mode', 'Manual Quantity')
            total_capital = exec_params.get('total_capital')
            
            # Calculate position size based on position sizing mode
            if position_sizing_mode == "Auto Calculate" and position_sizing_percent and total_capital:
                # Calculate quantity based on percentage of total capital
                max_investment = total_capital * (position_sizing_percent / 100)
                calculated_quantity = int(max_investment / current_price)
                self.logger.info(f"Auto calculated quantity: {calculated_quantity} (${max_investment:.2f} / ${current_price:.2f})")
            else:
                # Use risk-based calculation
                stop_loss_price = current_price * (1 - (execution.stop_loss or 2.0) / 100)
                calculated_quantity = await self.risk_manager.calculate_position_size(
                    execution.user_id, execution.risk_per_trade, current_price, stop_loss_price, db
                )
            
            # Use the smaller of calculated quantity or execution quantity
            trade_quantity = min(calculated_quantity, execution.quantity)
            
            # Validate the trade
            is_valid, reason = await self.risk_manager.validate_trade(
                execution.user_id, execution.instrument_token, trade_quantity, current_price, 'BUY', db
            )
            
            if not is_valid:
                self.logger.warning(f"Entry trade rejected for {execution.trading_symbol}: {reason}")
                return
            
            # Place order if API is available
            if api:
                order_result = await self._place_execution_order(
                    api, execution, trade_quantity, current_price, 'BUY'
                )
                
                if order_result:
                    self.logger.info(f"Entry order placed for {execution.trading_symbol}: {trade_quantity} shares at INR {current_price}")
                    
                    # Update execution record
                    execution.entry_price = current_price
                    execution.trades_executed += 1
                    execution.last_signal_at = datetime.now()
                    
                    # Update metrics
                    if task_key in self.strategy_metrics:
                        self.strategy_metrics[task_key]['trades_executed'] += 1
                        self.strategy_metrics[task_key]['status'] = 'position_opened'
                    
                    await db.commit()
                    
        except Exception as e:
            self.logger.error(f"Error handling execution entry signal: {e}")
    
    async def _handle_execution_exit_signal(self, execution: StrategyExecution, current_position: int, 
                                          data: pd.DataFrame, db: AsyncSession, api=None):
        """Handle exit signal for a strategy execution"""
        try:
            task_key = f"{execution.execution_id}_{execution.user_id}"
            current_price = data['close'].iloc[-1]
            
            # Determine transaction type based on current position
            transaction_type = 'SELL' if current_position > 0 else 'BUY'
            quantity = abs(current_position)
            
            # Place exit order if API is available
            if api:
                order_result = await self._place_execution_order(
                    api, execution, quantity, current_price, transaction_type
                )
                
                if order_result:
                    self.logger.info(f"Exit order placed for {execution.trading_symbol}: {quantity} shares at INR {current_price}")
                    
                    # Calculate P&L if we have entry price
                    if execution.entry_price:
                        if current_position > 0:  # Long position
                            pnl = (current_price - execution.entry_price) * quantity
                        else:  # Short position  
                            pnl = (execution.entry_price - current_price) * quantity
                        
                        execution.pnl += pnl
                        execution.exit_price = current_price
                        
                        # Update metrics
                        if task_key in self.strategy_metrics:
                            self.strategy_metrics[task_key]['pnl'] = execution.pnl
                    
                    # Update execution record
                    execution.trades_executed += 1
                    execution.last_signal_at = datetime.now()
                    
                    if task_key in self.strategy_metrics:
                        self.strategy_metrics[task_key]['trades_executed'] += 1
                        self.strategy_metrics[task_key]['status'] = 'position_closed'
                    
                    await db.commit()
                        
        except Exception as e:
            self.logger.error(f"Error handling execution exit signal: {e}")
    
    async def _check_trailing_stop(self, execution: StrategyExecution, current_position: int, data: pd.DataFrame) -> bool:
        """Check if trailing stop should be triggered"""
        try:
            exec_params = execution.execution_config.get('execution_params', {})
            if not exec_params.get('trailing_stop_enabled'):
                return False
                
            trailing_stop_percent = exec_params.get('trailing_stop_percent')
            trailing_stop_min = exec_params.get('trailing_stop_min')
            
            if not trailing_stop_percent:
                return False
            
            current_price = data['close'].iloc[-1]
            
            # Initialize or update trailing stop price
            if not hasattr(execution, 'trailing_stop_price') or not execution.trailing_stop_price:
                if current_position > 0:  # Long position
                    execution.trailing_stop_price = current_price * (1 - trailing_stop_percent / 100)
                else:  # Short position
                    execution.trailing_stop_price = current_price * (1 + trailing_stop_percent / 100)
                self.logger.info(f"Initialized trailing stop at {execution.trailing_stop_price} for {execution.trading_symbol}")
                return False
            
            # Update trailing stop if price moved favorably
            if current_position > 0:  # Long position
                new_trailing_stop = current_price * (1 - trailing_stop_percent / 100)
                if new_trailing_stop > execution.trailing_stop_price:
                    execution.trailing_stop_price = new_trailing_stop
                    self.logger.info(f"Updated trailing stop to {execution.trailing_stop_price} for {execution.trading_symbol}")
                
                # Check if trailing stop triggered
                if current_price <= execution.trailing_stop_price:
                    self.logger.info(f"Trailing stop triggered for {execution.trading_symbol}: {current_price} <= {execution.trailing_stop_price}")
                    return True
            else:  # Short position
                new_trailing_stop = current_price * (1 + trailing_stop_percent / 100)
                if new_trailing_stop < execution.trailing_stop_price:
                    execution.trailing_stop_price = new_trailing_stop
                    self.logger.info(f"Updated trailing stop to {execution.trailing_stop_price} for {execution.trading_symbol}")
                
                # Check if trailing stop triggered
                if current_price >= execution.trailing_stop_price:
                    self.logger.info(f"Trailing stop triggered for {execution.trading_symbol}: {current_price} >= {execution.trailing_stop_price}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trailing stop: {e}")
            return False
    
    async def _check_partial_exits(self, execution: StrategyExecution, current_position: int, 
                                 data: pd.DataFrame, db: AsyncSession, api=None) -> bool:
        """Check and execute partial exits"""
        try:
            exec_params = execution.execution_config.get('execution_params', {})
            partial_exits = exec_params.get('partial_exits', [])
            
            if not partial_exits or not execution.entry_price:
                return False
            
            current_price = data['close'].iloc[-1]
            
            # Calculate current profit percentage
            if current_position > 0:  # Long position
                profit_percent = ((current_price - execution.entry_price) / execution.entry_price) * 100
            else:  # Short position
                profit_percent = ((execution.entry_price - current_price) / execution.entry_price) * 100
            
            # Check each partial exit target
            for partial_exit in partial_exits:
                target_percent = partial_exit.get('target', 0)
                qty_percent = partial_exit.get('qty_percent', 0)
                
                if profit_percent >= target_percent:
                    # Calculate partial exit quantity
                    original_quantity = abs(current_position)
                    exit_quantity = int(original_quantity * (qty_percent / 100))
                    
                    if exit_quantity > 0:
                        # Execute partial exit
                        transaction_type = 'SELL' if current_position > 0 else 'BUY'
                        
                        if api:
                            order_result = await self._place_execution_order(
                                api, execution, exit_quantity, current_price, transaction_type
                            )
                            
                            if order_result:
                                self.logger.info(f"Partial exit executed for {execution.trading_symbol}: {exit_quantity} shares at {current_price} ({profit_percent:.1f}% profit)")
                                
                                # Update execution record
                                execution.trades_executed += 1
                                execution.last_signal_at = datetime.now()
                                
                                # Remove this partial exit target (executed)
                                partial_exits.remove(partial_exit)
                                exec_params['partial_exits'] = partial_exits
                                
                                await db.commit()
                                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking partial exits: {e}")
            return False
    
    async def _place_execution_order(self, api, execution: StrategyExecution, quantity: int, 
                                   price: float, transaction_type: str) -> bool:
        """Place an order for strategy execution"""
        try:
            # This is a placeholder for actual order placement
            # In real implementation, you'd use the actual broker API
            self.logger.info(f"MOCK ORDER: {transaction_type} {quantity} {execution.trading_symbol} at INR {price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing execution order: {e}")
            return False
    async def deactivate_strategy(self, strategy_id: str, user_id: str, db: AsyncSession) -> Dict[str, Any]:
        """Deactivate a strategy template and stop all running executions"""
        try:
            self.logger.info(f"Deactivating strategy template {strategy_id} for user {user_id}")
            
            # Find the strategy
            result = await db.execute(
                select(Strategy).filter(
                    and_(
                        Strategy.strategy_id == strategy_id,
                        Strategy.user_id == user_id
                    )
                )
            )
            strategy = result.scalars().first()
            
            if not strategy:
                return {
                    "success": False,
                    "message": "Strategy not found or access denied"
                }
            
            # Find all running executions for this strategy
            exec_result = await db.execute(
                select(StrategyExecution).filter(
                    and_(
                        StrategyExecution.strategy_id == strategy_id,
                        StrategyExecution.user_id == user_id,
                        StrategyExecution.status == "running"
                    )
                )
            )
            running_executions = exec_result.scalars().all()
            
            # Stop all running executions
            stopped_count = 0
            for execution in running_executions:
                stop_result = await self.stop_strategy_execution(execution.execution_id, user_id, db)
                if stop_result.get('success'):
                    stopped_count += 1
            
            # Update strategy status to inactive
            strategy.status = "inactive"
            strategy.updated_at = datetime.now()
            await db.commit()
            
            message = f"Strategy template deactivated successfully."
            if stopped_count > 0:
                message += f" Also stopped {stopped_count} running executions."
            
            self.logger.info(f"Strategy {strategy_id} deactivated successfully")
            
            return {
                "success": True,
                "message": message,
                "strategy_id": strategy_id,
                "status": "inactive",
                "stopped_executions": stopped_count
            }
            
        except Exception as e:
            await db.rollback()
            self.logger.error(f"Error deactivating strategy {strategy_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to deactivate strategy: {str(e)}"
            }
    
    async def _monitor_instrument_strategy(self, strategy_id: str, user_id: str, 
                                         instrument: Dict, strategy_data: Dict, 
                                         db: AsyncSession, api=None):
        """Monitor a single instrument for strategy conditions"""
        task_key = f"{strategy_id}_{instrument['token']}_{user_id}"
        trading_symbol = instrument['symbol']
        
        self.logger.info(f"Started monitoring {trading_symbol} for strategy {strategy_id}")
        
        try:
            while True:
                try:
                    # Update metrics
                    if task_key in self.strategy_metrics:
                        self.strategy_metrics[task_key]['last_check'] = datetime.now()
                    
                    # Check if trading is still allowed
                    trading_allowed, _ = await self.risk_manager.check_trading_allowed(user_id, db)
                    if not trading_allowed:
                        self.logger.warning(f"Trading suspended for user {user_id}, pausing strategy monitoring")
                        await asyncio.sleep(300)  # Wait 5 minutes before checking again
                        continue
                    
                    # Get current market data
                    historical_data = await self._get_market_data(trading_symbol, api)
                    if not historical_data or len(historical_data) < 50:
                        await asyncio.sleep(60)  # Wait 1 minute if no data
                        continue
                    
                    # Check current position
                    current_position = await self._get_current_position(user_id, instrument['token'], db)
                    
                    # Evaluate entry conditions if no position
                    if current_position == 0:
                        entry_signal = await self._evaluate_conditions(
                            strategy_data['entry_conditions'], historical_data
                        )
                        
                        if entry_signal:
                            await self._handle_entry_signal(
                                strategy_id, user_id, instrument, strategy_data, 
                                historical_data, db, api
                            )
                    
                    # Evaluate exit conditions if we have a position
                    elif current_position != 0:
                        exit_signal = await self._evaluate_conditions(
                            strategy_data['exit_conditions'], historical_data
                        )
                        
                        if exit_signal:
                            await self._handle_exit_signal(
                                strategy_id, user_id, instrument, current_position,
                                historical_data, db, api
                            )
                    
                    # Update position in metrics
                    if task_key in self.strategy_metrics:
                        self.strategy_metrics[task_key]['current_position'] = current_position
                    
                    # Wait before next check (based on timeframe)
                    timeframe = strategy_data.get('parameters', {}).get('timeframe', '5min')
                    sleep_seconds = self._get_sleep_duration(timeframe)
                    await asyncio.sleep(sleep_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Error in strategy monitoring for {trading_symbol}: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    
        except asyncio.CancelledError:
            self.logger.info(f"Strategy monitoring cancelled for {trading_symbol}")
        except Exception as e:
            self.logger.error(f"Fatal error in strategy monitoring for {trading_symbol}: {e}")
    
    async def _get_market_data(self, trading_symbol: str, api=None) -> pd.DataFrame:
        """Get historical market data for analysis"""
        try:
            # Get last 100 periods of data for indicator calculations
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            
            data = await get_historical_data(
                upstox_api=api,
                upstox_access_token=None,
                trading_symbol=trading_symbol,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
                unit="days",
                interval="1",
                source="default"
            )
            
            if hasattr(data, 'data') and data.data:
                df = pd.DataFrame(data.data)
                # Ensure column names are capitalized for technical indicators
                if 'close' in df.columns:
                    df['Close'] = df['close']
                if 'open' in df.columns:
                    df['Open'] = df['open']
                if 'high' in df.columns:
                    df['High'] = df['high']
                if 'low' in df.columns:
                    df['Low'] = df['low']
                if 'volume' in df.columns:
                    df['Volume'] = df['volume']
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {trading_symbol}: {e}")
            return pd.DataFrame()
    
    async def _evaluate_conditions(self, conditions: List[Dict], data: pd.DataFrame) -> bool:
        """Evaluate strategy conditions against market data"""
        try:
            if not conditions or data.empty:
                return False
            
            # Calculate all required indicators
            indicators = self._calculate_indicators(data)
            
            # Evaluate each condition
            for condition in conditions:
                if not await self._evaluate_single_condition(condition, indicators, data):
                    return False  # All conditions must be true (AND logic)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating conditions: {e}")
            return False
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}
        
        try:
            if len(data) < 50:
                return indicators
            
            # Price indicators
            indicators['Close Price'] = data['close'].iloc[-1]
            indicators['Open Price'] = data['open'].iloc[-1]
            indicators['High Price'] = data['high'].iloc[-1]
            indicators['Low Price'] = data['low'].iloc[-1]
            indicators['Volume'] = data['volume'].iloc[-1]
            
            # Moving averages
            sma_20_data = moving_average(data.copy(), 20)
            if 'MA_20' in sma_20_data.columns:
                indicators['SMA_20'] = sma_20_data['MA_20'].iloc[-1]
                
            sma_50_data = moving_average(data.copy(), 50)  
            if 'MA_50' in sma_50_data.columns:
                indicators['SMA_50'] = sma_50_data['MA_50'].iloc[-1]
                
            ema_14_data = exponential_moving_average(data.copy(), 14)
            if 'EMA_14' in ema_14_data.columns:
                indicators['EMA_14'] = ema_14_data['EMA_14'].iloc[-1]
                
            ema_21_data = exponential_moving_average(data.copy(), 21)
            if 'EMA_21' in ema_21_data.columns:
                indicators['EMA_21'] = ema_21_data['EMA_21'].iloc[-1]
            
            # RSI
            rsi_data = relative_strength_index(data.copy(), 14)
            if 'RSI' in rsi_data.columns:
                indicators['RSI_14'] = rsi_data['RSI'].iloc[-1]
            
            # MACD
            macd_data = macd(data.copy(), 12, 26)
            if 'MACD' in macd_data.columns:
                indicators['MACD'] = macd_data['MACD'].iloc[-1]
                indicators['MACD_Signal'] = macd_data['MACDsign'].iloc[-1] if 'MACDsign' in macd_data.columns else None
                indicators['MACD_Histogram'] = macd_data['MACDdiff'].iloc[-1] if 'MACDdiff' in macd_data.columns else None
            
            # Bollinger Bands
            bb_data = bollinger_bands(data.copy(), 20)
            if 'BB_up' in bb_data.columns and 'BB_dn' in bb_data.columns and 'BB_ma' in bb_data.columns:
                indicators['BB_Upper'] = bb_data['BB_up'].iloc[-1]
                indicators['BB_Middle'] = bb_data['BB_ma'].iloc[-1]
                indicators['BB_Lower'] = bb_data['BB_dn'].iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    async def _evaluate_single_condition(self, condition: Dict, indicators: Dict, data: pd.DataFrame) -> bool:
        """Evaluate a single condition"""
        try:
            left_value = self._get_condition_value(condition, 'left', indicators, data)
            right_value = self._get_condition_value(condition, 'right', indicators, data)
            comparison = condition['comparison']
            
            if left_value is None or right_value is None:
                return False
            
            if comparison == '>':
                return left_value > right_value
            elif comparison == '<':
                return left_value < right_value
            elif comparison == '>=':
                return left_value >= right_value
            elif comparison == '<=':
                return left_value <= right_value
            elif comparison == '==':
                return abs(left_value - right_value) < 0.01  # Allow small floating point differences
            elif comparison == 'Crosses Above':
                return self._check_crossover(condition, indicators, data, 'above')
            elif comparison == 'Crosses Below':
                return self._check_crossover(condition, indicators, data, 'below')
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _get_condition_value(self, condition: Dict, side: str, indicators: Dict, data: pd.DataFrame) -> Optional[float]:
        """Get the value for one side of a condition"""
        try:
            indicator_key = f'{side}_indicator'
            value_key = f'{side}_value'
            params_key = f'{side}_params'
            
            if condition.get(value_key) is not None:
                # Fixed value
                return float(condition[value_key])
            
            indicator_name = condition.get(indicator_key)
            if not indicator_name:
                return None
            
            params = condition.get(params_key, {})
            
            # Handle parameterized indicators
            if indicator_name == 'SMA' and params.get('period'):
                period = int(params['period'])
                sma_data = moving_average(data.copy(), period)
                column_name = f'MA_{period}'
                if column_name in sma_data.columns:
                    return sma_data[column_name].iloc[-1]
            elif indicator_name == 'EMA' and params.get('period'):
                period = int(params['period'])
                ema_data = exponential_moving_average(data.copy(), period)
                column_name = f'EMA_{period}'
                if column_name in ema_data.columns:
                    return ema_data[column_name].iloc[-1]
            elif indicator_name == 'RSI' and params.get('period'):
                period = int(params['period'])
                rsi_data = relative_strength_index(data.copy(), period)
                if 'RSI' in rsi_data.columns:
                    return rsi_data['RSI'].iloc[-1]
            
            # Handle direct indicator access
            return indicators.get(indicator_name)
            
        except Exception as e:
            self.logger.error(f"Error getting condition value: {e}")
            return None
    
    def _check_crossover(self, condition: Dict, indicators: Dict, data: pd.DataFrame, direction: str) -> bool:
        """Check if one indicator crosses above/below another"""
        try:
            # This is simplified - in a real implementation, you'd check the previous values
            # to confirm an actual crossover occurred
            left_value = self._get_condition_value(condition, 'left', indicators, data)
            right_value = self._get_condition_value(condition, 'right', indicators, data)
            
            if left_value is None or right_value is None:
                return False
            
            # For now, just check current relationship
            # TODO: Implement proper crossover detection using historical data
            if direction == 'above':
                return left_value > right_value
            else:
                return left_value < right_value
                
        except Exception as e:
            self.logger.error(f"Error checking crossover: {e}")
            return False
    
    async def _handle_entry_signal(self, strategy_id: str, user_id: str, instrument: Dict,
                                 strategy_data: Dict, data: pd.DataFrame, db: AsyncSession, api=None):
        """Handle entry signal by placing a trade"""
        try:
            task_key = f"{strategy_id}_{instrument['token']}_{user_id}"
            
            # Update metrics
            if task_key in self.strategy_metrics:
                self.strategy_metrics[task_key]['signals_generated'] += 1
            
            current_price = data['close'].iloc[-1]
            
            # Calculate position size using risk management
            risk_per_trade = instrument.get('risk_per_trade', 2.0)
            stop_loss_price = current_price * 0.98  # 2% stop loss (simplified)
            
            quantity = await self.risk_manager.calculate_position_size(
                user_id, risk_per_trade, current_price, stop_loss_price, db
            )
            
            # Validate trade
            is_valid, reason = await self.risk_manager.validate_trade(
                user_id, instrument['token'], quantity, current_price, 'BUY', db
            )
            
            if not is_valid:
                self.logger.warning(f"Trade rejected for {instrument['symbol']}: {reason}")
                return
            
            # Place order
            if api:
                order_result = await place_order(
                    api=api,
                    instrument_token=instrument['token'],
                    quantity=quantity,
                    price=current_price,
                    order_type='MARKET',
                    transaction_type='BUY',
                    product_type='CNC'
                )
                
                if order_result:
                    self.logger.info(f"Entry order placed for {instrument['symbol']}: {quantity} shares at ₹{current_price}")
                    
                    # Update metrics
                    if task_key in self.strategy_metrics:
                        self.strategy_metrics[task_key]['trades_executed'] += 1
                        self.strategy_metrics[task_key]['status'] = 'position_opened'
                    
        except Exception as e:
            self.logger.error(f"Error handling entry signal: {e}")
    
    async def _handle_exit_signal(self, strategy_id: str, user_id: str, instrument: Dict,
                                current_position: int, data: pd.DataFrame, db: AsyncSession, api=None):
        """Handle exit signal by closing position"""
        try:
            task_key = f"{strategy_id}_{instrument['token']}_{user_id}"
            current_price = data['close'].iloc[-1]
            
            # Determine transaction type based on current position
            transaction_type = 'SELL' if current_position > 0 else 'BUY'
            quantity = abs(current_position)
            
            # Place exit order
            if api:
                order_result = await place_order(
                    api=api,
                    instrument_token=instrument['token'],
                    quantity=quantity,
                    price=current_price,
                    order_type='MARKET',
                    transaction_type=transaction_type,
                    product_type='CNC'
                )
                
                if order_result:
                    self.logger.info(f"Exit order placed for {instrument['symbol']}: {quantity} shares at ₹{current_price}")
                    
                    # Update metrics
                    if task_key in self.strategy_metrics:
                        self.strategy_metrics[task_key]['trades_executed'] += 1
                        self.strategy_metrics[task_key]['status'] = 'position_closed'
                        
        except Exception as e:
            self.logger.error(f"Error handling exit signal: {e}")
    
    async def _get_current_position(self, user_id: str, instrument_token: str, db: AsyncSession) -> int:
        """Get current position for an instrument"""
        try:
            result = await db.execute(
                select(Position).filter(
                    and_(
                        Position.user_id == user_id,
                        Position.instrument_token == instrument_token
                    )
                )
            )
            position = result.scalars().first()
            return position.quantity if position else 0
            
        except Exception as e:
            self.logger.error(f"Error getting position: {e}")
            return 0
    
    def _get_sleep_duration(self, timeframe: str) -> int:
        """Get sleep duration based on timeframe"""
        timeframe_mapping = {
            '1min': 60,
            '3min': 180,
            '5min': 300,
            '15min': 900,
            '30min': 1800,
            '60min': 3600,
            'day': 3600  # Check once per hour for daily timeframe
        }
        return timeframe_mapping.get(timeframe, 300)  # Default to 5 minutes
    
    def get_strategy_metrics(self, strategy_id: str, user_id: str) -> Dict[str, Any]:
        """Get metrics for all instruments of a strategy"""
        try:
            strategy_metrics = {}
            for task_key, metrics in self.strategy_metrics.items():
                if task_key.startswith(f"{strategy_id}_") and task_key.endswith(f"_{user_id}"):
                    instrument_token = task_key.split('_')[1]
                    strategy_metrics[instrument_token] = metrics
            
            return strategy_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting strategy metrics: {e}")
            return {}
    
    def get_all_active_strategies(self) -> List[str]:
        """Get list of all active strategy task keys"""
        return list(self.active_strategies.keys())
    
    async def emergency_stop_execution(self, execution_id: str):
        """Emergency stop for execution - simplified version for emergency use"""
        try:
            self.logger.warning(f"Emergency stop requested for execution {execution_id}")
            # Find and cancel the monitoring task
            for task_key, task in list(self.active_strategies.items()):
                if execution_id in task_key:
                    task.cancel()
                    del self.active_strategies[task_key]
                    self.logger.info(f"Emergency stopped monitoring task for {execution_id}")
                    break
        except Exception as e:
            self.logger.error(f"Error in emergency stop for {execution_id}: {str(e)}")
    
    async def stop_all_strategies(self):
        """Stop all active strategies (for shutdown)"""
        for task_key, task in self.active_strategies.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.active_strategies.clear()
        self.strategy_metrics.clear()
        self.logger.info("All strategy monitoring stopped")