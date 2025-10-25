import asyncio
import json
import logging
import inspect
import os
import sys
from datetime import datetime, date, time as date_time, timedelta
from typing import Optional, Dict, Any, List, Callable
import pandas as pd
import numpy as np
import requests
import smtplib
from email.mime.text import MIMEText
# Force the same import path as main.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.database import db_manager, get_db
import pandas_ta as ta
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from sqlalchemy.sql import text, func
from fastapi import Depends, HTTPException
from kiteconnect import KiteConnect
import upstox_client
from backtesting import Backtest
from common_utils.backtesting_adapter import (
    RSIStrategy, MACDStrategy, BollingerBandsStrategy,
    prepare_data_for_backtesting, convert_backtesting_result
)
from common_utils.db_utils import async_fetch_query, async_execute_query
from common_utils.read_write_sql_data import load_sql_data
from common_utils.fetch_load_db_data import *
from common_utils.market_data import MarketData
from common_utils.upstox_utils import get_symbol_for_instrument
from .models import Order, ScheduledOrder, QueuedOrder, User, GTTOrder, Instrument
from .schemas import Order as OrderSchema, ScheduledOrder as ScheduledOrderSchema, OrderHistory, Trade, QuoteResponse, \
    OHLCResponse, LTPResponse, HistoricalDataResponse, Instrument as InstrumentSchema, HistoricalDataPoint, MFSIPResponse, MFSIPRequest
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
from common_utils.indicators import calculate_ema, calculate_rsi, calculate_linear_regression, calculate_atr, \
    calculate_macd, calculate_bollinger_bands, calculate_stochastic_oscillator, calculate_sma
from common_utils.predefined_strategies import check_macd_crossover, check_bollinger_band_signals, check_stochastic_signals, \
    check_support_resistance_breakout, macd_strategy, bollinger_band_strategy, rsi_strategy
try:
    from backend.app.ws_events import broadcast_event as ws_broadcast_event
except Exception:
    async def ws_broadcast_event(*args, **kwargs):
        return None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

strategy_tasks = {}
market_data = MarketData()

UPSTOX_TOKEN_EXPIRY_TIME = date_time(3, 30)
ZERODHA_TOKEN_EXPIRY_TIME = date_time(6, 0)
EMAIL_SENDER = os.getenv("SMTP_USER")
EMAIL_PASSWORD = os.getenv("SMTP_PASSWORD")


def get_next_expiry_time(target_time: date_time) -> datetime:
    now = datetime.now()
    target_datetime = datetime.combine(now.date(), target_time)
    if now > target_datetime:
        target_datetime += timedelta(days=1)
    return target_datetime

class TokenExpiredError(Exception):
    def __init__(self, broker: str):
        self.broker = broker
        self.message = f"{broker} access token has expired. Please re-authenticate."
        super().__init__(self.message)

async def send_email(subject: str, body: str, recipient: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = recipient
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(f"Email sent to {recipient}: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email to {recipient}: {str(e)}")

async def notify(title: str, message: str, recipient: Optional[str] = None, type: str = 'success'):
    logger.info(f"Notification: {title} - {message}")
    if recipient and EMAIL_SENDER and EMAIL_PASSWORD:
        await send_email(title, message, recipient)

class OrderMonitor:
    def __init__(self):
        self.running: bool = True
        self.polling_interval: int = 60
        self.monitor_tasks: List[asyncio.Task] = []
        logger.info("OrderMonitor initialized")

    async def run_scheduled_tasks(self, user_apis: Dict[str, Dict[str, Any]]):
        """CORRECTED: Proper database session handling for long-running tasks with dynamic user API loading"""
        logger.info("Starting OrderMonitor scheduled tasks")
        try:
            while self.running:
                try:
                    # CORRECT WAY: Use the session factory directly, not the dependency
                    session_factory = db_manager.get_session_factory('trading_db')
                    if not session_factory:
                        logger.error("Database session factory not available")
                        await asyncio.sleep(self.polling_interval)
                        continue

                    # Create a new session for this iteration
                    async with session_factory() as db:
                        try:
                            # Dynamically load active users from database if user_apis is empty or limited
                            active_user_apis = await self._get_active_user_apis(db, user_apis)
                            
                            # Process user APIs if available
                            if active_user_apis:
                                for user_id, apis in active_user_apis.items():
                                    try:
                                        upstox_api = apis.get("upstox", {})
                                        zerodha_api = apis.get("zerodha", {}).get("kite")

                                        if upstox_api or zerodha_api:
                                            logger.info(f"Sync Order status for pending orders for user {user_id}")
                                            await self.sync_order_statuses(upstox_api, zerodha_api, db, user_id)

                                    except Exception as user_error:
                                        logger.error(f"Error processing user {user_id}: {user_error}")

                                # Monitor trailing stop loss orders every 5 minutes (300 seconds)
                                current_time = datetime.now()
                                if not hasattr(self, '_last_trailing_check'):
                                    self._last_trailing_check = current_time - timedelta(minutes=4)

                                time_since_last_check = (current_time - self._last_trailing_check).total_seconds()
                                if time_since_last_check >= 300:  # 5 minutes
                                    logger.info("Running trailing stop loss monitoring check")
                                    await self.monitor_trailing_stop_loss_orders(active_user_apis, db)
                                    self._last_trailing_check = current_time
                            else:
                                logger.debug("No active user APIs available for processing")

                            # Commit any changes
                            await db.commit()

                        except Exception as processing_error:
                            logger.error(f"Error in scheduled tasks processing: {processing_error}")
                            await db.rollback()

                except Exception as session_error:
                    logger.error(f"Error with database session: {session_error}")

                # Wait before next iteration
                await asyncio.sleep(self.polling_interval)

        except asyncio.CancelledError:
            logger.info("OrderMonitor scheduled tasks cancelled")
            self.running = False
            await self.cancel_all_tasks()
            raise
        except Exception as e:
            logger.error(f"Unexpected error in run_scheduled_tasks: {e}")
            self.running = False
            await self.cancel_all_tasks()

    async def _get_active_user_apis(self, db: AsyncSession, provided_user_apis: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically load active user APIs from database.
        First tries to use provided user APIs, then loads from database if needed.
        """
        active_apis = {}
        
        try:
            # First, use any provided user APIs
            if provided_user_apis:
                active_apis.update(provided_user_apis)
                logger.debug(f"Using {len(provided_user_apis)} provided user APIs")
            
            # Get users who have pending orders, scheduled orders, or active GTT orders
            query = text("""
                SELECT DISTINCT u.user_id, u.upstox_access_token, u.upstox_access_token_expiry,
                       u.zerodha_access_token, u.zerodha_access_token_expiry,
                       u.upstox_api_key, u.upstox_api_secret, u.zerodha_api_key, u.zerodha_api_secret
                FROM users u
                WHERE u.user_id IN (
                    SELECT DISTINCT user_id FROM orders WHERE status IN ('open', 'pending', 'trigger pending', 'amo req received')
                    UNION
                    SELECT DISTINCT user_id FROM scheduled_orders WHERE status = 'PENDING'
                    UNION
                    SELECT DISTINCT user_id FROM gtt_orders WHERE status IN ('PENDING', 'active', 'scheduled')
                )
                AND (
                    (u.upstox_access_token IS NOT NULL AND u.upstox_access_token_expiry > NOW())
                    OR 
                    (u.zerodha_access_token IS NOT NULL AND u.zerodha_access_token_expiry > NOW())
                )
            """)
            
            result = await db.execute(query)
            users_with_active_orders = result.fetchall()
            
            # Initialize APIs for users not already in provided_user_apis
            for user_row in users_with_active_orders:
                user_id = user_row.user_id
                
                # Skip if we already have APIs for this user
                if user_id in active_apis:
                    continue
                    
                try:
                    # Import here to avoid circular imports
                    from backend.app.api_manager import initialize_user_apis
                    
                    user_apis_dict = await initialize_user_apis(user_id, db)
                    if user_apis_dict:
                        active_apis[user_id] = user_apis_dict
                        logger.info(f"Dynamically loaded APIs for user {user_id}")
                        
                except Exception as init_error:
                    logger.warning(f"Failed to initialize APIs for user {user_id}: {init_error}")
                    
            logger.info(f"Total active user APIs: {len(active_apis)}")
            
        except Exception as e:
            logger.error(f"Error loading active user APIs: {e}")
            # Return at least the provided APIs if database query fails
            return provided_user_apis if provided_user_apis else {}
            
        return active_apis

    async def monitor_trailing_stop_loss_orders(self, active_user_apis: Dict[str, Dict[str, Any]], db: AsyncSession):
        """Monitor and update trailing stop loss orders every 5 minutes"""
        try:
            user_ids = tuple(active_user_apis.keys()) if active_user_apis else ()
            if not user_ids:
                logger.debug("No active user APIs for trailing stop loss monitoring")
                return

            # Enhanced query with exit_order_id filtering for better tracking
            query = text("""
                SELECT * FROM orders 
                WHERE (
                    (is_trailing_stop_loss = true AND status IN ('complete', 'COMPLETE') 
                     AND exit_order_id IS NULL
                     AND remarks NOT LIKE '%triggered%')
                    OR
                    (status IN ('complete', 'COMPLETE') 
                     AND (stop_loss > 0 OR target > 0)
                     AND exit_order_id IS NULL
                     AND remarks NOT LIKE '%Stop Loss triggered%'
                     AND remarks NOT LIKE '%Target reached%'
                     AND remarks NOT LIKE '%Exit order%')
                )
                AND user_id = ANY(:user_ids)
            """)

            result = await db.execute(query, {"user_ids": list(user_ids)})
            orders_to_monitor = result.fetchall()

            logger.info(f"Found {len(orders_to_monitor)} orders to monitor (trailing stop loss and regular stop loss/target)")

            for order_row in orders_to_monitor:
                try:
                    user_id = order_row.user_id
                    order_id = order_row.order_id
                    trading_symbol = order_row.trading_symbol
                    instrument_token = order_row.instrument_token
                    transaction_type = order_row.transaction_type
                    quantity = order_row.quantity
                    entry_price = float(order_row.price) if order_row.price else 0.0

                    # Get order parameters
                    is_trailing_stop_loss = bool(order_row.is_trailing_stop_loss)
                    stop_loss = float(order_row.stop_loss or 0)
                    target = float(order_row.target or 0)

                    # Trailing stop loss parameters
                    trailing_stop_loss_percent = float(order_row.trailing_stop_loss_percent or 0)
                    trail_start_target_percent = float(order_row.trail_start_target_percent or 0)
                    trailing_activated = bool(order_row.trailing_activated)
                    highest_price = float(order_row.highest_price_achieved or entry_price)

                    # Skip if no monitoring parameters are set
                    if not is_trailing_stop_loss and stop_loss <= 0 and target <= 0:
                        continue

                    # Validate trailing stop loss parameters
                    if is_trailing_stop_loss and (trailing_stop_loss_percent <= 0 or trail_start_target_percent <= 0):
                        logger.warning(f"Invalid trailing parameters for order {order_id}")
                        continue

                    # Get user APIs
                    user_apis = active_user_apis.get(user_id, {})
                    upstox_api = user_apis.get("upstox", {}).get("market_data")
                    zerodha_api = user_apis.get("zerodha", {}).get("kite")

                    if not (upstox_api or zerodha_api):
                        logger.warning(f"No market data API available for user {user_id}")
                        continue

                    # Get current market price
                    try:
                        ltp_data = await get_ltp(upstox_api, zerodha_api, [instrument_token], db)
                        if not ltp_data:
                            logger.warning(f"No LTP data for {trading_symbol}")
                            continue

                        current_price = ltp_data[0].last_price
                        logger.debug(f"Current price for {trading_symbol}: {current_price}, Entry: {entry_price}")

                    except Exception as price_error:
                        logger.error(f"Error fetching current price for {trading_symbol}: {price_error}")
                        continue

                    # Check exit conditions in priority order
                    exit_triggered = False
                    exit_reason = None
                    exit_price = current_price

                    # Priority 1: Regular Stop Loss (highest priority for BUY orders when price goes down)
                    if stop_loss > 0:
                        if transaction_type == "BUY" and current_price <= stop_loss:
                            exit_triggered = True
                            exit_reason = "Stop Loss triggered"
                            exit_price = stop_loss
                        elif transaction_type == "SELL" and current_price >= stop_loss:
                            exit_triggered = True
                            exit_reason = "Stop Loss triggered"
                            exit_price = stop_loss

                    # Priority 2: Regular Target (high priority for profit taking)
                    if not exit_triggered and target > 0:
                        if transaction_type == "BUY" and current_price >= target:
                            exit_triggered = True
                            exit_reason = "Target reached"
                            exit_price = target
                        elif transaction_type == "SELL" and current_price <= target:
                            exit_triggered = True
                            exit_reason = "Target reached"
                            exit_price = target

                    # Priority 3: Trailing Stop Loss (only if no regular stop loss/target triggered)
                    if not exit_triggered and is_trailing_stop_loss:
                        # Update highest price if current price is higher (for BUY orders)
                        if transaction_type == "BUY" and current_price > highest_price:
                            highest_price = current_price

                            # Update highest price in database
                            update_query = text("""
                                UPDATE orders 
                                SET highest_price_achieved = :highest_price 
                                WHERE order_id = :order_id
                            """)
                            await db.execute(update_query, {
                                "highest_price": highest_price,
                                "order_id": order_id
                            })

                            logger.info(f"Updated highest price for {trading_symbol} to {highest_price}")

                        # Check if trailing should be activated
                        if transaction_type == "BUY":
                            price_gain_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                        else:
                            price_gain_percent = ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0

                        if not trailing_activated and price_gain_percent >= trail_start_target_percent:
                            # Activate trailing stop loss
                            trailing_activated = True

                            update_query = text("""
                                UPDATE orders 
                                SET trailing_activated = true 
                                WHERE order_id = :order_id
                            """)
                            await db.execute(update_query, {"order_id": order_id})

                            logger.info(
                                f"Activated trailing stop loss for {trading_symbol} at {current_price} ({price_gain_percent:.2f}% gain)")

                            # Send notification
                            result = await db.execute(select(User).filter(User.user_id == user_id))
                            user = result.scalars().first()
                            if user:
                                await notify(
                                    f"Trailing Stop Activated: {trading_symbol}",
                                    f"Trailing stop loss activated at {current_price} with {price_gain_percent:.2f}% gain",
                                    user.email
                                )

                        # Check trailing stop loss condition if activated
                        if trailing_activated:
                            # Calculate trailing stop price (percentage below highest price for BUY orders)
                            if transaction_type == "BUY":
                                trailing_stop_price = highest_price * (1 - trailing_stop_loss_percent / 100)
                                if current_price <= trailing_stop_price:
                                    exit_triggered = True
                                    exit_reason = "Trailing stop loss triggered"
                                    exit_price = trailing_stop_price
                            else:  # SELL orders
                                # For SELL orders, trailing stop moves up as price goes down
                                lowest_price = float(order_row.lowest_price_achieved or entry_price)
                                if current_price < lowest_price:
                                    lowest_price = current_price
                                    # Update lowest price in database
                                    update_query = text("""
                                        UPDATE orders 
                                        SET lowest_price_achieved = :lowest_price 
                                        WHERE order_id = :order_id
                                    """)
                                    await db.execute(update_query, {
                                        "lowest_price": lowest_price,
                                        "order_id": order_id
                                    })

                                trailing_stop_price = lowest_price * (1 + trailing_stop_loss_percent / 100)
                                if current_price >= trailing_stop_price:
                                    exit_triggered = True
                                    exit_reason = "Trailing stop loss triggered"
                                    exit_price = trailing_stop_price

                            logger.debug(
                                f"Trailing check for {trading_symbol}: Current={current_price}, TrailingStop={trailing_stop_price}, Highest={highest_price}")

                    # Execute exit order if any condition is triggered
                    if exit_triggered:
                        logger.info(f"{exit_reason} for {trading_symbol} at {current_price}")

                        try:
                            # Get appropriate API for placing order
                            order_api = user_apis.get("upstox", {}).get(
                                "order") if order_row.broker == "Upstox" else user_apis.get("zerodha", {}).get(
                                "kite")

                            if order_api:
                                # Place opposite transaction (exit order)
                                exit_transaction_type = "SELL" if transaction_type == "BUY" else "BUY"

                                # Set appropriate product type based on broker
                                product_type = "CNC" if order_row.broker == "Zerodha" else "D"
                                exit_response = await place_order(
                                    api=order_api,
                                    instrument_token=instrument_token,
                                    trading_symbol=trading_symbol,
                                    transaction_type=exit_transaction_type,
                                    quantity=quantity,
                                    price=0,  # Market order
                                    order_type="MARKET",
                                    product_type=product_type,
                                    broker=order_row.broker,
                                    db=db,
                                    user_id=user_id
                                )

                                # Handle Upstox response which contains order_ids (plural) as an array
                                if order_row.broker == "Upstox":
                                    if hasattr(exit_response.data, 'order_ids') and exit_response.data.order_ids:
                                        exit_order_id = exit_response.data.order_ids[0]  # Take the first order ID
                                    elif hasattr(exit_response.data, 'order_id'):
                                        exit_order_id = exit_response.data.order_id  # Fallback for single order_id
                                    else:
                                        exit_order_id = "unknown"  # Fallback if no order ID found
                                else:
                                    exit_order_id = exit_response

                                # Update original order with exit details and exit_order_id
                                update_fields = {
                                    "remarks": f"{exit_reason} at {current_price}. Exit order: {exit_order_id}",
                                    "exit_order_id": exit_order_id,  # Track the exit order ID
                                    "order_id": order_id
                                }

                                if "trailing" in exit_reason.lower():
                                    update_fields["trailing_stop_price"] = current_price

                                update_query = text("""
                                    UPDATE orders 
                                    SET remarks = :remarks, exit_order_id = :exit_order_id""" +
                                    (", trailing_stop_price = :trailing_stop_price" if "trailing_stop_price" in update_fields else "") + """
                                    WHERE order_id = :order_id
                                """)

                                await db.execute(update_query, update_fields)

                                logger.info(f"Placed exit order {exit_order_id} for {trading_symbol} - {exit_reason}")

                                # Send notification
                                result = await db.execute(select(User).filter(User.user_id == user_id))
                                user = result.scalars().first()
                                if user:
                                    if transaction_type == "BUY":
                                        profit_loss = (current_price - entry_price) * quantity
                                    else:
                                        profit_loss = (entry_price - current_price) * quantity

                                    await notify(
                                        f"{exit_reason}: {trading_symbol}",
                                        f"Exit order placed at {current_price}. Estimated P&L: Rs.{profit_loss:.2f}",
                                        user.email
                                    )
                            else:
                                logger.error(f"No order API available for user {user_id} to place exit order")

                        except Exception as exit_error:
                            logger.error(f"Error placing exit order for {trading_symbol}: {exit_error}")

                except Exception as order_error:
                    logger.error(f"Error processing order monitoring for order {order_row.order_id}: {order_error}")

            # Commit all database changes
            await db.commit()

        except Exception as e:
            logger.error(f"Error in monitor_trailing_stop_loss_orders: {e}")
            await db.rollback()

    async def sync_order_statuses(self, upstox_api: Optional[Any], zerodha_api: Optional[Any],
                                  db: AsyncSession, user_id: Optional[str] = None) -> bool:
        """Sync order statuses and GTT orders - ENHANCED: Now syncs both regular orders and GTT orders for users with active APIs"""
        try:
            if not (upstox_api or zerodha_api):
                logger.debug("No APIs available for order status sync")
                return True

            # ===== SYNC REGULAR ORDERS =====
            # Build the query with user_id filter if provided - Include PENDING status from place_order
            stmt = select(Order).where(
                func.lower(Order.status).in_([
                    status.lower() for status in [
                        "open", "pending", "PENDING", "trigger pending",
                        "amo req received", "after market order req received"
                    ]
                ])
            )

            # CRITICAL FIX: Add user_id filter to only fetch orders for the current user
            if user_id:
                stmt = stmt.where(Order.user_id == user_id)

            result = await db.execute(stmt)
            orders = result.scalars().all()

            if user_id:
                logger.info(f"Found {len(orders)} open/pending orders for user {user_id}")
            else:
                logger.info(f"Found {len(orders)} open/pending orders (all users)")

            # Sync regular orders
            for order in orders:
                try:
                    broker_response = None
                    order_status = None
                    broker_message = None
                    failure_reason = None
                    
                    if order.broker == "Upstox" and upstox_api:
                        broker_response = upstox_api["order_v2"].get_order_status(order_id=order.order_id)
                        order_status = broker_response.data.status
                        # Extract additional details from Upstox response
                        if hasattr(broker_response.data, 'status_message'):
                            broker_message = broker_response.data.status_message
                        if hasattr(broker_response.data, 'rejection_reason') and broker_response.data.rejection_reason:
                            failure_reason = broker_response.data.rejection_reason
                            
                    elif order.broker == "Zerodha" and zerodha_api:
                        broker_response = zerodha_api.order_history(order_id=order.order_id)
                        latest_order = broker_response[-1] if broker_response else {}
                        order_status = latest_order.get("status", "").lower()
                        broker_message = latest_order.get("status_message")
                        failure_reason = latest_order.get("status_message") if order_status in ["rejected", "cancelled"] else None
                    else:
                        logger.debug(f"Skipping order {order.order_id}: API not available for broker {order.broker}")
                        continue

                    if order_status and order_status.lower() != order.status.lower():
                        old_status = order.status
                        order.status = order_status.lower()
                        
                        # Prepare timestamp and tracking data
                        current_time = datetime.now()
                        update_data = {
                            'order_id': order.order_id,
                            'status': order.status,
                            'status_updated_at': current_time
                        }
                        
                        # Add broker message for all status changes
                        if broker_message:
                            update_data['broker_message'] = broker_message
                        
                        # Add specific event timestamps based on order events
                        # Note: completed/cancelled/rejected timestamps are handled by order_timestamp + status
                        if hasattr(order, 'trailing_activated') and order.trailing_activated and old_status != order.status:
                            if order.status in ["complete", "filled"] and 'trailing_activated_at' not in update_data:
                                update_data['trailing_activated_at'] = current_time
                        
                        # Check for stop loss triggers (you can enhance this logic based on your business rules)
                        if order.status in ["complete", "filled"] and hasattr(order, 'stop_loss') and order.stop_loss:
                            # This would need more sophisticated logic to detect if it was a stop loss trigger
                            # For now, we can add the timestamp when available from broker response
                            pass
                        
                        # Update the order with enhanced tracking
                        update_fields = ', '.join([f"{k} = :{k}" for k in update_data.keys() if k != 'order_id'])
                        update_query = text(f"""
                            UPDATE orders 
                            SET {update_fields}
                            WHERE order_id = :order_id
                        """)
                        
                        await db.execute(update_query, update_data)
                        
                        logger.info(f"Updated order {order.order_id} status from {old_status} to {order.status}")
                        # Broadcast order status update
                        try:
                            await ws_broadcast_event(order.user_id, "order_status_updated", {
                                "scope": "placed",
                                "broker": order.broker,
                                "order_id": order.order_id,
                                "trading_symbol": order.trading_symbol,
                                "status": str(order.status).upper()
                            })
                        except Exception:
                            pass
                        if failure_reason:
                            logger.info(f"Order {order.order_id} failure reason: {failure_reason}")
                        if broker_message:
                            logger.debug(f"Order {order.order_id} broker message: {broker_message}")

                        # ===== SYNC SIP ACTUAL TRADES =====
                        # Update corresponding sip_actual_trades records (only if they exist)
                        try:
                            # First check if the order exists in sip_actual_trades
                            check_sip_trade = text("""
                                SELECT COUNT(*) as count FROM sip_actual_trades 
                                WHERE order_id = :order_id
                            """)
                            
                            check_result = await db.execute(check_sip_trade, {'order_id': order.order_id})
                            sip_trade_exists = check_result.scalar() > 0
                            
                            if sip_trade_exists:
                                # Map order status to execution status
                                if order.status in ["complete", "filled"]:
                                    execution_status = "EXECUTED"
                                elif order.status in ["rejected", "cancelled"]:
                                    execution_status = "FAILED"
                                else:
                                    execution_status = "PENDING"
                                
                                # Prepare SIP update with optimized tracking
                                sip_update_data = {
                                    'execution_status': execution_status,
                                    'order_id': order.order_id
                                }
                                
                                # Add broker message and execution timestamp
                                if broker_message:
                                    sip_update_data['broker_message'] = broker_message
                                
                                # Set execution timestamp: NULL for failed, current_time for executed
                                if execution_status == "EXECUTED":
                                    sip_update_data['order_executed_at'] = current_time
                                elif execution_status == "FAILED":
                                    # NULL execution time indicates failure
                                    sip_update_data['order_executed_at'] = None
                                
                                # Build dynamic query for SIP update
                                sip_update_fields = ', '.join([f"{k} = :{k}" for k in sip_update_data.keys() if k != 'order_id'])
                                update_sip_trade = text(f"""
                                    UPDATE sip_actual_trades 
                                    SET {sip_update_fields}
                                    WHERE order_id = :order_id
                                """)
                                
                                await db.execute(update_sip_trade, sip_update_data)
                                logger.info(f"Updated sip_actual_trades execution_status to {execution_status} for order {order.order_id}")
                                
                                if failure_reason and execution_status == "FAILED":
                                    logger.info(f"SIP trade {order.order_id} failure reason: {failure_reason}")
                                
                                # Update portfolio totals when order is executed
                                if execution_status == "EXECUTED":
                                    await self._update_portfolio_totals_on_execution(order.order_id, db, upstox_api["market_data_v3"], zerodha_api)
                            else:
                                logger.debug(f"Order {order.order_id} not found in sip_actual_trades, skipping SIP update")
                            
                        except Exception as sip_error:
                            logger.warning(f"Could not update sip_actual_trades for order {order.order_id}: {sip_error}")

                        if order.status in ["complete", "rejected", "cancelled"]:
                            result = await db.execute(select(User).filter(User.user_id == order.user_id))
                            user = result.scalars().first()
                            if user:
                                await notify(
                                    f"Order Status Update: {order.order_id}",
                                    f"Order {order.order_id} for {order.trading_symbol} is {order.status}",
                                    user.email
                                )
                except Exception as e:
                    logger.error(f"Error syncing status for order {order.order_id}: {str(e)}")

            # ===== SYNC GTT ORDERS =====
            # Build GTT query with user_id filter if provided
            gtt_stmt = select(GTTOrder).where(GTTOrder.status.in_(["PENDING", "active", "scheduled"]))

            if user_id:
                gtt_stmt = gtt_stmt.where(GTTOrder.user_id == user_id)

            gtt_result = await db.execute(gtt_stmt)
            gtt_orders = gtt_result.scalars().all()

            if user_id:
                logger.info(f"Found {len(gtt_orders)} active GTT orders for user {user_id}")
            else:
                logger.info(f"Found {len(gtt_orders)} active GTT orders (all users)")

            # Sync GTT orders
            for gtt in gtt_orders:
                try:
                    gtt_status_updated = False

                    if gtt.broker == "Zerodha" and zerodha_api:
                        try:
                            # Get GTT status from Zerodha API
                            zerodha_gtt = zerodha_api.get_gtt(trigger_id=gtt.gtt_order_id)
                            api_status = zerodha_gtt.get("status", "").lower()
                            logger.debug(f"Zerodha GTT {gtt.gtt_order_id} status: {zerodha_gtt}")

                            if api_status and api_status != gtt.status.lower():
                                old_status = gtt.status
                                gtt.status = api_status
                                gtt_status_updated = True
                                logger.info(f"Updated GTT {gtt.gtt_order_id} status from {old_status} to {api_status}")

                        except Exception as api_error:
                            logger.debug(f"Could not fetch GTT status from Zerodha API for {gtt.gtt_order_id}: {api_error}")

                    elif gtt.broker == "Upstox" and upstox_api:
                        # For Upstox GTT orders, check if trigger conditions are met
                        try:
                            # Get GTT status from Zerodha API
                            upstox_gtt_raw = upstox_api["order"].get_gtt_order_details(gtt_order_id=gtt.gtt_order_id).data
                            logger.debug(f"Upstox GTT {gtt.gtt_order_id} status: {upstox_gtt_raw}")
                            api_status = None
                            for gtt_order in upstox_gtt_raw:
                                upstox_gtt = gtt_order.to_dict()
                                api_status = upstox_gtt.get("rules", [{"status": ""}])[0].get("status", "").lower()
                                logger.debug(f"Upstox GTT {gtt.gtt_order_id} status: {api_status}")
                                if api_status and api_status != gtt.status.lower():
                                    old_status = gtt.status
                                    gtt.status = api_status
                                    gtt_status_updated = True
                                    logger.info(f"Updated GTT {gtt.gtt_order_id} status from {old_status} to {api_status}")

                        except Exception as api_error:
                            logger.debug(f"Could not fetch GTT status from API for {gtt.gtt_order_id}: {api_error}")

                    # ===== SYNC SIP ACTUAL TRADES FOR GTT ORDERS =====
                    # Update corresponding sip_actual_trades records for GTT orders (only if they exist)
                    if gtt_status_updated:
                        try:
                            # First check if a SIP trade exists for this GTT order id
                            check_sip_trade = text("""
                                SELECT COUNT(*) as count FROM sip_actual_trades 
                                WHERE gtt_order_id = :gtt_order_id
                            """)
                            check_result = await db.execute(check_sip_trade, {'gtt_order_id': gtt.gtt_order_id})
                            sip_trade_exists = check_result.scalar() > 0

                            if sip_trade_exists:
                                # Map GTT status to execution status (normalize case)
                                status_lower = str(gtt.status).lower() if gtt.status is not None else ""
                                if status_lower in ["triggered", "completed"]:
                                    execution_status = "EXECUTED"
                                elif status_lower in ["cancelled"]:
                                    execution_status = "FAILED"
                                else:
                                    execution_status = "PENDING"

                                # Build SIP update data
                                sip_update_data = {
                                    'execution_status': execution_status,
                                    'gtt_order_id': gtt.gtt_order_id
                                }

                                # Set execution timestamp: now for EXECUTED, NULL for FAILED
                                current_time = datetime.now()
                                if execution_status == "EXECUTED":
                                    sip_update_data['order_executed_at'] = current_time
                                elif execution_status == "FAILED":
                                    sip_update_data['order_executed_at'] = None

                                sip_update_fields = ', '.join([f"{k} = :{k}" for k in sip_update_data.keys() if k != 'gtt_order_id'])
                                update_sip_trade = text(f"""
                                    UPDATE sip_actual_trades 
                                    SET {sip_update_fields}
                                    WHERE gtt_order_id = :gtt_order_id
                                """)

                                await db.execute(update_sip_trade, sip_update_data)
                                logger.info(f"Updated sip_actual_trades execution_status to {execution_status} for GTT order {gtt.gtt_order_id}")

                                # Update portfolio totals when GTT order is executed
                                if execution_status == "EXECUTED":
                                    await self._update_portfolio_totals_on_execution(gtt.gtt_order_id, db, upstox_api["market_data_v3"], zerodha_api)
                            else:
                                logger.debug(f"GTT order {gtt.gtt_order_id} not found in sip_actual_trades, skipping SIP update")

                        except Exception as sip_error:
                            logger.warning(f"Could not update sip_actual_trades for GTT order {gtt.gtt_order_id}: {sip_error}")

                        # Broadcast GTT status update
                        try:
                            await ws_broadcast_event(gtt.user_id, "gtt_status_updated", {
                                "scope": "gtt",
                                "broker": gtt.broker,
                                "gtt_order_id": gtt.gtt_order_id,
                                "trading_symbol": gtt.trading_symbol,
                                "status": str(gtt.status).upper()
                            })
                        except Exception:
                            pass

                    # Send notification if status changed
                    if gtt_status_updated and gtt.status in ["triggered", "cancelled", "completed"]:
                        result = await db.execute(select(User).filter(User.user_id == gtt.user_id))
                        user = result.scalars().first()
                        if user:
                            await notify(
                                f"GTT Order Update: {gtt.gtt_order_id}",
                                f"GTT order for {gtt.trading_symbol} status changed to {gtt.status}",
                                user.email
                            )

                except Exception as e:
                    logger.error(f"Error syncing GTT order {gtt.gtt_order_id}: {str(e)}")

            # Log summary
            if orders or gtt_orders:
                logger.info(f"Sync completed for user {user_id if user_id else 'all users'}: {len(orders)} orders, {len(gtt_orders)} GTT orders")
            else:
                logger.debug(f"No orders or GTT orders to sync for user {user_id if user_id else 'all users'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in sync_order_statuses: {str(e)}")
            return False

    async def _update_portfolio_totals_on_execution(self, order_id: str, db: AsyncSession, upstox_api: Optional[Any] = None, kite_api: Optional[Any] = None):
        """Update portfolio totals when an order is executed"""
        try:
            # Get the executed trade details
            trade_query = text("""
                SELECT portfolio_id, symbol, units, amount, price
                FROM sip_actual_trades 
                WHERE (order_id = :order_id OR gtt_order_id = :order_id) AND execution_status = 'EXECUTED'
            """)
            
            trade_result = await db.execute(trade_query, {'order_id': order_id})
            trade = trade_result.fetchone()
            
            if not trade:
                logger.warning(f"No executed trade found for order {order_id}")
                return
            
            portfolio_id, symbol, units, amount, price = trade
            
            # Get portfolio symbols for current value calculation
            portfolio_query = text("""
                SELECT symbols FROM sip_portfolios WHERE portfolio_id = :portfolio_id
            """)
            
            portfolio_result = await db.execute(portfolio_query, {'portfolio_id': portfolio_id})
            portfolio_data = portfolio_result.fetchone()
            
            if not portfolio_data:
                logger.warning(f"Portfolio {portfolio_id} not found")
                return
            
            symbols_json = portfolio_data[0]
            
            # Parse symbols safely
            try:
                if isinstance(symbols_json, str):
                    symbols_data = json.loads(symbols_json)
                elif isinstance(symbols_json, list):
                    symbols_data = symbols_json
                else:
                    logger.error(f"Invalid symbols data type for portfolio {portfolio_id}")
                    return
            except Exception as parse_error:
                logger.error(f"Error parsing symbols for portfolio {portfolio_id}: {parse_error}")
                return
            
            # Calculate current portfolio value using the optimized function
            try:
                current_portfolio_value = await calculate_portfolio_current_value(
                    portfolio_id=portfolio_id,
                    symbols_data=symbols_data,
                    upstox_api=upstox_api,
                    kite_api=kite_api,
                    trading_db=db
                )
            except Exception as value_error:
                logger.warning(f"Error calculating current portfolio value for {portfolio_id}: {value_error}")
                # Fallback: use the executed trade amount
                current_portfolio_value = amount
            
            # Update portfolio totals
            update_query = text("""
                UPDATE sip_portfolios 
                SET total_invested = total_invested + :amount,
                    current_units = current_units + :units,
                    current_value = :current_value,
                    updated_at = :now
                WHERE portfolio_id = :portfolio_id
            """)
            
            await db.execute(update_query, {
                'amount': amount,
                'units': units,
                'current_value': current_portfolio_value,
                'now': datetime.now(),
                'portfolio_id': portfolio_id
            })
            
            logger.info(f"Updated portfolio {portfolio_id} totals: +{amount} invested, +{units} units, current_value={current_portfolio_value}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio totals for order {order_id}: {str(e)}")


    async def _update_order_status(self, order_id: str, status: str, broker: str, db: AsyncSession):
        query = """
            UPDATE orders
            SET status = :status
            WHERE order_id = :order_id AND broker = :broker
        """
        await async_execute_query(db, text(query), {"status": status, "order_id": order_id, "broker": broker})

    async def cancel_all_tasks(self):
        self.running = False
        for task in self.monitor_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*[task for task in self.monitor_tasks if not task.done()], return_exceptions=True)
        self.monitor_tasks = []

    async def delete_gtt_order(self, api: Optional[Any], gtt_id: str, db: AsyncSession):
        try:
            result = await db.execute(select(GTTOrder).filter(GTTOrder.gtt_order_id == gtt_id))
            gtt_order = result.scalars().first()
            if not gtt_order:
                logger.error(f"GTT order {gtt_id} not found in database")
                raise HTTPException(status_code=404, detail=f"GTT order {gtt_id} not found")
            
            # Handle broker-specific cancellation
            if api and gtt_order.broker == "Upstox":
                try:
                    # Upstox SDK: cancel GTT
                    try:
                        body = upstox_client.GttCancelOrderRequest(gtt_order_id=gtt_id)
                        api.cancel_gtt_order(body=body)
                    except Exception:
                        # Some SDK versions may use path param
                        api.cancel_gtt_order(gtt_id)
                    logger.info(f"Upstox GTT order {gtt_id} cancelled successfully")
                except Exception as e:
                    logger.error(f"Error canceling Upstox GTT order {gtt_id}: {str(e)}")
                    raise Exception(f"Failed to cancel Upstox GTT order: {str(e)}")
            elif api and gtt_order.broker == "Zerodha":
                api.delete_gtt(trigger_id=gtt_id)

            # Mark as cancelled in DB instead of deleting
            update_query = """
                UPDATE gtt_orders
                SET status = :status
                WHERE gtt_order_id = :gtt_id
            """
            await async_execute_query(db, text(update_query), {"status": "CANCELLED", "gtt_id": gtt_id})
            result = await db.execute(select(User).filter(User.user_id == gtt_order.user_id))
            user = result.scalars().first()
            if user:
                await notify("GTT Order Cancelled", f"GTT order {gtt_id} cancelled", user.email)
            return {"status": "success", "message": f"GTT order {gtt_id} cancelled"}
        except Exception as e:
            logger.error(f"Error deleting GTT order {gtt_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error deleting GTT order: {str(e)}")

class OrderManager:
    def __init__(self, monitor: OrderMonitor):
        self.order_monitor = monitor
        self.scheduled_order_queue = []
        self.running = True
        self.order_lock = asyncio.Lock()
        self.market_open = date_time(9, 15)
        self.market_close = date_time(15, 30)
        logger.info("OrderManager initialized")

    async def start(self, user_apis=None):
        """CORRECTED: Start OrderManager with proper database session handling"""
        logger.info("OrderManager starting...")
        try:
            # Load scheduled orders first using the correct session pattern
            session_factory = db_manager.get_session_factory('trading_db')
            if not session_factory:
                logger.error("Database session factory not available")
                return

            async with session_factory() as db:
                try:
                    await self._load_scheduled_orders(db)
                    logger.info("Scheduled orders loaded successfully")
                except Exception as load_error:
                    logger.error(f"Error loading scheduled orders: {load_error}")

            # Start the continuous processing loop
            await self._process_scheduled_orders_loop(user_apis)

        except Exception as e:
            logger.error(f"Error starting OrderManager: {e}")

    async def _load_scheduled_orders(self, db: AsyncSession):
        """Load scheduled orders - same as before"""
        try:
            stmt = select(ScheduledOrder).where(ScheduledOrder.status == "PENDING")
            result = await db.execute(stmt)
            scheduled_orders = result.scalars().all()
            self.scheduled_order_queue = [ScheduledOrderSchema.from_orm(order).dict() for order in scheduled_orders]
            logger.info(f"Loaded {len(self.scheduled_order_queue)} scheduled orders")
        except Exception as e:
            logger.error(f"Error in load_scheduled_orders: {str(e)}")
            self.scheduled_order_queue = []

    async def _process_scheduled_orders_loop(self, user_apis: Dict[str, Dict[str, Any]]):
        """CORRECTED: Process scheduled orders with proper session management and dynamic user API loading"""
        logger.info("Starting scheduled order processing loop")
        try:
            while self.running:
                try:
                    # CORRECT WAY: Use session factory directly
                    session_factory = db_manager.get_session_factory('trading_db')
                    if not session_factory:
                        logger.error("Database session factory not available")
                        await asyncio.sleep(60)
                        continue

                    async with session_factory() as db:
                        try:
                            query = """
                                SELECT * FROM scheduled_orders
                                WHERE status = :status
                                ORDER BY schedule_datetime ASC
                            """
                            scheduled_orders = await async_fetch_query(db, text(query), {"status": "PENDING"})
                            now = datetime.now()

                            logger.info(f"Found {len(scheduled_orders)} pending scheduled orders at {now}")

                            for order in scheduled_orders:
                                try:
                                    # Parse schedule_datetime with multiple format support
                                    schedule_datetime = order["schedule_datetime"]
                                    if isinstance(schedule_datetime, str):
                                        # Try multiple datetime formats
                                        try:
                                            schedule_datetime = datetime.strptime(schedule_datetime, "%Y-%m-%d %H:%M:%S")
                                        except ValueError:
                                            try:
                                                schedule_datetime = datetime.strptime(schedule_datetime, "%Y-%m-%dT%H:%M:%S")
                                            except ValueError:
                                                schedule_datetime = datetime.fromisoformat(schedule_datetime.replace('Z', '+00:00'))

                                    logger.info(f"Order {order['scheduled_order_id']}: scheduled for {schedule_datetime}, current time {now}")

                                    # Check if it's time to execute (with some tolerance for execution delay)
                                    if schedule_datetime <= now:
                                        user_id = order["user_id"]
                                        logger.info(f"Attempting to execute scheduled order {order['scheduled_order_id']} for user {user_id}")

                                        # Dynamic API loading - check provided first, then load from DB if needed
                                        upstox_api = user_apis.get(user_id, {}).get("upstox", {}).get("order")
                                        zerodha_api = user_apis.get(user_id, {}).get("zerodha", {}).get("kite")

                                        # If APIs are not available in provided user_apis, try to load them
                                        if not (upstox_api or zerodha_api):
                                            logger.info(f"APIs not found in provided user_apis for user {user_id}, attempting to load from database")
                                            try:
                                                from backend.app.api_manager import initialize_user_apis
                                                user_apis_dict = await initialize_user_apis(user_id, db)
                                                if user_apis_dict:
                                                    # Update the global user_apis dict
                                                    user_apis[user_id] = user_apis_dict
                                                    upstox_api = user_apis_dict.get("upstox", {}).get("order")
                                                    zerodha_api = user_apis_dict.get("zerodha", {}).get("kite")
                                                    logger.info(f"Successfully loaded APIs for user {user_id}")
                                                else:
                                                    logger.error(f"Failed to initialize APIs for user {user_id}")
                                            except Exception as init_error:
                                                logger.error(f"Error initializing APIs for user {user_id}: {init_error}")

                                        api = upstox_api if order["broker"] == "Upstox" else zerodha_api
                                        if not api:
                                            logger.error(f"API not available for user {user_id}, broker {order['broker']}")
                                            # Mark order as failed due to API unavailability
                                            update_query = """
                                                UPDATE scheduled_orders
                                                SET status = 'FAILED', error_message = 'API not available'
                                                WHERE scheduled_order_id = :order_id
                                            """
                                            await async_execute_query(db, text(update_query),
                                                                      {"order_id": order["scheduled_order_id"]})
                                            continue

                                        logger.info(f"Executing scheduled order {order['scheduled_order_id']} with {order['broker']} API")

                                        # Execute the order
                                        response = await place_order(
                                            api=api,
                                            instrument_token=order["instrument_token"],
                                            trading_symbol=order["trading_symbol"],
                                            transaction_type=order["transaction_type"],
                                            quantity=order["quantity"],
                                            price=order["price"],
                                            order_type=order["order_type"],
                                            trigger_price=order["trigger_price"],
                                            is_amo=order["is_amo"],
                                            product_type=order["product_type"],
                                            validity="DAY",
                                            stop_loss=order["stop_loss"],
                                            target=order["target"],
                                            broker=order["broker"],
                                            db=db,
                                            upstox_apis=user_apis.get(user_id, {}).get("upstox", {}),
                                            kite_apis=user_apis.get(user_id, {}).get("zerodha", {}),
                                            user_id=user_id,
                                            order_monitor=self.order_monitor
                                        )

                                        executed_order_id = response.data.order_ids[0] if order["broker"] != 'Zerodha' else response

                                        # Update order status to executed
                                        update_query = """
                                            UPDATE scheduled_orders
                                            SET status = 'EXECUTED', executed_at = :executed_at, executed_order_id = :executed_order_id
                                            WHERE scheduled_order_id = :order_id
                                        """
                                        await async_execute_query(db, text(update_query), {
                                            "order_id": order["scheduled_order_id"],
                                            "executed_at": now,
                                            "executed_order_id": executed_order_id
                                        })

                                        logger.info(f"Successfully executed scheduled order {order['scheduled_order_id']}")

                                        # Send notification to user
                                        result = await db.execute(select(User).filter(User.user_id == user_id))
                                        user = result.scalars().first()
                                        if user:
                                            await notify(
                                                "Scheduled Order Executed",
                                                f"Scheduled order for {order['trading_symbol']} has been executed at {now}",
                                                user.email
                                            )

                                    else:
                                        # Log when orders are waiting to be executed
                                        time_diff = (schedule_datetime - now).total_seconds()
                                        if time_diff > 0:
                                            logger.info(f"Order {order['scheduled_order_id']} waiting {time_diff:.0f} seconds for execution")

                                except Exception as order_error:
                                    logger.error(f"Error processing scheduled order {order.get('scheduled_order_id', 'unknown')}: {str(order_error)}")
                                    # Mark order as failed
                                    try:
                                        update_query = """
                                            UPDATE scheduled_orders
                                            SET status = 'FAILED', error_message = :error_msg
                                            WHERE scheduled_order_id = :order_id
                                        """
                                        await async_execute_query(db, text(update_query), {
                                            "order_id": order.get("scheduled_order_id"),
                                            "error_msg": str(order_error)[:255]  # Limit error message length
                                        })
                                    except Exception as update_error:
                                        logger.error(f"Failed to update order status: {update_error}")

                            # Commit all changes
                            await db.commit()

                        except Exception as processing_error:
                            logger.error(f"Error in scheduled order processing: {processing_error}")
                            await db.rollback()

                except Exception as session_error:
                    logger.error(f"Error with database session in scheduled orders: {session_error}")

                # Wait before next iteration (check every 30 seconds for better responsiveness)
                await asyncio.sleep(30)

        except asyncio.CancelledError:
            logger.info("Scheduled order processing cancelled")
            self.running = False
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _process_scheduled_orders_loop: {str(e)}")
            self.running = False

    async def place_gtt_order(self, api, instrument_token, trading_symbol, transaction_type, quantity,
                              trigger_type, trigger_price, limit_price, last_price,
                              second_trigger_price=None, second_limit_price=None, rules: Optional[List[Dict[str, Any]]] = None,
                              broker="Zerodha", db: AsyncSession = None, user_id: str = "default_user"):
        try:
            # Handle Upstox GTT orders
            if broker == "Upstox":
                # Determine if caller provided full rules (preferred for Upstox)
                provided_rules = rules if isinstance(rules, list) and len(rules) > 0 else None

                # Basic validation for required Upstox GTT inputs
                if not instrument_token:
                    raise ValueError("Instrument key is required for Upstox GTT")
                if quantity <= 0:
                    raise ValueError("Quantity must be greater than 0 for GTT")
                # When rules are provided, do not mandate top-level trigger_price
                if not provided_rules:
                    if not trigger_price or trigger_price <= 0:
                        raise ValueError("Trigger price must be greater than 0 for GTT")
                # For rules-based Upstox OCO, don't require top-level second_trigger_price
                if not provided_rules:
                    if trigger_type in ("two_leg", "OCO") and (second_trigger_price is None or second_trigger_price <= 0):
                        raise ValueError("Second trigger price is required and must be > 0 for OCO GTT")

                # Decide type and rules
                # Normalize type from UI; Upstox expects SINGLE or MULTIPLE
                type_norm = (trigger_type or "single").upper()
                if type_norm == "OCO":
                    type_norm = "MULTIPLE"
                elif type_norm not in ("SINGLE", "MULTIPLE"):
                    type_norm = "SINGLE" if not provided_rules or len(provided_rules) == 1 else "MULTIPLE"

                # If caller provided full rules, use them as-is (validate minimal keys)
                if provided_rules:
                    normalized_rules = []
                    # Upstox typically supports up to 2 rules; trim extras
                    for r in provided_rules[:2]:
                        strat = str(r.get("strategy", "ENTRY")).upper()
                        # Map UI STOP_LOSS to API STOPLOSS
                        if strat == "STOP_LOSS":
                            strat = "STOPLOSS"
                        if strat not in ("ENTRY", "STOPLOSS", "TARGET"):
                            raise ValueError(f"Invalid rule strategy: {strat}")
                        ttype = str(r.get("trigger_type", "ABOVE")).upper()
                        # Upstox supports IMMEDIATE in addition to ABOVE/BELOW
                        if ttype not in ("ABOVE", "BELOW", "IMMEDIATE"):
                            raise ValueError(f"Invalid rule trigger_type: {ttype}")
                        tprice = float(r.get("trigger_price"))
                        if tprice <= 0:
                            raise ValueError("trigger_price must be > 0 for each rule")
                        rule_obj = {
                            "strategy": strat,
                            "trigger_type": ttype,
                            "trigger_price": tprice,
                        }
                        normalized_rules.append(rule_obj)
                    rules_to_send = normalized_rules
                else:
                    # Backward-compatible build when no rules array is provided
                    rules_to_send = [{
                        "strategy": "ENTRY",
                        "trigger_type": "ABOVE" if transaction_type == "BUY" else "BELOW",
                        "trigger_price": float(trigger_price),
                    }]
                    if type_norm == "MULTIPLE" and second_trigger_price is not None:
                        rules_to_send.append({
                            "strategy": "TARGET" if transaction_type == "BUY" else "STOPLOSS",
                            "trigger_type": "ABOVE" if transaction_type == "BUY" else "BELOW",
                            "trigger_price": float(second_trigger_price),
                        })

                body = {
                    "type": type_norm,
                    "quantity": int(quantity),
                    "product": "D",
                    "rules": rules_to_send,
                    "instrument_token": instrument_token,
                    "transaction_type": transaction_type,
                }

                logger.info(f"Placing Upstox GTT via SDK: {body}")
                # Use SDK (OrderApiV3) - synchronous SDK
                response = api.place_gtt_order(body=body)
                logger.debug(f"Upstox GTT Order placed successfully - {response}")
                
                gtt_id = response.data.gtt_order_ids[0]

                logger.info(f"Upstox GTT Order placed successfully - {gtt_id}")            

                gtt_id = str(uuid.uuid4())
                query = """
                    INSERT INTO gtt_orders (
                        gtt_order_id, instrument_token, trading_symbol, transaction_type, quantity,
                        trigger_type, trigger_price, limit_price, last_price, second_trigger_price, second_limit_price,
                        status, broker, created_at, user_id
                    ) VALUES (
                        :gtt_id, :instrument_token, :trading_symbol, :transaction_type, :quantity,
                        :trigger_type, :trigger_price, :limit_price, :last_price, :second_trigger_price,
                        :second_limit_price, :status, :broker, :created_at, :user_id
                    )
                """
                await async_execute_query(db, text(query), {
                    "gtt_id": gtt_id,
                    "instrument_token": instrument_token,
                    "trading_symbol": trading_symbol,
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "trigger_type": trigger_type,
                    "trigger_price": trigger_price,
                    "limit_price": limit_price,
                    "last_price": last_price,
                    "second_trigger_price": second_trigger_price,
                    "second_limit_price": second_limit_price,
                    "status": "PENDING",
                    "broker": broker,
                    "created_at": datetime.now(),
                    "user_id": user_id
                })
                result = await db.execute(select(User).filter(User.user_id == user_id))
                user = result.scalars().first()
                if user:
                    await notify("GTT Order Placed", f"GTT order {gtt_id} for {trading_symbol} placed", user.email)
                return {"status": "success", "gtt_id": gtt_id}
            elif broker == "Zerodha":
                # Normalize trigger type for Kite Connect
                ttype_in = (trigger_type or "single").strip().lower()
                kite_trigger_type = "single"
                if ttype_in in ("two-leg", "two_leg", "oco", "o-c-o"):
                    kite_trigger_type = "two-leg"

                condition = {
                    "exchange": "NSE",
                    "tradingsymbol": trading_symbol,
                    "last_price": last_price
                }
                if kite_trigger_type == "single":
                    condition["trigger_values"] = [trigger_price]
                    orders = [{
                        "exchange": "NSE",
                        "tradingsymbol": trading_symbol,
                        "product": "CNC",
                        "order_type": "LIMIT",
                        "transaction_type": transaction_type,
                        "quantity": quantity,
                        "price": limit_price
                    }]
                else:
                    condition["trigger_values"] = [trigger_price, second_trigger_price]
                    orders = [
                        {
                            "exchange": "NSE",
                            "tradingsymbol": trading_symbol,
                            "product": "CNC",
                            "order_type": "LIMIT",
                            "transaction_type": transaction_type,
                            "quantity": quantity,
                            "price": limit_price
                        },
                        {
                            "exchange": "NSE",
                            "tradingsymbol": trading_symbol,
                            "product": "CNC",
                            "order_type": "LIMIT",
                            "transaction_type": transaction_type,
                            "quantity": quantity,
                            "price": second_limit_price
                        }
                    ]
                
                logger.debug(f"Placing Zerodha GTT order - condition: {condition}\n Orders: {orders} \n Trigger type: {kite_trigger_type}")
                try:
                    response = api.place_gtt(
                    
                    trigger_type=kite_trigger_type,
                    tradingsymbol=trading_symbol,
                    exchange="NSE",
                    trigger_values=condition["trigger_values"],
                    last_price=condition["last_price"],
                    orders=orders
                )
                except Exception as ze:
                    # Log more details and rethrow for caller
                    logger.error(f"Zerodha place_gtt error for {trading_symbol}: {ze!r}")
                    raise
                logger.info(f"GTT Order placed successfully - {response}")
                gtt_id = str(response.get("trigger_id"))

            # Derive gtt_type if possible
            derived_gtt_type = "SINGLE"
            try:
                if rules and isinstance(rules, list) and len(rules) > 1:
                    derived_gtt_type = "MULTIPLE"
                elif trigger_type and str(trigger_type).lower() not in ("single", "two_leg"):
                    # Map unknowns conservatively
                    derived_gtt_type = "SINGLE"
                elif trigger_type and str(trigger_type).lower() == "two_leg":
                    derived_gtt_type = "MULTIPLE"
            except Exception:
                derived_gtt_type = "SINGLE"

            query = """
                INSERT INTO gtt_orders (
                    gtt_order_id, instrument_token, trading_symbol, transaction_type, quantity,
                    trigger_type, trigger_price, limit_price, last_price, second_trigger_price, second_limit_price,
                    status, broker, created_at, user_id, gtt_type, rules
                ) VALUES (
                    :gtt_id, :instrument_token, :trading_symbol, :transaction_type, :quantity,
                    :trigger_type, :trigger_price, :limit_price, :last_price, :second_trigger_price,
                    :second_limit_price, :status, :broker, :created_at, :user_id, :gtt_type, :rules
                )
            """
            await async_execute_query(db, text(query), {
                "gtt_id": gtt_id,
                "instrument_token": instrument_token,
                "trading_symbol": trading_symbol,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "trigger_type": trigger_type,
                "trigger_price": trigger_price,
                "limit_price": limit_price,
                "last_price": last_price,
                "second_trigger_price": second_trigger_price,
                "second_limit_price": second_limit_price,
                "status": "active",
                "broker": broker,
                "created_at": datetime.now(),
                "user_id": user_id,
                "gtt_type": derived_gtt_type,
                "rules": json.dumps(rules) if rules is not None else None
            })
            result = await db.execute(select(User).filter(User.user_id == user_id))
            user = result.scalars().first()
            if user:
                await notify("GTT Order Placed", f"GTT order {gtt_id} for {trading_symbol} placed", user.email)
            return {"status": "success", "gtt_id": gtt_id}
        except Exception as e:
            logger.error(f"Error placing GTT order for {trading_symbol}: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def modify_gtt_order(self, api, gtt_id: str, trigger_type: str, trigger_price: float, limit_price: float,
                               last_price: float, quantity: int, second_trigger_price: Optional[float] = None,
                               second_limit_price: Optional[float] = None, rules: Optional[List[Dict[str, Any]]] = None,
                               db: AsyncSession = None):
        try:
            stmt = select(GTTOrder).where(GTTOrder.gtt_order_id == gtt_id)
            result = await db.execute(stmt)
            gtt_order = result.scalars().first()
            if not gtt_order:
                raise ValueError(f"GTT order {gtt_id} not found")
            
            # Handle Upstox GTT modification
            if gtt_order.broker == "Upstox":
                try:
                    # Enhanced Upstox GTT modification based on API documentation
                    provided_rules = rules if isinstance(rules, list) and len(rules) > 0 else None
                    
                    # Determine GTT type based on trigger_type and rules
                    type_norm = (trigger_type or "SINGLE").upper()
                    if type_norm == "OCO":
                        type_norm = "MULTIPLE"
                    elif type_norm not in ("SINGLE", "MULTIPLE"):
                        type_norm = "SINGLE" if not provided_rules or len(provided_rules) == 1 else "MULTIPLE"

                    # Build rules array for Upstox API
                    if provided_rules:
                        # Use provided rules directly (preferred method)
                        normalized_rules = []
                        for r in provided_rules[:3]:  # Upstox supports max 3 rules
                            strat = str(r.get("strategy", "ENTRY")).upper()
                            # Map UI STOP_LOSS to API STOPLOSS
                            if strat == "STOP_LOSS":
                                strat = "STOPLOSS"
                            if strat not in ("ENTRY", "STOPLOSS", "TARGET"):
                                raise ValueError(f"Invalid rule strategy: {strat}")
                            
                            ttype = str(r.get("trigger_type", "ABOVE")).upper()
                            # Upstox supports ABOVE, BELOW, IMMEDIATE
                            if ttype not in ("ABOVE", "BELOW", "IMMEDIATE"):
                                raise ValueError(f"Invalid rule trigger_type: {ttype}")
                            
                            tprice = float(r.get("trigger_price"))
                            if tprice <= 0:
                                raise ValueError("trigger_price must be > 0 for each rule")
                            
                            rule_obj = {
                                "strategy": strat,
                                "trigger_type": ttype,
                                "trigger_price": tprice,
                            }
                            
                            # Add trailing_gap for STOPLOSS strategy if specified
                            if strat == "STOPLOSS" and r.get("trailing_gap"):
                                rule_obj["trailing_gap"] = float(r.get("trailing_gap"))
                            
                            normalized_rules.append(rule_obj)
                        rules_to_send = normalized_rules
                    else:
                        # Backward-compatible build when no rules array is provided
                        rules_to_send = [{
                            "strategy": "ENTRY",
                            "trigger_type": "ABOVE" if gtt_order.transaction_type == "BUY" else "BELOW",
                            "trigger_price": float(trigger_price)
                        }]
                        
                        if type_norm == "MULTIPLE" and second_trigger_price:
                            rules_to_send.append({
                                "strategy": "TARGET" if gtt_order.transaction_type == "BUY" else "STOPLOSS",
                                "trigger_type": "ABOVE" if gtt_order.transaction_type == "BUY" else "BELOW",
                                "trigger_price": float(second_trigger_price)
                            })

                    # Validate rules according to Upstox API requirements
                    if not rules_to_send:
                        raise ValueError("At least one rule is required for GTT order")
                    
                    if type_norm == "SINGLE" and len(rules_to_send) != 1:
                        raise ValueError("SINGLE type GTT must have exactly one rule")
                    
                    if type_norm == "MULTIPLE" and (len(rules_to_send) < 2 or len(rules_to_send) > 3):
                        raise ValueError("MULTIPLE type GTT must have 2-3 rules")
                    
                    # Ensure ENTRY strategy is present
                    entry_rules = [r for r in rules_to_send if r["strategy"] == "ENTRY"]
                    if not entry_rules:
                        raise ValueError("ENTRY strategy is required for GTT order")
                    
                    # Check for duplicate strategies
                    strategies = [r["strategy"] for r in rules_to_send]
                    if len(strategies) != len(set(strategies)):
                        raise ValueError("Duplicate strategies are not allowed in GTT rules")

                    # Build request body according to Upstox API documentation
                    body = {
                        "type": type_norm,
                        "quantity": int(quantity),
                        "rules": rules_to_send,
                        "gtt_order_id": gtt_id
                    }

                    logger.info(f"Modifying Upstox GTT order {gtt_id} with body: {body}")
                    
                    # Call Upstox API to modify GTT order
                    response = api.modify_gtt_order(body=body)
                    logger.info(f"Upstox GTT order {gtt_id} modified successfully: {response}")
                    
                except Exception as e:
                    logger.error(f"Error modifying Upstox GTT order {gtt_id}: {str(e)}")
                    raise Exception(f"Failed to modify Upstox GTT order: {str(e)}")
            
            # Handle Zerodha GTT modification  
            elif gtt_order.broker == "Zerodha":
                condition = {
                    "exchange": "NSE",
                    "tradingsymbol": gtt_order.trading_symbol,
                    "last_price": last_price
                }
                if trigger_type == "single":
                    condition["trigger_values"] = [trigger_price]
                    orders = [{
                        "exchange": "NSE",
                        "tradingsymbol": gtt_order.trading_symbol,
                        "product": "CNC",
                        "order_type": "LIMIT",
                        "transaction_type": gtt_order.transaction_type,
                        "quantity": quantity,
                        "price": limit_price
                    }]
                else:
                    condition["trigger_values"] = [trigger_price, second_trigger_price]
                    orders = [
                        {
                            "exchange": "NSE",
                            "tradingsymbol": gtt_order.trading_symbol,
                            "product": "CNC",
                            "order_type": "LIMIT",
                            "transaction_type": gtt_order.transaction_type,
                            "quantity": quantity,
                            "price": limit_price
                        },
                        {
                            "exchange": "NSE",
                            "tradingsymbol": gtt_order.trading_symbol,
                            "product": "CNC",
                            "order_type": "LIMIT",
                            "transaction_type": gtt_order.transaction_type,
                            "quantity": quantity,
                            "price": second_limit_price
                        }
                    ]
                api.modify_gtt(
                    trigger_id=gtt_id,
                    trigger_type=trigger_type,
                    tradingsymbol=gtt_order.trading_symbol,
                    exchange="NSE",
                    trigger_values=condition["trigger_values"],
                    last_price=condition["last_price"],
                    orders=orders
                )
            
            # Update database with modified values (including optional rules and gtt_type)
            # Compute gtt_type if rules suggest multi-leg
            derived_gtt_type = None
            try:
                if rules is not None:
                    derived_gtt_type = "MULTIPLE" if isinstance(rules, list) and len(rules) > 1 else "SINGLE"
                elif trigger_type is not None:
                    derived_gtt_type = "MULTIPLE" if str(trigger_type).lower() == "two_leg" else "SINGLE"
            except Exception:
                derived_gtt_type = None

            set_clauses = [
                "trigger_type = :trigger_type",
                "trigger_price = :trigger_price",
                "limit_price = :limit_price",
                "second_trigger_price = :second_trigger_price",
                "second_limit_price = :second_limit_price",
                "quantity = :quantity",
                "last_price = :last_price"
            ]
            if rules is not None:
                set_clauses.append("rules = :rules")
            if derived_gtt_type is not None:
                set_clauses.append("gtt_type = :gtt_type")

            query = f"""
                UPDATE gtt_orders
                SET {', '.join(set_clauses)}
                WHERE gtt_order_id = :gtt_id
            """
            params = {
                "gtt_id": gtt_id,
                "trigger_type": trigger_type,
                "trigger_price": trigger_price,
                "limit_price": limit_price,
                "second_trigger_price": second_trigger_price,
                "second_limit_price": second_limit_price,
                "quantity": quantity,
                "last_price": last_price
            }
            if rules is not None:
                params["rules"] = json.dumps(rules)  # Convert list to JSON string
            if derived_gtt_type is not None:
                params["gtt_type"] = derived_gtt_type

            await async_execute_query(db, text(query), params)
            
            result = await db.execute(select(User).filter(User.user_id == gtt_order.user_id))
            user = result.scalars().first()
            if user:
                await notify("GTT Order Modified", f"GTT order {gtt_id} modified", user.email)
            return {"status": "success", "gtt_id": gtt_id}
        except Exception as e:
            logger.error(f"Error modifying GTT order {gtt_id}: {str(e)}")
            return {"status": "error", "message": str(e)}

@retry(stop=stop_after_attempt(3), retry=retry_if_not_exception_type(TokenExpiredError))
async def init_upstox_api(db: AsyncSession, user_id: str, auth_code: Optional[str] = None) -> Dict[str, Any]:
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        upstox_apis = {"order": None, "portfolio": None, "market_data": None, "user": None}
        if user.upstox_access_token and user.upstox_access_token_expiry and datetime.now() < user.upstox_access_token_expiry:
            logger.info(f"Upstox access token still valid for user {user_id}, expires at {user.upstox_access_token_expiry}")
            config = upstox_client.Configuration()
            config.access_token = user.upstox_access_token
            api_client = upstox_client.ApiClient(config)
            upstox_apis = {
                "order": upstox_client.OrderApiV3(api_client),
                "order_v2": upstox_client.OrderApi(api_client),
                "portfolio": upstox_client.PortfolioApi(api_client),
                "market_data": upstox_client.MarketQuoteApi(api_client),
                "market_data_v3": upstox_client.MarketQuoteV3Api(api_client),
                "user": upstox_client.UserApi(api_client),
                "history": upstox_client.HistoryV3Api(api_client),
                "access_token": user.upstox_access_token
            }
        else:
            logger.info(f"Upstox access token expired or missing for user {user_id}")
            return {"user": None, "order": None, "portfolio": None, "market_data": None, "history": None}
        logger.info(f"Upstox APIs initialized for user {user_id}")
        return upstox_apis
    except Exception as e:
        logger.error(f"Error initializing Upstox APIs for user {user_id}: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), retry=retry_if_not_exception_type(TokenExpiredError))
async def init_zerodha_api(db: AsyncSession, user_id: str, request_token: Optional[str] = None) -> Dict[str, Any]:
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        kite_apis = {"kite": None}
        if user.zerodha_access_token and user.zerodha_access_token_expiry and datetime.now() < user.zerodha_access_token_expiry:
            logger.info(f"Zerodha access token still valid for user {user_id}, expires at {user.zerodha_access_token_expiry}")
            kite = KiteConnect(api_key=user.zerodha_api_key)
            kite.set_access_token(user.zerodha_access_token)
            kite_apis = {"kite": kite}
        else:
            logger.info(f"Zerodha access token expired or missing for user {user_id}")
            return {"kite": None}
        logger.info(f"Zerodha APIs initialized for user {user_id}")
        return kite_apis
    except Exception as e:
        logger.error(f"Error initializing Zerodha APIs for user {user_id}: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_upstox_access_token(db: AsyncSession, user_id: str, auth_code: str) -> Optional[str]:
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        if not all([user.upstox_api_key, user.upstox_api_secret]):
            logger.error(f"Missing Upstox API credentials for user {user_id}")
            return None
        redirect_uri = "https://api.upstox.com/v2/login"
        url = "https://api.upstox.com/v2/login/authorization/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": user.upstox_api_key,
            "client_secret": user.upstox_api_secret,
            "redirect_uri": redirect_uri
        }
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            access_token = response.json()["access_token"]
            expiry_time = get_next_expiry_time(UPSTOX_TOKEN_EXPIRY_TIME)
            user.upstox_access_token = access_token
            user.upstox_access_token_expiry = expiry_time
            await db.commit()
            logger.info(f"Upstox access token fetched for user {user_id}")
            await notify("Upstox Authentication", "Successfully authenticated with Upstox", user.email)
            return access_token
        else:
            logger.error(f"Failed to fetch Upstox access token: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error fetching Upstox access token for user {user_id}: {str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_zerodha_access_token(db: AsyncSession, user_id: str, request_token: str) -> Optional[str]:
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        if not all([user.zerodha_api_key, user.zerodha_api_secret]):
            logger.error(f"Missing Zerodha API credentials for user {user_id}")
            return None
        kite = KiteConnect(api_key=user.zerodha_api_key)
        response = kite.generate_session(request_token, user.zerodha_api_secret)
        access_token = response['access_token']
        expiry_time = get_next_expiry_time(ZERODHA_TOKEN_EXPIRY_TIME)
        if access_token:
            user.zerodha_access_token = access_token
            user.zerodha_access_token_expiry = expiry_time
            await db.commit()
            logger.info(f"Zerodha access token fetched for user {user_id}")
            await notify("Zerodha Authentication", "Successfully authenticated with Zerodha", user.email)
            return access_token
        else:
            logger.error(f"Failed to fetch Zerodha access token: {response}")
            return None
    except Exception as e:
        logger.error(f"Error fetching Zerodha access token for user {user_id}: {str(e)}")
        return None

async def place_order(api, instrument_token, trading_symbol, transaction_type, quantity, price=0, order_type="MARKET",
                      trigger_price=0, is_amo=False, product_type=None, validity='DAY', stop_loss=None, target=None,
                      broker="Upstox", db: AsyncSession = None, upstox_apis=None, kite_apis=None,
                      user_id: str = "default_user", order_monitor=None, is_trailing_stop_loss=False,
                      trailing_stop_loss_percent=None, trail_start_target_percent=None):
    try:
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if price < 0 or trigger_price < 0:
            raise ValueError("Price and trigger price cannot be negative")

        # Set default product type based on broker if not provided
        if product_type is None:
            product_type = "CNC" if broker == "Zerodha" else "D"

        # Validate trailing stop loss parameters
        if is_trailing_stop_loss:
            if not trailing_stop_loss_percent or trailing_stop_loss_percent <= 0:
                raise ValueError("Trailing stop loss percentage must be greater than 0")
            if not trail_start_target_percent or trail_start_target_percent <= 0:
                raise ValueError("Trail start target percentage must be greater than 0")

        # CRITICAL SECURITY FIX: Wrap order placement and database save in transaction
        primary_order_id = None
        response = None

        # Check if transaction is already active, if not start a new one
        transaction_started_here = False
        if not db.in_transaction():
            await db.begin()
            transaction_started_here = True

        try:
            # Step 1: Place order with broker
            if broker == "Upstox":
                order = upstox_client.PlaceOrderV3Request(
                    quantity=quantity,
                    product=product_type,
                    validity=validity,
                    price=price if order_type in ["LIMIT", "SL"] else 0,
                    tag="API Order",
                    instrument_token=instrument_token,
                    order_type=order_type,
                    transaction_type=transaction_type,
                    disclosed_quantity=0,
                    trigger_price=trigger_price if order_type in ["SL", "SL-M"] else 0,
                    is_amo=is_amo,
                    slice=True
                )
                response = api.place_order(order)
                logger.debug(f"Placed order: {response}")

                # Handle Upstox response which contains order_ids (plural) as an array
                if hasattr(response.data, 'order_ids') and response.data.order_ids:
                    primary_order_id = response.data.order_ids[0]  # Take the first order ID
                elif hasattr(response.data, 'order_id'):
                    primary_order_id = response.data.order_id  # Fallback for single order_id
                else:
                    raise ValueError("No order ID found in Upstox response")
            else:
                zerodha_validity = "DAY" if validity == "DAY" else "IOC"
                order_params = {
                    "tradingsymbol": trading_symbol,
                    "exchange": "NSE",
                    "transaction_type": transaction_type,
                    "order_type": order_type,
                    "quantity": quantity,
                    "product": product_type,
                    "validity": zerodha_validity,
                    "price": price if order_type in ["LIMIT", "SL"] else 0,
                    "trigger_price": trigger_price if order_type in ["SL", "SL-M"] else 0,
                    "tag": "API Order"
                }
                response = api.place_order(
                    variety=api.VARIETY_REGULAR if not is_amo else api.VARIETY_AMO,
                    **order_params
                )
                primary_order_id = response

            # Step 2: If broker order succeeds, prepare database record
            if not primary_order_id:
                raise ValueError("Order placement failed - no order ID received")

            order_data_dict = {
                "order_id": primary_order_id,
                "broker": broker,
                "trading_symbol": trading_symbol,
                "instrument_token": instrument_token,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "trigger_price": trigger_price,
                "product_type": product_type,
                "status": "PENDING",
                "remarks": "Order placed via API",
                "order_timestamp": datetime.now(),
                "user_id": user_id,
                "stop_loss": stop_loss if stop_loss is not None else 0,
                "target": target if target is not None else 0,
                "is_amo": is_amo
            }

            # Add trailing stop loss information if enabled
            if is_trailing_stop_loss:
                order_data_dict.update({
                    "is_trailing_stop_loss": True,
                    "trailing_stop_loss_percent": trailing_stop_loss_percent,
                    "trail_start_target_percent": trail_start_target_percent,
                    "trailing_activated": False,
                    "highest_price_achieved": None
                })
            else:
                order_data_dict.update({
                    "is_trailing_stop_loss": False,
                    "trailing_stop_loss_percent": None,
                    "trail_start_target_percent": None,
                    "trailing_activated": None,
                    "highest_price_achieved": None
                })

            # Step 3: Save to database within the same transaction
            order_data = pd.DataFrame([order_data_dict])
            await load_sql_data(order_data, "orders", load_type="append", index_required=False, db=db)

            # Commit transaction if we started it here
            if transaction_started_here:
                await db.commit()

            logger.info(f"Order successfully placed and saved: {primary_order_id} for {trading_symbol}, user {user_id}")

        except Exception as db_error:
            logger.error(f"Transaction failed - order {primary_order_id} might need manual cleanup: {db_error}")
            # Rollback transaction if we started it here
            if transaction_started_here:
                await db.rollback()
            # But we need to consider if the broker order was already placed
            if primary_order_id:
                logger.critical(f"CRITICAL: Order {primary_order_id} was placed with {broker} but database save failed. Manual intervention required!")
                # In production, you might want to:
                # 1. Send alert to admin
                # 2. Attempt order cancellation with broker
                # 3. Log to special error table for manual processing
            raise db_error

        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if user:
            order_type_msg = "Trailing Stop Loss Order" if is_trailing_stop_loss else "Regular Order"
            await notify(f"{order_type_msg} Placed: {transaction_type} for {trading_symbol}",
                         f"Order ID: {primary_order_id}, Quantity: {quantity}, Price: {price}", user.email)
        logger.info(f"Order placed: {primary_order_id} for {trading_symbol}, user {user_id}, trailing_stop_loss: {is_trailing_stop_loss}")

        # NOTE: Removed individual order monitoring since monitor_trailing_stop_loss_orders now handles all monitoring
        # The centralized monitoring function runs every 5 minutes and handles:
        # - Regular stop loss/target monitoring
        # - Trailing stop loss monitoring
        # - Better performance through batched processing

        return response
    except ValueError as ve:
        logger.error(f"Validation error placing {broker} order for {trading_symbol}: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error placing {broker} order for {trading_symbol}, user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")

async def modify_order(api, order_id: str, quantity: Optional[int] = None, order_type: Optional[str] = None,
                       price: Optional[float] = None, trigger_price: Optional[float] = None,
                       validity: Optional[str] = "DAY", broker: str = "Upstox", db: AsyncSession = None):
    try:
        stmt = select(Order).where(Order.order_id == order_id, Order.broker == broker)
        result = await db.execute(stmt)
        order = result.scalars().first()
        if not order:
            raise ValueError(f"Order {order_id} not found")
        modified_params = {
            "quantity": quantity if quantity is not None else order.quantity,
            "order_type": order_type if order_type is not None else order.order_type,
            "price": price if price is not None else order.price,
            "trigger_price": trigger_price if trigger_price is not None else order.trigger_price,
            "validity": validity
        }
        if broker == "Upstox":
            modify_request = upstox_client.ModifyOrderRequest(
                quantity=modified_params["quantity"],
                order_type=modified_params["order_type"],
                price=modified_params["price"],
                trigger_price=modified_params["trigger_price"],
                validity=modified_params["validity"],
                order_id=order_id  # Add order_id to the request
            )
            response = api.modify_order(modify_request)  # Pass only the request object
        else:
            zerodha_validity = "DAY" if modified_params["validity"] == "DAY" else "IOC"
            response = api.modify_order(
                variety=api.VARIETY_REGULAR if not order.is_amo else api.VARIETY_AMO,
                order_id=order_id,
                quantity=modified_params["quantity"],
                order_type=modified_params["order_type"],
                price=modified_params["price"] if modified_params["order_type"] in ["LIMIT", "SL"] else 0,
                trigger_price=modified_params["trigger_price"] if modified_params["order_type"] in ["SL", "SL-M"] else 0,
                validity=zerodha_validity
            )
        logger.info(f"Modified order: {response}")
        query = """
            UPDATE orders
            SET quantity = :quantity, order_type = :order_type, price = :price, trigger_price = :trigger_price
            WHERE order_id = :order_id AND broker = :broker
        """
        await async_execute_query(db, text(query), {
            "order_id": order_id,
            "broker": broker,
            "quantity": modified_params["quantity"],
            "order_type": modified_params["order_type"],
            "price": modified_params["price"],
            "trigger_price": modified_params["trigger_price"]
        })
        result = await db.execute(select(User).filter(User.user_id == order.user_id))
        user = result.scalars().first()
        if user:
            await notify(f"Order Modified: {order_id}",
                         f"Updated quantity: {quantity}, order_type: {order_type}", user.email)
        return {"status": "success", "order_id": order_id}
    except Exception as e:
        logger.error(f"Error modifying order {order_id}: {str(e)}")
        raise

async def get_order_book(upstox_api, kite_api):
    try:
        orders = []
        if upstox_api:
            upstox_orders = upstox_api.get_order_book().data
            for order in upstox_orders:
                order_dict = order.to_dict()
                orders.append({
                    "Broker": "Upstox",
                    "OrderID": order_dict.get("order_id", ""),
                    "Symbol": order_dict.get("trading_symbol", ""),
                    "Exchange": order_dict.get("exchange", ""),
                    "TransType": order_dict.get("transaction_type", ""),
                    "OrderType": order_dict.get("order_type", ""),
                    "Product": order_dict.get("product", ""),
                    "Quantity": order_dict.get("quantity", 0),
                    "Status": order_dict.get("status", ""),
                    "Price": order_dict.get("price", 0),
                    "TriggerPrice": order_dict.get("trigger_price", 0),
                    "AvgPrice": order_dict.get("average_price", 0),
                    "FilledQty": order_dict.get("filled_quantity", 0),
                    "OrderTime": order_dict.get("order_timestamp", ""),
                    "Remarks": order_dict.get("status_message", "")
                })
        if kite_api:
            zerodha_orders = kite_api.orders()
            for order in zerodha_orders:
                orders.append({
                    "Broker": "Zerodha",
                    "OrderID": order.get("order_id", ""),
                    "Symbol": order.get("tradingsymbol", ""),
                    "Exchange": order.get("exchange", ""),
                    "TransType": order.get("transaction_type", ""),
                    "OrderType": order.get("order_type", ""),
                    "Product": order.get("product", ""),
                    "Quantity": order.get("quantity", 0),
                    "Status": order.get("status", ""),
                    "Price": order.get("price", 0),
                    "TriggerPrice": order.get("trigger_price", 0),
                    "AvgPrice": order.get("average_price", 0),
                    "FilledQty": order.get("filled_quantity", 0),
                    "OrderTime": order.get("order_timestamp", ""),
                    "Remarks": order.get("status_message", "")
                })
        orders_df = pd.DataFrame(orders)
        return orders_df
    except Exception as e:
        logger.error(f"Error fetching order book: {str(e)}")
        raise

async def get_positions(upstox_api, zerodha_api):
    try:
        positions = []
        if upstox_api:
            upstox_positions = upstox_api.get_positions(api_version="v2").data
            for pos in upstox_positions:
                pos_dict = pos.to_dict()
                positions.append({
                    "Broker": "Upstox",
                    "Symbol": pos_dict.get("trading_symbol", ""),
                    "Exchange": pos_dict.get("exchange", ""),
                    "Product": pos_dict.get("product", ""),
                    "Quantity": pos_dict.get("quantity", 0),
                    "AvgPrice": pos_dict.get("average_price", 0),
                    "LastPrice": pos_dict.get("last_price", 0),
                    "PnL": pos_dict.get("pnl", 0),
                    "InstrumentToken": pos_dict.get("instrument_token", "")
                })
        if zerodha_api:
            zerodha_positions = zerodha_api.positions().get("net", [])
            for pos in zerodha_positions:
                positions.append({
                    "Broker": "Zerodha",
                    "Symbol": pos.get("tradingsymbol", ""),
                    "Exchange": pos.get("exchange", ""),
                    "Product": pos.get("product", ""),
                    "Quantity": pos.get("net_quantity", 0),
                    "AvgPrice": pos.get("average_price", 0),
                    "LastPrice": pos.get("last_price", 0),
                    "PnL": pos.get("pnl", 0),
                    "InstrumentToken": f"{pos.get('exchange')}|{pos.get('tradingsymbol')}"
                })
        positions_df = pd.DataFrame(positions)
        return positions_df
    except Exception as e:
        logger.error(f"Error fetching positions: {str(e)}")
        raise

async def get_portfolio(upstox_api, zerodha_api):
    try:
        holdings = []
        if upstox_api:
            upstox_holdings = upstox_api.get_holdings(api_version="v2").data
            for holding in upstox_holdings:
                holding_dict = holding.to_dict()
                holdings.append({
                    "Broker": "Upstox",
                    "Symbol": holding_dict.get("trading_symbol", ""),
                    "Exchange": holding_dict.get("exchange", ""),
                    "Quantity": holding_dict.get("quantity", 0),
                    "LastPrice": holding_dict.get("last_price", 0),
                    "AvgPrice": holding_dict.get("average_price", 0),
                    "PnL": holding_dict.get("pnl", 0),
                    "DayChange": holding_dict.get("day_change", 0) * holding_dict.get("quantity", 0),
                    "DayChangePct": holding_dict.get("day_change_percentage", 0),
                })
        if zerodha_api:
            zerodha_holdings = zerodha_api.holdings()
            for holding in zerodha_holdings:
                holdings.append({
                    "Broker": "Zerodha",
                    "Symbol": holding.get("tradingsymbol", ""),
                    "Exchange": holding.get("exchange", ""),
                    "Product": holding.get("product", ""),
                    "Quantity": holding.get("quantity", 0),
                    "LastPrice": holding.get("last_price", 0),
                    "AvgPrice": holding.get("average_price", 0),
                    "PnL": holding.get("pnl", 0),
                    "DayChange": holding.get("day_change", 0) * holding.get("quantity", 0),
                    "DayChangePct": holding.get("day_change_percentage", 0),
                })
        portfolio_df = pd.DataFrame(holdings)
        return portfolio_df
    except Exception as e:
        logger.error(f"Error fetching portfolio: {str(e)}")
        raise

async def get_funds_data(api, broker: str) -> dict:
    try:
        if not api:
            return {}
        if broker == "Upstox":
            funds_data = api.get_user_fund_margin(api_version="v2").data
            # Ensure funds_data is a dictionary
            if not isinstance(funds_data, dict):
                logger.error(f"Unexpected Upstox funds data format: {type(funds_data)}")
                return {}
            return {
                "equity": {
                    "available_margin": funds_data.get("equity", {}).get("available_margin", 0),
                    "used_margin": funds_data.get("equity", {}).get("used_margin", 0)
                },
                "commodity": {
                    "available_margin": funds_data.get("commodity", {}).get("available_margin", 0),
                    "used_margin": funds_data.get("commodity", {}).get("used_margin", 0)
                }
            }
        else:
            funds_data = api.margins()
            return funds_data
    except Exception as e:
        logger.error(f"Error fetching funds data for {broker}: {str(e)}")
        return {}

async def fetch_fno_snapshot_live_data(symbols: List[str]):
    live_data_dict = {}
    try:
        live_data = market_data.nse_get_fno_snapshot_live(mode="json").get('data')
        live_data_dict_full = {item['symbol']: item for item in live_data} if live_data else {}
        live_data_dict = {symbol: live_data_dict_full.get(symbol, {}) for symbol in symbols}
        if not live_data_dict:
            logger.warning(f"No live data found for symbols: {symbols}")
        else:
            logger.info(f"Fetched live data for symbols: {list(live_data_dict.keys())}")
    except Exception as e:
        logger.error(f"Error fetching FNO snapshot live data for {symbols}: {str(e)}")
    return live_data_dict

async def get_quotes_from_nse(instruments: List[str], db: AsyncSession = None) -> List[QuoteResponse]:
    try:
        instruments_map = await fetch_symbols_for_instruments(db, instruments)
        live_data = await fetch_fno_snapshot_live_data([inst.trading_symbol for inst in instruments_map])
        quotes = []
        for instrument in instruments_map:
            if live_data[instrument.trading_symbol]:
                raw_data = live_data[instrument.trading_symbol]
            else:
                logger.warning(f"No live data found for {instrument.trading_symbol}, fetching from NSE API")
                raw_data = market_data.nse_get_quote(instrument.trading_symbol, security_type="NIFTY_500")

            if not raw_data:
                logger.warning(f"No data found for instrument {instrument}")
                continue
            quotes.append(QuoteResponse(
                instrument_token=instrument.instrument_token,
                trading_symbol=instrument.trading_symbol,
                last_price=raw_data.get("lastPrice", 0.0),
                net_change=raw_data.get("change", 0.0),
                pct_change=raw_data.get("pChange", 0.0),
                volume=raw_data.get("totalTradedVolume", 0),
                average_price=raw_data.get("totalTradedValue", 0.0)/raw_data.get("totalTradedVolume", 1),
                ohlc={
                    "open": raw_data.get("open", 0.0),
                    "high": raw_data.get("dayHigh", 0.0),
                    "low": raw_data.get("dayLow", 0.0),
                    "close": raw_data.get("lastPrice", 0.0)
                },
                depth={}
            ))
        return quotes
    except Exception as e:
        logger.error(f"Error fetching quotes from NSE for {instruments}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_quotes(upstox_api, kite_api, instruments: List[str], db: AsyncSession = None) -> List[QuoteResponse]:
    if not upstox_api:
        try:
            return await get_quotes_from_nse(instruments, db)
        except Exception as e:
            logger.error(f"Error fetching quotes from NSE: {str(e)}")
            raise HTTPException(status_code=400, detail="Upstox API required for quotes")

    try:
        # Convert trading symbols to proper instrument keys
        instruments_map = await fetch_symbols_for_instruments(db, instruments)
        upstox_instrument_keys = [inst.instrument_token for inst in instruments_map]

        if not upstox_instrument_keys:
            logger.warning(f"No valid instrument keys found for: {instruments}")
            return await get_quotes_from_nse(instruments, db)

        logger.debug(f"Converting symbols {instruments} to instrument keys: {upstox_instrument_keys}")

        response = upstox_api.get_full_market_quote(symbol=",".join(upstox_instrument_keys), api_version="v2").data
        quotes = []

        for instrument, quote in response.items():
            quote_dict = quote.to_dict()

            # Find the corresponding trading symbol from our mapping
            trading_symbol = ""
            for inst_map in instruments_map:
                if inst_map.instrument_token == instrument:
                    trading_symbol = inst_map.trading_symbol
                    break

            quotes.append(QuoteResponse(
                instrument_token=instrument,
                trading_symbol=trading_symbol or instrument.split(":")[-1],
                last_price=quote_dict.get("last_price", 0.0),
                net_change=quote_dict.get("net_change", 0.0),
                pct_change=round((quote_dict.get("net_change", 0.0)/quote_dict.get("last_price", 1.0))*100, 2) if quote_dict.get("last_price", 0.0) > 0 else 0.0,
                volume=quote_dict.get("volume", 0),
                average_price=quote_dict.get("average_price"),
                ohlc={
                    "open": quote_dict.get("ohlc", {}).get("open", 0.0),
                    "high": quote_dict.get("ohlc", {}).get("high", 0.0),
                    "low": quote_dict.get("ohlc", {}).get("low", 0.0),
                    "close": quote_dict.get("ohlc", {}).get("close", 0.0)
                },
                depth=quote_dict.get("depth", {}),
            ))
        return quotes
    except Exception as e:
        logger.error(f"Error fetching quotes using upstox: {str(e)}")
        try:
            return await get_quotes_from_nse(instruments, db)
        except Exception as e:
            logger.error(f"Error fetching quotes from NSE fallback: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

async def get_ohlc_from_nse(instruments: List[str], db: AsyncSession = None) -> List[OHLCResponse]:
    try:
        instruments_map = await fetch_symbols_for_instruments(db, instruments)
        live_data = await fetch_fno_snapshot_live_data([inst.trading_symbol for inst in instruments_map])
        ohlc_data = []

        for instrument in instruments_map:
            if live_data[instrument.trading_symbol]:
                raw_data = live_data[instrument.trading_symbol]
            else:
                logger.warning(f"No live data found for {instrument.trading_symbol}, fetching from NSE API")
                raw_data = market_data.nse_get_quote(instrument.trading_symbol, security_type="NIFTY_500")

            ohlc_data.append(OHLCResponse(
                instrument_token=instrument.instrument_token,
                trading_symbol=instrument.trading_symbol,
                open=raw_data.get("open", 0.0),
                high=raw_data.get("dayHigh", 0.0),
                low=raw_data.get("dayLow", 0.0),
                close=raw_data.get("lastPrice", 0.0),
                previous_close=raw_data.get("previousClose", 0.0),
                volume=raw_data.get("totalTradedVolume", 0)
            ))
        return ohlc_data
    except Exception as e:
        logger.error(f"Error fetching OHLC data via NSE for {instruments}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_ohlc_openchart(instruments: List[str], db: AsyncSession = None) -> List[OHLCResponse]:
    try:
        instruments_map = await fetch_symbols_for_instruments(db, instruments)
        ohlc_data = []
        end_date_obj = datetime.today()
        start_date_obj = end_date_obj - timedelta(days=5)
        for instrument in instruments_map:
            raw_data = load_stock_history(instrument.trading_symbol, start_date_obj, end_date_obj, load=False)
            ohlc_data.append(OHLCResponse(
                instrument_token=instrument.instrument_token,
                trading_symbol=instrument.trading_symbol,
                open=raw_data.iloc[-1]['open'],
                high=raw_data.iloc[-1]['high'],
                low=raw_data.iloc[-1]['low'],
                close=raw_data.iloc[-1]['close'],
                volume=raw_data.iloc[-1]['volume'],
                previous_close=raw_data.iloc[-2]['close']
            ))
        return ohlc_data
    except Exception as e:
        logger.error(f"Error fetching OHLC data via openchart for {instruments}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_ohlc(upstox_api, kite_api, instruments: List[str], db: AsyncSession = None) -> List[OHLCResponse]:
    if not upstox_api:
        try:
            return await get_ohlc_from_nse(instruments, db)
        except Exception as e:
            logger.error(f"Error fetching OHLC data from NSE: {str(e)}")
            try:
                return await get_ohlc_openchart(instruments, db)
            except Exception as e:
                logger.error(f"Error fetching OHLC data from openchart: {str(e)}")
                raise HTTPException(status_code=400, detail="Upstox API required for OHLC")

    try:
        # Convert trading symbols to proper instrument keys
        instruments_map = await fetch_symbols_for_instruments(db, instruments)
        upstox_instrument_keys = [inst.instrument_token for inst in instruments_map]

        if not upstox_instrument_keys:
            logger.warning(f"No valid instrument keys found for: {instruments}")
            return await get_ohlc_from_nse(instruments, db)

        logger.debug(f"Converting symbols {instruments} to instrument keys: {upstox_instrument_keys}")

        response = upstox_api.get_market_quote_ohlc(interval="1d", instrument_key= ",".join(upstox_instrument_keys)).data
        logger.debug("OHLC Response: " + str(response))
        ohlc_data = []

        for instrument, ohlc in response.items():
            ohlc_dict = ohlc.to_dict()

            # Find the corresponding trading symbol from our mapping
            trading_symbol = ""
            for inst_map in instruments_map:
                if inst_map.instrument_token == instrument:
                    trading_symbol = inst_map.trading_symbol
                    break

            ohlc_data.append(OHLCResponse(
                instrument_token=instrument,
                trading_symbol=trading_symbol or instrument.split(":")[-1],
                open=ohlc_dict.get("live_ohlc", {}).get("open", 0.0),
                high=ohlc_dict.get("live_ohlc", {}).get("high", 0.0),
                low=ohlc_dict.get("live_ohlc", {}).get("low", 0.0),
                close=ohlc_dict.get("live_ohlc", {}).get("close", 0.0),
                previous_close=ohlc_dict.get("prev_ohlc", {}).get("close", 0.0) if ohlc_dict.get("prev_ohlc") else 0.0,
                volume=ohlc_dict.get("volume")
            ))
        return ohlc_data
    except Exception as e:
        logger.error(f"Error fetching OHLC data via Upstox: {str(e)}")
        try:
            return await get_ohlc_openchart(instruments, db)
        except Exception as e:
            logger.error(f"Error fetching OHLC data from openchart: {str(e)}")
            try:
                return await get_ohlc_from_nse(instruments, db)
            except Exception as e:
                logger.error(f"Error fetching OHLC data from NSE: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

async def get_ltp_from_nse(instruments: List[str], db: AsyncSession = None) -> List[LTPResponse]:
    try:
        instruments_map = await fetch_symbols_for_instruments(db, instruments)
        live_data = await fetch_fno_snapshot_live_data([inst.trading_symbol for inst in instruments_map])
        ohlc_data = []

        for instrument in instruments_map:
            if live_data[instrument.trading_symbol]:
                raw_data = live_data[instrument.trading_symbol]
            else:
                logger.warning(f"No live data found for {instrument.trading_symbol}, fetching from NSE API")
                raw_data = market_data.nse_symbol_quote(instrument.trading_symbol)

            ohlc_data.append(LTPResponse(
                instrument_token=instrument.instrument_token,
                trading_symbol=instrument.trading_symbol,
                last_price=raw_data.get("lastPrice", 0.0),
                volume=raw_data.get("totalTradedVolume", 0),
                previous_close=raw_data.get("previousClose", 0.0),
            ))
        return ohlc_data
    except Exception as e:
        logger.error(f"Error fetching LTP data via NSE for {instruments}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_ltp_openchart(instruments: List[str], db: AsyncSession = None) -> List[LTPResponse]:
    try:
        instruments_map = await fetch_symbols_for_instruments(db, instruments)
        ohlc_data = []
        end_date_obj = datetime.today()
        start_date_obj = end_date_obj - timedelta(days=5)

        for instrument in instruments_map:
            raw_data = load_stock_history(instrument.trading_symbol, start_date_obj, end_date_obj, load=False)
            ohlc_data.append(LTPResponse(
                instrument_token=instrument.instrument_token,
                trading_symbol=instrument.trading_symbol,
                last_price=raw_data.iloc[-1]['close'],
                volume=raw_data.iloc[-1]['volume'],
                previous_close=raw_data.iloc[-2]['close']
            ))
        return ohlc_data
    except Exception as e:
        logger.error(f"Error fetching LTP data via openchart for {instruments}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_ltp(upstox_api, kite_api, instruments: List[str], db: AsyncSession = None) -> List[LTPResponse]:
    if not upstox_api:
        try:
            return await get_ltp_from_nse(instruments, db)
        except Exception as e:
            logger.error(f"Error fetching LTP data from NSE: {str(e)}")
            try:
                return await get_ltp_openchart(instruments, db)
            except Exception as e:
                logger.error(f"Error fetching LTP data from openchart: {str(e)}")
                raise HTTPException(status_code=400, detail="Upstox API required for LTP")

    try:
        # Convert trading symbols to proper instrument keys
        instruments_map = await fetch_symbols_for_instruments(db, instruments)
        upstox_instrument_keys = [inst.instrument_token for inst in instruments_map]

        if not upstox_instrument_keys:
            logger.warning(f"No valid instrument keys found for: {instruments}")
            return await get_ltp_from_nse(instruments, db)

        logger.debug(f"Converting symbols {instruments} to instrument keys: {upstox_instrument_keys}")

        response = upstox_api.get_ltp(instrument_key=",".join(upstox_instrument_keys)).data
        logger.debug("LTP Response: " + str(response))
        ltp_data = []

        for instrument, quote in response.items():
            quote_dict = quote.to_dict()

            # Find the corresponding trading symbol from our mapping
            trading_symbol = ""
            for inst_map in instruments_map:
                if inst_map.instrument_token == instrument:
                    trading_symbol = inst_map.trading_symbol
                    break

            ltp_data.append(LTPResponse(
                instrument_token=instrument,
                trading_symbol=trading_symbol or instrument.split(":")[-1],
                last_price=quote_dict.get("last_price", 0.0),
                volume=quote_dict.get("volume", 0),
                previous_close=quote_dict.get("cp", 0.0),
            ))
        return ltp_data
    except Exception as e:
        logger.error(f"Error fetching LTP: {str(e)}")
        try:
            return await get_ltp_from_nse(instruments, db)
        except Exception as e:
            logger.error(f"Error fetching LTP data from NSE fallback: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

async def get_historical_data(upstox_api, upstox_access_token, trading_symbol: str,
                              from_date: str, to_date: str, unit: str,
                              interval: str, instrument: str = None, db: AsyncSession = None, nse_db: AsyncSession = None, source: str = "default") -> HistoricalDataResponse:
    logger.info(f"Fetching historical data for {trading_symbol} from {from_date} to {to_date}, unit: {unit}, interval: {interval}, source: {source}")
    data_points = []

    # Convert dates to datetime objects for comparison
    from_date_dt = datetime.strptime(from_date, "%Y-%m-%d")
    to_date_dt = datetime.strptime(to_date, "%Y-%m-%d")

    def normalize_timestamp(value):
        """Convert pandas/numpy timestamps to serializable datetime objects."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, date) and not isinstance(value, datetime):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, np.datetime64):
            return pd.Timestamp(value).to_pydatetime()
        if isinstance(value, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return value

    # Define data fetching functions
    async def fetch_from_db():
        """Fetch data from database with interval-specific table handling"""
        if not db or not nse_db:
            return []

        try:
            # Determine table name based on interval type
            base_table_name = trading_symbol.replace(" ", "_").replace("-", "_")

            if unit.lower() == 'day' and interval == '1':
                table_name = base_table_name  # Daily data in base table
            elif unit.lower() == 'week':
                table_name = f"{base_table_name}_W"  # Weekly data in _W table
            elif unit.lower() == 'month':
                table_name = f"{base_table_name}_M"  # Monthly data in _M table
            else:
                # For minute, hour or other intervals, only stored in day tables
                logger.info(f"Unit '{unit}' with interval '{interval}' not available in database, skipping DB source")
                return []

            logger.info(f"Trying database table '{table_name}' for {unit} interval historical data")

            query = f"""
                SELECT * FROM \"{table_name}\"
                WHERE timestamp >= :from_date AND timestamp <= :to_date
                ORDER BY timestamp
            """

            result = await async_fetch_query(
                nse_db,
                text(query),
                {"from_date": from_date_dt, "to_date": to_date_dt}
            )

            db_points = []
            if result:
                for row in result:
                    date_val = row.get('timestamp')
                    if isinstance(date_val, str):
                        date_val = datetime.strptime(date_val, "%Y-%m-%d")

                    db_points.append(HistoricalDataPoint(
                        timestamp=normalize_timestamp(date_val),
                        open=float(row.get('open', 0)),
                        high=float(row.get('high', 0)),
                        low=float(row.get('low', 0)),
                        close=float(row.get('close', 0)),
                        volume=int(row.get('volume', 0))
                    ))
                logger.info(f"Successfully fetched {len(db_points)} data points from database")
            return db_points
        except Exception as e:
            logger.error(f"Error fetching from database: {str(e)}")
            return []

    async def fetch_from_upstox():
        """Fetch data from Upstox API with proper interval handling"""
        if not upstox_api:
            return []

        try:
            logger.info(f"Trying Upstox API for {trading_symbol} with unit: {unit}, interval: {interval}")
            headers = {"Authorization": f"Bearer {upstox_access_token}"}

            # Map unit to Upstox API format (Upstox uses plural forms)
            upstox_unit_map = {
                'day': 'days',
                'minute': 'minutes',
                'hour': 'hours',
                'week': 'weeks',
                'month': 'months'
            }
            upstox_unit = upstox_unit_map.get(unit.lower(), unit.lower())

            # Use v3 API as per documentation reference (v2 doesn't support historical candles properly)
            url = f"https://api.upstox.com/v3/historical-candle/{instrument}/{upstox_unit}/{interval}/{to_date}/{from_date}"
            logger.info(f"Upstox API URL: {url}")
            response = requests.get(url, headers=headers)

            upstox_points = []
            if response.status_code == 200:
                candles = response.json().get("data", {}).get("candles", [])
                for candle in candles:
                    upstox_points.append(HistoricalDataPoint(
                        timestamp=datetime.strptime(candle[0], "%Y-%m-%dT%H:%M:%S%z"),
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=int(candle[5])
                    ))
                logger.info(f"Successfully fetched {len(upstox_points)} data points from Upstox API")
            else:
                logger.error(f"Upstox API error: {response.text}")
            return upstox_points
        except Exception as e:
            logger.warning(f"Failed to fetch data from Upstox API: {str(e)}")
            return []

    async def fetch_from_openchart():
        """Fetch data from openchart with proper interval mapping"""
        try:
            logger.info(f"Trying openchart for NSE stock: {trading_symbol} with unit: {unit}, interval: {interval}")

            # Map interval to openchart format
            openchart_interval = interval
            if unit.lower() == 'minute':
                openchart_interval = f"{interval}m"  # e.g., 5m, 15m, 30m
            elif unit.lower() == 'hour':
                openchart_interval = f"{interval}h"  # e.g., 1h, 4h
            elif unit.lower() == 'day':
                openchart_interval = f"{interval}d"  # e.g., 1d
            elif unit.lower() == 'week':
                openchart_interval = f"{interval}w"  # e.g., 1w
            elif unit.lower() == 'month':
                openchart_interval = f"{interval}M"  # e.g., 1M

            logger.info(f"Using openchart interval: {openchart_interval}")

            # Use asyncio.to_thread to run the synchronous load_stock_history in a thread
            import asyncio
            data = await asyncio.to_thread(load_stock_history, trading_symbol, from_date_dt, to_date_dt, interval=openchart_interval, load=False)

            openchart_points = []
            # Check if data is a DataFrame and not empty
            if hasattr(data, 'empty') and not data.empty:
                for _, row in data.iterrows():
                    openchart_points.append(HistoricalDataPoint(
                        timestamp=normalize_timestamp(row["timestamp"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"])
                    ))
                logger.info(f"Successfully fetched {len(openchart_points)} data points from openchart")
            else:
                logger.warning(f"openchart returned no data or invalid data type: {type(data)}")
            return openchart_points
        except Exception as e:
            logger.warning(f"Failed to fetch data from openchart: {str(e)}")
            return []

    # Enhanced data fetching logic based on source parameter and interval type
    if source == "default":
        # For day, week, month intervals: try database first
        if unit.lower() in ['day', 'week', 'month']:
            data_points = await fetch_from_db()
            if not data_points:
                data_points = await fetch_from_upstox()
            if not data_points:
                data_points = await fetch_from_openchart()
        else:
            # For minute, hour intervals: try Upstox first (better for intraday), then openchart
            data_points = await fetch_from_upstox()
            if not data_points:
                data_points = await fetch_from_openchart()
    elif source == "db":
        # Try database first, then fallback to APIs if DB doesn't have the interval
        data_points = await fetch_from_db()
        if not data_points:
            if unit.lower() in ['minute', 'hour'] or (unit.lower() == 'day' and interval != '1'):
                # For non-daily intervals, prefer Upstox over openchart
                data_points = await fetch_from_upstox()
                if not data_points:
                    data_points = await fetch_from_openchart()
            else:
                data_points = await fetch_from_upstox()
                if not data_points:
                    data_points = await fetch_from_openchart()
    elif source == "upstox":
        # Try Upstox first, then fallback to openchart if available
        data_points = await fetch_from_upstox()
        if not data_points:
            data_points = await fetch_from_openchart()
    elif source == "openchart":
        # Try openchart first, then fallback to Upstox
        data_points = await fetch_from_openchart()
        if not data_points:
            data_points = await fetch_from_upstox()
    else:
        logger.warning(f"Unknown source '{source}', using default order")
        # Use the default logic
        if unit.lower() in ['day', 'week', 'month']:
            data_points = await fetch_from_db()
            if not data_points:
                data_points = await fetch_from_upstox()
            if not data_points:
                data_points = await fetch_from_openchart()
        else:
            data_points = await fetch_from_upstox()
            if not data_points:
                data_points = await fetch_from_openchart()

    # If we still don't have data, raise an error
    if not data_points:
        logger.error(f"Could not retrieve historical data for {trading_symbol} from any source")
        raise HTTPException(status_code=404, detail=f"Historical data not available for {trading_symbol}")

    # Create the response with sorted data
    historical_data = HistoricalDataResponse(
        instrument_token=instrument,
        data=sorted(data_points, key=lambda x: x.timestamp)
    )

    logger.info(f"Returning {len(data_points)} historical data points for {trading_symbol}")
    return historical_data


async def fetch_instruments(db: AsyncSession, refresh: bool = False) -> List[InstrumentSchema]:
    try:
        if refresh:
            path = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
            instruments_df = pd.read_json(path)
            instruments_df = instruments_df[
                ['trading_symbol', 'name', 'instrument_key', 'exchange', 'instrument_type', 'segment']
            ][((instruments_df['segment'] == 'NSE_EQ') & (instruments_df['instrument_type'] == 'EQ') &
              (~instruments_df['name'].str.contains('TEST', case=False, na=False))) | (instruments_df['segment'] == 'NSE_INDEX') & (instruments_df['instrument_type'] == 'INDEX')
               | ((instruments_df['segment']=='NSE_COM') & (instruments_df['instrument_type']=='COM'))]

            #  check if instruments table exists and delete entries from it
            await db.execute(text("DROP TABLE IF EXISTS instruments"))
            await db.execute(text("CREATE TABLE instruments (instrument_token VARCHAR PRIMARY KEY, trading_symbol VARCHAR, name VARCHAR, exchange VARCHAR, instrument_type VARCHAR, segment VARCHAR)"))

            for _, row in instruments_df.iterrows():
                instrument = Instrument(
                    instrument_token=row["instrument_key"],
                    trading_symbol=row["trading_symbol"],
                    name=row["name"],
                    exchange=row["exchange"],
                    instrument_type=row["instrument_type"],
                    segment=row["segment"]
                )
                db.add(instrument)
            await db.commit()
            logger.info(f"Refreshed {len(instruments_df)} instruments in database")

        query = select(Instrument)
        result = await db.execute(query)
        instruments = result.scalars().all()
        if not instruments:
            raise HTTPException(status_code=404, detail="No instruments found")

        return [
            InstrumentSchema(
                instrument_token=inst.instrument_token,
                exchange=inst.exchange,
                trading_symbol=inst.trading_symbol,
                name=inst.name,
                instrument_type=inst.instrument_type,
                segment=inst.segment
            )
            for inst in instruments
        ]
    except requests.RequestException as re:
        logger.error(f"Network error fetching instruments: {str(re)}")
        raise HTTPException(status_code=502, detail="Network error")
    except Exception as e:
        logger.error(f"Error fetching instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_symbols_for_instruments(db: AsyncSession, instruments: List[str]) -> List[InstrumentSchema]:
    try:
        if not instruments:
            raise HTTPException(status_code=400, detail="No instruments provided")

        query = select(Instrument).where(
            or_(
                Instrument.instrument_token.in_(instruments),
                Instrument.trading_symbol.in_(instruments)
            )
)
        result = await db.execute(query)
        instruments = result.scalars().all()

        if not instruments:
            raise HTTPException(status_code=404, detail="No instruments found for the provided symbols")

        return [
            InstrumentSchema(
                instrument_token=inst.instrument_token,
                exchange=inst.exchange,
                trading_symbol=inst.trading_symbol,
                instrument_type=inst.instrument_type,
                segment=inst.segment
            )
            for inst in instruments
        ]
    except Exception as e:
        logger.error(f"Error fetching symbols for instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_order_history(upstox_api, kite_api, order_id: str, broker: str) -> List[OrderHistory]:
    try:
        history = []
        if broker == "Upstox" and upstox_api:
            response = upstox_api.get_order_details(order_id=order_id).data
            for entry in response:
                entry_dict = entry.to_dict()
                history.append(OrderHistory(
                    order_id=order_id,
                    status=entry_dict.get("status", ""),
                    timestamp=datetime.strptime(entry_dict.get("order_timestamp"), "%Y-%m-%d %H:%M:%S"),
                    price=entry_dict.get("price"),
                    quantity=entry_dict.get("quantity"),
                    remarks=entry_dict.get("status_message")
                ))
        elif broker == "Zerodha" and kite_api:
            response = kite_api.order_history(order_id=order_id)
            for entry in response:
                history.append(OrderHistory(
                    order_id=order_id,
                    status=entry.get("status", ""),
                    timestamp=datetime.strptime(entry.get("order_timestamp"), "%Y-%m-%d %H:%M:%S"),
                    price=entry.get("price"),
                    quantity=entry.get("quantity"),
                    remarks=entry.get("status_message")
                ))
        return history
    except Exception as e:
        logger.error(f"Error fetching order history for order {order_id}: {str(e)}")
        raise

def get_order_trades(upstox_api, kite_api, order_id: str, broker: str) -> List[Trade]:
    try:
        trades = []
        if broker == "Upstox" and upstox_api:
            response = upstox_api.get_trades_by_order(order_id=order_id).data
            for trade in response:
                trade_dict = trade.to_dict()
                trades.append(Trade(
                    trade_id=trade_dict.get("trade_id"),
                    order_id=order_id,
                    instrument_token=trade_dict.get("instrument_token"),
                    quantity=trade_dict.get("quantity"),
                    price=trade_dict.get("price"),
                    timestamp=datetime.strptime(trade_dict.get("trade_timestamp"), "%Y-%m-%d %H:%M:%S")
                ))
        elif broker == "Zerodha" and kite_api:
            response = kite_api.order_trades(order_id=order_id)
            for trade in response:
                trades.append(Trade(
                    trade_id=trade.get("trade_id"),
                    order_id=order_id,
                    instrument_token=trade.get("instrument_token", ""),
                    quantity=trade.get("quantity"),
                    price=trade.get("average_price"),
                    timestamp=datetime.strptime(trade.get("trade_timestamp"), "%Y-%m-%d %H:%M:%S")
                ))
        return trades
    except Exception as e:
        logger.error(f"Error fetching trades for order {order_id}: {str(e)}")
        raise

async def execute_strategy(api, strategy: str, instrument_token: str, quantity: int, stop_loss: float, take_profit: float,
                           broker: str, db: AsyncSession, user_id: str):
    try:
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if stop_loss < 0 or take_profit < 0:
            raise ValueError("Stop loss and take profit cannot be negative")
        # Parse strategy if it's a JSON string
        if strategy.startswith("{"):
            strategy_data = json.loads(strategy)
            strategy_name = strategy_data.get("name", "Custom Strategy")
            is_custom = True
        else:
            strategy_name = strategy
            is_custom = False
        # Get trading symbol from instrument token
        trading_symbol = get_symbol_for_instrument(instrument_token)
        data = await get_historical_data(upstox_api=api, upstox_access_token=None, trading_symbol=trading_symbol,
                                         from_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                                         to_date=datetime.now().strftime("%Y-%m-%d"), unit="days", interval="1", source="default")
        if not data.data:
            raise ValueError("Failed to fetch historical data")
        df = pd.DataFrame([{
            "timestamp": point.timestamp,
            "open": point.open,
            "high": point.high,
            "low": point.low,
            "close": point.close,
            "volume": point.volume
        } for point in data.data])
        signal = None

        if is_custom:
            entry_signals = await evaluate_custom_strategy(df, strategy_data.get("entry_conditions", []))
            signal = entry_signals.iloc[-1] if not entry_signals.empty else None
        elif strategy == "MACD Crossover":
            macd_line, signal_line, _ = calculate_macd(df)
            signal = check_macd_crossover(macd_line, signal_line)
        elif strategy == "Bollinger Bands":
            _, upper_band, lower_band = calculate_bollinger_bands(df)
            signal = check_bollinger_band_signals(df, upper_band, lower_band)
        elif strategy == "RSI Oversold/Overbought":
            rsi = calculate_rsi(df)
            signal = "BUY" if rsi.iloc[-1] < 30 else "SELL" if rsi.iloc[-1] > 70 else None
        elif strategy == "Stochastic Oscillator":
            k, d = calculate_stochastic_oscillator(df)
            signal = check_stochastic_signals(k, d)
        elif strategy == "Support/Resistance Breakout":
            signal = check_support_resistance_breakout(df)
        if signal:
            trading_symbol = get_symbol_for_instrument(instrument_token)
            current_price = df["close"].iloc[-1]
            stop_loss_price = current_price * (1 - stop_loss / 100) if signal == "BUY" else current_price * (1 + stop_loss / 100)
            take_profit_price = current_price * (1 + take_profit / 100) if signal == "BUY" else current_price * (1 - take_profit / 100)
            order_result = await place_order(
                api=api,
                instrument_token=instrument_token,
                trading_symbol=trading_symbol,
                transaction_type=signal,
                quantity=quantity,
                price=0,
                order_type="MARKET",
                trigger_price=0,
                is_amo=False,
                product_type="CNC",
                validity="DAY",
                stop_loss=stop_loss_price,
                target=take_profit_price,
                broker=broker,
                db=db,
                user_id=user_id
            )
            if order_result:
                if broker == "Zerodha":
                    order_id = order_result if order_result else None
                else:
                    if hasattr(order_result.data, 'order_ids') and order_result.data.order_ids:
                        order_id = order_result.data.order_ids[0]
                    elif hasattr(order_result.data, 'order_id'):
                        order_id = order_result.data.order_id
                    else:
                        order_id = "unknown" if order_result else None

                logger.info(f"Strategy {strategy} executed for {trading_symbol}, order ID {order_id}, user {user_id}")
                result = await db.execute(select(User).filter(User.user_id == user_id))
                user = result.scalars().first()
                if user:
                    await notify(f"Strategy Order Placed: {strategy}",
                                 f"Signal: {signal}, Order ID: {order_id}, Symbol: {trading_symbol}",
                                 user.email)
                return f"{signal} signal executed with order ID {order_id}"
            logger.warning(f"Strategy {strategy} signal detected but order placement failed for {trading_symbol}")
            return f"{signal} signal detected but order placement failed"
        logger.info(f"No trading signal detected for strategy {strategy}, symbol {instrument_token}")
        return "No trading signal detected"
    except ValueError as ve:
        logger.error(f"Validation error executing strategy {strategy} for {instrument_token}, user {user_id}: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error executing strategy {strategy} for {instrument_token}, user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute strategy: {str(e)}")

async def schedule_strategy_execution(api, strategy: str, instrument_token: str, quantity: int, stop_loss: float,
                                     take_profit: float, interval_minutes: int, run_hours: List[tuple], broker: str,
                                     db: AsyncSession, user_id: str):
    try:
        def is_market_open():
            now = datetime.now().time()
            weekday = datetime.now().weekday()
            if weekday >= 5:  # Skip weekends
                return False
            for start_hour, end_hour in run_hours:
                start_time = date_time(start_hour, 0)
                end_time = date_time(end_hour, 0)
                if start_time <= now <= end_time:
                    return True
            return False

        async def run_strategy():
            while True:
                if is_market_open():
                    result = await execute_strategy(
                        api, strategy, instrument_token, quantity, stop_loss, take_profit, broker, db, user_id
                    )
                    logger.info(f"Strategy execution result for {strategy}: {result}")
                    result = await db.execute(select(User).filter(User.user_id == user_id))
                    user = result.scalars().first()
                    if user:
                        await notify(f"Strategy Execution: {strategy}", result, user.email)
                else:
                    logger.info("Market closed. Waiting for next interval.")
                await asyncio.sleep(interval_minutes * 60)

        task = asyncio.create_task(run_strategy())
        logger.info(f"Scheduled strategy {strategy} for {instrument_token} every {interval_minutes} minutes")
        task_key = f"{user_id}_{strategy}_{instrument_token}"
        strategy_tasks[task_key] = task
        return {"status": "success", "message": f"Strategy {strategy} scheduled"}
    except Exception as e:
        logger.error(f"Error scheduling strategy {strategy}: {str(e)}")
        return {"status": "error", "message": str(e)}


async def stop_strategy_execution(strategy: str, instrument_token: str, user_id: str, db: AsyncSession):
    try:
        # Note: In a production system, store task references in a database or Redis
        # For simplicity, assume tasks are tracked in a global dict (not ideal)
        global strategy_tasks
        task_key = f"{user_id}_{strategy}_{instrument_token}"
        if task_key in strategy_tasks:
            task = strategy_tasks[task_key]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Stopped strategy {strategy} for {instrument_token}, user {user_id}")
            del strategy_tasks[task_key]
            result = await db.execute(select(User).filter(User.user_id == user_id))
            user = result.scalars().first()
            if user:
                await notify("Strategy Execution Stopped",
                             f"Stopped strategy {strategy} for {instrument_token}",
                             user.email)
            return {"status": "success", "message": f"Stopped strategy {strategy} for {instrument_token}"}
        else:
            return {"status": "error", "message": f"No active strategy {strategy} for {instrument_token}"}
    except Exception as e:
        logger.error(f"Error stopping strategy {strategy} for {instrument_token}, user {user_id}: {str(e)}")
        return {"status": "error", "message": str(e)}


async def backtest_strategy(trading_symbol: str, instrument_token: str, timeframe: str, strategy: str,
                            params: Dict, start_date: str, end_date: str, ws_callback: Optional[Callable] = None,
                            db: Optional[AsyncSession] = None, nse_db: Optional[AsyncSession] = None):
    """
    Main service to orchestrate a backtest, correctly handling single runs,
    custom strategies, and parameter optimization.
    """
    # Check if optimization is enabled
    optimization_config = params.get('optimization_config')
    if optimization_config and optimization_config.get('enabled'):
        logger.info("Using backtesting.py optimization")
        return await run_optimization_backtest(
            instrument_token, timeframe, strategy, params,
            start_date, end_date, db, ws_callback
        )

    logger.info(f"Starting backtest for {trading_symbol} from {start_date} to {end_date}")
    logger.info(f"Strategy: {strategy}, Timeframe: {timeframe}, Params: {params}")

    try:
        # --- 1. Data Fetching ---
        df = await get_historical_dataframe(trading_symbol, instrument_token, timeframe, start_date, end_date, db, nse_db)

        # --- 2. Strategy Identification ---
        strategy_func, strategy_name, is_custom = get_strategy_details(strategy)
        if ws_callback: await ws_callback({"progress": 0.1, "status": "Data fetched, preparing indicators..."})

        # --- 3. Prepare Indicators and Details for Logging ---
        indicator_details = []

        # Calculate indicators for all strategy types upfront
        if is_custom:
            strategy_data = json.loads(strategy)
            all_conditions = strategy_data.get("entry_conditions", []) + strategy_data.get("exit_conditions", [])
            unique_indicators = {}
            for cond in all_conditions:
                for side in ["left", "right"]:
                    name = cond.get(f"{side}_indicator")
                    if name and name != "Fixed Value":
                        iparams = cond.get(f"{side}_params", {})
                        key = (name, frozenset(iparams.items()));
                        unique_indicators[key] = (name, iparams)

            for name, iparams in unique_indicators.values():
                col_name = None
                display_name = f"{name}({', '.join(f'{v}' for v in iparams.values())})" if iparams else name
                if name == "EMA":
                    df.ta.ema(length=iparams.get("period", 20), append=True)
                    col_name = f"EMA_{iparams.get('period', 20)}"
                elif name == "RSI":
                    df.ta.rsi(length=iparams.get("period", 14), append=True)
                    col_name = f"RSI_{iparams.get('period', 14)}"
                elif name == "MACD":
                    df.ta.macd(fast=iparams.get("fast_period", 12), slow=iparams.get("slow_period", 26), signal=iparams.get("signal_period", 9), append=True)
                    # We need to add multiple columns for MACD
                    f, s, sig = iparams.get('fast_period', 12), iparams.get('slow_period', 26), iparams.get('signal_period', 9)
                    indicator_details.append({'column_name': f'MACD_{f}_{s}_{sig}', 'display_name': f'MACD({f},{s},{sig})'})
                    indicator_details.append({'column_name': f'MACDh_{f}_{s}_{sig}', 'display_name': f'Hist({f},{s},{sig})'})
                    indicator_details.append({'column_name': f'MACDs_{f}_{s}_{sig}', 'display_name': f'Signal({f},{s},{sig})'})
                elif name == "Bollinger Bands":
                    df.ta.bbands(length=iparams.get("period", 20), std=iparams.get("std_dev", 2), append=True)
                    p, std = iparams.get('period', 20), iparams.get('std_dev', 2.0)
                    indicator_details.append({'column_name': f'BBU_{p}_{std}', 'display_name': f'BBU({p},{std})'})
                    indicator_details.append({'column_name': f'BBM_{p}_{std}', 'display_name': f'BBM({p},{std})'})
                    indicator_details.append({'column_name': f'BBL_{p}_{std}', 'display_name': f'BBL({p},{std})'})
                elif name in ["Close Price", "Open Price", "High Price", "Low Price", "Volume"]:
                    col_name = name.split(' ')[0].lower()
                    display_name = name

                if col_name:
                    indicator_details.append({'column_name': col_name, 'display_name': display_name})
        else:
            # Prepare indicators for predefined strategies
            if strategy_name == "RSI Oversold/Overbought":
                p = params.get('period', 14)
                df.ta.rsi(length=p, append=True)
                indicator_details.append({'column_name': f'RSI_{p}', 'display_name': f'RSI({p})'})
            elif strategy_name == "MACD Crossover":
                f, s, sig = params.get('fast_period', 12), params.get('slow_period', 26), params.get('signal_period', 9)
                df.ta.macd(fast=f, slow=s, signal=sig, append=True)
                indicator_details.append({'column_name': f'MACD_{f}_{s}_{sig}', 'display_name': f'MACD({f},{s},{sig})'})
                indicator_details.append(
                    {'column_name': f'MACDs_{f}_{s}_{sig}', 'display_name': f'Signal({f},{s},{sig})'})
            elif strategy_name == "Bollinger Bands":
                 p, std = params.get('period', 20), params.get('num_std', 2.0)
                 df.ta.bbands(length=p, std=std, append=True)
                 indicator_details.append({'column_name': f'BBU_{p}_{float(std)}', 'display_name': f'BBU({p},{std})'})
                 indicator_details.append({'column_name': f'BBL_{p}_{float(std)}', 'display_name': f'BBL({p},{std})'})


        # --- 4. Run Optimization or Single Backtest ---
        enable_optimization = params.get("enable_optimization", False)
        final_response = {}

        if enable_optimization and not is_custom:
            logger.info(f"Starting optimization for strategy: {strategy_name}")
            results = await run_parameter_optimization(df, strategy_func, params, ws_callback, indicator_details=indicator_details)
            final_response = {"optimization_enabled": True, **results}
        elif is_custom and enable_optimization:
            logger.info(f"Starting optimization for custom strategy: {strategy_name}")
            results = await run_parameter_optimization_for_custom_strategy(df, strategy_data, params, ws_callback,
                                                                           indicator_details=indicator_details)
            final_response = {"optimization_enabled": True, **results}
        elif is_custom:
            logger.info(f"Running single backtest for custom strategy: {strategy_name}")
            results = await backtest_custom_strategy(df, strategy_data, params, ws_callback, indicator_details=indicator_details)
            final_response = {"optimization_enabled": False, **results}
        else:
            logger.info(f"Running single backtest for predefined strategy: {strategy_name}")
            results = await backtest_strategy_generic(df, strategy_func, params, ws_callback, indicator_details=indicator_details)
            final_response = {"optimization_enabled": False, **results}

        # --- 5. Final Formatting ---
        final_response.update({
            "StrategyName": strategy_name,
            "Instrument": instrument_token,
            "StartDate": start_date,
            "EndDate": end_date,
            "Timeframe": timeframe
        })

        # CRITICAL FIX: Always include OHLC data regardless of optimization
        chart_data = df.reset_index()
        chart_data['timestamp'] = chart_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        ohlc_data = chart_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')

        # Ensure OHLC data is in the response (single backtest or optimization)
        if final_response.get("optimization_enabled"):
            # For optimization, add OHLC to best_result
            if "best_result" in final_response and final_response["best_result"]:
                final_response["best_result"]["OHLC"] = ohlc_data
                final_response["best_result"]["Instrument"] = instrument_token
                final_response["best_result"]["StartDate"] = start_date
                final_response["best_result"]["EndDate"] = end_date
                final_response["best_result"]["Timeframe"] = timeframe
        else:
            # For single backtest, add OHLC to main response
            final_response["OHLC"] = ohlc_data

        final_response["ChartData"] = chart_data.to_dict(orient='records')
        final_response["IndicatorDetails"] = indicator_details

        return final_response

    except Exception as e:
        logger.error(f"Error in backtest_strategy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def run_parameter_optimization(df, strategy_func, params, ws_callback, indicator_details: Optional[List[Dict]] = None):
    """
    Runs multiple backtests with varying parameters to find the optimal set.
    """
    optimization_iterations = int(params.get("optimization_iterations", 10))
    stop_loss_range = params.get("stop_loss_range", [1.0, 5.0])
    take_profit_range = params.get("take_profit_range", [2.0, 6.0])
    trailing_stop_range = params.get("trailing_stop_range", [1.0, 3.0])

    param_combinations = []
    for _ in range(optimization_iterations):
        combo = {
            'stop_loss_percent': round(np.random.uniform(stop_loss_range[0], stop_loss_range[1]), 2)
        }

        # Add take profit optimization if specified
        if params.get("take_profit_range") and take_profit_range[1] > take_profit_range[0]:
            combo['take_profit_percent'] = round(np.random.uniform(take_profit_range[0], take_profit_range[1]), 2)

        # Add trailing stop optimization if specified
        if params.get("trailing_stop_range") and trailing_stop_range[1] > trailing_stop_range[0]:
            combo['trailing_stop_loss_percent'] = round(
                np.random.uniform(trailing_stop_range[0], trailing_stop_range[1]), 2)

        param_combinations.append(combo)

    all_runs = []
    best_result, best_metric = None, float('-inf')

    for i, combo in enumerate(param_combinations):
        iter_params = params.copy()
        iter_params.update(combo)

        # Run a full backtest for this parameter combination
        iter_result = await backtest_strategy_generic(df.copy(), strategy_func, iter_params, indicator_details=indicator_details)

        # Store results for comparison
        run_summary = {
            "parameters": combo,
            "TotalPNL": iter_result.get("TotalProfit", 0),
            "WinRate": iter_result.get("WinRate", 0),
            "TotalTrades": iter_result.get("TotalTrades", 0),
            "SharpeRatio": iter_result.get("SharpeRatio", 0),
            "MaxDrawdown": iter_result.get("MaxDrawdown", 0)
        }
        all_runs.append(run_summary)

        # Update best result based on Total PNL
        current_pnl = iter_result.get('TotalProfit', float('-inf'))
        if current_pnl > best_metric:
            best_metric = current_pnl
            best_result = iter_result
            # Add chart data to best result
            best_result["OptimizedParameters"] = combo

        if ws_callback:
            progress = 0.1 + ((i + 1) / optimization_iterations) * 0.9
            await ws_callback(
                {"progress": round(progress, 2), "status": f"Running optimization {i + 1}/{optimization_iterations}"})

    return {"best_result": best_result, "all_runs": all_runs}


async def backtest_custom_strategy(df, strategy_data, params, ws_callback, indicator_details: Optional[List[Dict]] = None):
    """
    Generates signals for a custom strategy and then uses the generic backtesting engine.
    """
    # Check if optimization is enabled
    enable_optimization = params.get("enable_optimization", False)

    if enable_optimization:
        # Use the specialized optimization function for custom strategies
        return await run_parameter_optimization_for_custom_strategy(df, strategy_data, params, ws_callback,
                                                                    indicator_details)
    else:
        # 1. Generate signals from custom conditions
        entry_signals = await evaluate_custom_strategy(df, strategy_data.get("entry_conditions", []), signal_type="BUY")
        exit_signals = await evaluate_custom_strategy(df, strategy_data.get("exit_conditions", []), signal_type="SELL")

        # 2. Combine entry and exit signals into a single column
        df['signal'] = entry_signals.where(entry_signals == 'BUY', exit_signals)

        # 3. Define a dummy function that returns the pre-calculated signals from the DataFrame
        def custom_strategy_func(df, **params):
            return df['signal']

        # 4. Call the generic backtesting engine with the custom signals
        result = await backtest_strategy_generic(df, custom_strategy_func, params, ws_callback, indicator_details=indicator_details)

        return result


async def run_parameter_optimization_for_custom_strategy(df, strategy_data, params, ws_callback,
                                                         indicator_details: Optional[List[Dict]] = None):
    """
    Enhanced parameter optimization specifically for custom strategies.
    Regenerates signals for each parameter combination.
    """
    optimization_iterations = int(params.get("optimization_iterations", 10))
    stop_loss_range = params.get("stop_loss_range", [1.0, 5.0])
    take_profit_range = params.get("take_profit_range", [2.0, 6.0])
    trailing_stop_range = params.get("trailing_stop_range", [1.0, 3.0])

    # Generate parameter combinations for multiple variables
    param_combinations = []
    for _ in range(optimization_iterations):
        combo = {
            'stop_loss_percent': round(np.random.uniform(stop_loss_range[0], stop_loss_range[1]), 2)
        }

        # Add take profit optimization if specified
        if params.get("take_profit_range") and take_profit_range[1] > take_profit_range[0]:
            combo['take_profit_percent'] = round(np.random.uniform(take_profit_range[0], take_profit_range[1]), 2)

        # Add trailing stop optimization if specified
        if params.get("trailing_stop_range") and trailing_stop_range[1] > trailing_stop_range[0]:
            combo['trailing_stop_loss_percent'] = round(
                np.random.uniform(trailing_stop_range[0], trailing_stop_range[1]), 2)

        param_combinations.append(combo)

    all_runs = []
    best_result, best_metric = None, float('-inf')

    for i, combo in enumerate(param_combinations):
        iter_params = params.copy()
        iter_params.update(combo)

        # CRITICAL FIX: Regenerate signals with new parameters for custom strategies
        df_copy = df.copy()

        # Generate fresh signals for this parameter combination
        entry_signals = await evaluate_custom_strategy(df_copy, strategy_data.get("entry_conditions", []),
                                                       signal_type="BUY")
        exit_signals = await evaluate_custom_strategy(df_copy, strategy_data.get("exit_conditions", []),
                                                      signal_type="SELL")

        # Combine signals
        df_copy['signal'] = entry_signals.where(entry_signals == 'BUY', exit_signals)

        # Create strategy function that uses the pre-calculated signals
        def custom_strategy_func(df, **params):
            return df['signal']

        # Run backtest with this parameter combination
        iter_result = await backtest_strategy_generic(df_copy, custom_strategy_func, iter_params,
                                                      indicator_details=indicator_details)

        # Store results for comparison
        run_summary = {
            "parameters": combo,
            "TotalPNL": iter_result.get("TotalProfit", 0),
            "WinRate": iter_result.get("WinRate", 0),
            "TotalTrades": iter_result.get("TotalTrades", 0),
            "SharpeRatio": iter_result.get("SharpeRatio", 0),
            "MaxDrawdown": iter_result.get("MaxDrawdown", 0)
        }
        all_runs.append(run_summary)

        # Update best result based on Total PNL
        current_pnl = iter_result.get('TotalProfit', float('-inf'))
        if current_pnl > best_metric:
            best_metric = current_pnl
            best_result = iter_result
            # Add chart data and optimized parameters to best result
            best_result["OptimizedParameters"] = combo

        if ws_callback:
            progress = 0.1 + ((i + 1) / optimization_iterations) * 0.9
            await ws_callback(
                {"progress": round(progress, 2), "status": f"Running optimization {i + 1}/{optimization_iterations}"})

    return {"best_result": best_result, "all_runs": all_runs}


async def backtest_strategy_generic(
        df: pd.DataFrame,
        strategy_func: Callable,
        params: Dict,
        ws_callback: Optional[Callable] = None,
        indicator_details: Optional[List[Dict]] = None
) -> Dict:
    """
    It preserves all original logic for complex exits (stop-loss, trailing stops,
    and partial take-profits) while adding the ability to capture and log the specific
    indicator values that trigger a trade.
    """
    df_copy = df.copy()

    # Generate signals if not already present
    if 'signal' not in df_copy.columns and strategy_func is not None:
        # This ensures backward compatibility if a df is passed without a signal column.
        # It filters params to avoid the original TypeError.
        import inspect
        strategy_params = inspect.signature(strategy_func).parameters
        filtered_params = {k: v for k, v in params.items() if k in strategy_params}
        signals = strategy_func(df_copy, **filtered_params)
        df_copy['signal'] = signals

    # --- State Variables ---
    quantity_in_position = 0.0
    entry_price = 0.0
    stop_loss_price = 0.0
    trailing_stop_price = 0.0
    take_profit_price = 0.0
    active_partial_exits = []
    tradebook = []

    # --- Parameters ---
    initial_investment = params.get("initial_investment", 100000.0)
    stop_loss_percent = params.get("stop_loss_percent", 0)
    trailing_stop_loss_percent = params.get("trailing_stop_loss_percent", 0)
    position_sizing_percent = params.get("position_sizing_percent", 0)
    take_profit_percent = params.get("take_profit_percent", 0)

    # Ensure partial exits are sorted by target percentage
    partial_exits = sorted(params.get("partial_exits", []), key=lambda x: x.get('target', 0))

    portfolio_value = initial_investment
    total_rows = len(df_copy)

    # --- Helper to capture indicator values ---
    def get_indicators_for_trade(row_tuple):
        if not indicator_details:
            return {}
        indicators_data = {}
        for detail in indicator_details:
            col_name = detail['column_name']
            display_name = detail['display_name']
            if hasattr(row_tuple, col_name):
                 value = getattr(row_tuple, col_name)
                 if pd.notna(value):
                    indicators_data[display_name] = round(value, 2)
        return indicators_data

    # --- Backtesting Loop ---
    for i, row in enumerate(df_copy.itertuples()):
        current_price = row.close

        if ws_callback and i % (total_rows // 20) == 0:
            progress = 0.3 + (i / total_rows) * 0.6
            await ws_callback({"progress": round(progress, 2), "status": f"Processing data: {i}/{total_rows}"})

        # --- CHECK EXIT CONDITIONS (if in a position) ---
        if quantity_in_position > 0:
            full_exit_reason = None
            exit_price = current_price
            indicators_for_exit = {}

            # Priority 1: Hard stops
            if stop_loss_price > 0 and current_price <= stop_loss_price:
                full_exit_reason = "Stop Loss"
                exit_price = stop_loss_price

            # Priority 2: Take profit (NEW - Higher priority than trailing stop)
            elif take_profit_price > 0 and current_price >= take_profit_price:
                full_exit_reason = "Take Profit"
                exit_price = take_profit_price

            # Priority 3: Trailing Stop-Loss
            elif trailing_stop_price > 0 and current_price <= trailing_stop_price:
                full_exit_reason = "Trailing Stop"
                exit_price = trailing_stop_price

            # Priority 4: Strategy based SELL signal
            elif row.signal == "SELL":
                full_exit_reason = "Strategy Signal"
                indicators_for_exit = get_indicators_for_trade(row)  # Capture indicators on signal exit

            # If a full exit was triggered, process it
            if full_exit_reason:
                profit = (exit_price - entry_price) * quantity_in_position
                portfolio_value += profit
                tradebook.append({
                    "Date": row.Index.isoformat(),
                    "Action": "SELL",
                    "Price": exit_price,
                    "Quantity": quantity_in_position,
                    "Profit": profit,
                    "PortfolioValue": portfolio_value,
                    "Reason": full_exit_reason,
                    "Indicators": indicators_for_exit
                })
                quantity_in_position = 0.0  # Exit full position
            else:
                # Priority 5: Partial take-profits (only if no full exit occurred)
                exited_partials = []
                for pe in active_partial_exits:
                    if current_price >= pe['price_level']:
                        qty_to_sell = pe['quantity']
                        profit = (pe['price_level'] - entry_price) * qty_to_sell
                        portfolio_value += profit
                        tradebook.append({
                            "Date": row.Index.isoformat(), "Action": "SELL", "Price": pe['price_level'],
                            "Quantity": qty_to_sell, "Profit": profit, "PortfolioValue": portfolio_value,
                            "Reason": f"Partial Target {pe['target_percent']}%",
                            "Indicators": get_indicators_for_trade(row)
                        })
                        quantity_in_position -= qty_to_sell
                        exited_partials.append(pe) # Mark this partial exit for removal

                # Remove triggered partials from the active list
                active_partial_exits = [pe for pe in active_partial_exits if pe not in exited_partials]

                # Clean up floating point dust
                if quantity_in_position <= 1e-6:
                    quantity_in_position = 0.0

            # Update trailing stop if still in position
            if quantity_in_position > 0 and trailing_stop_loss_percent > 0:
                new_trailing_stop = current_price * (1 - trailing_stop_loss_percent / 100)
                trailing_stop_price = max(trailing_stop_price, new_trailing_stop)

        # Reset all position-specific state if the position is fully closed
        if quantity_in_position == 0:
            entry_price = stop_loss_price = trailing_stop_price = take_profit_price = 0.0
            active_partial_exits.clear()

        # --- CHECK ENTRY CONDITION ---
        if quantity_in_position == 0 and row.signal == "BUY":
            entry_price = current_price

            # Determine position size based on portfolio value if specified
            sizing_base = portfolio_value if position_sizing_percent > 0 else initial_investment
            sizing_amount = sizing_base * (
                        position_sizing_percent / 100) if position_sizing_percent > 0 else initial_investment
            initial_quantity = sizing_amount / entry_price
            quantity_in_position = initial_quantity

            # Set the stop-loss and trailing-stop for the new position
            if stop_loss_percent > 0:
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100)

            # Set take profit
            if take_profit_percent > 0:
                take_profit_price = entry_price * (1 + take_profit_percent / 100)

            # Set trailing stop
            if trailing_stop_loss_percent > 0:
                trailing_stop_price = entry_price * (1 - trailing_stop_loss_percent / 100)

            # Setup partial exit targets for this new position
            if partial_exits:
                for pe_def in partial_exits:
                    target_pct, qty_pct = pe_def.get('target', 0), pe_def.get('qty_percent', 0)
                    if target_pct > 0 and qty_pct > 0:
                        active_partial_exits.append({
                            "price_level": entry_price * (1 + target_pct / 100),
                            "quantity": initial_quantity * (qty_pct / 100),
                            "target_percent": target_pct
                        })

            tradebook.append({
                "Date": row.Index.isoformat(),
                "Action": "BUY",
                "Price": entry_price,
                "Quantity": initial_quantity,
                "Profit": 0,
                "PortfolioValue": portfolio_value,
                "Reason": "Strategy Signal",
                "Indicators": get_indicators_for_trade(row)
            })

    # --- Post-processing: Close any open position at the end ---
    if quantity_in_position > 0:
        last_price = df_copy.iloc[-1].close
        profit = (last_price - entry_price) * quantity_in_position
        portfolio_value += profit
        tradebook.append({
            "Date": df_copy.index[-1].isoformat(), "Action": "SELL", "Price": last_price,
            "Quantity": quantity_in_position, "Profit": profit, "PortfolioValue": portfolio_value,
            "Reason": "End of Backtest", "Indicators": {}
        })

    # Calculate final metrics
    final_portfolio_value = portfolio_value
    total_profit = final_portfolio_value - initial_investment

    # Calculate trades and win rate
    trades_df = pd.DataFrame(tradebook)
    completed_trades = trades_df[trades_df['Action'] == 'SELL']
    total_trades = len(completed_trades)
    winning_trades = len(completed_trades[completed_trades['Profit'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        "InitialInvestment": initial_investment,
        "FinalPortfolioValue": final_portfolio_value,
        "TotalProfit": total_profit,
        "WinRate": win_rate,
        "TotalTrades": total_trades,
        "WinningTrades": winning_trades,
        "LosingTrades": total_trades - winning_trades,
        "Tradebook": tradebook
    }


async def run_optimization_backtest(instrument_token: str,
                                    timeframe: str,
                                    strategy: str,
                                    params: Dict,
                                    start_date: str,
                                    end_date: str,
                                    db,
                                    ws_callback: Optional[Callable] = None) -> Dict:
    """
    Run optimization using backtesting.py library
    """
    try:
        logger.info(f"Starting optimization for {strategy} on {instrument_token}")

        # Get historical data using your existing function
        df = await get_historical_dataframe(instrument_token, timeframe, start_date, end_date, db)

        if ws_callback:
            await ws_callback({"progress": 0.1, "status": "Data loaded, preparing optimization..."})

        # Prepare data for backtesting.py
        bt_data = prepare_data_for_backtesting(df)

        # Get optimization config
        opt_config = params.get('optimization_config', {})

        # Map strategy names to classes
        strategy_map = {
            'RSI Oversold/Overbought': RSIStrategy,
            'MACD Crossover': MACDStrategy,
            'Bollinger Bands': BollingerBandsStrategy
        }

        strategy_class = strategy_map.get(strategy)
        if not strategy_class:
            raise ValueError(f"Strategy '{strategy}' not supported for optimization")

        # Create backtest instance
        bt = Backtest(
            bt_data,
            strategy_class,
            cash=params.get('initial_investment', 100000),
            commission=0.002  # 0.2% commission
        )

        if ws_callback:
            await ws_callback({"progress": 0.2, "status": "Running optimization..."})

        # Extract optimization parameters
        opt_params = opt_config.get('parameters', {})
        max_tries = opt_config.get('max_tries', 50)
        optimize_for = opt_config.get('optimize_for', 'Return [%]')

        # Run optimization or single backtest
        if opt_params:
            result = bt.optimize(
                **opt_params,
                maximize=optimize_for,
                max_tries=max_tries,
                random_state=42
            )

            if ws_callback:
                await ws_callback({"progress": 0.9, "status": "Optimization complete!"})

            # Convert results
            converted_result = convert_backtesting_result(result, opt_params)
            converted_result['optimization_enabled'] = True

        else:
            # Run single backtest
            result = bt.run()
            converted_result = convert_backtesting_result(result)

        # Add metadata
        converted_result.update({
            "StrategyName": strategy,
            "Instrument": instrument_token
        })

        if ws_callback:
            await ws_callback({"progress": 1.0, "status": "Complete!"})

        return converted_result

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise e

async def evaluate_custom_strategy(df: pd.DataFrame, strategy_conditions: List[Dict],
                                   signal_type: str = "BUY") -> pd.Series:
    """
    Evaluates custom strategy conditions using the 'pandas-ta' library.
    This version dynamically calculates all required indicators and appends them
    as columns to the DataFrame, which is both efficient and clean.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_with_indicators = df.copy()

    # --- Step 1: Dynamically calculate and append all required indicators ---
    unique_indicators = {}
    for cond in strategy_conditions:
        for side in ["left", "right"]:
            indicator_name = cond.get(f"{side}_indicator")
            if indicator_name and indicator_name != "Fixed Value":
                params = cond.get(f"{side}_params", {})
                # Create a unique key to avoid redundant calculations
                key = (indicator_name, frozenset(params.items()))
                unique_indicators[key] = (indicator_name, params)

    def get_column_name_for_check(indicator, params):
        """Helper to construct the expected column name from pandas-ta."""
        if indicator == 'SMA': return f"SMA_{params.get('period', 20)}"
        if indicator == 'EMA': return f"EMA_{params.get('period', 14)}"
        if indicator == 'RSI': return f"RSI_{params.get('period', 14)}"
        if indicator == 'MACD': return f"MACD_{params.get('fast_period', 12)}_{params.get('slow_period', 26)}_{params.get('signal_period', 9)}"
        if indicator == 'Bollinger Bands':
            # Check for one of the bands, e.g., the middle one.
            return f"BBM_{params.get('period', 20)}_{params.get('std_dev', 2.0)}"
        return None

    for (indicator_name, params) in unique_indicators.values():
        try:
            # --- FIX: Check if the indicator column already exists ---
            expected_col = get_column_name_for_check(indicator_name, params)
            if expected_col and expected_col in df_with_indicators.columns:
                logger.debug(f"Indicator '{indicator_name}' with params {params} already found. Skipping calculation.")
                continue
            # --- END FIX ---

            # Map user-friendly names to pandas-ta function names and parameters
            if indicator_name == "SMA":
                df_with_indicators.ta.sma(length=params.get("period", 20), append=True)
            elif indicator_name == "EMA":
                df_with_indicators.ta.ema(length=params.get("period", 14), append=True)
            elif indicator_name == "RSI":
                df_with_indicators.ta.rsi(length=params.get("period", 14), append=True)
            elif indicator_name == "MACD":
                df_with_indicators.ta.macd(fast=params.get("fast_period", 12), slow=params.get("slow_period", 26),
                                           signal=params.get("signal_period", 9), append=True)
            elif indicator_name == "Bollinger Bands":
                df_with_indicators.ta.bbands(length=params.get("period", 20), std=params.get("std_dev", 2), append=True)
        except Exception as e:
            logger.error(f"Error calculating indicator '{indicator_name}' with pandas-ta: {e}")

    # --- Step 2: Evaluate conditions using the new indicator columns ---
    signals = pd.Series(index=df.index, dtype='object')

    for i in range(1, len(df_with_indicators)):
        all_conditions_met = True
        for cond in strategy_conditions:
            try:
                # --- Get Column Names from pandas-ta conventions ---
                def get_column_name(indicator, params):
                    if indicator == 'SMA':
                         return f"SMA_{params.get('period', 20)}"
                    if indicator == 'EMA': return f"EMA_{params.get('period', 14)}"
                    if indicator == 'RSI': return f"RSI_{params.get('period', 14)}"
                    if indicator == 'MACD': return f"MACD_{params.get('fast_period', 12)}_{params.get('slow_period', 26)}_{params.get('signal_period', 9)}"
                    if indicator == 'Bollinger Bands':
                        band = params.get('band', 'upper').upper()[0]  # BBU, BBM, BBL
                        return f"BB{band}_{params.get('period', 20)}_{params.get('std_dev', 2.0)}"
                    if indicator in ["Close Price", "Open Price", "High Price", "Low Price", "Volume"]:
                        return indicator.split(" ")[0].lower()
                    return None

                # --- Get Current Values ---
                left_col = get_column_name(cond["left_indicator"], cond.get("left_params", {}))
                right_col = get_column_name(cond["right_indicator"], cond.get("right_params", {}))

                left_current = float(cond["left_value"]) if cond["left_indicator"] == "Fixed Value" else \
                df_with_indicators[left_col].iloc[i]
                right_current = float(cond["right_value"]) if cond["right_indicator"] == "Fixed Value" else \
                df_with_indicators[right_col].iloc[i]

                if pd.isna(left_current) or pd.isna(right_current):
                    all_conditions_met = False
                    break

                # --- Evaluate Condition ---
                comparison = cond["comparison"]
                condition_result = False
                if comparison in ["Crosses Above", "Crosses Below"]:
                    left_prev = float(cond["left_value"]) if cond["left_indicator"] == "Fixed Value" else \
                    df_with_indicators[left_col].iloc[i - 1]
                    right_prev = float(cond["right_value"]) if cond["right_indicator"] == "Fixed Value" else \
                    df_with_indicators[right_col].iloc[i - 1]

                    if pd.isna(left_prev) or pd.isna(right_prev):
                        all_conditions_met = False
                        break

                    if comparison == "Crosses Above" and left_prev <= right_prev and left_current > right_current:
                        condition_result = True
                    elif comparison == "Crosses Below" and left_prev >= right_prev and left_current < right_current:
                        condition_result = True
                else:
                    condition_result = evaluate_condition(left_current, right_current, comparison)

                if not condition_result:
                    all_conditions_met = False
                    break
            except (KeyError, IndexError) as e:
                logger.error(f"Error accessing data for condition at index {i}: {cond}. Error: {e}")
                all_conditions_met = False
                break

        if all_conditions_met:
            signals.iloc[i] = signal_type

    return signals

async def backtest_short_sell(df, initial_investment: float, stop_loss_atr_mult: float, target_atr_mult: float):
    portfolio_value = initial_investment
    tradebook = []
    position = None
    entry_price = 0
    atr = calculate_atr(df)
    for i, row in df.iterrows():
        if position is None:
            signal = check_short_sell_signal(row, atr.iloc[i])
            if signal == "SELL":
                position = "short"
                entry_price = row["close"]
                stop_loss = entry_price * (1 + stop_loss_atr_mult * atr.iloc[i] / entry_price)
                target = entry_price * (1 - target_atr_mult * atr.iloc[i] / entry_price)
        elif position == "short":
            if row["high"] >= stop_loss:
                exit_price = stop_loss
                profit = (entry_price - exit_price) * initial_investment / entry_price
                portfolio_value += profit
                tradebook.append({
                    "Date": row["timestamp"],
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "Profit": profit,
                    "PortfolioValue": portfolio_value
                })
                position = None
            elif row["low"] <= target:
                exit_price = target
                profit = (entry_price - exit_price) * initial_investment / entry_price
                portfolio_value += profit
                tradebook.append({
                    "Date": row["timestamp"],
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "Profit": profit,
                    "PortfolioValue": portfolio_value
                })
                position = None
    total_trades = len(tradebook)
    winning_trades = len([t for t in tradebook if t["Profit"] > 0])
    total_profit = portfolio_value - initial_investment
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    yearly_summary = pd.DataFrame(tradebook).groupby(pd.to_datetime(tradebook["Date"]).dt.year).agg({
        "Profit": "sum",
        "PortfolioValue": "last"
    }).to_dict()
    return {
        "InitialInvestment": initial_investment,
        "StopLossATRMult": stop_loss_atr_mult,
        "TargetATRMult": target_atr_mult,
        "FinalPortfolioValue": portfolio_value,
        "TotalProfit": total_profit,
        "WinRate": win_rate,
        "TotalTrades": total_trades,
        "WinningTrades": winning_trades,
        "LosingTrades": total_trades - winning_trades,
        "YearlySummary": yearly_summary,
        "Tradebook": pd.DataFrame(tradebook)
    }

def check_short_sell_signal(row, atr):
    return "SELL" if row["close"] > row["open"] else None


# Helper functions to refactor backtest_strategy
async def get_historical_dataframe(trading_symbol, instrument_token, timeframe, start_date, end_date, db, nse_db):
    data = await get_historical_data(upstox_api=None, upstox_access_token=None, trading_symbol=trading_symbol,
                                     from_date=start_date, to_date=end_date, unit="days", interval=timeframe,
                                     instrument=instrument_token, db=db, nse_db=nse_db, source="default")
    if not data.data:
        raise ValueError(f"No historical data for {instrument_token}")

    df = pd.DataFrame([p.model_dump() for p in data.data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def get_strategy_details(strategy: str) -> (Optional[Callable], str, bool):
    if strategy.startswith("{"):
        try:
            strategy_data = json.loads(strategy)
            return None, strategy_data.get("name", "Custom Strategy"), True
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON for custom strategy: {strategy}")
    else:
        strategy_func_map = {
            "MACD Crossover": macd_strategy,
            "Bollinger Bands": bollinger_band_strategy,
            "RSI Oversold/Overbought": rsi_strategy,
        }
        strategy_func = strategy_func_map.get(strategy)
        if not strategy_func and strategy != "Short Sell Optimization":
            raise ValueError(f"Unsupported predefined strategy: {strategy}")
        return strategy_func, strategy, False

async def get_mf_sips(api, db: AsyncSession, user_id: str) -> List[MFSIPResponse]:
    try:
        sips = api.mf_sips()
        query = """
            SELECT * FROM mf_sips WHERE user_id = :user_id AND status = 'ACTIVE'
        """
        db_sips = await async_fetch_query(db, text(query), {"user_id": user_id})
        sip_list = []
        for sip in sips:
            db_sip = next((ds for ds in db_sips if ds["sip_id"] == sip["sip_id"]), None)
            if db_sip:
                sip_list.append(MFSIPResponse(
                    sip_id=sip["sip_id"],
                    scheme_code=sip["fund"],
                    amount=sip["amount"],
                    frequency=sip["frequency"],
                    start_date=datetime.strptime(sip["start_date"], "%Y-%m-%d"),
                    status=sip["status"],
                    user_id=user_id,
                    created_at=db_sip["created_at"]
                ))
        return sip_list
    except Exception as e:
        logger.error(f"Error listing SIPs for user {user_id}: {str(e)}")
        raise

def evaluate_condition(left_value, right_value, comparison):
    if comparison == ">":
        return left_value > right_value
    elif comparison == "<":
        return left_value < right_value
    elif comparison == ">=":
        return left_value >= right_value
    elif comparison == "<=":
        return left_value <= right_value
    elif comparison == "==":
        # Use a small epsilon for float comparison
        return abs(left_value - right_value) < 1e-9
    return False

async def calculate_portfolio_current_value(
    portfolio_id: str,
    symbols_data: List[Any],
    upstox_api=None,
    kite_api=None,
    trading_db: AsyncSession = None,
    fallback_price: float = 0.0
) -> float:
    """
    Optimized function to calculate current portfolio value from LTP data.
    This eliminates redundant code across multiple endpoints.
    
    Args:
        portfolio_id: Portfolio ID to calculate value for
        symbols_data: List of symbol configurations from portfolio
        upstox_api: Upstox API client (optional)
        kite_api: Zerodha API client (optional)
        trading_db: Database session
        fallback_price: Default price if LTP is unavailable
        
    Returns:
        float: Calculated current portfolio value
    """
    try:
        # Extract symbols from the portfolio
        portfolio_symbols = []
        for symbol_config in symbols_data:
            if isinstance(symbol_config, dict):
                symbol = symbol_config.get('symbol')
            else:
                symbol = symbol_config
            if symbol:
                portfolio_symbols.append(symbol)
        
        if not portfolio_symbols:
            logger.warning(f"No symbols found for portfolio {portfolio_id}")
            return 0.0
        
        # Get current market prices for all portfolio symbols
        logger.info(f"Requesting LTP for symbols: {portfolio_symbols}")
        ltp_data = await get_ltp(upstox_api.get("market_data_v3"), kite_api, portfolio_symbols, trading_db)
        logger.info(f"Received LTP data: {[f'{item.trading_symbol}:{item.last_price}' for item in ltp_data] if ltp_data else 'None'}")
        
        # Calculate current value based on units held and current prices
        calculated_current_value = 0.0
        
        # Get current units for each symbol from sip_actual_trades (only executed trades)
        for symbol in portfolio_symbols:
            # Get total units held for this symbol (only from executed trades)
            units_query = text("""
                SELECT COALESCE(SUM(units), 0) as total_units
                FROM sip_actual_trades 
                WHERE portfolio_id = :portfolio_id AND symbol = :symbol AND execution_status = 'EXECUTED'
            """)
            
            units_result = await trading_db.execute(units_query, {
                'portfolio_id': portfolio_id,
                'symbol': symbol
            })
            symbol_units = units_result.scalar() or 0.0
            
            # Get current price for this symbol with enhanced matching
            symbol_price = fallback_price
            if ltp_data:
                for ltp_item in ltp_data:
                    # Try exact match first
                    if ltp_item.trading_symbol == symbol:
                        symbol_price = ltp_item.last_price
                        break
                    # Try case-insensitive match
                    elif ltp_item.trading_symbol.upper() == symbol.upper():
                        symbol_price = ltp_item.last_price
                        break
                    # Try removing any suffix (like .NS)
                    elif ltp_item.trading_symbol.split('.')[0].upper() == symbol.upper():
                        symbol_price = ltp_item.last_price
                        break
            
            # Add to total portfolio value
            calculated_current_value += symbol_units * symbol_price
            logger.info(f"Symbol: {symbol}, Units: {symbol_units}, Price: {symbol_price}, Value: {symbol_units * symbol_price}")
        
        logger.info(f"Total calculated current value for portfolio {portfolio_id}: {calculated_current_value}")
        return calculated_current_value
        
    except Exception as value_error:
        logger.warning(f"Error calculating current value for portfolio {portfolio_id}: {value_error}, using fallback calculation")
        # Fallback: calculate based on stored units with fallback price (only executed trades)
        try:
            fallback_query = text("""
                SELECT COALESCE(SUM(units), 0) as total_units
                FROM sip_actual_trades 
                WHERE portfolio_id = :portfolio_id AND execution_status = 'EXECUTED'
            """)
            
            fallback_result = await trading_db.execute(fallback_query, {
                'portfolio_id': portfolio_id
            })
            total_units = fallback_result.scalar() or 0.0
            return total_units * fallback_price
        except Exception as fallback_error:
            logger.error(f"Fallback calculation also failed for portfolio {portfolio_id}: {fallback_error}")
            return 0.0
