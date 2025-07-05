import asyncio
import json
import logging
import inspect
import os
import sys
from datetime import datetime, time as date_time, timedelta
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
from sqlalchemy.future import select
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
from common_utils.fetch_db_data import *
from common_utils.utils import notify
from common_utils import indicators
from common_utils.upstox_utils import get_symbol_for_instrument
from models import Order, ScheduledOrder, QueuedOrder, User, GTTOrder, Instrument
from schemas import Order as OrderSchema, ScheduledOrder as ScheduledOrderSchema, OrderHistory, Trade, QuoteResponse, \
    OHLCResponse, LTPResponse, HistoricalDataResponse, Instrument as InstrumentSchema, HistoricalDataPoint, MFSIPResponse, MFSIPRequest
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
from common_utils.indicators import calculate_ema, calculate_rsi, calculate_linear_regression, calculate_atr, \
    calculate_macd, calculate_bollinger_bands, calculate_stochastic_oscillator, calculate_sma
from common_utils.predefined_strategies import check_macd_crossover, check_bollinger_band_signals, check_stochastic_signals, \
    check_support_resistance_breakout, macd_strategy, bollinger_band_strategy, rsi_strategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

strategy_tasks = {}

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
        self.order_queue: Optional[asyncio.Queue] = None
        self.monitor_tasks: List[asyncio.Task] = []
        logger.info("OrderMonitor initialized")

    async def initialize_queue(self):
        if self.order_queue is None:
            self.order_queue = asyncio.Queue()
            logger.info("Order queue initialized")

    async def run_scheduled_tasks(self, user_apis: Dict[str, Dict[str, Any]]):
        """Run periodic tasks for syncing order statuses and monitoring GTT orders - PROPER FIX"""
        logger.info("Starting OrderMonitor scheduled tasks")
        try:
            while self.running:
                try:
                    # CORRECT: Use the get_db dependency function properly as an async generator
                    async for db in get_db():
                        try:
                            # Process user APIs if available
                            if user_apis:
                                for user_id, apis in user_apis.items():
                                    try:
                                        upstox_api = apis.get("upstox", {}).get("order")
                                        zerodha_api = apis.get("zerodha", {}).get("kite")

                                        if upstox_api or zerodha_api:
                                            await self.sync_order_statuses(upstox_api, zerodha_api, db)

                                        if upstox_api:
                                            await self.monitor_gtt_orders(upstox_api, db)

                                    except Exception as user_error:
                                        logger.error(f"Error processing user {user_id}: {user_error}")
                            else:
                                logger.debug("No user APIs to process")

                        except Exception as processing_error:
                            logger.error(f"Error in scheduled tasks processing: {processing_error}")

                        # Important: Break after processing to avoid staying in the generator loop
                        break

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

    async def sync_order_statuses(self, upstox_api: Optional[Any], zerodha_api: Optional[Any],
                                  db: AsyncSession) -> bool:
        """Sync order statuses with proper error handling"""
        try:
            # Only proceed if we have at least one API
            if not (upstox_api or zerodha_api):
                logger.debug("No APIs available for order status sync")
                return True

            stmt = select(Order).where(
                func.lower(Order.status).in_([status.lower() for status in ["open", "pending", "trigger pending"]])
            )
            result = await db.execute(stmt)
            orders = result.scalars().all()

            if not orders:
                logger.debug("No open orders to sync")
                return True

            for order in orders:
                try:
                    if order.broker == "Upstox" and upstox_api:
                        order_status = upstox_api.get_order_status(order_id=order.order_id).data.status
                    elif order.broker == "Zerodha" and zerodha_api:
                        order_status = zerodha_api.order_history(order_id=order.order_id)[-1]["status"].lower()
                    else:
                        logger.debug(f"Skipping order {order.order_id}: API not available for broker {order.broker}")
                        continue

                    if order_status:
                        order.status = order_status.lower()
                        logger.info(f"Updated order {order.order_id} status to {order.status}")

                        if order.status in ["complete", "rejected", "cancelled"]:
                            result = await db.execute(select(User).filter(User.user_id == order.user_id))
                            user = result.scalars().first()
                            if user:
                                await notify(
                                    f"Order Status Update: {order.order_id}",
                                    f"Order {order.order_id} for {order.trading_symbol} is {order.status}",
                                    user.email
                                )
                except Exception as order_error:
                    logger.error(f"Error syncing order {order.order_id}: {order_error}")

            await db.commit()
            return True

        except Exception as e:
            logger.error(f"Error in sync_order_statuses: {e}")
            try:
                await db.rollback()
            except:
                pass  # Session might already be closed
            return False

    async def monitor_order(self, order_id: str, instrument_token: str, trading_symbol: str, transaction_type: str,
                           quantity: int, product_type: str, stop_loss_price: Optional[float], target_price: Optional[float],
                           api: Any, broker: str, db: AsyncSession, upstox_apis: Dict[str, Any], kite_apis: Dict[str, Any]):
        logger.info(f"Starting monitor_order for order {order_id}")
        task = asyncio.current_task()
        self.monitor_tasks.append(task)
        try:
            await self.initialize_queue()
            # Fetch the Order instance to access user_id
            result = await db.execute(select(Order).filter(Order.order_id == order_id, Order.broker == broker))
            order: Optional[Order] = result.scalars().first()
            if not order:
                logger.error(f"Order {order_id} not found in database")
                return

            if stop_loss_price or target_price:
                logger.info(
                    f"Queuing stop-loss/target for order {order_id}: stop_loss={stop_loss_price}, target={target_price}")
                sl_transaction = "BUY" if transaction_type == "SELL" else "SELL"
                if stop_loss_price:
                    await self.order_queue.put({
                        "queued_order_id": f"queued_{order_id}_sl",
                        "parent_order_id": order_id,
                        "instrument_token": instrument_token,
                        "trading_symbol": trading_symbol,
                        "transaction_type": sl_transaction,
                        "quantity": quantity,
                        "order_type": "LIMIT",
                        "price": stop_loss_price,
                        "trigger_price": 0,
                        "product_type": product_type,
                        "validity": "DAY",
                        "is_gtt": "False",
                        "status": "QUEUED",
                        "broker": broker,
                        "user_id": order.user_id
                    })
                if target_price:
                    await self.order_queue.put({
                        "queued_order_id": f"queued_{order_id}_target",
                        "parent_order_id": order_id,
                        "instrument_token": instrument_token,
                        "trading_symbol": trading_symbol,
                        "transaction_type": sl_transaction,
                        "quantity": quantity,
                        "order_type": "LIMIT",
                        "price": target_price,
                        "trigger_price": 0,
                        "product_type": product_type,
                        "validity": "DAY",
                        "is_gtt": "False",
                        "status": "QUEUED",
                        "broker": broker,
                        "user_id": order.user_id
                    })
                await self._store_queued_orders(db)
                result = await db.execute(select(User).filter(User.user_id == order.user_id))
                user = result.scalars().first()
                if user:
                    await notify("SL/Target Orders Queued", f"Queued SL and target for {trading_symbol}", user.email)

            max_attempts = 60
            attempt = 0
            backoff = 5
            max_backoff = 30
            while self.running and attempt < max_attempts:
                try:
                    order_status = (api.get_order_status(order_id=order_id).data.status if broker == "Upstox"
                                    else api.order_history(order_id=order_id)[-1]["status"].lower())
                    await self._update_order_status(order_id, order_status, broker, db)
                    if order_status.lower() == "complete":
                        await self._process_queued_orders(order_id, instrument_token, trading_symbol, api, broker, db,
                                                          upstox_apis, kite_apis)
                        break
                    elif order_status.lower() in ["rejected", "cancelled", "triggered"]:
                        await self._clear_queued_orders(order_id, db)
                        break
                    attempt += 1
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, max_backoff)
                except Exception as e:
                    logger.error(f"Error monitoring order {order_id}: {str(e)}")
                    break
        finally:
            if task in self.monitor_tasks:
                self.monitor_tasks.remove(task)

    async def monitor_gtt_orders(self, upstox_api: Optional[Any], db: AsyncSession):
        stmt = select(GTTOrder).where(GTTOrder.status == "PENDING", GTTOrder.broker == "Upstox")
        result = await db.execute(stmt)
        gtt_orders: List[GTTOrder] = result.scalars().all()
        for gtt in gtt_orders:
            try:
                ltp_data = await get_ltp(upstox_api, None, [gtt.instrument_token])
                ltp = ltp_data[0].last_price
                if gtt.trigger_type == "single" and ltp >= gtt.trigger_price:
                    await place_order(
                        api=upstox_api,
                        instrument_token=gtt.instrument_token,
                        trading_symbol=gtt.trading_symbol,
                        transaction_type=gtt.transaction_type,
                        quantity=gtt.quantity,
                        price=gtt.limit_price,
                        order_type="LIMIT",
                        broker="Upstox",
                        db=db,
                        user_id=gtt.user_id
                    )
                    gtt.status = "TRIGGERED"
                    await db.commit()
                    result = await db.execute(select(User).filter(User.user_id == gtt.user_id))
                    user = result.scalars().first()
                    if user:
                        await notify(
                            f"GTT Order Triggered: {gtt.gtt_order_id}",
                            f"GTT order for {gtt.trading_symbol} triggered at ₹{ltp}",
                            user.email
                        )
            except Exception as e:
                logger.error(f"Error monitoring GTT order {gtt.gtt_order_id}: {str(e)}")
        await asyncio.sleep(10)

    async def _update_order_status(self, order_id: str, status: str, broker: str, db: AsyncSession):
        query = """
            UPDATE orders 
            SET status = :status 
            WHERE order_id = :order_id AND broker = :broker
        """
        await async_execute_query(db, text(query), {"status": status, "order_id": order_id, "broker": broker})

    async def _store_queued_orders(self, db: AsyncSession):
        logger.info("Storing queued orders")
        while not self.order_queue.empty():
            order = await self.order_queue.get()
            logger.info(f"Inserting queued order: {order['queued_order_id']}")
            try:
                await async_execute_query(db, text("""
                    INSERT INTO queued_orders (
                        queued_order_id, parent_order_id, instrument_token, trading_symbol, transaction_type, 
                        quantity, order_type, price, trigger_price, product_type, validity, is_gtt, status,
                        broker, user_id)
                     VALUES (
                        :queued_order_id, :parent_order_id, :instrument_token, :trading_symbol, :transaction_type, 
                        :quantity, :order_type, :price, :trigger_price, :product_type, :validity, :is_gtt, :status,
                        :broker, :user_id)
                """), order)
                await db.commit()
                logger.info(f"Successfully inserted queued order: {order['queued_order_id']}")
            except Exception as e:
                logger.error(f"Error inserting queued order {order['queued_order_id']}: {str(e)}")
                await db.rollback()
                raise
            self.order_queue.task_done()

    async def _process_queued_orders(self, order_id: str, instrument_token: str, trading_symbol: str, api: Any,
                                     broker: str, db: AsyncSession, upstox_apis: Dict[str, Any], kite_apis: Dict[str, Any]):
        query = """
            SELECT * FROM queued_orders 
            WHERE parent_order_id = :order_id AND status = 'QUEUED'
        """
        queued_orders = await async_fetch_query(db, text(query), {"order_id": order_id})
        for row in queued_orders:
            try:
                await place_order(
                    api=api,
                    instrument_token=instrument_token,
                    trading_symbol=trading_symbol,
                    transaction_type=row["transaction_type"],
                    quantity=row["quantity"],
                    price=row["price"],
                    order_type=row["order_type"],
                    trigger_price=row["trigger_price"],
                    is_amo=False,
                    product_type=row["product_type"],
                    validity=row["validity"],
                    stop_loss=None,
                    target=None,
                    broker=broker,
                    db=db,
                    upstox_apis=upstox_apis,
                    kite_apis=kite_apis,
                    user_id=row.get("user_id", "default_user")
                )
                update_query = """
                    UPDATE queued_orders 
                    SET status = 'PLACED' 
                    WHERE queued_order_id = :queued_order_id
                """
                await async_execute_query(db, text(update_query), {"queued_order_id": row["queued_order_id"]})
            except Exception as e:
                logger.error(f"Error processing queued order for {order_id}: {str(e)}")

    async def _clear_queued_orders(self, order_id: str, db: AsyncSession):
        query = """
            UPDATE queued_orders 
            SET status = 'CANCELLED' 
            WHERE parent_order_id = :order_id AND status = 'QUEUED'
        """
        await async_execute_query(db, text(query), {"order_id": order_id})

    async def cancel_all_tasks(self):
        """Cancel all monitoring tasks"""
        self.running = False
        for task in self.monitor_tasks:
            if not task.done():
                task.cancel()

        if self.monitor_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.monitor_tasks, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some monitor tasks didn't cancel in time")

        self.monitor_tasks.clear()
        logger.info("All OrderMonitor tasks cancelled")

    async def delete_gtt_order(self, api: Optional[Any], gtt_id: str, db: AsyncSession):
        try:
            result = await db.execute(select(GTTOrder).filter(GTTOrder.gtt_order_id == gtt_id))
            gtt_order = result.scalars().first()
            if not gtt_order:
                logger.error(f"GTT order {gtt_id} not found in database")
                raise HTTPException(status_code=404, detail=f"GTT order {gtt_id} not found")
            if api:
                api.delete_gtt(trigger_id=gtt_id)
            query = "DELETE FROM gtt_orders WHERE gtt_order_id = :gtt_id"
            await async_execute_query(db, text(query), {"gtt_id": gtt_id})
            result = await db.execute(select(User).filter(User.user_id == gtt_order.user_id))
            user = result.scalars().first()
            if user:
                await notify("GTT Order Deleted", f"GTT order {gtt_id} deleted", user.email)
            return {"status": "success", "message": f"GTT order {gtt_id} deleted"}
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
        """Start OrderManager with proper database session handling - PROPER FIX"""
        logger.info("OrderManager starting...")
        try:
            # Load scheduled orders first using the correct pattern
            async for db in get_db():
                try:
                    await self._load_scheduled_orders(db)
                    logger.info("Scheduled orders loaded successfully")
                except Exception as load_error:
                    logger.error(f"Error loading scheduled orders: {load_error}")
                break  # Important: Break after one iteration

            # Start the continuous processing loop
            await self._process_scheduled_orders_loop(user_apis)

        except Exception as e:
            logger.error(f"Error starting OrderManager: {e}")

    async def _load_scheduled_orders(self, db: AsyncSession):
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
        """Continuous loop for processing scheduled orders - PROPER FIX"""
        logger.info("Starting scheduled orders processing loop")
        try:
            while self.running:
                try:
                    # Create a fresh database session for each iteration using the correct pattern
                    async for db in get_db():
                        try:
                            await self._process_scheduled_orders(db, user_apis)
                        except Exception as processing_error:
                            logger.error(f"Error processing scheduled orders: {processing_error}")
                        break  # Important: Break after one iteration

                except Exception as session_error:
                    logger.error(f"Error with database session in processing loop: {session_error}")

                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute

        except asyncio.CancelledError:
            logger.info("Scheduled order processing cancelled")
            self.running = False
            raise
        except Exception as e:
            logger.error(f"Critical error in scheduled orders loop: {e}")
            self.running = False

    async def _process_scheduled_orders(self, db: AsyncSession, user_apis: Dict[str, Dict[str, Any]]):
        """Process scheduled orders for a single iteration - FINAL FIX"""
        try:
            query = """
                SELECT * FROM scheduled_orders 
                WHERE status = :status
            """
            scheduled_orders = await async_fetch_query(db, text(query), {"status": "PENDING"})

            if not scheduled_orders:
                logger.debug("No pending scheduled orders found")
                return

            now = datetime.now()

            for order in scheduled_orders:
                try:
                    # Parse schedule_datetime if it's a string
                    schedule_datetime = order["schedule_datetime"]
                    if isinstance(schedule_datetime, str):
                        schedule_datetime = datetime.strptime(schedule_datetime, "%Y-%m-%d %H:%M:%S")

                    # Check if it's time to execute
                    if schedule_datetime <= now:
                        user_id = order["user_id"]
                        broker = order["broker"]

                        # Get user APIs
                        if user_id in user_apis:
                            apis = user_apis[user_id]

                            # Execute the scheduled order
                            if broker == "Upstox" and apis.get("upstox", {}).get("order"):
                                api = apis["upstox"]["order"]
                            elif broker == "Zerodha" and apis.get("zerodha", {}).get("kite"):
                                api = apis["zerodha"]["kite"]
                            else:
                                logger.warning(f"API not available for user {user_id}, broker {broker}")
                                continue

                            # Place the order
                            try:
                                # Use your existing place_order function
                                result = await place_order(
                                    api=api,
                                    instrument_token=order["instrument_token"],
                                    trading_symbol=order["trading_symbol"],
                                    transaction_type=order["transaction_type"],
                                    quantity=order["quantity"],
                                    price=order.get("price", 0),
                                    order_type=order["order_type"],
                                    trigger_price=order.get("trigger_price", 0),
                                    is_amo=order.get("is_amo", False),
                                    product_type=order["product_type"],
                                    validity=order.get("validity", "DAY"),
                                    stop_loss=order.get("stop_loss"),
                                    target=order.get("target"),
                                    broker=broker,
                                    db=db,
                                    upstox_apis=apis.get("upstox", {}),
                                    kite_apis=apis.get("zerodha", {}),
                                    user_id=user_id,
                                    order_monitor=self.order_monitor
                                )

                                # Update order status to EXECUTED
                                update_query = """
                                    UPDATE scheduled_orders 
                                    SET status = 'EXECUTED', executed_at = :executed_at
                                    WHERE scheduled_order_id = :order_id
                                """
                                await async_execute_query(db, text(update_query), {
                                    "executed_at": now,
                                    "order_id": order["scheduled_order_id"]
                                })

                                logger.info(
                                    f"✅ Executed scheduled order {order['scheduled_order_id']} for user {user_id}")

                            except Exception as execution_error:
                                logger.error(
                                    f"❌ Error executing scheduled order {order['scheduled_order_id']}: {execution_error}")

                                # Update order status to FAILED
                                update_query = """
                                    UPDATE scheduled_orders 
                                    SET status = 'FAILED', error_message = :error_message
                                    WHERE scheduled_order_id = :order_id
                                """
                                await async_execute_query(db, text(update_query), {
                                    "error_message": str(execution_error),
                                    "order_id": order["scheduled_order_id"]
                                })
                        else:
                            logger.warning(f"User {user_id} not found in user_apis")

                except Exception as order_error:
                    logger.error(f"Error processing scheduled order: {order_error}")

        except Exception as e:
            logger.error(f"Error in _process_scheduled_orders: {e}")

    async def _execute_scheduled_order(self, order, db: AsyncSession, upstox_apis=None, kite_apis=None):
        async with self.order_lock:
            try:
                schedule_datetime = pd.to_datetime(order.get("schedule_datetime"))
                if schedule_datetime and datetime.now() >= schedule_datetime:
                    api = upstox_apis["order"] if order["broker"] == "Upstox" else kite_apis["kite"]
                    if not api:
                        logger.warning(f"Cannot execute scheduled order {order['scheduled_order_id']}: API not initialized for broker {order['broker']}")
                        return
                    result = await place_order(
                        api=api,
                        instrument_token=order["instrument_token"],
                        trading_symbol=order["trading_symbol"],
                        transaction_type=order["transaction_type"],
                        quantity=order["quantity"],
                        price=order["price"],
                        order_type=order["order_type"],
                        trigger_price=order["trigger_price"],
                        is_amo=order["is_amo"] == "True",
                        product_type=order["product_type"],
                        validity=order["validity"],
                        stop_loss=order["stop_loss"],
                        target=order["target"],
                        broker=order["broker"],
                        db=db,
                        upstox_apis=upstox_apis,
                        kite_apis=kite_apis,
                        user_id=order.get("user_id", "default_user")
                    )
                    if result:
                        order_id = result.data.order_id if order["broker"] == "Upstox" else result
                        query = """
                            UPDATE scheduled_orders 
                            SET status = 'EXECUTED' 
                            WHERE scheduled_order_id = :order_id
                        """
                        await async_execute_query(db, text(query), {"order_id": order["scheduled_order_id"]})
                        query = "DELETE FROM scheduled_orders WHERE scheduled_order_id = :order_id"
                        await async_execute_query(db, text(query), {"order_id": order["scheduled_order_id"]})
                        logger.info(f"Scheduled order {order['scheduled_order_id']} executed with order ID {order_id}")
                        result = await db.execute(select(User).filter(User.user_id == order.get("user_id", "default_user")))
                        user = result.scalars().first()
                        if user:
                            await notify("Scheduled Order Executed", f"Order ID: {order_id}", user.email)
            except Exception as e:
                logger.error(f"Error executing scheduled order {order['scheduled_order_id']}: {str(e)}")

    async def place_gtt_order(self, api, instrument_token, trading_symbol, transaction_type, quantity,
                              trigger_type, trigger_price, limit_price, last_price,
                              second_trigger_price=None, second_limit_price=None, broker="Zerodha",
                              db: AsyncSession = None, user_id: str = "default_user"):
        try:
            if broker != "Zerodha":
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
            condition = {
                "exchange": "NSE",
                "tradingsymbol": trading_symbol,
                "last_price": last_price
            }
            if trigger_type == "single":
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
            response = api.place_gtt(
                trigger_type=trigger_type,
                tradingsymbol=trading_symbol,
                exchange="NSE",
                trigger_values=condition["trigger_values"],
                last_price=condition["last_price"],
                orders=orders
            )
            logger.info(f"GTT Order placed successfully - {response}")
            gtt_id = str(response.get("trigger_id"))

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
                "status": "active",
                "broker": broker,
                "created_at": datetime.now(),
                "user_id": user_id
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
                               second_limit_price: Optional[float] = None, db: AsyncSession = None):
        try:
            stmt = select(GTTOrder).where(GTTOrder.gtt_order_id == gtt_id)
            result = await db.execute(stmt)
            gtt_order = result.scalars().first()
            if not gtt_order:
                raise ValueError(f"GTT order {gtt_id} not found")
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
            query = """
                UPDATE gtt_orders 
                SET trigger_type = :trigger_type, trigger_price = :trigger_price, limit_price = :limit_price,
                    second_trigger_price = :second_trigger_price, second_limit_price = :second_limit_price,
                    quantity = :quantity, last_price = :last_price
                WHERE gtt_order_id = :gtt_id
            """
            await async_execute_query(db, text(query), {
                "gtt_id": gtt_id,
                "trigger_type": trigger_type,
                "trigger_price": trigger_price,
                "limit_price": limit_price,
                "second_trigger_price": second_trigger_price,
                "second_limit_price": second_limit_price,
                "quantity": quantity,
                "last_price": last_price
            })
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
                "order": upstox_client.OrderApi(api_client),
                "portfolio": upstox_client.PortfolioApi(api_client),
                "market_data": upstox_client.MarketQuoteApi(api_client),
                "user": upstox_client.UserApi(api_client),
                "history": upstox_client.HistoryApi(api_client)
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
                      trigger_price=0, is_amo=False, product_type="D", validity='DAY', stop_loss=None, target=None,
                      broker="Upstox", db: AsyncSession = None, upstox_apis=None, kite_apis=None,
                      user_id: str = "default_user", order_monitor=None):
    try:
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if price < 0 or trigger_price < 0:
            raise ValueError("Price and trigger price cannot be negative")
        if broker == "Upstox":
            order = upstox_client.PlaceOrderRequest(
                quantity=quantity,
                product=product_type,
                validity=validity,
                price=price,
                tag="StreamlitOrder",
                instrument_token=instrument_token,
                order_type=order_type,
                transaction_type=transaction_type,
                disclosed_quantity=0,
                trigger_price=trigger_price,
                is_amo=is_amo
            )
            response = api.place_order(order, api_version="v2")
            primary_order_id = response.data.order_id
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
                "tag": "StreamlitOrder"
            }
            response = api.place_order(
                variety=api.VARIETY_REGULAR if not is_amo else api.VARIETY_AMO,
                **order_params
            )
            primary_order_id = response
        order_data = pd.DataFrame([{
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
        }])

        await load_sql_data(order_data, "orders", load_type="append", index_required=False, db=db)
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if user:
            await notify(f"Order Placed: {transaction_type} for {trading_symbol}",
                         f"Order ID: {primary_order_id}, Quantity: {quantity}, Price: {price}", user.email)
        logger.info(f"Order placed: {primary_order_id} for {trading_symbol}, user {user_id}")

        if stop_loss or target:
            asyncio.create_task(order_monitor.monitor_order(
                primary_order_id, instrument_token, trading_symbol, transaction_type, quantity,
                product_type, stop_loss, target, api, broker, db, upstox_apis, kite_apis
            ))

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
                validity=modified_params["validity"]
            )
            response = api.modify_order(order_id, modify_request, api_version="v2")
        else:
            zerodha_validity = "DAY" if modified_params["validity"] == "DAY" else "IOC"
            response = api.modify_order(
                variety=api.VARIETY_REGULAR,
                order_id=order_id,
                quantity=modified_params["quantity"],
                order_type=modified_params["order_type"],
                price=modified_params["price"] if modified_params["order_type"] in ["LIMIT", "SL"] else 0,
                trigger_price=modified_params["trigger_price"] if modified_params["order_type"] in ["SL", "SL-M"] else 0,
                validity=zerodha_validity
            )
        query = """
            UPDATE orders 
            SET quantity = :quantity, order_type = :order_type, price = :price, trigger_price = :trigger_price, validity = :validity
            WHERE order_id = :order_id AND broker = :broker
        """
        await async_execute_query(db, text(query), {
            "order_id": order_id,
            "broker": broker,
            "quantity": modified_params["quantity"],
            "order_type": modified_params["order_type"],
            "price": modified_params["price"],
            "trigger_price": modified_params["trigger_price"],
            "validity": modified_params["validity"]
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
            upstox_orders = upstox_api.get_order_book(api_version="v2").data
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

async def get_quotes(upstox_api, kite_api, instruments: List[str]) -> List[QuoteResponse]:
    if not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API required for quotes")
    try:
        response = upstox_api.get_full_market_quote(",".join(instruments), api_version="v2").data
        quotes = []
        for instrument, quote in response.items():
            quote_dict = quote.to_dict()
            quotes.append(QuoteResponse(
                instrument_token=instrument,
                last_price=quote_dict.get("last_price", 0.0),
                volume=quote_dict.get("volume", 0),
                average_price=quote_dict.get("average_price"),
                ohlc={
                    "open": quote_dict.get("ohlc", {}).get("open", 0.0),
                    "high": quote_dict.get("ohlc", {}).get("high", 0.0),
                    "low": quote_dict.get("ohlc", {}).get("low", 0.0),
                    "close": quote_dict.get("ohlc", {}).get("close", 0.0)
                }
            ))
        return quotes
    except Exception as e:
        logger.error(f"Error fetching quotes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_ohlc(upstox_api, kite_api, instruments: List[str]) -> List[OHLCResponse]:
    if not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API required for OHLC")
    try:
        response = upstox_api.get_market_quote_ohlc(",".join(instruments), interval="1d", api_version="v2").data
        ohlc_data = []
        for instrument, ohlc in response.items():
            ohlc_dict = ohlc.to_dict()
            ohlc_data.append(OHLCResponse(
                instrument_token=instrument,
                open=ohlc_dict.get("ohlc", {}).get("open", 0.0),
                high=ohlc_dict.get("ohlc", {}).get("high", 0.0),
                low=ohlc_dict.get("ohlc", {}).get("low", 0.0),
                close=ohlc_dict.get("ohlc", {}).get("close", 0.0),
                volume=ohlc_dict.get("volume")
            ))
        return ohlc_data
    except Exception as e:
        logger.error(f"Error fetching OHLC data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_ltp(upstox_api, kite_api, instruments: List[str]) -> List[LTPResponse]:
    if not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API required for LTP")
    try:
        response = upstox_api.ltp(",".join(instruments), api_version="v2").data
        ltp_data = []
        for instrument, quote in response.items():
            quote_dict = quote.to_dict()
            ltp_data.append(LTPResponse(
                instrument_token=instrument,
                last_price=quote_dict.get("last_price", 0.0)
            ))
        return ltp_data
    except Exception as e:
        logger.error(f"Error fetching LTP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_and_load_index_data(index_name, db: AsyncSession = None) -> pd.DataFrame:
    try:
        url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_name.replace(' ', '%20')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json().get("data", [])
            df = pd.DataFrame(data)
            df["index_name"] = index_name
            df = df[["index_name", "symbol", "open", "dayHigh", "dayLow", "lastPrice", "previousClose","pChange", "yearHigh", "yearLow", "totalTradedVolume"]]
            await load_sql_data(df, "index_data", load_type="replace", index_required=False, db=db)
            return df
        logger.error(f"Failed to fetch index data: {response.text}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching index data: {str(e)}")
        return pd.DataFrame()


async def get_historical_data(upstox_api, upstox_access_token, instrument: str, from_date: str, to_date: str, unit: str,
                              interval: str, db: AsyncSession = None) -> HistoricalDataResponse:
    logger.info(f"Fetching historical data for {instrument} from {from_date} to {to_date}")
    data_points = []

    # Convert dates to datetime objects for comparison
    from_date_dt = datetime.strptime(from_date, "%Y-%m-%d")
    to_date_dt = datetime.strptime(to_date, "%Y-%m-%d")

    # Try openchart first for NSE stocks
    try:
        logger.info(f"Trying openchart for NSE stock: {instrument}")
        data = load_stock_history(instrument, from_date_dt, to_date_dt, interval=interval, load=False)

        if not data.empty:
            for _, row in data.iterrows():
                data_points.append(HistoricalDataPoint(
                    timestamp=row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"])
                ))
            logger.info(f"Successfully fetched {len(data_points)} data points from openchart")
    except Exception as e:
        logger.warning(f"Failed to fetch data from openchart: {str(e)}")

    # If openchart didn't work, try Upstox API
    if not data_points and upstox_api:
        try:
            logger.info(f"Trying Upstox API for {instrument}")
            headers = {"Authorization": f"Bearer {upstox_access_token}"}
            url = f"https://api.upstox.com/v3/historical-candle/{instrument}/{unit}/{interval}/{to_date}/{from_date}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                candles = response.json().get("data", {}).get("candles", [])
                for candle in candles:
                    data_points.append(HistoricalDataPoint(
                        timestamp=datetime.strptime(candle[0], "%Y-%m-%dT%H:%M:%S%z"),
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=int(candle[5])
                    ))
                logger.info(f"Successfully fetched {len(data_points)} data points from Upstox API")
            else:
                logger.error(f"Upstox API error: {response.text}")
        except Exception as e:
            logger.warning(f"Failed to fetch data from Upstox API: {str(e)}")

    # If both external sources failed, try the database
    if not data_points and db:
        try:
            # Extract clean table name from instrument
            table_name = instrument.split("_")[-1].lower()
            logger.info(f"Trying database table '{table_name}' for historical data")

            query = f"""
                SELECT * FROM {table_name}
                WHERE timestamp >= :from_date AND timestamp <= :to_date
                ORDER BY timestamp
            """

            # Use parameterized query safely
            result = await async_fetch_query(
                db,
                text(query),
                {"from_date": from_date_dt, "to_date": to_date_dt}
            )

            if result:
                for row in result:
                    # Convert database row to HistoricalDataPoint
                    date_val = row.get('timestamp')
                    if isinstance(date_val, str):
                        date_val = datetime.strptime(date_val, "%Y-%m-%d")

                    data_points.append(HistoricalDataPoint(
                        timestamp=date_val,
                        open=float(row.get('open', 0)),
                        high=float(row.get('high', 0)),
                        low=float(row.get('low', 0)),
                        close=float(row.get('close', 0)),
                        volume=int(row.get('volume', 0))
                    ))
                logger.info(f"Successfully fetched {len(data_points)} data points from database")
        except Exception as e:
            logger.error(f"Error fetching from database: {str(e)}")

    # If we still don't have data, raise an error
    if not data_points:
        logger.error(f"Could not retrieve historical data for {instrument} from any source")
        raise HTTPException(status_code=404, detail=f"Historical data not available for {instrument}")

    # Create the response with sorted data
    historical_data = HistoricalDataResponse(
        instrument_token=instrument,
        data=sorted(data_points, key=lambda x: x.timestamp)
    )

    logger.info(f"Returning {len(data_points)} historical data points for {instrument}")
    return historical_data


async def fetch_instruments(db: AsyncSession, refresh: bool = False) -> List[InstrumentSchema]:
    try:
        if refresh:
            path = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
            instruments_df = pd.read_json(path)
            instruments_df = instruments_df[
                ['trading_symbol', 'instrument_key', 'exchange', 'instrument_type', 'segment']
            ][(instruments_df['segment'] == 'NSE_EQ') & (instruments_df['instrument_type'] == 'EQ') &
              (~instruments_df['name'].str.contains('TEST', case=False, na=False))]

            await db.execute(text("DELETE FROM instruments"))
            for _, row in instruments_df.iterrows():
                instrument = Instrument(
                    instrument_token=row["instrument_key"],
                    trading_symbol=row["trading_symbol"],
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

def get_order_history(upstox_api, kite_api, order_id: str, broker: str) -> List[OrderHistory]:
    try:
        history = []
        if broker == "Upstox" and upstox_api:
            response = upstox_api.get_order_details(order_id=order_id, api_version="v2").data
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
            response = upstox_api.get_trades_by_order(order_id=order_id, api_version="v2").data
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
        data = await get_historical_data(upstox_api=api, kite_api=None, instrument=instrument_token,
                                         from_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                                         to_date=datetime.now().strftime("%Y-%m-%d"), interval="day")
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
            result = await place_order(
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
            if result:
                order_id = result.data.order_id if broker == "Upstox" else result
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


async def backtest_strategy(instrument_token: str, timeframe: str, strategy: str, params: Dict, start_date: str,
                            end_date: str, ws_callback: Optional[Callable] = None, db: Optional[AsyncSession] = None):
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

    logger.info(f"Starting backtest for {instrument_token} from {start_date} to {end_date}")
    logger.info(f"Strategy: {strategy}, Timeframe: {timeframe}, Params: {params}")

    try:
        # --- 1. Data Fetching ---
        df = await get_historical_dataframe(instrument_token, timeframe, start_date, end_date, db)

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
                    if indicator == 'SMA': return f"SMA_{params.get('period', 20)}"
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
async def get_historical_dataframe(instrument_token, timeframe, start_date, end_date, db):
    data = await get_historical_data(upstox_api=None, upstox_access_token=None, instrument=instrument_token,
                                     from_date=start_date, to_date=end_date, unit="days", interval=timeframe, db=db)
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

async def get_analytics_data(upstox_api, instrument_token: str, upstox_access_token: str, unit: str, interval: str,
                             ema_period: int, rsi_period: int, lr_period: int, stochastic_k: int, stochastic_d: int) -> dict:
    try:
        from_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        data = await get_historical_data(upstox_api, upstox_access_token, instrument_token,
                                         from_date, to_date, unit, interval)
        # Convert HistoricalDataResponse to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": point.timestamp,
                "open": point.open,
                "high": point.high,
                "low": point.low,
                "close": point.close,
                "volume": point.volume
            }
            for point in data.data
        ])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        sma = calculate_sma(df, ema_period)
        ema = calculate_ema(df, ema_period)
        rsi = calculate_rsi(df, rsi_period)
        lr = calculate_linear_regression(df, lr_period)
        macd_line, signal_line, _ = calculate_macd(df)  # Fixed: Unpack all three values
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df)
        stochastic_k_val, stochastic_d_val = calculate_stochastic_oscillator(df, stochastic_k, stochastic_d)
        atr = calculate_atr(df)
        # Replace NaN/inf with None
        result = {
            "sma": [None if pd.isna(x) or not np.isfinite(x) else x for x in sma],
            "ema": [None if pd.isna(x) or not np.isfinite(x) else x for x in ema],
            "rsi": [None if pd.isna(x) or not np.isfinite(x) else x for x in rsi],
            "lr": [None if pd.isna(x) or not np.isfinite(x) else x for x in lr],
            "macd": [None if pd.isna(x) or not np.isfinite(x) else x for x in macd_line],
            "signal": [None if pd.isna(x) or not np.isfinite(x) else x for x in signal_line],
            "bb_upper": [None if pd.isna(x) or not np.isfinite(x) else x for x in bb_upper],
            "bb_middle": [None if pd.isna(x) or not np.isfinite(x) else x for x in bb_middle],
            "bb_lower": [None if pd.isna(x) or not np.isfinite(x) else x for x in bb_lower],
            "stochastic_k": [None if pd.isna(x) or not np.isfinite(x) else x for x in stochastic_k_val],
            "stochastic_d": [None if pd.isna(x) or not np.isfinite(x) else x for x in stochastic_d_val],
            "atr": [None if pd.isna(x) or not np.isfinite(x) else x for x in atr]
        }
        return result
    except Exception as e:
        logger.error(f"Error in get_analytics_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def place_mf_sip(api, sip: MFSIPRequest, db: AsyncSession, user_id: str) -> MFSIPResponse:
    try:
        response = api.mf_sip(
            fund=sip.scheme_code,
            amount=sip.amount,
            frequency=sip.frequency,
            start_date=sip.start_date.strftime("%Y-%m-%d"),
            instalments=-1  # Ongoing SIP
        )
        sip_id = response.get("sip_id")
        sip_data = {
            "sip_id": sip_id,
            "scheme_code": sip.scheme_code,
            "amount": sip.amount,
            "frequency": sip.frequency,
            "start_date": sip.start_date,
            "status": "ACTIVE",
            "user_id": user_id,
            "created_at": datetime.now()
        }
        await async_execute_query(db, text("""
            INSERT INTO mf_sips (
                sip_id, scheme_code, amount, frequency, start_date, status, user_id, created_at
            ) VALUES (
                :sip_id, :scheme_code, :amount, :frequency, :start_date, :status, :user_id, :created_at
            )
        """), sip_data)
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if user:
            await notify("Mutual Fund SIP Created",
                         f"SIP {sip_id} for {sip.scheme_code} with amount ₹{sip.amount} created",
                         user.email)
        return MFSIPResponse(**sip_data)
    except Exception as e:
        logger.error(f"Error creating SIP for user {user_id}: {str(e)}")
        raise

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

async def cancel_mf_sip(api, sip_id: str, db: AsyncSession, user_id: str) -> dict:
    try:
        api.cancel_mf_sip(sip_id=sip_id)
        query = """
            UPDATE mf_sips 
            SET status = 'CANCELLED' 
            WHERE sip_id = :sip_id AND user_id = :user_id
        """
        await async_execute_query(db, text(query), {"sip_id": sip_id, "user_id": user_id})
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if user:
            await notify("Mutual Fund SIP Cancelled",
                         f"SIP {sip_id} cancelled",
                         user.email)
        return {"status": "success", "message": f"SIP {sip_id} cancelled"}
    except Exception as e:
        logger.error(f"Error cancelling SIP {sip_id} for user {user_id}: {str(e)}")
        raise

def get_indicator_value(df, indicator_name, params, index, indicators_module):
    if indicator_name == "Close Price":
        return df["close"].iloc[index]
    elif indicator_name == "Open Price":
        return df["open"].iloc[index]
    elif indicator_name == "High Price":
        return df["high"].iloc[index]
    elif indicator_name == "Low Price":
        return df["low"].iloc[index]
    elif indicator_name == "Volume":
        return df["volume"].iloc[index]
    elif indicator_name == "SMA":
        sma = indicators_module.calculate_sma(df, params.get("period", 20))
        return sma.iloc[index]
    elif indicator_name == "EMA":
        ema = indicators_module.calculate_ema(df, params.get("period", 14))
        return ema.iloc[index]
    elif indicator_name == "RSI":
        rsi = indicators_module.calculate_rsi(df, params.get("period", 14))
        return rsi.iloc[index]
    elif indicator_name == "MACD":
        macd, _, _ = indicators_module.calculate_macd(df, params.get("fast_period", 12), params.get("slow_period", 26), params.get("signal_period", 9))
        return macd.iloc[index]
    elif indicator_name == "Bollinger Bands":
        _, upper, lower = indicators_module.calculate_bollinger_bands(df, params.get("period", 20), params.get("std_dev", 2))
        return upper.iloc[index]  # Example: return upper band
    return None

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


if __name__ == "__main__":
    import asyncio
    from backend.app.database import get_db

    async def main():
        # Example parameters (replace with real values as needed)
        instrument_token = "SBIN"
        timeframe = "day"
        # strategy = "RSI Oversold/Overbought"  # or a custom strategy JSON string
        strategy = {"strategy_id": "1fe9328e-f8cd-405c-9db2-68353fdcb2c3",
                    "user_id": "4fbba468-6a86-4516-8236-2f8abcbfd2ef",
                    "broker": "Zerodha",
                    "name": "RSI",
                    "description": "",
                    "entry_conditions": [{"left_indicator": "RSI",
                                          "left_params": {"period": 14},
                                          "left_value": "",
                                          "comparison": "<=",
                                          "right_indicator": "Fixed Value",
                                          "right_params": "",
                                          "right_value": 30.0}],
                    "exit_conditions": [{"left_indicator": "RSI",
                                         "left_params": {"period": 14},
                                         "left_value": "",
                                         "comparison": ">=",
                                         "right_indicator": "Fixed Value",
                                         "right_params": "",
                                         "right_value": 75.0}],
                    "parameters": {"timeframe": "day",
                                   "position_sizing": 100},
                    "status": "inactive",
                    "created_at": "2025-06-21T17:59:00.011814",
                    "updated_at": "2025-06-21T18:02:08.444443"}
        params = {
            "initial_investment": 100000,
            "stop_loss_percent": 0,
            "trailing_stop_loss_percent": 0,
            "position_sizing_percent": 100,
            "enable_optimization": True,
            'partial_exits': [],
            'optimization_iterations': 20,
            'stop_loss_range': [1.0, 5.0]
        }
        start_date = "2024-01-20"
        end_date = "2025-06-20"

        result = await backtest_strategy(
            instrument_token, timeframe, json.dumps(strategy) if not isinstance(strategy, str) else strategy, params, start_date, end_date, db=get_db()
        )
        print(result)

    asyncio.run(main())