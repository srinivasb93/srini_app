import asyncio
import json
import logging
import os
from datetime import datetime, time as date_time, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import requests
import smtplib
from email.mime.text import MIMEText

import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.sql import text, func
from fastapi import Depends, HTTPException
from kiteconnect import KiteConnect
import upstox_client

from common_utils.db_utils import async_fetch_query, async_execute_query
from common_utils.read_write_sql_data import load_sql_data
from common_utils.fetch_db_data import *
from common_utils.utils import notify
from common_utils import indicators
from common_utils.upstox_utils import get_symbol_for_instrument, get_historical_data_latest, get_market_quote
from models import Order, ScheduledOrder, QueuedOrder, User, GTTOrder, Instrument
from schemas import Order as OrderSchema, ScheduledOrder as ScheduledOrderSchema, OrderHistory, Trade, QuoteResponse, \
    OHLCResponse, LTPResponse, HistoricalDataResponse, Instrument as InstrumentSchema, HistoricalDataPoint, MFSIPResponse, MFSIPRequest
from database import get_db
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
from common_utils.indicators import calculate_ema, calculate_rsi, calculate_linear_regression, calculate_atr, \
    calculate_macd, calculate_bollinger_bands, calculate_stochastic_oscillator, calculate_sma
from common_utils.strategies import check_macd_crossover, check_bollinger_band_signals, check_stochastic_signals, \
    check_support_resistance_breakout, macd_strategy, bollinger_band_strategy, rsi_strategy

logger = logging.getLogger(__name__)

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
        """Run periodic tasks for syncing order statuses and monitoring GTT orders."""
        logger.info("Starting OrderMonitor scheduled tasks")
        try:
            while self.running:
                async for db in get_db():
                    try:
                        for user_id, apis in user_apis.items():
                            upstox_api = apis["upstox"]["order"]
                            zerodha_api = apis["zerodha"]["kite"]
                            await self.sync_order_statuses(upstox_api, zerodha_api, db)
                            if upstox_api:
                                await self.monitor_gtt_orders(upstox_api, db)
                    except Exception as e:
                        logger.error(f"Error in scheduled tasks: {str(e)}")
                await asyncio.sleep(self.polling_interval)
        except asyncio.CancelledError:
            logger.info("OrderMonitor scheduled tasks cancelled")
            self.running = False
            await self.cancel_all_tasks()
            raise
        except Exception as e:
            logger.error(f"Unexpected error in run_scheduled_tasks: {str(e)}")
            self.running = False
            await self.cancel_all_tasks()

    async def sync_order_statuses(self, upstox_api: Optional[Any], zerodha_api: Optional[Any], db: AsyncSession) -> bool:
        try:
            stmt = select(Order).where(
                func.lower(Order.status).in_([status.lower() for status in ["open", "pending", "trigger pending"]])
            )
            result = await db.execute(stmt)
            orders = result.scalars().all()
            for order in orders:
                try:
                    if order.broker == "Upstox" and upstox_api:
                        order_status = upstox_api.get_order_status(order_id=order.order_id).data.status
                    elif order.broker == "Zerodha" and zerodha_api:
                        order_status = zerodha_api.order_history(order_id=order.order_id)[-1]["status"].lower()
                    else:
                        logger.warning(f"Skipping order {order.order_id}: API not initialized for broker {order.broker}")
                        continue
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
                except Exception as e:
                    logger.error(f"Error syncing status for order {order.order_id}: {str(e)}")
            await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error in sync_order_statuses: {str(e)}")
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
        async for db in get_db():
            await self._load_scheduled_orders(db)
        await self._process_scheduled_orders(db, user_apis)

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

    async def _process_scheduled_orders(self, db: AsyncSession, user_apis: Dict[str, Dict[str, Any]]):
        try:
            while self.running:
                query = """
                    SELECT * FROM scheduled_orders 
                    WHERE status = :status
                """
                scheduled_orders = await async_fetch_query(db, text(query), {"status": "PENDING"})
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
                            upstox_api = user_apis.get(user_id, {}).get("upstox", {}).get("order")
                            zerodha_api = user_apis.get(user_id, {}).get("zerodha", {}).get("kite")
                            api = upstox_api if order["broker"] == "Upstox" else zerodha_api
                            if not api:
                                logger.warning(f"API not initialized for user {user_id}, broker {order['broker']}")
                                continue
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
                                user_id=user_id
                            )
                            update_query = """
                                UPDATE scheduled_orders 
                                SET status = 'EXECUTED' 
                                WHERE scheduled_order_id = :order_id
                            """
                            await async_execute_query(db, text(update_query), {"order_id": order["scheduled_order_id"]})
                            logger.info(f"Scheduled order {order['scheduled_order_id']} executed")
                    except Exception as e:
                        logger.error(f"Error processing scheduled order {order['scheduled_order_id']}: {str(e)}")
                await asyncio.sleep(60)  # Check every minute
        except asyncio.CancelledError:
            logger.info("Scheduled order processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _process_scheduled_orders: {str(e)}")

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
            entry_signals = await evaluate_custom_strategy(df, strategy_data.get("entry_conditions", []),
                                                           indicators)
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

async def backtest_strategy(instrument_token: str, timeframe: str, strategy: str, params: Dict, start_date: str, end_date: str, ws_callback=None, db: AsyncSession = None):
    logger.debug(f"Starting backtest for strategy: {strategy}, instrument: {instrument_token}")
    try:
        if not instrument_token:
            raise ValueError("Instrument token is required")
        if not start_date or not end_date:
            raise ValueError("Start and end dates are required")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")

        data = await get_historical_data(upstox_api=None, upstox_access_token=None, instrument=instrument_token,
                                         from_date=start_date, to_date=end_date, unit="days", interval=timeframe, db=db)
        if not data.data:
            logger.error(f"No historical data for {instrument_token}")
            return {"error": {"code": "NO_DATA", "message": "Failed to fetch historical data"}}
        df = pd.DataFrame([{
            "timestamp": point.timestamp,
            "open": point.open,
            "high": point.high,
            "low": point.low,
            "close": point.close,
            "volume": point.volume
        } for point in data.data])
        logger.debug(f"Fetched {len(df)} data points for {instrument_token}")

        # Parse strategy
        if strategy.startswith("{"):
            try:
                strategy_data = json.loads(strategy)
                strategy_name = strategy_data.get("name", "Custom Strategy")
                is_custom = True
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON strategy: {strategy}")
                return {"error": {"code": "INVALID_STRATEGY", "message": "Invalid JSON strategy format"}}
        else:
            strategy_name = strategy
            is_custom = False
            logger.debug(f"Using predefined strategy: {strategy_name}")

        if ws_callback:
            await ws_callback({"progress": 0.1, "partial_result": {"message": "Starting backtest"}})

        if is_custom:
            result = await backtest_custom_strategy(df, strategy_data, params, ws_callback)
            result["StrategyName"] = strategy_name
            logger.debug(f"Custom strategy backtest completed: {strategy_name}")
            if ws_callback:
                await ws_callback({"progress": 1.0, "partial_result": result})
            return result
        else:
            strategy_func = {
                "MACD Crossover": macd_strategy,
                "Bollinger Bands": bollinger_band_strategy,
                "RSI Oversold/Overbought": rsi_strategy,
                "Short Sell Optimization": None  # Handled separately
            }.get(strategy_name)
            if not strategy_func and strategy_name != "Short Sell Optimization":
                logger.error(f"Unsupported strategy: {strategy_name}")
                return {"error": {"code": "INVALID_STRATEGY", "message": f"Unsupported strategy: {strategy_name}"}}
            if strategy_name == "Short Sell Optimization":
                initial_investment = params.get("initial_investment", 50000)
                stop_loss_range = params.get("stop_loss_range", [1.5, 2.5])
                target_range = params.get("target_range", [4.0, 6.0])
                stop_loss_values = [x for x in np.arange(stop_loss_range[0], stop_loss_range[1] + 0.5, 0.5)]
                target_values = [x for x in np.arange(target_range[0], target_range[1] + 0.5, 0.5)]
                best_result = None
                best_profit = float('-inf')
                total_iterations = len(stop_loss_values) * len(target_values)
                for i, (stop_loss_mult, target_mult) in enumerate([(sl, tg) for sl in stop_loss_values for tg in target_values]):
                    result = await backtest_short_sell(df, initial_investment, stop_loss_mult, target_mult)
                    result["StrategyName"] = strategy_name
                    if result["TotalProfit"] > best_profit:
                        best_profit = result["TotalProfit"]
                        best_result = result
                    if ws_callback:
                        await ws_callback({"progress": (i + 1) / total_iterations, "partial_result": result})
                logger.debug(f"Short Sell Optimization backtest completed: best profit {best_profit}")
                return best_result
            else:
                result = await backtest_strategy_generic(df, strategy_func, **params)
                result["StrategyName"] = strategy_name
                if ws_callback:
                    await ws_callback({"progress": 1.0, "partial_result": result})
                logger.debug(f"Generic strategy backtest completed: {strategy_name}")
                return result
    except ValueError as ve:
        logger.error(f"Validation error in backtest: {str(ve)}")
        return {"error": {"code": "INVALID_INPUT", "message": str(ve)}}
    except Exception as e:
        logger.error(f"Unexpected error backtesting strategy {strategy}: {str(e)}", exc_info=True)
        return {"error": {"code": "BACKTEST_FAILED", "message": str(e)}}

async def backtest_custom_strategy(df, strategy_data, params, ws_callback):
    initial_investment = params.get("initial_investment", 100000)
    tradebook = []
    position = None
    entry_price = 0
    portfolio_value = initial_investment
    entry_signals = await evaluate_custom_strategy(df, strategy_data.get("entry_conditions", []), indicators)
    exit_signals = await evaluate_custom_strategy(df, strategy_data.get("exit_conditions", []), indicators)
    for i, row in df.iterrows():
        if position is None and entry_signals.iloc[i] == "BUY":
            position = "long"
            entry_price = row["close"]
            tradebook.append({
                "Date": row["timestamp"].isoformat() if isinstance(row["timestamp"], pd.Timestamp) else row["timestamp"],
                "EntryPrice": entry_price,
                "Action": "BUY",
                "PortfolioValue": portfolio_value,
                "Quantity": initial_investment / entry_price
            })
        elif position == "long" and exit_signals.iloc[i] == "BUY":  # Using BUY as exit signal placeholder
            exit_price = row["close"]
            profit = (exit_price - entry_price) * (initial_investment / entry_price)
            portfolio_value += profit
            tradebook.append({
                "Date": row["timestamp"].isoformat() if isinstance(row["timestamp"], pd.Timestamp) else row["timestamp"],
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "Profit": profit,
                "PortfolioValue": portfolio_value,
                "Action": "SELL",
                "Quantity": initial_investment / entry_price
            })
            position = None
        if ws_callback:
            await ws_callback({"progress": (i + 1) / len(df), "partial_result": {"Tradebook": tradebook}})
    total_trades = len(tradebook)
    winning_trades = len([t for t in tradebook if t.get("Profit", 0) > 0])
    total_profit = portfolio_value - initial_investment
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    return {
        "InitialInvestment": initial_investment,
        "FinalPortfolioValue": portfolio_value,
        "TotalProfit": total_profit,
        "WinRate": win_rate,
        "TotalTrades": total_trades,
        "WinningTrades": winning_trades,
        "LosingTrades": total_trades - winning_trades,
        "Tradebook": tradebook
    }

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

async def backtest_strategy_generic(df, strategy_func, **params):
    df_copy = df.copy()
    signals = strategy_func(df_copy, **params)
    df_copy['signal'] = signals
    df_copy['position'] = 0
    df_copy['pnl'] = 0
    position = 0
    buy_price = 0
    tradebook = []
    initial_investment = params.get("initial_investment", 100000)
    portfolio_value = initial_investment
    for i, row in df_copy.iterrows():
        if row['signal'] == "BUY" and position == 0:
            position = 1
            buy_price = row['close']
            df_copy.at[i, 'position'] = position
            tradebook.append({
                "Date": row["timestamp"].isoformat() if isinstance(row["timestamp"], (pd.Timestamp, datetime)) else str(row["timestamp"]),
                "EntryPrice": float(buy_price),
                "Action": "BUY",
                "PortfolioValue": float(portfolio_value),
                "Quantity": float(initial_investment / buy_price)
            })
        elif row['signal'] == "SELL" and position == 1:
            position = 0
            sell_price = row['close']
            df_copy.at[i, 'position'] = position
            profit = (sell_price - buy_price) * (initial_investment / buy_price)
            portfolio_value += profit
            df_copy.at[i, 'pnl'] = profit
            tradebook.append({
                "Date": row["timestamp"].isoformat() if isinstance(row["timestamp"], (pd.Timestamp, datetime)) else str(row["timestamp"]),
                "EntryPrice": float(buy_price),
                "ExitPrice": float(sell_price),
                "Profit": float(profit),
                "PortfolioValue": float(portfolio_value),
                "Action": "SELL",
                "Quantity": float(initial_investment / buy_price)
            })
    df_copy['cumulative_pnl'] = df_copy['pnl'].cumsum()
    total_trades = df_copy['signal'].value_counts().sum()
    profitable_trades = len(df_copy[df_copy['pnl'] > 0])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    return {
        "data": df_copy.to_dict(orient="records"),
        "total_trades": total_trades,
        "win_rate": float(win_rate),
        "total_pnl": float(df_copy['cumulative_pnl'].iloc[-1]),
        "Tradebook": tradebook
    }

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

async def evaluate_custom_strategy(df, strategy_conditions: List[Dict], indicators_module):
    signals = pd.Series(index=df.index, dtype='object')
    signals[:] = None
    for i in range(1, len(df)):
        condition_met = True
        for cond in strategy_conditions:
            left_value = get_indicator_value(df, cond["left_indicator"], cond["left_params"], i, indicators_module)
            right_value = cond["right_value"] if cond["right_indicator"] == "Fixed Value" else get_indicator_value(df, cond["right_indicator"], cond["right_params"], i, indicators_module)
            if left_value is None or right_value is None:
                condition_met = False
                break
            comparison = cond["comparison"]
            if not evaluate_condition(left_value, right_value, comparison):
                condition_met = False
                break
        if condition_met:
            signals.iloc[i] = "BUY"  # Default to BUY for entry; adjust for exit
    if signals.dropna().empty:
        logger.warning(f"No signals generated for strategy conditions: {strategy_conditions}")
    return signals

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
        return abs(left_value - right_value) < 1e-6
    elif comparison == "Crosses Above":
        return left_value > right_value  # Simplified; needs previous value check
    elif comparison == "Crosses Below":
        return left_value < right_value
    return False

if __name__ == "__main__":
    import asyncio
    from backend.app.database import get_db
    from datetime import datetime, timedelta


    async def main():
            try:
                # Example parameters - replace with actual test values
                instrument_key = "NSE_EQ|INE002A01018"  # Example: Reliance Industries
                to_date_str = datetime.now().strftime("%Y-%m-%d")
                from_date_str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

                print(f"Fetching historical data for {instrument_key} from {from_date_str} to {to_date_str}")

                historical_data_response = await get_historical_data(
                    upstox_api=None,  # Provide a mock or real API object if needed
                    upstox_access_token=None,  # Provide a token if testing Upstox API path
                    instrument="RELIANCE",
                    from_date=from_date_str,
                    to_date=to_date_str,
                    unit="days",  # or "minutes"
                    interval="1",  # "1" for 1Day or "1" for 1minute etc.
                    db=get_db()
                )
                if historical_data_response and historical_data_response.data:
                    print(f"Fetched {len(historical_data_response.data)} data points.")
                    for point in historical_data_response.data[:5]:  # Print first 5 points
                        print(f"Timestamp: {point.timestamp}, Close: {point.close}")
                else:
                    print("No historical data found or an error occurred.")

            except Exception as e:
                print(f"An error occurred in main: {e}")


    asyncio.run(main())
