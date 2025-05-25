import asyncio
import json
import logging
import os
from datetime import datetime, time as date_time, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import pyotp
import requests
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.sql import text, func
from fastapi import Depends
from kiteconnect import KiteConnect
import upstox_client
from common_utils.db_utils import async_fetch_query, async_execute_query
from common_utils.read_write_sql_data import load_sql_data
from common_utils.utils import notify
from common_utils.upstox_utils import get_symbol_for_instrument
from models import Order, ScheduledOrder, QueuedOrder, User, GTTOrder
from schemas import Order as OrderSchema, ScheduledOrder as ScheduledOrderSchema, OrderHistory, Trade, QuoteResponse, \
    OHLCResponse, LTPResponse, HistoricalDataResponse, Instrument, HistoricalDataPoint
from database import get_db
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

logger = logging.getLogger(__name__)

# Define token expiry times
UPSTOX_TOKEN_EXPIRY_TIME = date_time(3, 30)  # 3:30 AM IST
ZERODHA_TOKEN_EXPIRY_TIME = date_time(6, 0)  # 6:00 AM IST

def get_next_expiry_time(target_time: date_time) -> datetime:
    """Calculate the next occurrence of the target time (e.g., 3:30 AM IST or 6:00 AM IST)."""
    now = datetime.now()
    target_datetime = datetime.combine(now.date(), target_time)
    if now > target_datetime:
        target_datetime += timedelta(days=1)
    return target_datetime

class TokenExpiredError(Exception):
    """Custom exception for expired tokens."""
    def __init__(self, broker: str):
        self.broker = broker
        self.message = f"{broker} access token has expired. Please re-authenticate."
        super().__init__(self.message)

class OrderMonitor:
    def __init__(self):
        self.running = True
        self.polling_interval = 60
        self.order_queue = None
        self.monitor_tasks = []
        logger.info("OrderMonitor initialized")

    async def initialize_queue(self):
        if self.order_queue is None:
            self.order_queue = asyncio.Queue()
            logger.info("Order queue initialized")

    async def sync_order_statuses(self, upstox_api, zerodha_api, db: AsyncSession):
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
                except Exception as e:
                    logger.error(f"Error syncing status for order {order.order_id}: {str(e)}")
            await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error in sync_order_statuses: {str(e)}")
            return False

    async def monitor_order(self, order_id, instrument_token, trading_symbol, transaction_type, quantity, product_type,
                            stop_loss_price, target_price, api, broker, db: AsyncSession, upstox_apis, kite_apis):
        task = asyncio.current_task()
        self.monitor_tasks.append(task)
        try:
            await self.initialize_queue()
            if stop_loss_price or target_price:
                sl_transaction = "BUY" if transaction_type == "SELL" else "SELL"
                if stop_loss_price:
                    await self.order_queue.put({
                        "queued_order_id": f"queued_{order_id}_sl",
                        "parent_order_id": order_id,
                        "instrument_token": instrument_token,
                        "trading_symbol": trading_symbol,
                        "transaction_type": sl_transaction,
                        "quantity": quantity,
                        "order_type": "SL-M",
                        "price": 0,
                        "trigger_price": stop_loss_price,
                        "product_type": product_type,
                        "validity": "DAY",
                        "is_gtt": "False",
                        "status": "QUEUED"
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
                        "status": "QUEUED"
                    })
                await self._store_queued_orders(db)
                notify("SL/Target Orders Queued", f"Queued SL and target for {trading_symbol}")

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

    async def _get_pending_orders(self, db: AsyncSession):
        query = """
            SELECT order_id, status, broker, instrument_token, trading_symbol 
            FROM orders 
            WHERE status IN ('open', 'pending', 'trigger pending')
        """
        return pd.DataFrame(await async_fetch_query(db, text(query), {}))

    async def _update_order_status(self, order_id, status, broker, db: AsyncSession):
        query = """
            UPDATE orders 
            SET status = :status 
            WHERE order_id = :order_id AND broker = :broker
        """
        await async_execute_query(db, text(query), {"status": status, "order_id": order_id, "broker": broker})

    async def _store_queued_orders(self, db: AsyncSession):
        while not self.order_queue.empty():
            order = await self.order_queue.get()
            await async_execute_query(db, text("""
                INSERT INTO queued_orders (
                    queued_order_id, parent_order_id, instrument_token, trading_symbol, transaction_type, 
                    quantity, order_type, price, trigger_price, product_type, validity, is_gtt, status
                ) VALUES (
                    :queued_order_id, :parent_order_id, :instrument_token, :trading_symbol, :transaction_type, 
                    :quantity, :order_type, :price, :trigger_price, :product_type, :validity, :is_gtt, :status
                )
            """), order)
            self.order_queue.task_done()

    async def _process_queued_orders(self, order_id, instrument_token, trading_symbol, api, broker, db: AsyncSession,
                                     upstox_apis, kite_apis):
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

    async def _clear_queued_orders(self, order_id, db: AsyncSession):
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

    async def run_scheduled_tasks(self, upstox_api=None, zerodha_api=None, user_apis=None):
        await self.initialize_queue()
        logger.info("Starting periodic sync tasks")
        try:
            async for db in get_db():
                # Fetch all users
                result = await db.execute(select(User))
                users = result.scalars().all()
                for user in users:
                    # Get user-specific APIs
                    if user_apis and user.user_id in user_apis:
                        upstox_api = user_apis[user.user_id]["upstox"]["order"]
                        zerodha_api = user_apis[user.user_id]["zerodha"]["kite"]
                    else:
                        upstox_api = None
                        zerodha_api = None
                    await self.sync_order_statuses(upstox_api, zerodha_api, db)
                while self.running:
                    await asyncio.sleep(self.polling_interval)
                    async for db_inner in get_db():
                        for user in users:
                            if user_apis and user.user_id in user_apis:
                                upstox_api = user_apis[user.user_id]["upstox"]["order"]
                                zerodha_api = user_apis[user.user_id]["zerodha"]["kite"]
                            else:
                                upstox_api = None
                                zerodha_api = None
                            await self.sync_order_statuses(upstox_api, zerodha_api, db_inner)
        except asyncio.CancelledError:
            logger.info("Periodic sync tasks cancelled")
            raise
        finally:
            logger.info("Stopped periodic sync tasks")

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
        await self._process_scheduled_orders(user_apis)

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

    async def _process_scheduled_orders(self, user_apis=None):
        logger.info("Starting scheduled order processing")
        try:
            while self.running:
                async with self.order_lock:
                    for order in self.scheduled_order_queue[:]:
                        async for db in get_db():
                            user_id = order.get("user_id", "default_user")
                            if user_apis and user_id in user_apis:
                                upstox_apis = user_apis[user_id]["upstox"]
                                kite_apis = user_apis[user_id]["zerodha"]
                            else:
                                logger.warning(f"Skipping scheduled order {order['scheduled_order_id']}: APIs not initialized for user {user_id}")
                                continue
                            await self._execute_scheduled_order(order, db, upstox_apis, kite_apis)
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Scheduled order processing cancelled")
            raise

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
                        self.scheduled_order_queue = [
                            o for o in self.scheduled_order_queue if
                            o["scheduled_order_id"] != order["scheduled_order_id"]
                        ]
                        query = "DELETE FROM scheduled_orders WHERE scheduled_order_id = :order_id"
                        await async_execute_query(db, text(query), {"order_id": order["scheduled_order_id"]})
            except Exception as e:
                logger.error(f"Error executing scheduled order {order['scheduled_order_id']}: {str(e)}")

    async def place_gtt_order(self, api, instrument_token, trading_symbol, transaction_type, quantity,
                              trigger_type, trigger_price, limit_price, last_price,
                              second_trigger_price=None, second_limit_price=None, broker="Zerodha",
                              db: AsyncSession = None, user_id: str = "default_user"):
        try:
            if broker != "Zerodha":
                await place_order(api, instrument_token, transaction_type, quantity, limit_price,
                                  "LIMIT", trigger_price, False, "CNC", "DAY", None, None, broker, db, user_id=user_id)
                return {"status": "success", "gtt_id": None}
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
            gtt_id = response.get("trigger_id")
            query = """
                INSERT INTO gtt_orders (
                    gtt_order_id, instrument_token, trading_symbol, transaction_type, quantity, 
                    trigger_type, trigger_price, limit_price, second_trigger_price, second_limit_price, 
                    status, broker, created_at, user_id
                ) VALUES (
                    :gtt_id, :instrument_token, :trading_symbol, :transaction_type, :quantity, 
                    :trigger_type, :trigger_price, :limit_price, :second_trigger_price, 
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
                "second_trigger_price": second_trigger_price,
                "second_limit_price": second_limit_price,
                "status": "active",
                "broker": broker,
                "created_at": datetime.now(),
                "user_id": user_id
            })
            return {"status": "success", "gtt_id": gtt_id}
        except Exception as e:
            logger.error(f"Error placing GTT order for {trading_symbol}: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def modify_gtt_order(self, api, gtt_id: str, trigger_type: str, trigger_price: float, limit_price: float,
                               second_trigger_price: Optional[float] = None, second_limit_price: Optional[float] = None,
                               db: AsyncSession = None):
        try:
            # Fetch the existing GTT order to get details
            stmt = select(GTTOrder).where(GTTOrder.gtt_order_id == gtt_id)
            result = await db.execute(stmt)
            gtt_order = result.scalars().first()
            if not gtt_order:
                raise ValueError(f"GTT order {gtt_id} not found")

            condition = {
                "exchange": "NSE",
                "tradingsymbol": gtt_order.trading_symbol,
                "last_price": trigger_price  # Approximate with trigger price as we don't have last price here
            }
            if trigger_type == "single":
                condition["trigger_values"] = [trigger_price]
                orders = [{
                    "exchange": "NSE",
                    "tradingsymbol": gtt_order.trading_symbol,
                    "product": "CNC",
                    "order_type": "LIMIT",
                    "transaction_type": gtt_order.transaction_type,
                    "quantity": gtt_order.quantity,
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
                        "quantity": gtt_order.quantity,
                        "price": limit_price
                    },
                    {
                        "exchange": "NSE",
                        "tradingsymbol": gtt_order.trading_symbol,
                        "product": "CNC",
                        "order_type": "LIMIT",
                        "transaction_type": gtt_order.transaction_type,
                        "quantity": gtt_order.quantity,
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
            # Update the database
            query = """
                UPDATE gtt_orders 
                SET trigger_type = :trigger_type, trigger_price = :trigger_price, limit_price = :limit_price,
                    second_trigger_price = :second_trigger_price, second_limit_price = :second_limit_price
                WHERE gtt_order_id = :gtt_id
            """
            await async_execute_query(db, text(query), {
                "gtt_id": gtt_id,
                "trigger_type": trigger_type,
                "trigger_price": trigger_price,
                "limit_price": limit_price,
                "second_trigger_price": second_trigger_price,
                "second_limit_price": second_limit_price
            })
            return {"status": "success", "gtt_id": gtt_id}
        except Exception as e:
            logger.error(f"Error modifying GTT order {gtt_id}: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def delete_gtt_order(self, api, gtt_id: str, db: AsyncSession):
        try:
            api.delete_gtt(trigger_id=gtt_id)
            query = "DELETE FROM gtt_orders WHERE gtt_order_id = :gtt_id"
            await async_execute_query(db, text(query), {"gtt_id": gtt_id})
            return {"status": "success", "message": f"GTT order {gtt_id} deleted"}
        except Exception as e:
            logger.error(f"Error deleting GTT order {gtt_id}: {str(e)}")
            return {"status": "error", "message": str(e)}

@retry(stop=stop_after_attempt(3), retry=retry_if_not_exception_type(TokenExpiredError))
async def init_upstox_api(db: AsyncSession, user_id: str, auth_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize Upstox APIs for the given user.

    Args:
        db: Database session
        user_id: User ID
        auth_code: Authorization code for Upstox (optional)

    Returns:
        Dict of Upstox APIs or None if initialization fails
    """
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        upstox_apis = {"order": None, "portfolio": None, "market_data": None, "user": None}

        # Initialize Upstox APIs
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
            logger.info(f"Upstox access token expired or missing for user {user_id}, fetching new token")
            if not user.upstox_access_token or not user.upstox_access_token_expiry:
                logger.warning(f"No Upstox access token for user {user_id}")
                return {"user": None, "order": None, "portfolio": None, "market_data": None, "history": None}

            if user.upstox_access_token_expiry < datetime.now():
                logger.warning(f"Upstox access token expired for user {user_id}")
                raise TokenExpiredError(broker="Upstox")

        logger.info(f"Upstox APIs initialized for user {user_id}")
        return upstox_apis
    except Exception as e:
        logger.error(f"Error initializing Upstox APIs for user {user_id}: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), retry=retry_if_not_exception_type(TokenExpiredError))
async def init_zerodha_api(db: AsyncSession, user_id: str, request_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize Zerodha APIs for the given user.

    Args:
        db: Database session
        user_id: User ID
        request_token: Request token for Zerodha (optional)

    Returns:
        Dict of Zerodha APIs or None if initialization fails
    """
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        kite_apis = {"kite": None}

        # Initialize Zerodha APIs
        if user.zerodha_access_token and user.zerodha_access_token_expiry and datetime.now() < user.zerodha_access_token_expiry:
            logger.info(
                f"Zerodha access token still valid for user {user_id}, expires at {user.zerodha_access_token_expiry}")
            kite = KiteConnect(api_key=user.zerodha_api_key)
            kite.set_access_token(user.zerodha_access_token)
            kite_apis = {"kite": kite}
        else:
            logger.info(f"Zerodha access token expired or missing for user {user_id}, fetching new token")
            if not user.zerodha_access_token or not user.zerodha_access_token_expiry:
                logger.warning(f"No Zerodha access token for user {user_id}")
                return {"kite": None}

            if user.zerodha_access_token_expiry < datetime.now():
                logger.warning(f"Zerodha access token expired for user {user_id}")
                raise TokenExpiredError(broker="Zerodha")

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

        if not auth_code:
            logger.error(f"Missing auth_code for Upstox token generation for user {user_id}")
            raise ValueError(
                f"Missing auth_code for Upstox token generation for user {user_id}. "
                "Please obtain a new auth_code by following these steps:\n"
                f"1. Visit https://api.upstox.com/v2/login/authorization/dialog?client_id={user.upstox_api_key}&redirect_uri=https://your-redirect-uri&response_type=code\n"
                "2. Log in and authorize the application to get the 'code' from the redirect URL.\n"
                "3. Call the /auth/upstox/ endpoint with the 'auth_code' query parameter (e.g., /auth/upstox/?auth_code=<code>)"
            )

        # Construct the Upstox login URL
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
        # Make the request to fetch the access token
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            access_token = response.json()["access_token"]
            expiry_time = get_next_expiry_time(UPSTOX_TOKEN_EXPIRY_TIME)

            user.upstox_access_token = access_token
            user.upstox_access_token_expiry = expiry_time
            await db.commit()
            logger.info(f"Upstox access token fetched for user {user_id}")
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
            logger.error(f"Missing Zerodha API Key & Secret for user {user_id}")
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
                      user_id: str = "default_user"):
    try:
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
        await load_sql_data(order_data, "orders", load_type="append", index_required=False, db=db,
                            database="trading_db")
        if stop_loss or target:
            asyncio.create_task(OrderMonitor().monitor_order(
                primary_order_id, instrument_token, trading_symbol, transaction_type, quantity,
                product_type, stop_loss, target, api, broker, db, upstox_apis, kite_apis
            ))
        return response
    except Exception as e:
        logger.error(f"Error placing {broker} order: {str(e)}")
        raise

async def modify_order(api, order_id: str, quantity: Optional[int] = None, order_type: Optional[str] = None,
                       price: Optional[float] = None, trigger_price: Optional[float] = None,
                       validity: Optional[str] = "DAY", broker: str = "Upstox", db: AsyncSession = None):
    try:
        # Fetch the existing order to get current details
        stmt = select(Order).where(Order.order_id == order_id, Order.broker == broker)
        result = await db.execute(stmt)
        order = result.scalars().first()
        if not order:
            raise ValueError(f"Order {order_id} not found")

        # Prepare modified parameters
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

        # Update the database
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
        return {"status": "success", "order_id": order_id}
    except Exception as e:
        logger.error(f"Error modifying order {order_id}: {str(e)}")
        raise

def get_order_book(upstox_api, kite_api):
    orders = []
    try:
        # Fetch Upstox orders
        if upstox_api:
            upstox_orders = upstox_api.get_order_book(api_version="v2").data
            for order in upstox_orders:
                order_dict = order.to_dict()
                orders.append({
                    "Broker": "Upstox",
                    "Order ID": order_dict.get("order_id", ""),
                    "Symbol": order_dict.get("trading_symbol", ""),
                    "Exchange": order_dict.get("exchange", ""),
                    "Trans. Type": order_dict.get("transaction_type", ""),
                    "Order Type": order_dict.get("order_type", ""),
                    "Product": order_dict.get("product", ""),
                    "Quantity": order_dict.get("quantity", 0),
                    "Status": order_dict.get("status", ""),
                    "Price": order_dict.get("price", 0),
                    "Trigger Price": order_dict.get("trigger_price", 0),
                    "Avg. Price": order_dict.get("average_price", 0),
                    "Filled Qty": order_dict.get("filled_quantity", 0),
                    "Order Time": order_dict.get("order_timestamp", ""),
                    "Remarks": order_dict.get("status_message", "")
                })
    except Exception as e:
        logger.error(f"Error fetching Upstox orders: {str(e)}")
        raise

    try:
        # Fetch Zerodha orders
        if kite_api:
            zerodha_orders = kite_api.orders()
            for order in zerodha_orders:
                orders.append({
                    "Broker": "Zerodha",
                    "Order ID": order.get("order_id", ""),
                    "Symbol": order.get("tradingsymbol", ""),
                    "Exchange": order.get("exchange", ""),
                    "Trans. Type": order.get("transaction_type", ""),
                    "Order Type": order.get("order_type", ""),
                    "Product": order.get("product", ""),
                    "Quantity": order.get("quantity", 0),
                    "Status": order.get("status", ""),
                    "Price": order.get("price", 0),
                    "Trigger Price": order.get("trigger_price", 0),
                    "Avg. Price": order.get("average_price", 0),
                    "Filled Qty": order.get("filled_quantity", 0),
                    "Order Time": order.get("order_timestamp", ""),
                    "Remarks": order.get("status_message", "")
                })
    except Exception as e:
        logger.error(f"Error fetching Zerodha orders: {str(e)}")
        raise

    return pd.DataFrame(orders)

def get_positions(upstox_api, zerodha_api):
    try:
        positions = []
        try:
            # Upstox positions
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
                        "Avg. Price": pos_dict.get("average_price", 0),
                        "Last Price": pos_dict.get("last_price", 0),
                        "P&L": pos_dict.get("pnl", 0),
                        "Instrument Token": pos_dict.get("instrument_token", "")
                    })
        except Exception as e:
            logger.error(f"Failed to fetch Upstox positions: {e}")

        try:
            # Zerodha positions
            if zerodha_api:
                zerodha_positions = zerodha_api.positions().get("net", [])
                for pos in zerodha_positions:
                    positions.append({
                        "Broker": "Zerodha",
                        "Symbol": pos.get("tradingsymbol", ""),
                        "Exchange": pos.get("exchange", ""),
                        "Product": pos.get("product", ""),
                        "Quantity": pos.get("net_quantity", 0),
                        "Avg. Price": pos.get("average_price", 0),
                        "Last Price": pos.get("last_price", 0),
                        "P&L": pos.get("pnl", 0),
                        "Instrument Token": f"{pos.get('exchange')}|{pos.get('tradingsymbol')}"
                    })
        except Exception as e:
            logger.error(f"Failed to fetch Zerodha positions: {e}")

        return pd.DataFrame(positions)
    except Exception as e:
        logger.error(f"Error fetching positions: {str(e)}")
        raise

def get_portfolio(upstox_api, zerodha_api):
    try:
        holdings = []
        try:
            # Upstox portfolio
            if upstox_api:
                upstox_holdings = upstox_api.get_holdings(api_version="v2").data
                for holding in upstox_holdings:
                    holding_dict = holding.to_dict()
                    holdings.append({
                        "Broker": "Upstox",
                        "Symbol": holding_dict.get("trading_symbol", ""),
                        "Exchange": holding_dict.get("exchange", ""),
                        "Quantity": holding_dict.get("quantity", 0),
                        "Last Price": holding_dict.get("last_price", 0),
                        "Avg. Price": holding_dict.get("average_price", 0),
                        "P&L": holding_dict.get("pnl", 0),
                        "Day Change": holding_dict.get("day_change", 0) * holding_dict.get("quantity", 0),
                        "Day Change %": holding_dict.get("day_change_percentage", 0),
                    })
        except Exception as e:
            logger.error(f"Failed to fetch Upstox portfolio: {e}")

        try:
            # Zerodha portfolio
            if zerodha_api:
                zerodha_holdings = zerodha_api.holdings()
                for holding in zerodha_holdings:
                    holdings.append({
                        "Broker": "Zerodha",
                        "Symbol": holding.get("tradingsymbol", ""),
                        "Exchange": holding.get("exchange", ""),
                        "Quantity": holding.get("quantity", 0),
                        "Last Price": holding.get("last_price", 0),
                        "Avg. Price": holding.get("average_price", 0),
                        "P&L": holding.get("pnl", 0),
                        "Day Change": holding.get("day_change", 0) * holding.get("quantity", 0),
                        "Day Change %": holding.get("day_change_percentage", 0),
                    })
        except Exception as e:
            logger.error(f"Failed to fetch Zerodha portfolio: {e}")

        return pd.DataFrame(holdings)
    except Exception as e:
        logger.error(f"Error fetching portfolio: {str(e)}")
        raise

def get_gtt_orders(zerodha_api):
    """Fetch GTT orders (Zerodha only)."""
    try:
        if not zerodha_api:
            return pd.DataFrame()
        gtt_orders = zerodha_api.get_gtts()
        gtt_list = [{
            "Broker": "Zerodha",
            "GTT ID": gtt["id"],
            "Symbol": gtt["condition"]["tradingsymbol"],
            "Exchange": gtt["condition"]["exchange"],
            "Transaction Type": gtt["orders"][0]["transaction_type"],
            "Quantity": gtt["orders"][0]["quantity"],
            "Trigger Price": gtt["condition"]["trigger_values"][0],
            "Limit Price": gtt["orders"][0]["price"],
            "Status": gtt["status"],
            "Created At": gtt["created_at"],
            "Expires At": gtt["expires_at"],
            "Result": gtt["orders"][0]["result"]["order_result"]["status"] if gtt["orders"][0]["result"] else None,
        } for gtt in gtt_orders]
        return pd.DataFrame(gtt_list)
    except Exception as e:
        logger.error(f"Failed to fetch GTT orders: {e}")
        return pd.DataFrame()

def get_funds_data(api, broker):
    """Fetch funds data for Upstox or Zerodha."""
    try:
        if not api:
            return {}
        if broker == "Upstox":
            funds_data = api.get_user_fund_margin(api_version="v2").data
        else:
            funds_data = api.margins()

        return funds_data
    except Exception as e:
        logger.error(f"Error fetching funds data: {str(e)}")
        return None

def get_quotes(upstox_api, kite_api, instruments: List[str]) -> List[QuoteResponse]:
    try:
        quotes = []
        if upstox_api:
            # Upstox expects a list of instrument tokens in the format "exchange:token"
            response = upstox_api.get_full_market_quote(",".join(instruments), api_version="v2").data
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
        if kite_api:
            response = kite_api.quote(instruments)
            for instrument, quote in response.items():
                quotes.append(QuoteResponse(
                    instrument_token=instrument,
                    last_price=quote.get("last_price", 0.0),
                    volume=quote.get("volume", 0),
                    average_price=quote.get("average_price"),
                    ohlc={
                        "open": quote.get("ohlc", {}).get("open", 0.0),
                        "high": quote.get("ohlc", {}).get("high", 0.0),
                        "low": quote.get("ohlc", {}).get("low", 0.0),
                        "close": quote.get("ohlc", {}).get("close", 0.0)
                    }
                ))
        return quotes
    except Exception as e:
        logger.error(f"Error fetching quotes: {str(e)}")
        raise

def get_ohlc(upstox_api, kite_api, instruments: List[str]) -> List[OHLCResponse]:
    try:
        ohlc_data = []
        if upstox_api:
            response = upstox_api.get_full_market_quote(",".join(instruments), api_version="v2").data
            for instrument, ohlc in response.items():
                ohlc_dict = ohlc.to_dict()
                ohlc_data.append(OHLCResponse(
                    instrument_token=instrument,
                    open=ohlc_dict.get("open", 0.0),
                    high=ohlc_dict.get("high", 0.0),
                    low=ohlc_dict.get("low", 0.0),
                    close=ohlc_dict.get("close", 0.0),
                    volume=ohlc_dict.get("volume")
                ))
        if kite_api:
            response = kite_api.ohlc(instruments)
            for instrument, ohlc in response.items():
                ohlc_data.append(OHLCResponse(
                    instrument_token=instrument,
                    open=ohlc.get("ohlc", {}).get("open", 0.0),
                    high=ohlc.get("ohlc", {}).get("high", 0.0),
                    low=ohlc.get("ohlc", {}).get("low", 0.0),
                    close=ohlc.get("ohlc", {}).get("close", 0.0),
                    volume=ohlc.get("volume")
                ))
        return ohlc_data
    except Exception as e:
        logger.error(f"Error fetching OHLC data: {str(e)}")
        raise

def get_ltp(upstox_api, kite_api, instruments: List[str]) -> List[LTPResponse]:
    try:
        ltp_data = []
        if upstox_api:
            response = upstox_api.get_full_market_quote(",".join(instruments), api_version="v2").data
            for instrument, quote in response.items():
                quote_dict = quote.to_dict()
                ltp_data.append(LTPResponse(
                    instrument_token=instrument,
                    last_price=quote_dict.get("last_price", 0.0)
                ))
        if kite_api:
            response = kite_api.ltp(instruments)
            for instrument, ltp in response.items():
                ltp_data.append(LTPResponse(
                    instrument_token=instrument,
                    last_price=ltp.get("last_price", 0.0)
                ))
        return ltp_data
    except Exception as e:
        logger.error(f"Error fetching LTP: {str(e)}")
        raise

def get_historical_data(upstox_api, kite_api, instrument: str, from_date: str, to_date: str, interval: str) -> HistoricalDataResponse:
    try:
        data_points = []
        if upstox_api:
            response = upstox_api.get_historical_candle_data1(
                instrument_key=instrument,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                api_version="v2"
            ).data.candles
            for candle in response:
                data_points.append(HistoricalDataPoint(
                    timestamp=datetime.strptime(candle[0], "%Y-%m-%dT%H:%M:%S%z"),
                    open=candle[1],
                    high=candle[2],
                    low=candle[3],
                    close=candle[4],
                    volume=candle[5]
                ))
        if kite_api:
            response = kite_api.historical_data(
                instrument_token=instrument.split(":")[1] if ":" in instrument else instrument,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=False
            )
            for candle in response:
                data_points.append(HistoricalDataPoint(
                    timestamp=datetime.strptime(candle["date"], "%Y-%m-%dT%H:%M:%S%z"),
                    open=candle["open"],
                    high=candle["high"],
                    low=candle["low"],
                    close=candle["close"],
                    volume=candle.get("volume")
                ))
        return HistoricalDataResponse(instrument_token=instrument, data=data_points)
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise

def get_instruments(upstox_api, kite_api, exchange: Optional[str] = None) -> List[Instrument]:
    try:
        instruments = []
        if upstox_api:
            response = upstox_api.instruments(api_version="v2").data
            for inst in response:
                inst_dict = inst.to_dict()
                if exchange and inst_dict.get("exchange") != exchange:
                    continue
                instruments.append(Instrument(
                    instrument_token=inst_dict.get("instrument_key"),
                    exchange=inst_dict.get("exchange"),
                    trading_symbol=inst_dict.get("trading_symbol"),
                    name=inst_dict.get("name"),
                    instrument_type=inst_dict.get("instrument_type"),
                    segment=inst_dict.get("segment")
                ))
        if kite_api:
            response = kite_api.instruments(exchange=exchange) if exchange else kite_api.instruments()
            for inst in response:
                instruments.append(Instrument(
                    instrument_token=str(inst["instrument_token"]),
                    exchange=inst["exchange"],
                    trading_symbol=inst["tradingsymbol"],
                    name=inst.get("name"),
                    instrument_type=inst["instrument_type"],
                    segment=inst["segment"]
                ))
        return instruments
    except Exception as e:
        logger.error(f"Error fetching instruments: {str(e)}")
        raise

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