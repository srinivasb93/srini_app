from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket
from fastapi.security import OAuth2PasswordRequestForm
from typing import List, Optional
import logging
from datetime import datetime, timedelta
import uuid
from database import get_db, init_engine
from sqlalchemy.future import select
import sys
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
import pandas as pd
import asyncio
import bcrypt
from collections import defaultdict
from uuid import uuid4

# Add the project_root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from models import Order, ScheduledOrder as ScheduledOrderModel, AutoOrder as AutoOrderModel, \
    GTTOrder as GTTOrderModel, User, Strategy
from backend.app.routes import data
from services import (
    OrderManager, OrderMonitor, init_upstox_api, init_zerodha_api,
    fetch_upstox_access_token, fetch_zerodha_access_token, place_order,
    get_order_book, get_positions, get_portfolio, get_funds_data,
    get_quotes, get_ohlc, get_ltp, get_historical_data, fetch_instruments,
    get_order_history, get_order_trades, modify_order, TokenExpiredError,
    execute_strategy, backtest_strategy, get_analytics_data, place_mf_sip,
    get_mf_sips, cancel_mf_sip, schedule_strategy_execution, stop_strategy_execution
)
from common_utils.db_utils import async_fetch_query
from common_utils.read_write_sql_data import load_sql_data
from schemas import (
    PlaceOrderRequest, UserCreate, UserResponse, ScheduledOrder, ScheduledOrderRequest, AutoOrder, AutoOrderRequest,
    GTTOrder, GTTOrderRequest, ProfileResponse, MarginResponse, QuoteResponse, OHLCResponse, LTPResponse,
    HistoricalDataResponse, Instrument, OrderHistory, Trade, ModifyOrderRequest, StrategyRequest, StrategyResponse,
    BacktestRequest, MFSIPRequest, MFSIPResponse
)
from auth import UserManager, oauth2_scheme

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Algo Trading App",
    description="API for managing algorithmic trading with Upstox and Zerodha",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "orders", "description": "Order management endpoints"},
        {"name": "portfolio", "description": "Portfolio and positions endpoints"},
        {"name": "market-data", "description": "Market data endpoints"},
        {"name": "gtt-orders", "description": "Good Till Triggered (GTT) order endpoints"},
        {"name": "algo-trading", "description": "Algorithmic trading endpoints"},
        {"name": "analytics", "description": "Analytics and indicators"},
        {"name": "mutual-funds", "description": "Mutual fund operations"}
    ]
)

# Include routers
app.include_router(data.router)

# Initialize OrderMonitor and OrderManager
order_monitor = OrderMonitor()
order_manager = OrderManager(monitor=order_monitor)

# User-specific APIs storage
user_apis = {}

# Custom in-memory rate limiter
class InMemoryRateLimiter:
    def __init__(self, times: int, seconds: int):
        self.times = times
        self.seconds = seconds
        self.requests = defaultdict(list)

    async def __call__(self, request: Request):
        client_ip = request.client.host
        now = datetime.now()
        # Remove expired timestamps
        self.requests[client_ip] = [ts for ts in self.requests[client_ip] if now - ts < timedelta(seconds=self.seconds)]
        # Check if limit exceeded
        if len(self.requests[client_ip]) >= self.times:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        # Add new request timestamp
        self.requests[client_ip].append(now)

# Rate limiter instances
auth_limiter = InMemoryRateLimiter(times=10, seconds=60)
order_limiter = InMemoryRateLimiter(times=20, seconds=60)
gtt_limiter = InMemoryRateLimiter(times=5, seconds=60)
portfolio_limiter = InMemoryRateLimiter(times=10, seconds=60)
backtest_limiter = InMemoryRateLimiter(times=10, seconds=60)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    user_id = UserManager.verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id

async def initialize_user_apis(user_id: str, db: AsyncSession, force_reinitialize: bool = False):
    if force_reinitialize or user_id not in user_apis:
        try:
            upstox_apis = await init_upstox_api(db, user_id)
            kite_apis = await init_zerodha_api(db, user_id)
            user_apis[user_id] = {"upstox": upstox_apis, "zerodha": kite_apis}
        except TokenExpiredError as e:
            logger.info(f"Token expired during initialization for user {user_id}: {e.message}")
            user_apis[user_id] = {
                "upstox": {"user": None, "order": None, "portfolio": None, "market_data": None, "history": None},
                "zerodha": {"kite": None}
            }
            return user_apis[user_id]
        except Exception as e:
            logger.error(f"Failed to initialize APIs for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize APIs: {str(e)}")
    return user_apis[user_id]

@app.on_event("startup")
async def startup_event():
    session_factory, _ = await init_engine()
    logger.info("Database engine and session factory initialized successfully")
    asyncio.create_task(order_monitor.run_scheduled_tasks(user_apis=user_apis))
    asyncio.create_task(order_manager.start(user_apis=user_apis))
    logger.info("Background tasks scheduled")

@app.on_event("shutdown")
async def shutdown_event():
    await order_monitor.cancel_all_tasks()
    logger.info("Application shutdown")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Trading Backend API"}

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse, tags=["auth"], dependencies=[Depends(auth_limiter)])
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.email == user.email))
    db_user = result.scalars().first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user_id = str(uuid.uuid4())
    new_user = User(
        user_id=user_id,
        email=user.email,
        hashed_password=hashed_password,
        upstox_api_key=user.upstox_api_key,
        upstox_api_secret=user.upstox_api_secret,
        zerodha_api_key=user.zerodha_api_key,
        zerodha_api_secret=user.zerodha_api_secret,
        created_at=datetime.now()
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return UserResponse.from_orm(new_user)

@app.post("/auth/login", response_model=dict, tags=["auth"], dependencies=[Depends(auth_limiter)])
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.email == form_data.username))
    user = result.scalars().first()
    if not user or not bcrypt.checkpw(form_data.password.encode('utf-8'), user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    try:
        await initialize_user_apis(user.user_id, db, force_reinitialize=True)
    except TokenExpiredError as e:
        logger.info(f"Token expired for user {user.user_id} during login")
    token = UserManager.create_access_token(data={"sub": user.user_id})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/upstox/", tags=["auth"], dependencies=[Depends(auth_limiter)])
async def auth_upstox(auth_code: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        access_token = await fetch_upstox_access_token(db, user_id, auth_code)
        if access_token:
            user_apis[user_id] = await initialize_user_apis(user_id, db, force_reinitialize=True)
            return {"status": "success", "message": "Upstox authentication successful"}
        else:
            raise HTTPException(status_code=400, detail="Failed to authenticate with Upstox")
    except Exception as e:
        logger.error(f"Error in Upstox auth: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/zerodha/", tags=["auth"], dependencies=[Depends(auth_limiter)])
async def auth_zerodha(request_token: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        access_token = await fetch_zerodha_access_token(db, user_id, request_token)
        if access_token:
            user_apis[user_id] = await initialize_user_apis(user_id, db, force_reinitialize=True)
            return {"status": "success", "message": "Zerodha authentication successful"}
        else:
            raise HTTPException(status_code=400, detail="Failed to authenticate with Zerodha")
    except Exception as e:
        logger.error(f"Error in Zerodha auth: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/profile/{broker}", response_model=ProfileResponse, tags=["portfolio"])
async def get_profile(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")

    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["upstox"]["user"] if broker == "Upstox" else user_apis_dict["zerodha"]["kite"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{broker} API not initialized")

    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if broker == "Zerodha":
            profile = api.profile()
            return ProfileResponse(
                user_id=user_id,
                email=user.email,
                name=profile.get("user_name"),
                broker=broker
            )
        else:  # Upstox
            profile = api.get_profile(api_version="v2").data  # Returns a ProfileData object
            return ProfileResponse(
                user_id=user_id,
                email=user.email,
                name=profile.user_name,
                broker=broker
            )
    except Exception as e:
        logger.error(f"Error fetching {broker} profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/margins/{broker}", response_model=MarginResponse, tags=["portfolio"], dependencies=[Depends(portfolio_limiter)])
async def get_margins(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["upstox"]["user"] if broker == "Upstox" else user_apis_dict["zerodha"]["kite"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{broker} API not initialized")
    try:
        margins = await get_funds_data(api, broker)
        if broker == "Upstox":
            response = MarginResponse(
                equity={
                    "available": margins.get("equity", {}).get("available_margin", 0),
                    "used": margins.get("equity", {}).get("used_margin", 0)
                },
                commodity={
                    "available": margins.get("commodity", {}).get("available_margin", 0),
                    "used": margins.get("commodity", {}).get("used_margin", 0)
                },
                broker=broker
            )
        else:
            response = MarginResponse(
                equity={
                    "available": margins["equity"]["available"]["live_balance"],
                    "used": margins["equity"]["utilised"]["debits"]
                },
                commodity={
                    "available": margins["commodity"]["available"]["live_balance"],
                    "used": margins["commodity"]["utilised"]["debits"]
                },
                broker=broker
            )
        return response
    except Exception as e:
        logger.error(f"Error fetching {broker} margins: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Order endpoints
@app.post("/orders/", tags=["orders"], dependencies=[Depends(order_limiter)])
async def place_new_order(order: PlaceOrderRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["upstox"]["order"] if order.broker == "Upstox" else user_apis_dict["zerodha"]["kite"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{order.broker} API not initialized")
    try:
        if order.schedule_datetime:
            scheduled_order_id = str(uuid.uuid4())

            order_data = {
                "scheduled_order_id": scheduled_order_id,
                "broker": order.broker,
                "instrument_token": order.instrument_token,
                "trading_symbol": order.trading_symbol,
                "transaction_type": order.transaction_type,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "price": order.price,
                "trigger_price": order.trigger_price,
                "product_type": order.product_type,
                "schedule_datetime": datetime.strptime(order.schedule_datetime,
                                                  "%Y-%m-%d %H:%M:%S") if order.schedule_datetime else None,
                "stop_loss": order.stop_loss,
                "target": order.target,
                "status": "PENDING",
                "is_amo": str(order.is_amo),
                "user_id": user_id
            }
            await load_sql_data(pd.DataFrame([order_data]), "scheduled_orders", load_type="append", index_required=False, db=db)
            await order_manager._load_scheduled_orders(db)
            return {"status": "success", "message": f"Scheduled order {scheduled_order_id} created"}
        else:
            response = await place_order(
                api=api,
                instrument_token=order.instrument_token,
                trading_symbol=order.trading_symbol,
                transaction_type=order.transaction_type,
                quantity=order.quantity,
                price=order.price,
                order_type=order.order_type,
                trigger_price=order.trigger_price,
                is_amo=order.is_amo,
                product_type=order.product_type,
                validity=order.validity,
                stop_loss=order.stop_loss,
                target=order.target,
                broker=order.broker,
                db=db,
                upstox_apis=user_apis_dict["upstox"],
                kite_apis=user_apis_dict["zerodha"],
                user_id=user_id,
                order_monitor=order_monitor,
            )
            return {"status": "success", "order_id": response.data.order_id if order.broker == "Upstox" else response}
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{broker}", tags=["orders"])
async def get_orders(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    try:
        query = """
            SELECT * FROM orders 
            WHERE user_id = :user_id AND broker = :broker
        """
        orders = pd.DataFrame(await async_fetch_query(db, text(query), {"user_id": user_id, "broker": broker}))
        return orders.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching {broker} orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/orders/{order_id}", tags=["orders"])
async def cancel_order(order_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Order).filter(Order.order_id == order_id, Order.user_id == user_id))
    order = result.scalars().first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["upstox"]["order"] if order.broker == "Upstox" else user_apis_dict["zerodha"]["kite"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{order.broker} API not initialized")
    try:
        if order.broker == "Upstox":
            api.cancel_order(order_id=order_id, api_version="v2")
        else:
            api.cancel_order(order_id=order_id, variety="regular" if "amo" not in order.status else "amo")
        order.status = "cancelled"
        await db.commit()
        return {"status": "success", "message": f"Order {order_id} cancelled"}
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/orders/{order_id}/modify", tags=["orders"])
async def modify_existing_order(order_id: str, modify_request: ModifyOrderRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Order).filter(Order.order_id == order_id, Order.user_id == user_id))
    order = result.scalars().first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["upstox"]["order"] if order.broker == "Upstox" else user_apis_dict["zerodha"]["kite"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{order.broker} API not initialized")
    try:
        response = await modify_order(
            api=api,
            order_id=order_id,
            quantity=modify_request.quantity,
            order_type=modify_request.order_type,
            price=modify_request.price,
            trigger_price=modify_request.trigger_price,
            validity=modify_request.validity,
            broker=order.broker,
            db=db
        )
        return response
    except Exception as e:
        logger.error(f"Error modifying order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{broker}/{order_id}", response_model=List[OrderHistory], tags=["orders"])
async def get_order_details(broker: str, order_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["order"]
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not (upstox_api if broker == "Upstox" else kite_api):
        raise HTTPException(status_code=400, detail=f"{broker} API not initialized")
    try:
        history = await get_order_history(upstox_api, kite_api, order_id, broker)
        return history
    except Exception as e:
        logger.error(f"Error fetching order history for {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{broker}/{order_id}/trades", response_model=List[Trade], tags=["orders"])
async def get_trades_for_order(broker: str, order_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["order"]
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not (upstox_api if broker == "Upstox" else kite_api):
        raise HTTPException(status_code=400, detail=f"{broker} API not initialized")
    try:
        trades = await get_order_trades(upstox_api, kite_api, order_id, broker)
        return trades
    except Exception as e:
        logger.error(f"Error fetching trades for order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order-book/{broker}", tags=["orders"])
async def get_order_book_data(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["order"] if broker == "Upstox" else None
    kite_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    if not (upstox_api or kite_api):
        raise HTTPException(status_code=400, detail=f"{broker} API not initialized")
    try:
        order_book = await get_order_book(upstox_api, kite_api)
        return order_book.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching {broker} order book: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scheduled-orders/", response_model=ScheduledOrder, tags=["orders"])
async def create_scheduled_order(order: ScheduledOrderRequest, user_id: str = Depends(get_current_user),
                                db: AsyncSession = Depends(get_db)):
    scheduled_order = ScheduledOrderModel(
        scheduled_order_id=str(uuid.uuid4()),
        broker=order.broker,
        instrument_token=order.instrument_token,
        trading_symbol=order.trading_symbol,
        transaction_type=order.transaction_type,
        quantity=order.quantity,
        order_type=order.order_type,
        price=order.price,
        trigger_price=order.trigger_price,
        product_type=order.product_type,
        schedule_datetime=order.schedule_datetime,
        stop_loss=order.stop_loss,
        target=order.target,
        status="PENDING",
        is_amo=order.is_amo,
        user_id=user_id
    )
    db.add(scheduled_order)
    await db.commit()
    await db.refresh(scheduled_order)
    await order_manager._load_scheduled_orders(db)
    return ScheduledOrder.from_orm(scheduled_order)

@app.post("/auto-orders/", response_model=AutoOrder, tags=["orders"])
async def create_auto_order(order: AutoOrderRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    auto_order = AutoOrderModel(
        auto_order_id=str(uuid.uuid4()),
        instrument_token=order.instrument_token,
        trading_symbol=order.trading_symbol,
        transaction_type=order.transaction_type,
        risk_per_trade=order.risk_per_trade,
        stop_loss_type=order.stop_loss_type,
        stop_loss_value=order.stop_loss_value,
        target_value=order.target_value,
        atr_period=order.atr_period,
        product_type=order.product_type,
        order_type=order.order_type,
        limit_price=order.limit_price,
        user_id=user_id,
        broker=order.broker
    )
    db.add(auto_order)
    await db.commit()
    await db.refresh(auto_order)
    return AutoOrder.from_orm(auto_order)

@app.post("/gtt-orders/", tags=["gtt-orders"])
async def create_gtt_order(order: GTTOrderRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["zerodha"]["kite"] if order.broker == "Zerodha" else user_apis_dict["upstox"]["order"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{order.broker} API not initialized")
    try:
        response = await order_manager.place_gtt_order(
            api=api,
            instrument_token=order.instrument_token,
            trading_symbol=order.trading_symbol,
            transaction_type=order.transaction_type,
            quantity=order.quantity,
            trigger_type=order.trigger_type,
            trigger_price=order.trigger_price,
            limit_price=order.limit_price,
            last_price=order.last_price,
            second_trigger_price=order.second_trigger_price,
            second_limit_price=order.second_limit_price,
            broker=order.broker,
            db=db,
            user_id=user_id
        )
        return response
    except Exception as e:
        logger.error(f"Error placing GTT order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gtt-orders/{broker}", response_model=List[GTTOrder], tags=["gtt-orders"])
async def get_gtt_orders(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    if broker == "Zerodha" and not api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        if broker == "Zerodha":
            gtt_orders = api.get_gtts()
            result = [
                GTTOrder(
                    gtt_order_id=str(gtt["id"]),
                    instrument_token="",  # Not directly available, needs mapping
                    trading_symbol=gtt["condition"]["tradingsymbol"],
                    transaction_type=gtt["orders"][0]["transaction_type"],
                    quantity=gtt["orders"][0]["quantity"],
                    trigger_type=gtt["type"],
                    trigger_price=gtt["condition"]["trigger_values"][0],
                    limit_price=gtt["orders"][0]["price"],
                    last_price=gtt["condition"]["last_price"],
                    second_trigger_price=gtt["condition"]["trigger_values"][1] if len(gtt["condition"]["trigger_values"]) > 1 else None,
                    second_limit_price=gtt["orders"][1]["price"] if len(gtt["orders"]) > 1 else None,
                    status=gtt["status"],
                    broker=broker,
                    created_at=datetime.strptime(gtt["created_at"], "%Y-%m-%d %H:%M:%S"),
                    user_id=user_id
                )
                for gtt in gtt_orders
            ]
        else:
            result = await db.execute(select(GTTOrderModel).filter(GTTOrderModel.user_id == user_id, GTTOrderModel.broker == broker))
            result = result.scalars().all()
        return result
    except Exception as e:
        logger.error(f"Error fetching {broker} GTT orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gtt-orders/{broker}/{gtt_id}", response_model=GTTOrder, tags=["gtt-orders"])
async def get_gtt_order(broker: str, gtt_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    if broker == "Zerodha" and not api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        if broker == "Zerodha":
            gtt = api.get_gtt(trigger_id=gtt_id)
            return GTTOrder(
                gtt_order_id=str(gtt["id"]),
                instrument_token="",  # Not directly available
                trading_symbol=gtt["condition"]["tradingsymbol"],
                transaction_type=gtt["orders"][0]["transaction_type"],
                quantity=gtt["orders"][0]["quantity"],
                trigger_type=gtt["type"],
                trigger_price=gtt["condition"]["trigger_values"][0],
                limit_price=gtt["orders"][0]["price"],
                last_price=gtt["condition"]["last_price"],
                second_trigger_price=gtt["condition"]["trigger_values"][1] if len(gtt["condition"]["trigger_values"]) > 1 else None,
                second_limit_price=gtt["orders"][1]["price"] if len(gtt["orders"]) > 1 else None,
                status=gtt["status"],
                broker=broker,
                created_at=datetime.strptime(gtt["created_at"], "%Y-%m-%d %H:%M:%S"),
                user_id=user_id
            )
        else:
            result = await db.execute(select(GTTOrder).filter(GTTOrder.gtt_order_id == gtt_id, GTTOrder.user_id == user_id))
            gtt = result.scalars().first()
            if not gtt:
                raise HTTPException(status_code=404, detail="GTT order not found")
            return gtt
    except Exception as e:
        logger.error(f"Error fetching GTT order {gtt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/gtt-orders/{broker}/{gtt_id}", tags=["gtt-orders"])
async def modify_gtt_order(broker: str, gtt_id: str, order: GTTOrderRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker != "Zerodha":
        raise HTTPException(status_code=400, detail="GTT modification only supported for Zerodha")
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["zerodha"]["kite"]
    if not api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        response = await order_manager.modify_gtt_order(
            api=api,
            gtt_id=gtt_id,
            trigger_type=order.trigger_type,
            trigger_price=order.trigger_price,
            limit_price=order.limit_price,
            last_price=order.last_price,
            quantity=order.quantity,
            second_trigger_price=order.second_trigger_price,
            second_limit_price=order.second_limit_price,
            db=db
        )
        return response
    except Exception as e:
        logger.error(f"Error modifying GTT order {gtt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/gtt-orders/{broker}/{gtt_id}", tags=["gtt-orders"])
async def delete_gtt_order(broker: str, gtt_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    if broker == "Zerodha" and not api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        response = await order_monitor.delete_gtt_order(api, gtt_id, db)
        return response
    except Exception as e:
        logger.error(f"Error deleting GTT order {gtt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions/{broker}", tags=["portfolio"])
async def get_positions_data(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["portfolio"] if broker == "Upstox" else None
    zerodha_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    if not (upstox_api or zerodha_api):
        raise HTTPException(status_code=400, detail=f"{broker} API not initialized")
    try:
        positions = await get_positions(upstox_api, zerodha_api)
        return positions.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching {broker} positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/{broker}", tags=["portfolio"])
async def get_portfolio_data(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["portfolio"] if broker == "Upstox" else None
    zerodha_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    if not (upstox_api or zerodha_api):
        raise HTTPException(status_code=400, detail=f"{broker} API not initialized")
    try:
        portfolio = await get_portfolio(upstox_api, zerodha_api)
        return portfolio.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching {broker} portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trade-history/{broker}", tags=["portfolio"])
async def get_trade_history(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    try:
        query = """
            SELECT * FROM trade_history 
            WHERE user_id = :user_id AND broker = :broker
        """
        trades = pd.DataFrame(await async_fetch_query(db, text(query), {"user_id": user_id, "broker": broker}))
        return trades.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching {broker} trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/funds/{broker}", tags=["portfolio"])
async def get_funds(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["upstox"]["user"] if broker == "Upstox" else user_apis_dict["zerodha"]["kite"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{broker} API not initialized")
    try:
        funds_data = await get_funds_data(api, broker)
        # Ensure consistent response format
        if broker == "Upstox":
            response = {
                "equity": {
                    "available": funds_data.get("equity", {}).get("available_margin", 0),
                    "used": funds_data.get("equity", {}).get("used_margin", 0)
                },
                "commodity": {
                    "available": funds_data.get("commodity", {}).get("available_margin", 0),
                    "used": funds_data.get("commodity", {}).get("used_margin", 0)
                },
                "broker": broker
            }
        else:  # Zerodha
            response = {
                "equity": {
                    "available": funds_data.get("equity", {}).get("available", {}).get("live_balance", 0),
                    "used": funds_data.get("equity", {}).get("utilised", {}).get("debits", 0)
                },
                "commodity": {
                    "available": funds_data.get("commodity", {}).get("available", {}).get("live_balance", 0),
                    "used": funds_data.get("commodity", {}).get("utilised", {}).get("debits", 0)
                },
                "broker": broker
            }
        return response
    except Exception as e:
        logger.error(f"Error fetching {broker} funds: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quotes/{broker}", response_model=List[QuoteResponse], tags=["market-data"])
async def get_market_quotes(broker: str, instruments: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    # upstox_api = user_apis_dict["upstox"]["history"] if broker == "Upstox" else None
    # kite_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    upstox_api = user_apis_dict["upstox"]["market_data"]
    if not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API not initialized")
    try:
        instrument_list = instruments.split(",")
        quotes = await get_quotes(upstox_api, None, instrument_list)
        return quotes
    except Exception as e:
        logger.error(f"Error fetching {broker} quotes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ohlc/{broker}", response_model=List[OHLCResponse], tags=["market-data"])
async def get_ohlc_data(broker: str, instruments: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    # upstox_api = user_apis_dict["upstox"]["history"] if broker == "Upstox" else None
    # kite_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    upstox_api = user_apis_dict["upstox"]["market_data"]
    if not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API not initialized")
    try:
        instrument_list = instruments.split(",")
        ohlc_data = await get_ohlc(upstox_api, None, instrument_list)
        return ohlc_data
    except Exception as e:
        logger.error(f"Error fetching {broker} OHLC data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ltp/{broker}", response_model=List[LTPResponse], tags=["market-data"])
async def get_ltp_data(broker: str, instruments: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    # upstox_api = user_apis_dict["upstox"]["history"] if broker == "Upstox" else None
    # kite_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    upstox_api = user_apis_dict["upstox"]["market_data"]
    if not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API not initialized")
    try:
        instrument_list = instruments.split(",")
        ltp_data = await get_ltp(upstox_api, None, instrument_list)
        return ltp_data
    except Exception as e:
        logger.error(f"Error fetching {broker} LTP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical-data/{broker}", response_model=HistoricalDataResponse, tags=["market-data"])
async def get_historical(broker: str, instrument: str, from_date: str, to_date: str, unit: str,  interval: str,
                        user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    # upstox_api = user_apis_dict["upstox"]["history"] if broker == "Upstox" else None
    # kite_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    upstox_api = user_apis_dict["upstox"]["history"]
    if not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API not initialized")
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        historical_data = await get_historical_data(upstox_api, user.upstox_access_token, instrument, from_date, to_date, unit, interval)
        return historical_data
    except Exception as e:
        logger.error(f"Error fetching {broker} historical data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/instruments/{broker}", response_model=List[Instrument], tags=["market-data"])
async def get_instruments_list(
    broker: str,
    exchange: Optional[str] = None,
    refresh: bool = False,
    db: AsyncSession = Depends(get_db)
):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    try:
        instruments = await fetch_instruments(db, refresh=refresh)
        if exchange:
            instruments = [inst for inst in instruments if inst.exchange == exchange]
        logger.info(f"Fetched {len(instruments)} instruments for broker {broker}, exchange {exchange}")
        return instruments
    except Exception as e:
        logger.error(f"Error fetching instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutual-funds/instruments", tags=["mutual-funds"])
async def get_mutual_fund_instruments(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        instruments = kite_api.mf_instruments()
        return instruments
    except Exception as e:
        logger.error(f"Error fetching mutual fund instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mutual-funds/orders", tags=["mutual-funds"])
async def place_mutual_fund_order(
    scheme_code: str, amount: float, transaction_type: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        order = kite_api.place_mf_order(
            tradingsymbol=scheme_code,
            transaction_type=transaction_type,
            amount=amount,
        )
        return {"status": "success", "order_id": order["order_id"]}
    except Exception as e:
        logger.error(f"Error placing mutual fund order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/mutual-funds/orders/{order_id}", tags=["mutual-funds"])
async def cancel_mutual_fund_order(order_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        kite_api.cancel_mf_order(order_id=order_id)
        return {"status": "success", "message": f"Mutual fund order {order_id} cancelled"}
    except Exception as e:
        logger.error(f"Error cancelling mutual fund order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutual-funds/orders", tags=["mutual-funds"])
async def get_mutual_fund_orders(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        orders = kite_api.mf_orders()
        return orders
    except Exception as e:
        logger.error(f"Error fetching mutual fund orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutual-funds/holdings", tags=["mutual-funds"])
async def get_mutual_fund_holdings(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        holdings = kite_api.mf_holdings()
        return holdings
    except Exception as e:
        logger.error(f"Error fetching mutual fund holdings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutual-funds/sips", response_model=List[MFSIPResponse], tags=["mutual-funds"])
async def get_mutual_fund_sips(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        sips = await get_mf_sips(kite_api, db, user_id)
        return sips
    except Exception as e:
        logger.error(f"Error fetching mutual fund SIPs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/algo-trading/execute", tags=["algo-trading"])
async def execute_strategy_endpoint(request: StrategyRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["upstox"]["order"] if request.broker == "Upstox" else user_apis_dict["zerodha"]["kite"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{request.broker} API not initialized")
    try:
        result = await execute_strategy(
            api=api,
            strategy=request.strategy,
            instrument_token=request.instrument_token,
            quantity=request.quantity,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            broker=request.broker,
            db=db,
            user_id=user_id
        )
        return {"status": "success", "message": result}
    except Exception as e:
        logger.error(f"Error executing strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/algo-trading/backtest", tags=["algo-trading"], dependencies=[Depends(backtest_limiter)])
async def backtest_strategy_endpoint(request: BacktestRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        async def ws_callback(data):
            # Placeholder for WebSocket progress (handled in WebSocket endpoint)
            pass
        result = await backtest_strategy(
            instrument_token=request.instrument_token,
            timeframe=request.timeframe,
            strategy=request.strategy,
            params=request.params,
            start_date=request.start_date,
            end_date=request.end_date,
            ws_callback=ws_callback
        )
        return result
    except Exception as e:
        logger.error(f"Error backtesting strategy: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "BACKTEST_FAILED", "message": str(e)}})

@app.post("/strategies/", response_model=StrategyResponse, tags=["algo-trading"])
async def create_strategy(strategy: StrategyRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        new_strategy = Strategy(
            strategy_id=str(uuid4()),
            user_id=user_id,
            broker=strategy.broker,
            name=strategy.name,
            description=strategy.description,
            entry_conditions=strategy.entry_conditions,
            exit_conditions=strategy.exit_conditions,
            parameters=strategy.parameters
        )
        db.add(new_strategy)
        await db.commit()
        await db.refresh(new_strategy)
        return StrategyResponse.from_orm(new_strategy)
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "CREATE_FAILED", "message": str(e)}})

@app.post("/algo-trading/schedule", tags=["algo-trading"], dependencies=[Depends(gtt_limiter)])
async def schedule_strategy(request: StrategyRequest, interval_minutes: int = 5, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["upstox"]["order"] if request.broker == "Upstox" else user_apis_dict["zerodha"]["kite"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{request.broker} API not initialized")
    try:
        result = await schedule_strategy_execution(
            api=api,
            strategy=request.strategy,
            instrument_token=request.instrument_token,
            quantity=request.quantity,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            interval_minutes=interval_minutes,
            run_hours=[(9, 15), (15, 30)],
            broker=request.broker,
            db=db,
            user_id=user_id
        )
        return result
    except Exception as e:
        logger.error(f"Error scheduling strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/algo-trading/stop", tags=["algo-trading"], dependencies=[Depends(gtt_limiter)])
async def stop_strategy(strategy: str, instrument_token: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        result = await stop_strategy_execution(strategy, instrument_token, user_id, db)
        return result
    except Exception as e:
        logger.error(f"Error stopping strategy {strategy} for {instrument_token}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies/{broker}", response_model=List[StrategyResponse], tags=["algo-trading"])
async def get_strategies(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_BROKER", "message": "Invalid broker"}})
    try:
        result = await db.execute(select(Strategy).filter(Strategy.user_id == user_id, Strategy.broker == broker))
        strategies = result.scalars().all()
        return [StrategyResponse.from_orm(s) for s in strategies]
    except Exception as e:
        logger.error(f"Error fetching strategies: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "FETCH_FAILED", "message": str(e)}})

@app.get("/strategies/{strategy_id}", response_model=StrategyResponse, tags=["algo-trading"])
async def get_strategy(strategy_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(Strategy).filter(Strategy.strategy_id == strategy_id, Strategy.user_id == user_id))
        strategy = result.scalars().first()
        if not strategy:
            raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Strategy not found"}})
        return StrategyResponse.from_orm(strategy)
    except Exception as e:
        logger.error(f"Error fetching strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "FETCH_FAILED", "message": str(e)}})

@app.post("/strategies/{strategy_id}/{action}", response_model=dict, tags=["algo-trading"])
async def toggle_strategy(strategy_id: str, action: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if action not in ["activate", "deactivate"]:
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_ACTION", "message": "Invalid action"}})
    try:
        result = await db.execute(select(Strategy).filter(Strategy.strategy_id == strategy_id, Strategy.user_id == user_id))
        strategy = result.scalars().first()
        if not strategy:
            raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Strategy not found"}})
        strategy.status = "active" if action == "activate" else "inactive"
        await db.commit()
        return {"status": strategy.status, "strategy_id": strategy_id}
    except Exception as e:
        logger.error(f"Error toggling strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "TOGGLE_FAILED", "message": str(e)}})

@app.delete("/strategies/{strategy_id}", response_model=dict, tags=["algo-trading"])
async def delete_strategy(strategy_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(Strategy).filter(Strategy.strategy_id == strategy_id, Strategy.user_id == user_id))
        strategy = result.scalars().first()
        if not strategy:
            raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Strategy not found"}})
        await db.execute(text("DELETE FROM strategies WHERE strategy_id = :strategy_id"), {"strategy_id": strategy_id})
        await db.commit()
        return {"success": True, "message": "Strategy deleted"}
    except Exception as e:
        logger.error(f"Error deleting strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "DELETE_FAILED", "message": str(e)}})

# Add WebSocket for backtest progress
@app.websocket("/ws/backtest/{user_id}")
async def websocket_backtest(websocket: WebSocket, user_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            await websocket.send_json({"progress": data.get("progress", 0), "partial_result": data.get("partial_result")})
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
    finally:
        await websocket.close()

@app.get("/analytics/{broker}", tags=["analytics"])
async def get_analytics(
    instrument_token: str,
    unit: str = "days",
    interval: str = "1",
    ema_period: int = 20,
    rsi_period: int = 14,
    lr_period: int = 20,
    stochastic_k: int = 14,
    stochastic_d: int = 3,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["history"]
    if not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API not initialized")
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        data = await get_analytics_data(
            upstox_api=upstox_api,
            upstox_access_token=user.upstox_access_token,
            instrument_token=instrument_token,
            unit=unit,
            interval=interval,
            ema_period=ema_period,
            rsi_period=rsi_period,
            lr_period=lr_period,
            stochastic_k=stochastic_k,
            stochastic_d=stochastic_d
        )
        return data
    except Exception as e:
        logger.error(f"Error fetching analytics data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mutual-funds/sips", response_model=MFSIPResponse, tags=["mutual-funds"], dependencies=[Depends(gtt_limiter)])
async def create_mf_sip(sip: MFSIPRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        response = await place_mf_sip(kite_api, sip, db, user_id)
        return response
    except Exception as e:
        logger.error(f"Error creating SIP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutual-funds/sips", response_model=List[MFSIPResponse], tags=["mutual-funds"], dependencies=[Depends(portfolio_limiter)])
async def list_mf_sips(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        sips = await get_mf_sips(kite_api, db, user_id)
        return sips
    except Exception as e:
        logger.error(f"Error listing SIPs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/mutual-funds/sips/{sip_id}", tags=["mutual-funds"], dependencies=[Depends(gtt_limiter)])
async def delete_mf_sip(sip_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"]
    if not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    try:
        response = await cancel_mf_sip(kite_api, sip_id, db, user_id)
        return response
    except Exception as e:
        logger.error(f"Error cancelling SIP {sip_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scheduled-orders/{broker}",  response_model=List[ScheduledOrder], tags=["orders"])
async def get_scheduled_orders(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    try:
        query = select(ScheduledOrderModel).filter(ScheduledOrderModel.user_id == user_id, ScheduledOrderModel.broker == broker)
        result = await db.execute(query)
        orders = result.scalars().all()
        return [ScheduledOrder.from_orm(order) for order in orders]
    except Exception as e:
        logger.error(f"Error fetching scheduled orders for {broker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auto-orders/{broker}", response_model=List[AutoOrder],tags=["orders"])
async def get_auto_orders(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    try:
        query = select(AutoOrderModel).filter(AutoOrderModel.user_id == user_id, AutoOrderModel.broker == broker)
        result = await db.execute(query)
        orders = result.scalars().all()
        return [AutoOrder.from_orm(order) for order in orders]
    except Exception as e:
        logger.error(f"Error fetching auto orders for {broker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

