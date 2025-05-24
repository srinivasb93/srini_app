from fastapi import FastAPI, Depends, HTTPException, Request, status
from typing import List, Optional
import logging
from datetime import datetime
import uuid
from kiteconnect import KiteConnect
from database import get_db, init_engine
from sqlalchemy.future import select
import sys
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
import pandas as pd
import asyncio
from dotenv import load_dotenv
import bcrypt

# Add the project_root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from models import Order, ScheduledOrder, AutoOrder, GTTOrder, User
from services import (
    OrderManager, OrderMonitor, init_upstox_api, init_zerodha_api,
    fetch_upstox_access_token, fetch_zerodha_access_token, place_order,
    get_order_book, get_positions, get_portfolio, get_funds_data
)
from common_utils.db_utils import async_fetch_query, async_execute_query
from common_utils.read_write_sql_data import load_sql_data
from schemas import PlaceOrderRequest, UserCreate, UserResponse, ScheduledOrder, AutoOrder, GTTOrderRequest
from auth import UserManager, Token
from fastapi.security import OAuth2PasswordRequestForm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Algo Trading App",
    description="API for managing algorithmic trading with Zerodha and Upstox",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "orders", "description": "Order management endpoints"}
    ]
)
load_dotenv()

DATABASE = "trading_db"

# Initialize global API storage (for backward compatibility)
app.upstox_apis = {"order": None, "portfolio": None, "market_data": None, "user": None}
app.kite_apis = {"kite": None}

# Initialize per-user API storage
app.user_apis = {}  # Format: {user_id: {"upstox": upstox_apis, "zerodha": kite_apis}}


@app.on_event("startup")
async def startup_event():
    logger.info("Starting application")
    await init_engine()  # Initializes global session_factory

    # Initialize OrderMonitor and OrderManager
    monitor = OrderMonitor()
    app.order_monitor = monitor
    app.order_manager = OrderManager(monitor)
    try:
        # Since APIs are now initialized per user, pass user_apis to OrderMonitor
        app.monitor_task = asyncio.create_task(
            monitor.run_scheduled_tasks(user_apis=app.user_apis))
        app.manager_task = asyncio.create_task(app.order_manager.start(app.user_apis))
    except Exception as e:
        logger.error(f"Failed to initialize database tasks: {str(e)}")
        app.monitor_task = None
        app.manager_task = None
    logger.info("Application initialized")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")
    if hasattr(app, "order_monitor"):
        await app.order_monitor.cancel_all_tasks()
    if hasattr(app, "monitor_task") and app.monitor_task:
        app.monitor_task.cancel()
    if hasattr(app, "manager_task") and app.manager_task:
        app.manager_task.cancel()
    # Reset global APIs
    app.upstox_apis = {"order": None, "portfolio": None, "market_data": None, "user": None}
    app.kite_apis = {"kite": None}
    # Clear per-user APIs
    app.user_apis = {}
    logger.info("Application shutdown complete")


# Authentication Endpoints
@app.post("/auth/register", tags=["auth"])
async def register(user_create: UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(User).filter(User.email == user_create.email))
        if result.scalars().first():
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = bcrypt.hashpw(user_create.password.encode("utf-8"), bcrypt.gensalt())

        db_user = User(
            user_id=str(uuid.uuid4()),
            email=user_create.email,
            hashed_password=hashed_password.decode("utf-8"),
            upstox_api_key=user_create.upstox_api_key,
            upstox_api_secret=user_create.upstox_api_secret,
            upstox_username=user_create.upstox_username,
            upstox_password=user_create.upstox_password,
            upstox_totp_token=user_create.upstox_totp_token,
            zerodha_api_key=user_create.zerodha_api_key,
            zerodha_api_secret=user_create.zerodha_api_secret,
            zerodha_username=user_create.zerodha_username,
            zerodha_password=user_create.zerodha_password,
            zerodha_totp_token=user_create.zerodha_totp_token,
            created_at=datetime.now()
        )

        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        return {"status": "success", "user_id": db_user.user_id}
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/login", tags=["auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    logger.info(f"Login attempt with username: {form_data.username}")

    if not form_data.username or "@" not in form_data.username:
        raise HTTPException(status_code=422, detail="Invalid email format")
    if not form_data.password:
        raise HTTPException(status_code=422, detail="Password cannot be empty")
    result = await db.execute(select(User).filter(User.email == form_data.username))
    user = result.scalars().first()
    if not user:
        logger.error(f"User not found: {form_data.username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    try:
        password_bytes = form_data.password.encode("utf-8")
        hashed_password_bytes = user.hashed_password.encode("utf-8")
        if not bcrypt.checkpw(password_bytes, hashed_password_bytes):
            logger.error("Password mismatch")
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        logger.error(f"Password check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # Initialize APIs for the user
    try:
        upstox_apis = await init_upstox_api(db, user.user_id)
        kite_apis = await init_zerodha_api(db, user.user_id)
        app.user_apis[user.user_id] = {
            "upstox": upstox_apis,
            "zerodha": kite_apis
        }
        # Update global APIs (for backward compatibility)
        app.upstox_apis = upstox_apis
        app.kite_apis = kite_apis
        logger.info(f"APIs initialized for user {user.user_id} during login")
    except Exception as e:
        logger.error(f"Error initializing APIs for user {user.user_id} during login: {str(e)}")
        # Ensure the user_apis entry is created even if initialization fails
        app.user_apis[user.user_id] = {
            "upstox": {"order": None, "portfolio": None, "market_data": None, "user": None},
            "zerodha": {"kite": None}
        }

    access_token = UserManager.create_access_token(data={"sub": user.user_id})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/health/", tags=["health"])
async def health_check(db: AsyncSession = Depends(get_db)):
    return {"status": "healthy", "database": "connected"}


@app.post("/initialize/", tags=["initialize"])
async def initialize(current_user: User = Depends(UserManager.get_current_user)):
    return {"status": "success", "message": "Application initialized"}


@app.post("/auth/upstox/", tags=["auth"])
async def auth_upstox(
        auth_code: str,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    try:
        token = await fetch_upstox_access_token(db, current_user.user_id, auth_code)
        if not token:
            raise HTTPException(
                status_code=400,
                detail="Failed to fetch Upstox access token. Ensure the auth_code is valid and try again."
            )
        # Initialize Upstox APIs and update user-specific storage
        upstox_apis = await init_upstox_api(db, current_user.user_id, auth_code)
        if current_user.user_id not in app.user_apis:
            app.user_apis[current_user.user_id] = {
                "upstox": upstox_apis,
                "zerodha": {"kite": None}
            }
        else:
            app.user_apis[current_user.user_id]["upstox"] = upstox_apis
        # Update global APIs (for backward compatibility)
        app.upstox_apis = upstox_apis
        logger.info("Reinitialized Upstox APIs after fetching new token")
        result = await db.execute(select(User).filter(User.user_id == current_user.user_id))
        user = result.scalars().first()
        return {
            "status": "success",
            "access_token": token,
            "expires_at": user.upstox_access_token_expiry.isoformat() if user.upstox_access_token_expiry else None
        }
    except ValueError as e:
        logger.error(f"ValueError in auth_upstox for user {current_user.user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in auth_upstox for user {current_user.user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upstox authentication failed: {str(e)}")


@app.post("/auth/zerodha/", tags=["auth"])
async def auth_zerodha(
        request_token: str,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    try:
        token = await fetch_zerodha_access_token(db, current_user.user_id, request_token)
        if not token:
            raise HTTPException(
                status_code=400,
                detail="Failed to fetch Zerodha access token. Please check your credentials and try again."
            )
        # Initialize Zerodha APIs and update user-specific storage
        kite_apis = await init_zerodha_api(db, current_user.user_id, request_token)
        if current_user.user_id not in app.user_apis:
            app.user_apis[current_user.user_id] = {
                "upstox": {"order": None, "portfolio": None, "market_data": None, "user": None},
                "zerodha": kite_apis
            }
        else:
            app.user_apis[current_user.user_id]["zerodha"] = kite_apis
        # Update global APIs (for backward compatibility)
        app.kite_apis = kite_apis
        logger.info("Reinitialized Zerodha APIs after fetching new token")
        result = await db.execute(select(User).filter(User.user_id == current_user.user_id))
        user = result.scalars().first()
        return {
            "status": "success",
            "access_token": token,
            "expires_at": user.zerodha_access_token_expiry.isoformat() if user.zerodha_access_token_expiry else None
        }
    except ValueError as e:
        logger.error(f"ValueError in auth_zerodha for user {current_user.user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in auth_zerodha for user {current_user.user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Zerodha authentication failed: {str(e)}")


@app.post("/orders/", tags=["orders"])
async def place_order_endpoint(
        order_data: PlaceOrderRequest,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received request body for /orders/: {order_data.dict()}")

    # Fetch user-specific APIs
    if current_user.user_id not in app.user_apis:
        raise HTTPException(status_code=400, detail="APIs not initialized for this user. Please log in again.")
    user_apis = app.user_apis[current_user.user_id]
    upstox_apis = user_apis["upstox"]
    kite_apis = user_apis["zerodha"]

    # Use the appropriate API based on the broker
    api = upstox_apis["order"] if order_data.broker == "Upstox" else kite_apis["kite"]

    # Check if the required API is initialized
    if order_data.broker == "Upstox" and not upstox_apis["order"]:
        raise HTTPException(status_code=400,
                            detail="Upstox API not initialized. Please authenticate via /auth/upstox/.")
    if order_data.broker == "Zerodha" and not kite_apis["kite"]:
        raise HTTPException(status_code=400,
                            detail="Zerodha API not initialized. Please authenticate via /auth/zerodha/.")

    try:
        result = await place_order(
            api=api,
            instrument_token=order_data.instrument_token,
            transaction_type=order_data.transaction_type,
            quantity=order_data.quantity,
            price=order_data.price,
            order_type=order_data.order_type,
            trigger_price=order_data.trigger_price,
            is_amo=order_data.is_amo,
            product_type=order_data.product_type,
            validity=order_data.validity,
            stop_loss=order_data.stop_loss,
            target=order_data.target,
            broker=order_data.broker,
            db=db,
            upstox_apis=upstox_apis,
            kite_apis=kite_apis,
            user_id=current_user.user_id
        )
        if result:
            order_id = result.data.order_id if order_data.broker == "Upstox" else result
            return {"status": "success", "order_id": order_id}
        raise HTTPException(status_code=500, detail="Failed to place order")
    except ValueError as e:
        logger.error(f"ValueError in place_order_endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in place_order_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orders/{broker}", response_model=List[dict], tags=["orders"])
async def get_orders(
        broker: str,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker specified")

    query = "SELECT * FROM orders WHERE user_id = :user_id AND broker = :broker"
    params = {"user_id": current_user.user_id, "broker": broker}
    try:
        orders = await async_fetch_query(db, text(query), params)
        return orders
    except Exception as e:
        logger.error(f"Failed to fetch orders for broker {broker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/orders/{order_id}", tags=["orders"])
async def cancel_order(
        order_id: str,
        broker: str,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    if current_user.user_id not in app.user_apis:
        raise HTTPException(status_code=400, detail="APIs not initialized for this user. Please log in again.")
    user_apis = app.user_apis[current_user.user_id]
    upstox_apis = user_apis["upstox"]
    kite_apis = user_apis["zerodha"]

    api = upstox_apis["order"] if broker == "Upstox" else kite_apis["kite"]

    # Check if the required API is initialized
    if broker == "Upstox" and not upstox_apis["order"]:
        raise HTTPException(status_code=400,
                            detail="Upstox API not initialized. Please authenticate via /auth/upstox/.")
    if broker == "Zerodha" and not kite_apis["kite"]:
        raise HTTPException(status_code=400,
                            detail="Zerodha API not initialized. Please authenticate via /auth/zerodha/.")

    try:
        if broker == "Upstox":
            response = api.cancel_order(order_id, api_version="2.0")
        else:
            response = api.cancel_order(variety=api.VARIETY_REGULAR, order_id=order_id)
        return {"status": "success", "message": f"Order {order_id} cancelled"}
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gtt-orders/", tags=["orders"])
async def place_gtt_order_endpoint(
        gtt_data: GTTOrderRequest,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    if gtt_data.broker != "Zerodha":
        raise HTTPException(status_code=400, detail="GTT orders only supported for Zerodha")

    if current_user.user_id not in app.user_apis:
        raise HTTPException(status_code=400, detail="APIs not initialized for this user. Please log in again.")
    kite_apis = app.user_apis[current_user.user_id]["zerodha"]

    # Check if Zerodha API is initialized
    if not kite_apis["kite"]:
        raise HTTPException(status_code=400,
                            detail="Zerodha API not initialized. Please authenticate via /auth/zerodha/.")

    try:
        result = await app.order_manager.place_gtt_order(
            api=kite_apis["kite"],
            instrument_token=gtt_data.instrument_token,
            trading_symbol=gtt_data.trading_symbol,
            transaction_type=gtt_data.transaction_type,
            quantity=gtt_data.quantity,
            trigger_type=gtt_data.trigger_type,
            trigger_price=gtt_data.trigger_price,
            limit_price=gtt_data.limit_price,
            last_price=gtt_data.last_price,
            second_trigger_price=gtt_data.second_trigger_price,
            second_limit_price=gtt_data.second_limit_price,
            broker=gtt_data.broker,
            db=db
        )
        if result["status"] == "success":
            return {"status": "success", "gtt_id": result["gtt_id"]}
        raise HTTPException(status_code=500, detail=result["message"])
    except Exception as e:
        logger.error(f"Error in place_gtt_order_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scheduled-orders/", tags=["orders"])
async def schedule_order(
        order_data: ScheduledOrder,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    # Validate required fields
    if not order_data.broker or order_data.broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Broker must be specified and must be 'Upstox' or 'Zerodha'")
    if not order_data.instrument_token:
        raise HTTPException(status_code=400, detail="Instrument token is required")
    if not order_data.transaction_type:
        raise HTTPException(status_code=400, detail="Transaction type is required")
    if not order_data.quantity:
        raise HTTPException(status_code=400, detail="Quantity is required")
    if not order_data.order_type:
        raise HTTPException(status_code=400, detail="Order type is required")
    if not order_data.product_type:
        raise HTTPException(status_code=400, detail="Product type is required")

    # Set default values and required fields
    order_data.scheduled_order_id = f"scheduled_{uuid.uuid4().hex.upper()[0:6]}"
    order_data.status = "PENDING"
    order_data.is_amo = order_data.is_amo if order_data.is_amo is not None else False
    order_data.user_id = current_user.user_id

    scheduled_order_df = pd.DataFrame([order_data.dict()])
    try:
        await load_sql_data(scheduled_order_df, "scheduled_orders", load_type="append", index_required=False, db=db,
                            database=DATABASE)
        if current_user.user_id not in app.user_apis:
            raise HTTPException(status_code=400, detail="APIs not initialized for this user. Please log in again.")
        upstox_apis = app.user_apis[current_user.user_id]["upstox"]
        kite_apis = app.user_apis[current_user.user_id]["zerodha"]
        if order_data.broker == "Upstox" and not upstox_apis["order"]:
            raise HTTPException(status_code=400,
                                detail="Upstox API not initialized. Please authenticate via /auth/upstox/.")
        if order_data.broker == "Zerodha" and not kite_apis["kite"]:
            raise HTTPException(status_code=400,
                                detail="Zerodha API not initialized. Please authenticate via /auth/zerodha/.")
        await app.order_manager._execute_scheduled_order(order_data.dict(), db, upstox_apis, kite_apis)
        return {"status": "success", "order_id": order_data.scheduled_order_id}
    except Exception as e:
        logger.error(f"Error in schedule_order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auto-orders/", tags=["orders"])
async def add_auto_order(
        auto_order_data: AutoOrder,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    # Validate required fields
    if not auto_order_data.instrument_token:
        raise HTTPException(status_code=400, detail="Instrument token is required")
    if not auto_order_data.transaction_type:
        raise HTTPException(status_code=400, detail="Transaction type is required")

    auto_order_data.auto_order_id = f"auto_{uuid.uuid4().hex.upper()[0:6]}"
    auto_order_data.user_id = current_user.user_id
    auto_order_df = pd.DataFrame([auto_order_data.dict()])
    try:
        await load_sql_data(auto_order_df, "auto_orders", load_type="append", index_required=False, db=db,
                            database=DATABASE)
        return {"status": "success", "auto_order_id": auto_order_data.auto_order_id}
    except Exception as e:
        logger.error(f"Error in add_auto_order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/order-book/{broker}", tags=["orders"])
async def get_order_book_endpoint(
        broker: str,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker specified")

    if current_user.user_id not in app.user_apis:
        raise HTTPException(status_code=400, detail="APIs not initialized for this user. Please log in again.")
    upstox_apis = app.user_apis[current_user.user_id]["upstox"]
    kite_apis = app.user_apis[current_user.user_id]["zerodha"]

    # Check if the specified broker's API is initialized
    if broker == "Upstox" and not upstox_apis["order"]:
        raise HTTPException(status_code=400,
                            detail="Upstox API not initialized. Please authenticate via /auth/upstox/.")
    if broker == "Zerodha" and not kite_apis["kite"]:
        raise HTTPException(status_code=400,
                            detail="Zerodha API not initialized. Please authenticate via /auth/zerodha/.")

    try:
        # Pass only the relevant API based on the broker
        upstox_api = upstox_apis["order"] if broker == "Upstox" else None
        kite_api = kite_apis["kite"] if broker == "Zerodha" else None
        orders_df = get_order_book(upstox_api, kite_api)
        return orders_df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error in get_order_book_endpoint for broker {broker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions/{broker}", tags=["portfolio"])
async def get_positions_endpoint(
        broker: str,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker specified")

    if current_user.user_id not in app.user_apis:
        raise HTTPException(status_code=400, detail="APIs not initialized for this user. Please log in again.")
    upstox_apis = app.user_apis[current_user.user_id]["upstox"]
    kite_apis = app.user_apis[current_user.user_id]["zerodha"]

    # Check if the specified broker's API is initialized
    if broker == "Upstox" and not upstox_apis["portfolio"]:
        raise HTTPException(status_code=400,
                            detail="Upstox API not initialized. Please authenticate via /auth/upstox/.")
    if broker == "Zerodha" and not kite_apis["kite"]:
        raise HTTPException(status_code=400,
                            detail="Zerodha API not initialized. Please authenticate via /auth/zerodha/.")

    try:
        # Pass only the relevant API based on the broker
        upstox_api = upstox_apis["portfolio"] if broker == "Upstox" else None
        kite_api = kite_apis["kite"] if broker == "Zerodha" else None
        positions_df = get_positions(upstox_api, kite_api)
        return positions_df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error in get_positions_endpoint for broker {broker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/{broker}", tags=["portfolio"])
async def get_portfolio_endpoint(
        broker: str,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker specified")

    if current_user.user_id not in app.user_apis:
        raise HTTPException(status_code=400, detail="APIs not initialized for this user. Please log in again.")
    upstox_apis = app.user_apis[current_user.user_id]["upstox"]
    kite_apis = app.user_apis[current_user.user_id]["zerodha"]

    # Check if the specified broker's API is initialized
    if broker == "Upstox" and not upstox_apis["portfolio"]:
        raise HTTPException(status_code=400,
                            detail="Upstox API not initialized. Please authenticate via /auth/upstox/.")
    if broker == "Zerodha" and not kite_apis["kite"]:
        raise HTTPException(status_code=400,
                            detail="Zerodha API not initialized. Please authenticate via /auth/zerodha/.")

    try:
        # Pass only the relevant API based on the broker
        upstox_api = upstox_apis["portfolio"] if broker == "Upstox" else None
        kite_api = kite_apis["kite"] if broker == "Zerodha" else None
        portfolio_df = get_portfolio(upstox_api, kite_api)
        return portfolio_df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error in get_portfolio_endpoint for broker {broker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trade-history/{broker}", response_model=List[dict], tags=["portfolio"])
async def get_trade_history(
        broker: str,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker specified")

    query = "SELECT * FROM trade_history WHERE user_id = :user_id AND broker = :broker ORDER BY exit_time DESC LIMIT 10"
    params = {"user_id": current_user.user_id, "broker": broker}
    try:
        trades = await async_fetch_query(db, text(query), params)
        return trades
    except Exception as e:
        logger.error(f"Error in get_trade_history for broker {broker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/funds/{broker}", tags=["portfolio"])
async def get_funds_data_endpoint(
        broker: str,
        current_user: User = Depends(UserManager.get_current_user),
        db: AsyncSession = Depends(get_db)
):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker specified")

    if current_user.user_id not in app.user_apis:
        raise HTTPException(status_code=400, detail="APIs not initialized for this user. Please log in again.")
    upstox_apis = app.user_apis[current_user.user_id]["upstox"]
    kite_apis = app.user_apis[current_user.user_id]["zerodha"]

    # Check if the required API is initialized
    if broker == "Upstox" and not upstox_apis["user"]:
        raise HTTPException(status_code=400,
                            detail="Upstox API not initialized. Please authenticate via /auth/upstox/.")
    if broker == "Zerodha" and not kite_apis["kite"]:
        raise HTTPException(status_code=400,
                            detail="Zerodha API not initialized. Please authenticate via /auth/zerodha/.")

    try:
        logger.info(f"Inside {broker} funds")
        funds_data = get_funds_data(
            upstox_apis if broker == "Upstox" else kite_apis,
            broker
        )
        logger.info("Funds data fetched successfully")
        logger.info(funds_data)
        if not funds_data:
            raise HTTPException(status_code=404, detail="Funds data not found")
        return funds_data
    except Exception as e:
        logger.error(f"Error in get_funds_data_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))