import sys
import os
import logging
import asyncio

# Ensure consistent import path - add this RIGHT after your existing path manipulation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create logs directory if it doesn't exist (use absolute path)
app_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(app_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging to project root logs directory for easier access
root_logs_dir = os.path.join(project_root, 'logs') 
os.makedirs(root_logs_dir, exist_ok=True)
backend_log_file = os.path.join(root_logs_dir, 'backend.log')

# Configure root logger to capture all logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Remove any existing handlers to avoid duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# Create and configure file handler
file_handler = logging.FileHandler(backend_log_file, mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Create and configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Configure uvicorn loggers to use our handlers
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers = [file_handler, console_handler]
uvicorn_logger.setLevel(logging.INFO)

uvicorn_access_logger = logging.getLogger("uvicorn.access")  
uvicorn_access_logger.handlers = [file_handler, console_handler]
uvicorn_access_logger.setLevel(logging.INFO)

uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_error_logger.handlers = [file_handler, console_handler]
uvicorn_error_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Log startup information
logger.info("="*50)
logger.info("TRADING APPLICATION STARTING")
logger.info(f"App directory: {app_dir}")
logger.info(f"Project root: {project_root}")
logger.info(f"Backend log file: {backend_log_file}")
logger.info("="*50)

# Now import everything else in the same order as before
import aiohttp
from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, Query
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timedelta
import uuid
import pandas as pd
import bcrypt
from collections import defaultdict
from uuid import uuid4

from backend.app.database import (
    init_databases,
    cleanup_databases,
    db_manager,
    get_db,
    get_nsedata_db
)

from backend.app.api_manager import initialize_user_apis
from backend.app.routes.sip_routes import sip_router
from backend.app.routes.watchlist import watchlist_router
from backend.app.routes.equity_data_routes import equity_router, nse_openchart
from backend.app.routes.mutual_fund_routes import mf_router
from backend.app.routes.scanner_routes import router as scanner_router
from backend.app.routes import data

# Continue with your other imports...
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text, func
from sqlalchemy import and_, or_

# Rest of your imports exactly as they were
# For standalone debugging - add this check
try:
    from .models import Order, ScheduledOrder as ScheduledOrderModel, AutoOrder as AutoOrderModel, GTTOrder as GTTOrderModel, User, Strategy, Position, StrategyExecution
    from .services import (
    OrderManager, OrderMonitor, fetch_upstox_access_token, fetch_zerodha_access_token, place_order,
    get_order_book, get_positions, get_portfolio, get_funds_data, get_quotes, get_ohlc, get_ltp,
    get_historical_data, fetch_instruments, get_order_history, get_order_trades, modify_order, TokenExpiredError,
    execute_strategy, backtest_strategy, get_mf_sips, schedule_strategy_execution, stop_strategy_execution,
        get_quotes_from_nse, get_ltp_openchart, get_ltp_from_nse, get_ohlc_openchart, get_ohlc_from_nse
    )
    from .schemas import (
        PlaceOrderRequest, UserCreate, UserResponse, ScheduledOrder, ScheduledOrderRequest, AutoOrder, AutoOrderRequest,
        GTTOrder, GTTOrderRequest, ProfileResponse, MarginResponse, QuoteResponse, OHLCResponse, LTPResponse,
        HistoricalDataResponse, Instrument, OrderHistory, Trade, ModifyOrderRequest, StrategyRequest, StrategyResponse,
        BacktestRequest, MFSIPResponse, StrategyExecutionRequest, PartialExit, MarketDataRequest, HistoricalDataRequest
    )
except ImportError:
    # Fallback to absolute imports for standalone debugging
    from backend.app.models import Order, ScheduledOrder as ScheduledOrderModel, AutoOrder as AutoOrderModel, GTTOrder as GTTOrderModel, User, Strategy, Position, StrategyExecution
    from backend.app.services import (
        OrderManager, OrderMonitor, fetch_upstox_access_token, fetch_zerodha_access_token, place_order,
        get_order_book, get_positions, get_portfolio, get_funds_data, get_quotes, get_ohlc, get_ltp,
        get_historical_data, fetch_instruments, get_order_history, get_order_trades, modify_order, TokenExpiredError,
        execute_strategy, backtest_strategy, get_mf_sips, schedule_strategy_execution, stop_strategy_execution,
        get_quotes_from_nse, get_ltp_openchart, get_ltp_from_nse, get_ohlc_openchart, get_ohlc_from_nse
    )
    from backend.app.schemas import (
        PlaceOrderRequest, UserCreate, UserResponse, ScheduledOrder, ScheduledOrderRequest, AutoOrder, AutoOrderRequest,
        GTTOrder, GTTOrderRequest, ProfileResponse, MarginResponse, QuoteResponse, OHLCResponse, LTPResponse,
        HistoricalDataResponse, Instrument, OrderHistory, Trade, ModifyOrderRequest, StrategyRequest, StrategyResponse,
        BacktestRequest, MFSIPResponse, StrategyExecutionRequest, PartialExit, MarketDataRequest, HistoricalDataRequest
    )
from backend.app.strategy_engine import StrategyExecutionEngine
from backend.app.risk_manager import RiskManager
from backend.app.cache_manager import cache_manager, cache_user_data, cache_portfolio_data, CacheConfig, invalidate_user_cache
from common_utils.upstox_utils import get_symbol_for_instrument
from common_utils.db_utils import async_fetch_query
from common_utils.utils import sanitize_floats
from common_utils.read_write_sql_data import load_sql_data
from common_utils.market_data import MarketData
# Auth import with fallback
try:
    from .auth import UserManager, oauth2_scheme
except ImportError:
    from backend.app.auth import UserManager, oauth2_scheme

# Initialize global variables that existing functions depend on
order_monitor = None
order_manager = None
background_tasks = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simplified lifespan with reliable startup and shutdown"""
    global order_monitor, order_manager, user_apis, background_tasks

    logger.info("Starting Modern Trading Platform...")
    
    # Ensure logging configuration persists after uvicorn initialization
    logger.info("Reconfiguring logging to ensure all logs go to file...")
    
    # Re-apply our logging configuration to handle uvicorn overrides
    root_logger = logging.getLogger()
    
    # Ensure our file handler is still attached to all relevant loggers
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        specific_logger = logging.getLogger(logger_name)
        if file_handler not in specific_logger.handlers:
            specific_logger.addHandler(file_handler)
        specific_logger.setLevel(logging.INFO)
    
    logger.info("Logging configuration verified - all logs will be written to file")
    startup_tasks = []

    try:
        # Step 1: Initialize databases
        logger.info("Initializing database system...")
        await init_databases()
        logger.info("Databases initialized successfully")
        # One-time, idempotent schema adjustments for GTT
        try:
            session_factory = db_manager.get_session_factory('trading_db')
            if session_factory:
                async with session_factory() as db:
                    try:
                        await db.execute(text("ALTER TABLE gtt_orders ADD COLUMN IF NOT EXISTS gtt_type TEXT"))
                    except Exception as e:
                        logger.debug(f"gtt_type migration note: {e}")
                    try:
                        await db.execute(text("ALTER TABLE gtt_orders ADD COLUMN IF NOT EXISTS rules JSON"))
                    except Exception as e:
                        logger.debug(f"rules migration note: {e}")
                    try:
                        await db.commit()
                    except Exception as e:
                        logger.warning(f"Schema migration commit warning: {e}")
        except Exception as mig_err:
            logger.warning(f"Non-fatal schema migration issue: {mig_err}")

        # Step 2: Initialize services
        logger.info("Initializing services...")
        order_monitor = OrderMonitor()
        order_manager = OrderManager(monitor=order_monitor)
        market_data = MarketData()
        user_apis = {}
        logger.info("Services initialized")

        # Step 3: Start background services with error handling
        logger.info("Starting background services...")

        async def safe_monitor_task():
            try:
                await order_monitor.run_scheduled_tasks(user_apis=user_apis)
            except Exception as e:
                logger.error(f"OrderMonitor error: {e}")

        async def safe_manager_task():
            try:
                await order_manager.start(user_apis=user_apis)
            except Exception as e:
                logger.error(f"OrderManager error: {e}")

        # Create tasks with error handling
        monitor_task = asyncio.create_task(safe_monitor_task(), name="monitor")
        manager_task = asyncio.create_task(safe_manager_task(), name="manager")
        startup_tasks = [monitor_task, manager_task]
        background_tasks = startup_tasks.copy()

        logger.info("Background services started")
        logger.info("Trading platform is ready!")

        # Initialize strategy engine and risk manager
        app.state.strategy_engine = StrategyExecutionEngine()
        app.state.risk_manager = RiskManager()
        
        # Initialize market hours manager
        from .market_hours_manager import MarketHoursManager
        app.state.market_hours_manager = MarketHoursManager(app.state.strategy_engine)
        await app.state.market_hours_manager.start_monitoring()
        
        logger.info("Strategy execution engine, risk manager, and market hours manager initialized")
        
        # Debug: Log all registered routes
        logger.info("Registered routes:")
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                logger.info(f"  {route.methods} {route.path}")
        
        # Store in app state
        app.state.order_monitor = order_monitor
        app.state.order_manager = order_manager
        app.state.user_apis = user_apis
        app.state.market_data = market_data

        # Application runs here
        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Cancel any tasks that were started
        for task in startup_tasks:
            if not task.done():
                task.cancel()
        raise
    finally:
        # Simple, reliable shutdown
        logger.info("Shutting down...")

        try:
            # Step 1: Stop services
            if order_monitor:
                order_monitor.running = False
            if order_manager:
                order_manager.running = False

            # Step 2: Cancel background tasks quickly
            for task in background_tasks:
                if not task.done():
                    task.cancel()

            # Step 3: Wait briefly for cancellation (don't hang)
            if background_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*background_tasks, return_exceptions=True),
                        timeout=2.0  # Short timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Tasks didn't cancel in time - proceeding")

            # Step 4: Close databases
            await cleanup_databases()

            logger.info("Shutdown completed")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")

app = FastAPI(
    title="Modern Algorithmic Trading Platform",
    description="Clean, scalable trading platform with multi-database architecture",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "orders", "description": "Order management endpoints"},
        {"name": "portfolio", "description": "Portfolio and positions endpoints"},
        {"name": "market-data", "description": "Market data endpoints"},
        {"name": "gtt-orders", "description": "Good Till Triggered (GTT) order endpoints"},
        {"name": "algo-trading", "description": "Algorithmic trading endpoints"},
        {"name": "mutual-funds", "description": "Mutual fund operations"},
        {"name": "sip-strategy", "description": "SIP strategy endpoints"},
        {"name": "system", "description": "System health and monitoring"}
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests for monitoring"""
    start_time = asyncio.get_event_loop().time()

    try:
        response = await call_next(request)
        process_time = asyncio.get_event_loop().time() - start_time

        # Log only slow requests or errors to reduce noise
        if process_time > 1.0 or response.status_code >= 400:
            logger.warning(
                f"{request.method} {request.url.path} "
                f"- Status: {response.status_code} "
                f"- Time: {process_time:.3f}s"
            )

        response.headers["X-Process-Time"] = str(process_time)
        return response

    except Exception as e:
        process_time = asyncio.get_event_loop().time() - start_time
        logger.error(
            f"{request.method} {request.url.path} "
            f"- Error: {str(e)} "
            f"- Time: {process_time:.3f}s"
        )
        raise

# Include routers
app.include_router(sip_router, tags=["sip-strategy"])
app.include_router(data.router, prefix="/api", tags=["data"])
app.include_router(watchlist_router, prefix="/api", tags=["watchlist"])
app.include_router(scanner_router, prefix="/api", tags=["scanner"])
app.include_router(equity_router)
app.include_router(mf_router)

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

@app.get("/health", tags=["system"])
async def health_check():
    """Comprehensive application health check"""
    try:
        # Check database health
        db_health = await db_manager.health_check()

        # Determine overall status
        overall_status = "healthy"
        unhealthy_databases = []

        for db_name, status in db_health.items():
            if status.get('status') != 'healthy':
                overall_status = "unhealthy"
                unhealthy_databases.append(db_name)

        # Check global services
        services_status = {
            "order_monitor": "initialized" if order_monitor else "not_initialized",
            "order_manager": "initialized" if order_manager else "not_initialized",
            "user_apis_count": len(user_apis),
            "background_tasks_count": len(background_tasks)
        }

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0",
            "databases": db_health,
            "services": services_status,
            "unhealthy_databases": unhealthy_databases,
            "active_asyncio_tasks": len([task for task in asyncio.all_tasks() if not task.done()]),
            "environment": os.getenv("ENVIRONMENT", "development")
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.get("/", tags=["system"])
async def root():
    """Root endpoint with basic application information"""
    return {
        "message": "Modern Algorithmic Trading Platform",
        "version": "2.1.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs_url": "/docs",
        "health_check_url": "/health",
        "features": [
            "Enhanced database connection management",
            "Proper resource cleanup",
            "Global service compatibility",
            "Comprehensive monitoring"
        ]
    }

# ============================================================================
# APPLICATION STARTUP/SHUTDOWN EVENTS
# ============================================================================
@app.get("/system/info")
async def get_system_info():
    """Get detailed system information"""
    try:
        system_info = {
            "platform": "Modern Trading Platform",
            "version": "3.0.0",
            "databases": {},
            "active_users": len(user_apis),
            "uptime": datetime.now().isoformat()
        }

        # Database information
        for db_name, config in db_manager.db_configs.items():
            session_factory = db_manager.get_session_factory(db_name)

            db_info = {
                "description": config["description"],
                "status": "connected" if session_factory else "disconnected",
                "tables": 0
            }

            # Get table count
            if session_factory:
                try:
                    async with session_factory() as session:
                        result = await session.execute(text("""
                            SELECT COUNT(*) FROM information_schema.tables 
                            WHERE table_schema = 'public'
                        """))
                        db_info["tables"] = result.scalar()
                except Exception:
                    db_info["tables"] = "unknown"

            system_info["databases"][db_name] = db_info

        return system_info

    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/register", response_model=UserResponse, tags=["auth"])
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register new user"""
    # Check if user exists
    result = await db.execute(select(User).filter(User.email == user.email))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = User(
        user_id=str(uuid.uuid4()),
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

    logger.info(f"New user registered: {user.email}")
    return UserResponse.from_orm(new_user)


@app.post("/auth/login", tags=["auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    """User login"""
    # Verify user credentials
    result = await db.execute(select(User).filter(User.email == form_data.username))
    user = result.scalars().first()

    if not user or not bcrypt.checkpw(form_data.password.encode('utf-8'), user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Initialize user APIs
    try:
        await initialize_user_apis(user.user_id, db, force_reinitialize=True)
    except TokenExpiredError:
        logger.info(f"API tokens expired for user {user.user_id}")

    # Create access token
    token = UserManager.create_access_token(data={"sub": user.user_id})

    logger.info(f"User logged in: {user.email}")
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.user_id
    }


@app.post("/auth/upstox", tags=["auth"])
async def auth_upstox(auth_code: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Authenticate Upstox"""
    try:
        access_token = await fetch_upstox_access_token(db, user_id, auth_code)
        if access_token:
            user_apis[user_id] = await initialize_user_apis(user_id, db, force_reinitialize=True)
            return {"status": "success", "message": "Upstox authentication successful"}
        else:
            raise HTTPException(status_code=400, detail="Upstox authentication failed")
    except Exception as e:
        logger.error(f"Upstox auth error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/auth/token-status/{broker}", tags=["auth"])
async def get_token_status(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get token expiry status for a broker"""
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if broker == "Zerodha":
            token = user.zerodha_access_token
            expiry = user.zerodha_access_token_expiry
        else:  # Upstox
            token = user.upstox_access_token
            expiry = user.upstox_access_token_expiry
        
        is_valid = token and expiry and datetime.now() < expiry
        
        return {
            "broker": broker,
            "has_token": bool(token),
            "is_valid": is_valid,
            "expires_at": expiry.isoformat() if expiry else None,
            "expires_in_hours": (expiry - datetime.now()).total_seconds() / 3600 if is_valid else 0
        }
    except Exception as e:
        logger.error(f"Error checking {broker} token status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/auth/revoke-token/{broker}", tags=["auth"])
async def revoke_token(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Revoke broker access token"""
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    
    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if broker == "Zerodha":
            user.zerodha_access_token = None
            user.zerodha_access_token_expiry = None
        else:  # Upstox
            user.upstox_access_token = None
            user.upstox_access_token_expiry = None
        
        await db.commit()
        
        # Clear from user_apis cache
        if user_id in user_apis:
            del user_apis[user_id]
        
        logger.info(f"{broker} token revoked for user {user_id}")
        return {"status": "success", "message": f"{broker} token revoked successfully"}
    except Exception as e:
        logger.error(f"Error revoking {broker} token: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    
    # CRITICAL SECURITY FIX: Add comprehensive risk validation for all manual orders
    if hasattr(app.state, 'risk_manager'):
        try:
            # Comprehensive trading check (includes emergency stop, circuit breaker, market hours, user limits)
            trading_allowed, reason = await app.state.risk_manager.comprehensive_trading_check(user_id, db)
            if not trading_allowed:
                logger.warning(f"Order rejected for user {user_id}: {reason}")
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": {
                            "code": "TRADING_SUSPENDED", 
                            "message": f"Trading not allowed: {reason}"
                        }
                    }
                )
            
            # Validate the specific trade parameters
            trade_value = abs(order.quantity * (order.price if order.price > 0 else 100.0))  # Use price or estimate
            is_valid, validation_msg = await app.state.risk_manager.validate_trade(
                user_id, 
                order.instrument_token, 
                order.quantity, 
                order.price if order.price > 0 else 100.0,  # Use provided price or estimate for market orders
                order.transaction_type, 
                db
            )
            
            if not is_valid:
                logger.warning(f"Order validation failed for user {user_id}: {validation_msg}")
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": {
                            "code": "RISK_VALIDATION_FAILED", 
                            "message": validation_msg
                        }
                    }
                )
            
            logger.info(f"Risk validation passed for user {user_id}, order value: INR {trade_value:,.2f}")
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions as-is
        except Exception as e:
            logger.error(f"Risk validation error for user {user_id}: {e}")
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": {
                        "code": "RISK_VALIDATION_ERROR", 
                        "message": f"Risk validation failed: {str(e)}"
                    }
                }
            )
    else:
        logger.error("Risk manager not initialized - this is a critical security issue")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": {
                    "code": "RISK_MANAGER_UNAVAILABLE", 
                    "message": "Risk management system unavailable"
                }
            }
        )
    
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
                "is_amo": order.is_amo,
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
                is_trailing_stop_loss=order.is_trailing_stop_loss,
                trailing_stop_loss_percent=order.trailing_stop_loss_percent,
                trail_start_target_percent=order.trail_start_target_percent
            )
            
            # Invalidate user's order and position cache after successful order placement
            await cache_manager.delete_pattern(f"orders:*{user_id}*{order.broker}*")
            await cache_manager.delete_pattern(f"positions:*{user_id}*{order.broker}*")
            
            # Handle different response formats for different brokers
            if order.broker == "Upstox":
                # Upstox returns response object with order_ids array
                if hasattr(response.data, 'order_ids') and response.data.order_ids:
                    order_id = response.data.order_ids[0]  # Take the first order ID
                elif hasattr(response.data, 'order_id'):
                    order_id = response.data.order_id  # Fallback for single order_id
                else:
                    order_id = "unknown"  # Fallback if no order ID found
            else:
                # Zerodha returns just the order_id directly
                order_id = response

            return {"status": "success", "order_id": order_id}
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{broker}", tags=["orders"])
async def get_orders(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    
    # Check cache first
    cache_key = cache_manager.generate_cache_key("orders", user_id, broker)
    cached_orders = await cache_manager.get(cache_key)
    if cached_orders:
        return cached_orders
    
    try:
        query = """
            SELECT * FROM orders 
            WHERE user_id = :user_id AND broker = :broker
        """
        orders = pd.DataFrame(await async_fetch_query(db, text(query), {"user_id": user_id, "broker": broker}))

        # Handle NaN, infinity, and negative infinity values for JSON compliance
        if not orders.empty:
            # Replace infinite values with None
            orders = orders.replace([float('inf'), float('-inf')], None)

            # Handle NaN values by converting to None
            import numpy as np
            orders = orders.replace([np.nan], None)

            # Additional safety: convert to native Python types and handle any remaining problematic values
            orders_dict = orders.to_dict(orient="records")

            # Clean the dictionary to ensure JSON compliance
            cleaned_orders = []
            for order in orders_dict:
                cleaned_order = {}
                for key, value in order.items():
                    if pd.isna(value) or value is np.nan or value == float('inf') or value == float('-inf'):
                        cleaned_order[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        # Convert numpy types to native Python types
                        if np.isfinite(value):
                            cleaned_order[key] = value.item()
                        else:
                            cleaned_order[key] = None
                    else:
                        cleaned_order[key] = value
                cleaned_orders.append(cleaned_order)

            # Cache the cleaned orders for 30 seconds (orders change frequently)
            await cache_manager.set(cache_key, cleaned_orders, CacheConfig.ORDERS)
            return cleaned_orders
        else:
            empty_result = []
            await cache_manager.set(cache_key, empty_result, CacheConfig.ORDERS)
            return empty_result
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
            api.cancel_order(order_id=order_id)
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
    upstox_api = user_apis_dict["upstox"]["order_v2"]
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
    upstox_api = user_apis_dict["upstox"]["order_v2"]
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
    upstox_api = user_apis_dict["upstox"]["order_v2"] if broker == "Upstox" else None
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

    # Ensure user APIs are available in the global user_apis
    try:
        user_apis_dict = await initialize_user_apis(user_id, db)
        if user_apis_dict and hasattr(app.state, 'user_apis'):
            app.state.user_apis[user_id] = user_apis_dict
            logger.info(f"Updated user APIs for user {user_id} in scheduled order creation")
    except Exception as e:
        logger.warning(f"Could not initialize user APIs for scheduled order: {e}")

    # Reload scheduled orders in OrderManager
    if hasattr(app.state, 'order_manager') and app.state.order_manager:
        try:
            await app.state.order_manager._load_scheduled_orders(db)
            logger.info("Reloaded scheduled orders in OrderManager")
        except Exception as e:
            logger.error(f"Error reloading scheduled orders: {e}")

    return ScheduledOrder.from_orm(scheduled_order)

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

@app.delete("/scheduled-orders/{scheduled_order_id}", tags=["orders"])
async def delete_scheduled_order(scheduled_order_id: str, broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    try:
        query = select(ScheduledOrderModel).filter(ScheduledOrderModel.scheduled_order_id == scheduled_order_id, ScheduledOrderModel.user_id == user_id, ScheduledOrderModel.broker == broker)
        result = await db.execute(query)
        scheduled_order = result.scalars().first()
        if not scheduled_order:
            raise HTTPException(status_code=404, detail="Scheduled order not found")
        await db.delete(scheduled_order)
        await db.commit()
        return {"status": "success", "message": f"Scheduled order {scheduled_order_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting scheduled order {scheduled_order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/scheduled-orders/{scheduled_order_id}", tags=["orders"])
async def modify_scheduled_order(scheduled_order_id: str, order: ScheduledOrderRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if order.broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    try:
        query = select(ScheduledOrderModel).filter(ScheduledOrderModel.scheduled_order_id == scheduled_order_id, ScheduledOrderModel.user_id == user_id, ScheduledOrderModel.broker == order.broker)
        result = await db.execute(query)
        scheduled_order = result.scalars().first()
        if not scheduled_order:
            raise HTTPException(status_code=404, detail="Scheduled order not found")

        # Update fields
        scheduled_order.instrument_token = order.instrument_token
        scheduled_order.trading_symbol = order.trading_symbol
        scheduled_order.transaction_type = order.transaction_type
        scheduled_order.quantity = order.quantity
        scheduled_order.order_type = order.order_type
        scheduled_order.price = order.price
        scheduled_order.trigger_price = order.trigger_price
        scheduled_order.product_type = order.product_type
        scheduled_order.schedule_datetime = order.schedule_datetime
        scheduled_order.stop_loss = order.stop_loss
        scheduled_order.target = order.target
        scheduled_order.is_amo = order.is_amo

        await db.commit()
        await db.refresh(scheduled_order)

        return ScheduledOrder.from_orm(scheduled_order)
    except Exception as e:
        logger.error(f"Error modifying scheduled order {scheduled_order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/gtt-orders/", tags=["gtt-orders"])
async def create_gtt_order(order: GTTOrderRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    api = user_apis_dict["zerodha"]["kite"] if order.broker == "Zerodha" else user_apis_dict["upstox"]["order"]
    if not api:
        raise HTTPException(status_code=400, detail=f"{order.broker} API not initialized")
    
    # Add risk validation for GTT orders
    if hasattr(app.state, 'risk_manager'):
        try:
            trading_allowed, reason = await app.state.risk_manager.check_trading_allowed(user_id, db)
            if not trading_allowed:
                raise HTTPException(
                    status_code=400, 
                    detail={"error": {"code": "TRADING_SUSPENDED", "message": f"GTT order rejected: {reason}"}}
                )
            
            # Validate GTT order size
            # Determine an effective price for validation (handles None for Upstox rules flow)
            effective_price = 0.0
            if order.limit_price is not None and order.limit_price > 0:
                effective_price = order.limit_price
            elif order.trigger_price is not None and order.trigger_price > 0:
                effective_price = order.trigger_price

            is_valid, validation_msg = await app.state.risk_manager.validate_trade(
                user_id, order.instrument_token, order.quantity,
                effective_price,
                order.transaction_type, db
            )
            
            if not is_valid:
                raise HTTPException(
                    status_code=400, 
                    detail={"error": {"code": "RISK_VALIDATION_FAILED", "message": f"GTT order validation failed: {validation_msg}"}}
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GTT order risk validation error: {str(e)}")
    
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
            rules=order.rules,
            broker=order.broker,
            db=db,
            user_id=user_id
        )
        # Normalize error responses to HTTP errors
        if isinstance(response, dict) and response.get('status') == 'error':
            raise HTTPException(status_code=400, detail=response.get('message', 'GTT order failed'))
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error placing GTT order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gtt-orders/{broker}", response_model=List[GTTOrder], tags=["gtt-orders"])
async def get_gtt_orders(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    upstox_api = user_apis_dict["upstox"].get("order") if broker == "Upstox" else None
    
    if broker == "Zerodha" and not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    elif broker == "Upstox" and not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API not initialized")
        
    try:
        if broker == "Zerodha":
            gtt_orders = kite_api.get_gtts()
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
                    user_id=user_id,
                    gtt_type="SINGLE" if gtt["type"] == "single" else "MULTIPLE"
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
    kite_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    upstox_api = user_apis_dict["upstox"].get("order") if broker == "Upstox" else None
    
    if broker == "Zerodha" and not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    elif broker == "Upstox" and not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API not initialized")
        
    try:
        if broker == "Zerodha":
            gtt = kite_api.get_gtt(trigger_id=gtt_id)
            # Build rules array for Zerodha GTT orders
            rules = []
            if gtt["condition"]["trigger_values"]:
                rules.append({
                    "strategy": "ENTRY",
                    "trigger_type": "ABOVE" if gtt["orders"][0]["transaction_type"] == "BUY" else "BELOW",
                    "trigger_price": gtt["condition"]["trigger_values"][0],
                    "rule_index": 0
                })
            
            if len(gtt["condition"]["trigger_values"]) > 1:
                # Determine strategy based on transaction type
                strategy = "TARGET" if gtt["orders"][0]["transaction_type"] == "BUY" else "STOPLOSS"
                rules.append({
                    "strategy": strategy,
                    "trigger_type": "ABOVE" if gtt["orders"][0]["transaction_type"] == "BUY" else "BELOW",
                    "trigger_price": gtt["condition"]["trigger_values"][1],
                    "rule_index": 1
                })
            
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
                user_id=user_id,
                # Add rules for consistency
                gtt_type="SINGLE" if gtt["type"] == "single" else "MULTIPLE",
                rules=rules
            )
        elif broker == "Upstox":
            try:
                # Get GTT order details from Upstox API
                # Note: Upstox API method name might be different
                try:
                    gtt = upstox_api.get_gtt_order_details(gtt_order_id=gtt_id).data[0].to_dict()
                    logger.debug(f"GTT order details: {gtt}")
                except AttributeError:
                    # If no method exists, fallback to database
                    logger.warning("Upstox API method for getting GTT order not found, using database fallback")
                    result = await db.execute(select(GTTOrderModel).filter(GTTOrderModel.gtt_order_id == gtt_id, GTTOrderModel.user_id == user_id))
                    gtt_order = result.scalars().first()
                    if not gtt_order:
                        raise HTTPException(status_code=404, detail="GTT order not found")
                    
                    # Enhance database GTT order with additional fields for frontend compatibility
                    # Build rules array from existing fields for frontend compatibility
                    rules = []
                    if gtt_order.trigger_price:
                        rules.append({
                            "strategy": "ENTRY",
                            "trigger_type": "ABOVE" if gtt_order.transaction_type == "BUY" else "BELOW",
                            "trigger_price": gtt_order.trigger_price,
                            "rule_index": 0
                        })
                    
                    if gtt_order.second_trigger_price:
                        # Determine strategy based on transaction type
                        strategy = "TARGET" if gtt_order.transaction_type == "BUY" else "STOPLOSS"
                        rules.append({
                            "strategy": strategy,
                            "trigger_type": "ABOVE" if gtt_order.transaction_type == "BUY" else "BELOW",
                            "trigger_price": gtt_order.second_trigger_price,
                            "rule_index": 1
                        })
                    
                    # Create enhanced GTTOrder object with all fields
                    enhanced_order = GTTOrder(
                        gtt_order_id=gtt_order.gtt_order_id,
                        instrument_token=gtt_order.instrument_token,
                        trading_symbol=gtt_order.trading_symbol,
                        transaction_type=gtt_order.transaction_type,
                        quantity=gtt_order.quantity,
                        trigger_type=gtt_order.trigger_type,
                        trigger_price=gtt_order.trigger_price,
                        limit_price=gtt_order.limit_price,
                        last_price=gtt_order.last_price,
                        second_trigger_price=gtt_order.second_trigger_price,
                        second_limit_price=gtt_order.second_limit_price,
                        status=gtt_order.status,
                        broker=gtt_order.broker,
                        created_at=gtt_order.created_at,
                        user_id=gtt_order.user_id,
                        # Add Upstox-specific enrichments
                        gtt_type="SINGLE" if gtt_order.trigger_type == "single" else "MULTIPLE",
                        rules=rules
                    )
                    return enhanced_order
                
                # Extract rules from the response - handle all 3 rules
                rules = gtt.get("rules", [])
                
                # Get all rules for comprehensive trigger handling
                first_rule = rules[0] if len(rules) > 0 else {}
                second_rule = rules[1] if len(rules) > 1 else {}
                third_rule = rules[2] if len(rules) > 2 else {}
                
                # Determine trigger type based on number of rules
                gtt_type = gtt.get("type", "SINGLE")
                trigger_type = "single" if gtt_type == "SINGLE" else "two_leg"
                
                # Extract trigger prices and other details from all rules
                trigger_price = first_rule.get("trigger_price", 0.0)
                second_trigger_price = second_rule.get("trigger_price") if second_rule else None
                third_trigger_price = third_rule.get("trigger_price") if third_rule else None
                
                # For Upstox, limit_price is typically the same as trigger_price for ENTRY strategy
                # but this might need adjustment based on your business logic
                limit_price = trigger_price
                second_limit_price = second_trigger_price
                third_limit_price = third_trigger_price
                
                # Get transaction type from the first rule or fallback
                transaction_type = first_rule.get("transaction_type", "BUY")
                
                # Extract trading symbol from instrument token if available
                instrument_token = gtt.get("instrument_token", "")
                trading_symbol = instrument_token.split("|")[-1] if instrument_token and "|" in instrument_token else ""
                
                # Build comprehensive rules list for frontend use
                enhanced_rules = []
                for i, rule in enumerate(rules):
                    enhanced_rule = {
                        "strategy": rule.get("strategy", ""),
                        "trigger_type": rule.get("trigger_type", ""),
                        "trigger_price": rule.get("trigger_price", 0.0),
                        "rule_index": i
                    }
                    # Add trailing_gap if present
                    if rule.get("trailing_gap"):
                        enhanced_rule["trailing_gap"] = rule.get("trailing_gap")
                    enhanced_rules.append(enhanced_rule)
                
                return GTTOrder(
                    gtt_order_id=str(gtt.get("gtt_order_id", gtt_id)),
                    instrument_token=instrument_token,
                    trading_symbol=trading_symbol,
                    transaction_type=transaction_type,
                    quantity=gtt.get("quantity", 0),
                    trigger_type=trigger_type,
                    trigger_price=trigger_price,
                    limit_price=limit_price,
                    last_price=0.0,  # Upstox doesn't provide last_price in GTT details
                    second_trigger_price=second_trigger_price,
                    second_limit_price=second_limit_price,
                    status=gtt.get("status", "").lower(),
                    broker=broker,
                    created_at=datetime.now(),
                    user_id=user_id,
                    # Upstox-specific enrichments
                    gtt_type=gtt_type,
                    rules=enhanced_rules
                )
            except Exception as e:
                logger.error(f"Error fetching Upstox GTT order {gtt_id}: {str(e)}")
                # Fallback to database if API call fails
                result = await db.execute(select(GTTOrderModel).filter(GTTOrderModel.gtt_order_id == gtt_id, GTTOrderModel.user_id == user_id))
                gtt_model = result.scalars().first()
                if not gtt_model:
                    raise HTTPException(status_code=404, detail="GTT order not found")
                return GTTOrder.from_orm(gtt_model)
        else:
            result = await db.execute(select(GTTOrderModel).filter(GTTOrderModel.gtt_order_id == gtt_id, GTTOrderModel.user_id == user_id))
            gtt_model = result.scalars().first()
            if not gtt_model:
                raise HTTPException(status_code=404, detail="GTT order not found")
            return GTTOrder.from_orm(gtt_model)
    except Exception as e:
        logger.error(f"Error fetching GTT order {gtt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/gtt-orders/{broker}/{gtt_id}", tags=["gtt-orders"])
async def modify_gtt_order(broker: str, gtt_id: str, order: GTTOrderRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user_apis_dict = await initialize_user_apis(user_id, db)
    kite_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    upstox_api = user_apis_dict["upstox"].get("order") if broker == "Upstox" else None
    if broker == "Zerodha" and not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    if broker == "Upstox" and not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API not initialized")
    try:
        api_client = kite_api if broker == "Zerodha" else upstox_api
        response = await order_manager.modify_gtt_order(
            api=api_client,
            gtt_id=gtt_id,
            trigger_type=order.trigger_type,
            trigger_price=order.trigger_price,
            limit_price=order.limit_price,
            last_price=order.last_price,
            quantity=order.quantity,
            second_trigger_price=order.second_trigger_price,
            second_limit_price=order.second_limit_price,
            rules=order.rules,  # Pass rules for Upstox GTT orders
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
    kite_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    upstox_api = user_apis_dict["upstox"].get("order") if broker == "Upstox" else None
    if broker == "Zerodha" and not kite_api:
        raise HTTPException(status_code=400, detail="Zerodha API not initialized")
    if broker == "Upstox" and not upstox_api:
        raise HTTPException(status_code=400, detail="Upstox API not initialized")
    try:
        response = await order_monitor.delete_gtt_order(kite_api if broker == "Zerodha" else upstox_api, gtt_id, db)
        return response
    except Exception as e:
        logger.error(f"Error deleting GTT order {gtt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions/{broker}", tags=["portfolio"])
async def get_positions_data(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    
    # Check cache first
    cache_key = cache_manager.generate_cache_key("positions", user_id, broker)
    cached_positions = await cache_manager.get(cache_key)
    if cached_positions:
        return cached_positions
    
    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["portfolio"] if broker == "Upstox" else None
    zerodha_api = user_apis_dict["zerodha"]["kite"] if broker == "Zerodha" else None
    if not (upstox_api or zerodha_api):
        raise HTTPException(status_code=400, detail=f"{broker} API not initialized")
    try:
        positions = await get_positions(upstox_api, zerodha_api)
        positions_data = positions.to_dict(orient="records")
        
        # Cache the result for 1 minute (positions can change frequently)
        await cache_manager.set(cache_key, positions_data, CacheConfig.POSITIONS)
        
        return positions_data
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

# ============================================================================
# MARKET DATA ENDPOINTS
# ============================================================================

# Helper function to process market data parameters
async def process_market_data_params(
    instruments: Optional[str] = None,
    trading_symbols: Optional[str] = None,
    db: AsyncSession = None
) -> List[str]:
    """
    Process market data parameters to get a unified list of instruments/symbols.
    Accepts either instruments or trading_symbols and returns a combined list.
    """
    if not instruments and not trading_symbols:
        raise HTTPException(status_code=400, detail="Either instruments or trading_symbols must be provided")

    combined_list = []

    if instruments:
        combined_list.extend(instruments.split(","))

    if trading_symbols:
        combined_list.extend(trading_symbols.split(","))

    # Remove duplicates while preserving order
    seen = set()
    unique_list = []
    for item in combined_list:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            unique_list.append(item)

    return unique_list

@app.get("/quotes/{broker}", response_model=List[QuoteResponse], tags=["market-data"])
async def get_market_quotes(
    broker: str,
    instruments: Optional[str] = Query(None, description="Comma-separated instrument tokens (e.g., 'NSE_EQ|INE002A01018,NSE_EQ|INE009A01021')"),
    trading_symbols: Optional[str] = Query(None, description="Comma-separated trading symbols (e.g., 'RELIANCE,TCS')"),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get market quotes using either instrument tokens or trading symbols"""
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")

    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["market_data"]

    if not upstox_api:
        logger.debug(f"No upstox api found for {user_id}. Trying with NSE data")

    try:
        # Process both parameter types
        instrument_list = await process_market_data_params(instruments, trading_symbols, db)
        quotes = await get_quotes(upstox_api, None, instrument_list, db)
        return quotes
    except Exception as e:
        logger.error(f"Error fetching {broker} quotes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ohlc/{broker}", response_model=List[OHLCResponse], tags=["market-data"])
async def get_ohlc_data(
    broker: str,
    instruments: Optional[str] = Query(None, description="Comma-separated instrument tokens (e.g., 'NSE_EQ|INE002A01018,NSE_EQ|INE009A01021')"),
    trading_symbols: Optional[str] = Query(None, description="Comma-separated trading symbols (e.g., 'RELIANCE,TCS')"),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get OHLC data using either instrument tokens or trading symbols"""
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")

    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["market_data_v3"]

    if not upstox_api:
        logger.debug(f"No upstox api found for {user_id}. Trying with NSE data")

    try:
        # Process both parameter types
        instrument_list = await process_market_data_params(instruments, trading_symbols, db)
        ohlc_data = await get_ohlc(upstox_api, None, instrument_list, db)
        return ohlc_data
    except Exception as e:
        logger.error(f"Error fetching {broker} OHLC data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ltp/{broker}", response_model=List[LTPResponse], tags=["market-data"])
async def get_ltp_data(
    broker: str,
    instruments: Optional[str] = Query(None, description="Comma-separated instrument tokens (e.g., 'NSE_EQ|INE002A01018,NSE_EQ|INE009A01021')"),
    trading_symbols: Optional[str] = Query(None, description="Comma-separated trading symbols (e.g., 'RELIANCE,TCS')"),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get Last Traded Price using either instrument tokens or trading symbols"""
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")

    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["market_data_v3"]

    if not upstox_api:
        logger.debug(f"No upstox api found for {user_id}. Trying with NSE data")
    try:
        # Process both parameter types
        instrument_list = await process_market_data_params(instruments, trading_symbols, db)
        ltp_data = await get_ltp(upstox_api, None, instrument_list, db)
        return ltp_data
    except Exception as e:
        logger.error(f"Error fetching {broker} LTP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical-data/{broker}", response_model=HistoricalDataResponse, tags=["market-data"])
async def get_historical(
    broker: str,
    instrument: Optional[str] = Query(None, description="Instrument token (e.g., 'NSE_EQ|INE002A01018')"),
    trading_symbol: Optional[str] = Query(None, description="Trading symbol (e.g., 'RELIANCE')"),
    from_date: str = Query(..., description="Start date in YYYY-MM-DD format", example="2025-01-01"),
    to_date: str = Query(..., description="End date in YYYY-MM-DD format", example="2025-12-31"),
    unit: str = Query(..., description="Time unit: 'minute', 'day', 'week', 'month'"),
    interval: str = Query(..., description="Interval value: '1', '3', '5', '10', '15', '30', '60'"),
    source: str = Query("default", description="Data source: 'default', 'upstox', 'nse'"),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    nse_db: AsyncSession = Depends(get_nsedata_db)
):
    """
    Get historical data using either instrument token or trading symbol.

    **Parameters:**
    - **instrument**: Instrument token in format 'EXCHANGE_SEGMENT|ISIN' (e.g., 'NSE_EQ|INE002A01018')
    - **trading_symbol**: Trading symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
    - **from_date**: Start date in YYYY-MM-DD format
    - **to_date**: End date in YYYY-MM-DD format
    - **unit**: Time unit for data points
        - minute: Minute-wise data
        - day: Daily data
        - week: Weekly data
        - month: Monthly data
    - **interval**: Interval value
        - For minute: 1, 3, 5, 10, 15, 30, 60
        - For day/week/month: usually 1
    - **source**: Data source preference
        - default: Auto-select best available source
        - upstox: Use Upstox API
        - nse: Use NSE direct data

    **Example requests:**
    - `/historical-data/Upstox?trading_symbol=RELIANCE&from_date=2024-01-01&to_date=2024-12-31&unit=day&interval=1`
    - `/historical-data/Upstox?instrument=NSE_EQ|INE002A01018&from_date=2024-01-01&to_date=2024-12-31&unit=minute&interval=15`
    """
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")

    # Validate that either instrument or trading_symbol is provided
    if not instrument and not trading_symbol:
        raise HTTPException(status_code=400, detail="Either instrument or trading_symbol must be provided")

    # Validate date format
    try:
        datetime.strptime(from_date, '%Y-%m-%d')
        datetime.strptime(to_date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    # Validate unit and interval
    valid_units = ["minute", "day", "week", "month"]
    valid_intervals = ["1", "3", "5", "10", "15", "30", "60"]

    if unit not in valid_units:
        raise HTTPException(status_code=400, detail=f"Unit must be one of: {', '.join(valid_units)}")

    if interval not in valid_intervals:
        raise HTTPException(status_code=400, detail=f"Interval must be one of: {', '.join(valid_intervals)}")

    user_apis_dict = await initialize_user_apis(user_id, db)
    upstox_api = user_apis_dict["upstox"]["history"]

    if not upstox_api:
        logger.debug(f"No upstox api found for {user_id}. Trying with NSE data")

    try:
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Determine which parameter to use and get trading symbol if needed
        final_instrument = instrument
        final_trading_symbol = trading_symbol

        if instrument and not trading_symbol:
            # Get trading symbol from instrument token
            final_trading_symbol = await get_symbol_for_instrument(instrument)
        elif trading_symbol and not instrument:
            # Use trading symbol directly, instrument can be None
            final_instrument = trading_symbol  # Services can handle both formats

        historical_data = await get_historical_data(
            upstox_api,
            user.upstox_access_token,
            final_trading_symbol,
            from_date,
            to_date,
            unit,
            interval,
            final_instrument,
            db,
            nse_db,
            source
        )
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
async def backtest_strategy_endpoint(request: BacktestRequest, user_id: str = Depends(get_current_user),
                                     db: AsyncSession = Depends(get_db), nse_db: AsyncSession = Depends(get_nsedata_db)):
    try:
        async def ws_callback(data):
            # Placeholder for WebSocket progress (handled in WebSocket endpoint)
            pass
        result = await backtest_strategy(
            trading_symbol=request.trading_symbol,
            instrument_token=request.instrument_token,
            timeframe=request.timeframe,
            strategy=request.strategy,
            params=request.params,
            start_date=request.start_date,
            end_date=request.end_date,
            ws_callback=ws_callback,
            db=db,
            nse_db=nse_db
        )

        result = sanitize_floats(result)

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

@app.get("/strategies/status-summary", response_model=dict, tags=["algo-trading"])
async def get_strategies_with_execution_status(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get all strategies with their current execution status for consistent UI display"""
    try:
        # Get all strategies with their execution information
        result = await db.execute(
            select(Strategy, func.count(StrategyExecution.execution_id).label("active_executions"))
            .outerjoin(
                StrategyExecution, 
                and_(
                    Strategy.strategy_id == StrategyExecution.strategy_id,
                    StrategyExecution.status == "running"
                )
            )
            .filter(Strategy.user_id == user_id)
            .group_by(Strategy.strategy_id)
            .order_by(Strategy.created_at.desc())
        )
        strategies_with_counts = result.all()
        
        strategy_list = []
        for strategy, active_executions in strategies_with_counts:
            strategy_data = {
                "strategy_id": strategy.strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "broker": strategy.broker,
                "status": strategy.status,
                "active_executions": active_executions,
                "execution_status": "running" if active_executions > 0 else "stopped",
                "display_status": "Running" if active_executions > 0 else ("Ready to Execute" if strategy.status == "active" else "Draft"),
                "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
                "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None,
                "entry_conditions": strategy.entry_conditions,
                "exit_conditions": strategy.exit_conditions,
                "parameters": strategy.parameters
            }
            strategy_list.append(strategy_data)
        
        # Summary statistics
        total_strategies = len(strategy_list)
        running_strategies = sum(1 for s in strategy_list if s["active_executions"] > 0)
        ready_strategies = sum(1 for s in strategy_list if s["status"] == "active" and s["active_executions"] == 0)
        draft_strategies = sum(1 for s in strategy_list if s["status"] == "inactive")
        
        return {
            "strategies": strategy_list,
            "summary": {
                "total": total_strategies,
                "running": running_strategies,
                "ready": ready_strategies,
                "draft": draft_strategies
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching strategy status summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies/all/{broker}", response_model=List[StrategyResponse], tags=["algo-trading"])
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

@app.get("/strategies/{broker}/live", tags=["algo-trading"])
async def get_live_strategies(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get all live/active strategies for a broker"""
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_BROKER", "message": "Invalid broker"}})
    try:
        result = await db.execute(select(Strategy).filter(Strategy.user_id == user_id, Strategy.broker == broker, Strategy.status == "active"))
        strategies = result.scalars().all()
        return [StrategyResponse.from_orm(s) for s in strategies]
    except Exception as e:
        logger.error(f"Error fetching live strategies: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "FETCH_FAILED", "message": str(e)}})

@app.get("/strategies/{broker}/execution-status", tags=["algo-trading"])
async def get_strategies_with_execution_status(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get all strategies with their execution status and counts"""
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_BROKER", "message": "Invalid broker"}})
    
    try:
        # Get all strategies for the user
        strategy_result = await db.execute(
            select(Strategy).filter(
                and_(
                    Strategy.user_id == user_id, 
                    Strategy.broker == broker
                )
            )
        )
        strategies = strategy_result.scalars().all()
        
        # Get execution counts for each strategy
        execution_result = await db.execute(
            select(
                StrategyExecution.strategy_id,
                StrategyExecution.status,
                func.count(StrategyExecution.execution_id).label('count')
            ).filter(
                StrategyExecution.user_id == user_id
            ).group_by(
                StrategyExecution.strategy_id,
                StrategyExecution.status
            )
        )
        execution_counts = execution_result.fetchall()
        
        # Organize execution data by strategy
        execution_data = {}
        for row in execution_counts:
            strategy_id = row[0]
            status = row[1]
            count = row[2]
            
            if strategy_id not in execution_data:
                execution_data[strategy_id] = {
                    'running': 0,
                    'stopped': 0,
                    'completed': 0,
                    'failed': 0,
                    'total': 0
                }
            
            execution_data[strategy_id][status] = count
            execution_data[strategy_id]['total'] += count
        
        # Build response with execution status
        strategies_with_status = []
        for strategy in strategies:
            strategy_dict = {
                'strategy_id': strategy.strategy_id,
                'name': strategy.name,
                'description': strategy.description,
                'status': strategy.status,
                'entry_conditions': strategy.entry_conditions,
                'exit_conditions': strategy.exit_conditions,
                'parameters': strategy.parameters,
                'created_at': strategy.created_at.isoformat() if strategy.created_at else None,
                'updated_at': strategy.updated_at.isoformat() if strategy.updated_at else None,
                'execution_status': execution_data.get(strategy.strategy_id, {
                    'running': 0, 'stopped': 0, 'completed': 0, 'failed': 0, 'total': 0
                })
            }
            strategies_with_status.append(strategy_dict)
        
        return strategies_with_status
        
    except Exception as e:
        logger.error(f"Error fetching strategies with execution status: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "FETCH_FAILED", "message": str(e)}})

@app.get("/strategies/{broker}/statistics", tags=["algo-trading"])
async def get_strategy_statistics(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get comprehensive strategy statistics for a broker"""
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_BROKER", "message": "Invalid broker"}})
    try:
        # Get strategy counts by status
        result = await db.execute(select(Strategy.status, func.count(Strategy.strategy_id)).filter(
            Strategy.user_id == user_id, Strategy.broker == broker
        ).group_by(Strategy.status))
        status_counts = dict(result.fetchall())
        
        # Get total strategies
        total_strategies = sum(status_counts.values())
        
        # Get recent strategies
        result = await db.execute(select(Strategy).filter(
            Strategy.user_id == user_id, Strategy.broker == broker
        ).order_by(Strategy.created_at.desc()).limit(5))
        recent_strategies = result.scalars().all()
        
        return {
            "total_strategies": total_strategies,
            "status_breakdown": status_counts,
            "recent_strategies": [StrategyResponse.from_orm(s) for s in recent_strategies],
            "broker": broker,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching strategy statistics: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "STATS_FAILED", "message": str(e)}})

@app.get("/strategies/{broker}/performance", tags=["algo-trading"])
async def get_strategy_performance(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    query = """
        SELECT s.strategy_id, s.name, t.pnl, t.entry_time, t.exit_time
        FROM strategies s
        LEFT JOIN trade_history t ON s.user_id = t.user_id AND t.broker = s.broker
        WHERE s.user_id = :user_id AND s.broker = :broker AND s.status = 'active'
    """
    trades = pd.DataFrame(await async_fetch_query(db, text(query), {"user_id": user_id, "broker": broker}))
    performance = trades.groupby("strategy_id").agg({
        "pnl": ["sum", "count"],
        "entry_time": "min",
        "exit_time": "max"
    }).to_dict()
    return performance

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

@app.post("/strategies/{strategy_id}/execute", response_model=dict, tags=["strategy-execution"])
async def execute_strategy(
    strategy_id: str,
    execution_request: StrategyExecutionRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Execute a strategy on specific instruments with risk parameters"""
    try:
        logger.info(f"Executing strategy {strategy_id} for user {user_id}")
        logger.info(f"Execution request: {execution_request}")
        
        # Get user API for broker operations
        user_api = None
        if hasattr(app.state, 'user_apis') and user_id in app.state.user_apis:
            # First get the strategy to know which broker to use
            result = await db.execute(select(Strategy).filter(Strategy.strategy_id == strategy_id, Strategy.user_id == user_id))
            strategy = result.scalars().first()
            if strategy:
                broker_apis = app.state.user_apis[user_id]
                user_api = broker_apis.get(strategy.broker.lower())
        
        # Execute strategy using strategy engine
        if hasattr(app.state, 'strategy_engine'):
            execution_result = await app.state.strategy_engine.execute_strategy(
                strategy_id=strategy_id,
                user_id=user_id,
                instrument_token=execution_request.instrument_token,
                trading_symbol=execution_request.trading_symbol,
                quantity=execution_request.quantity,
                risk_per_trade=execution_request.risk_per_trade,
                stop_loss=execution_request.stop_loss,
                take_profit=execution_request.take_profit,
                position_sizing_percent=execution_request.position_sizing_percent,
                position_sizing_mode=execution_request.position_sizing_mode,
                total_capital=execution_request.total_capital,
                timeframe=execution_request.timeframe,
                trailing_stop_enabled=execution_request.trailing_stop_enabled,
                trailing_stop_percent=execution_request.trailing_stop_percent,
                trailing_stop_min=execution_request.trailing_stop_min,
                partial_exits=[{"target": pe.target, "qty_percent": pe.qty_percent} for pe in execution_request.partial_exits],
                db=db,
                api=user_api
            )
            
            logger.info(f"Execution result: {execution_result}")
            
            if not execution_result.get('success'):
                raise HTTPException(
                    status_code=400,
                    detail={"error": {"code": "EXECUTION_FAILED", "message": execution_result.get('message', 'Unknown error')}}
                )
            
            return execution_result
        else:
            raise HTTPException(status_code=500, detail="Strategy execution engine not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategies/{strategy_id}/{action}", response_model=dict, tags=["algo-trading"])
async def toggle_strategy(strategy_id: str, action: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if action not in ["activate", "deactivate"]:
        raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_ACTION", "message": "Invalid action"}})
    try:
        result = await db.execute(select(Strategy).filter(Strategy.strategy_id == strategy_id, Strategy.user_id == user_id))
        strategy = result.scalars().first()
        if not strategy:
            raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Strategy not found"}})
        
        # Get user API for broker operations
        user_api = None
        if hasattr(app.state, 'user_apis') and user_id in app.state.user_apis:
            broker_apis = app.state.user_apis[user_id]
            user_api = broker_apis.get(strategy.broker.lower())
        
        # Use strategy execution engine for real activation/deactivation
        if action == "activate":
            logger.info(f"Attempting to activate strategy {strategy_id} for user {user_id}")
            if hasattr(app.state, 'strategy_engine'):
                activation_result = await app.state.strategy_engine.activate_strategy(
                    strategy_id, user_id, db, user_api
                )
                logger.info(f"Strategy activation result: {activation_result}")
                
                if not activation_result.get('success'):
                    logger.error(f"Strategy activation failed: {activation_result.get('message')}")
                    # Don't update database status on failure
                    raise HTTPException(
                        status_code=400, 
                        detail={"error": {"code": "ACTIVATION_FAILED", "message": activation_result.get('message', 'Unknown error')}}
                    )
                else:
                    logger.info(f"Strategy {strategy_id} activated successfully")
            else:
                logger.warning("Strategy engine not available, using fallback activation")
                # Fallback: just update status
                strategy.status = "active"
        else:  # deactivate
            if hasattr(app.state, 'strategy_engine'):
                deactivation_result = await app.state.strategy_engine.deactivate_strategy(
                    strategy_id, user_id, db
                )
                if not deactivation_result.get('success'):
                    logger.warning(f"Strategy deactivation had issues: {deactivation_result.get('message')}")
            strategy.status = "inactive"
        
        await db.commit()
        
        return {
            "status": strategy.status, 
            "strategy_id": strategy_id,
            "message": f"Strategy {action}d successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "TOGGLE_FAILED", "message": str(e)}})

# New endpoints for strategy execution and risk management

@app.get("/strategies/{strategy_id}/debug", response_model=dict, tags=["algo-trading"])
async def debug_strategy(strategy_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Debug endpoint to check strategy data structure"""
    try:
        result = await db.execute(select(Strategy).filter(Strategy.strategy_id == strategy_id, Strategy.user_id == user_id))
        strategy = result.scalars().first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {
            "strategy_id": strategy_id,
            "name": strategy.name,
            "status": strategy.status,
            "description": strategy.description,
            "broker": strategy.broker,
            "parameters": strategy.parameters,
            "entry_conditions": strategy.entry_conditions,
            "exit_conditions": strategy.exit_conditions,
            "created_at": strategy.created_at.isoformat() if strategy.created_at else None
        }
    except Exception as e:
        logger.error(f"Error debugging strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies/{strategy_id}/metrics", response_model=dict, tags=["algo-trading"])
async def get_strategy_metrics(strategy_id: str, user_id: str = Depends(get_current_user)):
    """Get real-time metrics for an active strategy"""
    try:
        if hasattr(app.state, 'strategy_engine'):
            metrics = app.state.strategy_engine.get_strategy_metrics(strategy_id, user_id)
            return {
                "strategy_id": strategy_id,
                "metrics": metrics,
                "active_instruments": len(metrics)
            }
        return {"strategy_id": strategy_id, "metrics": {}, "active_instruments": 0}
    except Exception as e:
        logger.error(f"Error fetching strategy metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-strategies/{broker}", response_model=list, tags=["algo-trading"])
async def get_active_strategies(broker: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get all active strategies for a broker"""
    if broker not in ["Upstox", "Zerodha"]:
        raise HTTPException(status_code=400, detail="Invalid broker")
    try:
        result = await db.execute(
            select(Strategy).filter(
                Strategy.user_id == user_id,
                Strategy.broker == broker,
                Strategy.status == "active"
            )
        )
        strategies = result.scalars().all()
        
        return [
            {
                "strategy_id": s.strategy_id,
                "name": s.name,
                "description": s.description,
                "status": s.status,
                "broker": s.broker,
                "parameters": s.parameters,
                "created_at": s.created_at.isoformat() if s.created_at else None
            }
            for s in strategies
        ]
    except Exception as e:
        logger.error(f"Error fetching active strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk-management/metrics", response_model=dict, tags=["risk-management"])
async def get_risk_metrics(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get comprehensive risk management metrics for the user"""
    try:
        # Check cache first
        cache_key = cache_manager.generate_cache_key("risk_metrics", user_id)
        cached_metrics = await cache_manager.get(cache_key)
        if cached_metrics:
            return cached_metrics
        
        if hasattr(app.state, 'risk_manager'):
            risk_metrics = await app.state.risk_manager.get_risk_metrics(user_id, db)
            
            # Cache for 2 minutes (risk metrics are calculated values that don't change too frequently)
            await cache_manager.set(cache_key, risk_metrics, CacheConfig.RISK_METRICS)
            
            return risk_metrics
        return {}
    except Exception as e:
        logger.error(f"Error fetching risk metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk-management/emergency-stop", response_model=dict, tags=["risk-management"])
async def activate_emergency_stop(reason: str = "Manual activation via API", user_id: str = Depends(get_current_user)):
    """Activate emergency stop - halts all trading immediately"""
    try:
        if hasattr(app.state, 'risk_manager'):
            success = app.state.risk_manager.activate_emergency_stop(f"{reason} (by user {user_id})")
            if success:
                logger.critical(f"Emergency stop activated by user {user_id}: {reason}")
                return {"status": "success", "message": "Emergency stop activated", "active": True}
            else:
                raise HTTPException(status_code=500, detail="Failed to activate emergency stop")
        else:
            raise HTTPException(status_code=500, detail="Risk manager not available")
    except Exception as e:
        logger.error(f"Error activating emergency stop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk-management/emergency-stop/deactivate", response_model=dict, tags=["risk-management"])
async def deactivate_emergency_stop(user_id: str = Depends(get_current_user)):
    """Deactivate emergency stop - requires proper authorization"""
    try:
        if hasattr(app.state, 'risk_manager'):
            success = app.state.risk_manager.deactivate_emergency_stop(f"User {user_id}")
            if success:
                logger.warning(f"Emergency stop deactivated by user {user_id}")
                return {"status": "success", "message": "Emergency stop deactivated", "active": False}
            else:
                raise HTTPException(status_code=500, detail="Failed to deactivate emergency stop")
        else:
            raise HTTPException(status_code=500, detail="Risk manager not available")
    except Exception as e:
        logger.error(f"Error deactivating emergency stop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk-management/system-health", response_model=dict, tags=["risk-management"])
async def get_system_health(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get system health status and circuit breaker status"""
    try:
        if hasattr(app.state, 'risk_manager'):
            healthy, reason = await app.state.risk_manager.check_system_health(db)
            market_open, market_reason = app.state.risk_manager.is_market_hours()
            
            return {
                "system_healthy": healthy,
                "health_reason": reason,
                "circuit_breaker_active": app.state.risk_manager.circuit_breaker_triggered,
                "emergency_stop_active": app.state.risk_manager.emergency_stop_active,
                "market_open": market_open,
                "market_status": market_reason,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Risk manager not available")
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/preferences", response_model=dict, tags=["user"])
async def get_user_preferences(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get user preferences including risk management settings"""
    try:
        # Check cache first
        cache_key = cache_manager.generate_cache_key("user_preferences", user_id)
        cached_preferences = await cache_manager.get(cache_key)
        if cached_preferences:
            return cached_preferences
        
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        preferences_result = {
            "status": "success",
            "preferences": user.preferences or {}
        }
        
        # Cache preferences for 30 minutes
        await cache_manager.set(cache_key, preferences_result, CacheConfig.USER_PREFERENCES)
        
        return preferences_result
    except Exception as e:
        logger.error(f"Error fetching user preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def validate_risk_preferences(preferences: dict) -> tuple[bool, str]:
    """
    Validate user preference values to prevent financial risks
    Returns: (is_valid, error_message)
    """
    try:
        # Daily loss limit validation
        if 'daily_loss_limit' in preferences:
            daily_loss_limit = float(preferences['daily_loss_limit'])
            if daily_loss_limit < 1000:
                return False, "Daily loss limit must be at least ₹1,000"
            if daily_loss_limit > 10000000:  # 1 crore limit
                return False, "Daily loss limit cannot exceed ₹1,00,00,000"
        
        # Position size limit validation  
        if 'position_size_limit' in preferences:
            position_size_limit = float(preferences['position_size_limit'])
            if position_size_limit < 5000:
                return False, "Position size limit must be at least ₹5,000"
            if position_size_limit > 50000000:  # 5 crore limit
                return False, "Position size limit cannot exceed ₹5,00,00,000"
        
        # Max open positions validation
        if 'max_open_positions' in preferences:
            max_open_positions = int(preferences['max_open_positions'])
            if max_open_positions < 1:
                return False, "Maximum open positions must be at least 1"
            if max_open_positions > 100:
                return False, "Maximum open positions cannot exceed 100"
        
        # Risk per trade validation
        if 'risk_per_trade' in preferences:
            risk_per_trade = float(preferences['risk_per_trade'])
            if risk_per_trade < 0.1:
                return False, "Risk per trade must be at least 0.1%"
            if risk_per_trade > 25.0:
                return False, "Risk per trade cannot exceed 25%"
        
        # Max portfolio risk validation
        if 'max_portfolio_risk' in preferences:
            max_portfolio_risk = float(preferences['max_portfolio_risk'])
            if max_portfolio_risk < 1.0:
                return False, "Maximum portfolio risk must be at least 1%"
            if max_portfolio_risk > 100.0:
                return False, "Maximum portfolio risk cannot exceed 100%"
                
        # Auto-stop trading validation (must be boolean and cannot be disabled if loss limit exceeded)
        if 'auto_stop_trading' in preferences:
            if not isinstance(preferences['auto_stop_trading'], bool):
                return False, "Auto-stop trading must be true or false"
        
        # Validate order type preferences
        if 'default_order_type' in preferences:
            valid_order_types = ['MARKET', 'LIMIT', 'SL', 'SL-M']
            if preferences['default_order_type'] not in valid_order_types:
                return False, f"Default order type must be one of: {', '.join(valid_order_types)}"
        
        # Validate product type preferences  
        if 'default_product_type' in preferences:
            valid_product_types = ['CNC', 'MIS', 'NRML', 'D']
            if preferences['default_product_type'] not in valid_product_types:
                return False, f"Default product type must be one of: {', '.join(valid_product_types)}"
        
        return True, "Validation passed"
        
    except (ValueError, TypeError) as e:
        return False, f"Invalid preference value format: {str(e)}"
    except Exception as e:
        return False, f"Preference validation error: {str(e)}"

@app.post("/user/preferences", response_model=dict, tags=["user"])
async def update_user_preferences(preferences: dict, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Update user preferences including risk management settings with validation"""
    try:
        # CRITICAL SECURITY FIX: Validate all preference updates
        is_valid, validation_message = validate_risk_preferences(preferences)
        if not is_valid:
            logger.warning(f"Invalid preference update attempted by user {user_id}: {validation_message}")
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": {
                        "code": "INVALID_PREFERENCES", 
                        "message": validation_message
                    }
                }
            )
        
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # If user currently has daily loss exceeded, prevent disabling auto-stop
        if 'auto_stop_trading' in preferences and not preferences['auto_stop_trading']:
            if hasattr(app.state, 'risk_manager'):
                try:
                    risk_settings = await app.state.risk_manager.get_user_risk_settings(user_id, db)
                    daily_pnl = await app.state.risk_manager.get_daily_pnl(user_id, db)
                    if daily_pnl <= -risk_settings['daily_loss_limit']:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": {
                                    "code": "CANNOT_DISABLE_AUTO_STOP",
                                    "message": "Cannot disable auto-stop trading when daily loss limit is exceeded"
                                }
                            }
                        )
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error checking risk status for preference update: {e}")
        
        # Merge with existing preferences
        current_prefs = user.preferences or {}
        current_prefs.update(preferences)
        user.preferences = current_prefs
        
        await db.commit()
        
        # Invalidate cached preferences and related data after successful update
        await cache_manager.delete_pattern(f"user_preferences:*{user_id}*")
        await cache_manager.delete_pattern(f"risk_metrics:*{user_id}*")  # Risk metrics depend on preferences
        
        logger.info(f"Preferences updated successfully for user {user_id}: {list(preferences.keys())}")
        
        return {
            "status": "success",
            "message": "Preferences updated successfully",
            "updated_preferences": list(preferences.keys())
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error updating user preferences for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")

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

@app.put("/strategies/{strategy_id}", response_model=StrategyResponse, tags=["algo-trading"])
async def update_strategy(strategy_id: str, strategy_update: StrategyRequest, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Update an existing strategy"""
    try:
        result = await db.execute(select(Strategy).filter(Strategy.strategy_id == strategy_id, Strategy.user_id == user_id))
        strategy = result.scalars().first()
        if not strategy:
            raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Strategy not found"}})
        
        # Update strategy fields
        strategy.name = strategy_update.name
        strategy.description = strategy_update.description
        strategy.entry_conditions = strategy_update.entry_conditions
        strategy.exit_conditions = strategy_update.exit_conditions
        strategy.parameters = strategy_update.parameters
        strategy.broker = strategy_update.broker
        strategy.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(strategy)
        
        logger.info(f"Strategy {strategy_id} updated successfully by user {user_id}")
        return StrategyResponse.from_orm(strategy)
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "UPDATE_FAILED", "message": str(e)}})

@app.post("/strategies/{strategy_id}/duplicate", response_model=StrategyResponse, tags=["algo-trading"])
async def duplicate_strategy(strategy_id: str, user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Duplicate an existing strategy"""
    try:
        result = await db.execute(select(Strategy).filter(Strategy.strategy_id == strategy_id, Strategy.user_id == user_id))
        original_strategy = result.scalars().first()
        if not original_strategy:
            raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Strategy not found"}})
        
        # Create duplicate strategy
        new_strategy = Strategy(
            strategy_id=str(uuid4()),
            user_id=user_id,
            broker=original_strategy.broker,
            name=f"{original_strategy.name} (Copy)",
            description=original_strategy.description,
            entry_conditions=original_strategy.entry_conditions,
            exit_conditions=original_strategy.exit_conditions,
            parameters=original_strategy.parameters,
            status="inactive"  # Start as inactive
        )
        
        db.add(new_strategy)
        await db.commit()
        await db.refresh(new_strategy)
        
        logger.info(f"Strategy {strategy_id} duplicated successfully by user {user_id}")
        return StrategyResponse.from_orm(new_strategy)
    except Exception as e:
        logger.error(f"Error duplicating strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": {"code": "DUPLICATE_FAILED", "message": str(e)}})

# ============================================================================
# EMERGENCY CONTROL ENDPOINTS
# ============================================================================

@app.post("/executions/emergency-stop", tags=["emergency"])
async def emergency_stop_all_executions(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Emergency stop all running strategy executions"""
    try:
        logger.warning(f"EMERGENCY STOP activated by user {user_id}")

        # Get all running executions for this user
        result = await db.execute(
            select(StrategyExecution).filter(
                StrategyExecution.user_id == user_id,
                StrategyExecution.status == 'running'
            )
        )
        running_executions = result.scalars().all()

        stopped_count = 0
        for execution in running_executions:
            try:
                # Update execution status to stopped
                execution.status = 'emergency_stopped'
                execution.stopped_at = datetime.now()
                stopped_count += 1

                # Here you would also call the strategy engine to actually stop the execution
                if hasattr(app.state, 'strategy_engine'):
                    await app.state.strategy_engine.emergency_stop_execution(execution.execution_id)

            except Exception as e:
                logger.error(f"Error stopping execution {execution.execution_id}: {e}")

        await db.commit()

        return {
            "status": "success",
            "message": f"Emergency stop activated - {stopped_count} executions stopped",
            "executions_stopped": stopped_count
        }

    except Exception as e:
        logger.error(f"Emergency stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategies/pause-all", tags=["strategy-control"])
async def pause_all_strategies(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Pause all active strategies for the user"""
    try:
        logger.info(f"Pausing all strategies for user {user_id}")

        # Get all active strategies for this user
        result = await db.execute(
            select(Strategy).filter(
                Strategy.user_id == user_id,
                Strategy.status == 'active'
            )
        )
        active_strategies = result.scalars().all()

        paused_count = 0
        for strategy in active_strategies:
            try:
                strategy.status = 'paused'
                strategy.updated_at = datetime.now()
                paused_count += 1

                # Also pause any running executions
                exec_result = await db.execute(
                    select(StrategyExecution).filter(
                        StrategyExecution.strategy_id == strategy.strategy_id,
                        StrategyExecution.status == 'running'
                    )
                )
                running_executions = exec_result.scalars().all()

                for execution in running_executions:
                    execution.status = 'paused'
                    if hasattr(app.state, 'strategy_engine'):
                        await app.state.strategy_engine.pause_execution(execution.execution_id)

            except Exception as e:
                logger.error(f"Error pausing strategy {strategy.strategy_id}: {e}")

        await db.commit()

        return {
            "status": "success",
            "message": f"Paused {paused_count} strategies",
            "strategies_paused": paused_count
        }

    except Exception as e:
        logger.error(f"Pause all strategies error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategies/start-all", tags=["strategy-control"])
async def start_all_strategies(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Start/resume all paused strategies for the user"""
    try:
        logger.info(f"Starting all strategies for user {user_id}")

        # Get all paused strategies for this user
        result = await db.execute(
            select(Strategy).filter(
                Strategy.user_id == user_id,
                Strategy.status == 'paused'
            )
        )
        paused_strategies = result.scalars().all()

        started_count = 0
        for strategy in paused_strategies:
            try:
                strategy.status = 'active'
                strategy.updated_at = datetime.now()
                started_count += 1

                # Also resume any paused executions
                exec_result = await db.execute(
                    select(StrategyExecution).filter(
                        StrategyExecution.strategy_id == strategy.strategy_id,
                        StrategyExecution.status == 'paused'
                    )
                )
                paused_executions = exec_result.scalars().all()

                for execution in paused_executions:
                    execution.status = 'running'
                    if hasattr(app.state, 'strategy_engine'):
                        await app.state.strategy_engine.resume_execution(execution.execution_id)

            except Exception as e:
                logger.error(f"Error starting strategy {strategy.strategy_id}: {e}")

        await db.commit()

        return {
            "status": "success",
            "message": f"Started {started_count} strategies",
            "strategies_started": started_count
        }

    except Exception as e:
        logger.error(f"Start all strategies error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/executions/active-strategies", tags=["executions"])
async def get_active_strategy_summary(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get summary of active strategies (one record per strategy-symbol combination)"""
    try:
        result = await db.execute(
            select(StrategyExecution, Strategy.name)
            .join(Strategy, StrategyExecution.strategy_id == Strategy.strategy_id, isouter=True)
            .filter(
                StrategyExecution.user_id == user_id,
                StrategyExecution.status == "running"
            )
            .order_by(StrategyExecution.started_at.desc())
        )
        execution_data_list = result.all()

        # Group by strategy_id + trading_symbol to avoid duplicates
        strategy_summary = {}
        for execution, strategy_name in execution_data_list:
            key = f"{execution.strategy_id}_{execution.trading_symbol}"
            
            if key not in strategy_summary:
                strategy_summary[key] = {
                    "execution_id": execution.execution_id,
                    "strategy_id": execution.strategy_id,
                    "strategy_name": strategy_name or "Unknown",
                    "trading_symbol": execution.trading_symbol,
                    "status": execution.status,
                    "total_pnl": float(execution.pnl or 0),
                    "total_trades": execution.trades_executed or 0,
                    "total_signals": execution.signals_generated or 0,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "last_signal_at": execution.last_signal_at.isoformat() if execution.last_signal_at else None,
                    "execution_count": 1,
                    "latest_execution": execution.execution_id
                }
            else:
                # Aggregate metrics from multiple executions
                summary = strategy_summary[key]
                summary["total_pnl"] += float(execution.pnl or 0)
                summary["total_trades"] += (execution.trades_executed or 0)
                summary["total_signals"] += (execution.signals_generated or 0)
                summary["execution_count"] += 1
                
                # Keep the most recent execution details
                if execution.started_at and (not summary["started_at"] or execution.started_at.isoformat() > summary["started_at"]):
                    summary["latest_execution"] = execution.execution_id
                    summary["last_signal_at"] = execution.last_signal_at.isoformat() if execution.last_signal_at else None

        return {
            "active_strategies": list(strategy_summary.values()),
            "total_active": len(strategy_summary)
        }
        
    except Exception as e:
        logger.error(f"Error fetching active strategy summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/executions", tags=["executions"])
async def get_all_executions(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get all strategy executions for the user"""
    try:
        result = await db.execute(
            select(StrategyExecution, Strategy.name)
            .join(Strategy, StrategyExecution.strategy_id == Strategy.strategy_id, isouter=True)
            .filter(StrategyExecution.user_id == user_id)
            .order_by(StrategyExecution.started_at.desc())
        )
        execution_data_list = result.all()

        execution_list = []
        for execution, strategy_name in execution_data_list:
            execution_data = {
                "execution_id": execution.execution_id,
                "strategy_id": execution.strategy_id,
                "strategy_name": strategy_name or "Unknown",
                "trading_symbol": execution.trading_symbol,
                "status": execution.status,
                "pnl": float(execution.pnl or 0),
                "trades_executed": execution.trades_executed or 0,
                "signals_generated": execution.signals_generated or 0,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "last_signal_at": execution.last_signal_at.isoformat() if execution.last_signal_at else None,
                "stopped_at": execution.stopped_at.isoformat() if execution.stopped_at else None
            }
            execution_list.append(execution_data)

        return execution_list

    except Exception as e:
        logger.error(f"Error fetching executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/executions/{execution_id}/stop", tags=["executions"])
async def stop_execution(
    execution_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Stop a specific strategy execution"""
    try:
        result = await db.execute(
            select(StrategyExecution).filter(
                StrategyExecution.execution_id == execution_id,
                StrategyExecution.user_id == user_id
            )
        )
        execution = result.scalar_one_or_none()

        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")

        if execution.status == 'running':
            execution.status = 'stopped'
            execution.stopped_at = datetime.now()

            # Call strategy engine to stop execution
            if hasattr(app.state, 'strategy_engine'):
                await app.state.strategy_engine.stop_strategy_execution(execution_id, user_id, db)

            await db.commit()

            return {
                "status": "success",
                "message": f"Execution {execution_id} stopped successfully"
            }
        else:
            return {
                "status": "info",
                "message": f"Execution is already {execution.status}"
            }

    except Exception as e:
        logger.error(f"Error stopping execution {execution_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/executions/{execution_id}/metrics", tags=["executions"])
async def get_execution_metrics(
    execution_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed metrics for a specific execution"""
    try:
        result = await db.execute(
            select(StrategyExecution, Strategy.name)
            .join(Strategy, StrategyExecution.strategy_id == Strategy.strategy_id, isouter=True)
            .filter(
                StrategyExecution.execution_id == execution_id,
                StrategyExecution.user_id == user_id
            )
        )
        result_data = result.first()

        if not result_data:
            raise HTTPException(status_code=404, detail="Execution not found")

        execution, strategy_name = result_data
        metrics = {
            "execution_id": execution.execution_id,
            "strategy_name": strategy_name or "Unknown",
            "trading_symbol": execution.trading_symbol,
            "status": execution.status,
            "pnl": float(execution.pnl or 0),
            "trades_executed": execution.trades_executed or 0,
            "signals_generated": execution.signals_generated or 0,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "last_signal_at": execution.last_signal_at.isoformat() if execution.last_signal_at else None,
            "stopped_at": execution.stopped_at.isoformat() if execution.stopped_at else None,
            "risk_parameters": {
                "risk_per_trade": execution.risk_per_trade or 2.0,
                "stop_loss": execution.stop_loss,
                "take_profit": execution.take_profit,
                "quantity": execution.quantity
            },
            "performance_metrics": {
                "total_trades": execution.trades_executed or 0,
                "total_signals": execution.signals_generated or 0,
                "total_pnl": float(execution.pnl or 0),
                "success_rate": 0  # Calculate based on actual trade data
            }
        }

        return {"metrics": metrics}

    except Exception as e:
        logger.error(f"Error fetching execution metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies/{broker}/list", tags=["strategies"])
async def list_strategies_for_execution(
    broker: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get list of strategies available for execution (all strategies, not just active)"""
    try:
        result = await db.execute(
            select(Strategy).filter(
                Strategy.user_id == user_id,
                Strategy.broker.ilike(broker)  # Case-insensitive broker matching
            ).order_by(Strategy.created_at.desc())
        )
        strategies = result.scalars().all()

        strategy_list = []
        for strategy in strategies:
            strategy_data = {
                "strategy_id": strategy.strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "status": strategy.status,
                "broker": strategy.broker,
                "entry_conditions": strategy.entry_conditions or [],
                "exit_conditions": strategy.exit_conditions or [],
                "parameters": strategy.parameters or {},
                "created_at": strategy.created_at.isoformat() if strategy.created_at else None
            }
            strategy_list.append(strategy_data)

        return strategy_list

    except Exception as e:
        logger.error(f"Error fetching strategies list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add WebSocket for backtest progress
@app.websocket("/ws/backtest/{user_id}")
async def websocket_backtest(websocket: WebSocket, user_id: str):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_json()
                logger.debug(f"WebSocket received data for user {user_id}: {data}")
                await websocket.send_json(data)
            except aiohttp.WSServerHandshakeError as e:
                logger.warning(f"WebSocket handshake error for user {user_id}: {str(e)}")
                break
            except aiohttp.WebSocketError as e:
                logger.error(f"WebSocket error for user {user_id}: {str(e)}")
                break
    except asyncio.CancelledError:
        # Graceful shutdown
        pass
    except Exception as e:
        logger.error(f"Unexpected WebSocket error for user {user_id}: {str(e)}")
    finally:
        try:
            if websocket.closed:
                logger.info(f"WebSocket already closed for user {user_id}")
            else:
                await websocket.close(code=1000, reason="Normal closure")
                logger.info(f"WebSocket closed for user {user_id}")
        except Exception as e:
            logger.error(f"Error closing WebSocket for user {user_id}: {str(e)}")

# Orders WebSocket (basic accept to avoid 403 and allow future push updates)
@app.websocket("/ws/orders/{user_id}")
async def websocket_orders(websocket: WebSocket, user_id: str):
    await websocket.accept()
    await ws_register_client(user_id, websocket)
    logger.info(f"Orders WebSocket connected for user {user_id}")
    try:
        while True:
            await asyncio.sleep(30)
    except asyncio.CancelledError:
        # Graceful shutdown without error logging
        pass
    except Exception as e:
        logger.warning(f"Orders WebSocket closed for user {user_id}: {e}")
    finally:
        try:
            await ws_unregister_client(user_id, websocket)
            if not websocket.closed:
                await websocket.close()
        except Exception:
            pass

# Test endpoint for main.py database access
@app.get("/debug/test-main-db")
async def test_main_database_simple(db: AsyncSession = Depends(get_db)):
    """Simple test of main.py database access"""
    try:
        result = await db.execute(text("SELECT 1 as test"))
        test_result = result.fetchone()

        return {
            "success": True,
            "test_result": test_result[0] if test_result else None,
            "message": "Main.py database access working",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Main.py database test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/test-nsedata-db")
async def test_main_database_simple(db: AsyncSession = Depends(get_nsedata_db)):
    """Simple test of main.py database access"""
    try:
        result = await db.execute(text("SELECT 1 as test"))
        test_result = result.fetchone()

        return {
            "success": True,
            "test_result": test_result[0] if test_result else None,
            "message": "Main.py database access working",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Main.py database test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    try:
        metrics = {
            "active_users": len(user_apis),
            "database_connections": len([
                db for db, status in {
                    db_name: "connected" if db_manager.get_session_factory(db_name) else "disconnected"
                    for db_name in db_manager.db_configs.keys()
                }.items() if status == "connected"
            ]),
            "timestamp": datetime.now().isoformat()
        }

        # Add database-specific metrics
        for db_name in db_manager.db_configs.keys():
            session_factory = db_manager.get_session_factory(db_name)
            if session_factory:
                try:
                    async with session_factory() as session:
                        if db_name == "nsedata":
                            # Get symbol count
                            result = await session.execute(text("""
                                SELECT COUNT(*) FROM pg_tables 
                                WHERE schemaname = 'public' AND tablename ~ '^[A-Z]+$'
                            """))
                            metrics[f"{db_name}_symbols"] = result.scalar()
                        elif db_name == "trading_db":
                            # Get user count
                            result = await session.execute(text("SELECT COUNT(*) FROM users"))
                            metrics[f"{db_name}_users"] = result.scalar()
                except Exception:
                    pass

        return metrics

    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STRATEGY EXECUTION ENDPOINTS  
# ============================================================================

@app.post("/executions/cleanup", response_model=dict, tags=["executions"])
async def cleanup_old_executions(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Clean up old/duplicate strategy executions"""
    try:
        # Get all executions grouped by strategy and symbol
        result = await db.execute(
            select(StrategyExecution)
            .filter(StrategyExecution.user_id == user_id)
            .order_by(StrategyExecution.started_at.desc())
        )
        all_executions = result.scalars().all()
        
        # Group by strategy_id + trading_symbol
        execution_groups = {}
        for execution in all_executions:
            key = f"{execution.strategy_id}_{execution.trading_symbol}"
            if key not in execution_groups:
                execution_groups[key] = []
            execution_groups[key].append(execution)
        
        cleaned_count = 0
        for group_key, executions in execution_groups.items():
            if len(executions) > 1:
                # Keep the most recent running execution, mark others as stopped
                running_executions = [e for e in executions if e.status == "running"]
                if len(running_executions) > 1:
                    # Keep the most recent, stop others
                    running_executions.sort(key=lambda x: x.started_at, reverse=True)
                    for execution in running_executions[1:]:  # Skip the first (most recent)
                        execution.status = "stopped"
                        execution.stopped_at = datetime.now()
                        cleaned_count += 1
        
        await db.commit()
        
        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} duplicate executions",
            "cleaned_count": cleaned_count
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up executions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/executions/{execution_id}/stop", response_model=dict, tags=["strategy-execution"])
async def stop_strategy_execution(
    execution_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Stop a running strategy execution"""
    try:
        if hasattr(app.state, 'strategy_engine'):
            stop_result = await app.state.strategy_engine.stop_strategy_execution(execution_id, user_id, db)
            
            if not stop_result.get('success'):
                raise HTTPException(
                    status_code=400,
                    detail={"error": {"code": "STOP_FAILED", "message": stop_result.get('message', 'Unknown error')}}
                )
            
            return stop_result
        else:
            raise HTTPException(status_code=500, detail="Strategy execution engine not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping strategy execution {execution_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add WebSocket for backtest progress
@app.websocket("/ws/backtest/{user_id}")
async def websocket_backtest(websocket: WebSocket, user_id: str):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_json()
                logger.debug(f"WebSocket received data for user {user_id}: {data}")
                await websocket.send_json(data)
            except aiohttp.WSServerHandshakeError as e:
                logger.warning(f"WebSocket handshake error for user {user_id}: {str(e)}")
                break
            except aiohttp.WebSocketError as e:
                logger.error(f"WebSocket error for user {user_id}: {str(e)}")
                break
    except asyncio.CancelledError:
        # Graceful shutdown
        pass
    except Exception as e:
        logger.error(f"Unexpected WebSocket error for user {user_id}: {str(e)}")
    finally:
        try:
            if websocket.closed:
                logger.info(f"WebSocket already closed for user {user_id}")
            else:
                await websocket.close(code=1000, reason="Normal closure")
                logger.info(f"WebSocket closed for user {user_id}")
        except Exception as e:
            logger.error(f"Error closing WebSocket for user {user_id}: {str(e)}")


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import asyncio

    async def debug_execute_strategy_endpoint(db):
        """Debug function to test execute_strategy endpoint standalone"""
        try:
            # Initialize app state similar to startup
            if not hasattr(app.state, 'strategy_engine'):
                app.state.strategy_engine = StrategyExecutionEngine()
            if not hasattr(app.state, 'risk_manager'):
                app.state.risk_manager = RiskManager()
            if not hasattr(app.state, 'user_apis'):
                app.state.user_apis = {}

            # Create execution request object
            execution_data = {
                "instrument_token": "IDEA",
                "trading_symbol": "NSE_EQ|INE669E01016",
                "quantity": 100,
                "risk_per_trade": 2.0,
                "stop_loss": 2.0,
                "take_profit": 5.0,
                "position_sizing_percent": 10.0,
                "position_sizing_mode": "Manual Quantity",
                "total_capital": 100000.0,
                "timeframe": "day",
                "trailing_stop_enabled": False,
                "trailing_stop_percent": None,
                "trailing_stop_min": None,
                "partial_exits": []
            }

            # Convert to StrategyExecutionRequest object
            execution_request = StrategyExecutionRequest(**execution_data)

            # Debug parameters
            strategy_id = "51da78a4-3ef3-4f84-a490-90bba5f8d86f"
            user_id = "e4269837-0ccd-484f-af70-a5dfa2abe230"

            print(f"🔍 Debug: Testing execute_strategy endpoint")
            print(f"Strategy ID: {strategy_id}")
            print(f"User ID: {user_id}")
            print(f"Execution Data: {execution_data}")

            # Call the actual execute_strategy function
            result = await execute_strategy(
                strategy_id=strategy_id,
                execution_request=execution_request,
                user_id=user_id,
                db=db
            )

            print(f"✅ Success: {result}")
            return result

        except Exception as e:
            print(f"❌ Error in debug_execute_strategy_endpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    async def main():
        # from backend.app.routes.sip_routes import run_sip_backtest, SIPBacktestRequest, SIPConfigRequest
        # from fastapi import BackgroundTasks
        from backend.app.database import get_db, get_nsedata_db
        from backend.app.routes.data import fetch_table_data
        from backend.app.routes.sip_routes import get_sip_portfolios, get_portfolio_performance

        #
        # request_config = SIPBacktestRequest(
        #     symbols=["GOLDBEES", "ITBEES"],
        #     start_date="2020-01-01",
        #     end_date="2025-06-30",
        #     config=SIPConfigRequest(
        #         fixed_investment=3000,
        #         max_amount_in_a_month=6000,
        #         major_drawdown_threshold=-10,
        #         minor_drawdown_threshold=-4,
        #         extreme_drawdown_threshold=-15,
        #         major_drawdown_inv_multiplier=3,
        #         minor_drawdown_inv_multiplier=2,
        #         extreme_drawdown_inv_multiplier=4,
        #         rolling_window=100,
        #         fallback_day=28,
        #         min_investment_gap_days=7,
        #         price_reduction_threshold=5,
        #         force_remaining_investment=True
        #     )
        # )
        #
        # background_tasks = BackgroundTasks()
        db_gen = get_db()
        db = await db_gen.__anext__()
        #
        nsedata_db_gen = get_nsedata_db()
        nsedata_db = await nsedata_db_gen.__anext__()

        # await run_sip_backtest(request_config, background_tasks, trading_db=db, nsedata_db=nsedata_db, enable_monthly_limits=False)
        # print(await get_market_quotes(broker='Upstox', instruments="NSE_EQ|INE669E01016,NSE_EQ|INE002A01018", user_id="4fbba468-6a86-4516-8236-2f8abcbfd2ef", db=db))
        # await get_quote(symbol='RELIANCE', source='nsepython', fallback_sources=['nsetools', 'openchart'],
        #                         user_id="4fbba468-6a86-4516-8236-2f8abcbfd2ef", db=db)
        # await get_historical_data(upstox_api=None, upstox_access_token='None', instrument='RELIANCE', from_date='2023-01-01', to_date='2023-10-01', unit='days', interval='1', db=nsedata_db)
        print(await get_ohlc_data(broker='Zerodha', instruments='NSE_EQ|INE002A01018,NSE_EQ|INE669E01016,NSE_EQ|INE176A01028', user_id="4fbba468-6a86-4516-8236-2f8abcbfd2ef", db=db))
        # print(await get_ltp_openchart(instruments=['NSE_EQ|INE002A01018', 'NSE_EQ|INE669E01016'], db=db))
        # print(await get_ltp_from_nse(instruments=['NSE_EQ|INE002A01018', 'NSE_EQ|INE669E01016'], db=db))
        # print(await get_ohlc_openchart(instruments=['NSE_EQ|INE002A01018', 'NSE_EQ|INE669E01016'], db=db))
        # print(await get_ltp(upstox_api=None, kite_api=None, instruments=['GOLDBEES', 'SBIN'], db=db))
        # print(await get_ohlc_from_nse(instruments=['NSE_EQ|INE002A01018', 'NSE_EQ|INE669E01016'], db=db))
        # print(await get_quotes_from_nse(instruments=['NSE_EQ|INE002A01018', 'NSE_EQ|INE669E01016'], db=db))
        # Call place_new_order with appropriate parameters
        # order_payload = PlaceOrderRequest(
        #     trading_symbol='IDEA',
        #     instrument_token="NSE_EQ|INE002A01018",
        #     transaction_type="BUY",
        #     quantity=1,
        #     order_type="LIMIT",
        #     product_type="CNC",
        #     validity="DAY",
        #     price=6.58,
        #     trigger_price=0.0,
        #     stop_loss=6.54,
        #     target=6.75,
        #     is_trailing_stop_loss=True,
        #     is_amo=True,
        #     trailing_stop_loss_percent=2,
        #     trail_start_target_percent=3,
        #     broker="Zerodha"
        # )
        # order_status = await place_new_order(
        #     order=order_payload,
        #     user_id="e4269837-0ccd-484f-af70-a5dfa2abe230",
        #     db=db
        # )
        # print(f"Order Status: {order_status}")
        # await get_ohlc()
        # Debug execute_strategy endpoint
        # await debug_execute_strategy_endpoint(db)
        # print(await get_portfolio_performance(portfolio_id="pf_e4269837-0ccd-484f-af70-a5dfa2abe230_GOLDBEES_1755439421", user_id="e4269837-0ccd-484f-af70-a5dfa2abe230", trading_db=db))
        # print(await fetch_table_data(table_name='ALL_STOCKS', filters="\"STK_INDEX\" LIKE '%NIFTY%'", required_db='nsedata', trading_db=db, nsedata_db=nsedata_db))
    asyncio.run(main())

