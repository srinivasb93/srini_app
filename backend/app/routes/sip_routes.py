"""
Complete SIP Strategy API Routes with fixed data fetching and all missing endpoints
Addresses: Correct table name handling, proper database queries, missing endpoints
"""
import os

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, desc
from typing import List, Dict, Optional, Any, Union, Literal
from datetime import datetime, timedelta, date
from pydantic import BaseModel, validator
import json
import logging
import uuid
import numpy as np
from enum import Enum
import pandas as pd
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore
import pytz
import traceback
from backend.app.database import db_manager
from backend.app.services import OrderManager


# Clean imports - multi-database architecture
from backend.app.database import get_db, get_nsedata_db
from backend.app.auth import UserManager, oauth2_scheme
from backend.app.strategies.enhanced_sip_strategy import (
    EnhancedSIPStrategy, SIPConfig, Trade, EnhancedSIPStrategyWithLimits, BenchmarkSIPCalculator)

logger = logging.getLogger(__name__)

# Enhanced Scheduler Configuration
def create_enhanced_scheduler():
    """Create scheduler with production-ready configuration"""

    # Job stores
    jobstores = {
        'default': MemoryJobStore()
    }

    # Executors with proper thread/process management
    executors = {
        'default': AsyncIOExecutor(),
    }

    # Job defaults
    job_defaults = {
        'coalesce': True,  # Combine multiple pending executions into one
        'max_instances': 1,  # Only one instance of each job at a time
        'misfire_grace_time': 300  # 5 minutes grace period for missed jobs
    }

    scheduler = AsyncIOScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=pytz.timezone('Asia/Kolkata')  # IST timezone
    )

    return scheduler

# Create global scheduler instance
scheduler = create_enhanced_scheduler()

class SchedulerStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"

class JobTriggerType(str, Enum):
    CRON = "cron"
    INTERVAL = "interval"
    DATE = "date"

# Pydantic models for request/response
class SchedulerConfigRequest(BaseModel):
    job_id: str
    trigger_type: JobTriggerType
    # For CRON triggers
    cron_expression: Optional[str] = None
    # For interval triggers
    interval_seconds: Optional[int] = None
    interval_minutes: Optional[int] = None
    interval_hours: Optional[int] = None
    # For date triggers
    run_date: Optional[datetime] = None
    # Common job settings
    timezone: str = "Asia/Kolkata"
    max_instances: int = 1
    coalesce: bool = True
    misfire_grace_time: int = 300
    replace_existing: bool = True

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    next_run_time: Optional[datetime]
    trigger_info: Dict[str, Any]
    max_instances: int
    pending_instances: int

class SchedulerStatusResponse(BaseModel):
    scheduler_running: bool
    total_jobs: int
    active_jobs: List[JobStatusResponse]
    timezone: str
    uptime_seconds: float


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    user_id = UserManager.verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    return obj

def parse_date_string(date_str: str) -> date:
    """Convert date string to datetime.date object"""
    try:
        if isinstance(date_str, str):
            # Parse various date formats
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d']:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            # If no format works, try parsing as ISO format
            return datetime.fromisoformat(date_str).date()
        elif isinstance(date_str, datetime):
            return date_str.date()
        elif isinstance(date_str, date):
            return date_str
        else:
            raise ValueError(f"Cannot parse date: {date_str}")
    except Exception as e:
        logger.error(f"Error parsing date '{date_str}': {e}")
        raise ValueError(f"Invalid date format: {date_str}")

def safe_json_parse(data, field_name: str = "data", default=None):
    """Safely parse JSON data that might already be parsed"""
    try:
        if data is None:
            return default or ([] if field_name in ['symbols', 'list'] else {})
        elif isinstance(data, str):
            if data.strip() == "":
                return default or ([] if field_name in ['symbols', 'list'] else {})
            return json.loads(data)
        elif isinstance(data, (list, dict)):
            return data  # Already parsed
        else:
            logger.warning(f"Unexpected {field_name} type: {type(data)}")
            return default or ([] if field_name in ['symbols', 'list'] else {})
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {field_name}: {e}")
        return default or ([] if field_name in ['symbols', 'list'] else {})
    except Exception as e:
        logger.error(f"Unexpected error parsing {field_name}: {e}")
        return default or ([] if field_name in ['symbols', 'list'] else {})


class SIPSymbolConfig(BaseModel):
    """Configuration for individual symbol in multi-symbol portfolio"""
    symbol: str
    allocation_percentage: float = 100.0
    config: Optional[Dict] = None

    @validator('allocation_percentage')
    def validate_allocation(cls, v):
        if v <= 0 or v > 100:
            raise ValueError('Allocation percentage must be between 0 and 100')
        return v


class SIPConfigRequest(BaseModel):
    """Enhanced SIP Configuration with monthly investment limits"""
    fixed_investment: float = 5000
    max_amount_in_a_month: Optional[float] = 20000
    drawdown_threshold_1: float = -10.0
    drawdown_threshold_2: float = -4.0
    investment_multiplier_1: float = 2.0
    investment_multiplier_2: float = 3.0
    investment_multiplier_3: float = 5.0
    rolling_window: int = 100
    fallback_day: int = 22
    min_investment_gap_days: int = 5
    price_reduction_threshold: float = 4.0  # 4% price reduction for multiple signals

    @validator('fixed_investment')
    def validate_investment(cls, v):
        if v <= 0:
            raise ValueError('Investment amount must be positive')
        return v

    @validator('max_amount_in_a_month', always=True)
    def validate_monthly_limit(cls, v, values):
        if v is None:
            # Default to 4 times the fixed investment
            fixed_investment = values.get('fixed_investment', 5000)
            return fixed_investment * 4
        if v <= 0:
            raise ValueError('Monthly limit must be positive')
        fixed_investment = values.get('fixed_investment', 5000)
        if v < fixed_investment:
            raise ValueError('Monthly limit cannot be less than fixed investment')
        return v

    @validator('price_reduction_threshold')
    def validate_price_threshold(cls, v):
        if v <= 0:
            raise ValueError('Price reduction threshold must be positive')
        return v


class SIPBacktestRequest(BaseModel):
    """Enhanced backtest request with monthly limits"""
    symbols: List[str]
    start_date: str
    end_date: str
    config: SIPConfigRequest

    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError('At least one symbol required')
        return v


class SIPMultiPortfolioRequest(BaseModel):
    """Multi-symbol portfolio creation"""
    portfolio_name: str
    symbols: List[SIPSymbolConfig]
    default_config: SIPConfigRequest
    auto_rebalance: bool = False
    rebalance_frequency_days: int = 30

    @validator('symbols')
    def validate_symbols_allocation(cls, v):
        if not v:
            raise ValueError('At least one symbol required')

        total_allocation = sum(symbol.allocation_percentage for symbol in v)
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError('Total allocation must equal 100%')

        return v


class SIPPortfolioRequest(BaseModel):
    symbol: str
    portfolio_name: Optional[str] = None
    config: SIPConfigRequest


class SIPBacktestResponse(BaseModel):
    """Enhanced backtest response with monthly tracking"""
    backtest_id: str
    symbol: str
    strategy_name: str
    total_investment: float
    final_portfolio_value: float
    total_return_percent: float
    cagr: float
    max_drawdown: Optional[float]
    sharpe_ratio: Optional[float]
    volatility: Optional[float]
    num_trades: int
    num_skipped: int = 0
    monthly_limit_exceeded: int = 0
    price_threshold_skipped: int = 0
    max_amount_in_a_month: float
    price_reduction_threshold: float
    monthly_summary: Dict[str, Any] = {}
    created_at: datetime


class SIPPortfolioResponse(BaseModel):
    portfolio_id: str
    symbol: str
    portfolio_name: Optional[str]
    status: str
    total_invested: float
    current_units: float
    current_value: float
    next_investment_date: Optional[datetime]
    created_at: datetime


class SIPSignalResponse(BaseModel):
    signal_id: str
    symbol: str
    signal_type: str
    recommended_amount: float
    multiplier: float
    current_price: float
    drawdown_percent: Optional[float]
    signal_strength: str
    created_at: datetime


class InvestmentReportRequest(BaseModel):
    symbols: List[str]
    config: Optional[SIPConfigRequest] = None
    report_type: str = "comprehensive"  # "quick", "comprehensive", "detailed"
    include_risk_assessment: bool = True
    include_allocation_suggestions: bool = True

    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError('At least one symbol required')
        if len(v) > 20:
            raise ValueError('Maximum 20 symbols allowed per report')
        return v


class InvestmentReportResponse(BaseModel):
    report_id: str
    report_generated: str
    analysis_period: str
    overall_metrics: Dict[str, Any]
    portfolio_recommendation: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    symbol_reports: Dict[str, Any]
    disclaimer: str


# Create router
sip_router = APIRouter(prefix="/sip", tags=["sip-strategy"])


# ============================================================================
# BACKGROUND SIGNAL PROCESSING
# ============================================================================
def merge_symbol_config_with_fallback(symbol_config: Dict, fallback_config: Dict) -> Dict:
    """
    Merge symbol-specific config with fallback config.
    Symbol config takes precedence, fallback config fills missing values.
    """
    try:
        # Extract symbol's individual config
        symbol_individual_config = symbol_config.get('config', {})

        # If symbol config is empty or only has placeholder data, use fallback
        if (not symbol_individual_config or
                symbol_individual_config == {"additionalProp1": {}} or
                not isinstance(symbol_individual_config, dict) or
                len(symbol_individual_config) == 0):
            logger.info(f"Using fallback config for symbol {symbol_config.get('symbol')}")
            return fallback_config.copy()

        # Check if symbol config has actual SIP parameters
        sip_params = ['fixed_investment', 'drawdown_threshold_1', 'rolling_window', 'fallback_day']
        has_sip_params = any(param in symbol_individual_config for param in sip_params)

        if not has_sip_params:
            logger.info(f"Symbol config lacks SIP parameters, using fallback for {symbol_config.get('symbol')}")
            return fallback_config.copy()

        # Merge configs - symbol config overrides fallback
        merged_config = fallback_config.copy()
        merged_config.update(symbol_individual_config)

        logger.info(f"Using merged config for symbol {symbol_config.get('symbol')}: symbol-specific values applied")
        return merged_config

    except Exception as e:
        logger.error(f"Error merging configs: {e}, using fallback")
        return fallback_config.copy()


async def get_monthly_investment_total(portfolio_id: str, symbol: str, trading_db: AsyncSession) -> float:
    """Get total investment for a symbol in current month"""
    try:
        current_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month_start = (current_month_start.replace(day=28) + timedelta(days=4)).replace(day=1)

        query = text("""
            SELECT COALESCE(SUM(amount), 0) as total_invested
            FROM sip_actual_trades
            WHERE portfolio_id = :portfolio_id 
            AND symbol = :symbol
            AND timestamp >= :month_start
            AND timestamp < :next_month
        """)

        result = await trading_db.execute(query, {
            'portfolio_id': portfolio_id,
            'symbol': symbol,
            'month_start': current_month_start,
            'next_month': next_month_start
        })

        total = result.scalar() or 0.0
        logger.debug(f"Monthly investment total for {symbol}: ₹{total:.2f}")
        return total

    except Exception as e:
        logger.error(f"Error fetching monthly investment total for {symbol}: {e}")
        return 0.0


async def get_investment_signals_with_monthly_limits(
        symbol: str,
        config_dict: Dict,
        strategy,
        nsedata_db: AsyncSession,
        trading_db: AsyncSession,
        portfolio_id: str
) -> Dict:
    """
    Generate investment signals using existing enhanced strategy with monthly limits
    - Uses the actual get_next_investment_signals from EnhancedSIPStrategy
    - Adds monthly investment tracking on top
    - Matches backtesting logic exactly
    """
    try:
        # Fetch recent data (last 6 months for better analysis)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

        # Get historical data using existing method
        data = await strategy.fetch_data_from_db_async(symbol, start_date, end_date)

        if data.empty:
            return {
                "signal": "NO_DATA",
                "confidence": 0,
                "recommended_amount": 0,
                "current_price": 0,
                "message": "Insufficient data for analysis"
            }

        # Create SIPConfig object from dictionary
        config_obj = SIPConfig(
            fixed_investment=config_dict.get('fixed_investment', 5000),
            drawdown_threshold_1=config_dict.get('drawdown_threshold_1', -10.0),
            drawdown_threshold_2=config_dict.get('drawdown_threshold_2', -4.0),
            investment_multiplier_1=config_dict.get('investment_multiplier_1', 2.0),
            investment_multiplier_2=config_dict.get('investment_multiplier_2', 3.0),
            investment_multiplier_3=config_dict.get('investment_multiplier_3', 5.0),
            rolling_window=config_dict.get('rolling_window', 100),
            fallback_day=config_dict.get('fallback_day', 22),
            min_investment_gap_days=config_dict.get('min_investment_gap_days', 5),
            max_amount_in_a_month=config_dict.get('max_amount_in_a_month',
                                                  config_dict.get('fixed_investment', 5000) * 4),
            price_reduction_threshold=config_dict.get('price_reduction_threshold', 4.0)
        )

        # Use existing get_next_investment_signals method - this gives us the SAME logic as backtesting
        signals = strategy.get_next_investment_signals(data, config_obj)

        # Ensure signals is a dictionary and has required fields
        if not isinstance(signals, dict):
            signals = {
                "signal": "ERROR",
                "confidence": 0,
                "recommended_amount": config_obj.fixed_investment,
                "current_price": float(data['close'].iloc[-1]) if not data.empty else 0,
                "message": "Invalid signal format from strategy"
            }

        # Ensure current_price is set
        if 'current_price' not in signals:
            signals['current_price'] = float(data['close'].iloc[-1]) if not data.empty else 0

        # Check monthly investment limits AFTER getting base signals
        monthly_invested = await get_monthly_investment_total(portfolio_id, symbol, trading_db)
        max_monthly = config_obj.max_amount_in_a_month
        available_monthly_budget = max(0, max_monthly - monthly_invested)

        # Apply monthly limit constraints to recommended amount
        recommended_amount = signals.get('recommended_amount', config_obj.fixed_investment)

        # If no budget available, override signal
        if available_monthly_budget < config_obj.fixed_investment:
            signals.update({
                "signal": "MONTHLY_LIMIT_REACHED",
                "confidence": 0,
                "recommended_amount": 0,
                "message": f"Monthly limit reached. Available: ₹{available_monthly_budget:.2f}"
            })
        elif recommended_amount > available_monthly_budget:
            # Adjust amount to available budget
            signals.update({
                'recommended_amount': available_monthly_budget,
                'amount_adjusted': True,
                'original_amount': recommended_amount,
                'message': f"Amount adjusted for monthly limit. Available: ₹{available_monthly_budget:.2f}"
            })

        # Add monthly tracking metadata (same as backtesting)
        signals.update({
            "monthly_invested_so_far": monthly_invested,
            "monthly_budget_remaining": available_monthly_budget,
            "monthly_limit": max_monthly
        })

        return signals

    except Exception as e:
        logger.error(f"Error generating enhanced signals for {symbol}: {e}")
        return {
            "signal": "ERROR",
            "confidence": 0,
            "recommended_amount": 0,
            "current_price": 0,
            "message": f"Signal generation failed: {str(e)}"
        }


async def get_user_apis_for_gtt(user_id: str, trading_db: AsyncSession) -> Dict:
    """
    Get user APIs for GTT order placement
    FIXED: Uses direct database queries instead of importing from main.py
    """
    try:
        # Instead of importing from main.py, let's recreate the user API logic here
        # This avoids circular imports

        from backend.app.services import init_upstox_api, init_zerodha_api

        # Initialize APIs directly
        user_apis_dict = {
            "upstox": {"order": None, "quotes": None},
            "zerodha": {"kite": None, "quotes": None}
        }

        try:
            # Initialize Zerodha APIs
            kite_apis = await init_zerodha_api(trading_db, user_id)
            if kite_apis and "kite" in kite_apis:
                user_apis_dict["zerodha"]["kite"] = kite_apis["kite"]
                logger.info(f"Zerodha API initialized for user {user_id}")
        except Exception as zerodha_error:
            logger.warning(f"Zerodha API initialization failed for user {user_id}: {zerodha_error}")

        try:
            # Initialize Upstox APIs
            upstox_apis = await init_upstox_api(trading_db, user_id)
            if upstox_apis and "order" in upstox_apis:
                user_apis_dict["upstox"]["order"] = upstox_apis["order"]
                logger.info(f"Upstox API initialized for user {user_id}")
        except Exception as upstox_error:
            logger.warning(f"Upstox API initialization failed for user {user_id}: {upstox_error}")

        return user_apis_dict

    except Exception as e:
        logger.error(f"Error getting user APIs for {user_id}: {e}")
        return {}


async def get_instrument_token_for_symbol(symbol: str, trading_db: AsyncSession) -> Optional[str]:
    """Get instrument token for symbol - required for GTT order placement"""
    try:
        # Query instruments table for the symbol
        query = text("""
            SELECT instrument_token 
            FROM instruments 
            WHERE trading_symbol = :symbol 
            AND exchange = 'NSE'
            LIMIT 1
        """)

        result = await trading_db.execute(query, {'symbol': symbol})
        row = result.fetchone()

        if row:
            return str(row[0])
        else:
            logger.warning(f"Instrument token not found for symbol: {symbol}")
            return None

    except Exception as e:
        logger.error(f"Error fetching instrument token for {symbol}: {e}")
        return None


async def get_order_manager_instance() -> Optional[OrderManager]:
    """
    Get OrderManager instance without circular import
    FIXED: Creates new instance instead of importing global one
    """
    try:
        from backend.app.services import OrderMonitor

        # Create fresh instances to avoid circular import
        order_monitor = OrderMonitor()
        order_manager = OrderManager(monitor=order_monitor)

        logger.debug("Created new OrderManager instance for GTT operations")
        return order_manager

    except Exception as e:
        logger.error(f"Error creating OrderManager instance: {e}")
        return None


async def place_gtt_order_via_existing_api(
        user_id: str,
        symbol: str,
        quantity: int,
        trigger_price: float,
        limit_price: float,
        current_price: float,
        trading_db: AsyncSession
) -> Dict:
    """
    Place GTT order using OrderManager.place_gtt_order method
    FIXED: Creates OrderManager instance instead of importing global one
    """
    try:
        # Get user APIs without circular import
        user_apis_dict = await get_user_apis_for_gtt(user_id, trading_db)

        if not user_apis_dict:
            logger.error(f"Could not get user APIs for user {user_id}")
            return {"status": "error", "gtt_id": None, "message": "User APIs not available"}

        # Get Zerodha API (assuming Zerodha for SIP orders)
        broker = "Zerodha"
        api = user_apis_dict.get("zerodha", {}).get("kite")

        if not api:
            logger.error(f"Zerodha API not initialized for user {user_id}")
            return {"status": "error", "gtt_id": None, "message": "Zerodha API not initialized"}

        # Get instrument token for the symbol
        instrument_token = await get_instrument_token_for_symbol(symbol, trading_db)

        if not instrument_token:
            logger.error(f"Could not find instrument token for {symbol}")
            return {"status": "error", "gtt_id": None, "message": f"Instrument token not found for {symbol}"}

        # Create OrderManager instance (avoids circular import)
        order_manager = await get_order_manager_instance()

        if not order_manager:
            logger.error("Could not create OrderManager instance")
            return {"status": "error", "gtt_id": None, "message": "OrderManager not available"}

        # Use OrderManager.place_gtt_order method
        response = await order_manager.place_gtt_order(
            api=api,
            instrument_token=instrument_token,
            trading_symbol=symbol,
            transaction_type="BUY",
            quantity=quantity,
            trigger_type="single",
            trigger_price=trigger_price,
            limit_price=limit_price,
            last_price=current_price,
            second_trigger_price=None,
            second_limit_price=None,
            broker=broker,
            db=trading_db,
            user_id=user_id
        )

        if response.get("status") == "success":
            gtt_id = response.get("gtt_id")
            logger.info(f"GTT order placed successfully via OrderManager: {gtt_id} for {symbol}")

            return {
                "status": "success",
                "gtt_id": gtt_id,
                "message": "GTT order placed successfully via OrderManager"
            }
        else:
            error_msg = response.get("message", "Unknown error from OrderManager")
            logger.error(f"OrderManager GTT placement failed for {symbol}: {error_msg}")
            return {
                "status": "error",
                "gtt_id": None,
                "message": f"GTT placement failed: {error_msg}"
            }

    except Exception as e:
        logger.error(f"Error placing GTT order via OrderManager for {symbol}: {e}")
        return {
            "status": "error",
            "gtt_id": None,
            "message": f"GTT order placement failed: {str(e)}"
        }

async def daily_signal_check():
    """Enhanced background task with proper session management and error handling"""
    job_start_time = datetime.now()
    logger.info(f"🔄 Starting daily signal check at {job_start_time.strftime('%Y-%m-%d %H:%M:%S IST')}")

    # Metrics tracking
    metrics = {
        'start_time': job_start_time,
        'portfolios_processed': 0,
        'signals_generated': 0,
        'errors_encountered': 0,
        'symbols_processed': []
    }

    trading_db = None
    nsedata_db = None

    try:
        # Create async generators and get sessions from them
        trading_db_generator = get_db()
        nsedata_db_generator = get_nsedata_db()

        # Get sessions from the async generators
        trading_db = await trading_db_generator.__anext__()
        nsedata_db = await nsedata_db_generator.__anext__()
        logger.info("✅ Database sessions acquired successfully")

        # Get all active portfolios
        portfolios_query = text("""
            SELECT portfolio_id, user_id, symbols, config, 
                   total_invested, current_units, next_investment_date
            FROM sip_portfolios 
            WHERE status = 'active'
        """)

        result = await trading_db.execute(portfolios_query)
        portfolios = result.fetchall()

        logger.info(f"📊 Found {len(portfolios)} active portfolios to process")

        if not portfolios:
            logger.warning("⚠️ No active portfolios found")
            return

        # Initialize strategy
        strategy = EnhancedSIPStrategy(
            nsedata_session=nsedata_db,
            trading_session=trading_db
        )

        # Process each portfolio
        for portfolio in portfolios:
            portfolio_start_time = datetime.now()
            portfolio_id, user_id, symbols_json, config_json, total_invested, current_units, next_investment_date = portfolio

            try:
                # Safe JSON parsing with validation
                if isinstance(symbols_json, str):
                    symbols_data = json.loads(symbols_json)
                else:
                    symbols_data = symbols_json or []

                if isinstance(config_json, str):
                    config_dict = json.loads(config_json)
                else:
                    config_dict = config_json or {}


                config = SIPConfig(**config_dict)
                portfolio_signals_generated = 0

                # Process each symbol in the portfolio
                for symbol_config in symbols_data:
                    symbol_start_time = datetime.now()

                    # Extract and validate symbol name
                    if isinstance(symbol_config, dict):
                        symbol = symbol_config.get('symbol')
                        allocation_pct = symbol_config.get('allocation_percentage', 100.0)
                        if not symbol:
                            logger.warning(f"Symbol config missing 'symbol' key: {symbol_config}")
                            continue
                    elif isinstance(symbol_config, str):
                        symbol = symbol_config
                        allocation_pct = 100.0
                        # Convert string to dict format for consistency
                        symbol_config = {'symbol': symbol, 'allocation_percentage': allocation_pct, 'config': {}}
                    else:
                        logger.warning(f"Invalid symbol config type: {type(symbol_config)}")
                        continue

                    # Validate symbol
                    if not symbol or len(symbol.strip()) == 0:
                        logger.warning(f"Empty or invalid symbol: '{symbol}'")
                        continue

                    # Normalize symbol name
                    symbol = symbol.strip().upper()
                    metrics['symbols_processed'].append(symbol)

                    try:
                        # Check timing constraints
                        current_date = datetime.now().date()

                        # Check last investment date to ensure minimum gap
                        last_trade_query = text("""
                                            SELECT MAX(timestamp) FROM sip_actual_trades 
                                            WHERE portfolio_id = :portfolio_id AND symbol = :symbol
                                        """)

                        last_trade_result = await trading_db.execute(
                            last_trade_query, {'portfolio_id': portfolio_id, 'symbol': symbol}
                        )
                        last_trade_date = last_trade_result.scalar()

                        # SOLUTION 1: Merge symbol config with fallback config
                        merged_config = merge_symbol_config_with_fallback(symbol_config, config_dict)

                        # Enforce minimum gap
                        min_gap_days = merged_config.get('min_investment_gap_days', 5)
                        if last_trade_date:
                            days_since_last = (current_date - last_trade_date.date()).days
                            if days_since_last < min_gap_days:
                                logger.info(
                                    f"⏭️ Skipping {symbol}: minimum gap not met "
                                    f"({days_since_last}/{min_gap_days} days)"
                                )
                                continue

                        # SOLUTION 2: Use enhanced signal generation with monthly limits (same as backtest)
                        try:
                            signals = await get_investment_signals_with_monthly_limits(
                                symbol=symbol,
                                config_dict=merged_config,
                                strategy=strategy,
                                nsedata_db=nsedata_db,
                                trading_db=trading_db,
                                portfolio_id=portfolio_id
                            )
                        except Exception as signal_error:
                            logger.error(f"Error generating signals for {symbol}: {signal_error}")
                            signals = {
                                "signal": "ERROR",
                                "confidence": 0,
                                "recommended_amount": 0,
                                "current_price": 0,
                                "message": f"Signal generation failed: {str(signal_error)}"
                            }

                        # Validate signals response
                        if not isinstance(signals, dict):
                            logger.error(f"Invalid signals response for {symbol}: {type(signals)}")
                            continue

                        # Add metadata to signals
                        signals.update({
                            'symbol': symbol,
                            'allocation_percentage': allocation_pct,
                            'portfolio_id': portfolio_id,
                            'config_used': 'merged' if merged_config != config_dict else 'fallback'
                        })

                        # Adjust recommended amount based on allocation percentage
                        if 'recommended_amount' in signals and allocation_pct != 100.0:
                            base_amount = signals['recommended_amount']
                            allocated_amount = (base_amount * allocation_pct) / 100.0
                            signals['allocated_amount'] = allocated_amount
                            signals['base_amount'] = base_amount
                            signals['recommended_amount'] = allocated_amount

                        portfolio_signals_generated += 1

                        logger.info(
                            f"✅ Generated signals for {symbol}: {signals.get('signal', 'UNKNOWN')} "
                            f"(confidence: {signals.get('confidence', 0):.2f}, "
                            f"amount: ₹{signals.get('recommended_amount', 0):,.2f})"
                        )

                        # SOLUTION 3: GTT order integration
                        try:
                            if signals.get('signal') not in ['NO_DATA', 'ERROR']:
                                save_result = await save_signal_with_gtt_order(
                                    portfolio_id, symbol, signals, trading_db, merged_config
                                )

                                if save_result.get('status') == 'success':
                                    logger.info(f"✅ Signal and REAL GTT order saved successfully for {symbol}")
                                    metrics['signals_generated'] += 1
                                elif save_result.get('status') in ['partial_success', 'signal_only']:
                                    logger.info(
                                        f"⚠️ Signal saved (GTT partial/none) for {symbol}: {save_result.get('message')}")
                                    metrics['signals_generated'] += 1
                                else:
                                    logger.error(f"❌ Failed to save signal for {symbol}: {save_result.get('message')}")
                                    metrics['errors_encountered'] += 1
                            else:
                                logger.info(f"ℹ️ No actionable signal for {symbol}: {signals.get('signal')}")

                        except Exception as save_error:
                            logger.error(f"Error saving signal for {symbol}: {save_error}")
                            metrics['errors_encountered'] += 1

                    except asyncio.TimeoutError:
                        logger.error(f"Timeout processing {symbol}")
                        metrics['errors_encountered'] += 1
                        continue
            except Exception as symbol_error:
                logger.error(f"Error processing {symbol}: {symbol_error}")
                metrics['errors_encountered'] += 1
                continue

            # Log symbol processing time
            symbol_processing_time = (datetime.now() - symbol_start_time).total_seconds()
            logger.debug(f"⏱️ Processed {symbol} in {symbol_processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Critical error in daily signal check: {e}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        metrics['errors_encountered'] += 1

    finally:
        # Calculate and log final metrics
        job_end_time = datetime.now()
        total_processing_time = (job_end_time - job_start_time).total_seconds()

        metrics.update({
            'end_time': job_end_time,
            'total_processing_time_seconds': total_processing_time,
            'unique_symbols_count': len(set(metrics['symbols_processed']))
        })

        # Comprehensive summary log
        success_rate = (
            (metrics['signals_generated'] / max(len(metrics['symbols_processed']), 1)) * 100
            if metrics['symbols_processed'] else 0
        )

        logger.info(
            f"🏁 Daily signal check completed in {total_processing_time:.2f}s\n"
            f"📊 Final Summary:\n"
            f"   • Portfolios processed: {metrics['portfolios_processed']}\n"
            f"   • Total symbols checked: {len(metrics['symbols_processed'])}\n"
            f"   • Unique symbols: {metrics['unique_symbols_count']}\n"
            f"   • Signals generated: {metrics['signals_generated']}\n"
            f"   • Success rate: {success_rate:.1f}%\n"
            f"   • Errors encountered: {metrics['errors_encountered']}\n"
            f"   • Average time per portfolio: {total_processing_time / max(metrics['portfolios_processed'], 1):.2f}s\n"
            f"   • Memory efficient: Sessions properly managed ✅"
        )

        # Store metrics for monitoring (if trading_db is still available)
        try:
            await store_job_metrics(trading_db, 'daily_signal_check', metrics)
        except Exception as metric_error:
            logger.warning(f"Could not store job metrics: {metric_error}")

        try:
            if trading_db:
                await trading_db_generator.aclose()
            if nsedata_db:
                await nsedata_db_generator.aclose()
        except Exception as cleanup_error:
            logger.debug(f"Session cleanup not completed: {cleanup_error}")


async def monitor_and_update_gtt_status():
    """
    Monitor GTT orders and update trade records when they execute
    This should be called periodically to sync GTT status
    """
    try:
        db_gen = get_db()
        trading_db = await db_gen.__anext__()
        # Get active GTT orders
        active_gtt_query = text("""
            SELECT t.trade_id, t.gtt_order_id, t.symbol, t.portfolio_id, g.status
            FROM sip_actual_trades t
            JOIN gtt_orders g ON t.gtt_order_id = g.gtt_order_id
            WHERE t.execution_status = 'GTT_PENDING'
            AND g.status IN ('ACTIVE', 'TRIGGERED', 'CANCELLED')
        """)

        result = await trading_db.execute(active_gtt_query)
        gtt_trades = result.fetchall()

        for trade_row in gtt_trades:
            trade_id, gtt_order_id, symbol, portfolio_id, gtt_status = trade_row

            # Update trade status based on GTT status
            new_execution_status = {
                'TRIGGERED': 'EXECUTED',
                'CANCELLED': 'CANCELLED',
                'ACTIVE': 'GTT_PENDING'
            }.get(gtt_status, 'GTT_PENDING')

            if new_execution_status != 'GTT_PENDING':
                update_trade_query = text("""
                    UPDATE sip_actual_trades 
                    SET execution_status = :status, updated_at = :now
                    WHERE trade_id = :trade_id
                """)

                await trading_db.execute(update_trade_query, {
                    'status': new_execution_status,
                    'trade_id': trade_id,
                    'now': datetime.now()
                })

                logger.info(f"Updated trade {trade_id} status to {new_execution_status}")

        await trading_db.commit()

    except Exception as e:
        logger.error(f"Error monitoring GTT status: {e}")
        await trading_db.rollback()
    finally:
        try:
            if trading_db:
                await db_gen.aclose()
        except Exception as cleanup_error:
            logger.debug(f"Session cleanup not completed: {cleanup_error}")


async def store_job_metrics(db_session: AsyncSession, job_name: str, metrics: dict):
    """Store job execution metrics for monitoring"""
    try:
        insert_metrics = text("""
            INSERT INTO job_execution_metrics 
            (job_name, execution_start, execution_end, total_time_seconds, 
             portfolios_processed, signals_generated, errors_count, metadata)
            VALUES (:job_name, :start_time, :end_time, :total_time, 
                    :portfolios, :signals, :errors, :metadata)
        """)

        await db_session.execute(insert_metrics, {
            'job_name': job_name,
            'start_time': metrics['start_time'],
            'end_time': metrics['end_time'],
            'total_time': metrics['total_processing_time_seconds'],
            'portfolios': metrics['portfolios_processed'],
            'signals': metrics['signals_generated'],
            'errors': metrics['errors_encountered'],
            'metadata': json.dumps({
                'unique_symbols_count': metrics['unique_symbols_count'],
                'symbols_processed': metrics['symbols_processed'][:50],  # Limit for storage
                'success_rate': (metrics['signals_generated'] / max(len(metrics['symbols_processed']), 1)) * 100
            })
        })

        await db_session.commit()
        logger.debug("Job metrics stored successfully")

    except Exception as e:
        logger.error(f"Failed to store job metrics: {e}")
        await db_session.rollback()


def calculate_next_investment_date(current_date: datetime.date, config: SIPConfig) -> datetime.date:
    """Calculate next investment date based on fallback day"""
    next_month = current_date.replace(day=1) + timedelta(days=32)
    next_month = next_month.replace(day=1)

    try:
        next_investment = next_month.replace(day=config.fallback_day)
    except ValueError:
        # Handle months with fewer days
        next_investment = next_month.replace(day=min(config.fallback_day, 28))

    return next_investment


async def save_signal_to_database_with_real_gtt(
        signal_id: str,
        portfolio_id: str,
        symbol: str,
        signals: Dict,
        config: Dict,
        trading_db: AsyncSession,
        gtt_order_id: Optional[str] = None
) -> Dict:
    """Save signal record to database with GTT order linking"""
    try:
        # Prepare serializable signal data
        serializable_signals = {}
        for key, value in signals.items():
            try:
                if isinstance(value, (datetime, date)):
                    serializable_signals[key] = value.isoformat()
                elif hasattr(value, 'item'):  # numpy scalar
                    serializable_signals[key] = value.item()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    serializable_signals[key] = value
                else:
                    serializable_signals[key] = str(value)
            except Exception:
                serializable_signals[key] = str(value)

        signal_data_json = json.dumps(serializable_signals)

        # Calculate GTT trigger price
        current_price = signals.get('current_price', 0)
        gtt_trigger_price = current_price * 0.994 if current_price > 0 else 0  # 0.2% below current price

        insert_signal = text("""
            INSERT INTO sip_signals 
            (signal_id, portfolio_id, symbol, signal_type, recommended_amount, 
             multiplier, current_price, drawdown_percent, signal_strength, 
             gtt_trigger_price, signal_data, gtt_order_id, gtt_status,
             created_at, expires_at)
            VALUES (:signal_id, :portfolio_id, :symbol, :signal_type, 
                    :recommended_amount, :multiplier, :current_price, 
                    :drawdown_percent, :signal_strength, :gtt_trigger_price,
                    :signal_data, :gtt_order_id, :gtt_status,
                    :created_at, :expires_at)
        """)

        await trading_db.execute(insert_signal, {
            'signal_id': signal_id,
            'portfolio_id': portfolio_id,
            'symbol': symbol,
            'signal_type': signals.get('signal', 'NORMAL'),
            'recommended_amount': float(signals.get('recommended_amount', config.get('fixed_investment', 5000))),
            'multiplier': float(signals.get('confidence', 1.0)),
            'current_price': float(current_price),
            'drawdown_percent': float(signals.get('drawdown_100', 0)),
            'signal_strength': min(max(float(signals.get('confidence', 1.0)) * 100, 0), 100),
            'gtt_trigger_price': float(gtt_trigger_price),
            'signal_data': signal_data_json,
            'gtt_order_id': gtt_order_id,
            'gtt_status': 'ACTIVE' if gtt_order_id else 'NONE',
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(days=7)
        })

        await trading_db.commit()

        return {"status": "success", "signal_id": signal_id}

    except Exception as e:
        logger.error(f"Error saving signal to database: {e}")
        await trading_db.rollback()
        raise


async def create_gtt_trade_record(
        portfolio_id: str,
        symbol: str,
        gtt_order_id: str,
        trigger_price: float,
        quantity: int,
        amount: float,
        trading_db: AsyncSession
) -> str:
    """
    Create trade record for GTT order placement
    This ensures GTT orders count towards minimum gap tracking
    """
    try:
        # Generate trade ID
        trade_id = f"gtt_{portfolio_id}_{int(datetime.now().timestamp())}"

        # Insert trade record with GTT-specific details
        insert_trade = text("""
            INSERT INTO sip_actual_trades 
            (trade_id, portfolio_id, symbol, timestamp, price, units, amount, 
             trade_type, execution_status, gtt_order_id)
            VALUES (:trade_id, :portfolio_id, :symbol, :timestamp, :price, :units, :amount, 
                    :trade_type, :execution_status, :gtt_order_id)
        """)

        await trading_db.execute(insert_trade, {
            'trade_id': trade_id,
            'portfolio_id': portfolio_id,
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': trigger_price,
            'units': quantity,
            'amount': amount,
            'trade_type': 'GTT_PLACED',
            'execution_status': 'GTT_PENDING',
            'gtt_order_id': gtt_order_id
        })

        logger.info(f"✅ Created trade record {trade_id} for GTT order {gtt_order_id}")
        return trade_id

    except Exception as e:
        logger.error(f"Error creating GTT trade record: {e}")
        raise


async def update_portfolio_after_gtt_placement(
        portfolio_id: str,
        amount: float,
        units: int,
        config: dict,
        trading_db: AsyncSession
):
    """
    Update portfolio totals after GTT placement
    This ensures portfolio tracking is consistent
    """
    try:
        # Calculate next investment date
        current_date = datetime.now().date()
        next_investment_date = calculate_next_investment_date(current_date, SIPConfig(**config))

        # Update portfolio with GTT investment
        update_query = text("""
            UPDATE sip_portfolios 
            SET total_invested = total_invested + :amount,
                current_units = current_units + :units,
                next_investment_date = :next_date,
                updated_at = :now
            WHERE portfolio_id = :portfolio_id
        """)

        await trading_db.execute(update_query, {
            'amount': amount,
            'units': units,
            'next_date': next_investment_date,
            'now': datetime.now(),
            'portfolio_id': portfolio_id
        })

        logger.info(f"✅ Updated portfolio {portfolio_id}: +₹{amount:,.2f}, +{units} units")

    except Exception as e:
        logger.error(f"Error updating portfolio after GTT placement: {e}")
        raise

async def save_signal_with_gtt_order(portfolio_id: str, symbol: str, signals: dict,
                                     trading_db: AsyncSession, config: dict):
    """
    REAL GTT Integration using OrderManager.place_gtt_order
    FIXED: No circular imports - creates OrderManager instance locally
    1. Saves signal to database
    2. Places actual GTT order through broker API integration
    3. Links signal with GTT order ID
    4. Handles errors gracefully
    """
    signal_id =  f"sig_{str(uuid.uuid4())}_{int(datetime.now().timestamp())}"
    gtt_order_id = None

    try:
        current_price = signals.get('current_price', 0)
        recommended_amount = signals.get('recommended_amount', 0)

        # Validate signal before processing
        if signals.get('signal') in ['NO_DATA', 'ERROR', 'MONTHLY_LIMIT_REACHED']:
            logger.info(f"Saving signal without GTT order for {symbol}: {signals.get('signal')}")
            await save_signal_to_database_with_real_gtt(
                signal_id, portfolio_id, symbol, signals, config, trading_db, None
            )
            return {
                "status": "signal_only",
                "signal_id": signal_id,
                "message": f"Signal saved: {signals.get('signal')}"
            }

        # Validate price and amount for GTT order
        if recommended_amount <= 0 or current_price <= 0:
            logger.warning(f"Invalid amount or price for {symbol}: amount={recommended_amount}, price={current_price}")
            await save_signal_to_database_with_real_gtt(
                signal_id, portfolio_id, symbol, signals, config, trading_db, None
            )
            return {
                "status": "signal_only",
                "signal_id": signal_id,
                "message": "Signal saved without GTT (invalid price/amount)"
            }

        # Calculate GTT parameters
        gtt_trigger_price = current_price * 0.994  # 0.6% below current price
        limit_price = current_price * 0.992  # 0.8% below current price
        quantity = max(1, int(recommended_amount / current_price))  # Ensure minimum 1 quantity

        # Get user_id from portfolio
        user_query = text("SELECT user_id FROM sip_portfolios WHERE portfolio_id = :portfolio_id")
        user_result = await trading_db.execute(user_query, {'portfolio_id': portfolio_id})
        user_row = user_result.fetchone()
        user_id = user_row[0] if user_row else None

        if not user_id:
            logger.error(f"Could not find user_id for portfolio {portfolio_id}")
            await save_signal_to_database_with_real_gtt(
                signal_id, portfolio_id, symbol, signals, config, trading_db, None
            )
            return {
                "status": "signal_only",
                "signal_id": signal_id,
                "message": "Signal saved without GTT (user not found)"
            }

        # Place GTT order
        logger.info(
            f"Placing GTT order for {symbol} : trigger=₹{gtt_trigger_price:.2f}, qty={quantity}")

        gtt_result = await place_gtt_order_via_existing_api(
            user_id=user_id,
            symbol=symbol,
            quantity=quantity,
            trigger_price=gtt_trigger_price,
            limit_price=limit_price,
            current_price=current_price,
            trading_db=trading_db
        )

        # Save signal with GTT order ID (if successful)
        if gtt_result.get('status') == 'success':
            gtt_order_id = gtt_result.get('gtt_id')

        await save_signal_to_database_with_real_gtt(
            signal_id, portfolio_id, symbol, signals, config, trading_db, gtt_order_id
        )

        if gtt_order_id:
            await create_gtt_trade_record(
                portfolio_id=portfolio_id,
                symbol=symbol,
                gtt_order_id=gtt_order_id,
                trigger_price=gtt_trigger_price,
                quantity=quantity,
                amount=recommended_amount,
                trading_db=trading_db
            )

            # SOLUTION: Update portfolio totals
            await update_portfolio_after_gtt_placement(
                portfolio_id=portfolio_id,
                amount=recommended_amount,
                units=quantity,
                config=config,
                trading_db=trading_db
            )

            logger.info(
                f"✅ REAL GTT order flow successful for {symbol}:\n"
                f"   Signal ID: {signal_id}\n"
                f"   GTT Order ID: {gtt_order_id}\n"
                f"   Quantity: {quantity} units\n"
                f"   Trigger Price: ₹{gtt_trigger_price:.2f}\n"
                f"   Amount: ₹{recommended_amount:,.2f}\n"
                f"   Placed via: OrderManager (no circular import)"
            )

            return {
                "status": "success",
                "signal_id": signal_id,
                "gtt_order_id": gtt_order_id,
                "trade_created": True,
                "message": "Signal saved, GTT order placed, and trade recorded for tracking"
            }
        else:
            logger.warning(f"GTT order failed for {symbol}, signal saved without GTT")
            return {
                "status": "partial_success",
                "signal_id": signal_id,
                "gtt_order_id": None,
                "trade_created": False,
                "message": f"Signal saved but GTT order failed: {gtt_result.get('message', 'Unknown error')}"
            }

    except Exception as e:
        logger.error(f"Error in GTT order flow for {symbol}: {e}")
        try:
            await trading_db.rollback()
            # Try to save signal only as fallback
            await save_signal_to_database_with_real_gtt(
                signal_id, portfolio_id, symbol, signals, config, trading_db, None
            )
            return {
                "status": "error_recovered",
                "signal_id": signal_id,
                "gtt_order_id": None,
                "trade_created": False,
                "message": f"Error occurred but signal saved: {str(e)}"
            }
        except Exception as fallback_error:
            logger.error(f"Complete failure for {symbol}: {fallback_error}")
            return {
                "status": "error",
                "signal_id": None,
                "gtt_order_id": None,
                "trade_created": False,
                "message": f"Complete failure: {str(e)}"
            }


async def update_next_investment_date(portfolio_id: str, next_date: datetime.date,
                                      trading_db: AsyncSession):
    """Update next investment date in portfolio"""
    try:
        update_query = text("""
            UPDATE sip_portfolios 
            SET next_investment_date = :next_date, updated_at = :now
            WHERE portfolio_id = :portfolio_id
        """)

        await trading_db.execute(update_query, {
            'portfolio_id': portfolio_id,
            'next_date': next_date,
            'now': datetime.now()
        })

        logger.info(f"Updated next investment date for {portfolio_id} to {next_date}")

    except Exception as e:
        logger.error(f"Error updating next investment date: {e}")

# ============================================================================
# HEALTH AND TESTING ENDPOINTS
# ============================================================================

@sip_router.get("/health")
async def health_check():
    """Enhanced SIP service health check"""
    return {
        "status": "healthy",
        "service": "Enhanced SIP Strategy API",
        "version": "4.0.0",
        "fixes_implemented": [
            "✅ Correct data fetching with proper table names",
            "✅ Fixed database query structure",
            "✅ Enhanced error handling for missing data",
            "✅ Proper symbol case handling",
            "✅ All missing endpoints restored"
        ],
        "background_scheduler": scheduler.running,
        "timestamp": datetime.now().isoformat()
    }


@sip_router.get("/database/status")
async def check_database_status(
        trading_db: AsyncSession = Depends(get_db),
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Check database connectivity and test data access"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "databases": {}
    }

    # Test trading database
    try:
        result = await trading_db.execute(text("SELECT 1 as test, NOW() as server_time"))
        test_result = result.fetchone()
        status["databases"]["trading_db"] = {
            "status": "connected",
            "server_time": test_result[1].isoformat() if test_result else None,
            "purpose": "User data, portfolios, signals, backtest results"
        }
    except Exception as e:
        status["databases"]["trading_db"] = {"status": "error", "error": str(e)}

    # Test NSE data database
    try:
        result = await nsedata_db.execute(text("SELECT 1 as test, NOW() as server_time"))
        test_result = result.fetchone()
        status["databases"]["nsedata_db"] = {
            "status": "connected",
            "server_time": test_result[1].isoformat() if test_result else None,
            "purpose": "Stock market data, historical prices"
        }
    except Exception as e:
        status["databases"]["nsedata_db"] = {"status": "error", "error": str(e)}

    # Overall status
    all_connected = all(db["status"] == "connected" for db in status["databases"].values())
    status["overall"] = "healthy" if all_connected else "degraded"

    return status


# ============================================================================
# SYMBOLS AND DATA AVAILABILITY
# ============================================================================

@sip_router.get("/symbols")
async def get_available_symbols(
        limit: int = 100,
        offset: int = 0,
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Get available symbols with enhanced filtering"""
    try:
        # Get symbols from table names in the database
        symbols_query = text("""
            SELECT tablename
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tableowner != 'postgres'
            AND tablename NOT IN ('STOCKS_IN_DB', 'spatial_ref_sys')
            ORDER BY tablename
            LIMIT :limit OFFSET :offset
        """)

        result = await nsedata_db.execute(symbols_query, {"limit": limit, "offset": offset})
        symbols_data = result.fetchall()

        symbols = [row[0] for row in symbols_data]

        return {
            "symbols": symbols,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": len(symbols) == limit
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.get("/market-data/{symbol}")
async def get_market_data_info(
        symbol: str,
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Get market data information for a symbol with proper table name handling"""
    try:
        # First check if table exists
        table_exists_query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = :table_name
            )
        """)

        exists_result = await nsedata_db.execute(table_exists_query, {"table_name": symbol})
        table_exists = exists_result.scalar()

        if not table_exists:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in database")

        # Get comprehensive symbol information using proper quoting
        query = text(f"""
            SELECT 
                COUNT(*) as total_records,
                MIN(timestamp) as data_start,
                MAX(timestamp) as data_end,
                AVG(close) as avg_price,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(volume) as avg_volume
            FROM "{symbol}"
        """)

        result = await nsedata_db.execute(query)
        stats = result.fetchone()

        # Get recent data sample
        recent_query = text(f"""
            SELECT timestamp, open, high, low, close, volume
            FROM "{symbol}"
            ORDER BY timestamp DESC
            LIMIT 10
        """)

        recent_result = await nsedata_db.execute(recent_query)
        recent_data = recent_result.fetchall()

        return {
            "symbol": symbol,
            "data_quality": {
                "total_records": stats[0] if stats else 0,
                "data_start": stats[1].isoformat() if stats and stats[1] else None,
                "data_end": stats[2].isoformat() if stats and stats[2] else None,
                "coverage_days": (stats[2] - stats[1]).days if stats and stats[1] and stats[2] else 0
            },
            "price_stats": {
                "avg_price": round(float(stats[3]), 2) if stats and stats[3] else 0,
                "min_price": round(float(stats[4]), 2) if stats and stats[4] else 0,
                "max_price": round(float(stats[5]), 2) if stats and stats[5] else 0,
                "avg_volume": int(stats[6]) if stats and stats[6] else 0
            },
            "recent_data": [
                {
                    "timestamp": row[0].isoformat(),
                    "ohlc": {
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4])
                    },
                    "volume": int(row[5])
                }
                for row in recent_data
            ],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found or data unavailable")


# ============================================================================
# CORE BACKTESTING ENDPOINTS
# ============================================================================

@sip_router.post("/backtest", response_model=List[Dict])
async def run_sip_backtest(
        request: SIPBacktestRequest,
        background_tasks: BackgroundTasks,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db),
        nsedata_db: AsyncSession = Depends(get_nsedata_db),
        enable_monthly_limits: bool = True,
):
    """
    Enhanced SIP strategy backtest with benchmark comparison

    NEW FEATURES:
    1. Includes regular SIP benchmark (₹5000 on 15th of every month)
    2. Monthly investment limits and price thresholds
    3. Comprehensive comparison metrics
    """
    try:
        # Create enhanced strategy
        if enable_monthly_limits:
            strategy = EnhancedSIPStrategyWithLimits(
                nsedata_session=nsedata_db,
                trading_session=trading_db
            )
        else:
            strategy = EnhancedSIPStrategy(
                nsedata_session=nsedata_db,
                trading_session=trading_db
            )

        # Create benchmark calculator
        benchmark_calculator = BenchmarkSIPCalculator(
            monthly_amount=request.config.fixed_investment,
            investment_day=15
        )

        logger.info(f"🚀 Starting enhanced SIP backtest with benchmark for user {user_id}")
        logger.info(f"📊 Symbols: {request.symbols}")
        logger.info(f"📅 Period: {request.start_date} to {request.end_date}")
        logger.info(f"💰 Monthly limit: ₹{request.config.max_amount_in_a_month:,.2f}")
        logger.info(f"📉 Price threshold: {request.config.price_reduction_threshold}%")

        results = []

        for symbol in request.symbols:
            try:
                # Run strategy backtest
                strategy_result = await strategy.run_backtest(
                    symbol,
                    request.start_date,
                    request.end_date,
                    request.config
                )

                # Run benchmark calculation
                benchmark_result = await benchmark_calculator.calculate_benchmark(
                    symbol,
                    request.start_date,
                    request.end_date,
                    nsedata_db
                )

                if strategy_result and benchmark_result:
                    # Combine strategy and benchmark results
                    combined_result = {
                        **strategy_result,
                        'benchmark': benchmark_result,
                        'comparison': _calculate_comparison_metrics(strategy_result, benchmark_result)
                    }

                    results.append(combined_result)
                    logger.info(f"✅ Completed enhanced backtest with benchmark for {symbol}")
                else:
                    logger.warning(f"⚠️ No data or trades for {symbol}")

            except Exception as symbol_error:
                logger.error(f"❌ Error processing {symbol}: {symbol_error}")
                continue

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No valid results generated for any of the specified symbols"
            )

        # Save results to database (background task)
        background_tasks.add_task(
            save_enhanced_backtest_results,
            results, user_id, request, trading_db
        )

        logger.info(f"✅ Enhanced backtest with benchmark completed for {len(results)} symbols")
        return results

    except Exception as e:
        logger.error(f"Enhanced backtest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


async def save_enhanced_backtest_results(
        results: List[Dict],
        user_id: str,
        request: SIPBacktestRequest,
        trading_db: AsyncSession
):
    """Save enhanced backtest results with benchmark data"""
    try:
        # Create enhanced results table with benchmark columns
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS sip_backtest_results (
                backtest_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                strategy_name VARCHAR NOT NULL,
                total_investment FLOAT NOT NULL,
                final_portfolio_value FLOAT NOT NULL,
                total_return_percent FLOAT NOT NULL,
                cagr FLOAT NOT NULL,
                num_trades INTEGER NOT NULL,
                num_skipped INTEGER DEFAULT 0,
                monthly_limit_exceeded INTEGER DEFAULT 0,
                price_threshold_skipped INTEGER DEFAULT 0,
                max_amount_in_a_month FLOAT NOT NULL,
                price_reduction_threshold FLOAT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                config_used JSONB NOT NULL,
                trades JSONB,
                skipped_investments JSONB,
                monthly_summary JSONB,
                benchmark_data JSONB,
                comparison_metrics JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        await trading_db.execute(create_table_query)

        # Insert results with benchmark data
        for result in results:
            backtest_id = f"bt_{uuid.uuid4()}_{int(datetime.now().timestamp())}"

            # Prepare data for insertion
            insert_query = text("""
                INSERT INTO sip_backtest_results (
                    backtest_id, user_id, symbol, strategy_name, total_investment,
                    final_portfolio_value, total_return_percent, cagr, num_trades,
                    num_skipped, monthly_limit_exceeded, price_threshold_skipped,
                    max_amount_in_a_month, price_reduction_threshold,
                    start_date, end_date, config_used, trades, skipped_investments,
                    monthly_summary, benchmark_data, comparison_metrics
                ) VALUES (
                    :backtest_id, :user_id, :symbol, :strategy_name, :total_investment,
                    :final_portfolio_value, :total_return_percent, :cagr, :num_trades,
                    :num_skipped, :monthly_limit_exceeded, :price_threshold_skipped,
                    :max_amount_in_a_month, :price_reduction_threshold,
                    :start_date, :end_date, :config_used, :trades, :skipped_investments,
                    :monthly_summary, :benchmark_data, :comparison_metrics
                )
            """)

            # Convert date strings to date objects
            start_date_obj = datetime.strptime(request.start_date, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(request.end_date, '%Y-%m-%d').date()

            await trading_db.execute(insert_query, {
                'backtest_id': backtest_id,
                'user_id': user_id,
                'symbol': result['symbol'],
                'strategy_name': result['strategy_name'],
                'total_investment': float(result['total_investment']),
                'final_portfolio_value': float(result['final_portfolio_value']),
                'total_return_percent': float(result['total_return_percent']),
                'cagr': float(result['cagr_percent']) / 100,
                'num_trades': int(result['num_trades']),
                'num_skipped': int(result.get('num_skipped', 0)),
                'monthly_limit_exceeded': int(result.get('monthly_limit_exceeded', 0)),
                'price_threshold_skipped': int(result.get('price_threshold_skipped', 0)),
                'max_amount_in_a_month': float(request.config.max_amount_in_a_month),
                'price_reduction_threshold': float(request.config.price_reduction_threshold),
                'start_date': start_date_obj,
                'end_date': end_date_obj,
                'config_used': json.dumps(request.config.dict()),
                'trades': json.dumps(result.get('trades', [])),
                'skipped_investments': json.dumps(result.get('skipped_investments', [])),
                'monthly_summary': json.dumps(result.get('monthly_summary', {})),
                'benchmark_data': json.dumps(result.get('benchmark', {})),
                'comparison_metrics': json.dumps(result.get('comparison', {}))
            })

        await trading_db.commit()
        logger.info(f"✅ Saved enhanced backtest results with benchmark data for {len(results)} symbols")

    except Exception as e:
        logger.error(f"Error saving enhanced backtest results: {e}")
        await trading_db.rollback()
        raise

@sip_router.get("/backtest/history", response_model=List[Dict])
async def get_sip_backtest_history(
        limit: int = 10,
        offset: int = 0,
        symbol: Optional[str] = None,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get enhanced SIP backtest history with monthly limit analytics"""
    try:
        base_query = """
            SELECT backtest_id, symbol, strategy_name, total_investment, 
                   final_portfolio_value, total_return_percent, cagr, 
                   num_trades, num_skipped, monthly_limit_exceeded, 
                   price_threshold_skipped, max_amount_in_a_month,
                   price_reduction_threshold, monthly_summary, created_at
            FROM sip_backtest_results 
            WHERE user_id = :user_id
        """
        params = {"user_id": user_id}

        if symbol:
            base_query += " AND symbol = :symbol"
            params["symbol"] = symbol

        base_query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        params.update({"limit": limit, "offset": offset})

        result = await trading_db.execute(text(base_query), params)
        backtest_results = result.fetchall()

        enhanced_results = []
        for row in backtest_results:
            try:
                monthly_summary = safe_json_parse(row[13])
            except (json.JSONDecodeError, TypeError):
                monthly_summary = {}

            # Calculate additional analytics
            total_months = len(monthly_summary)
            avg_monthly_utilization = 0
            if total_months > 0 and row[11] > 0:  # max_amount_in_a_month > 0
                utilizations = [
                    (month_data.get('total_invested', 0) / row[11]) * 100
                    for month_data in monthly_summary.values()
                    if isinstance(month_data, dict)
                ]
                avg_monthly_utilization = sum(utilizations) / len(utilizations) if utilizations else 0

            enhanced_result = {
                "backtest_id": row[0],
                "symbol": row[1],
                "strategy_name": row[2],
                "total_investment": float(row[3]) if row[3] else 0,
                "final_portfolio_value": float(row[4]) if row[4] else 0,
                "total_return_percent": float(row[5]) if row[5] else 0,
                "cagr_percent": float(row[6]) * 100 if row[6] else 0,
                "num_trades": int(row[7]) if row[7] else 0,
                "num_skipped": int(row[8]) if row[8] else 0,
                "monthly_limit_exceeded": int(row[9]) if row[9] else 0,
                "price_threshold_skipped": int(row[10]) if row[10] else 0,
                "max_amount_in_a_month": float(row[11]) if row[11] else 0,
                "price_reduction_threshold": float(row[12]) if row[12] else 4.0,
                "monthly_analytics": {
                    "total_months": total_months,
                    "avg_monthly_utilization_percent": round(avg_monthly_utilization, 2),
                    "months_with_limits_hit": sum(1 for data in monthly_summary.values()
                                                  if
                                                  isinstance(data, dict) and data.get('total_invested', 0) >= row[11])
                },
                "created_at": row[14].isoformat() if row[14] else None
            }
            enhanced_results.append(enhanced_result)

        return enhanced_results

    except Exception as e:
        logger.error(f"Error fetching enhanced backtest history: {e}")
        return []


async def calculate_enhanced_opportunity_cost(
        symbol: str,
        skipped_investments: List[Dict],
        trades: List[Dict],
        start_date: date,
        end_date: date,
        user_id: str,
        trading_db: AsyncSession
) -> Dict[str, float]:
    """
    Enhanced opportunity cost calculation with multiple fallback methods
    """
    try:
        total_skipped_amount = sum(
            skip.get('intended_amount', 0)
            for skip in skipped_investments
            if isinstance(skip, dict)
        )

        if total_skipped_amount <= 0:
            return None

        # Method 1: Try to get final price from trades
        final_price_from_trades = None
        if isinstance(trades, list) and trades:
            for trade in reversed(trades):  # Start from the last trade
                if isinstance(trade, dict) and trade.get('price', 0) > 0:
                    final_price_from_trades = float(trade['price'])
                    break

        # Method 2: Fetch current/recent market price from database
        final_price_from_db = await get_final_price_from_market_data(
            symbol, end_date, trading_db
        )

        # Method 3: Use average price from successful trades as fallback
        avg_trade_price = None
        if isinstance(trades, list) and trades:
            valid_prices = [
                float(trade['price'])
                for trade in trades
                if isinstance(trade, dict) and trade.get('price', 0) > 0
            ]
            if valid_prices:
                avg_trade_price = sum(valid_prices) / len(valid_prices)

        # Select the best final price (priority order)
        final_price = (
                final_price_from_trades or
                final_price_from_db or
                avg_trade_price or
                0
        )

        if final_price <= 0:
            logger.warning(f"Could not determine final price for {symbol}, skipping opportunity cost calculation")
            return {
                "total_skipped_amount": total_skipped_amount,
                "estimated_final_value": 0,
                "opportunity_cost": -total_skipped_amount,
                "opportunity_cost_percent": -100,
                "final_price_used": 0,
                "price_source": "unavailable",
                "calculation_method": "total_loss_assumed"
            }

        # Calculate estimated units that would have been purchased
        estimated_units = 0
        skipped_with_prices = 0
        skipped_without_prices = 0

        for skip in skipped_investments:
            if isinstance(skip, dict):
                intended_amount = skip.get('intended_amount', 0)
                skip_price = skip.get('current_price', 0) or skip.get('price', 0)

                if skip_price > 0:
                    estimated_units += intended_amount / skip_price
                    skipped_with_prices += 1
                else:
                    # For skipped investments without price, use average price from trades
                    if avg_trade_price and avg_trade_price > 0:
                        estimated_units += intended_amount / avg_trade_price
                    skipped_without_prices += 1

        # Calculate final estimated value
        estimated_final_value = estimated_units * final_price
        opportunity_cost = estimated_final_value - total_skipped_amount
        opportunity_cost_percent = (opportunity_cost / total_skipped_amount) * 100 if total_skipped_amount > 0 else 0

        # Determine price source for transparency
        if final_price_from_trades:
            price_source = "final_trade"
        elif final_price_from_db:
            price_source = "market_data"
        elif avg_trade_price:
            price_source = "average_trade_price"
        else:
            price_source = "unknown"

        return {
            "total_skipped_amount": round(total_skipped_amount, 2),
            "estimated_final_value": round(estimated_final_value, 2),
            "opportunity_cost": round(opportunity_cost, 2),
            "opportunity_cost_percent": round(opportunity_cost_percent, 2),
            "estimated_units": round(estimated_units, 4),
            "final_price_used": round(final_price, 2),
            "price_source": price_source,
            "skipped_investments_count": len(skipped_investments),
            "skipped_with_prices": skipped_with_prices,
            "skipped_without_prices": skipped_without_prices,
            "calculation_method": "enhanced_multi_source"
        }

    except Exception as e:
        logger.error(f"Error calculating enhanced opportunity cost: {e}")
        return {
            "total_skipped_amount": total_skipped_amount,
            "estimated_final_value": 0,
            "opportunity_cost": -total_skipped_amount,
            "opportunity_cost_percent": -100,
            "error": str(e),
            "calculation_method": "error_fallback"
        }


async def get_final_price_from_market_data(
        symbol: str,
        end_date: date,
        trading_db: AsyncSession
) -> Optional[float]:
    """
    Get the final/recent price from market data as fallback
    """
    try:
        # Try to get price from the market data around the end date
        from backend.app.database import get_nsedata_db

        # Get nsedata session
        nsedata_db = None
        try:
            # This is a simplified approach - you might need to adjust based on your setup
            nsedata_db = get_nsedata_db()
        except:
            return None

        if not nsedata_db:
            return None

        # Query for price around the end date
        price_query = text(f"""
            SELECT close
            FROM "{symbol.upper()}"
            WHERE timestamp <= :end_date
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        async with nsedata_db() as session:
            result = await session.execute(price_query, {"end_date": end_date})
            price_row = result.fetchone()

            if price_row and price_row[0]:
                return float(price_row[0])

    except Exception as e:
        logger.warning(f"Could not fetch final price from market data for {symbol}: {e}")

    return None


# Additional helper function to validate skipped investments data
def validate_skipped_investments_data(skipped_investments: List[Dict]) -> Dict[str, Any]:
    """
    Validate and analyze the structure of skipped investments data
    """
    if not isinstance(skipped_investments, list):
        return {"valid": False, "reason": "Not a list"}

    total_count = len(skipped_investments)
    valid_count = 0
    with_price_count = 0
    without_price_count = 0

    for skip in skipped_investments:
        if isinstance(skip, dict):
            valid_count += 1
            if skip.get('current_price', 0) > 0 or skip.get('price', 0) > 0:
                with_price_count += 1
            else:
                without_price_count += 1

    return {
        "valid": True,
        "total_count": total_count,
        "valid_dict_count": valid_count,
        "with_price_count": with_price_count,
        "without_price_count": without_price_count,
        "data_quality": "good" if with_price_count > total_count * 0.7 else "poor"
    }

@sip_router.get("/analytics/monthly-limits/{symbol}")
async def get_monthly_limits_analytics(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get detailed analytics for monthly investment limits for a specific symbol - ENHANCED VERSION"""
    try:
        # Base query for getting backtest results with monthly data
        query = """
            SELECT backtest_id, total_investment, final_portfolio_value,
                   max_amount_in_a_month, monthly_limit_exceeded, 
                   price_threshold_skipped, monthly_summary, trades,
                   skipped_investments, created_at, start_date, end_date
            FROM sip_backtest_results 
            WHERE user_id = :user_id AND symbol = :symbol
        """
        params = {"user_id": user_id, "symbol": symbol}

        if start_date:
            query += " AND created_at >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND created_at <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY created_at DESC LIMIT 10"

        result = await trading_db.execute(text(query), params)
        backtest_data = result.fetchall()

        if not backtest_data:
            return {
                "symbol": symbol,
                "message": "No backtest data found for the specified criteria",
                "analytics": {}
            }

        # Aggregate analytics across all backtests
        total_backtests = len(backtest_data)
        total_monthly_limits_hit = sum(row[4] for row in backtest_data)
        total_price_threshold_skips = sum(row[5] for row in backtest_data)

        monthly_utilization_data = []
        opportunity_cost_analysis = []

        for row in backtest_data:
            # Use safe_json_parse for all JSON fields
            monthly_summary = safe_json_parse(row[6])  # monthly_summary
            trades = safe_json_parse(row[7])  # trades
            skipped_investments = safe_json_parse(row[8])  # skipped_investments

            # Calculate monthly utilization
            for month, data in monthly_summary.items():
                if isinstance(data, dict):
                    utilization_percent = (data.get('total_invested', 0) / row[3]) * 100 if row[3] > 0 else 0
                    monthly_utilization_data.append({
                        "month": month,
                        "utilization_percent": round(utilization_percent, 2),
                        "amount_invested": data.get('total_invested', 0),
                        "monthly_limit": row[3],
                        "num_investments": data.get('num_investments', 0)
                    })

            # ENHANCED OPPORTUNITY COST CALCULATION
            if isinstance(skipped_investments, list) and skipped_investments:
                opportunity_cost_result = await calculate_enhanced_opportunity_cost(
                    symbol, skipped_investments, trades, row[10], row[11], user_id, trading_db
                )

                if opportunity_cost_result:
                    opportunity_cost_analysis.append({
                        "backtest_id": row[0],
                        **opportunity_cost_result
                    })

        # Calculate overall statistics
        avg_monthly_utilization = (
                sum(data["utilization_percent"] for data in monthly_utilization_data) /
                len(monthly_utilization_data)
        ) if monthly_utilization_data else 0

        avg_opportunity_cost = (
                sum(analysis["opportunity_cost_percent"] for analysis in opportunity_cost_analysis) /
                len(opportunity_cost_analysis)
        ) if opportunity_cost_analysis else 0

        analytics = {
            "symbol": symbol,
            "analysis_period": f"{start_date or 'inception'} to {end_date or 'latest'}",
            "overall_statistics": {
                "total_backtests_analyzed": total_backtests,
                "total_monthly_limits_hit": total_monthly_limits_hit,
                "total_price_threshold_skips": total_price_threshold_skips,
                "avg_monthly_utilization_percent": round(avg_monthly_utilization, 2),
                "avg_opportunity_cost_percent": round(avg_opportunity_cost, 2)
            },
            "monthly_utilization_trend": monthly_utilization_data[-12:],  # Last 12 months
            "opportunity_cost_analysis": opportunity_cost_analysis,
            "recommendations": _generate_monthly_limit_recommendations(
                avg_monthly_utilization, total_monthly_limits_hit, total_price_threshold_skips
            ),
            "timestamp": datetime.now().isoformat()
        }

        return analytics

    except Exception as e:
        logger.error(f"Error generating monthly limits analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_monthly_limit_recommendations(avg_utilization: float,
                                            limits_hit: int,
                                            price_skips: int) -> List[str]:
    """Generate recommendations based on monthly limit analytics"""
    recommendations = []

    if avg_utilization < 50:
        recommendations.append(
            f"Low monthly utilization ({avg_utilization:.1f}%). Consider reducing monthly limits or increasing fixed investment."
        )
    elif avg_utilization > 90:
        recommendations.append(
            f"High monthly utilization ({avg_utilization:.1f}%). Consider increasing monthly limits for more opportunities."
        )

    if limits_hit > 5:
        recommendations.append(
            f"Monthly limits hit {limits_hit} times. Consider increasing max_amount_in_a_month for better opportunity capture."
        )

    if price_skips > 10:
        recommendations.append(
            f"Price threshold caused {price_skips} skips. Consider lowering price_reduction_threshold for more frequent investments."
        )
    elif price_skips < 2:
        recommendations.append(
            "Very few price threshold skips. Consider increasing price_reduction_threshold for more selective investing."
        )

    if not recommendations:
        recommendations.append("Configuration appears well-balanced for current market conditions.")

    return recommendations


# Additional function to handle config_used parsing
def safe_parse_config(config_data):
    """Safely parse config data that might already be a Python object"""
    if config_data is None:
        return {}
    elif isinstance(config_data, dict):
        return config_data
    elif isinstance(config_data, str):
        try:
            return json.loads(config_data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse config JSON: {config_data}")
            return {}
    else:
        logger.warning(f"Unexpected config data type: {type(config_data)}")
        return {}

# Fixed optimize config function
@sip_router.post("/optimize-config")
async def optimize_sip_config(
        symbol: str,
        target_monthly_utilization: float = 80.0,
        risk_tolerance: str = "moderate",  # conservative, moderate, aggressive
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """
    Optimize SIP configuration based on historical performance and user preferences - FIXED VERSION
    """
    try:
        # Get historical performance data
        query = """
            SELECT total_investment, final_portfolio_value, monthly_limit_exceeded,
                   price_threshold_skipped, max_amount_in_a_month, price_reduction_threshold,
                   config_used, monthly_summary
            FROM sip_backtest_results 
            WHERE user_id = :user_id AND symbol = :symbol
            ORDER BY created_at DESC LIMIT 5
        """

        result = await trading_db.execute(text(query), {"user_id": user_id, "symbol": symbol})
        historical_data = result.fetchall()

        if not historical_data:
            # Return default optimized config for new users
            return _get_default_optimized_config(risk_tolerance)

        # Analyze historical performance with fixed JSON parsing
        analysis = _analyze_historical_performance(historical_data, target_monthly_utilization)

        # Generate optimized configuration
        optimized_config = _generate_optimized_config(analysis, risk_tolerance)

        return {
            "symbol": symbol,
            "risk_tolerance": risk_tolerance,
            "target_monthly_utilization": target_monthly_utilization,
            "analysis_summary": analysis,
            "optimized_config": optimized_config,
            "expected_improvements": _calculate_expected_improvements(analysis, optimized_config),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error optimizing SIP config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _analyze_historical_performance(historical_data: List, target_utilization: float) -> Dict:
    """Analyze historical performance to identify optimization opportunities"""
    total_returns = []
    monthly_utilizations = []
    limit_hits = 0
    price_skips = 0

    for row in historical_data:
        # Calculate return
        if row[0] > 0:  # total_investment > 0
            return_pct = ((row[1] / row[0]) - 1) * 100
            total_returns.append(return_pct)

        # Analyze monthly utilization
        monthly_summary = safe_json_parse(row[7])
        for month_data in monthly_summary.values():
            if row[4] > 0:  # max_amount_in_a_month > 0
                utilization = (month_data.get('total_invested', 0) / row[4]) * 100
                monthly_utilizations.append(utilization)

        limit_hits += row[2]  # monthly_limit_exceeded
        price_skips += row[3]  # price_threshold_skipped

    avg_return = sum(total_returns) / len(total_returns) if total_returns else 0
    avg_utilization = sum(monthly_utilizations) / len(monthly_utilizations) if monthly_utilizations else 0

    return {
        "avg_return_percent": round(avg_return, 2),
        "avg_monthly_utilization": round(avg_utilization, 2),
        "total_limit_hits": limit_hits,
        "total_price_skips": price_skips,
        "utilization_gap": target_utilization - avg_utilization,
        "performance_issues": _identify_performance_issues(avg_utilization, limit_hits, price_skips)
    }


def _identify_performance_issues(avg_utilization: float, limit_hits: int, price_skips: int) -> List[str]:
    """Identify specific performance issues"""
    issues = []

    if avg_utilization < 50:
        issues.append("underutilized_monthly_budget")
    elif avg_utilization > 95:
        issues.append("budget_constraints")

    if limit_hits > 10:
        issues.append("frequent_limit_hits")

    if price_skips > 15:
        issues.append("excessive_price_filtering")
    elif price_skips < 2:
        issues.append("insufficient_price_filtering")

    return issues


def _generate_optimized_config(analysis: Dict, risk_tolerance: str) -> Dict:
    """Generate optimized configuration based on analysis and risk tolerance"""
    base_config = SIPConfigRequest()

    # Risk tolerance multipliers
    risk_multipliers = {
        "conservative": {"monthly": 3.0, "price_threshold": 5.0, "multipliers": [1.5, 2.0, 2.5]},
        "moderate": {"monthly": 4.0, "price_threshold": 4.0, "multipliers": [2.0, 3.0, 4.0]},
        "aggressive": {"monthly": 5.0, "price_threshold": 3.0, "multipliers": [2.5, 4.0, 6.0]}
    }

    risk_params = risk_multipliers.get(risk_tolerance, risk_multipliers["moderate"])

    # Adjust based on analysis
    monthly_multiplier = risk_params["monthly"]
    price_threshold = risk_params["price_threshold"]

    # Adjust monthly limit based on utilization
    if "underutilized_monthly_budget" in analysis["performance_issues"]:
        monthly_multiplier *= 0.8  # Reduce monthly limit
    elif "budget_constraints" in analysis["performance_issues"]:
        monthly_multiplier *= 1.3  # Increase monthly limit

    # Adjust price threshold based on skipping behavior
    if "excessive_price_filtering" in analysis["performance_issues"]:
        price_threshold *= 0.8  # Lower threshold for more investments
    elif "insufficient_price_filtering" in analysis["performance_issues"]:
        price_threshold *= 1.2  # Higher threshold for selectivity

    optimized_config = {
        "fixed_investment": base_config.fixed_investment,
        "max_amount_in_a_month": base_config.fixed_investment * monthly_multiplier,
        "price_reduction_threshold": round(price_threshold, 1),
        "drawdown_threshold_1": -12.0 if risk_tolerance == "conservative" else -8.0 if risk_tolerance == "moderate" else -5.0,
        "drawdown_threshold_2": -6.0 if risk_tolerance == "conservative" else -4.0 if risk_tolerance == "moderate" else -2.0,
        "investment_multiplier_1": risk_params["multipliers"][0],
        "investment_multiplier_2": risk_params["multipliers"][1],
        "investment_multiplier_3": risk_params["multipliers"][2],
        "rolling_window": base_config.rolling_window,
        "fallback_day": base_config.fallback_day,
        "min_investment_gap_days": 7 if risk_tolerance == "conservative" else 5 if risk_tolerance == "moderate" else 3
    }

    return optimized_config


def _get_default_optimized_config(risk_tolerance: str) -> Dict:
    """Get default optimized config for new users"""
    base_config = SIPConfigRequest()
    optimized = _generate_optimized_config(
        {"performance_issues": [], "utilization_gap": 0},
        risk_tolerance
    )

    return {
        "symbol": "N/A",
        "risk_tolerance": risk_tolerance,
        "analysis_summary": {"message": "No historical data available - using defaults"},
        "optimized_config": optimized,
        "expected_improvements": ["Optimized for selected risk tolerance"],
        "timestamp": datetime.now().isoformat()
    }


def _calculate_expected_improvements(analysis: Dict, optimized_config: Dict) -> List[str]:
    """Calculate expected improvements from optimized configuration"""
    improvements = []

    if "underutilized_monthly_budget" in analysis["performance_issues"]:
        improvements.append(
            f"Reduced monthly limit to ₹{optimized_config['max_amount_in_a_month']:,.0f} for better capital efficiency")

    if "budget_constraints" in analysis["performance_issues"]:
        improvements.append(
            f"Increased monthly limit to ₹{optimized_config['max_amount_in_a_month']:,.0f} for more opportunities")

    if "excessive_price_filtering" in analysis["performance_issues"]:
        improvements.append(
            f"Lowered price threshold to {optimized_config['price_reduction_threshold']}% for more frequent investments")

    if "insufficient_price_filtering" in analysis["performance_issues"]:
        improvements.append(
            f"Raised price threshold to {optimized_config['price_reduction_threshold']}% for better selectivity")

    if not improvements:
        improvements.append("Configuration fine-tuned based on risk tolerance")

    improvements.append(
        f"Expected monthly utilization: {75 + (5 if 'aggressive' in str(optimized_config) else 0)}%-85%")

    return improvements


# ============================================================================
# BATCH PROCESSING AND UTILITIES
# ============================================================================

@sip_router.post("/batch-backtest/multi-configs")
async def run_backtest_with_limits_multiple_configs(
        symbols: List[str],
        configs: List[SIPConfigRequest],
        start_date: str,
        end_date: str,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db),
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """
    Run batch backtests with different configurations and benchmark comparison
    """
    try:
        strategy = EnhancedSIPStrategyWithLimits(
            nsedata_session=nsedata_db,
            trading_session=trading_db
        )

        benchmark_calculator = BenchmarkSIPCalculator()

        batch_results = []

        # Calculate benchmark once for all configs
        benchmark_results = {}
        for symbol in symbols:
            benchmark_results[symbol] = await benchmark_calculator.calculate_benchmark(
                symbol, start_date, end_date, nsedata_db
            )

        for config in configs:
            config_results = []

            for symbol in symbols:
                try:
                    result = await strategy.run_backtest(symbol, start_date, end_date, config)
                    if result:
                        # Add benchmark comparison
                        if symbol in benchmark_results:
                            result['benchmark'] = benchmark_results[symbol]
                            result['comparison'] = _calculate_comparison_metrics(
                                result, benchmark_results[symbol]
                            )
                        config_results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch backtest for {symbol}: {e}")

            batch_results.append({
                "config": config.dict(),
                "results": config_results,
                "summary": _calculate_batch_summary(config_results),
                "benchmark_summary": _calculate_benchmark_summary(benchmark_results)
            })

        return {
            "batch_id": f"batch_{uuid.uuid4()}_{int(datetime.now().timestamp())}",
            "symbols": symbols,
            "period": f"{start_date} to {end_date}",
            "configurations_tested": len(configs),
            "batch_results": batch_results,
            "best_config_recommendation": _recommend_best_config(batch_results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_batch_summary(results: List[Dict]) -> Dict:
    """Calculate summary statistics for a batch of results"""
    if not results:
        return {}

    total_investment = sum(r['total_investment'] for r in results)
    total_value = sum(r['final_portfolio_value'] for r in results)
    avg_return = sum(r['total_return_percent'] for r in results) / len(results)
    total_trades = sum(r['num_trades'] for r in results)
    total_skipped = sum(r.get('num_skipped', 0) for r in results)

    return {
        "total_symbols": len(results),
        "total_investment": total_investment,
        "total_portfolio_value": total_value,
        "overall_return_percent": ((total_value / total_investment) - 1) * 100 if total_investment > 0 else 0,
        "average_return_percent": avg_return,
        "total_trades": total_trades,
        "total_skipped": total_skipped
    }


def _calculate_benchmark_summary(benchmark_results: Dict) -> Dict:
    """Calculate summary statistics for benchmark results"""
    if not benchmark_results:
        return {}

    valid_results = [r for r in benchmark_results.values() if r.get('total_investment', 0) > 0]
    if not valid_results:
        return {}

    total_investment = sum(r['total_investment'] for r in valid_results)
    total_value = sum(r['final_portfolio_value'] for r in valid_results)
    avg_return = sum(r['total_return_percent'] for r in valid_results) / len(valid_results)
    total_trades = sum(r['num_trades'] for r in valid_results)

    return {
        "total_symbols": len(valid_results),
        "total_investment": total_investment,
        "total_portfolio_value": total_value,
        "overall_return_percent": ((total_value / total_investment) - 1) * 100 if total_investment > 0 else 0,
        "average_return_percent": avg_return,
        "total_trades": total_trades,
        "strategy_name": "Regular SIP Benchmark"
    }


def _recommend_best_config(batch_results: List[Dict]) -> Dict:
    """Recommend the best configuration based on performance vs benchmark"""
    if not batch_results:
        return {"recommendation": "No valid configurations to analyze"}

    best_config = None
    best_score = float('-inf')

    for batch_result in batch_results:
        results = batch_result.get('results', [])
        if not results:
            continue

        # Calculate average outperformance vs benchmark
        outperformances = []
        for result in results:
            comparison = result.get('comparison', {})
            outperformance = comparison.get('return_outperformance_percent', 0)
            outperformances.append(outperformance)

        if outperformances:
            avg_outperformance = sum(outperformances) / len(outperformances)

            # Score based on outperformance and consistency
            consistency_bonus = 1.0 if all(x >= 0 for x in outperformances) else 0.5
            score = avg_outperformance * consistency_bonus

            if score > best_score:
                best_score = score
                best_config = batch_result['config']

    if best_config:
        return {
            "recommended_config": best_config,
            "average_outperformance": best_score,
            "reason": f"Best average outperformance vs benchmark: {best_score:.2f}%"
        }
    else:
        return {"recommendation": "No configuration outperformed benchmark consistently"}


# ============================================================================
# PORTFOLIO MANAGEMENT ENDPOINTS
# ============================================================================

@sip_router.post("/portfolio/multi", response_model=Dict[str, str])
async def create_multi_symbol_portfolio(
        request: SIPMultiPortfolioRequest,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Create multi-symbol SIP portfolio"""
    try:
        # Create enhanced portfolios table
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS sip_portfolios (
                portfolio_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                portfolio_name VARCHAR NOT NULL,
                portfolio_type VARCHAR DEFAULT 'single',
                symbols JSONB,
                config JSONB NOT NULL,
                status VARCHAR DEFAULT 'active',
                total_invested FLOAT DEFAULT 0,
                current_units FLOAT DEFAULT 0,
                current_value FLOAT DEFAULT 0,
                next_investment_date DATE,
                auto_rebalance BOOLEAN DEFAULT FALSE,
                rebalance_frequency_days INTEGER DEFAULT 30,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await trading_db.execute(create_table_query)

        # Generate portfolio ID
        portfolio_id = f"pf_{user_id}_multi_{int(datetime.now().timestamp())}"

        # Prepare symbols data
        symbols_data = []
        for symbol_config in request.symbols:
            symbols_data.append({
                "symbol": symbol_config.symbol,
                "allocation_percentage": symbol_config.allocation_percentage,
                "config": symbol_config.config or {}
            })

        # Calculate next investment date
        next_investment_date = calculate_next_investment_date(
            datetime.now().date(),
            SIPConfig(**request.default_config.dict())
        )

        # Insert portfolio
        insert_query = text("""
            INSERT INTO sip_portfolios 
            (portfolio_id, user_id, portfolio_name, portfolio_type, symbols, config, 
             next_investment_date, auto_rebalance, rebalance_frequency_days, created_at)
            VALUES (:portfolio_id, :user_id, :portfolio_name, :portfolio_type, :symbols, 
                    :config, :next_investment_date, :auto_rebalance, :rebalance_frequency_days, :created_at)
        """)

        await trading_db.execute(insert_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id,
            'portfolio_name': request.portfolio_name,
            'portfolio_type': 'multi',
            'symbols': json.dumps(symbols_data),
            'config': json.dumps(request.default_config.dict()),
            'next_investment_date': next_investment_date,
            'auto_rebalance': request.auto_rebalance,
            'rebalance_frequency_days': request.rebalance_frequency_days,
            'created_at': datetime.now()
        })

        await trading_db.commit()

        logger.info(f"Created multi-symbol portfolio {portfolio_id} with {len(symbols_data)} symbols")

        return {
            "portfolio_id": portfolio_id,
            "status": "created",
            "portfolio_type": "multi",
            "symbols_count": str(len(symbols_data))
        }

    except Exception as e:
        logger.error(f"Error creating multi-symbol portfolio: {e}")
        await trading_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.post("/portfolio", response_model=Dict[str, str])
async def create_sip_portfolio(
        request: SIPPortfolioRequest,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Create single-symbol SIP portfolio"""
    try:
        # Create enhanced portfolios table
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS sip_portfolios (
                portfolio_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                portfolio_name VARCHAR,
                portfolio_type VARCHAR DEFAULT 'single',
                symbols JSONB,
                config JSONB NOT NULL,
                status VARCHAR DEFAULT 'active',
                total_invested FLOAT DEFAULT 0,
                current_units FLOAT DEFAULT 0,
                current_value FLOAT DEFAULT 0,
                next_investment_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await trading_db.execute(create_table_query)

        # Generate portfolio ID
        portfolio_id = f"pf_{user_id}_{request.symbol}_{int(datetime.now().timestamp())}"

        # Calculate next investment date
        config = SIPConfig(**request.config.dict())
        next_investment_date = calculate_next_investment_date(datetime.now().date(), config)

        # Prepare single symbol data
        symbols_data = [{
            "symbol": request.symbol,
            "allocation_percentage": 100.0,
            "config": {}
        }]

        # Insert portfolio
        insert_query = text("""
            INSERT INTO sip_portfolios 
            (portfolio_id, user_id, portfolio_name, portfolio_type, symbols, config, 
             next_investment_date, created_at)
            VALUES (:portfolio_id, :user_id, :portfolio_name, :portfolio_type, :symbols, 
                    :config, :next_investment_date, :created_at)
        """)

        await trading_db.execute(insert_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id,
            'portfolio_name': request.portfolio_name or f"{request.symbol} SIP Portfolio",
            'portfolio_type': 'single',
            'symbols': json.dumps(symbols_data),
            'config': json.dumps(config.__dict__),
            'next_investment_date': next_investment_date,
            'created_at': datetime.now()
        })

        await trading_db.commit()

        logger.info(f"Created single-symbol portfolio {portfolio_id} for {request.symbol}")

        return {"portfolio_id": portfolio_id, "status": "created"}

    except Exception as e:
        logger.error(f"Error creating SIP portfolio: {e}")
        await trading_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.get("/portfolio")
async def get_sip_portfolios(
        status: Optional[str] = None,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get user's SIP portfolios with enhanced information - FIXED VERSION"""
    try:
        base_query = """
            SELECT portfolio_id, portfolio_name, portfolio_type, symbols, status, 
                   total_invested, current_units, current_value, 
                   next_investment_date, auto_rebalance, created_at
            FROM sip_portfolios 
            WHERE user_id = :user_id
        """
        params = {"user_id": user_id}

        if status:
            base_query += " AND status = :status"
            params["status"] = status
        else:
            # Default to exclude deleted portfolios
            base_query += " AND status != 'deleted'"

        base_query += " ORDER BY created_at DESC"

        result = await trading_db.execute(text(base_query), params)
        portfolios = result.fetchall()

        enhanced_portfolios = []

        for row in portfolios:
            try:
                # CRITICAL FIX: Safe JSON parsing for symbols
                symbols_raw = row[3]  # symbols column

                if symbols_raw is None:
                    symbols_data = []
                elif isinstance(symbols_raw, str):
                    # It's a JSON string, parse it
                    try:
                        symbols_data = json.loads(symbols_raw)
                    except json.JSONDecodeError as json_err:
                        logger.error(f"JSON decode error for portfolio {row[0]} symbols: {json_err}")
                        symbols_data = []
                elif isinstance(symbols_raw, (list, dict)):
                    # It's already parsed (JSONB from PostgreSQL)
                    symbols_data = symbols_raw
                else:
                    logger.warning(f"Unexpected symbols data type for portfolio {row[0]}: {type(symbols_raw)}")
                    symbols_data = []

                # Ensure symbols_data is a list
                if not isinstance(symbols_data, list):
                    logger.warning(f"Symbols data is not a list for portfolio {row[0]}: {type(symbols_data)}")
                    symbols_data = []

                portfolio_info = {
                    "portfolio_id": row[0],
                    "portfolio_name": row[1],
                    "portfolio_type": row[2],
                    "symbols": symbols_data,
                    "status": row[4],
                    "total_invested": row[5] or 0,
                    "current_units": row[6] or 0,
                    "current_value": row[7] or 0,
                    "next_investment_date": row[8].isoformat() if row[8] else None,
                    "auto_rebalance": row[9] or False,
                    "created_at": row[10].isoformat() if row[10] else None,
                    "symbols_count": len(symbols_data)
                }

                enhanced_portfolios.append(portfolio_info)

            except Exception as portfolio_error:
                logger.error(f"Error processing portfolio {row[0]}: {portfolio_error}")
                # Skip this portfolio but continue processing others
                continue

        logger.info(f"✅ Successfully fetched {len(enhanced_portfolios)} portfolios for user {user_id}")
        return enhanced_portfolios

    except Exception as e:
        logger.error(f"Error fetching SIP portfolios: {e}")
        return []


@sip_router.put("/portfolio/{portfolio_id}/cancel")
async def cancel_sip_portfolio(
        portfolio_id: str,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Cancel SIP portfolio (mark as cancelled)"""
    try:
        # Update status to cancelled instead of deleted
        update_query = text("""
            UPDATE sip_portfolios 
            SET status = 'cancelled', updated_at = :now
            WHERE portfolio_id = :portfolio_id AND user_id = :user_id
        """)

        result = await trading_db.execute(update_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id,
            'now': datetime.now()
        })

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        await trading_db.commit()

        logger.info(f"Cancelled SIP portfolio {portfolio_id}")

        return {"status": "cancelled", "portfolio_id": portfolio_id}

    except Exception as e:
        logger.error(f"Error cancelling portfolio: {e}")
        await trading_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.delete("/portfolio/{portfolio_id}")
async def delete_sip_portfolio(
        portfolio_id: str,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Permanently delete SIP portfolio and all related data"""
    try:
        # Verify ownership first
        check_query = text("""
            SELECT portfolio_id FROM sip_portfolios 
            WHERE portfolio_id = :portfolio_id AND user_id = :user_id
        """)

        result = await trading_db.execute(check_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id
        })

        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Portfolio not found")

        # Delete related data first (due to foreign key constraints)
        delete_signals = text("""
            DELETE FROM sip_signals WHERE portfolio_id = :portfolio_id
        """)
        await trading_db.execute(delete_signals, {'portfolio_id': portfolio_id})

        delete_trades = text("""
            DELETE FROM sip_actual_trades WHERE portfolio_id = :portfolio_id
        """)
        await trading_db.execute(delete_trades, {'portfolio_id': portfolio_id})

        # Delete portfolio
        delete_portfolio = text("""
            DELETE FROM sip_portfolios WHERE portfolio_id = :portfolio_id
        """)
        result = await trading_db.execute(delete_portfolio, {'portfolio_id': portfolio_id})

        await trading_db.commit()

        logger.info(f"Permanently deleted portfolio {portfolio_id} and all related data")

        return {"status": "permanently_deleted", "portfolio_id": portfolio_id}

    except Exception as e:
        logger.error(f"Error deleting portfolio: {e}")
        await trading_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SIGNAL GENERATION ENDPOINTS
# ============================================================================

@sip_router.get("/signals/{portfolio_id}")
async def get_investment_signals(
        portfolio_id: str,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db),
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Get investment signals for ALL symbols in a SIP portfolio - FIXED FOR MULTI-SYMBOL"""
    try:
        # Verify portfolio ownership
        portfolio_query = text("""
            SELECT symbols, config, portfolio_type FROM sip_portfolios 
            WHERE portfolio_id = :portfolio_id AND user_id = :user_id
        """)

        result = await trading_db.execute(portfolio_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id
        })
        portfolio_data = result.fetchone()

        if not portfolio_data:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        symbols_json, config_json, portfolio_type = portfolio_data

        # CRITICAL FIX: Safe JSON parsing with type checking
        try:
            # Parse symbols safely
            if isinstance(symbols_json, str):
                symbols_data = json.loads(symbols_json)
            elif isinstance(symbols_json, list):
                symbols_data = symbols_json  # Already parsed
            else:
                logger.error(f"Unexpected symbols_json type: {type(symbols_json)}")
                raise ValueError(f"Invalid symbols data type: {type(symbols_json)}")

            # Parse config safely
            if isinstance(config_json, str):
                config_dict = json.loads(config_json)
            elif isinstance(config_json, dict):
                config_dict = config_json  # Already parsed
            else:
                logger.error(f"Unexpected config_json type: {type(config_json)}")
                raise ValueError(f"Invalid config data type: {type(config_json)}")

            config = SIPConfig(**config_dict)

            logger.info(f"Successfully parsed portfolio data for {portfolio_id}")
            logger.info(
                f"Portfolio type: {portfolio_type}, Symbols count: {len(symbols_data) if isinstance(symbols_data, list) else 'N/A'}")

        except Exception as parse_error:
            logger.error(f"JSON parsing error for portfolio {portfolio_id}: {parse_error}")
            raise HTTPException(status_code=500, detail=f"Portfolio data parsing failed: {str(parse_error)}")

        # Validate symbols_data structure
        if not isinstance(symbols_data, list) or not symbols_data:
            logger.error(f"Invalid symbols_data structure: {symbols_data}")
            raise HTTPException(status_code=400, detail="Invalid or empty symbols in portfolio")

        # Create strategy instance with both sessions
        strategy = EnhancedSIPStrategy(
            nsedata_session=nsedata_db,
            trading_session=trading_db
        )

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # FIXED: Process ALL symbols in the portfolio, not just the first one
        all_signals = {}
        successful_signals = 0
        failed_signals = 0

        for symbol_config in symbols_data:
            try:
                # FIXED: Safe symbol extraction with validation
                if isinstance(symbol_config, dict) and 'symbol' in symbol_config:
                    symbol = symbol_config['symbol']
                    allocation_pct = symbol_config.get('allocation_percentage', 100.0 / len(symbols_data))
                elif isinstance(symbol_config, str):
                    symbol = symbol_config  # Direct string symbol
                    allocation_pct = 100.0 / len(symbols_data)  # Equal allocation
                else:
                    logger.error(f"Invalid symbol config structure: {symbol_config}")
                    continue

                if not symbol or not isinstance(symbol, str):
                    logger.warning(f"Skipping invalid symbol: {symbol}")
                    continue

                logger.info(f"Processing signals for symbol: {symbol} ({allocation_pct}% allocation)")

                # Generate signals for this symbol with timeout protection
                try:
                    logger.debug(f"Fetching data for {symbol} from {start_date} to {end_date}")

                    # Fetch data with timeout protection
                    data = await asyncio.wait_for(
                        strategy.fetch_data_from_db_async(symbol, start_date, end_date),
                        timeout=30.0
                    )

                    if data.empty:
                        logger.warning(f"No data available for {symbol}")
                        all_signals[symbol] = {
                            "signal": "NO_DATA",
                            "confidence": 0,
                            "message": f"No recent data available for {symbol}",
                            "symbol": symbol,
                            "allocation_percentage": allocation_pct,
                            "analysis_timestamp": datetime.now().isoformat()
                        }
                        failed_signals += 1
                        continue

                    logger.debug(f"Fetched {len(data)} data points for {symbol}")

                    # Generate signals with timeout protection
                    signals = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, strategy.get_next_investment_signals, data, config
                        ),
                        timeout=45.0
                    )

                    # Validate signals response
                    if not isinstance(signals, dict):
                        raise ValueError(f"Invalid signals response type: {type(signals)}")

                    # Add metadata to signals
                    signals['symbol'] = symbol
                    signals['allocation_percentage'] = allocation_pct
                    signals['portfolio_id'] = portfolio_id
                    signals['data_points_used'] = len(data)

                    # Adjust recommended amount based on allocation
                    if 'recommended_amount' in signals:
                        base_amount = signals['recommended_amount']
                        allocated_amount = (base_amount * allocation_pct) / 100.0
                        signals['allocated_amount'] = allocated_amount
                        signals['base_amount'] = base_amount

                    all_signals[symbol] = signals
                    successful_signals += 1

                    logger.info(
                        f"✅ Generated signals for {symbol}: {signals.get('signal', 'UNKNOWN')} (confidence: {signals.get('confidence', 0)})")

                    # Save signal to database if investment is recommended
                    try:
                        if signals.get('signal') not in ['NO_DATA', 'ERROR']:
                            await save_signal_with_gtt_order(portfolio_id, symbol, signals, trading_db, config)
                            logger.info(f"✅ Saved signal and GTT order for {symbol}")
                    except Exception as save_error:
                        logger.error(f"Error saving signal for {symbol}: {save_error}")
                        # Don't fail the entire request if saving fails
                        signals['save_warning'] = f"Signal generated but not saved: {str(save_error)}"

                except asyncio.TimeoutError:
                    logger.error(f"Timeout generating signals for {symbol}")
                    all_signals[symbol] = {
                        "signal": "ERROR",
                        "confidence": 0,
                        "message": f"Signal generation timeout for {symbol}",
                        "symbol": symbol,
                        "allocation_percentage": allocation_pct,
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                    failed_signals += 1
                except Exception as signal_error:
                    logger.error(f"Error generating signals for {symbol}: {signal_error}")
                    all_signals[symbol] = {
                        "signal": "ERROR",
                        "confidence": 0,
                        "message": f"Signal generation failed: {str(signal_error)}",
                        "symbol": symbol,
                        "allocation_percentage": allocation_pct,
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                    failed_signals += 1

            except Exception as symbol_processing_error:
                logger.error(f"Critical error processing symbol {symbol_config}: {symbol_processing_error}")
                failed_signals += 1
                continue

        # Generate portfolio-level summary
        portfolio_summary = {
            "portfolio_id": portfolio_id,
            "portfolio_type": portfolio_type or "single",
            "total_symbols": len(symbols_data),
            "successful_signals": successful_signals,
            "failed_signals": failed_signals,
            "overall_confidence": 0,
            "overall_signal": "NORMAL",
            "total_recommended_amount": 0,
            "analysis_timestamp": datetime.now().isoformat()
        }

        # Calculate overall metrics
        if successful_signals > 0:
            confidences = [signals.get('confidence', 0) for signals in all_signals.values() if
                           signals.get('confidence', 0) > 0]
            if confidences:
                portfolio_summary["overall_confidence"] = sum(confidences) / len(confidences)

            # Determine overall signal strength
            buy_signals = sum(1 for signals in all_signals.values() if signals.get('signal') in ['BUY', 'STRONG_BUY'])
            weak_buy_signals = sum(1 for signals in all_signals.values() if signals.get('signal') == 'WEAK_BUY')

            if buy_signals > len(symbols_data) * 0.5:
                portfolio_summary["overall_signal"] = "BUY"
            elif (buy_signals + weak_buy_signals) > len(symbols_data) * 0.3:
                portfolio_summary["overall_signal"] = "WEAK_BUY"
            elif portfolio_summary["overall_confidence"] < 0.3:
                portfolio_summary["overall_signal"] = "AVOID"

            # Calculate total recommended amount
            total_amount = sum(signals.get('allocated_amount', signals.get('recommended_amount', 0))
                               for signals in all_signals.values()
                               if signals.get('signal') not in ['NO_DATA', 'ERROR'])
            portfolio_summary["total_recommended_amount"] = total_amount

        # ENHANCED RESPONSE: Return both individual signals and portfolio summary
        response = {
            "portfolio_summary": portfolio_summary,
            "symbol_signals": all_signals,
            "symbols_processed": list(all_signals.keys()),
            "processing_stats": {
                "successful": successful_signals,
                "failed": failed_signals,
                "success_rate": round((successful_signals / len(symbols_data)) * 100, 2) if symbols_data else 0
            }
        }

        logger.info(
            f"✅ Successfully processed {successful_signals}/{len(symbols_data)} symbols for portfolio {portfolio_id}")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Critical error getting investment signals for portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-symbol signal generation failed: {str(e)}")


@sip_router.get("/signals")
async def get_all_signals(
        active_only: bool = True,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get all investment signals for user's portfolios - ENHANCED FOR MULTI-SYMBOL"""
    try:
        base_query = """
            SELECT s.signal_id, s.symbol, s.signal_type, s.recommended_amount,
                   s.multiplier, s.current_price, s.drawdown_percent, 
                   s.signal_strength, s.gtt_trigger_price, s.created_at,
                   p.portfolio_name, p.portfolio_type, p.portfolio_id
            FROM sip_signals s
            JOIN sip_portfolios p ON s.portfolio_id = p.portfolio_id
            WHERE p.user_id = :user_id
        """
        params = {"user_id": user_id}

        if active_only:
            base_query += """ 
                AND s.is_processed = FALSE 
                AND s.expires_at > :now
            """
            params["now"] = datetime.now()

        base_query += " ORDER BY p.portfolio_name, s.symbol, s.created_at DESC"

        result = await trading_db.execute(text(base_query), params)
        signals = result.fetchall()

        # Group signals by portfolio for better organization
        portfolio_signals = {}

        for row in signals:
            portfolio_id = row[12]  # portfolio_id from query
            portfolio_name = row[10]  # portfolio_name
            portfolio_type = row[11]  # portfolio_type

            if portfolio_id not in portfolio_signals:
                portfolio_signals[portfolio_id] = {
                    "portfolio_id": portfolio_id,
                    "portfolio_name": portfolio_name,
                    "portfolio_type": portfolio_type,
                    "signals": []
                }

            signal_data = {
                "signal_id": row[0],
                "symbol": row[1],
                "signal_type": row[2],
                "recommended_amount": row[3],
                "multiplier": row[4],
                "current_price": row[5],
                "drawdown_percent": row[6],
                "signal_strength": row[7],
                "gtt_trigger_price": row[8],
                "created_at": row[9].isoformat() if row[9] else None
            }

            portfolio_signals[portfolio_id]["signals"].append(signal_data)

        # Also return flat list for backward compatibility
        flat_signals = [
            {
                "signal_id": row[0],
                "symbol": row[1],
                "signal_type": row[2],
                "recommended_amount": row[3],
                "multiplier": row[4],
                "current_price": row[5],
                "drawdown_percent": row[6],
                "signal_strength": row[7],
                "gtt_trigger_price": row[8],
                "created_at": row[9].isoformat() if row[9] else None,
                "portfolio_name": row[10],
                "portfolio_type": row[11],
                "portfolio_id": row[12]
            } for row in signals
        ]

        return {
            "signals": flat_signals,  # Backward compatibility
            "portfolio_signals": list(portfolio_signals.values()),  # Enhanced grouping
            "total_signals": len(signals),
            "total_portfolios": len(portfolio_signals),
            "active_only": active_only,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return {
            "signals": [],
            "portfolio_signals": [],
            "total_signals": 0,
            "total_portfolios": 0,
            "error": str(e)
        }

@sip_router.get("/signals/symbol/{symbol}")
async def get_symbol_signals(
        symbol: str,
        active_only: bool = True,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get all signals for a specific symbol across all portfolios"""
    try:
        base_query = """
            SELECT s.signal_id, s.signal_type, s.recommended_amount,
                   s.multiplier, s.current_price, s.drawdown_percent, 
                   s.signal_strength, s.gtt_trigger_price, s.created_at,
                   p.portfolio_name, p.portfolio_type, p.portfolio_id
            FROM sip_signals s
            JOIN sip_portfolios p ON s.portfolio_id = p.portfolio_id
            WHERE p.user_id = :user_id AND s.symbol = :symbol
        """
        params = {"user_id": user_id, "symbol": symbol.upper()}

        if active_only:
            base_query += """ 
                AND s.is_processed = FALSE 
                AND s.expires_at > :now
            """
            params["now"] = datetime.now()

        base_query += " ORDER BY s.created_at DESC"

        result = await trading_db.execute(text(base_query), params)
        signals = result.fetchall()

        return {
            "symbol": symbol.upper(),
            "signals": [
                {
                    "signal_id": row[0],
                    "signal_type": row[1],
                    "recommended_amount": row[2],
                    "multiplier": row[3],
                    "current_price": row[4],
                    "drawdown_percent": row[5],
                    "signal_strength": row[6],
                    "gtt_trigger_price": row[7],
                    "created_at": row[8].isoformat() if row[8] else None,
                    "portfolio_name": row[9],
                    "portfolio_type": row[10],
                    "portfolio_id": row[11]
                } for row in signals
            ],
            "total_signals": len(signals),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching signals for symbol {symbol}: {e}")
        return {"symbol": symbol, "signals": [], "error": str(e)}


# ============================================================================
# EXECUTION ENDPOINTS
# ============================================================================

@sip_router.post("/execute/{portfolio_id}")
async def execute_sip_investment(
        portfolio_id: str,
        amount: Optional[float] = None,
        place_gtt_orders: bool = True,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Execute SIP investment with minimum gap enforcement - FIXED VERSION"""
    try:
        # Verify portfolio and get details
        portfolio_query = text("""
            SELECT symbols, config FROM sip_portfolios 
            WHERE portfolio_id = :portfolio_id AND user_id = :user_id AND status = 'active'
        """)

        result = await trading_db.execute(portfolio_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id
        })
        portfolio_data = result.fetchone()

        if not portfolio_data:
            raise HTTPException(status_code=404, detail="Active portfolio not found")

        symbols_json, config_json = portfolio_data

        # FIXED: Safe JSON parsing
        try:
            # Parse symbols safely
            if isinstance(symbols_json, str):
                symbols_data = json.loads(symbols_json)
            elif isinstance(symbols_json, list):
                symbols_data = symbols_json
            else:
                raise ValueError(f"Invalid symbols data type: {type(symbols_json)}")

            # Parse config safely
            if isinstance(config_json, str):
                config_dict = json.loads(config_json)
            elif isinstance(config_json, dict):
                config_dict = config_json
            else:
                raise ValueError(f"Invalid config data type: {type(config_json)}")

            config = SIPConfig(**config_dict)

        except Exception as parse_error:
            logger.error(f"JSON parsing error for portfolio {portfolio_id}: {parse_error}")
            raise HTTPException(status_code=500, detail=f"Portfolio data parsing failed: {str(parse_error)}")

        # Validate symbols data
        if not isinstance(symbols_data, list) or not symbols_data:
            raise HTTPException(status_code=400, detail="Invalid or empty symbols in portfolio")

        # Create trades table if not exists
        create_trades_table = text("""
                        CREATE TABLE IF NOT EXISTS sip_actual_trades (
                            trade_id VARCHAR PRIMARY KEY,
                            portfolio_id VARCHAR NOT NULL,
                            symbol VARCHAR NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            price FLOAT NOT NULL,
                            units FLOAT NOT NULL,
                            amount FLOAT NOT NULL,
                            trade_type VARCHAR DEFAULT 'BUY',
                            execution_status VARCHAR DEFAULT 'EXECUTED'
                        )
                    """)
        await trading_db.execute(create_trades_table)

        # Check last investment date for minimum gap enforcement
        last_trade_query = text("""
            SELECT MAX(timestamp) FROM sip_actual_trades 
            WHERE portfolio_id = :portfolio_id
        """)

        last_trade_result = await trading_db.execute(
            last_trade_query, {'portfolio_id': portfolio_id}
        )
        last_trade_date = last_trade_result.scalar()

        current_date = datetime.now().date()

        if last_trade_date:
            days_since_last = (current_date - last_trade_date.date()).days
            if days_since_last < config.min_investment_gap_days:
                raise HTTPException(
                    status_code=400,
                    detail=f"Minimum {config.min_investment_gap_days} days gap required. "
                           f"Last investment was {days_since_last} days ago."
                )

        # Execute investments for each symbol
        executed_trades = []
        total_invested = 0
        total_units = 0

        for symbol_config in symbols_data:
            try:
                # Extract symbol safely
                if isinstance(symbol_config, dict):
                    symbol = symbol_config.get('symbol')
                    allocation_pct = symbol_config.get('allocation_percentage', 100.0)
                else:
                    symbol = symbol_config  # Direct string
                    allocation_pct = 100.0 / len(symbols_data)  # Equal allocation

                if not symbol:
                    logger.warning(f"Skipping invalid symbol config: {symbol_config}")
                    continue

                # Calculate investment amount for this symbol
                base_amount = amount or config.fixed_investment
                symbol_amount = (base_amount * allocation_pct) / 100.0

                # Mock execution (replace with actual broker integration)
                execution_price = 150.0  # Replace with actual market price
                units_bought = symbol_amount / execution_price

                # Record trade
                trade_id = f"trade_{portfolio_id}_{symbol}_{int(datetime.now().timestamp())}"

                insert_trade = text("""
                    INSERT INTO sip_actual_trades 
                    (trade_id, portfolio_id, symbol, timestamp, price, units, amount, trade_type)
                    VALUES (:trade_id, :portfolio_id, :symbol, :timestamp, :price, :units, :amount, :trade_type)
                """)

                await trading_db.execute(insert_trade, {
                    'trade_id': trade_id,
                    'portfolio_id': portfolio_id,
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'price': execution_price,
                    'units': units_bought,
                    'amount': symbol_amount,
                    'trade_type': 'BUY'
                })

                executed_trades.append({
                    "symbol": symbol,
                    "amount": symbol_amount,
                    "price": execution_price,
                    "units": units_bought,
                    "trade_id": trade_id
                })

                total_invested += symbol_amount
                total_units += units_bought

            except Exception as symbol_error:
                logger.error(f"Error executing trade for symbol {symbol_config}: {symbol_error}")
                continue

        if not executed_trades:
            raise HTTPException(status_code=400, detail="No trades could be executed")

        # Update portfolio totals and next investment date
        next_investment_date = calculate_next_investment_date(current_date, config)

        update_query = text("""
            UPDATE sip_portfolios 
            SET total_invested = total_invested + :amount,
                current_units = current_units + :units,
                current_value = (current_units + :units) * 150.0,
                next_investment_date = :next_date,
                updated_at = :now
            WHERE portfolio_id = :portfolio_id
        """)

        await trading_db.execute(update_query, {
            'amount': total_invested,
            'units': total_units,
            'next_date': next_investment_date,
            'now': datetime.now(),
            'portfolio_id': portfolio_id
        })

        await trading_db.commit()

        logger.info(f"✅ Successfully executed {len(executed_trades)} trades for portfolio {portfolio_id}")

        return {
            "status": "success",
            "portfolio_id": portfolio_id,
            "total_investment_amount": total_invested,
            "executed_trades": executed_trades,
            "next_investment_date": next_investment_date.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing SIP investment: {e}")
        await trading_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@sip_router.get("/analytics/portfolio/{portfolio_id}")
async def get_detailed_portfolio_analytics(
        portfolio_id: str,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get comprehensive portfolio analytics - FIXED VERSION"""
    try:
        # Verify portfolio ownership
        portfolio_query = text("""
            SELECT portfolio_name, symbols, total_invested, current_units, created_at 
            FROM sip_portfolios 
            WHERE portfolio_id = :portfolio_id AND user_id = :user_id
        """)

        result = await trading_db.execute(portfolio_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id
        })
        portfolio_data = result.fetchone()

        if not portfolio_data:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        portfolio_name, symbols_json, total_invested, current_units, created_at = portfolio_data

        # FIXED: Safe JSON parsing for symbols
        try:
            if isinstance(symbols_json, str):
                symbols_data = json.loads(symbols_json)
            elif isinstance(symbols_json, (list, dict)):
                symbols_data = symbols_json
            else:
                symbols_data = []
        except Exception as parse_error:
            logger.error(f"Error parsing symbols for portfolio {portfolio_id}: {parse_error}")
            symbols_data = []

        # Get trade history with detailed analytics
        trades_query = text("""
            SELECT symbol, timestamp, price, units, amount, trade_type, execution_status
            FROM sip_actual_trades 
            WHERE portfolio_id = :portfolio_id
            ORDER BY timestamp ASC
        """)

        result = await trading_db.execute(trades_query, {'portfolio_id': portfolio_id})
        trades = result.fetchall()

        # Calculate performance metrics per symbol
        symbol_analytics = {}
        total_current_value = 0

        for symbol_config in symbols_data:
            if isinstance(symbol_config, dict):
                symbol = symbol_config.get('symbol')
                allocation_pct = symbol_config.get('allocation_percentage', 0)
            else:
                symbol = symbol_config
                allocation_pct = 100.0 / len(symbols_data) if symbols_data else 0

            if not symbol:
                continue

            symbol_trades = [t for t in trades if t[0] == symbol]

            if symbol_trades:
                symbol_invested = sum(float(t[4]) for t in symbol_trades)  # amount column
                symbol_units = sum(float(t[3]) for t in symbol_trades)  # units column
                avg_price = symbol_invested / symbol_units if symbol_units > 0 else 0

                # Current value (placeholder price - replace with actual market price)
                current_price = 150.0
                symbol_current_value = symbol_units * current_price
                symbol_return = ((symbol_current_value / symbol_invested) - 1) * 100 if symbol_invested > 0 else 0

                symbol_analytics[symbol] = {
                    "invested": symbol_invested,
                    "units": symbol_units,
                    "avg_buy_price": avg_price,
                    "current_price": current_price,
                    "current_value": symbol_current_value,
                    "return_percent": symbol_return,
                    "trades_count": len(symbol_trades),
                    "allocation_percent": allocation_pct
                }

                total_current_value += symbol_current_value

        # Overall portfolio metrics
        total_return = ((total_current_value / total_invested) - 1) * 100 if total_invested > 0 else 0

        days_invested = (datetime.now() - created_at).days if created_at else 0
        years_invested = days_invested / 365.25
        cagr = (total_current_value / total_invested) ** (
                    1 / years_invested) - 1 if years_invested > 0 and total_invested > 0 else 0

        return {
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio_name,
            "total_invested": total_invested or 0,
            "current_value": total_current_value,
            "total_return_percent": total_return,
            "cagr_percent": cagr * 100,
            "days_invested": days_invested,
            "total_trades": len(trades),
            "symbols_analytics": symbol_analytics,
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting portfolio analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================

@sip_router.get("/config/defaults")
async def get_default_config():
    """Get enhanced SIP configuration with monthly limits and price thresholds"""
    config = SIPConfigRequest()
    return {
        "enhanced_config": {
            "fixed_investment": config.fixed_investment,
            "max_amount_in_a_month": config.max_amount_in_a_month,
            "price_reduction_threshold": config.price_reduction_threshold,
            "drawdown_threshold_1": config.drawdown_threshold_1,
            "drawdown_threshold_2": config.drawdown_threshold_2,
            "investment_multiplier_1": config.investment_multiplier_1,
            "investment_multiplier_2": config.investment_multiplier_2,
            "investment_multiplier_3": config.investment_multiplier_3,
            "rolling_window": config.rolling_window,
            "fallback_day": config.fallback_day,
            "min_investment_gap_days": config.min_investment_gap_days
        },
        "new_features": {
            "monthly_investment_limits": {
                "description": "Enforces monthly investment limits per symbol",
                "default_calculation": "4 × fixed_investment",
                "configurable": True
            },
            "price_reduction_threshold": {
                "description": "Required price reduction for multiple signals in same month",
                "default_value": "4.0%",
                "purpose": "Prevents excessive buying without significant price drops"
            }
        },
        "description": "Enhanced SIP strategy with monthly limits and price threshold controls",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# MISSING ENDPOINTS FROM ORIGINAL IMPLEMENTATION
# ============================================================================

@sip_router.get("/performance/{portfolio_id}")
async def get_portfolio_performance(
        portfolio_id: str,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get performance metrics for a SIP portfolio - FIXED VERSION"""
    try:
        # Verify portfolio ownership
        portfolio_query = text("""
            SELECT symbols, total_invested, current_units, created_at 
            FROM sip_portfolios 
            WHERE portfolio_id = :portfolio_id AND user_id = :user_id
        """)

        result = await trading_db.execute(portfolio_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id
        })
        portfolio_data = result.fetchone()

        if not portfolio_data:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        symbols_json, total_invested, current_units, created_at = portfolio_data

        # FIXED: Safe JSON parsing
        try:
            if isinstance(symbols_json, str):
                symbols_data = json.loads(symbols_json)
            elif isinstance(symbols_json, (list, dict)):
                symbols_data = symbols_json
            else:
                symbols_data = []
        except Exception as parse_error:
            logger.error(f"Error parsing symbols for performance {portfolio_id}: {parse_error}")
            symbols_data = []

        # Calculate basic performance metrics
        current_value = (current_units or 0) * 150.0  # Mock current price
        total_return = ((current_value / (total_invested or 1)) - 1) * 100 if total_invested else 0

        days_invested = (datetime.now() - created_at).days if created_at else 1
        years_invested = max(days_invested / 365.25, 0.01)  # Prevent division by zero
        cagr = ((current_value / (total_invested or 1)) ** (1 / years_invested) - 1) * 100 if total_invested else 0

        return {
            "portfolio_id": portfolio_id,
            "performance_summary": {
                "total_invested": total_invested or 0,
                "current_value": current_value,
                "total_return_percent": total_return,
                "cagr_percent": cagr,
                "days_invested": days_invested
            },
            "symbols_count": len(symbols_data),
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.get("/strategies/compare/{symbol}")
async def compare_strategies(
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        user_id: str = Depends(get_current_user),
        nsedata_db: AsyncSession = Depends(get_nsedata_db),
        enable_monthly_limits: bool = True
):
    """Compare different SIP strategy configurations with benchmark"""
    try:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        strategy = EnhancedSIPStrategy(nsedata_session=nsedata_db)
        benchmark_calculator = BenchmarkSIPCalculator()

        # Define strategy variants
        strategies = {
            "Conservative": SIPConfig(
                fixed_investment=5000,
                drawdown_threshold_1=-15.0,
                drawdown_threshold_2=-8.0,
                investment_multiplier_1=1.5,
                investment_multiplier_2=2.0,
                investment_multiplier_3=2.5
            ),
            "Balanced": SIPConfig(
                fixed_investment=5000,
                drawdown_threshold_1=-10.0,
                drawdown_threshold_2=-5.0,
                investment_multiplier_1=2.0,
                investment_multiplier_2=3.0,
                investment_multiplier_3=4.0
            ),
            "Aggressive": SIPConfig(
                fixed_investment=5000,
                drawdown_threshold_1=-5.0,
                drawdown_threshold_2=-2.0,
                investment_multiplier_1=3.0,
                investment_multiplier_2=5.0,
                investment_multiplier_3=8.0
            )
        }

        # Calculate benchmark first
        benchmark_result = await benchmark_calculator.calculate_benchmark(
            symbol, start_date, end_date, nsedata_db
        )

        results = {}
        for strategy_name, config in strategies.items():
            logger.info(f"Running {strategy_name} strategy for {symbol}")

            if enable_monthly_limits:
                backtest_results = await strategy.run_batch_backtest_with_monthly_limit([symbol], start_date, end_date, config)
            else:
                backtest_results = await strategy.run_batch_backtest([symbol], start_date, end_date, config)

            if symbol in backtest_results:
                result = backtest_results[symbol]
                roi = ((result.final_portfolio_value / result.total_investment) - 1) * 100

                strategy_data = {
                    "total_investment": result.total_investment,
                    "final_value": result.final_portfolio_value,
                    "total_return_percent": roi,
                    "cagr_percent": result.cagr * 100,
                    "max_drawdown_percent": result.max_drawdown * 100 if result.max_drawdown else 0,
                    "sharpe_ratio": result.sharpe_ratio,
                    "volatility_percent": result.volatility * 100 if result.volatility else 0,
                    "num_trades": len(result.trades),
                    "average_buy_price": result.average_buy_price
                }

                # Add benchmark comparison
                if benchmark_result.get('total_investment', 0) > 0:
                    strategy_data['vs_benchmark'] = {
                        'return_outperformance': roi - benchmark_result['total_return_percent'],
                        'cagr_outperformance': (result.cagr * 100) - benchmark_result['cagr_percent'],
                        'trade_difference': len(result.trades) - benchmark_result['num_trades']
                    }

                results[strategy_name] = strategy_data

        return {
            "symbol": symbol,
            "analysis_period": f"{start_date} to {end_date}",
            "benchmark": benchmark_result,
            "strategies": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Strategy comparison failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.post("/quick-test")
async def quick_sip_test(
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        investment_amount: float = 5000,
        user_id: str = Depends(get_current_user),
        nsedata_db: AsyncSession = Depends(get_nsedata_db),
        enable_monthly_limits: bool = True
):
    """Quick SIP test for rapid analysis with benchmark comparison"""
    try:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Create a basic config for quick testing
        config = SIPConfig(
            fixed_investment=investment_amount,
            drawdown_threshold_1=-10.0,
            drawdown_threshold_2=-4.0,
            investment_multiplier_1=2.0,
            investment_multiplier_2=3.0,
            investment_multiplier_3=5.0
        )

        strategy = EnhancedSIPStrategy(nsedata_session=nsedata_db)
        benchmark_calculator = BenchmarkSIPCalculator(monthly_amount=investment_amount)

        # Run quick backtest
        if enable_monthly_limits:
            results = await strategy.run_batch_backtest_with_monthly_limit(symbols, start_date, end_date, config)
        else:
            results = await strategy.run_batch_backtest(symbols, start_date, end_date, config)

        if not results:
            return {"message": "No data available for quick test", "symbols": symbols}

        # Calculate benchmark for all symbols
        quick_results = []
        for symbol, result in results.items():
            roi = ((result.final_portfolio_value / result.total_investment) - 1) * 100

            # Calculate benchmark
            benchmark = await benchmark_calculator.calculate_benchmark(
                symbol, start_date, end_date, nsedata_db
            )

            symbol_result = {
                "symbol": symbol,
                "strategy": {
                    "invested": result.total_investment,
                    "final_value": result.final_portfolio_value,
                    "return_percent": roi,
                    "cagr_percent": result.cagr * 100,
                    "trades": len(result.trades)
                },
                "benchmark": {
                    "invested": benchmark.get('total_investment', 0),
                    "final_value": benchmark.get('final_portfolio_value', 0),
                    "return_percent": benchmark.get('total_return_percent', 0),
                    "cagr_percent": benchmark.get('cagr_percent', 0),
                    "trades": benchmark.get('num_trades', 0)
                }
            }

            # Add comparison
            if benchmark.get('total_investment', 0) > 0:
                symbol_result["comparison"] = {
                    "outperformance": roi - benchmark['total_return_percent'],
                    "recommendation": "Strategy" if roi > benchmark['total_return_percent'] else "Benchmark"
                }

            quick_results.append(symbol_result)

        return {
            "test_period": f"{start_date} to {end_date}",
            "investment_per_symbol": investment_amount,
            "results": quick_results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.get("/templates")
async def get_strategy_templates():
    """Get predefined SIP strategy templates"""
    templates = {
        "conservative": {
            "name": "Conservative SIP",
            "description": "Lower risk approach with smaller multipliers and deeper drawdown thresholds",
            "config": {
                "fixed_investment": 5000,
                "drawdown_threshold_1": -15.0,
                "drawdown_threshold_2": -8.0,
                "investment_multiplier_1": 1.5,
                "investment_multiplier_2": 2.0,
                "investment_multiplier_3": 2.5,
                "rolling_window": 100,
                "fallback_day": 22,
                "min_investment_gap_days": 7
            },
            "risk_level": "Low",
            "suitable_for": ["Conservative investors", "Stable income seekers", "Risk-averse portfolios"]
        },
        "balanced": {
            "name": "Balanced SIP",
            "description": "Moderate risk approach balancing opportunity and safety",
            "config": {
                "fixed_investment": 5000,
                "drawdown_threshold_1": -10.0,
                "drawdown_threshold_2": -5.0,
                "investment_multiplier_1": 2.0,
                "investment_multiplier_2": 3.0,
                "investment_multiplier_3": 4.0,
                "rolling_window": 100,
                "fallback_day": 22,
                "min_investment_gap_days": 5
            },
            "risk_level": "Medium",
            "suitable_for": ["Balanced investors", "Long-term growth seekers", "Moderate risk tolerance"]
        },
        "aggressive": {
            "name": "Aggressive SIP",
            "description": "High risk, high reward approach with aggressive multipliers",
            "config": {
                "fixed_investment": 5000,
                "drawdown_threshold_1": -5.0,
                "drawdown_threshold_2": -2.0,
                "investment_multiplier_1": 3.0,
                "investment_multiplier_2": 5.0,
                "investment_multiplier_3": 8.0,
                "rolling_window": 50,
                "fallback_day": 15,
                "min_investment_gap_days": 3
            },
            "risk_level": "High",
            "suitable_for": ["Risk-tolerant investors", "Growth-focused portfolios", "Active management style"]
        }
    }

    return {
        "templates": templates,
        "default_recommendation": "balanced",
        "customization_note": "All templates can be customized based on individual risk tolerance",
        "timestamp": datetime.now().isoformat()
    }

async def save_investment_report_to_db(
        report_id: str,
        user_id: str,
        symbols: List[str],
        report_data: Dict,
        trading_db: AsyncSession
):
    """Save investment report to database (background task) - FIXED VERSION"""
    try:
        # Create table if it doesn't exist
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS investment_reports (
                report_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                symbols JSONB NOT NULL,
                report_data JSONB NOT NULL,
                report_summary JSONB,
                report_type VARCHAR DEFAULT 'comprehensive',
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await trading_db.execute(create_table_query)

        # Prepare summary for quick access
        summary = {
            "total_symbols": len(symbols),
            "analyzed_symbols": report_data.get("overall_metrics", {}).get("analyzed_symbols", 0),
            "overall_action": report_data.get("portfolio_recommendation", {}).get("portfolio_action", "UNKNOWN"),
            "risk_level": report_data.get("risk_assessment", {}).get("overall_risk_level", "UNKNOWN"),
            "avg_confidence": report_data.get("overall_metrics", {}).get("avg_confidence", 0)
        }

        # FIXED: Ensure proper JSON serialization
        try:
            symbols_json = json.dumps(symbols)
            report_data_json = json.dumps(report_data, default=str)  # Use default=str for datetime objects
            summary_json = json.dumps(summary)
        except Exception as serialize_error:
            logger.error(f"JSON serialization error: {serialize_error}")
            # Fallback serialization
            symbols_json = json.dumps([str(s) for s in symbols])
            report_data_json = json.dumps({"error": "Serialization failed", "original_error": str(serialize_error)})
            summary_json = json.dumps({"error": "Summary serialization failed"})

        # Insert report
        insert_query = text("""
            INSERT INTO investment_reports 
            (report_id, user_id, symbols, report_data, report_summary, report_type, generated_at)
            VALUES (:report_id, :user_id, :symbols, :report_data, :report_summary, :report_type, :generated_at)
        """)

        await trading_db.execute(insert_query, {
            'report_id': report_id,
            'user_id': user_id,
            'symbols': symbols_json,
            'report_data': report_data_json,
            'report_summary': summary_json,
            'report_type': 'comprehensive',
            'generated_at': datetime.now()
        })

        await trading_db.commit()
        logger.info(f"✅ Saved investment report {report_id} to database")

    except Exception as e:
        logger.error(f"Error saving investment report to DB: {e}")
        await trading_db.rollback()
        raise  # Re-raise to handle in calling function


@sip_router.post("/reports/investment", response_model=InvestmentReportResponse)
async def generate_comprehensive_investment_report(
        request: InvestmentReportRequest,
        background_tasks: BackgroundTasks,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db),
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Generate comprehensive investment report for multiple symbols"""
    try:
        # Create strategy instance
        strategy = EnhancedSIPStrategy(
            nsedata_session=nsedata_db,
            trading_session=trading_db
        )

        # Use provided config or default
        config = SIPConfig(**request.config.dict()) if request.config else SIPConfig()

        logger.info(f"🔄 Generating investment report for user {user_id}")
        logger.info(f"📊 Symbols: {request.symbols}")
        logger.info(f"📋 Report type: {request.report_type}")

        # Generate the comprehensive report
        report = await strategy.generate_investment_report(request.symbols, config)

        if not report or report.get('status') == 'ERROR':
            raise HTTPException(
                status_code=500,
                detail=report.get('message', 'Failed to generate investment report')
            )

        # Generate unique report ID
        report_id = f"rpt_{user_id}_{int(datetime.now().timestamp())}"

        # Save report to database in background
        background_tasks.add_task(
            save_investment_report_to_db,
            report_id, user_id, request.symbols, report, trading_db
        )

        # Prepare response
        response_data = {
            "report_id": report_id,
            "report_generated": report["report_generated"],
            "analysis_period": report["analysis_period"],
            "overall_metrics": report["overall_metrics"],
            "portfolio_recommendation": report["portfolio_recommendation"],
            "risk_assessment": report["risk_assessment"],
            "symbol_reports": report["symbol_reports"],
            "disclaimer": report["disclaimer"]
        }

        logger.info(f"✅ Investment report generated successfully: {report_id}")

        return response_data

    except Exception as e:
        logger.error(f"Error generating investment report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@sip_router.get("/reports/quick/{symbols}")
async def generate_quick_investment_report(
        symbols: str,  # Comma-separated symbols
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db),
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Generate quick investment report for symbols (comma-separated)"""
    try:
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]

        if not symbol_list:
            raise HTTPException(status_code=400, detail="No valid symbols provided")

        if len(symbol_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols for quick report")

        # Create strategy instance
        strategy = EnhancedSIPStrategy(
            nsedata_session=nsedata_db,
            trading_session=trading_db
        )

        config = SIPConfig()  # Use default config for quick reports

        logger.info(f"🚀 Generating quick report for: {symbol_list}")

        # Generate quick recommendations
        recommendations = await strategy.get_portfolio_recommendations(symbol_list, config)

        # Get basic statistics for each symbol
        symbol_stats = {}
        for symbol in symbol_list:
            try:
                quick_check = await strategy.quick_symbol_check(symbol)
                symbol_stats[symbol] = quick_check
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
                symbol_stats[symbol] = {"available": False, "error": str(e)}

        return {
            "report_type": "quick",
            "symbols": symbol_list,
            "generated_at": datetime.now().isoformat(),
            "portfolio_recommendations": recommendations,
            "symbol_statistics": symbol_stats,
            "next_steps": [
                "Review individual symbol recommendations",
                "Consider generating comprehensive report for detailed analysis",
                "Monitor signals for investment opportunities"
            ]
        }

    except Exception as e:
        logger.error(f"Error generating quick report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.get("/reports/history")
async def get_investment_report_history(
        limit: int = 10,
        offset: int = 0,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get user's investment report history - FIXED VERSION"""
    try:
        query = text("""
            SELECT report_id, symbols, report_summary, generated_at, report_type
            FROM investment_reports 
            WHERE user_id = :user_id
            ORDER BY generated_at DESC
            LIMIT :limit OFFSET :offset
        """)

        result = await trading_db.execute(query, {
            'user_id': user_id,
            'limit': limit,
            'offset': offset
        })

        reports = result.fetchall()

        report_list = []

        for row in reports:
            try:
                # FIXED: Safe JSON parsing for symbols column (row[1])
                symbols_raw = row[1]
                if symbols_raw is None:
                    symbols_data = []
                elif isinstance(symbols_raw, str):
                    try:
                        symbols_data = json.loads(symbols_raw)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in symbols for report {row[0]}")
                        symbols_data = []
                elif isinstance(symbols_raw, list):
                    symbols_data = symbols_raw  # Already parsed by PostgreSQL JSONB
                else:
                    logger.warning(f"Unexpected symbols type for report {row[0]}: {type(symbols_raw)}")
                    symbols_data = []

                # FIXED: Safe JSON parsing for report_summary column (row[2])
                summary_raw = row[2]
                if summary_raw is None:
                    summary_data = {}
                elif isinstance(summary_raw, str):
                    try:
                        summary_data = json.loads(summary_raw)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in summary for report {row[0]}")
                        summary_data = {}
                elif isinstance(summary_raw, dict):
                    summary_data = summary_raw  # Already parsed by PostgreSQL JSONB
                else:
                    logger.warning(f"Unexpected summary type for report {row[0]}: {type(summary_raw)}")
                    summary_data = {}

                report_item = {
                    "report_id": row[0],
                    "symbols": symbols_data,
                    "summary": summary_data,
                    "generated_at": row[3].isoformat() if row[3] else None,
                    "report_type": row[4] or "comprehensive"
                }

                report_list.append(report_item)

            except Exception as report_error:
                logger.error(f"Error processing report {row[0]}: {report_error}")
                # Skip this report but continue processing others
                continue

        logger.info(f"✅ Successfully fetched {len(report_list)} reports for user {user_id}")
        return report_list

    except Exception as e:
        logger.error(f"Error fetching report history: {e}")
        return []

@sip_router.get("/reports/{report_id}")
async def get_investment_report_details(
        report_id: str,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get detailed investment report by ID - FIXED VERSION"""
    try:
        query = text("""
            SELECT report_data, generated_at, symbols
            FROM investment_reports 
            WHERE report_id = :report_id AND user_id = :user_id
        """)

        result = await trading_db.execute(query, {
            'report_id': report_id,
            'user_id': user_id
        })

        report_row = result.fetchone()

        if not report_row:
            raise HTTPException(status_code=404, detail="Report not found")

        # FIXED: Safe JSON parsing for all columns
        try:
            # Parse report_data (row[0])
            report_data_raw = report_row[0]
            if report_data_raw is None:
                report_data = {}
            elif isinstance(report_data_raw, str):
                try:
                    report_data = json.loads(report_data_raw)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in report_data for report {report_id}")
                    report_data = {}
            elif isinstance(report_data_raw, dict):
                report_data = report_data_raw  # Already parsed
            else:
                logger.warning(f"Unexpected report_data type: {type(report_data_raw)}")
                report_data = {}

            # Parse symbols (row[2])
            symbols_raw = report_row[2]
            if symbols_raw is None:
                symbols_data = []
            elif isinstance(symbols_raw, str):
                try:
                    symbols_data = json.loads(symbols_raw)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in symbols for report {report_id}")
                    symbols_data = []
            elif isinstance(symbols_raw, list):
                symbols_data = symbols_raw  # Already parsed
            else:
                logger.warning(f"Unexpected symbols type: {type(symbols_raw)}")
                symbols_data = []

            return {
                "report_id": report_id,
                "report_data": report_data,
                "generated_at": report_row[1].isoformat() if report_row[1] else None,
                "symbols": symbols_data
            }

        except Exception as parse_error:
            logger.error(f"Error parsing report details for {report_id}: {parse_error}")
            raise HTTPException(status_code=500, detail=f"Report data parsing failed: {str(parse_error)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching report details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_comparison_metrics(strategy_result: Dict, benchmark_result: Dict) -> Dict:
    """Calculate comparison metrics between strategy and benchmark"""
    try:
        strategy_return = strategy_result.get('total_return_percent', 0)
        benchmark_return = benchmark_result.get('total_return_percent', 0)

        strategy_cagr = strategy_result.get('cagr_percent', 0)
        benchmark_cagr = benchmark_result.get('cagr_percent', 0)

        strategy_investment = strategy_result.get('total_investment', 0)
        benchmark_investment = benchmark_result.get('total_investment', 0)

        strategy_trades = strategy_result.get('num_trades', 0)
        benchmark_trades = benchmark_result.get('num_trades', 0)

        # Calculate outperformance
        return_outperformance = strategy_return - benchmark_return
        cagr_outperformance = strategy_cagr - benchmark_cagr

        # Calculate efficiency metrics
        investment_efficiency = ((
                                             strategy_investment / benchmark_investment) - 1) * 100 if benchmark_investment > 0 else 0
        trade_efficiency = strategy_trades - benchmark_trades

        # Determine recommendation
        recommendation = "Strategy outperforms benchmark" if return_outperformance > 0 else "Benchmark performs better"

        return {
            'return_outperformance_percent': float(return_outperformance),
            'cagr_outperformance_percent': float(cagr_outperformance),
            'investment_efficiency_percent': float(investment_efficiency),
            'trade_difference': trade_efficiency,
            'strategy_vs_benchmark': {
                'strategy_return': float(strategy_return),
                'benchmark_return': float(benchmark_return),
                'strategy_cagr': float(strategy_cagr),
                'benchmark_cagr': float(benchmark_cagr),
                'strategy_investment': float(strategy_investment),
                'benchmark_investment': float(benchmark_investment)
            },
            'recommendation': recommendation,
            'performance_summary': f"Strategy {'outperforms' if return_outperformance > 0 else 'underperforms'} benchmark by {abs(return_outperformance):.2f}%"
        }

    except Exception as e:
        logger.error(f"Error calculating comparison metrics: {e}")
        return {
            'error': 'Could not calculate comparison metrics',
            'return_outperformance_percent': 0.0,
            'cagr_outperformance_percent': 0.0,
            'recommendation': 'Comparison not available'
        }


@sip_router.get("/benchmark/test/{symbol}")
async def test_benchmark_calculation(
        symbol: str,
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        monthly_amount: float = 5000,
        investment_day: int = 15,
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Test benchmark calculation for a specific symbol"""
    try:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        benchmark_calculator = BenchmarkSIPCalculator(
            monthly_amount=monthly_amount,
            investment_day=investment_day
        )

        result = await benchmark_calculator.calculate_benchmark(
            symbol, start_date, end_date, nsedata_db
        )

        return {
            "test_parameters": {
                "symbol": symbol,
                "period": f"{start_date} to {end_date}",
                "monthly_amount": monthly_amount,
                "investment_day": investment_day
            },
            "benchmark_result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Benchmark test failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SCHEDULER STATUS AND CONTROL ENDPOINTS
# ============================================================================

@sip_router.get("/scheduler/status", response_model=Dict)
async def get_scheduler_status():
    """Get comprehensive scheduler status and job information"""
    try:
        # Calculate uptime if scheduler is running
        uptime = 0
        if scheduler.running:
            # Approximation - would need to track start time for exact uptime
            uptime = 3600  # Placeholder - implement proper uptime tracking

        jobs_info = []
        for job in scheduler.get_jobs():
            # FIXED: Handle case when scheduler is not running
            next_run_time = None
            if scheduler.running and hasattr(job, 'next_run_time') and job.next_run_time:
                next_run_time = job.next_run_time.isoformat()

            job_info = {
                "job_id": job.id,
                "status": "running" if scheduler.running else "paused",
                "next_run_time": next_run_time,
                "trigger_info": {
                    "trigger_type": type(job.trigger).__name__,
                    "trigger_details": str(job.trigger)
                },
                "max_instances": job.max_instances,
                "pending_instances": 0  # Simplified for now
            }
            jobs_info.append(job_info)

        return {
            "scheduler_running": scheduler.running,
            "total_jobs": len(scheduler.get_jobs()),
            "active_jobs": jobs_info,
            "timezone": str(scheduler.timezone),
            "uptime_seconds": uptime
        }

    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler status: {str(e)}")


@sip_router.post("/scheduler/start")
async def start_scheduler():
    """Start the scheduler if it's not running"""
    try:
        if not scheduler.running:
            scheduler.start()
            logger.info("✅ Scheduler started via API")
            return {"status": "success", "message": "Scheduler started successfully"}
        else:
            return {"status": "info", "message": "Scheduler is already running"}

    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start scheduler: {str(e)}")


@sip_router.post("/scheduler/pause")
async def pause_scheduler():
    """Pause the scheduler (stops scheduling new jobs but keeps existing ones)"""
    try:
        if scheduler.running:
            scheduler.pause()
            logger.info("⏸️ Scheduler paused via API")
            return {"status": "success", "message": "Scheduler paused successfully"}
        else:
            return {"status": "info", "message": "Scheduler is not running"}

    except Exception as e:
        logger.error(f"Error pausing scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause scheduler: {str(e)}")


@sip_router.post("/scheduler/resume")
async def resume_scheduler():
    """Resume the scheduler if it's paused"""
    try:
        if scheduler.state == 2:  # STATE_PAUSED = 2
            scheduler.resume()
            logger.info("▶️ Scheduler resumed via API")
            return {"status": "success", "message": "Scheduler resumed successfully"}
        else:
            return {"status": "info", "message": "Scheduler is not paused"}

    except Exception as e:
        logger.error(f"Error resuming scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume scheduler: {str(e)}")


@sip_router.post("/scheduler/shutdown")
async def shutdown_scheduler():
    """Shutdown the scheduler completely"""
    try:
        if scheduler.running:
            scheduler.shutdown(wait=True)
            logger.info("🛑 Scheduler shutdown via API")
            return {"status": "success", "message": "Scheduler shutdown successfully"}
        else:
            return {"status": "info", "message": "Scheduler is not running"}

    except Exception as e:
        logger.error(f"Error shutting down scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to shutdown scheduler: {str(e)}")


# ============================================================================
# JOB MANAGEMENT ENDPOINTS
# ============================================================================

@sip_router.get("/scheduler/jobs")
async def list_jobs():
    """List all scheduled jobs with their details"""
    try:
        jobs = []
        for job in scheduler.get_jobs():
            # FIXED: Safe access to next_run_time
            next_run_time = None
            if scheduler.running and hasattr(job, 'next_run_time') and job.next_run_time:
                next_run_time = job.next_run_time.isoformat()

            job_detail = {
                "job_id": job.id,
                "function_name": job.func.__name__,
                "next_run_time": next_run_time,
                "trigger_type": type(job.trigger).__name__,
                "trigger_info": str(job.trigger),
                "max_instances": job.max_instances,
                "coalesce": job.coalesce,
                "misfire_grace_time": job.misfire_grace_time,
                "scheduler_running": scheduler.running
            }
            jobs.append(job_detail)

        return {
            "total_jobs": len(jobs),
            "jobs": jobs,
            "scheduler_running": scheduler.running,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@sip_router.get("/scheduler/jobs/{job_id}")
async def get_job_details(job_id: str):
    """Get detailed information about a specific job"""
    try:
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        # FIXED: Safe access to next_run_time
        next_run_time = None
        if scheduler.running and hasattr(job, 'next_run_time') and job.next_run_time:
            next_run_time = job.next_run_time.isoformat()

        return {
            "job_id": job.id,
            "function_name": job.func.__name__,
            "next_run_time": next_run_time,
            "trigger_type": type(job.trigger).__name__,
            "trigger_info": str(job.trigger),
            "max_instances": job.max_instances,
            "coalesce": job.coalesce,
            "misfire_grace_time": job.misfire_grace_time,
            "scheduler_running": scheduler.running,
            "scheduler_state": scheduler.state
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job details: {str(e)}")


@sip_router.delete("/scheduler/jobs/{job_id}")
async def remove_job(job_id: str):
    """Remove a specific job from the scheduler"""
    try:
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        scheduler.remove_job(job_id)
        logger.info(f"🗑️ Job '{job_id}' removed via API")

        return {
            "status": "success",
            "message": f"Job '{job_id}' removed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove job: {str(e)}")


@sip_router.post("/scheduler/jobs/{job_id}/run")
async def run_job_now(job_id: str):
    """Execute a job immediately (outside of its normal schedule)"""
    try:
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        # FIXED: Different approach for running job immediately
        if scheduler.running:
            # If scheduler is running, modify next_run_time
            job.modify(next_run_time=datetime.now())
            logger.info(f"🚀 Job '{job_id}' scheduled to run immediately")
            message = f"Job '{job_id}' scheduled to run immediately"
        else:
            # If scheduler is not running, execute the job function directly
            logger.info(f"🚀 Executing job '{job_id}' directly (scheduler not running)")
            asyncio.create_task(job.func())
            message = f"Job '{job_id}' executed directly"

        return {
            "status": "success",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running job immediately: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run job immediately: {str(e)}")


@sip_router.post("/scheduler/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a specific job"""
    try:
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        scheduler.pause_job(job_id)
        logger.info(f"⏸️ Job '{job_id}' paused via API")

        return {
            "status": "success",
            "message": f"Job '{job_id}' paused successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause job: {str(e)}")


@sip_router.post("/scheduler/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job"""
    try:
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        scheduler.resume_job(job_id)
        logger.info(f"▶️ Job '{job_id}' resumed via API")

        return {
            "status": "success",
            "message": f"Job '{job_id}' resumed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume job: {str(e)}")


# ============================================================================
# JOB CONFIGURATION ENDPOINTS
# ============================================================================

@sip_router.put("/scheduler/jobs/{job_id}/configure")
async def update_job_configuration(job_id: str, config: SchedulerConfigRequest):
    """Update job configuration (trigger, timing, etc.)"""
    try:
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        # Prepare new trigger based on type
        trigger = None
        timezone_obj = pytz.timezone(config.timezone)

        if config.trigger_type == JobTriggerType.CRON:
            if not config.cron_expression:
                raise HTTPException(status_code=400, detail="cron_expression required for CRON trigger")

            # Validate cron expression
            try:
                # Parse cron expression to validate it
                import croniter
                if not croniter.is_valid(config.cron_expression):
                    raise ValueError("Invalid cron expression")

                trigger = CronTrigger.from_crontab(config.cron_expression, timezone=timezone_obj)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid cron expression: {str(e)}")

        elif config.trigger_type == JobTriggerType.INTERVAL:
            from apscheduler.triggers.interval import IntervalTrigger

            interval_kwargs = {}
            if config.interval_seconds:
                interval_kwargs['seconds'] = config.interval_seconds
            if config.interval_minutes:
                interval_kwargs['minutes'] = config.interval_minutes
            if config.interval_hours:
                interval_kwargs['hours'] = config.interval_hours

            if not interval_kwargs:
                raise HTTPException(status_code=400, detail="At least one interval parameter required")

            trigger = IntervalTrigger(**interval_kwargs, timezone=timezone_obj)

        elif config.trigger_type == JobTriggerType.DATE:
            from apscheduler.triggers.date import DateTrigger

            if not config.run_date:
                raise HTTPException(status_code=400, detail="run_date required for DATE trigger")

            trigger = DateTrigger(run_date=config.run_date, timezone=timezone_obj)

        # Update the job
        job.modify(
            trigger=trigger,
            max_instances=config.max_instances,
            coalesce=config.coalesce,
            misfire_grace_time=config.misfire_grace_time
        )

        logger.info(f"🔧 Job '{job_id}' configuration updated via API")

        return {
            "status": "success",
            "message": f"Job '{job_id}' configuration updated successfully",
            "new_config": {
                "trigger_type": config.trigger_type,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "max_instances": config.max_instances,
                "coalesce": config.coalesce,
                "misfire_grace_time": config.misfire_grace_time
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update job configuration: {str(e)}")


# ============================================================================
# PREDEFINED JOB MANAGEMENT ENDPOINTS
# ============================================================================

@sip_router.post("/scheduler/jobs/daily-signal-check/configure")
async def configure_daily_signal_check(
        trigger_type: str = "interval",  # "cron" or "interval"
        cron_expression: Optional[str] = None,
        interval_minutes: Optional[int] = 1,
        timezone: str = "Asia/Kolkata",
        max_instances: int = 1
):
    """Configure the daily signal check job with predefined settings"""
    try:
        # Validate inputs
        if trigger_type not in ["cron", "interval"]:
            raise HTTPException(status_code=400, detail="trigger_type must be 'cron' or 'interval'")

        if trigger_type == "cron" and not cron_expression:
            raise HTTPException(status_code=400, detail="cron_expression required for cron trigger")

        if trigger_type == "interval" and not interval_minutes:
            raise HTTPException(status_code=400, detail="interval_minutes required for interval trigger")

        # Remove existing job if it exists
        if scheduler.get_job('daily_signal_check'):
            scheduler.remove_job('daily_signal_check')
            logger.info("Removed existing daily_signal_check job")

        timezone_obj = pytz.timezone(timezone)

        # Create appropriate trigger
        if trigger_type == "cron":
            trigger = CronTrigger.from_crontab(cron_expression, timezone=timezone_obj)
        else:  # interval
            from apscheduler.triggers.interval import IntervalTrigger
            trigger = IntervalTrigger(minutes=interval_minutes, timezone=timezone_obj)

        # Add the job with new configuration
        scheduler.add_job(
            daily_signal_check,
            trigger,
            id='daily_signal_check',
            name='Daily Signal Check',
            replace_existing=True,
            max_instances=max_instances,
            coalesce=True
        )

        # FIXED: Safe access to job details
        job = scheduler.get_job('daily_signal_check')
        next_run_time = None

        if scheduler.running and job and hasattr(job, 'next_run_time') and job.next_run_time:
            next_run_time = job.next_run_time.isoformat()
        elif not scheduler.running:
            next_run_time = "Will be calculated when scheduler starts"

        logger.info(f"🔧 Daily signal check job reconfigured: {trigger}")

        return {
            "status": "success",
            "message": "Daily signal check job configured successfully",
            "configuration": {
                "job_id": "daily_signal_check",
                "trigger_type": trigger_type,
                "trigger_details": str(trigger),
                "next_run_time": next_run_time,
                "max_instances": max_instances,
                "timezone": timezone,
                "scheduler_running": scheduler.running
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring daily signal check: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure daily signal check: {str(e)}")


@sip_router.get("/scheduler/presets")
async def get_scheduler_presets():
    """Get predefined scheduler configurations for common use cases"""
    return {
        "presets": {
            "development": {
                "name": "Development Mode",
                "description": "Frequent execution for testing",
                "daily_signal_check": {
                    "trigger_type": "interval",
                    "interval_minutes": 1,
                    "timezone": "Asia/Kolkata"
                }
            },
            "production_conservative": {
                "name": "Production Conservative",
                "description": "Once daily at market open",
                "daily_signal_check": {
                    "trigger_type": "cron",
                    "cron_expression": "0 9 * * 1-5",  # 9 AM weekdays
                    "timezone": "Asia/Kolkata"
                }
            },
            "production_aggressive": {
                "name": "Production Aggressive",
                "description": "Multiple times during market hours",
                "daily_signal_check": {
                    "trigger_type": "cron",
                    "cron_expression": "0 9,12,15 * * 1-5",  # 9 AM, 12 PM, 3 PM weekdays
                    "timezone": "Asia/Kolkata"
                }
            },
            "testing": {
                "name": "Testing Mode",
                "description": "Every 5 minutes for intensive testing",
                "daily_signal_check": {
                    "trigger_type": "interval",
                    "interval_minutes": 5,
                    "timezone": "Asia/Kolkata"
                }
            }
        },
        "common_cron_expressions": {
            "every_minute": "* * * * *",
            "every_5_minutes": "*/5 * * * *",
            "every_hour": "0 * * * *",
            "market_open": "0 9 * * 1-5",
            "market_close": "30 15 * * 1-5",
            "twice_daily": "0 9,15 * * 1-5",
            "weekly_monday": "0 9 * * 1",
            "end_of_month": "0 9 28-31 * *"
        }
    }


@sip_router.post("/scheduler/apply-preset/{preset_name}")
async def apply_scheduler_preset(preset_name: str):
    """Apply a predefined scheduler configuration"""
    try:
        presets = await get_scheduler_presets()

        if preset_name not in presets["presets"]:
            raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

        preset_config = presets["presets"][preset_name]["daily_signal_check"]

        # Configure the daily signal check job
        await configure_daily_signal_check(
            trigger_type=JobTriggerType(preset_config["trigger_type"]),
            cron_expression=preset_config.get("cron_expression"),
            interval_minutes=preset_config.get("interval_minutes"),
            timezone=preset_config["timezone"]
        )

        logger.info(f"📋 Applied scheduler preset: {preset_name}")

        return {
            "status": "success",
            "message": f"Scheduler preset '{preset_name}' applied successfully",
            "preset_details": presets["presets"][preset_name]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying scheduler preset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply scheduler preset: {str(e)}")


# ============================================================================
# SCHEDULER MONITORING AND METRICS
# ============================================================================

@sip_router.get("/scheduler/metrics")
async def get_scheduler_metrics():
    """Get scheduler performance metrics and statistics"""
    try:
        jobs = scheduler.get_jobs()

        metrics = {
            "scheduler_status": "running" if scheduler.running else "stopped",
            "total_jobs": len(jobs),
            "job_breakdown": {},
            "next_executions": [],
            "system_info": {
                "timezone": str(scheduler.timezone),
                "executor_count": len(scheduler._executors),
                "jobstore_count": len(scheduler._jobstores)
            }
        }

        # Job breakdown by type
        for job in jobs:
            func_name = job.func.__name__
            if func_name not in metrics["job_breakdown"]:
                metrics["job_breakdown"][func_name] = 0
            metrics["job_breakdown"][func_name] += 1

        # Next 10 executions
        next_runs = []
        for job in jobs:
            if job.next_run_time:
                next_runs.append({
                    "job_id": job.id,
                    "function": job.func.__name__,
                    "next_run": job.next_run_time.isoformat(),
                    "trigger_type": type(job.trigger).__name__
                })

        # Sort by next run time
        next_runs.sort(key=lambda x: x["next_run"])
        metrics["next_executions"] = next_runs[:10]

        return metrics

    except Exception as e:
        logger.error(f"Error getting scheduler metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler metrics: {str(e)}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_cron_expression(cron_expr: str) -> bool:
    """Validate cron expression format"""
    try:
        from croniter import croniter
        return croniter.is_valid(cron_expr)
    except ImportError:
        # Fallback validation - basic format check
        parts = cron_expr.split()
        return len(parts) == 5 and all(
            part.replace('*', '1').replace('/', '1').replace('-', '1').replace(',', '1').isdigit() or part == '*' for
            part in parts)


async def get_job_execution_history(job_id: str, limit: int = 50):
    """Get execution history for a specific job (requires job_execution_metrics table)"""
    try:
        async with db_manager.get_session_factory('trading_db')() as db:
            query = text("""
                SELECT execution_start, execution_end, total_time_seconds, 
                       portfolios_processed, signals_generated, errors_count
                FROM job_execution_metrics 
                WHERE job_name = :job_id 
                ORDER BY execution_start DESC 
                LIMIT :limit
            """)

            result = await db.execute(query, {"job_id": job_id, "limit": limit})
            history = result.fetchall()

            return [
                {
                    "execution_start": row[0].isoformat(),
                    "execution_end": row[1].isoformat() if row[1] else None,
                    "duration_seconds": row[2],
                    "portfolios_processed": row[3],
                    "signals_generated": row[4],
                    "errors_count": row[5]
                }
                for row in history
            ]
    except Exception as e:
        logger.error(f"Error getting job execution history: {e}")
        return []


@sip_router.get("/scheduler/jobs/{job_id}/history")
async def get_job_history(job_id: str, limit: int = 50):
    """Get execution history for a specific job"""
    try:
        history = await get_job_execution_history(job_id, limit)

        return {
            "job_id": job_id,
            "total_executions": len(history),
            "history": history
        }

    except Exception as e:
        logger.error(f"Error getting job history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job history: {str(e)}")

# ============================================================================
# EMERGENCY CONTROLS
# ============================================================================

@sip_router.post("/scheduler/emergency-stop")
async def emergency_stop():
    """EMERGENCY: Stop all scheduled jobs immediately"""
    try:
        paused_jobs = []
        for job in scheduler.get_jobs():
            scheduler.pause_job(job.id)
            paused_jobs.append(job.id)

        logger.critical(f"🚨 EMERGENCY STOP: {len(paused_jobs)} jobs paused")

        return {
            "status": "EMERGENCY_STOPPED",
            "message": f"Emergency stop executed - {len(paused_jobs)} jobs paused",
            "paused_jobs": paused_jobs,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")


@sip_router.post("/scheduler/resume-from-emergency")
async def resume_from_emergency():
    """Resume all jobs after emergency stop"""
    try:
        resumed_jobs = []
        for job in scheduler.get_jobs():
            scheduler.resume_job(job.id)
            resumed_jobs.append(job.id)

        logger.info(f"✅ Emergency recovery: {len(resumed_jobs)} jobs resumed")

        return {
            "status": "RECOVERED",
            "message": f"Emergency recovery completed - {len(resumed_jobs)} jobs resumed",
            "resumed_jobs": resumed_jobs,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in emergency recovery: {e}")
        raise HTTPException(status_code=500, detail=f"Emergency recovery failed: {str(e)}")


# ============================================================================
# TRIGGER MANUAL EXECUTION
# ============================================================================

@sip_router.post("/scheduler/trigger/daily-signal-check")
async def trigger_signal_check_now():
    """Run the daily signal check immediately in background"""
    try:
        # Run the job immediately in background
        asyncio.create_task(daily_signal_check())

        logger.info("🚀 Daily signal check triggered manually")

        return {
            "status": "triggered",
            "message": "Daily signal check started immediately in background",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error triggering signal check: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger signal check: {str(e)}")


# ============================================================================
# DEVELOPMENT HELPERS
# ============================================================================

@sip_router.post("/scheduler/dev/create-test-job")
async def create_test_job():
    """Create a simple test job for development testing"""
    try:
        async def test_job():
            logger.info("🧪 Test job executed!")
            return "Test job completed"

        scheduler.add_job(
            test_job,
            CronTrigger(minute='*/2', timezone=pytz.timezone('Asia/Kolkata')),
            id='test_job',
            name='Test Job',
            replace_existing=True,
            max_instances=1
        )

        return {
            "status": "success",
            "message": "Test job created (runs every 2 minutes)",
            "job_id": "test_job"
        }

    except Exception as e:
        logger.error(f"Error creating test job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create test job: {str(e)}")


@sip_router.delete("/scheduler/dev/remove-test-job")
async def remove_test_job():
    """Remove the test job"""
    try:
        if scheduler.get_job('test_job'):
            scheduler.remove_job('test_job')
            return {"status": "success", "message": "Test job removed"}
        else:
            return {"status": "info", "message": "Test job not found"}

    except Exception as e:
        logger.error(f"Error removing test job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove test job: {str(e)}")