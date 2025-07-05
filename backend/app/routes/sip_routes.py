"""
Complete SIP Strategy API Routes with fixed data fetching and all missing endpoints
Addresses: Correct table name handling, proper database queries, missing endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, desc
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, date
from pydantic import BaseModel, validator
import json
import logging
import uuid
import numpy as np
import pandas as pd
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Clean imports - multi-database architecture
from backend.app.database import get_db, get_nsedata_db
from backend.app.auth import UserManager, oauth2_scheme
from backend.app.strategies.enhanced_sip_strategy import (EnhancedSIPStrategy, SIPConfig, Trade,
                                                          EnhancedSIPStrategyWithLimits)

logger = logging.getLogger(__name__)

# Global scheduler for background tasks
scheduler = AsyncIOScheduler()


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
    max_amount_in_a_month: Optional[float] = None  # Defaults to 4x fixed_investment
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


class BenchmarkSIPCalculator:
    """
    Calculate regular SIP benchmark performance
    Fixed ₹5000 investment on 15th of every month (no conditions)
    """

    def __init__(self, monthly_amount: float = 5000, investment_day: int = 15):
        self.monthly_amount = monthly_amount
        self.investment_day = investment_day

    async def calculate_benchmark(self, symbol: str, start_date: str, end_date: str,
                                  nsedata_db: AsyncSession) -> Dict:
        """Calculate benchmark SIP performance for comparison"""
        try:
            logger.info(f"🎯 Calculating benchmark SIP for {symbol}: ₹{self.monthly_amount} on {self.investment_day}th")

            # Fetch the same data as strategy
            data = await self._fetch_data_from_db_async(symbol, start_date, end_date, nsedata_db)

            if data.empty:
                logger.warning(f"No data available for benchmark calculation: {symbol}")
                return self._empty_benchmark_result()

            # Initialize tracking variables
            total_investment = 0.0
            total_units = 0.0
            benchmark_trades = []

            # Track months we've already invested in
            invested_months = set()

            # Simulate regular monthly SIP
            for i, row in data.iterrows():
                current_date = row['timestamp']
                current_price = row['close']

                # Create month key (YYYY-MM)
                month_key = f"{current_date.year}-{current_date.month:02d}"

                # Check if we should invest this month
                should_invest = False

                # Primary condition: 15th of the month
                if current_date.day == self.investment_day and month_key not in invested_months:
                    should_invest = True

                # Fallback: Last available day of month if 15th not available
                elif month_key not in invested_months:
                    # Check if this is the last day of data for this month
                    next_day_month = None
                    if i + 1 < len(data):
                        next_day_month = data.iloc[i + 1]['timestamp'].month

                    # If next day is different month or this is last data point
                    if next_day_month != current_date.month or i == len(data) - 1:
                        # And we haven't invested this month yet
                        should_invest = True

                if should_invest:
                    # Execute benchmark investment
                    units_bought = self.monthly_amount / current_price
                    total_investment += self.monthly_amount
                    total_units += units_bought

                    # Record trade
                    trade = {
                        'date': current_date.strftime('%Y-%m-%d'),
                        'price': float(current_price),
                        'units': float(units_bought),
                        'amount': float(self.monthly_amount),
                        'total_investment': float(total_investment),
                        'total_units': float(total_units),
                        'portfolio_value': float(total_units * current_price),
                        'trade_type': 'regular_sip'
                    }
                    benchmark_trades.append(trade)

                    # Mark month as invested
                    invested_months.add(month_key)

                    logger.debug(f"📅 Benchmark SIP: {current_date.strftime('%Y-%m-%d')} "
                                 f"₹{self.monthly_amount:,.2f} @ ₹{current_price:.2f}")

            # Calculate final portfolio value
            final_price = data.iloc[-1]['close']
            final_portfolio_value = total_units * final_price

            # Calculate performance metrics
            total_return_percent = ((final_portfolio_value / total_investment) - 1) * 100 if total_investment > 0 else 0

            # Calculate CAGR
            start_timestamp = data.iloc[0]['timestamp']
            end_timestamp = data.iloc[-1]['timestamp']
            years = (end_timestamp - start_timestamp).days / 365.25
            cagr_percent = ((final_portfolio_value / total_investment) ** (
                        1 / years) - 1) * 100 if years > 0 and total_investment > 0 else 0

            # Calculate average buy price
            avg_buy_price = total_investment / total_units if total_units > 0 else 0

            benchmark_result = {
                'strategy_name': 'Regular SIP Benchmark',
                'description': f'₹{self.monthly_amount:,.0f} invested on {self.investment_day}th of every month',
                'total_investment': float(total_investment),
                'final_portfolio_value': float(final_portfolio_value),
                'total_units': float(total_units),
                'average_buy_price': float(avg_buy_price),
                'total_return_percent': float(total_return_percent),
                'cagr_percent': float(cagr_percent),
                'num_trades': len(benchmark_trades),
                'trades': benchmark_trades,
                'final_price': float(final_price),
                'period': f"{start_date} to {end_date}",
                'monthly_investment': float(self.monthly_amount),
                'investment_day': self.investment_day
            }

            logger.info(f"✅ Benchmark SIP completed for {symbol}:")
            logger.info(f"   📊 Investment: ₹{total_investment:,.2f}")
            logger.info(f"   💰 Final Value: ₹{final_portfolio_value:,.2f}")
            logger.info(f"   📈 CAGR: {cagr_percent:.2f}%")
            logger.info(f"   🔄 Total Trades: {len(benchmark_trades)}")

            return benchmark_result

        except Exception as e:
            logger.error(f"Error calculating benchmark SIP for {symbol}: {e}")
            return self._empty_benchmark_result()

    async def _fetch_data_from_db_async(self, symbol: str, start_date: str, end_date: str,
                                        nsedata_db: AsyncSession) -> pd.DataFrame:
        """Fetch market data for benchmark calculation"""
        try:
            # Convert string dates to datetime objects
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()

            query = text(f"""
                SELECT timestamp, open, high, low, close, volume
                FROM "{symbol.upper()}"
                WHERE timestamp BETWEEN :start_date AND :end_date
                ORDER BY timestamp ASC
            """)

            result = await nsedata_db.execute(query, {
                'start_date': start_date_obj,
                'end_date': end_date_obj
            })

            rows = result.fetchall()

            if not rows:
                logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()

            # Convert to DataFrame
            data = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

            return data.dropna()

        except Exception as e:
            logger.error(f"Error fetching data for benchmark: {e}")
            return pd.DataFrame()

    def _empty_benchmark_result(self) -> Dict:
        """Return empty benchmark result for error cases"""
        return {
            'strategy_name': 'Regular SIP Benchmark',
            'description': f'₹{self.monthly_amount:,.0f} invested on {self.investment_day}th of every month',
            'total_investment': 0.0,
            'final_portfolio_value': 0.0,
            'total_units': 0.0,
            'average_buy_price': 0.0,
            'total_return_percent': 0.0,
            'cagr_percent': 0.0,
            'num_trades': 0,
            'trades': [],
            'final_price': 0.0,
            'period': '',
            'monthly_investment': float(self.monthly_amount),
            'investment_day': self.investment_day,
            'error': 'No data available for benchmark calculation'
        }


# Create router
sip_router = APIRouter(prefix="/sip", tags=["sip-strategy"])


# ============================================================================
# BACKGROUND SIGNAL PROCESSING
# ============================================================================

async def daily_signal_check():
    """Background task to check for signals daily at 8 AM - FIXED JSON handling"""
    logger.info("🔄 Starting daily signal check at 8 AM")

    try:
        # Get database sessions
        from backend.app.database import AsyncSessionLocal, AsyncNSESessionLocal

        async with AsyncSessionLocal() as trading_db:
            async with AsyncNSESessionLocal() as nsedata_db:
                # Get all active portfolios
                portfolios_query = text("""
                    SELECT portfolio_id, user_id, symbols, config, 
                           total_invested, current_units, next_investment_date
                    FROM sip_portfolios 
                    WHERE status = 'active'
                """)

                result = await trading_db.execute(portfolios_query)
                portfolios = result.fetchall()

                logger.info(f"Processing {len(portfolios)} active portfolios")

                strategy = EnhancedSIPStrategy(
                    nsedata_session=nsedata_db,
                    trading_session=trading_db
                )

                for portfolio in portfolios:
                    portfolio_id, user_id, symbols_json, config_json, total_invested, current_units, next_investment_date = portfolio

                    try:
                        # FIXED: Safe JSON parsing
                        if isinstance(symbols_json, str):
                            symbols_data = json.loads(symbols_json)
                        else:
                            symbols_data = symbols_json

                        if isinstance(config_json, str):
                            config_dict = json.loads(config_json)
                        else:
                            config_dict = config_json

                        config = SIPConfig(**config_dict)

                        # Process each symbol in the portfolio
                        for symbol_config in symbols_data:
                            if isinstance(symbol_config, dict):
                                symbol = symbol_config.get('symbol')
                            else:
                                symbol = symbol_config  # Direct string

                            if not symbol:
                                logger.warning(f"Skipping invalid symbol config: {symbol_config}")
                                continue

                            # Check if it's time for next investment
                            current_date = datetime.now().date()

                            # Skip if next investment date is in future
                            if next_investment_date and next_investment_date.date() > current_date:
                                continue

                            # Check last investment date to ensure minimum gap
                            last_trade_query = text("""
                                SELECT MAX(timestamp) FROM sip_actual_trades 
                                WHERE portfolio_id = :portfolio_id AND symbol = :symbol
                            """)

                            last_trade_result = await trading_db.execute(
                                last_trade_query, {'portfolio_id': portfolio_id, 'symbol': symbol}
                            )
                            last_trade_date = last_trade_result.scalar()

                            # Enforce minimum gap
                            if last_trade_date:
                                days_since_last = (current_date - last_trade_date.date()).days
                                if days_since_last < config.min_investment_gap_days:
                                    logger.info(
                                        f"Skipping {symbol} - only {days_since_last} days since last investment")
                                    continue

                            # Generate signals
                            try:
                                end_date = datetime.now().strftime('%Y-%m-%d')
                                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

                                data = await strategy.fetch_data_from_db_async(symbol, start_date, end_date)
                                if not data.empty:
                                    signals = strategy.get_next_investment_signals(data, config)

                                    # Save signal if valid
                                    if signals.get('signal') not in ['NO_DATA', 'ERROR']:
                                        await save_signal_with_gtt_order(portfolio_id, symbol, signals, trading_db,
                                                                         config)
                                        logger.info(f"✅ Generated daily signal for {symbol}")

                            except Exception as signal_error:
                                logger.error(f"Error generating daily signal for {symbol}: {signal_error}")

                    except Exception as portfolio_error:
                        logger.error(f"Error processing portfolio {portfolio_id}: {portfolio_error}")

    except Exception as e:
        logger.error(f"Critical error in daily signal check: {e}")


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


async def save_signal_with_gtt_order(portfolio_id: str, symbol: str, signals: Dict[str, Any],
                                     trading_db: AsyncSession, config: SIPConfig):
    """Save investment signal with GTT order - FIXED JSON handling"""
    try:
        signal_id = f"sig_{portfolio_id}_{symbol}_{int(datetime.now().timestamp())}"

        # Determine signal strength
        confidence = signals.get('confidence', 0)
        if confidence > 0.7:
            signal_strength = "high"
        elif confidence > 0.4:
            signal_strength = "medium"
        else:
            signal_strength = "low"

        # Calculate GTT trigger price (slightly below current for dip buying)
        current_price = signals.get('current_price', 0)
        gtt_trigger_price = current_price * 0.98  # 2% below current price

        # Create signals table if not exists
        create_signals_table = text("""
            CREATE TABLE IF NOT EXISTS sip_signals (
                signal_id VARCHAR PRIMARY KEY,
                portfolio_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                signal_type VARCHAR NOT NULL,
                recommended_amount FLOAT,
                multiplier FLOAT,
                current_price FLOAT,
                drawdown_percent FLOAT,
                signal_strength VARCHAR,
                is_processed BOOLEAN DEFAULT FALSE,
                gtt_order_id VARCHAR,
                gtt_trigger_price FLOAT,
                signal_data JSONB,  -- Store full signal data
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        await trading_db.execute(create_signals_table)

        # FIXED: Proper JSON serialization for signal_data
        try:
            # Ensure all values in signals are JSON serializable
            serializable_signals = {}
            for key, value in signals.items():
                if isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                    serializable_signals[key] = value
                elif hasattr(value, 'isoformat'):  # datetime objects
                    serializable_signals[key] = value.isoformat()
                else:
                    serializable_signals[key] = str(value)  # Convert to string as fallback

            signal_data_json = json.dumps(serializable_signals)

        except Exception as json_error:
            logger.error(f"JSON serialization error for signals: {json_error}")
            # Fallback to basic signal data
            signal_data_json = json.dumps({
                "signal": signals.get('signal', 'UNKNOWN'),
                "confidence": signals.get('confidence', 0),
                "message": signals.get('message', 'Signal data serialization failed')
            })

        insert_signal = text("""
            INSERT INTO sip_signals 
            (signal_id, portfolio_id, symbol, signal_type, recommended_amount, 
             multiplier, current_price, drawdown_percent, signal_strength, 
             gtt_trigger_price, signal_data, created_at, expires_at)
            VALUES (:signal_id, :portfolio_id, :symbol, :signal_type, 
                    :recommended_amount, :multiplier, :current_price, 
                    :drawdown_percent, :signal_strength, :gtt_trigger_price,
                    :signal_data, :created_at, :expires_at)
        """)

        await trading_db.execute(insert_signal, {
            'signal_id': signal_id,
            'portfolio_id': portfolio_id,
            'symbol': symbol,
            'signal_type': signals.get('signal', 'NORMAL'),
            'recommended_amount': signals.get('recommended_amount', config.fixed_investment),
            'multiplier': signals.get('confidence', 1.0),
            'current_price': current_price,
            'drawdown_percent': signals.get('drawdown_100'),
            'signal_strength': signal_strength,
            'gtt_trigger_price': gtt_trigger_price,
            'signal_data': signal_data_json,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(days=7)
        })

        await trading_db.commit()
        logger.info(f"✅ Signal and GTT order created for portfolio {portfolio_id}, symbol {symbol}")

    except Exception as e:
        logger.error(f"Error saving signal with GTT order: {e}")
        await trading_db.rollback()
        raise  # Re-raise to handle in calling function


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


# Start the scheduler
@sip_router.on_event("startup")
async def start_scheduler():
    """Start the background scheduler"""
    scheduler.add_job(
        daily_signal_check,
        CronTrigger(hour=8, minute=0),  # Daily at 8:00 AM
        id='daily_signal_check',
        replace_existing=True
    )
    scheduler.start()
    logger.info("✅ Background scheduler started - daily signal check at 8:00 AM")


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
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
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
        strategy = EnhancedSIPStrategyWithLimits(
            nsedata_session=nsedata_db,
            trading_session=trading_db
        )

        # Create benchmark calculator
        benchmark_calculator = BenchmarkSIPCalculator(
            monthly_amount=5000,
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
            backtest_id = str(uuid.uuid4())

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
                monthly_summary = json.loads(row[13]) if row[13] else {}
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


@sip_router.get("/analytics/monthly-limits/{symbol}")
async def get_monthly_limits_analytics(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get detailed analytics for monthly investment limits for a specific symbol"""
    try:
        # Base query for getting backtest results with monthly data
        query = """
            SELECT backtest_id, total_investment, final_portfolio_value,
                   max_amount_in_a_month, monthly_limit_exceeded, 
                   price_threshold_skipped, monthly_summary, trades,
                   skipped_investments, created_at
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
            monthly_summary = json.loads(row[6]) if row[6] else {}
            trades = json.loads(row[7]) if row[7] else []
            skipped_investments = json.loads(row[8]) if row[8] else []

            # Calculate monthly utilization
            for month, data in monthly_summary.items():
                utilization_percent = (data.get('total_invested', 0) / row[3]) * 100 if row[3] > 0 else 0
                monthly_utilization_data.append({
                    "month": month,
                    "utilization_percent": round(utilization_percent, 2),
                    "amount_invested": data.get('total_invested', 0),
                    "monthly_limit": row[3],
                    "num_investments": data.get('num_investments', 0)
                })

            # Calculate opportunity cost for skipped investments
            total_skipped_amount = sum(skip.get('intended_amount', 0) for skip in skipped_investments)
            if total_skipped_amount > 0 and trades:
                # Use final portfolio value to estimate opportunity cost
                final_price = trades[-1].get('price', 0) if trades else 0
                if final_price > 0:
                    estimated_units = sum(skip.get('intended_amount', 0) / skip.get('price', 1)
                                          for skip in skipped_investments if skip.get('price', 0) > 0)
                    estimated_value = estimated_units * final_price
                    opportunity_cost = estimated_value - total_skipped_amount

                    opportunity_cost_analysis.append({
                        "backtest_id": row[0],
                        "total_skipped_amount": total_skipped_amount,
                        "estimated_final_value": estimated_value,
                        "opportunity_cost": opportunity_cost,
                        "opportunity_cost_percent": (opportunity_cost / total_skipped_amount) * 100
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


@sip_router.post("/optimize-config")
async def optimize_sip_config(
        symbol: str,
        target_monthly_utilization: float = 80.0,
        risk_tolerance: str = "moderate",  # conservative, moderate, aggressive
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """
    Optimize SIP configuration based on historical performance and user preferences

    This endpoint analyzes past backtest results and suggests optimal configuration
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

        # Analyze historical performance
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
        monthly_summary = json.loads(row[7]) if row[7] else {}
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

@sip_router.post("/batch-backtest")
async def run_batch_backtest_with_limits(
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
            "batch_id": str(uuid.uuid4()),
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
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
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
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
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


@sip_router.get("/health-check")
async def sip_strategy_health_check():
    """Health check endpoint for SIP strategy system"""
    return {
        "status": "healthy",
        "version": "2.0.0-enhanced",
        "features": {
            "monthly_investment_limits": "active",
            "price_reduction_threshold": "active",
            "enhanced_tracking": "active",
            "batch_processing": "active",
            "config_optimization": "active"
        },
        "timestamp": datetime.now().isoformat()
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