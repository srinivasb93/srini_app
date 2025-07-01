# backend/app/routes/sip_routes.py - Comprehensive Implementation
"""
Complete SIP Strategy API Routes with all endpoints for frontend integration
Includes: Backtesting, Portfolio Management, Signal Generation, Performance Analytics
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, desc
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, validator
import json
import logging
import uuid

# Clean imports - multi-database architecture
from backend.app.database import get_db, get_nsedata_db
from backend.app.auth import UserManager, oauth2_scheme
from backend.app.strategies.enhanced_sip_strategy import EnhancedSIPStrategy, SIPConfig, Trade

logger = logging.getLogger(__name__)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    user_id = UserManager.verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id


# ============================================================================
# PYDANTIC MODELS - Complete Set
# ============================================================================

class SIPConfigRequest(BaseModel):
    fixed_investment: float = 5000
    drawdown_threshold_1: float = -10.0
    drawdown_threshold_2: float = -4.0
    investment_multiplier_1: float = 2.0
    investment_multiplier_2: float = 3.0
    investment_multiplier_3: float = 5.0
    rolling_window: int = 100
    fallback_day: int = 22

    @validator('fixed_investment')
    def validate_investment(cls, v):
        if v <= 0:
            raise ValueError('Investment amount must be positive')
        return v


class SIPBacktestRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    config: SIPConfigRequest

    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError('At least one symbol required')
        return v


class SIPPortfolioRequest(BaseModel):
    symbol: str
    portfolio_name: Optional[str] = None
    config: SIPConfigRequest


class SIPBacktestResponse(BaseModel):
    backtest_id: str
    symbol: str
    strategy_name: str
    total_investment: float
    final_portfolio_value: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    volatility: Optional[float]
    num_trades: int
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


# Create router
sip_router = APIRouter(prefix="/sip", tags=["sip-strategy"])


# ============================================================================
# DATABASE TESTING ENDPOINTS
# ============================================================================

@sip_router.get("/health")
async def health_check():
    """SIP service health check"""
    return {
        "status": "healthy",
        "service": "Comprehensive SIP Strategy API",
        "version": "3.0.0",
        "architecture": "multi-database",
        "features": [
            "Enhanced technical indicators",
            "Dynamic investment amounts",
            "Portfolio management",
            "Real-time signal generation",
            "Performance analytics",
            "Multi-strategy comparison",
            "Live portfolio tracking"
        ],
        "timestamp": datetime.now().isoformat()
    }


@sip_router.get("/database/status")
async def check_database_status(
        trading_db: AsyncSession = Depends(get_db),
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Check database connectivity and status"""
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
# CORE BACKTESTING ENDPOINTS
# ============================================================================

@sip_router.post("/backtest", response_model=List[SIPBacktestResponse])
async def run_sip_backtest(
        request: SIPBacktestRequest,
        background_tasks: BackgroundTasks,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db),
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Run comprehensive SIP strategy backtest"""
    try:
        # Create strategy with proper database sessions
        strategy = EnhancedSIPStrategy(
            nsedata_session=nsedata_db,
            trading_session=trading_db
        )
        config = SIPConfig(**request.config.dict())

        logger.info(f"🚀 Starting SIP backtest for user {user_id}")
        logger.info(f"📊 Symbols: {request.symbols}")
        logger.info(f"📅 Period: {request.start_date} to {request.end_date}")

        # Run backtest
        results = await strategy.run_batch_backtest(
            request.symbols,
            request.start_date,
            request.end_date,
            config
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No data found for the specified symbols and date range"
            )

        logger.info(f"✅ Backtest completed for {len(results)} symbols")

        # Save results in background
        background_tasks.add_task(save_backtest_results, results, user_id, trading_db)

        # Convert to response format
        response_data = []
        for symbol, result in results.items():
            response_data.append(SIPBacktestResponse(
                backtest_id=f"bt_{user_id}_{symbol}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                strategy_name=result.strategy_name,
                total_investment=result.total_investment,
                final_portfolio_value=result.final_portfolio_value,
                cagr=result.cagr,
                max_drawdown=result.max_drawdown,
                sharpe_ratio=result.sharpe_ratio,
                volatility=result.volatility,
                num_trades=len(result.trades),
                created_at=datetime.now()
            ))

        return response_data

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@sip_router.get("/backtest/history", response_model=List[SIPBacktestResponse])
async def get_sip_backtest_history(
        limit: int = 10,
        offset: int = 0,
        symbol: Optional[str] = None,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get SIP backtest history for user"""
    try:
        base_query = """
            SELECT backtest_id, symbol, strategy_name, total_investment, 
                   final_portfolio_value, cagr, max_drawdown, sharpe_ratio, 
                   volatility, num_trades, created_at
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

        return [
            SIPBacktestResponse(
                backtest_id=row[0],
                symbol=row[1],
                strategy_name=row[2],
                total_investment=row[3],
                final_portfolio_value=row[4],
                cagr=row[5],
                max_drawdown=row[6],
                sharpe_ratio=row[7],
                volatility=row[8],
                num_trades=row[9],
                created_at=row[10]
            ) for row in backtest_results
        ]

    except Exception as e:
        logger.error(f"Error fetching backtest history: {e}")
        return []


# ============================================================================
# PORTFOLIO MANAGEMENT ENDPOINTS
# ============================================================================

@sip_router.post("/portfolio", response_model=Dict[str, str])
async def create_sip_portfolio(
        request: SIPPortfolioRequest,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Create a new SIP portfolio for live tracking"""
    try:
        # Create portfolios table if it doesn't exist
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS sip_portfolios (
                portfolio_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                portfolio_name VARCHAR,
                config JSON NOT NULL,
                status VARCHAR DEFAULT 'active',
                total_invested FLOAT DEFAULT 0,
                current_units FLOAT DEFAULT 0,
                current_value FLOAT DEFAULT 0,
                next_investment_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await trading_db.execute(create_table_query)

        # Generate portfolio ID
        portfolio_id = f"pf_{user_id}_{request.symbol}_{int(datetime.now().timestamp())}"

        # Insert portfolio
        config = SIPConfig(**request.config.dict())
        insert_query = text("""
            INSERT INTO sip_portfolios 
            (portfolio_id, user_id, symbol, portfolio_name, config, status, created_at)
            VALUES (:portfolio_id, :user_id, :symbol, :portfolio_name, :config, :status, :created_at)
        """)

        await trading_db.execute(insert_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id,
            'symbol': request.symbol,
            'portfolio_name': request.portfolio_name or f"{request.symbol} SIP Portfolio",
            'config': json.dumps(config.__dict__),
            'status': 'active',
            'created_at': datetime.now()
        })

        await trading_db.commit()

        return {"portfolio_id": portfolio_id, "status": "created"}

    except Exception as e:
        logger.error(f"Error creating SIP portfolio: {e}")
        await trading_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.get("/portfolio", response_model=List[SIPPortfolioResponse])
async def get_sip_portfolios(
        status: Optional[str] = None,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get user's SIP portfolios"""
    try:
        base_query = """
            SELECT portfolio_id, symbol, portfolio_name, status, 
                   total_invested, current_units, current_value, 
                   next_investment_date, created_at
            FROM sip_portfolios 
            WHERE user_id = :user_id
        """
        params = {"user_id": user_id}

        if status:
            base_query += " AND status = :status"
            params["status"] = status

        base_query += " ORDER BY created_at DESC"

        result = await trading_db.execute(text(base_query), params)
        portfolios = result.fetchall()

        return [
            SIPPortfolioResponse(
                portfolio_id=row[0],
                symbol=row[1],
                portfolio_name=row[2],
                status=row[3],
                total_invested=row[4] or 0,
                current_units=row[5] or 0,
                current_value=row[6] or 0,
                next_investment_date=row[7],
                created_at=row[8]
            ) for row in portfolios
        ]

    except Exception as e:
        logger.error(f"Error fetching SIP portfolios: {e}")
        return []


@sip_router.get("/performance/{portfolio_id}")
async def get_portfolio_performance(
        portfolio_id: str,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get performance metrics for a SIP portfolio"""
    try:
        # Verify portfolio ownership
        portfolio_query = text("""
            SELECT symbol, total_invested, current_units, created_at 
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

        symbol, total_invested, current_units, created_at = portfolio_data

        # Get trade history
        trades_query = text("""
            SELECT timestamp, price, units, amount, trade_type
            FROM sip_actual_trades 
            WHERE portfolio_id = :portfolio_id
            ORDER BY timestamp ASC
        """)

        result = await trading_db.execute(trades_query, {'portfolio_id': portfolio_id})
        trades = result.fetchall()

        # Calculate performance metrics (using placeholder current price)
        current_price = 150.0  # You'd fetch this from market data API
        current_value = (current_units or 0) * current_price
        total_invested = total_invested or 0

        total_return = (current_value - total_invested) / total_invested if total_invested > 0 else 0

        # Calculate CAGR
        days_invested = (datetime.now() - created_at).days
        years_invested = days_invested / 365.25
        cagr = (current_value / total_invested) ** (
                    1 / years_invested) - 1 if years_invested > 0 and total_invested > 0 else 0

        # Average buy price
        avg_buy_price = total_invested / current_units if current_units and current_units > 0 else 0

        return {
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "total_invested": total_invested,
            "current_units": current_units or 0,
            "current_value": current_value,
            "current_price": current_price,
            "total_return_percent": total_return * 100,
            "cagr_percent": cagr * 100,
            "average_buy_price": avg_buy_price,
            "num_investments": len(trades),
            "days_invested": days_invested,
            "unrealized_pnl": current_value - total_invested
        }

    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
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
    """Get investment signals for a SIP portfolio"""
    try:
        # Verify portfolio ownership
        portfolio_query = text("""
            SELECT symbol, config FROM sip_portfolios 
            WHERE portfolio_id = :portfolio_id AND user_id = :user_id
        """)

        result = await trading_db.execute(portfolio_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id
        })
        portfolio_data = result.fetchone()

        if not portfolio_data:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        symbol, config_json = portfolio_data
        config = SIPConfig(**json.loads(config_json))

        # Generate signals
        strategy = EnhancedSIPStrategy(nsedata_session=nsedata_db)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        data = await strategy.fetch_data_from_db_async(symbol, start_date, end_date)

        if data.empty:
            raise HTTPException(status_code=404, detail="No recent data available")

        signals = strategy.get_next_investment_signals(data, config)

        # Save signal to database if investment is recommended
        if signals.get('signal') not in ['NO_DATA', 'ERROR']:
            await save_signal_to_db(portfolio_id, symbol, signals, trading_db)

        return signals

    except Exception as e:
        logger.error(f"Error getting investment signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.get("/signals", response_model=List[SIPSignalResponse])
async def get_all_signals(
        active_only: bool = True,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Get all investment signals for user's portfolios"""
    try:
        base_query = """
            SELECT s.signal_id, s.symbol, s.signal_type, s.recommended_amount,
                   s.multiplier, s.current_price, s.drawdown_percent, 
                   s.signal_strength, s.created_at
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

        base_query += " ORDER BY s.created_at DESC"

        result = await trading_db.execute(text(base_query), params)
        signals = result.fetchall()

        return [
            SIPSignalResponse(
                signal_id=row[0],
                symbol=row[1],
                signal_type=row[2],
                recommended_amount=row[3],
                multiplier=row[4],
                current_price=row[5],
                drawdown_percent=row[6],
                signal_strength=row[7],
                created_at=row[8]
            ) for row in signals
        ]

    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return []


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
    """Execute SIP investment for a portfolio"""
    try:
        # Verify portfolio and get details
        portfolio_query = text("""
            SELECT symbol, config FROM sip_portfolios 
            WHERE portfolio_id = :portfolio_id AND user_id = :user_id AND status = 'active'
        """)

        result = await trading_db.execute(portfolio_query, {
            'portfolio_id': portfolio_id,
            'user_id': user_id
        })
        portfolio_data = result.fetchone()

        if not portfolio_data:
            raise HTTPException(status_code=404, detail="Active portfolio not found")

        symbol, config_json = portfolio_data
        config = SIPConfig(**json.loads(config_json))

        # Use provided amount or default from config
        investment_amount = amount or config.fixed_investment

        # Placeholder for current price (integrate with your market data API)
        current_price = 100.0
        units = investment_amount / current_price

        # Create actual trades table if it doesn't exist
        create_trades_table = text("""
            CREATE TABLE IF NOT EXISTS sip_actual_trades (
                trade_id VARCHAR PRIMARY KEY,
                portfolio_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                price FLOAT NOT NULL,
                units FLOAT NOT NULL,
                amount FLOAT NOT NULL,
                trade_type VARCHAR DEFAULT 'Manual',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await trading_db.execute(create_trades_table)

        # Log the trade
        trade_id = f"tr_{portfolio_id}_{int(datetime.now().timestamp())}"
        insert_trade = text("""
            INSERT INTO sip_actual_trades 
            (trade_id, portfolio_id, symbol, timestamp, price, units, amount, trade_type, created_at)
            VALUES (:trade_id, :portfolio_id, :symbol, :timestamp, :price, :units, :amount, :trade_type, :created_at)
        """)

        await trading_db.execute(insert_trade, {
            'trade_id': trade_id,
            'portfolio_id': portfolio_id,
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': current_price,
            'units': units,
            'amount': investment_amount,
            'trade_type': 'Manual',
            'created_at': datetime.now()
        })

        # Update portfolio totals
        update_query = text("""
            UPDATE sip_portfolios 
            SET total_invested = total_invested + :amount,
                current_units = current_units + :units,
                current_value = (current_units + :units) * :price,
                updated_at = :now
            WHERE portfolio_id = :portfolio_id
        """)

        await trading_db.execute(update_query, {
            'amount': investment_amount,
            'units': units,
            'price': current_price,
            'now': datetime.now(),
            'portfolio_id': portfolio_id
        })

        await trading_db.commit()

        return {
            "status": "success",
            "trade_id": trade_id,
            "investment_amount": investment_amount,
            "units_purchased": units,
            "price": current_price
        }

    except Exception as e:
        logger.error(f"Error executing SIP investment: {e}")
        await trading_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STRATEGY COMPARISON ENDPOINTS
# ============================================================================

@sip_router.get("/strategies/compare/{symbol}")
async def compare_strategies(
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        user_id: str = Depends(get_current_user),
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Compare different SIP strategy configurations"""
    try:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        strategy = EnhancedSIPStrategy(nsedata_session=nsedata_db)

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

        results = {}
        for strategy_name, config in strategies.items():
            logger.info(f"Running {strategy_name} strategy for {symbol}")

            backtest_results = await strategy.run_batch_backtest([symbol], start_date, end_date, config)

            if symbol in backtest_results:
                result = backtest_results[symbol]
                roi = ((result.final_portfolio_value / result.total_investment) - 1) * 100

                results[strategy_name] = {
                    "total_investment": result.total_investment,
                    "final_value": result.final_portfolio_value,
                    "total_return_percent": roi,
                    "cagr_percent": result.cagr * 100,
                    "max_drawdown_percent": result.max_drawdown * 100 if result.max_drawdown else 0,
                    "sharpe_ratio": result.sharpe_ratio,
                    "volatility_percent": result.volatility * 100 if result.volatility else 0,
                    "num_trades": len(result.trades),
                    "avg_buy_price": result.average_buy_price
                }

        return {
            "symbol": symbol,
            "analysis_period": f"{start_date} to {end_date}",
            "strategies": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Strategy comparison failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MARKET DATA AND UTILITIES
# ============================================================================

@sip_router.get("/symbols")
async def get_available_symbols(
        limit: int = 100,
        offset: int = 0,
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Get available symbols with pagination"""
    try:
        # Get symbols with pagination
        symbols_query = text("""
            SELECT tablename
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tableowner != 'postgres'
            AND tablename ~ '^[A-Z][A-Z0-9]*$'
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
    """Get market data information for a symbol"""
    try:
        # Convert string dates to date objects for PostgreSQL
        def safe_date_convert(date_str: str):
            try:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except Exception:
                return datetime.now().date()

        # Get comprehensive symbol information
        query = text(f"""
            SELECT 
                COUNT(*) as total_records,
                MIN(timestamp) as data_start,
                MAX(timestamp) as data_end,
                AVG(close) as avg_price,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(volume) as avg_volume
            FROM public."{symbol}"
        """)

        result = await nsedata_db.execute(query)
        stats = result.fetchone()

        # Get recent data sample
        recent_query = text(f"""
            SELECT timestamp, open, high, low, close, volume
            FROM public."{symbol}"
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
# QUICK TESTING ENDPOINT (FIXED)
# ============================================================================

@sip_router.post("/quick-test")
async def quick_sip_test(
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        investment_amount: float = 5000,
        user_id: str = Depends(get_current_user),
        nsedata_db: AsyncSession = Depends(get_nsedata_db)
):
    """Quick SIP test without full backtesting - FIXED VERSION"""
    try:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # FIXED: Convert string dates to date objects
        def safe_date_convert(date_str: str):
            try:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except Exception as e:
                logger.error(f"Date conversion error: {e}")
                return datetime.now().date()

        start_datetime = safe_date_convert(start_date)
        end_datetime = safe_date_convert(end_date)

        results = {}

        for symbol in symbols:
            try:
                # FIXED: Simple query with proper date handling
                query = text(f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        AVG(close) as avg_price,
                        MIN(close) as min_price,
                        MAX(close) as max_price,
                        STDDEV(close) as volatility
                    FROM public."{symbol}"
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                """)

                result = await nsedata_db.execute(query, {
                    'start_date': start_datetime,
                    'end_date': end_datetime
                })

                row = result.fetchone()

                if row and row[0] > 0:
                    total_rows = row[0]
                    avg_price = float(row[1]) if row[1] else 0
                    min_price = float(row[2]) if row[2] else 0
                    max_price = float(row[3]) if row[3] else 0
                    volatility = float(row[4]) if row[4] else 0

                    # Estimate SIP performance
                    estimated_months = max(1, total_rows // 20)  # Rough monthly estimation
                    total_investment = investment_amount * estimated_months

                    if avg_price > 0:
                        estimated_units = total_investment / avg_price
                        estimated_value = estimated_units * avg_price  # Conservative estimate
                        simple_return = ((max_price / min_price) - 1) * 100 if min_price > 0 else 0
                    else:
                        estimated_units = 0
                        estimated_value = total_investment
                        simple_return = 0

                    results[symbol] = {
                        "status": "success",
                        "data_quality": {
                            "total_records": total_rows,
                            "estimated_months": estimated_months,
                            "price_range": f"₹{min_price:.2f} - ₹{max_price:.2f}"
                        },
                        "price_analysis": {
                            "avg_price": avg_price,
                            "min_price": min_price,
                            "max_price": max_price,
                            "volatility_percent": round((volatility / avg_price) * 100, 2) if avg_price > 0 else 0
                        },
                        "sip_estimation": {
                            "monthly_investment": investment_amount,
                            "total_investment": total_investment,
                            "estimated_final_value": estimated_value,
                            "estimated_return_percent": simple_return,
                            "estimated_units": estimated_units
                        }
                    }
                else:
                    results[symbol] = {
                        "status": "no_data",
                        "message": f"No data found for {symbol} in specified period"
                    }

            except Exception as e:
                logger.error(f"Quick test error for {symbol}: {e}")
                results[symbol] = {
                    "status": "error",
                    "error": str(e)
                }

        return {
            "test_results": results,
            "parameters": {
                "symbols": symbols,
                "period": f"{start_date} to {end_date}",
                "monthly_investment": investment_amount
            },
            "disclaimer": "These are rough estimations. Use full backtest for accurate analysis.",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Quick SIP test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONFIGURATION TEMPLATES
# ============================================================================

@sip_router.get("/config/templates")
async def get_config_templates():
    """Get predefined SIP configuration templates"""
    templates = {
        "conservative": {
            "name": "Conservative SIP",
            "description": "Low risk, steady investment approach",
            "config": {
                "fixed_investment": 5000,
                "drawdown_threshold_1": -15.0,
                "drawdown_threshold_2": -8.0,
                "investment_multiplier_1": 1.5,
                "investment_multiplier_2": 2.0,
                "investment_multiplier_3": 2.5,
                "rolling_window": 200,
                "fallback_day": 15
            },
            "risk_level": "Low",
            "suitable_for": ["First-time investors", "Risk-averse investors", "Long-term wealth preservation"]
        },
        "balanced": {
            "name": "Balanced SIP",
            "description": "Moderate risk with balanced growth potential",
            "config": {
                "fixed_investment": 5000,
                "drawdown_threshold_1": -10.0,
                "drawdown_threshold_2": -5.0,
                "investment_multiplier_1": 2.0,
                "investment_multiplier_2": 3.0,
                "investment_multiplier_3": 4.0,
                "rolling_window": 100,
                "fallback_day": 22
            },
            "risk_level": "Medium",
            "suitable_for": ["Experienced investors", "Balanced portfolio seekers", "Medium-term goals"]
        },
        "aggressive": {
            "name": "Aggressive SIP",
            "description": "Higher risk for potentially higher returns",
            "config": {
                "fixed_investment": 5000,
                "drawdown_threshold_1": -5.0,
                "drawdown_threshold_2": -2.0,
                "investment_multiplier_1": 3.0,
                "investment_multiplier_2": 5.0,
                "investment_multiplier_3": 8.0,
                "rolling_window": 50,
                "fallback_day": 28
            },
            "risk_level": "High",
            "suitable_for": ["Risk-tolerant investors", "Growth-focused portfolios", "Shorter investment horizons"]
        }
    }

    return {
        "templates": templates,
        "default_recommendation": "balanced",
        "customization_note": "All templates can be customized based on individual risk tolerance",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def save_backtest_results(results: Dict, user_id: str, trading_db: AsyncSession):
    """Save backtest results to database (background task)"""
    try:
        # Create table if it doesn't exist
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS sip_backtest_results (
                backtest_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                strategy_name VARCHAR NOT NULL,
                total_investment FLOAT NOT NULL,
                final_portfolio_value FLOAT NOT NULL,
                cagr FLOAT,
                max_drawdown FLOAT,
                sharpe_ratio FLOAT,
                volatility FLOAT,
                num_trades INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await trading_db.execute(create_table_query)

        # Insert results
        for symbol, result in results.items():
            backtest_id = f"bt_{user_id}_{symbol}_{int(datetime.now().timestamp())}"

            insert_query = text("""
                INSERT INTO sip_backtest_results 
                (backtest_id, user_id, symbol, strategy_name, total_investment, 
                 final_portfolio_value, cagr, max_drawdown, sharpe_ratio, volatility, 
                 num_trades, created_at)
                VALUES (:backtest_id, :user_id, :symbol, :strategy_name, :total_investment,
                        :final_portfolio_value, :cagr, :max_drawdown, :sharpe_ratio, 
                        :volatility, :num_trades, :created_at)
            """)

            await trading_db.execute(insert_query, {
                'backtest_id': backtest_id,
                'user_id': user_id,
                'symbol': symbol,
                'strategy_name': result.strategy_name,
                'total_investment': result.total_investment,
                'final_portfolio_value': result.final_portfolio_value,
                'cagr': result.cagr,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'volatility': result.volatility,
                'num_trades': len(result.trades),
                'created_at': datetime.now()
            })

        await trading_db.commit()
        logger.info(f"✅ Background: Saved backtest results for {len(results)} symbols")

    except Exception as e:
        logger.error(f"Background: Error saving backtest results: {e}")
        await trading_db.rollback()


async def save_signal_to_db(portfolio_id: str, symbol: str, signals: Dict, trading_db: AsyncSession):
    """Save investment signal to database"""
    try:
        # Create signals table if it doesn't exist
        create_signals_table = text("""
            CREATE TABLE IF NOT EXISTS sip_signals (
                signal_id VARCHAR PRIMARY KEY,
                portfolio_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                signal_type VARCHAR NOT NULL,
                recommended_amount FLOAT NOT NULL,
                multiplier FLOAT NOT NULL,
                current_price FLOAT NOT NULL,
                drawdown_percent FLOAT,
                signal_strength VARCHAR NOT NULL,
                is_processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW(),
                expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '7 days'
            )
        """)
        await trading_db.execute(create_signals_table)

        # Determine signal strength
        confidence = signals.get('confidence', 0)
        if confidence >= 0.8:
            signal_strength = "high"
        elif confidence >= 0.5:
            signal_strength = "medium"
        else:
            signal_strength = "low"

        # Insert signal
        signal_id = f"sig_{portfolio_id}_{int(datetime.now().timestamp())}"
        insert_signal = text("""
            INSERT INTO sip_signals 
            (signal_id, portfolio_id, symbol, signal_type, recommended_amount, 
             multiplier, current_price, drawdown_percent, signal_strength, 
             created_at, expires_at)
            VALUES (:signal_id, :portfolio_id, :symbol, :signal_type, 
                    :recommended_amount, :multiplier, :current_price, 
                    :drawdown_percent, :signal_strength, :created_at, :expires_at)
        """)

        await trading_db.execute(insert_signal, {
            'signal_id': signal_id,
            'portfolio_id': portfolio_id,
            'symbol': symbol,
            'signal_type': signals.get('signal', 'NORMAL'),
            'recommended_amount': signals.get('recommended_amount', 5000),
            'multiplier': signals.get('confidence', 1.0),
            'current_price': signals.get('current_price', 0),
            'drawdown_percent': signals.get('drawdown_100'),
            'signal_strength': signal_strength,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(days=7)
        })

        await trading_db.commit()
        logger.info(f"✅ Signal saved for portfolio {portfolio_id}")

    except Exception as e:
        logger.error(f"Error saving signal: {e}")
        await trading_db.rollback()


# ============================================================================
# ADDITIONAL ENDPOINTS FOR FRONTEND COMPATIBILITY
# ============================================================================

@sip_router.get("/config/defaults")
async def get_default_config():
    """Get default SIP configuration"""
    config = SIPConfig()
    return {
        "default_config": {
            "fixed_investment": config.fixed_investment,
            "drawdown_threshold_1": config.drawdown_threshold_1,
            "drawdown_threshold_2": config.drawdown_threshold_2,
            "investment_multiplier_1": config.investment_multiplier_1,
            "investment_multiplier_2": config.investment_multiplier_2,
            "investment_multiplier_3": config.investment_multiplier_3,
            "rolling_window": config.rolling_window,
            "fallback_day": config.fallback_day
        },
        "description": "Default SIP strategy configuration with dynamic investment amounts",
        "enhanced_features_available": True
    }


@sip_router.delete("/portfolio/{portfolio_id}")
async def delete_portfolio(
        portfolio_id: str,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Delete a SIP portfolio"""
    try:
        # Verify ownership and update status
        update_query = text("""
            UPDATE sip_portfolios 
            SET status = 'deleted', updated_at = :now
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
        return {"status": "deleted", "portfolio_id": portfolio_id}

    except Exception as e:
        logger.error(f"Error deleting portfolio: {e}")
        await trading_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.put("/portfolio/{portfolio_id}")
async def update_portfolio(
        portfolio_id: str,
        portfolio_name: Optional[str] = None,
        config: Optional[SIPConfigRequest] = None,
        user_id: str = Depends(get_current_user),
        trading_db: AsyncSession = Depends(get_db)
):
    """Update portfolio settings"""
    try:
        updates = []
        params = {'portfolio_id': portfolio_id, 'user_id': user_id, 'now': datetime.now()}

        if portfolio_name is not None:
            updates.append("portfolio_name = :portfolio_name")
            params['portfolio_name'] = portfolio_name

        if config is not None:
            updates.append("config = :config")
            params['config'] = json.dumps(config.dict())

        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        updates.append("updated_at = :now")

        update_query = text(f"""
            UPDATE sip_portfolios 
            SET {', '.join(updates)}
            WHERE portfolio_id = :portfolio_id AND user_id = :user_id
        """)

        result = await trading_db.execute(update_query, params)

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        await trading_db.commit()
        return {"status": "updated", "portfolio_id": portfolio_id}

    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")
        await trading_db.rollback()
        raise HTTPException(status_code=500, detail=str(e))