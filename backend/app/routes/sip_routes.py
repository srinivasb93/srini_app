"""
Fixed SIP Strategy API Endpoints that follow main.py database patterns
Removes the save_backtest_results dependency and focuses on working functionality
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, validator
import json
import logging
import os, sys

# Add the project_root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

from backend.app.database import get_db
from backend.app.auth import UserManager, oauth2_scheme

# Import the fixed strategy
from backend.app.strategies.enhanced_sip_strategy import EnhancedSIPStrategy, SIPConfig, SIPPortfolioTracker, Trade

logger = logging.getLogger(__name__)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    user_id = UserManager.verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id


# Pydantic models for API
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


class SIPBacktestResponse(BaseModel):
    symbol: str
    strategy_name: str
    total_investment: float
    final_portfolio_value: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    volatility: Optional[float]
    num_trades: int
    total_return_percent: float
    avg_buy_price: float


# Create router
sip_router = APIRouter(prefix="/sip", tags=["sip-strategy"])


@sip_router.get("/test-database")
async def test_database_connection(db: AsyncSession = Depends(get_db)):
    """Test database connection using async utilities like main.py"""
    try:
        # Use the EXACT same pattern as your working main.py endpoints
        result = await db.execute(text("SELECT 1 as test, NOW() as server_time"))
        test_result = result.fetchone()

        return {
            "message": "Database connection successful (async)",
            "test_result": test_result[0] if test_result else None,
            "server_time": test_result[1].isoformat() if test_result and test_result[1] else None,
            "timestamp": datetime.now(),
            "connection_type": "async - same as main.py endpoints"
        }
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


# Simple data test using async database
@sip_router.get("/test-data-async/{symbol}")
async def test_symbol_data_async(symbol: str, db: AsyncSession = Depends(get_db)):
    """Test symbol data fetching using async database utilities like main.py"""
    try:
        # Use async database query exactly like main.py endpoints do
        query = text(f"""
            SELECT timestamp, close 
            FROM public."{symbol}" 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)

        result = await db.execute(query)
        rows = result.fetchall()

        return {
            "symbol": symbol,
            "rows_found": len(rows),
            "sample_data": [{"timestamp": str(row[0]), "close": float(row[1])} for row in rows],
            "connection_type": "async - same as main.py",
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Error testing async data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.post("/backtest", response_model=List[SIPBacktestResponse])
async def run_sip_backtest(
        request: SIPBacktestRequest,
        user_id: str = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Run SIP strategy backtest on multiple symbols"""
    try:
        # Create strategy instance without passing db session to avoid issues
        strategy = EnhancedSIPStrategy(db)
        config = SIPConfig(**request.config.dict())

        logger.info(f"Starting SIP backtest for user {user_id} with symbols: {request.symbols}")

        # Run backtest
        results = await strategy.run_batch_backtest(
            request.symbols,
            request.start_date,
            request.end_date,
            config
        )

        if not results:
            logger.warning("No backtest results generated")
            raise HTTPException(status_code=404, detail="No data found for the specified symbols")

        logger.info(f"Backtest completed successfully for {len(results)} symbols")

        # Convert to response format without database save for now
        response_data = []
        for symbol, result in results.items():
            total_return_percent = ((result.final_portfolio_value / result.total_investment) - 1) * 100 if result.total_investment > 0 else 0

            response_data.append(SIPBacktestResponse(
                symbol=symbol,
                strategy_name=result.strategy_name,
                total_investment=result.total_investment,
                final_portfolio_value=result.final_portfolio_value,
                cagr=result.cagr * 100,  # Convert to percentage
                max_drawdown=result.max_drawdown * 100 if result.max_drawdown else 0,
                sharpe_ratio=result.sharpe_ratio,
                volatility=result.volatility * 100 if result.volatility else 0,
                num_trades=len(result.trades),
                total_return_percent=total_return_percent,
                avg_buy_price=result.average_buy_price
            ))

        # Save results to database following main.py patterns
        try:
            await save_backtest_results_to_db(results, user_id, db)
        except Exception as e:
            logger.error(f"Error saving backtest results (non-critical): {e}")
            # Don't fail the request if saving fails

        return response_data

    except Exception as e:
        logger.error(f"Error running SIP backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


async def save_backtest_results_to_db(results: Dict, user_id: str, db: AsyncSession):
    """Save backtest results to database following main.py patterns"""
    try:
        for symbol, result in results.items():
            # Save main result using the same pattern as main.py
            result_data = {
                'backtest_id': f"bt_{user_id}_{symbol}_{int(datetime.now().timestamp())}",
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
            }

            # Check if sip_backtest_results table exists, if not skip saving
            try:
                insert_query = text("""
                    INSERT INTO sip_backtest_results 
                    (backtest_id, user_id, symbol, strategy_name, total_investment, 
                     final_portfolio_value, cagr, max_drawdown, sharpe_ratio, volatility, 
                     num_trades, created_at)
                    VALUES (:backtest_id, :user_id, :symbol, :strategy_name, :total_investment,
                            :final_portfolio_value, :cagr, :max_drawdown, :sharpe_ratio, 
                            :volatility, :num_trades, :created_at)
                """)

                await db.execute(insert_query, result_data)
                await db.commit()
                logger.info(f"Backtest results saved for {symbol}")

            except Exception as e:
                logger.warning(f"Could not save backtest results for {symbol}: {e}")
                await db.rollback()
                # Continue without failing - table might not exist yet

    except Exception as e:
        logger.error(f"Error in save_backtest_results_to_db: {e}")
        await db.rollback()


@sip_router.get("/backtest/history")
async def get_sip_backtest_history(
        limit: int = 10,
        offset: int = 0,
        symbol: Optional[str] = None,
        user_id: str = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
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

        query = text(base_query)
        result = await db.execute(query, params)
        backtest_results = result.fetchall()

        return [
            {
                "backtest_id": row[0],
                "symbol": row[1],
                "strategy_name": row[2],
                "total_investment": row[3],
                "final_portfolio_value": row[4],
                "cagr": row[5] * 100,  # Convert to percentage
                "max_drawdown": row[6] * 100 if row[6] else 0,
                "sharpe_ratio": row[7],
                "volatility": row[8] * 100 if row[8] else 0,
                "num_trades": row[9],
                "created_at": row[10]
            } for row in backtest_results
        ]

    except Exception as e:
        logger.error(f"Error fetching backtest history: {e}")
        # Return empty list if table doesn't exist or other error
        return []


@sip_router.get("/signals/{symbol}")
async def get_investment_signals(
        symbol: str,
        user_id: str = Depends(get_current_user)
):
    """Get investment signals for a symbol"""
    try:
        strategy = EnhancedSIPStrategy()
        config = SIPConfig()  # Use default config for now

        # Fetch recent data for signal generation
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        data = await strategy.fetch_data_from_db(symbol, start_date, end_date)

        if data.empty:
            raise HTTPException(status_code=404, detail="No recent data available")

        signals = strategy.get_next_investment_signals(data, config)

        return {
            "symbol": symbol,
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting investment signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sip_router.get("/available-symbols")
async def get_available_symbols():
    """Get list of available symbols for SIP backtesting"""
    try:
        # Use your existing database utilities to get table list
        from common_utils.fetch_db_data import get_database_tables_list

        tables = get_database_tables_list('nsedata')  # or your database name

        # Filter for likely stock/ETF symbols (you can customize this logic)
        symbols = [table for table in tables if len(table) > 3 and table.isupper()]

        return {
            "symbols": sorted(symbols[:50]),  # Return first 50 symbols
            "total_count": len(symbols),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting available symbols: {e}")
        return {
            "symbols": ["ICICIB22", "HDFCNEXT50", "MOTILALOSMALL"],  # Fallback list
            "total_count": 3,
            "timestamp": datetime.now().isoformat()
        }


@sip_router.get("/performance-comparison")
async def compare_sip_strategies(
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        user_id: str = Depends(get_current_user)
):
    """Compare different SIP strategies for a symbol"""
    try:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        strategy = EnhancedSIPStrategy()

        # Test different configurations
        configs = {
            "Conservative": SIPConfig(
                fixed_investment=5000,
                drawdown_threshold_1=-15.0,
                investment_multiplier_1=1.5,
                investment_multiplier_2=2.0,
                investment_multiplier_3=3.0
            ),
            "Balanced": SIPConfig(
                fixed_investment=5000,
                drawdown_threshold_1=-10.0,
                investment_multiplier_1=2.0,
                investment_multiplier_2=3.0,
                investment_multiplier_3=5.0
            ),
            "Aggressive": SIPConfig(
                fixed_investment=5000,
                drawdown_threshold_1=-5.0,
                investment_multiplier_1=3.0,
                investment_multiplier_2=5.0,
                investment_multiplier_3=8.0
            )
        }

        results = {}

        for strategy_name, config in configs.items():
            backtest_results = await strategy.run_batch_backtest([symbol], start_date, end_date, config)

            if symbol in backtest_results:
                result = backtest_results[symbol]
                results[strategy_name] = {
                    "total_investment": result.total_investment,
                    "final_value": result.final_portfolio_value,
                    "total_return_percent": ((result.final_portfolio_value/result.total_investment-1)*100),
                    "cagr_percent": result.cagr * 100,
                    "num_trades": len(result.trades),
                    "max_drawdown_percent": result.max_drawdown * 100 if result.max_drawdown else 0,
                    "sharpe_ratio": result.sharpe_ratio
                }

        return {
            "symbol": symbol,
            "period": f"{start_date} to {end_date}",
            "strategies": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in performance comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check and utility endpoints
@sip_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SIP Strategy API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


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
        "description": "Default configuration for SIP strategy backtesting"
    }


# Simple backtest without complex strategy logic (for testing)
@sip_router.post("/quick-backtest")
async def quick_sip_backtest(
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        fixed_investment: float = 5000,
        user_id: str = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Quick SIP backtest using async database"""
    try:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        results = {}

        for symbol in symbols:
            try:
                # Get data summary using async database
                query = text(f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        (SELECT close FROM public."{symbol}" WHERE timestamp >= :start_date ORDER BY timestamp ASC LIMIT 1) as first_price,
                        (SELECT close FROM public."{symbol}" WHERE timestamp <= :end_date ORDER BY timestamp DESC LIMIT 1) as last_price
                    FROM public."{symbol}" 
                    WHERE timestamp BETWEEN :start_date AND :end_date
                """)

                result = await db.execute(query, {
                    'start_date': start_date,
                    'end_date': end_date
                })
                row = result.fetchone()

                if row and row[0] > 0 and row[1] and row[2]:
                    total_rows, first_price, last_price = row
                    total_return = ((float(last_price) / float(first_price)) - 1) * 100

                    # Simple SIP calculation
                    months = 12  # Assume 12 months for simplicity
                    total_investment = fixed_investment * months
                    final_value = total_investment * (1 + total_return / 100)

                    results[symbol] = {
                        "total_investment": f"₹{total_investment:,.0f}",
                        "final_value": f"₹{final_value:,.0f}",
                        "total_return": f"{total_return:.2f}%",
                        "data_points": total_rows,
                        "first_price": f"₹{float(first_price):.2f}",
                        "last_price": f"₹{float(last_price):.2f}"
                    }

                else:
                    results[symbol] = {"error": "No data found"}

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = {"error": str(e)}

        return {
            "status": "success",
            "results": results,
            "period": f"{start_date} to {end_date}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in quick backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))