# backend/app/strategies/enhanced_sip_strategy.py - Fixed Multi-Database Version
"""
Enhanced SIP Strategy with proper multi-database support
Fixes the database connectivity issue by using the correct database for stock data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
import json
from dataclasses import dataclass, asdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import uuid

# Import synchronous utilities as fallback
from common_utils.fetch_db_data import get_table_data, get_engine

logger = logging.getLogger(__name__)


@dataclass
class SIPConfig:
    """Configuration for SIP strategy"""
    fixed_investment: float = 5000
    drawdown_threshold_1: float = -10.0
    drawdown_threshold_2: float = -4.0
    investment_multiplier_1: float = 2.0
    investment_multiplier_2: float = 3.0
    investment_multiplier_3: float = 5.0
    rolling_window: int = 100
    fallback_day: int = 22


@dataclass
class Trade:
    """Represents a single trade in the SIP strategy"""
    timestamp: datetime
    price: float
    units: float
    amount: float
    drawdown: Optional[float]
    portfolio_value: float
    trade_type: str
    total_investment: float


@dataclass
class SIPResults:
    """Results from SIP backtesting"""
    strategy_name: str
    total_investment: float
    final_portfolio_value: float
    total_units: float
    average_buy_price: float
    cagr: float
    trades: List[Trade]
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    volatility: Optional[float] = None


class SIPPortfolioTracker:
    """Enhanced portfolio tracking for SIP strategy"""

    def __init__(self):
        self.total_units = 0.0
        self.total_investment = 0.0
        self.trades: List[Trade] = []
        self.max_portfolio_value = 0.0
        self.max_drawdown = 0.0

    def execute_investment(self, price: float, amount: float, timestamp: datetime,
                           drawdown: Optional[float] = None) -> Trade:
        """Execute an investment and track the trade"""
        units = amount / price
        self.total_units += units
        self.total_investment += amount

        current_portfolio_value = self.total_units * price
        self.max_portfolio_value = max(self.max_portfolio_value, current_portfolio_value)

        # Calculate drawdown from peak
        if self.max_portfolio_value > 0:
            current_drawdown = (current_portfolio_value - self.max_portfolio_value) / self.max_portfolio_value
            self.max_drawdown = min(self.max_drawdown, current_drawdown)

        trade = Trade(
            timestamp=timestamp,
            price=price,
            units=units,
            amount=amount,
            drawdown=drawdown,
            portfolio_value=current_portfolio_value,
            trade_type="BUY",
            total_investment=self.total_investment
        )

        self.trades.append(trade)
        return trade

    def get_current_value(self, current_price: float) -> float:
        """Get current portfolio value"""
        return self.total_units * current_price

    def get_average_buy_price(self) -> float:
        """Calculate average buy price"""
        if self.total_units > 0:
            return self.total_investment / self.total_units
        return 0.0


class EnhancedSIPStrategy:
    """Enhanced SIP Strategy with multi-database support"""

    def __init__(self, nsedata_session: AsyncSession = None, trading_session: AsyncSession = None):
        self.nsedata_session = nsedata_session  # For stock data
        self.trading_session = trading_session  # For saving results

    async def fetch_data_from_db_async(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data using async database utilities - FIXED VERSION"""
        try:
            if not self.nsedata_session:
                logger.error("No nsedata session provided")
                return await self._fetch_data_sync_fallback(symbol, start_date, end_date)

            # Convert string dates to datetime objects for PostgreSQL
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d').date()

            # Use the CORRECT database session for stock data
            query = text(f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM public."{symbol}" 
                WHERE timestamp BETWEEN :start_date AND :end_date 
                ORDER BY timestamp ASC
            """)

            result = await self.nsedata_session.execute(query, {
                'start_date': start_datetime,
                'end_date': end_datetime
            })

            rows = result.fetchall()

            if rows:
                # Convert to DataFrame
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                data = pd.DataFrame(rows, columns=columns)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data[['open', 'high', 'low', 'close', 'volume']] = data[
                    ['open', 'high', 'low', 'close', 'volume']].astype(float)

                logger.info(f"✅ Successfully fetched {len(data)} rows for {symbol} using async nsedata DB")
                return data
            else:
                logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} using async DB: {e}")
            # Fallback to sync method
            return await self._fetch_data_sync_fallback(symbol, start_date, end_date)

    async def _fetch_data_sync_fallback(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fallback to synchronous data fetching - FIXED VERSION"""
        try:
            logger.info(f"Using sync fallback for {symbol}")

            # FIXED: Create a simple sync query without using the complex fetch_db_data
            # This avoids the config.ini dependency issue

            # Try direct database connection using the same credentials as async
            import os
            from sqlalchemy import create_engine

            # Use environment variables or default values
            db_url = os.getenv("NSEDATA_URL", "postgresql://trading_user:password123@localhost:5432/nsedata")
            # Convert asyncpg URL to psycopg2 for sync
            sync_db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

            def fetch_sync():
                try:
                    engine = create_engine(sync_db_url)
                    query = f"""
                        SELECT timestamp, open, high, low, close, volume 
                        FROM public."{symbol}" 
                        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}' 
                        ORDER BY timestamp ASC
                    """

                    import pandas as pd
                    data = pd.read_sql(query, engine)
                    engine.dispose()
                    return data
                except Exception as e:
                    logger.error(f"Sync query failed: {e}")
                    return pd.DataFrame()

            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, fetch_sync)

            if not data.empty:
                logger.info(f"✅ Sync fallback successful for {symbol}: {len(data)} rows")
                return data
            else:
                logger.warning(f"No data found via sync fallback for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Sync fallback also failed for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical indicators for better entry/exit points"""
        if data.empty:
            return data

        data = data.copy()

        try:
            # Enhanced drawdown calculation with multiple timeframes
            data['RecentHigh_50'] = data['close'].rolling(window=50).max()
            data['RecentHigh_100'] = data['close'].rolling(window=100).max()
            data['RecentHigh_200'] = data['close'].rolling(window=200).max()

            data['Drawdown_50'] = (data['close'] - data['RecentHigh_50']) / data['RecentHigh_50'] * 100
            data['Drawdown_100'] = (data['close'] - data['RecentHigh_100']) / data['RecentHigh_100'] * 100
            data['Drawdown_200'] = (data['close'] - data['RecentHigh_200']) / data['RecentHigh_200'] * 100

            # Volatility indicators
            data['Returns'] = data['close'].pct_change()
            data['Volatility_20'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)

            # RSI for momentum
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # Moving averages for trend
            data['SMA_50'] = data['close'].rolling(window=50).mean()
            data['SMA_200'] = data['close'].rolling(window=200).mean()

            return data

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data

    def determine_investment_amount(self, current_price: float, data: pd.DataFrame,
                                    config: SIPConfig, current_row_index: int) -> float:
        """Enhanced investment amount calculation based on market conditions"""
        try:
            if current_row_index < config.rolling_window:
                return config.fixed_investment

            # Get current market conditions
            current_data = data.iloc[current_row_index]
            drawdown_50 = current_data.get('Drawdown_50', 0)
            drawdown_100 = current_data.get('Drawdown_100', 0)
            rsi = current_data.get('RSI', 50)
            volatility = current_data.get('Volatility_20', 0)

            # Base investment
            investment_amount = config.fixed_investment

            # Increase investment during significant drawdowns
            if drawdown_100 <= config.drawdown_threshold_1:  # -10% or worse
                investment_amount *= config.investment_multiplier_3  # 5x
                logger.info(f"Severe drawdown detected ({drawdown_100:.2f}%), increasing to {investment_amount}")
            elif drawdown_100 <= config.drawdown_threshold_2:  # -4% to -10%
                investment_amount *= config.investment_multiplier_2  # 3x
                logger.info(f"Moderate drawdown detected ({drawdown_100:.2f}%), increasing to {investment_amount}")
            elif drawdown_50 <= -2:  # Minor drawdown
                investment_amount *= config.investment_multiplier_1  # 2x

            # Additional adjustments based on RSI (oversold conditions)
            if rsi < 30:  # Oversold
                investment_amount *= 1.2
            elif rsi > 70:  # Overbought - reduce investment
                investment_amount *= 0.8

            # Volatility adjustment - invest more during high volatility (opportunity)
            if volatility > 0.3:  # High volatility
                investment_amount *= 1.1

            return round(investment_amount, 2)

        except Exception as e:
            logger.error(f"Error determining investment amount: {e}")
            return config.fixed_investment

    async def run_backtest(self, symbol: str, start_date: str, end_date: str,
                           config: SIPConfig) -> Optional[SIPResults]:
        """Run enhanced SIP backtest for a single symbol"""
        try:
            logger.info(f"🚀 Starting SIP backtest for {symbol} from {start_date} to {end_date}")

            # Fetch data
            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)

            # Initialize portfolio tracker
            portfolio = SIPPortfolioTracker()

            # Simulate monthly SIP investments
            for i, row in data.iterrows():
                # Invest on fallback day of each month or first available day
                if row['timestamp'].day == config.fallback_day or (
                        i > 0 and row['timestamp'].month != data.iloc[i - 1]['timestamp'].month
                ):
                    current_price = row['close']
                    investment_amount = self.determine_investment_amount(
                        current_price, data, config, i
                    )

                    # Execute investment
                    drawdown = row.get('Drawdown_100', 0)
                    trade = portfolio.execute_investment(
                        price=current_price,
                        amount=investment_amount,
                        timestamp=row['timestamp'],
                        drawdown=drawdown
                    )

                    logger.debug(f"Investment: {trade.amount} at {trade.price} on {trade.timestamp.date()}")

            # Calculate final results
            if not portfolio.trades:
                logger.warning(f"No trades executed for {symbol}")
                return None

            final_price = data.iloc[-1]['close']
            final_portfolio_value = portfolio.get_current_value(final_price)

            # Calculate CAGR
            years = (data.iloc[-1]['timestamp'] - data.iloc[0]['timestamp']).days / 365.25
            cagr = ((final_portfolio_value / portfolio.total_investment) ** (1 / years)) - 1 if years > 0 else 0

            # Calculate Sharpe ratio and volatility
            if len(data) > 1:
                returns = data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = (cagr - 0.05) / volatility if volatility > 0 else 0  # Assuming 5% risk-free rate
            else:
                volatility = None
                sharpe_ratio = None

            results = SIPResults(
                strategy_name="Enhanced SIP Strategy",
                total_investment=portfolio.total_investment,
                final_portfolio_value=final_portfolio_value,
                total_units=portfolio.total_units,
                average_buy_price=portfolio.get_average_buy_price(),
                cagr=cagr,
                trades=portfolio.trades,
                max_drawdown=portfolio.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility
            )

            logger.info(f"✅ Backtest completed for {symbol}: "
                        f"Investment: ₹{portfolio.total_investment:,.2f}, "
                        f"Final Value: ₹{final_portfolio_value:,.2f}, "
                        f"CAGR: {cagr * 100:.2f}%")

            return results

        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")
            return None

    async def run_batch_backtest(self, symbols: List[str], start_date: str,
                                 end_date: str, config: SIPConfig) -> Dict[str, SIPResults]:
        """Run backtest for multiple symbols"""
        results = {}

        for symbol in symbols:
            try:
                result = await self.run_backtest(symbol, start_date, end_date, config)
                if result:
                    results[symbol] = result
                else:
                    logger.warning(f"Skipping {symbol} - no valid backtest result")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        logger.info(f"Batch backtest completed. Processed {len(results)} out of {len(symbols)} symbols")
        return results

    def get_next_investment_signals(self, data: pd.DataFrame, config: SIPConfig) -> Dict[str, Any]:
        """Generate investment signals for the next investment"""
        try:
            if data.empty:
                return {"signal": "NO_DATA", "confidence": 0}

            # Calculate indicators for latest data
            data_with_indicators = self.calculate_technical_indicators(data)
            latest = data_with_indicators.iloc[-1]

            drawdown_100 = latest.get('Drawdown_100', 0)
            rsi = latest.get('RSI', 50)
            volatility = latest.get('Volatility_20', 0)

            # Determine signal strength
            signal_strength = "NORMAL"
            confidence = 0.5
            recommended_multiplier = 1.0

            if drawdown_100 <= config.drawdown_threshold_1:
                signal_strength = "STRONG_BUY"
                confidence = 0.9
                recommended_multiplier = config.investment_multiplier_3
            elif drawdown_100 <= config.drawdown_threshold_2:
                signal_strength = "BUY"
                confidence = 0.7
                recommended_multiplier = config.investment_multiplier_2
            elif rsi < 30:
                signal_strength = "OVERSOLD_BUY"
                confidence = 0.6
                recommended_multiplier = config.investment_multiplier_1

            return {
                "signal": signal_strength,
                "confidence": confidence,
                "recommended_amount": config.fixed_investment * recommended_multiplier,
                "current_price": latest['close'],
                "drawdown_100": drawdown_100,
                "rsi": rsi,
                "volatility": volatility,
                "reasoning": self._generate_signal_reasoning(latest, config)
            }

        except Exception as e:
            logger.error(f"Error generating investment signals: {e}")
            return {"signal": "ERROR", "confidence": 0, "error": str(e)}

    def _generate_signal_reasoning(self, latest_data: pd.Series, config: SIPConfig) -> str:
        """Generate human-readable reasoning for the investment signal"""
        drawdown = latest_data.get('Drawdown_100', 0)
        rsi = latest_data.get('RSI', 50)

        reasons = []

        if drawdown <= config.drawdown_threshold_1:
            reasons.append(f"Severe drawdown of {drawdown:.1f}% presents excellent buying opportunity")
        elif drawdown <= config.drawdown_threshold_2:
            reasons.append(f"Moderate drawdown of {drawdown:.1f}% suggests good entry point")

        if rsi < 30:
            reasons.append(f"RSI of {rsi:.1f} indicates oversold conditions")
        elif rsi > 70:
            reasons.append(f"RSI of {rsi:.1f} suggests overbought market - consider reducing investment")

        if not reasons:
            reasons.append("Normal market conditions - regular SIP investment recommended")

        return "; ".join(reasons)