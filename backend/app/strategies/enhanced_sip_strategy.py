"""
Fixed SIP Strategy Module that follows main.py database patterns
Uses existing sync utilities for data fetching and properly handles async operations
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

# Import your existing synchronous database utilities
from common_utils.fetch_db_data import get_table_data

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


class EnhancedSIPStrategy:
    """Enhanced SIP Strategy that follows main.py database patterns"""

    def __init__(self, db_session: AsyncSession = None):
        self.db_session = db_session

    async def fetch_data_from_db_async(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data using async database utilities"""
        try:
            query = text(f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM public."{symbol}" 
                WHERE timestamp BETWEEN :start_date AND :end_date 
                ORDER BY timestamp ASC
            """)

            result = await self.db_session.execute(query, {
                'start_date': start_date,
                'end_date': end_date
            })

            rows = result.fetchall()

            if rows:
                # Convert to DataFrame
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                data = pd.DataFrame(rows, columns=columns)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data[['open', 'high', 'low', 'close', 'volume']] = data[
                    ['open', 'high', 'low', 'close', 'volume']].astype(float)

                logger.info(f"Successfully fetched {len(data)} rows for {symbol} using async DB")
                return data
            else:
                logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} using async DB: {e}")
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

            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            data['RSI'] = 100 - (100 / (1 + rs))

            # Moving averages
            data['SMA_50'] = data['close'].rolling(window=50).mean()
            data['SMA_200'] = data['close'].rolling(window=200).mean()

            # Fill NaN values using forward fill then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            # Return original data if calculation fails
            return data

        return data

    def optimize_entry_conditions(self, row: pd.Series, config: SIPConfig) -> Tuple[bool, float, str]:
        """Enhanced entry condition logic with multiple factors"""

        try:
            # Basic drawdown condition
            drawdown_trigger = row.get('Drawdown_100', 0) < config.drawdown_threshold_1

            # Additional filters for better timing
            volatility_filter = row.get('Volatility_20', 0) > 0.15
            rsi_oversold = row.get('RSI', 50) < 40
            trend_support = row['close'] > row.get('SMA_200', row['close'])

            multiplier = 1.0
            trade_type = "Regular"

            if drawdown_trigger:
                if row.get('Drawdown_100', 0) < config.drawdown_threshold_1 * 1.5:
                    multiplier = config.investment_multiplier_3
                    trade_type = "Aggressive (5x)"
                elif row.get('Drawdown_100', 0) < config.drawdown_threshold_1 * 1.2:
                    multiplier = config.investment_multiplier_2
                    trade_type = "Enhanced (3x)"
                else:
                    multiplier = config.investment_multiplier_1
                    trade_type = "Dynamic (2x)"

                # Enhance multiplier based on additional conditions
                if volatility_filter and rsi_oversold:
                    multiplier *= 1.2
                    trade_type += " + Enhanced"

            return drawdown_trigger, multiplier, trade_type

        except Exception as e:
            logger.error(f"Error in optimize_entry_conditions: {e}")
            return False, 1.0, "Regular"

    def backtest_enhanced_strategy(self, data: pd.DataFrame, config: SIPConfig) -> SIPResults:
        """Enhanced backtesting with improved logic and metrics"""

        if data.empty:
            logger.error("No data provided for backtesting")
            return SIPResults(
                strategy_name="Enhanced Dynamic SIP",
                total_investment=0,
                final_portfolio_value=0,
                total_units=0,
                average_buy_price=0,
                cagr=0,
                trades=[],
                max_drawdown=0
            )

        try:
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)

            portfolio = {
                'units': 0.0,
                'total_investment': 0.0,
                'trades': []
            }

            # Monthly processing
            data['YearMonth'] = data['timestamp'].dt.to_period('M')
            unique_months = data['YearMonth'].unique()

            for month in unique_months:
                month_data = data[data['YearMonth'] == month].copy()
                monthly_investments = 0

                for idx, row in month_data.iterrows():
                    # Check entry conditions
                    should_invest, multiplier, trade_type = self.optimize_entry_conditions(row, config)

                    if should_invest and row['close'] > 0:  # Ensure valid price
                        investment_amount = config.fixed_investment * multiplier
                        units_bought = investment_amount / row['close']

                        portfolio['units'] += units_bought
                        portfolio['total_investment'] += investment_amount
                        monthly_investments += 1

                        trade = Trade(
                            timestamp=row['timestamp'],
                            price=row['close'],
                            units=units_bought,
                            amount=investment_amount,
                            drawdown=row.get('Drawdown_100'),
                            portfolio_value=portfolio['units'] * row['close'],
                            trade_type=trade_type,
                            total_investment=portfolio['total_investment']
                        )
                        portfolio['trades'].append(trade)

                # Fallback investment if no investments made
                if monthly_investments == 0 and len(month_data) > 0:
                    fallback_row = month_data.iloc[-1]  # Last trading day

                    if fallback_row['close'] > 0:  # Ensure valid price
                        units_bought = config.fixed_investment / fallback_row['close']
                        portfolio['units'] += units_bought
                        portfolio['total_investment'] += config.fixed_investment

                        trade = Trade(
                            timestamp=fallback_row['timestamp'],
                            price=fallback_row['close'],
                            units=units_bought,
                            amount=config.fixed_investment,
                            drawdown=fallback_row.get('Drawdown_100'),
                            portfolio_value=portfolio['units'] * fallback_row['close'],
                            trade_type="Fallback",
                            total_investment=portfolio['total_investment']
                        )
                        portfolio['trades'].append(trade)

            # Calculate final metrics
            if portfolio['total_investment'] == 0:
                logger.error("No investments made during backtesting period")
                return SIPResults(
                    strategy_name="Enhanced Dynamic SIP",
                    total_investment=0,
                    final_portfolio_value=0,
                    total_units=0,
                    average_buy_price=0,
                    cagr=0,
                    trades=[],
                    max_drawdown=0
                )

            final_value = portfolio['units'] * data['close'].iloc[-1]
            num_years = max((data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]).days / 365.25, 0.1)

            cagr = (final_value / portfolio['total_investment']) ** (1 / num_years) - 1

            # Calculate additional metrics
            volatility = 0
            max_drawdown = 0
            sharpe_ratio = 0

            if len(portfolio['trades']) > 1:
                try:
                    trade_df = pd.DataFrame([asdict(t) for t in portfolio['trades']])
                    trade_df['portfolio_returns'] = trade_df['portfolio_value'].pct_change()

                    volatility = trade_df['portfolio_returns'].std() * np.sqrt(12) if len(trade_df) > 1 else 0
                    max_drawdown = self._calculate_max_drawdown(trade_df['portfolio_value'])
                    sharpe_ratio = (cagr - 0.05) / volatility if volatility > 0 else 0
                except Exception as e:
                    logger.error(f"Error calculating additional metrics: {e}")

            return SIPResults(
                strategy_name="Enhanced Dynamic SIP",
                total_investment=portfolio['total_investment'],
                final_portfolio_value=final_value,
                total_units=portfolio['units'],
                average_buy_price=portfolio['total_investment'] / portfolio['units'] if portfolio['units'] > 0 else 0,
                cagr=cagr,
                trades=portfolio['trades'],
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility
            )

        except Exception as e:
            logger.error(f"Error in backtest_enhanced_strategy: {e}")
            return SIPResults(
                strategy_name="Enhanced Dynamic SIP",
                total_investment=0,
                final_portfolio_value=0,
                total_units=0,
                average_buy_price=0,
                cagr=0,
                trades=[],
                max_drawdown=0
            )

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown from portfolio values"""
        try:
            peak = portfolio_values.cummax()
            drawdown = (portfolio_values - peak) / peak
            return drawdown.min()
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0

    async def run_batch_backtest(self, symbols: List[str], start_date: str, end_date: str,
                                 config: SIPConfig) -> Dict[str, SIPResults]:
        """Run backtesting on multiple symbols"""
        results = {}

        for symbol in symbols:
            try:
                logger.info(f"Backtesting {symbol}...")
                data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue

                result = self.backtest_enhanced_strategy(data, config)
                results[symbol] = result
                logger.info(f"Backtest completed for {symbol}: CAGR {result.cagr*100:.2f}%")

            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")

        return results

    def get_next_investment_signals(self, data: pd.DataFrame, config: SIPConfig) -> Dict:
        """Get investment signals for current market conditions"""
        if data.empty:
            return {}

        try:
            data = self.calculate_technical_indicators(data)
            latest = data.iloc[-1]

            should_invest, multiplier, trade_type = self.optimize_entry_conditions(latest, config)

            return {
                'should_invest': should_invest,
                'investment_multiplier': multiplier,
                'trade_type': trade_type,
                'current_price': latest['close'],
                'drawdown_100': latest.get('Drawdown_100', 0),
                'rsi': latest.get('RSI', 50),
                'volatility': latest.get('Volatility_20', 0),
                'recommended_amount': config.fixed_investment * multiplier if should_invest else config.fixed_investment,
                'next_fallback_date': self._get_next_fallback_date().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting investment signals: {e}")
            return {}

    def _get_next_fallback_date(self) -> datetime:
        """Calculate next fallback investment date (last working day of month)"""
        today = datetime.now()
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)

        last_day = next_month - timedelta(days=1)

        # Adjust for weekends (simple approximation)
        while last_day.weekday() > 4:  # Saturday=5, Sunday=6
            last_day -= timedelta(days=1)

        return last_day


# Simplified portfolio tracker that follows main.py patterns
class SIPPortfolioTracker:
    """Track real SIP portfolio performance using main.py database patterns"""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def create_sip_portfolio(self, user_id: str, symbol: str, config: SIPConfig) -> str:
        """Create a new SIP portfolio for tracking"""
        portfolio_id = str(uuid.uuid4())

        try:
            # Use the same pattern as main.py for database operations
            query = text("""
                INSERT INTO sip_portfolios 
                (portfolio_id, user_id, symbol, config, created_at, status)
                VALUES (:portfolio_id, :user_id, :symbol, :config, :created_at, :status)
            """)

            await self.db_session.execute(query, {
                'portfolio_id': portfolio_id,
                'user_id': user_id,
                'symbol': symbol,
                'config': json.dumps(asdict(config)),
                'created_at': datetime.now(),
                'status': 'active'
            })

            await self.db_session.commit()
            return portfolio_id

        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"Error creating SIP portfolio: {e}")
            raise

    async def log_sip_investment(self, portfolio_id: str, trade: Trade) -> None:
        """Log actual SIP investment"""
        try:
            query = text("""
                INSERT INTO sip_actual_trades 
                (trade_id, portfolio_id, timestamp, price, units, amount, trade_type)
                VALUES (:trade_id, :portfolio_id, :timestamp, :price, :units, :amount, :trade_type)
            """)

            await self.db_session.execute(query, {
                'trade_id': str(uuid.uuid4()),
                'portfolio_id': portfolio_id,
                'timestamp': trade.timestamp,
                'price': trade.price,
                'units': trade.units,
                'amount': trade.amount,
                'trade_type': trade.trade_type
            })

            await self.db_session.commit()

        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"Error logging SIP investment: {e}")
            raise