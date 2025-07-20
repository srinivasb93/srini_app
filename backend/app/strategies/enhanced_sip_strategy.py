# backend/app/strategies/enhanced_sip_strategy.py - COMPLETE VERSION 3
"""
Enhanced SIP Strategy - COMPLETE IMPLEMENTATION
Fixed data fetching, comprehensive analysis, and full feature set
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
import json
import os
from dataclasses import dataclass, asdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, create_engine
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SIPConfig:
    """Enhanced configuration for SIP strategy"""
    fixed_investment: float = 5000
    major_drawdown_threshold: float = -10.0
    minor_drawdown_threshold: float = -4.0
    extreme_drawdown_threshold: float = -15.0
    minor_drawdown_inv_multiplier: float = 1.75
    major_drawdown_inv_multiplier: float = 3.0
    extreme_drawdown_inv_multiplier: float = 4.0  # For extreme opportunities
    rolling_window: int = 100
    fallback_day: int = 28
    min_investment_gap_days: int = 5
    max_amount_in_a_month: Optional[float] = None
    price_reduction_threshold: float = 4.0
    force_remaining_investment: bool = True  # Force invest remaining amount

@dataclass
class Trade:
    """Enhanced trade representation"""
    timestamp: datetime
    price: float
    units: float
    amount: float
    drawdown: Optional[float]
    portfolio_value: float
    trade_type: str
    total_investment: float
    symbol: Optional[str] = None

@dataclass
class SIPResults:
    """Enhanced results from SIP backtesting"""
    strategy_name: str
    total_investment: float
    final_portfolio_value: float
    total_units: float
    average_buy_price: float
    cagr: float
    trades: List[Trade]
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    volatility: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbol: Optional[str] = None


class SIPPortfolioTracker:
    """Enhanced portfolio tracker with minimum gap enforcement"""

    def __init__(self, min_gap_days: int = 5):
        self.trades: List[Trade] = []
        self.total_investment = 0.0
        self.total_units = 0.0
        self.max_drawdown = 0.0
        self.min_gap_days = min_gap_days
        self.last_investment_date: Optional[datetime] = None

    def can_invest(self, current_date: datetime) -> bool:
        """Check if investment is allowed based on minimum gap"""
        if not self.last_investment_date:
            return True
        days_since_last = (current_date.date() - self.last_investment_date.date()).days
        return days_since_last >= self.min_gap_days

    def execute_investment(self, price: float, amount: float,
                          timestamp: datetime, drawdown: Optional[float] = None,
                          symbol: Optional[str] = None,
                          force: bool = False) -> Optional[Trade]:
        """Execute investment with gap enforcement"""
        # Check minimum gap unless forced
        if not force and not self.can_invest(timestamp):
            logger.debug(f"Investment skipped due to minimum gap requirement")
            return None

        units = amount / price
        self.total_investment += amount
        self.total_units += units

        # Update max drawdown
        if drawdown and drawdown < self.max_drawdown:
            self.max_drawdown = drawdown

        # Determine trade type based on conditions
        trade_type = "Regular SIP"
        if drawdown and drawdown <= -10:
            trade_type = "Drawdown Opportunity"
        elif drawdown and drawdown <= -4:
            trade_type = "Moderate Dip"

        current_value = self.total_units * price

        trade = Trade(
            timestamp=timestamp,
            price=price,
            units=units,
            amount=amount,
            drawdown=drawdown,
            portfolio_value=current_value,
            trade_type=trade_type,
            total_investment=self.total_investment,
            symbol=symbol
        )

        self.trades.append(trade)
        self.last_investment_date = timestamp
        return trade

    def get_current_value(self, current_price: float) -> float:
        """Get current portfolio value"""
        return self.total_units * current_price

    def get_average_buy_price(self) -> float:
        """Calculate average buy price"""
        if self.total_units > 0:
            return self.total_investment / self.total_units
        return 0.0

    def get_days_since_last_investment(self, current_date: datetime) -> int:
        """Get days since last investment"""
        if not self.last_investment_date:
            return float('inf')
        return (current_date.date() - self.last_investment_date.date()).days


class BenchmarkSIPCalculator:
    """
    FIXED: Calculate regular SIP benchmark with consistent monthly investments
    This should generate MORE trades than the original (66+ trades for proper comparison)
    """

    def __init__(self, monthly_amount: float = 5000, investment_day: int = 15):
        self.monthly_amount = monthly_amount
        self.investment_day = investment_day

    async def calculate_benchmark(self, symbol: str, start_date: str, end_date: str,
                                  nsedata_db: AsyncSession) -> Dict:
        """FIXED: Calculate benchmark SIP with consistent monthly investments"""
        try:
            logger.info(
                f"🎯 Calculating FIXED benchmark SIP for {symbol}: ₹{self.monthly_amount} on {self.investment_day}th")

            # Fetch data using same method as enhanced strategy
            strategy = EnhancedSIPStrategy(nsedata_session=nsedata_db)
            data = await strategy.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                logger.warning(f"No data available for benchmark calculation: {symbol}")
                return self._empty_benchmark_result()

            # Initialize tracking
            total_investment = 0.0
            total_units = 0.0
            benchmark_trades = []
            invested_months = set()

            # FIXED: Ensure consistent monthly investments
            # Group data by month to ensure we invest every month
            data['year_month'] = data['timestamp'].dt.to_period('M')

            for month_period in data['year_month'].unique():
                month_data = data[data['year_month'] == month_period].copy()
                month_key = str(month_period)

                if month_key in invested_months:
                    continue

                # Find best investment date for this month
                investment_date = None
                investment_price = None

                # Priority 1: 15th of the month (if available)
                day_15_data = month_data[month_data['timestamp'].dt.day == self.investment_day]
                if not day_15_data.empty:
                    investment_date = day_15_data.iloc[0]['timestamp']
                    investment_price = day_15_data.iloc[0]['close']
                else:
                    # Priority 2: Closest date to 15th
                    month_data['day_diff'] = abs(month_data['timestamp'].dt.day - self.investment_day)
                    closest_idx = month_data['day_diff'].idxmin()
                    investment_date = month_data.loc[closest_idx, 'timestamp']
                    investment_price = month_data.loc[closest_idx, 'close']

                # Execute investment
                if investment_price is not None and investment_price > 0:
                    units_bought = self.monthly_amount / investment_price
                    total_investment += self.monthly_amount
                    total_units += units_bought

                    # Record trade
                    trade = {
                        'date': investment_date.strftime('%Y-%m-%d'),
                        'price': float(investment_price),
                        'units': float(units_bought),
                        'amount': float(self.monthly_amount),
                        'total_investment': float(total_investment),
                        'total_units': float(total_units),
                        'portfolio_value': float(total_units * investment_price),
                        'return_on_investment': ((total_units * investment_price) / total_investment - 1) * 100,
                        'trade_type': 'regular_sip',
                        'month': month_key
                    }
                    benchmark_trades.append(trade)
                    invested_months.add(month_key)

                    logger.debug(f"📅 Benchmark SIP: {investment_date.strftime('%Y-%m-%d')} "
                                 f"₹{self.monthly_amount:,.2f} @ ₹{investment_price:.2f}")

            # Calculate final metrics
            if total_units > 0 and benchmark_trades:
                final_price = data.iloc[-1]['close']
                final_portfolio_value = total_units * final_price
                total_return_percent = ((final_portfolio_value / total_investment) - 1) * 100

                # Calculate CAGR
                start_timestamp = data.iloc[0]['timestamp']
                end_timestamp = data.iloc[-1]['timestamp']
                years = (end_timestamp - start_timestamp).days / 365.25
                cagr_percent = ((final_portfolio_value / total_investment) ** (1 / years) - 1) * 100 if years > 0 else 0

                avg_buy_price = total_investment / total_units

                benchmark_result = {
                    'strategy_name': 'Regular SIP Benchmark',
                    'description': f'₹{self.monthly_amount:,.0f} invested monthly (targeting {self.investment_day}th)',
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
                    'investment_day': self.investment_day,
                    'months_invested': len(invested_months)
                }

                logger.info(f"✅ FIXED Benchmark SIP completed for {symbol}:")
                logger.info(f"   📊 Investment: ₹{total_investment:,.2f}")
                logger.info(f"   💰 Final Value: ₹{final_portfolio_value:,.2f}")
                logger.info(f"   📈 CAGR: {cagr_percent:.2f}%")
                logger.info(f"   🔄 Total Trades: {len(benchmark_trades)}")

                return benchmark_result
            else:
                return self._empty_benchmark_result()

        except Exception as e:
            logger.error(f"Error calculating benchmark for {symbol}: {e}")
            return self._empty_benchmark_result()

    def _empty_benchmark_result(self) -> Dict:
        """Return empty benchmark result for error cases"""
        return {
            'strategy_name': 'Regular SIP Benchmark',
            'description': 'No data available',
            'total_investment': 0.0,
            'final_portfolio_value': 0.0,
            'total_units': 0.0,
            'average_buy_price': 0.0,
            'total_return_percent': 0.0,
            'cagr_percent': 0.0,
            'num_trades': 0,
            'trades': [],
            'error': 'No data available'
        }


class EnhancedSIPStrategy:
    """Base SIP strategy for compatibility"""

    def __init__(self, nsedata_session: AsyncSession = None, trading_session: AsyncSession = None):
        self.nsedata_session = nsedata_session
        self.trading_session = trading_session
        self.monthly_tracker = None

    async def fetch_data_from_db_async(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """FIXED: Fetch data using proper table name handling and error recovery"""
        try:
            if not self.nsedata_session:
                logger.warning("No nsedata session provided, falling back to sync method")

            # Convert string dates to datetime objects for PostgreSQL
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d').date()

            # FIXED: First check if table exists
            table_exists_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = :table_name
                )
            """)

            exists_result = await self.nsedata_session.execute(table_exists_query, {"table_name": symbol})
            table_exists = exists_result.scalar()

            if not table_exists:
                logger.warning(f"Table {symbol} does not exist in database")
                return pd.DataFrame()


            # FIXED: Use proper table name quoting for PostgreSQL
            query = text(f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM "{symbol}" 
                WHERE timestamp BETWEEN :start_date AND :end_date 
                ORDER BY timestamp ASC
            """)

            result = await self.nsedata_session.execute(query, {
                'start_date': start_datetime,
                'end_date': end_datetime
            })

            rows = result.fetchall()

            if not rows:
                logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove any rows with NaN values
            df = df.dropna()

            logger.info(f"✅ Fetched {len(df)} records for {symbol} from async database")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} using async method: {e}")
            return pd.DataFrame()

    async def run_backtest(self, symbol: str, start_date: str, end_date: str,
                           config: SIPConfig) -> Optional[Dict]:
        """Run enhanced SIP backtest with proper error handling"""
        try:
            logger.info(f"🚀 Starting enhanced SIP backtest for {symbol} from {start_date} to {end_date}")

            # Initialize monthly tracker with enhanced config
            self.monthly_tracker = MonthlyInvestmentTracker(
                max_monthly_amount=config.max_amount_in_a_month,
                price_reduction_threshold=config.price_reduction_threshold
            )

            # Fetch data
            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)

            # Initialize portfolio tracker with minimum gap enforcement
            portfolio = SIPPortfolioTracker(min_gap_days=config.min_investment_gap_days)

            # Track investment opportunities and skipped investments
            total_opportunities = 0
            skipped_due_to_gap = 0
            trades = []

            # Simulate SIP investments with enhanced logic
            for i, row in data.iterrows():
                current_date = row['timestamp']

                # Check if it's a regular SIP date (fallback day of month)
                is_sip_date = (
                        current_date.day == config.fallback_day or
                        (i > 0 and current_date.month != data.iloc[i - 1]['timestamp'].month)
                )

                # Check for drawdown opportunities regardless of SIP date
                drawdown_100 = row.get('Drawdown_100', 0)
                is_drawdown_opportunity = (
                        drawdown_100 <= config.minor_drawdown_threshold  # -4% or worse
                )

                should_invest = is_sip_date or is_drawdown_opportunity

                if should_invest:
                    total_opportunities += 1

                    # Check if investment is allowed (minimum gap enforcement)
                    if portfolio.can_invest(current_date):
                        current_price = row['close']
                        investment_amount, is_extreme = self.determine_investment_amount(
                            current_price, data, config, i
                        )
                        logger.info(
                            f"Investment amount {investment_amount} on {current_date.date()} at price {current_price:.2f}")

                        # Execute investment
                        trade = portfolio.execute_investment(
                            price=current_price,
                            amount=investment_amount,
                            timestamp=current_date,
                            drawdown=drawdown_100,
                            symbol=symbol
                        )

                        if trade:
                            logger.debug(
                                f"Investment: ₹{trade.amount:,.2f} at ₹{trade.price:.2f} on {trade.timestamp.date()}")
                            trades.append(asdict(trade))
                            # Record the investment in tracker
                            self.monthly_tracker.record_investment(
                                symbol, current_date, investment_amount, current_price, is_extreme
                            )
                    else:
                        skipped_due_to_gap += 1
                        days_since_last = portfolio.get_days_since_last_investment(current_date)
                        logger.debug(
                            f"Skipped investment on {current_date.date()} - only {days_since_last} days since last investment")

            # Calculate final results
            if not portfolio.trades:
                logger.warning(f"No trades executed for {symbol}")
                return None

            final_price = data.iloc[-1]['close']
            final_portfolio_value = portfolio.get_current_value(final_price)

            # Calculate CAGR
            start_timestamp = data.iloc[0]['timestamp']
            end_timestamp = data.iloc[-1]['timestamp']
            years = (end_timestamp - start_timestamp).days / 365.25
            cagr = ((final_portfolio_value / portfolio.total_investment) ** (1 / years)) - 1 if years > 0 else 0

            # Calculate Sharpe ratio and volatility
            if len(data) > 1:
                returns = data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = (cagr - 0.05) / volatility if volatility > 0 else 0  # Assuming 5% risk-free rate
            else:
                volatility = None
                sharpe_ratio = None

            # results = SIPResults(
            #     strategy_name="Enhanced SIP Strategy",
            #     total_investment=portfolio.total_investment,
            #     final_portfolio_value=final_portfolio_value,
            #     total_units=portfolio.total_units,
            #     average_buy_price=portfolio.get_average_buy_price(),
            #     cagr=cagr,
            #     trades=portfolio.trades,
            #     max_drawdown=portfolio.max_drawdown,
            #     sharpe_ratio=sharpe_ratio,
            #     volatility=volatility,
            #     start_date=start_date,
            #     end_date=end_date,
            #     symbol=symbol
            # )

            logger.info(f"✅ Enhanced backtest completed for {symbol}:")
            logger.info(f"   📊 Investment: ₹{portfolio.total_investment:,.2f}")
            logger.info(f"   💰 Final Value: ₹{final_portfolio_value:,.2f}")
            logger.info(f"   📈 CAGR: {cagr * 100:.2f}%")
            logger.info(f"   🔄 Total Trades: {len(portfolio.trades)}")
            logger.info(f"   ⏭️  Opportunities: {total_opportunities}, Skipped due to gap: {skipped_due_to_gap}")

            return self._calculate_enhanced_results(
                symbol, portfolio.total_investment, final_portfolio_value, portfolio.total_units,
                trades, config, start_date, end_date, data,0, skipped_due_to_gap
            )

        except Exception as e:
            logger.error(f"Error running enhanced backtest for {symbol}: {e}")
            return None

    async def run_batch_backtest(self, symbols: List[str], start_date: str,
                                 end_date: str, config: SIPConfig) -> Dict[str, SIPResults]:
        """Run enhanced backtest for multiple symbols"""
        results = {}

        logger.info(f"🚀 Starting batch backtest for {len(symbols)} symbols")
        logger.info(f"Config Used: {config}")

        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol}...")
                result = await self.run_backtest(symbol, start_date, end_date, config)
                if result:
                    results[symbol] = result
                    logger.info(f"✅ {symbol} completed successfully")
                else:
                    logger.warning(f"❌ {symbol} failed - no valid backtest result")
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue

        logger.info(f"✅ Batch backtest completed. Processed {len(results)} out of {len(symbols)} symbols")
        return results

    async def run_batch_backtest_with_monthly_limit(self, symbols: List[str], start_date: str,
                                 end_date: str, config: SIPConfig) -> Dict[str, SIPResults]:
        """Run enhanced backtest for multiple symbols"""
        results = {}

        logger.info(f"🚀 Starting batch backtest for {len(symbols)} symbols")
        logger.info(f"Config Used: {config}")

        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol}...")

                # Use the enhanced strategy with limits
                enhanced_strategy = EnhancedSIPStrategyWithLimits(
                    nsedata_session=self.nsedata_session,
                    trading_session=self.trading_session
                )

                # Convert SIPConfig to request-like object
                config_request = type('obj', (object,), {
                    'fixed_investment': config.fixed_investment,
                    'max_amount_in_a_month': config.max_amount_in_a_month or (config.fixed_investment * 4),
                    'extreme_drawdown_threshold': config.extreme_drawdown_threshold,
                    'price_reduction_threshold': config.price_reduction_threshold,
                    'major_drawdown_threshold': config.major_drawdown_threshold,
                    'minor_drawdown_threshold': config.minor_drawdown_threshold,
                    'minor_drawdown_inv_multiplier': config.minor_drawdown_inv_multiplier,
                    'major_drawdown_inv_multiplier': config.major_drawdown_inv_multiplier,
                    'extreme_drawdown_inv_multiplier': config.extreme_drawdown_inv_multiplier,
                    'rolling_window': config.rolling_window,
                    'fallback_day': config.fallback_day,
                    'min_investment_gap_days': config.min_investment_gap_days,
                    'force_remaining_investment': config.force_remaining_investment

                })

                result_dict = await enhanced_strategy.run_backtest(
                    symbol, start_date, end_date, config_request
                )

                if result_dict:
                    # Convert dict result to SIPResults object for compatibility
                    trades_list = []
                    for trade_data in result_dict.get('trades', []):
                        trade = Trade(
                            timestamp=datetime.strptime(trade_data['date'], '%Y-%m-%d'),
                            price=trade_data['price'],
                            units=trade_data['units'],
                            amount=trade_data['amount'],
                            drawdown=trade_data.get('drawdown'),
                            portfolio_value=trade_data['portfolio_value'],
                            trade_type=trade_data['trade_type'],
                            total_investment=trade_data['total_investment'],
                            symbol=symbol
                        )
                        trades_list.append(trade)

                    sip_result = SIPResults(
                        strategy_name=result_dict['strategy_name'],
                        total_investment=result_dict['total_investment'],
                        final_portfolio_value=result_dict['final_portfolio_value'],
                        total_units=result_dict['total_units'],
                        average_buy_price=result_dict['average_buy_price'],
                        cagr=result_dict['cagr_percent'] / 100,  # Convert to decimal
                        trades=trades_list,
                        start_date=start_date,
                        end_date=end_date,
                        symbol=symbol
                    )

                    results[symbol] = sip_result
                    logger.info(f"✅ {symbol} completed successfully")
                else:
                    logger.warning(f"❌ {symbol} failed - no valid backtest result")

            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue

        logger.info(f"✅ Batch backtest completed. Results for {len(results)} symbols")
        return results

    def _calculate_next_fallback_date(self, current_date: datetime.date, fallback_day: int) -> datetime.date:
        """Calculate next fallback investment date"""
        try:
            # If current day is before fallback day this month, use this month
            if current_date.day < fallback_day:
                try:
                    return current_date.replace(day=fallback_day)
                except ValueError:
                    # Handle months with fewer days
                    next_next_month = current_date.replace(day=1) + timedelta(days=32)
                    next_next_month = next_next_month.replace(day=1)
                    last_day = (next_next_month - timedelta(days=1)).day
                    return current_date.replace(day=min(fallback_day, last_day))
            else:
                # Use next month
                next_month = current_date.replace(day=1) + timedelta(days=32)
                next_month = next_month.replace(day=1)
                try:
                    return next_month.replace(day=fallback_day)
                except ValueError:
                    # Handle months with fewer days
                    next_next_month = next_month + timedelta(days=32)
                    next_next_month = next_next_month.replace(day=1)
                    last_day = (next_next_month - timedelta(days=1)).day
                    return next_month.replace(day=min(fallback_day, last_day))
        except Exception as e:
            logger.error(f"Error calculating next fallback date: {e}")
            return current_date + timedelta(days=30)

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators with proper error handling and column naming"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame passed to calculate_technical_indicators")
                return df

            # Create a copy to avoid modifying original data
            data = df.copy()

            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return df

            # Sort by timestamp to ensure proper calculation
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp')

            logger.debug("Starting technical indicators calculation...")

            try:
                # Moving Averages with proper error handling
                data['SMA_20'] = data['close'].rolling(window=20, min_periods=1).mean()
                data['SMA_50'] = data['close'].rolling(window=50, min_periods=1).mean()
                data['SMA_200'] = data['close'].rolling(window=200, min_periods=1).mean()

                # Exponential Moving Averages
                data['EMA_12'] = data['close'].ewm(span=12, min_periods=1).mean()
                data['EMA_26'] = data['close'].ewm(span=26, min_periods=1).mean()

                logger.debug("✅ Moving averages calculated")

            except Exception as ma_error:
                logger.error(f"Error calculating moving averages: {ma_error}")

            try:
                # RSI Calculation (Relative Strength Index)
                def calculate_rsi(prices, window=14):
                    if len(prices) < window:
                        return pd.Series([50] * len(prices), index=prices.index)

                    delta = prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

                    # Handle division by zero
                    rs = gain / loss.replace(0, np.nan)
                    rsi = 100 - (100 / (1 + rs))

                    return rsi.fillna(50)  # Fill NaN with neutral RSI value

                data['RSI'] = calculate_rsi(data['close'])
                logger.debug("✅ RSI calculated")

            except Exception as rsi_error:
                logger.error(f"Error calculating RSI: {rsi_error}")
                data['RSI'] = 50  # Default neutral RSI

            try:
                # Williams %R
                def calculate_williams_r(high, low, close, window=14):
                    if len(close) < window:
                        return pd.Series([-50] * len(close), index=close.index)

                    highest_high = high.rolling(window=window, min_periods=1).max()
                    lowest_low = low.rolling(window=window, min_periods=1).min()

                    # Handle division by zero
                    denominator = highest_high - lowest_low
                    williams_r = -100 * (highest_high - close) / denominator.replace(0, np.nan)

                    return williams_r.fillna(-50)

                data['Williams_R'] = calculate_williams_r(data['high'], data['low'], data['close'])
                logger.debug("✅ Williams %R calculated")

            except Exception as wr_error:
                logger.error(f"Error calculating Williams %R: {wr_error}")
                data['Williams_R'] = -50  # Default neutral Williams %R

            try:
                # MACD (Moving Average Convergence Divergence)
                ema_12 = data['EMA_12']
                ema_26 = data['EMA_26']
                data['MACD'] = ema_12 - ema_26
                data['MACD_Signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

                logger.debug("✅ MACD calculated")

            except Exception as macd_error:
                logger.error(f"Error calculating MACD: {macd_error}")
                data['MACD'] = 0
                data['MACD_Signal'] = 0
                data['MACD_Histogram'] = 0

            try:
                # Bollinger Bands
                def calculate_bollinger_bands(prices, window=20, num_std=2):
                    sma = prices.rolling(window=window, min_periods=1).mean()
                    std = prices.rolling(window=window, min_periods=1).std()

                    upper_band = sma + (std * num_std)
                    lower_band = sma - (std * num_std)

                    # Calculate position within bands (0 = lower band, 1 = upper band)
                    bb_position = (prices - lower_band) / (upper_band - lower_band).replace(0, np.nan)
                    bb_position = bb_position.fillna(0.5)  # Default to middle

                    return upper_band, lower_band, bb_position

                data['Bollinger_Upper'], data['Bollinger_Lower'], data['BB_Position'] = calculate_bollinger_bands(
                    data['close'])
                logger.debug("✅ Bollinger Bands calculated")

            except Exception as bb_error:
                logger.error(f"Error calculating Bollinger Bands: {bb_error}")
                data['Bollinger_Upper'] = data['close'] * 1.02
                data['Bollinger_Lower'] = data['close'] * 0.98
                data['BB_Position'] = 0.5

            try:
                # Drawdown calculations
                def calculate_drawdowns(prices, windows=[20, 50, 100]):
                    drawdowns = {}
                    for window in windows:
                        if len(prices) >= window:
                            rolling_max = prices.rolling(window=window, min_periods=1).max()
                            drawdown = (prices - rolling_max) / rolling_max * 100
                            drawdowns[f'Drawdown_{window}'] = drawdown
                        else:
                            # If not enough data, calculate from available data
                            cummax = prices.expanding().max()
                            drawdown = (prices - cummax) / cummax * 100
                            drawdowns[f'Drawdown_{window}'] = drawdown

                    return drawdowns

                drawdowns = calculate_drawdowns(data['close'])
                for key, value in drawdowns.items():
                    data[key] = value

                logger.debug("✅ Drawdowns calculated")

            except Exception as dd_error:
                logger.error(f"Error calculating drawdowns: {dd_error}")
                data['Drawdown_20'] = 0
                data['Drawdown_50'] = 0
                data['Drawdown_100'] = 0

            try:
                # Volatility calculation
                def calculate_volatility(prices, window=20):
                    returns = prices.pct_change()
                    volatility = returns.rolling(window=window, min_periods=1).std()
                    return volatility.fillna(0)

                data['Volatility_20'] = calculate_volatility(data['close'])
                logger.debug("✅ Volatility calculated")

            except Exception as vol_error:
                logger.error(f"Error calculating volatility: {vol_error}")
                data['Volatility_20'] = 0

            try:
                # Commodity Channel Index (CCI)
                def calculate_cci(high, low, close, window=20):
                    if len(close) < window:
                        return pd.Series([0] * len(close), index=close.index)

                    tp = (high + low + close) / 3  # Typical Price
                    sma_tp = tp.rolling(window=window, min_periods=1).mean()

                    # Mean Absolute Deviation
                    mad = tp.rolling(window=window, min_periods=1).apply(
                        lambda x: np.mean(np.abs(x - x.mean())), raw=False
                    )

                    # Handle division by zero
                    cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
                    return cci.fillna(0)

                data['CCI'] = calculate_cci(data['high'], data['low'], data['close'])
                logger.debug("✅ CCI calculated")

            except Exception as cci_error:
                logger.error(f"Error calculating CCI: {cci_error}")
                data['CCI'] = 0

            try:
                # Average True Range (ATR)
                def calculate_atr(high, low, close, window=14):
                    if len(close) < 2:
                        return pd.Series([0] * len(close), index=close.index)

                    prev_close = close.shift(1)
                    true_range = pd.concat([
                        high - low,
                        abs(high - prev_close),
                        abs(low - prev_close)
                    ], axis=1).max(axis=1)

                    atr = true_range.rolling(window=window, min_periods=1).mean()
                    return atr.fillna(0)

                data['ATR'] = calculate_atr(data['high'], data['low'], data['close'])
                logger.debug("✅ ATR calculated")

            except Exception as atr_error:
                logger.error(f"Error calculating ATR: {atr_error}")
                data['ATR'] = 0

            # Validate all indicators were added
            expected_indicators = [
                'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
                'RSI', 'Williams_R', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'Bollinger_Upper', 'Bollinger_Lower', 'BB_Position',
                'Drawdown_20', 'Drawdown_50', 'Drawdown_100',
                'Volatility_20', 'CCI', 'ATR'
            ]

            missing_indicators = [ind for ind in expected_indicators if ind not in data.columns]
            if missing_indicators:
                logger.warning(f"Missing indicators after calculation: {missing_indicators}")
                # Add default values for missing indicators
                for indicator in missing_indicators:
                    data[indicator] = 0

            logger.info(f"✅ Technical indicators calculated successfully. Added {len(expected_indicators)} indicators.")

            # Log a sample of the calculated indicators for debugging
            if not data.empty:
                latest_row = data.iloc[-1]
                logger.debug(f"Sample indicators for latest data point:")
                logger.debug(f"RSI: {latest_row.get('RSI', 'N/A')}")
                logger.debug(f"Drawdown_100: {latest_row.get('Drawdown_100', 'N/A')}")
                logger.debug(f"BB_Position: {latest_row.get('BB_Position', 'N/A')}")

            return data

        except Exception as e:
            logger.error(f"Critical error in calculate_technical_indicators: {e}")
            # Return original DataFrame with default indicator columns to prevent failures
            try:
                default_indicators = {
                    'SMA_20': df['close'] if 'close' in df.columns else 0,
                    'SMA_50': df['close'] if 'close' in df.columns else 0,
                    'SMA_200': df['close'] if 'close' in df.columns else 0,
                    'EMA_12': df['close'] if 'close' in df.columns else 0,
                    'EMA_26': df['close'] if 'close' in df.columns else 0,
                    'RSI': 50,
                    'Williams_R': -50,
                    'MACD': 0,
                    'MACD_Signal': 0,
                    'MACD_Histogram': 0,
                    'Bollinger_Upper': df['close'] * 1.02 if 'close' in df.columns else 1,
                    'Bollinger_Lower': df['close'] * 0.98 if 'close' in df.columns else 1,
                    'BB_Position': 0.5,
                    'Drawdown_20': 0,
                    'Drawdown_50': 0,
                    'Drawdown_100': 0,
                    'Volatility_20': 0,
                    'CCI': 0,
                    'ATR': 0
                }

                for col, default_val in default_indicators.items():
                    df[col] = default_val

                return df
            except Exception as fallback_error:
                logger.error(f"Even fallback failed: {fallback_error}")
                return df

    def determine_investment_amount(self, current_price: float, data: pd.DataFrame,
                                    config: SIPConfig, index: int) -> Tuple[float, bool]:
        """Enhanced investment amount determination with multiple factors"""
        try:
            base_amount = config.fixed_investment

            if index >= len(data):
                return base_amount, False  # No data available for this index

            row = data.iloc[index]

            # Get technical indicators
            drawdown_50 = row.get('Drawdown_50', 0)
            drawdown_100 = row.get('Drawdown_100', 0)
            rsi = row.get('RSI', 50)
            bb_position = row.get('BB_Position', 0.5)
            macd_histogram = row.get('MACD_Histogram', 0)
            williams_r = row.get('Williams_R', -50)
            cci = row.get('CCI', 0)

            investment_amount = base_amount

            # ENHANCEMENT 3: Check for extreme opportunity (>15% drawdown)
            if drawdown_100 <= config.extreme_drawdown_threshold:
                extreme_amount = investment_amount * config.extreme_drawdown_inv_multiplier
                logger.info(f"🔥 EXTREME OPPORTUNITY: {abs(drawdown_100):.1f}% drawdown "
                            f"- investing ₹{extreme_amount:,.2f} (4x)")
                return extreme_amount, True

            # Regular drawdown logic
            if drawdown_100 <= config.major_drawdown_threshold:
                multiplier = config.major_drawdown_inv_multiplier
                investment_amount = config.fixed_investment * multiplier
                logger.info(f"📉 Major drawdown: {abs(drawdown_100):.1f}% "
                            f"- investing ₹{investment_amount:,.2f} ({multiplier}x)")

            elif drawdown_100 <= config.minor_drawdown_threshold:
                multiplier = config.minor_drawdown_inv_multiplier
                investment_amount = config.fixed_investment * multiplier
                logger.info(f"📊 Moderate drawdown: {abs(drawdown_100):.1f}% "
                            f"- investing ₹{investment_amount:,.2f} ({multiplier}x)")


            # RSI-based adjustments (oversold/overbought conditions)
            if 0 < rsi < 25:  # Extremely oversold - major opportunity
                investment_amount *= 1.5
                logger.info(f"Extremely oversold RSI ({rsi:.2f}), increasing investment by 50% to {investment_amount}")
            elif 0 < rsi < 30:  # Oversold
                investment_amount *= 1.2
                logger.info(f"Oversold RSI ({rsi:.2f}), increasing investment by 20% to {investment_amount}")
            elif rsi > 75:  # Extremely overbought - reduce investment
                investment_amount *= 0.6
                logger.info(f"Extremely overbought RSI ({rsi:.2f}), reducing investment by 40% to {investment_amount}")
            elif 75 > rsi > 70:  # Overbought - reduce investment
                investment_amount *= 0.8
                logger.info(f"Overbought RSI ({rsi:.2f}), reducing investment by 20% to {investment_amount}")

            # Williams %R adjustments
            if williams_r < -80 and cci < -100:  # Oversold
                logger.info(f"Investment amount before Williams %R/CCI adjustments: {investment_amount}")
                investment_amount *= 1.01
                logger.info(
                    f"Oversold Williams %R ({williams_r:.2f}), CCI ({cci:.2f}), increasing investment by 10% to {investment_amount}")
            elif williams_r > -20 and cci > 100:  # Overbought
                investment_amount *= 0.95
                logger.info(
                    f"Overbought Williams %R ({williams_r:.2f}), CCI ({cci:.2f}) reducing investment by 10% to {investment_amount}")

            # Bollinger Bands position adjustments
            if bb_position < 0.1:  # Near lower band - opportunity
                logger.info(f"Investment amount before Bollinger Band adjustments: {investment_amount}")
                investment_amount *= 1.02
                logger.info(f"Near Bollinger lower band, increasing investment by 15% to {investment_amount}")
            elif bb_position > 0.9:  # Near upper band - caution
                investment_amount *= .9
                logger.info(f"Near Bollinger upper band, reducing investment by 15% to {investment_amount}")

            # MACD momentum adjustments
            if macd_histogram > 0:  # Positive momentum
                logger.info(f"Investment amount before MACD adjustments: {investment_amount}")
                investment_amount *= 1.01
                logger.info(
                    f"Positive MACD momentum ({macd_histogram:.2f}), increasing investment by 5% to {investment_amount}")
            elif macd_histogram < -0.5:  # Strong negative momentum - opportunity
                investment_amount *= 1.05
                logger.info(
                    f"Strong negative MACD momentum ({macd_histogram:.2f}), increasing investment by 10% to {investment_amount}")


            # Cap the maximum multiplier to prevent excessive investments
            max_allowed = base_amount * 5
            if investment_amount > max_allowed:
                investment_amount = max_allowed
                logger.warning(f"Capping investment at maximum allowed: {max_allowed}")

            # Ensure minimum investment
            min_allowed = base_amount * 0.5  # Minimum 50% of base amount
            if investment_amount < min_allowed:
                investment_amount = min_allowed

            return round(investment_amount, 2), False

        except Exception as e:
            logger.error(f"Error determining investment amount: {e}")
            return config.fixed_investment, False

    def _generate_symbol_recommendation(self, signals: Dict, stats: Dict, data_quality: Dict) -> Dict[str, Any]:
        """Generate recommendation for individual symbol - WITH NULL SAFETY"""
        try:
            # CRITICAL FIX: Handle None parameters
            if signals is None:
                signals = {"signal": "ERROR", "confidence": 0}
            if stats is None:
                stats = {"status": "ERROR", "price_statistics": {}, "technical_indicators": {}, "risk_metrics": {}}
            if data_quality is None:
                data_quality = {"status": "ERROR"}

            recommendation = "HOLD"
            priority = "MEDIUM"
            confidence_score = signals.get('confidence', 0)
            signal_type = signals.get('signal', 'NORMAL')
            considerations = []

            # Base recommendation on signal
            if signal_type == 'STRONG_BUY':
                recommendation = "STRONG BUY"
                priority = "HIGH"
                considerations.append("Strong buy signal detected")
            elif signal_type == 'BUY':
                recommendation = "BUY"
                priority = "HIGH"
                considerations.append("Buy signal detected")
            elif signal_type == 'WEAK_BUY':
                recommendation = "CONSIDER BUY"
                priority = "MEDIUM"
                considerations.append("Weak buy signal")
            elif signal_type == 'AVOID':
                recommendation = "AVOID"
                priority = "LOW"
                considerations.append("Avoid signal detected")
            elif signal_type == 'ERROR' or signal_type == 'NO_DATA':
                recommendation = "NO RECOMMENDATION"
                priority = "LOW"
                confidence_score = 0
                considerations.append("Insufficient data or analysis error")

            # Adjust based on data quality
            quality_status = data_quality.get('status', 'UNKNOWN')
            if quality_status == 'POOR' or quality_status == 'ERROR':
                recommendation = "NO RECOMMENDATION"
                priority = "LOW"
                confidence_score = 0
                considerations.append("Poor data quality - recommendation unreliable")
            elif quality_status == 'FAIR':
                confidence_score *= 0.8
                considerations.append("Fair data quality - recommendation with caution")

            # Adjust based on statistics
            if stats.get('status') == 'SUCCESS':
                price_stats = stats.get('price_statistics', {})
                tech_indicators = stats.get('technical_indicators', {})
                risk_metrics = stats.get('risk_metrics', {})

                # Volatility consideration
                volatility = price_stats.get('volatility_annualized', 0)
                if volatility and volatility > 40:
                    considerations.append("High volatility - suitable for risk-tolerant investors")
                elif volatility and volatility < 20:
                    considerations.append("Low volatility - suitable for conservative investors")

                # RSI consideration
                rsi = tech_indicators.get('rsi')
                if rsi and rsi < 30:
                    considerations.append("Oversold conditions detected. Consider for buying")
                elif rsi and rsi > 70:
                    considerations.append("Overbought conditions detected. Consider for selling or avoiding")

                # Trend consideration
                sma_20 = tech_indicators.get('sma_20')
                sma_50 = tech_indicators.get('sma_50')
                current_price = price_stats.get('current_price', 0)

                if sma_20 and sma_50 and current_price:
                    if current_price > sma_20 > sma_50:
                        considerations.append("Strong upward trend. Current Price > SMA(20) > SMA(50)")
                    elif current_price < sma_20 < sma_50:
                        considerations.append("Strong downward trend. Current Price < SMA(20) < SMA(50)")

                # Risk consideration
                max_drawdown = risk_metrics.get('max_drawdown')
                if max_drawdown and abs(max_drawdown) > 30:
                    considerations.append("High historical drawdown risk. Consider for high-risk portfolios")

            return {
                "recommendation": recommendation,
                "priority": priority,
                "confidence_score": round(confidence_score, 3),
                "signal_type": signal_type,
                "considerations": considerations,
                "investment_horizon": "Long-term" if signal_type in ['STRONG_BUY', 'BUY'] else "Short-term",
                "risk_tolerance": "High" if signal_type == 'STRONG_BUY' else "Medium" if signal_type == 'BUY' else "Low"
            }

        except Exception as e:
            logger.error(f"Error generating symbol recommendation: {e}")
            # Always return structured response
            return {
                "recommendation": "ERROR - Unable to generate recommendation",
                "priority": "LOW",
                "confidence_score": 0,
                "signal_type": "ERROR",
                "considerations": [f"Analysis error: {str(e)}"],
                "investment_horizon": "N/A",
                "risk_tolerance": "N/A",
                "error": str(e)
            }

    async def validate_symbol_data_quality(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Validate data quality for a symbol"""
        try:
            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                return {
                    "symbol": symbol,
                    "status": "NO_DATA",
                    "message": "No data available for the specified date range",
                    "data_points": 0,
                    "coverage_percent": 0
                }

            # Calculate expected trading days (approximate)
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            total_days = (end_dt - start_dt).days
            expected_trading_days = total_days * 5 / 7  # Rough estimate

            actual_data_points = len(data)
            coverage_percent = (actual_data_points / expected_trading_days) * 100 if expected_trading_days > 0 else 0

            # Check for data gaps
            data_gaps = []
            if len(data) > 1:
                data['date_diff'] = data['timestamp'].diff().dt.days
                large_gaps = data[data['date_diff'] > 7]  # Gaps larger than a week
                if not large_gaps.empty:
                    data_gaps = [
                        {
                            "start_date": row['timestamp'].strftime('%Y-%m-%d'),
                            "gap_days": int(row['date_diff'])
                        }
                        for _, row in large_gaps.iterrows()
                    ]

            # Data quality assessment
            if coverage_percent >= 90:
                quality_status = "EXCELLENT"
            elif coverage_percent >= 75:
                quality_status = "GOOD"
            elif coverage_percent >= 50:
                quality_status = "FAIR"
            else:
                quality_status = "POOR"

            return {
                "symbol": symbol,
                "status": quality_status,
                "data_points": actual_data_points,
                "expected_data_points": int(expected_trading_days),
                "coverage_percent": round(coverage_percent, 2),
                "date_range": {
                    "requested_start": start_date,
                    "requested_end": end_date,
                    "actual_start": data.iloc[0]['timestamp'].strftime('%Y-%m-%d'),
                    "actual_end": data.iloc[-1]['timestamp'].strftime('%Y-%m-%d')
                },
                "data_gaps": data_gaps,
                "price_range": {
                    "min": float(data['close'].min()),
                    "max": float(data['close'].max()),
                    "avg": float(data['close'].mean())
                },
                "volume_stats": {
                    "avg_volume": int(data['volume'].mean()),
                    "max_volume": int(data['volume'].max()),
                    "min_volume": int(data['volume'].min())
                }
            }

        except Exception as e:
            logger.error(f"Error validating data quality for {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "ERROR",
                "message": f"Error validating data: {str(e)}",
                "data_points": 0,
                "coverage_percent": 0
            }

    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics"""
        try:
            if data.empty:
                return {"status": "NO_DATA"}

            returns = data['close'].pct_change().dropna()

            if len(returns) < 2:
                return {"status": "INSUFFICIENT_DATA"}

            # Max drawdown
            max_drawdown = self._calculate_max_drawdown(data['close'])

            # Sharpe ratio (assuming 5% risk-free rate)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)

            # Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(returns)

            # Value at Risk (VaR) at 5% confidence level
            value_at_risk_5 = returns.quantile(0.05) if len(returns) > 0 else 0

            # Expected Shortfall (CVaR)
            expected_shortfall = returns[returns <= value_at_risk_5].mean() if len(
                returns[returns <= value_at_risk_5]) > 0 else 0

            return {
                "status": "SUCCESS",
                "max_drawdown": float(max_drawdown),
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio),
                "value_at_risk_5": float(value_at_risk_5 * 100),
                "expected_shortfall": float(expected_shortfall * 100)
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                "status": "ERROR",
                "message": str(e)
            }

    def _calculate_price_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price-related statistics"""
        try:
            if data.empty:
                return {"status": "NO_DATA"}

            current_price = data.iloc[-1]['close']

            # Price changes
            price_change_1d = current_price - data.iloc[-2]['close'] if len(data) > 1 else 0
            price_change_percent_1d = (price_change_1d / data.iloc[-2]['close'] * 100) if len(data) > 1 and \
                                                                                          data.iloc[-2][
                                                                                              'close'] > 0 else 0

            # 52-week high/low (approx 252 trading days)
            year_data = data.tail(252)
            high_52w = year_data['high'].max()
            low_52w = year_data['low'].min()

            # Annualized volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0

            # Average volume
            average_volume = data['volume'].mean() if 'volume' in data.columns else 0

            return {
                "status": "SUCCESS",
                "current_price": float(current_price),
                "price_change_1d": float(price_change_1d),
                "price_change_percent_1d": float(price_change_percent_1d),
                "high_52w": float(high_52w),
                "low_52w": float(low_52w),
                "volatility_annualized": float(volatility),
                "average_volume": int(average_volume)
            }

        except Exception as e:
            logger.error(f"Error calculating price statistics: {e}")
            return {
                "status": "ERROR",
                "message": str(e)
            }

    async def get_symbol_statistics(self, symbol: str, days: int = 365) -> Dict[str, Any]:
        """Get comprehensive statistics for a symbol - FIXED VERSION"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            logger.debug(f"Getting statistics for {symbol} from {start_date} to {end_date}")

            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                return {
                    "symbol": symbol,
                    "status": "NO_DATA",
                    "message": "No data available",
                    "price_statistics": {},
                    "technical_indicators": {},
                    "risk_metrics": {}
                }

            try:
                # CRITICAL FIX: Calculate technical indicators FIRST
                logger.debug(f"Calculating technical indicators for {symbol}")
                data_with_indicators = self.calculate_technical_indicators(data)

                if data_with_indicators.empty:
                    raise ValueError("Technical indicators calculation returned empty data")

                # Get the latest data point after indicators are calculated
                latest_idx = -1
                latest_row = data_with_indicators.iloc[latest_idx]

                # Price statistics
                price_stats = {
                    "current_price": float(data.iloc[-1]['close']),
                    "price_change_1d": float(data.iloc[-1]['close'] - data.iloc[-2]['close']) if len(data) > 1 else 0,
                    "price_change_percent_1d": float(
                        ((data.iloc[-1]['close'] / data.iloc[-2]['close']) - 1) * 100) if len(data) > 1 else 0,
                    "high_52w": float(data['high'].max()),
                    "low_52w": float(data['low'].min()),
                    "volatility_annualized": float(data['close'].pct_change().std() * np.sqrt(252) * 100),
                    "average_volume": int(data['volume'].mean())
                }

                # FIXED: Technical indicators extraction with proper error handling
                tech_indicators = {}

                try:
                    # Extract all technical indicators with safe defaults
                    indicator_mapping = {
                        'rsi': 'RSI',
                        'williams_r': 'Williams_R',
                        'cci': 'CCI',
                        'sma_20': 'SMA_20',
                        'sma_50': 'SMA_50',
                        'sma_200': 'SMA_200',
                        'ema_12': 'EMA_12',
                        'ema_26': 'EMA_26',
                        'macd': 'MACD',
                        'macd_signal': 'MACD_Signal',
                        'macd_histogram': 'MACD_Histogram',
                        'bollinger_upper': 'Bollinger_Upper',
                        'bollinger_lower': 'Bollinger_Lower',
                        'bb_position': 'BB_Position',
                        'atr': 'ATR',
                        'drawdown_20': 'Drawdown_20',
                        'drawdown_50': 'Drawdown_50',
                        'drawdown_100': 'Drawdown_100',
                        'volatility_20': 'Volatility_20'
                    }

                    for key, col_name in indicator_mapping.items():
                        if col_name in data_with_indicators.columns:
                            value = latest_row[col_name]
                            if pd.notna(value):
                                tech_indicators[key] = float(value)
                            else:
                                tech_indicators[key] = None
                                logger.debug(f"NaN value for {key} in {symbol}")
                        else:
                            tech_indicators[key] = None
                            logger.warning(f"Missing column {col_name} for {key} in {symbol}")

                    logger.debug(f"✅ Extracted {len(tech_indicators)} technical indicators for {symbol}")

                    # Log some key indicators for debugging
                    logger.debug(
                        f"{symbol} - RSI: {tech_indicators.get('rsi')}, Drawdown_100: {tech_indicators.get('drawdown_100')}")

                except Exception as tech_error:
                    logger.error(f"Error extracting technical indicators for {symbol}: {tech_error}")
                    # Provide default technical indicators to prevent downstream errors
                    tech_indicators = {
                        'rsi': 50.0,
                        'williams_r': -50.0,
                        'cci': 0.0,
                        'sma_20': float(price_stats['current_price']),
                        'sma_50': float(price_stats['current_price']),
                        'sma_200': float(price_stats['current_price']),
                        'ema_12': float(price_stats['current_price']),
                        'ema_26': float(price_stats['current_price']),
                        'macd': 0.0,
                        'macd_signal': 0.0,
                        'macd_histogram': 0.0,
                        'bollinger_upper': float(price_stats['current_price']) * 1.02,
                        'bollinger_lower': float(price_stats['current_price']) * 0.98,
                        'bb_position': 0.5,
                        'atr': 0.0,
                        'drawdown_20': 0.0,
                        'drawdown_50': 0.0,
                        'drawdown_100': 0.0,
                        'volatility_20': 0.0
                    }

                # Risk metrics with safe calculations
                risk_metrics = {}
                try:
                    returns = data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        risk_metrics = {
                            "max_drawdown": float(self._calculate_max_drawdown(data['close'])),
                            "sharpe_ratio": float(self._calculate_sharpe_ratio(returns)) if len(returns) > 30 else None,
                            "sortino_ratio": float(self._calculate_sortino_ratio(returns)) if len(
                                returns) > 30 else None,
                            "value_at_risk_5": float(returns.quantile(0.05) * 100),
                            "expected_shortfall": float(returns[returns <= returns.quantile(0.05)].mean() * 100) if len(
                                returns[returns <= returns.quantile(0.05)]) > 0 else None
                        }
                    else:
                        risk_metrics = {
                            "max_drawdown": 0.0,
                            "sharpe_ratio": None,
                            "sortino_ratio": None,
                            "value_at_risk_5": 0.0,
                            "expected_shortfall": None
                        }
                except Exception as risk_error:
                    logger.error(f"Error calculating risk metrics for {symbol}: {risk_error}")
                    risk_metrics = {
                        "max_drawdown": 0.0,
                        "sharpe_ratio": None,
                        "sortino_ratio": None,
                        "value_at_risk_5": 0.0,
                        "expected_shortfall": None
                    }

                logger.info(f"✅ Successfully calculated statistics for {symbol}")

                return {
                    "symbol": symbol,
                    "status": "SUCCESS",
                    "data_points": len(data),
                    "analysis_period": f"{start_date} to {end_date}",
                    "price_statistics": price_stats,
                    "technical_indicators": tech_indicators,  # This was empty before - now properly populated
                    "risk_metrics": risk_metrics,
                    "last_updated": datetime.now().isoformat()
                }

            except Exception as calc_error:
                logger.error(f"Error calculating statistics for {symbol}: {calc_error}")
                # Return structured error response instead of None
                return {
                    "symbol": symbol,
                    "status": "CALCULATION_ERROR",
                    "message": f"Error calculating statistics: {str(calc_error)}",
                    "price_statistics": {
                        "current_price": float(data.iloc[-1]['close']) if not data.empty else 0,
                        "price_change_1d": 0,
                        "price_change_percent_1d": 0,
                        "high_52w": float(data['high'].max()) if not data.empty else 0,
                        "low_52w": float(data['low'].min()) if not data.empty else 0,
                        "volatility_annualized": 0,
                        "average_volume": int(data['volume'].mean()) if not data.empty else 0
                    },
                    "technical_indicators": {
                        'rsi': 50.0,
                        'williams_r': -50.0,
                        'cci': 0.0,
                        'sma_20': None,
                        'sma_50': None,
                        'sma_200': None,
                        'ema_12': None,
                        'ema_26': None,
                        'macd': 0.0,
                        'macd_signal': 0.0,
                        'macd_histogram': 0.0,
                        'bollinger_upper': None,
                        'bollinger_lower': None,
                        'bb_position': 0.5,
                        'atr': 0.0,
                        'drawdown_20': 0.0,
                        'drawdown_50': 0.0,
                        'drawdown_100': 0.0,
                        'volatility_20': 0.0
                    },
                    "risk_metrics": {
                        "max_drawdown": 0.0,
                        "sharpe_ratio": None,
                        "sortino_ratio": None,
                        "value_at_risk_5": 0.0,
                        "expected_shortfall": None
                    }
                }

        except Exception as e:
            logger.error(f"Critical error getting statistics for {symbol}: {e}")
            # CRITICAL: Always return a structured dictionary, never None
            return {
                "symbol": symbol,
                "status": "ERROR",
                "message": f"Failed to get statistics: {str(e)}",
                "price_statistics": {},
                "technical_indicators": {},
                "risk_metrics": {}
            }

    async def get_investment_signals(self, symbol: str, config: SIPConfig) -> Dict[str, Any]:
        """Get investment signals for a symbol"""
        try:
            logger.info(f"🎯 Getting enhanced investment signals for {symbol}")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                return {
                    "symbol": symbol,
                    "status": "NO_DATA",
                    "signals": {}
                }

            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)

            latest = data.iloc[-1]

            signals = {
                "symbol": symbol,
                "status": "SUCCESS",
                "signals": {
                    "drawdown_signal": {
                        "current_drawdown": float(latest.get('Drawdown_100', 0)),
                        "signal_strength": "STRONG" if latest.get('Drawdown_100',
                                                                  0) <= config.major_drawdown_threshold else "WEAK",
                        "recommended_action": "BUY" if latest.get('Drawdown_100',
                                                                  0) <= config.minor_drawdown_threshold else "HOLD"
                    },
                    "price_position": {
                        "current_price": float(latest['close']),
                        "vs_20_sma": float(((latest['close'] / latest.get('SMA_20', latest['close'])) - 1) * 100),
                        "vs_50_sma": float(((latest['close'] / latest.get('SMA_50', latest['close'])) - 1) * 100)
                    }
                },
                "overall_signal": "BUY" if latest.get('Drawdown_100', 0) <= config.minor_drawdown_threshold else "HOLD"
            }

            return signals

        except Exception as e:
            logger.error(f"Error getting investment signals for {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "ERROR",
                "message": f"Error calculating signals: {str(e)}"
            }

    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        try:
            if data.empty or len(data) < 20:
                return {"status": "INSUFFICIENT_DATA"}

            # Get recent data for calculation
            recent_data = data.tail(50)  # Last 50 days
            current_price = recent_data.iloc[-1]['close']

            # Calculate pivot points
            high = recent_data['high'].max()
            low = recent_data['low'].min()
            close = recent_data['close'].iloc[-1]

            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)

            # Dynamic support and resistance based on local extremes
            highs = recent_data['high'].rolling(window=5).max()
            lows = recent_data['low'].rolling(window=5).min()

            # Find significant levels
            resistance_levels = []
            support_levels = []

            for i in range(5, len(recent_data) - 5):
                # Check for resistance (local high)
                if (recent_data.iloc[i]['high'] == highs.iloc[i] and
                        recent_data.iloc[i]['high'] > recent_data.iloc[i - 1]['high'] and
                        recent_data.iloc[i]['high'] > recent_data.iloc[i + 1]['high']):
                    resistance_levels.append(recent_data.iloc[i]['high'])

                # Check for support (local low)
                if (recent_data.iloc[i]['low'] == lows.iloc[i] and
                        recent_data.iloc[i]['low'] < recent_data.iloc[i - 1]['low'] and
                        recent_data.iloc[i]['low'] < recent_data.iloc[i + 1]['low']):
                    support_levels.append(recent_data.iloc[i]['low'])

            # Get closest levels to current price
            resistance_levels = sorted(set(resistance_levels), reverse=True)
            support_levels = sorted(set(support_levels), reverse=True)

            nearest_resistance = None
            nearest_support = None

            for level in resistance_levels:
                if level > current_price:
                    nearest_resistance = level
                    break

            for level in support_levels:
                if level < current_price:
                    nearest_support = level
                    break

            return {
                "status": "SUCCESS",
                "pivot_points": {
                    "pivot": float(pivot),
                    "resistance_1": float(r1),
                    "resistance_2": float(r2),
                    "support_1": float(s1),
                    "support_2": float(s2)
                },
                "dynamic_levels": {
                    "nearest_resistance": float(nearest_resistance) if nearest_resistance else None,
                    "nearest_support": float(nearest_support) if nearest_support else None,
                    "resistance_distance": float(
                        (nearest_resistance / current_price - 1) * 100) if nearest_resistance else None,
                    "support_distance": float((current_price / nearest_support - 1) * 100) if nearest_support else None
                },
                "current_price": float(current_price)
            }

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {
                "status": "ERROR",
                "message": f"Error calculating support/resistance: {str(e)}"
            }

    def _generate_portfolio_recommendation(self, overall_metrics: Dict, symbol_reports: Dict) -> Dict[str, Any]:
        """Generate overall portfolio recommendation"""
        try:
            total_symbols = overall_metrics["total_symbols"]
            analyzed_symbols = overall_metrics["analyzed_symbols"]
            strong_buy_signals = overall_metrics["strong_buy_signals"]
            buy_signals = overall_metrics["buy_signals"]
            avoid_signals = overall_metrics["avoid_signals"]
            avg_confidence = overall_metrics["avg_confidence"]

            # Determine overall portfolio action
            if analyzed_symbols == 0:
                portfolio_action = "UNABLE_TO_ANALYZE"
                portfolio_confidence = 0
            elif strong_buy_signals / analyzed_symbols > 0.5:
                portfolio_action = "AGGRESSIVE_INVESTMENT"
                portfolio_confidence = min(avg_confidence + 0.2, 1.0)
            elif (strong_buy_signals + buy_signals) / analyzed_symbols > 0.4:
                portfolio_action = "MODERATE_INVESTMENT"
                portfolio_confidence = avg_confidence
            elif avoid_signals / analyzed_symbols > 0.5:
                portfolio_action = "DEFENSIVE_STANCE"
                portfolio_confidence = max(avg_confidence - 0.2, 0.1)
            else:
                portfolio_action = "BALANCED_APPROACH"
                portfolio_confidence = avg_confidence

            # Generate specific recommendations
            recommendations = []

            if portfolio_action == "AGGRESSIVE_INVESTMENT":
                recommendations.extend([
                    "Consider increasing SIP amounts for symbols with strong buy signals",
                    "Take advantage of current market opportunities",
                    "Monitor positions closely due to higher risk",
                    "Consider staggered investments to manage timing risk"
                ])
            elif portfolio_action == "MODERATE_INVESTMENT":
                recommendations.extend([
                    "Continue regular SIP investments",
                    "Consider slight increases in high-confidence symbols",
                    "Maintain diversified approach",
                    "Review portfolio monthly for rebalancing opportunities"
                ])
            elif portfolio_action == "DEFENSIVE_STANCE":
                recommendations.extend([
                    "Consider reducing investment amounts",
                    "Focus on high-quality, low-risk symbols",
                    "Wait for better market conditions",
                    "Maintain cash reserves for future opportunities"
                ])
            else:
                recommendations.extend([
                    "Maintain current investment strategy",
                    "Regular portfolio review recommended",
                    "Stay disciplined with SIP approach",
                    "Consider gradual position adjustments based on market signals"
                ])

            # Risk assessment
            risk_level = "UNKNOWN"
            if avg_confidence > 0.7:
                risk_level = "LOW_TO_MODERATE"
            elif avg_confidence > 0.5:
                risk_level = "MODERATE"
            elif avg_confidence > 0.3:
                risk_level = "MODERATE_TO_HIGH"
            else:
                risk_level = "HIGH"

            # Calculate suggested portfolio allocation
            total_allocation = 100.0
            allocation_suggestions = {}

            for symbol, report in symbol_reports.items():
                if report.get('status') == 'SUCCESS':
                    recommendation = report.get('recommendation', {})
                    suggested_allocation = recommendation.get('suggested_allocation', 0)
                    allocation_suggestions[symbol] = suggested_allocation

            # Normalize allocations to 100%
            total_suggested = sum(allocation_suggestions.values())
            if total_suggested > 0:
                allocation_suggestions = {
                    symbol: (allocation / total_suggested) * total_allocation
                    for symbol, allocation in allocation_suggestions.items()
                }

            return {
                "portfolio_action": portfolio_action,
                "portfolio_confidence": round(portfolio_confidence, 3),
                "risk_level": risk_level,
                "recommendations": recommendations,
                "allocation_suggestions": allocation_suggestions,
                "next_review_date": (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                "key_metrics": {
                    "analysis_coverage": f"{analyzed_symbols}/{total_symbols}",
                    "positive_signals": strong_buy_signals + buy_signals,
                    "negative_signals": avoid_signals,
                    "avg_confidence": round(avg_confidence, 3)
                }
            }

        except Exception as e:
            logger.error(f"Error generating portfolio recommendation: {e}")
            return {
                "portfolio_action": "ERROR",
                "portfolio_confidence": 0,
                "risk_level": "UNKNOWN",
                "recommendations": ["Unable to generate recommendations due to analysis error"],
                "error": str(e)
            }

    def _generate_risk_assessment(self, symbol_reports: Dict, overall_metrics: Dict) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        try:
            risk_summary = {
                "overall_risk_level": "UNKNOWN",
                "risk_factors": [],
                "mitigation_strategies": [],
                "portfolio_diversification": {
                    "score": 0,
                    "assessment": "UNKNOWN"
                }
            }

            # Analyze individual symbol risks
            high_risk_count = 0
            medium_risk_count = 0
            low_risk_count = 0

            volatility_scores = []
            correlation_risk = False

            for symbol, report in symbol_reports.items():
                if report.get('status') == 'SUCCESS':
                    recommendation = report.get('recommendation', {})
                    risk_tolerance = recommendation.get('risk_tolerance', 'UNKNOWN')

                    if risk_tolerance == 'LOW':
                        high_risk_count += 1
                    elif risk_tolerance == 'MEDIUM':
                        medium_risk_count += 1
                    elif risk_tolerance == 'HIGH':
                        low_risk_count += 1

                    # Collect volatility data
                    stats = report.get('statistics', {})
                    if stats.get('status') == 'SUCCESS':
                        price_stats = stats.get('price_statistics', {})
                        volatility = price_stats.get('volatility_annualized', 0)
                        volatility_scores.append(volatility)

            total_analyzed = high_risk_count + medium_risk_count + low_risk_count

            # Determine overall risk level
            if total_analyzed > 0:
                high_risk_ratio = high_risk_count / total_analyzed
                medium_risk_ratio = medium_risk_count / total_analyzed

                if high_risk_ratio > 0.5:
                    risk_summary["overall_risk_level"] = "HIGH"
                elif high_risk_ratio > 0.3 or medium_risk_ratio > 0.6:
                    risk_summary["overall_risk_level"] = "MEDIUM_TO_HIGH"
                elif medium_risk_ratio > 0.3:
                    risk_summary["overall_risk_level"] = "MEDIUM"
                else:
                    risk_summary["overall_risk_level"] = "LOW_TO_MEDIUM"

            # Identify risk factors
            if high_risk_count > 0:
                risk_summary["risk_factors"].append(f"{high_risk_count} high-risk symbols in portfolio")

            if volatility_scores:
                avg_volatility = sum(volatility_scores) / len(volatility_scores)
                if avg_volatility > 40:
                    risk_summary["risk_factors"].append("High average portfolio volatility")
                elif avg_volatility > 30:
                    risk_summary["risk_factors"].append("Moderate portfolio volatility")

            # Data quality risks
            data_quality_summary = overall_metrics.get("data_quality_summary", {})
            poor_data_count = data_quality_summary.get("poor", 0) + data_quality_summary.get("fair", 0)
            if poor_data_count > 0:
                risk_summary["risk_factors"].append(f"{poor_data_count} symbols with poor data quality")

            # Market concentration risk
            analyzed_symbols = overall_metrics.get("analyzed_symbols", 0)
            if analyzed_symbols < 5:
                risk_summary["risk_factors"].append("Limited diversification (fewer than 5 symbols)")

            # Generate mitigation strategies
            if high_risk_count > 0:
                risk_summary["mitigation_strategies"].append("Reduce allocation to high-risk symbols")
                risk_summary["mitigation_strategies"].append("Implement stricter stop-loss mechanisms")

            if avg_volatility > 35:
                risk_summary["mitigation_strategies"].append("Consider volatility-based position sizing")
                risk_summary["mitigation_strategies"].append("Implement gradual entry strategies")

            if analyzed_symbols < 8:
                risk_summary["mitigation_strategies"].append("Increase portfolio diversification")
                risk_summary["mitigation_strategies"].append("Consider adding symbols from different sectors")

            risk_summary["mitigation_strategies"].extend([
                "Regular portfolio rebalancing",
                "Maintain emergency cash reserves",
                "Monitor market conditions closely",
                "Review risk tolerance periodically"
            ])

            # Portfolio diversification score
            diversification_score = min(analyzed_symbols * 10, 100)  # Max 100 for 10+ symbols
            if diversification_score >= 80:
                diversification_assessment = "EXCELLENT"
            elif diversification_score >= 60:
                diversification_assessment = "GOOD"
            elif diversification_score >= 40:
                diversification_assessment = "FAIR"
            else:
                diversification_assessment = "POOR"

            risk_summary["portfolio_diversification"] = {
                "score": diversification_score,
                "assessment": diversification_assessment
            }

            return risk_summary

        except Exception as e:
            logger.error(f"Error generating risk assessment: {e}")
            return {
                "overall_risk_level": "ERROR",
                "risk_factors": ["Unable to assess risk due to analysis error"],
                "mitigation_strategies": ["Review portfolio manually"],
                "error": str(e)
            }

    def _calculate_enhanced_results(self, symbol: str, total_investment: float,
                                  final_portfolio_value: float, total_units: float,
                                  trades: List[Dict], config, start_date: str,
                                  end_date: str, data, monthly_exceeded_count: int,
                                  price_threshold_skipped: int) -> Dict:
        """Calculate comprehensive performance metrics"""

        total_return_percent = ((final_portfolio_value / total_investment) - 1) * 100 if total_investment > 0 else 0

        # Calculate CAGR
        start_timestamp = data.iloc[0]['timestamp']
        end_timestamp = data.iloc[-1]['timestamp']
        years = (end_timestamp - start_timestamp).days / 365.25
        cagr_percent = ((final_portfolio_value / total_investment) ** (1 / years) - 1) * 100 if years > 0 and total_investment > 0 else 0

        # Get monthly summary and analytics
        monthly_summary = self.monthly_tracker.get_monthly_summary(symbol)
        skipped_investments = self.monthly_tracker.skipped_investments

        # Count different trade types
        extreme_trades = len([t for t in trades if t.get('is_extreme', False)])
        force_trades = len([t for t in trades if t.get('is_force', False)])
        regular_trades = len(trades) - extreme_trades - force_trades

        results = {
            'symbol': symbol,
            'strategy_name': 'Enhanced SIP V4 with Comprehensive Fixes',
            'period': f"{start_date} to {end_date}",
            'total_investment': float(total_investment),
            'final_portfolio_value': float(final_portfolio_value),
            'total_units': float(total_units),
            'average_buy_price': float(total_investment / total_units) if total_units > 0 else 0,
            'total_return_percent': float(total_return_percent),
            'cagr_percent': float(cagr_percent),
            'num_trades': len(trades),
            'num_skipped': len(skipped_investments),
            'trade_breakdown': {
                'regular_trades': regular_trades,
                'extreme_trades': extreme_trades,
                'force_trades': force_trades
            },
            'monthly_limit_exceeded': monthly_exceeded_count,
            'price_threshold_skipped': price_threshold_skipped,
            'trades': trades,
            'skipped_investments': skipped_investments,
            'monthly_summary': monthly_summary,
            'config': config,
            'enhancements_applied': [
                'Force remaining investment at month end',
                'Holiday handling for regular SIP dates',
                'Extreme opportunity detection (>15% drawdown gets 4x investment)',
                'Monthly investment limits with extreme bypass',
                'Comprehensive trade categorization'
            ]
        }

        logger.info(f"✅ Enhanced backtest V4 completed for {symbol}")
        logger.info(f"📊 Results: {regular_trades} regular + {extreme_trades} extreme + {force_trades} force trades")
        logger.info(f"💰 Total investment: ₹{total_investment:,.2f}, Final value: ₹{final_portfolio_value:,.2f}")
        logger.info(f"📈 Return: {total_return_percent:.2f}%, CAGR: {cagr_percent:.2f}%")

        return results

    def _calculate_max_drawdown(self, prices) -> float:
        """Calculate maximum drawdown safely"""
        try:
            if len(prices) < 2:
                return 0.0
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return float(drawdown.min() * 100)
        except Exception:
            return 0.0

    def _calculate_sharpe_ratio(self, returns, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio safely"""
        try:
            if len(returns) < 2:
                return 0.0
            excess_returns = returns - (risk_free_rate / 252)
            if excess_returns.std() == 0:
                return 0.0
            return float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))
        except Exception:
            return 0.0

    def _calculate_sortino_ratio(self, returns, risk_free_rate: float = 0.05) -> float:
        """Calculate Sortino ratio safely"""
        try:
            if len(returns) < 2:
                return 0.0
            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            return float(excess_returns.mean() / downside_returns.std() * np.sqrt(252))
        except Exception:
            return 0.0


class MonthlyInvestmentTracker:
    """
    Tracks monthly investments with limits and price threshold logic
    Implements the core requirements:
    1. Monthly investment limits (default 4x fixed_investment)
    2. 4% price reduction threshold for multiple signals in same month
    """

    def __init__(self, max_monthly_amount: float, price_reduction_threshold: float = 4.0):
        self.max_monthly_amount = max_monthly_amount
        self.price_reduction_threshold = price_reduction_threshold / 100
        self.monthly_investments = {}
        self.skipped_investments = []
        self.extreme_trades_made = {}

    def get_month_key(self, date: datetime) -> str:
        """Generate month key for tracking"""
        return f"{date.year}-{date.month:02d}"

    def has_extreme_trade_this_month(self, symbol: str, current_date: datetime) -> bool:
        """Check if an extreme trade (4x investment) was made this month"""
        month_key = self.get_month_key(current_date)
        return self.extreme_trades_made.get(f"{symbol}_{month_key}", False)

    def mark_extreme_trade(self, symbol: str, current_date: datetime):
        """Mark that an extreme trade was made this month"""
        month_key = self.get_month_key(current_date)
        self.extreme_trades_made[f"{symbol}_{month_key}"] = True
        logger.info(f"🔥 Marked extreme trade for {symbol} in {month_key}")

    def can_invest(self, symbol: str, current_date: datetime,
                   intended_amount: float, current_price: float,
                   is_extreme_opportunity: bool = False) -> Dict:
        """
        Check if investment is allowed based on monthly limits and price thresholds
        1. Extreme opportunities (>15% drawdown) bypass monthly limits
        2. Remaining amount force investment at month end
        """
        month_key = self.get_month_key(current_date)

        # Initialize symbol tracking
        if symbol not in self.monthly_investments:
            self.monthly_investments[symbol] = {}

        if month_key not in self.monthly_investments[symbol]:
            self.monthly_investments[symbol][month_key] = {
                'total_invested': 0.0,
                'investments': [],
                'last_price': None,
                'investment_count': 0,
                'extreme_trade_made': False
            }

        month_data = self.monthly_investments[symbol][month_key]
        current_invested = month_data['total_invested']
        remaining_budget = self.max_monthly_amount - current_invested

        # ENHANCEMENT 3: Handle extreme opportunities (>15% drawdown)
        if is_extreme_opportunity:
            # Check if we've already made an extreme trade this month
            if self.has_extreme_trade_this_month(symbol, current_date):
                self.record_skip(symbol, current_date, intended_amount,
                                 "Extreme trade already made this month", current_price)
                return {
                    'can_invest': False,
                    'suggested_amount': 0.0,
                    'reason': 'Extreme trade already made this month',
                    'remaining_budget': remaining_budget
                }

            # For extreme opportunities, allow 4x investment regardless of monthly limit
            extreme_amount = intended_amount  # This should already be 4x from caller
            logger.info(f"🔥 EXTREME OPPORTUNITY: {symbol} - Investing ₹{extreme_amount:,.2f} "
                        f"(bypassing monthly limit of ₹{self.max_monthly_amount:,.2f})")

            # Mark that we've made an extreme trade this month
            self.mark_extreme_trade(symbol, current_date)
            month_data['extreme_trade_made'] = True

            return {
                'can_invest': True,
                'suggested_amount': extreme_amount,
                'reason': 'Extreme opportunity - bypassing monthly limit',
                'remaining_budget': 0.0,  # No more investments this month
                'is_extreme': True
            }

        # Check if extreme trade was made this month - if so, skip all other investments
        if month_data.get('extreme_trade_made', False):
            self.record_skip(symbol, current_date, intended_amount,
                             "Extreme trade made this month - skipping other investments", current_price)
            return {
                'can_invest': False,
                'suggested_amount': 0.0,
                'reason': 'Extreme trade made this month',
                'remaining_budget': 0.0
            }

        # Check 1: Monthly limit for regular investments
        if current_invested >= self.max_monthly_amount:
            self.record_skip(symbol, current_date, intended_amount,
                             "Monthly investment limit reached", current_price)
            return {
                'can_invest': False,
                'suggested_amount': 0.0,
                'reason': 'Monthly limit exceeded',
                'remaining_budget': 0.0
            }

        # Check 2: Adjust amount if it would exceed monthly limit
        if current_invested + intended_amount > self.max_monthly_amount:
            adjusted_amount = remaining_budget
            if adjusted_amount < intended_amount * 0.1:  # Less than 10% of intended
                self.record_skip(symbol, current_date, intended_amount,
                                 "Insufficient remaining monthly budget", current_price)
                return {
                    'can_invest': False,
                    'suggested_amount': 0.0,
                    'reason': 'Insufficient remaining budget',
                    'remaining_budget': remaining_budget
                }
        else:
            adjusted_amount = intended_amount

        # Check 3: Price reduction threshold for multiple investments in same month
        if month_data['investment_count'] > 0 and month_data['last_price'] is not None:
            price_reduction = (month_data['last_price'] - current_price) / month_data['last_price']

            if price_reduction < self.price_reduction_threshold:
                self.record_skip(symbol, current_date, intended_amount,
                                 f"Price reduction {price_reduction * 100:.1f}% < threshold {self.price_reduction_threshold * 100:.1f}%",
                                 current_price)
                return {
                    'can_invest': False,
                    'suggested_amount': 0.0,
                    'reason': 'Price reduction threshold not met',
                    'remaining_budget': remaining_budget
                }

        # All checks passed
        return {
            'can_invest': True,
            'suggested_amount': adjusted_amount,
            'reason': 'Investment approved',
            'remaining_budget': remaining_budget - adjusted_amount
        }

    def force_remaining_investment(self, symbol: str, current_date: datetime,
                                   current_price: float, fixed_investment: float) -> Optional[Dict]:
        """
        ENHANCEMENT 1: Force invest remaining amount if there's budget left at month end
        """
        month_key = self.get_month_key(current_date)

        if symbol not in self.monthly_investments or month_key not in self.monthly_investments[symbol]:
            return None

        month_data = self.monthly_investments[symbol][month_key]
        current_invested = month_data['total_invested']
        remaining_budget = fixed_investment - current_invested

        # Only force if we have significant remaining budget (at least 20% of fixed investment)
        min_threshold = fixed_investment * 0.2

        if remaining_budget >= min_threshold and not month_data.get('extreme_trade_made', False):
            logger.info(f"💰 FORCE INVESTMENT: {symbol} - Remaining budget ₹{remaining_budget:,.2f} "
                        f"on {current_date.strftime('%Y-%m-%d')}")
            return {
                'can_invest': True,
                'suggested_amount': remaining_budget,
                'reason': 'Force investment - remaining monthly budget',
                'remaining_budget': 0.0,
                'is_force_investment': True
            }

        return None

    def record_investment(self, symbol: str, date: datetime, amount: float,
                          price: float, is_extreme: bool = False) -> None:
        """Record a successful investment"""
        month_key = self.get_month_key(date)

        if symbol not in self.monthly_investments:
            self.monthly_investments[symbol] = {}
        if month_key not in self.monthly_investments[symbol]:
            self.monthly_investments[symbol][month_key] = {
                'total_invested': 0.0,
                'investments': [],
                'last_price': None,
                'investment_count': 0,
                'extreme_trade_made': False
            }

        month_data = self.monthly_investments[symbol][month_key]
        month_data['total_invested'] += amount
        month_data['last_price'] = price
        month_data['investment_count'] += 1

        if is_extreme:
            month_data['extreme_trade_made'] = True

        month_data['investments'].append({
            'date': date.strftime('%Y-%m-%d'),
            'amount': amount,
            'price': price,
            'units': amount / price,
            'is_extreme': is_extreme
        })

        logger.debug(f"💰 Recorded investment: {symbol} ₹{amount:,.2f} @ ₹{price:.2f} "
                     f"(Month total: ₹{month_data['total_invested']:,.2f})")

    def record_skip(self, symbol: str, date: datetime, intended_amount: float,
                    reason: str, current_price: float) -> None:
        """Record a skipped investment with reason"""
        skip_record = {
            'symbol': symbol,
            'date': date.strftime('%Y-%m-%d'),
            'intended_amount': intended_amount,
            'reason': reason,
            'current_price': current_price,
            'month_key': self.get_month_key(date)
        }

        self.skipped_investments.append(skip_record)
        logger.info(f"⏭️ Skipped investment: {symbol} ₹{intended_amount:,.2f} - {reason}")

    def get_monthly_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get comprehensive monthly investment summary"""
        if symbol:
            # Summary for specific symbol
            if symbol not in self.monthly_investments:
                return {}

            summary = {}
            for month_key, data in self.monthly_investments[symbol].items():
                summary[month_key] = {
                    'total_invested': data['total_invested'],
                    'num_investments': len(data['investments']),
                    'remaining_budget': self.max_monthly_amount - data['total_invested'],
                    'budget_utilization_percent': (data['total_invested'] / self.max_monthly_amount) * 100,
                    'investments': data['investments']
                }
            return summary
        else:
            # Summary for all symbols
            all_summary = {}
            for symbol, symbol_data in self.monthly_investments.items():
                all_summary[symbol] = self.get_monthly_summary(symbol)
            return all_summary


class EnhancedSIPStrategyWithLimits(EnhancedSIPStrategy):
    """Enhanced SIP Strategy with monthly investment limits and price thresholds"""

    def __init__(self, nsedata_session: AsyncSession = None, trading_session: AsyncSession = None):
        super().__init__()
        self.nsedata_session = nsedata_session
        self.trading_session = trading_session
        self.monthly_tracker = None

    async def run_backtest(self, symbol: str, start_date: str, end_date: str,
                           config: SIPConfig) -> Optional[Dict]:
        """
        Enhanced backtest with monthly limits and price thresholds

        This method implements both requirements:
        1. Monthly investment limits with default 4x fixed_investment
        2. 4% price reduction threshold for multiple signals in same month
        """
        try:
            logger.info(f"🚀 Starting enhanced SIP backtest for {symbol}")
            logger.info(f"📅 Period: {start_date} to {end_date}")
            logger.info(f"💰 Config: Fixed=₹{config.fixed_investment:,.2f}, "
                        f"Monthly=₹{config.max_amount_in_a_month:,.2f}, "
                        f"Extreme=₹{config.fixed_investment * config.extreme_drawdown_inv_multiplier:,.2f}")

            # Initialize monthly tracker with enhanced config
            self.monthly_tracker = MonthlyInvestmentTracker(
                max_monthly_amount=config.max_amount_in_a_month,
                price_reduction_threshold=config.price_reduction_threshold
            )

            # Fetch and prepare data
            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)

            # Initialize tracking variables
            total_investment = 0.0
            total_units = 0.0
            trades = []
            monthly_exceeded_count = 0
            price_threshold_skipped = 0

            # Enhanced simulation with monthly limits and price thresholds
            for i, row in data.iterrows():
                current_date = row['timestamp']
                current_price = row['close']

                # Determine if this is an investment opportunity
                should_invest, investment_reason = self._should_invest(row, config, i, data, symbol)

                if should_invest:
                    logger.info(f"Investment signal: {investment_reason},"
                                f" Date: {current_date.strftime('%Y-%m-%d')},"
                                f" Price: ₹{current_price:.2f}")
                    # Calculate investment amount based on market conditions
                    base_investment_amount, is_extreme  = self.determine_investment_amount(
                        current_price, data, config, i
                    )
                    logger.info(f"Investment amount: {base_investment_amount}")

                    # Check if investment is allowed (APPLY MONTHLY LIMITS AND PRICE THRESHOLD LOGIC)
                    investment_check = self.monthly_tracker.can_invest(
                        symbol, current_date, base_investment_amount, current_price, is_extreme
                    )

                    if investment_check['can_invest']:
                        # Execute investment with approved/adjusted amount
                        final_amount = investment_check['suggested_amount']
                        units_bought = final_amount / current_price

                        total_investment += final_amount
                        total_units += units_bought

                        # Record the investment in tracker
                        self.monthly_tracker.record_investment(
                            symbol, current_date, final_amount, current_price,
                            is_extreme or investment_check.get('is_extreme', False)
                        )

                        # Record trade
                        trade = {
                            'symbol': symbol,
                            'date': current_date.strftime('%Y-%m-%d'),
                            'price': float(current_price),
                            'units': float(units_bought),
                            'amount': float(final_amount),
                            'total_investment': float(total_investment),
                            'total_units': float(total_units),
                            'portfolio_value': float(total_units * current_price),
                            "return_on_investment": float((total_units * current_price - total_investment) / total_investment * 100),
                            'trade_type': investment_reason,
                            'drawdown': float(row.get('Drawdown_100', 0)),
                            'original_intended_amount': float(base_investment_amount),
                            'amount_adjusted': final_amount != base_investment_amount,
                            'is_extreme': is_extreme or investment_check.get('is_extreme', False),
                            'is_force': investment_check.get('is_force_investment', False)
                        }
                        trades.append(trade)

                        logger.debug(f"💰 Investment executed: {symbol} ₹{final_amount:,.2f}"
                                     f" @ ₹{current_price:.2f} ({units_bought:.2f} units)")

                    else:
                        # Track skip reasons
                        skip_reason = investment_check['reason']
                        if 'Monthly limit' in skip_reason:
                            monthly_exceeded_count += 1
                        elif 'Price reduction' in skip_reason:
                            price_threshold_skipped += 1

                        logger.debug(f"⏭️ Investment skipped: {symbol} - {skip_reason}")

                if self._is_month_end_or_fallback_approaching(current_date, config, i, data):
                    force_check = self.monthly_tracker.force_remaining_investment(
                        symbol, current_date, current_price, config.fixed_investment
                    )

                    if force_check and force_check['can_invest']:
                        final_amount = force_check['suggested_amount']
                        units_bought = final_amount / current_price
                        total_investment += final_amount
                        total_units += units_bought

                        # Record the force investment
                        self.monthly_tracker.record_investment(
                            symbol, current_date, final_amount, current_price, False
                        )

                        # Create trade record
                        trade = {
                            'symbol': symbol,
                            'date': current_date.strftime('%Y-%m-%d'),
                            'price': current_price,
                            'amount': final_amount,
                            'units': units_bought,
                            'portfolio_value': total_units * current_price,
                            'total_investment': total_investment,
                            'drawdown': row.get('Drawdown_100', 0),
                            'trade_type': 'force_remaining_investment',
                            'is_extreme': False,
                            'is_force': True
                        }
                        trades.append(trade)

                        logger.info(f"🔄 FORCE INVESTMENT: ₹{final_amount:,.2f} "
                                    f"(remaining budget) at ₹{current_price:.2f}")

            # Calculate final portfolio value
            final_price = data.iloc[-1]['close']
            final_portfolio_value = total_units * final_price

            # Return comprehensive results
            return self._calculate_enhanced_results(
                symbol, total_investment, final_portfolio_value, total_units,
                trades, config, start_date, end_date, data,
                monthly_exceeded_count, price_threshold_skipped
            )

        except Exception as e:
            logger.error(f"Error running enhanced backtest for {symbol}: {e}")
            return None

    async def get_next_investment_signals(self, symbol: str, config: SIPConfig) -> Dict[str, Any]:
        """Generate enhanced investment signals with robust error handling"""
        try:
            logger.info(f"🎯 Getting enhanced investment signals for {symbol}")

            # Get recent data for analysis
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)
            if data.empty:
                return {
                    "signal": "NO_DATA",
                    "confidence": 0,
                    "recommended_amount": 0,
                    "current_price": 0,
                    "message": "Insufficient data for analysis",
                    "symbol": symbol
                }

            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)

            # Get current market data
            current_row = data.iloc[-1]
            current_date = current_row['timestamp']
            current_price = current_row['close']
            drawdown_100 = current_row.get('Drawdown_100', 0)
            rsi = current_row.get('RSI', 0)

            # Initialize monthly tracker for current analysis
            if not self.monthly_tracker:
                self.monthly_tracker = MonthlyInvestmentTracker(
                    max_monthly_amount=config.max_amount_in_a_month,
                    price_reduction_threshold=config.price_reduction_threshold
                )

            # Check investment signals with enhanced logic
            should_invest, investment_reason = self._should_invest(
                current_row, config, len(data) - 1, data, symbol
            )

            if not should_invest:
                return {
                    "signal": "NORMAL",
                    "confidence": 0.1,
                    "recommended_amount": 0,
                    "current_price": current_price,
                    "message": f"No investment signal: {investment_reason}",
                    "symbol": symbol,
                    "drawdown_100": drawdown_100,
                    "analysis_timestamp": datetime.now().isoformat()
                }

            # Calculate investment amount with enhanced logic
            base_amount, is_extreme = self.determine_investment_amount(
                current_price, data, config, len(data) - 1
            )

            # Check monthly limits and constraints
            investment_check = self.monthly_tracker.can_invest(
                symbol, current_date, base_amount, current_price, is_extreme
            )

            if not investment_check['can_invest']:
                # Check for force investment opportunity
                force_check = self.monthly_tracker.force_remaining_investment(
                    symbol, current_date, current_price, config.fixed_investment
                )

                if force_check and force_check['can_invest']:
                    return {
                        "signal": "BUY",
                        "confidence": 0.8,
                        "recommended_amount": force_check['suggested_amount'],
                        "current_price": current_price,
                        "investment_multiplier": force_check['suggested_amount'] / config.fixed_investment,
                        "drawdown_100": drawdown_100,
                        "message": "Force investment - remaining monthly budget",
                        "symbol": symbol,
                        "is_force_investment": True,
                        "is_extreme_opportunity": False,
                        "is_holiday_makeup": False,
                        "trade_type": "force_remaining_investment",
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "signal": "SKIP",
                        "confidence": 0,
                        "recommended_amount": 0,
                        "current_price": current_price,
                        "message": investment_check['reason'],
                        "symbol": symbol,
                        "analysis_timestamp": datetime.now().isoformat()
                    }

            # Determine signal strength and type
            final_amount = investment_check['suggested_amount']
            investment_multiplier = final_amount / config.fixed_investment

            # Determine signal type based on conditions
            if is_extreme:
                signal_type = "STRONG_BUY"
                confidence = 0.95
                trade_type = "extreme_opportunity"
            elif investment_reason.startswith("sip_makeup") or investment_reason.startswith("last_trading_day"):
                signal_type = "BUY"
                confidence = 0.7
                trade_type = "holiday_makeup"
            elif drawdown_100 <= config.major_drawdown_threshold:
                signal_type = "STRONG_BUY"
                confidence = 0.85
                trade_type = "major_drawdown"
            elif drawdown_100 <= config.minor_drawdown_threshold:
                signal_type = "BUY"
                confidence = 0.75
                trade_type = "moderate_drawdown"
            else:
                signal_type = "WEAK_BUY"
                confidence = 0.6
                trade_type = "regular_sip"

            # RSI analysis
            if rsi < 30:
                signal_type = "STRONG_BUY"
                confidence = 0.95
                trade_type = "Oversold_condition"
            elif rsi > 70:
                confidence -= 0.5

            return {
                "signal": signal_type,
                "confidence": confidence,
                "recommended_amount": final_amount,
                "current_price": current_price,
                "investment_multiplier": investment_multiplier,
                "drawdown_100": drawdown_100,
                "rsi": rsi,
                "message": f"{trade_type.replace('_', ' ').title()} - {investment_reason}",
                "symbol": symbol,
                "is_extreme_opportunity": is_extreme,
                "is_force_investment": investment_check.get('is_force_investment', False),
                "is_holiday_makeup": "makeup" in investment_reason or "last_trading_day" in investment_reason,
                "trade_type": trade_type,
                "remaining_monthly_budget": investment_check.get('remaining_budget', 0),
                "analysis_timestamp": datetime.now().isoformat(),
                "next_fallback_date": self._calculate_next_fallback_date(current_date.date(),
                                                                         config.fallback_day).strftime('%Y-%m-%d')
            }

        except Exception as e:
            logger.error(f"Error getting enhanced investment signals for {symbol}: {e}")
            return {
                "signal": "ERROR",
                "confidence": 0,
                "recommended_amount": 0,
                "current_price": 0,
                "message": f"Signal analysis failed: {str(e)}",
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat()
            }

    async def get_portfolio_recommendations(self, symbols: List[str],
                                            config: SIPConfig) -> Dict[str, Any]:
        """Get recommendations for multiple symbols in a portfolio"""
        try:
            recommendations = {}
            overall_signal = "NORMAL"
            total_confidence = 0

            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            for symbol in symbols:
                try:
                    signal = await self.get_next_investment_signals(symbol, config)
                    recommendations[symbol] = signal
                    total_confidence += signal.get('confidence', 0)
                except Exception as e:
                    logger.error(f"Error getting recommendation for {symbol}: {e}")
                    recommendations[symbol] = {
                        "signal": "ERROR",
                        "confidence": 0,
                        "message": f"Error analyzing {symbol}"
                    }

            # Determine overall portfolio signal
            if recommendations:
                avg_confidence = total_confidence / len(recommendations)
                strong_buy_count = sum(1 for r in recommendations.values() if r.get('signal') == 'STRONG_BUY')
                buy_count = sum(1 for r in recommendations.values() if r.get('signal') == 'BUY')

                if strong_buy_count > len(symbols) * 0.5:
                    overall_signal = "STRONG_BUY"
                elif (strong_buy_count + buy_count) > len(symbols) * 0.3:
                    overall_signal = "BUY"
                elif avg_confidence < 0.3:
                    overall_signal = "AVOID"

            return {
                "overall_signal": overall_signal,
                "overall_confidence": avg_confidence if recommendations else 0,
                "symbol_recommendations": recommendations,
                "analysis_timestamp": datetime.now().isoformat(),
                "symbols_analyzed": len(recommendations)
            }

        except Exception as e:
            logger.error(f"Error getting portfolio recommendations: {e}")
            return {
                "overall_signal": "ERROR",
                "overall_confidence": 0,
                "symbol_recommendations": {},
                "error": str(e)
            }

    async def generate_investment_report(self, symbols: List[str], config: SIPConfig) -> Dict[str, Any]:
        """Generate comprehensive investment report for multiple symbols - OPTIMIZED VERSION"""
        try:
            logger.info(f"Generating investment report for {len(symbols)} symbols")

            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            symbol_reports = {}
            overall_metrics = {
                "total_symbols": len(symbols),
                "analyzed_symbols": 0,
                "successful_analyses": 0,
                "strong_buy_signals": 0,
                "buy_signals": 0,
                "avoid_signals": 0,
                "error_count": 0,
                "avg_confidence": 0,
                "data_quality_summary": {
                    "excellent": 0,
                    "good": 0,
                    "fair": 0,
                    "poor": 0,
                    "error": 0
                }
            }

            total_confidence = 0
            successful_recommendations = 0

            # Process symbols with better error isolation
            for symbol in symbols:
                try:
                    logger.info(f"Analyzing {symbol} for investment report")

                    # Get data quality with timeout protection
                    try:
                        data_quality = await asyncio.wait_for(
                            self.validate_symbol_data_quality(symbol, start_date, end_date),
                            timeout=30.0  # 30 second timeout per symbol
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout validating data quality for {symbol}")
                        data_quality = {
                            "symbol": symbol,
                            "status": "ERROR",
                            "message": "Data validation timeout",
                            "data_points": 0,
                            "coverage_percent": 0
                        }
                    except Exception as dq_error:
                        logger.error(f"Error validating data quality for {symbol}: {dq_error}")
                        data_quality = {
                            "symbol": symbol,
                            "status": "ERROR",
                            "message": f"Data validation failed: {str(dq_error)}",
                            "data_points": 0,
                            "coverage_percent": 0
                        }

                    # Update data quality summary
                    quality_status = data_quality.get('status', 'ERROR').lower()
                    if quality_status in overall_metrics["data_quality_summary"]:
                        overall_metrics["data_quality_summary"][quality_status] += 1
                    else:
                        overall_metrics["data_quality_summary"]["error"] += 1

                    # Skip if no data available
                    if data_quality.get('status') in ['NO_DATA', 'ERROR']:
                        symbol_reports[symbol] = {
                            "status": "DATA_UNAVAILABLE",
                            "message": data_quality.get('message', 'Data not available'),
                            "data_quality": data_quality,
                            "statistics": None,
                            "investment_signals": None,
                            "recommendation": {
                                "recommendation": "SKIP - No Data",
                                "priority": "LOW",
                                "confidence_score": 0,
                                "considerations": ["No historical data available"]
                            }
                        }
                        overall_metrics["error_count"] += 1
                        continue

                    # Get statistics with error handling
                    try:
                        stats = await asyncio.wait_for(
                            self.get_symbol_statistics(symbol),
                            timeout=45.0  # 45 second timeout for statistics
                        )
                        # Ensure stats is never None
                        if stats is None:
                            raise ValueError("get_symbol_statistics returned None")

                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout getting statistics for {symbol}")
                        stats = {
                            "symbol": symbol,
                            "status": "TIMEOUT",
                            "message": "Statistics calculation timeout",
                            "price_statistics": {},
                            "technical_indicators": {},
                            "risk_metrics": {}
                        }
                    except Exception as stats_error:
                        logger.error(f"Error getting statistics for {symbol}: {stats_error}")
                        stats = {
                            "symbol": symbol,
                            "status": "ERROR",
                            "message": f"Statistics calculation failed: {str(stats_error)}",
                            "price_statistics": {},
                            "technical_indicators": {},
                            "risk_metrics": {}
                        }

                    # Get investment signals with error handling
                    try:
                        data = await self.fetch_data_from_db_async(symbol, start_date, end_date)
                        if not data.empty:
                            signals = await self.get_next_investment_signals(symbol, config)
                            # Ensure signals is never None
                            if signals is None:
                                signals = {"signal": "ERROR", "confidence": 0, "message": "Signal generation failed"}
                        else:
                            signals = {"signal": "NO_DATA", "confidence": 0, "message": "No data for signals"}

                    except Exception as signal_error:
                        logger.error(f"Error getting signals for {symbol}: {signal_error}")
                        signals = {
                            "signal": "ERROR",
                            "confidence": 0,
                            "message": f"Signal generation failed: {str(signal_error)}"
                        }

                    # Generate recommendation with null safety
                    try:
                        recommendation = self._generate_symbol_recommendation(signals, stats, data_quality)
                        if recommendation is None:
                            raise ValueError("_generate_symbol_recommendation returned None")
                    except Exception as rec_error:
                        logger.error(f"Error generating recommendation for {symbol}: {rec_error}")
                        recommendation = {
                            "recommendation": "ERROR - Recommendation Failed",
                            "priority": "LOW",
                            "confidence_score": 0,
                            "signal_type": "ERROR",
                            "considerations": [f"Recommendation error: {str(rec_error)}"],
                            "error": str(rec_error)
                        }

                    # Compile report for this symbol
                    symbol_reports[symbol] = {
                        "status": "SUCCESS" if stats.get('status') == 'SUCCESS' else "PARTIAL",
                        "data_quality": data_quality,
                        "statistics": stats,
                        "investment_signals": signals,
                        "recommendation": recommendation
                    }

                    # Update overall metrics
                    overall_metrics["analyzed_symbols"] += 1

                    if stats.get('status') == 'SUCCESS':
                        overall_metrics["successful_analyses"] += 1

                    confidence_val = signals.get('confidence', 0)
                    if confidence_val > 0:
                        total_confidence += confidence_val
                        successful_recommendations += 1

                    signal_type = signals.get('signal', 'NORMAL')
                    if signal_type == 'STRONG_BUY':
                        overall_metrics["strong_buy_signals"] += 1
                    elif signal_type == 'BUY':
                        overall_metrics["buy_signals"] += 1
                    elif signal_type == 'AVOID':
                        overall_metrics["avoid_signals"] += 1

                except Exception as symbol_error:
                    logger.error(f"Critical error analyzing {symbol}: {symbol_error}")
                    symbol_reports[symbol] = {
                        "status": "CRITICAL_ERROR",
                        "message": f"Analysis failed: {str(symbol_error)}",
                        "error": str(symbol_error)
                    }
                    overall_metrics["error_count"] += 1

            # Calculate overall metrics safely
            if successful_recommendations > 0:
                overall_metrics["avg_confidence"] = total_confidence / successful_recommendations
            else:
                overall_metrics["avg_confidence"] = 0

            # Generate portfolio recommendation with error handling
            try:
                portfolio_recommendation = self._generate_portfolio_recommendation(overall_metrics, symbol_reports)
            except Exception as portfolio_error:
                logger.error(f"Error generating portfolio recommendation: {portfolio_error}")
                portfolio_recommendation = {
                    "portfolio_action": "MANUAL_REVIEW",
                    "recommendations": ["Portfolio analysis failed - manual review required"],
                    "error": str(portfolio_error)
                }

            # Generate risk assessment with error handling
            try:
                risk_assessment = self._generate_risk_assessment(symbol_reports, overall_metrics)
            except Exception as risk_error:
                logger.error(f"Error generating risk assessment: {risk_error}")
                risk_assessment = {
                    "overall_risk_level": "UNKNOWN",
                    "risk_factors": ["Risk assessment failed"],
                    "mitigation_strategies": ["Manual risk review required"],
                    "error": str(risk_error)
                }

            return {
                "report_generated": datetime.now().isoformat(),
                "analysis_period": f"{start_date} to {end_date}",
                "overall_metrics": overall_metrics,
                "portfolio_recommendation": portfolio_recommendation,
                "risk_assessment": risk_assessment,
                "symbol_reports": symbol_reports,
                "processing_summary": {
                    "total_symbols_requested": len(symbols),
                    "symbols_analyzed": overall_metrics["analyzed_symbols"],
                    "successful_analyses": overall_metrics["successful_analyses"],
                    "error_count": overall_metrics["error_count"],
                    "success_rate": round((overall_metrics["successful_analyses"] / len(symbols)) * 100, 2) if len(
                        symbols) > 0 else 0
                },
                "disclaimer": "This report is for educational purposes only and should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions."
            }

        except Exception as e:
            logger.error(f"Critical error generating investment report: {e}")
            return {
                "status": "CRITICAL_ERROR",
                "message": f"Failed to generate report: {str(e)}",
                "report_generated": datetime.now().isoformat(),
                "error": str(e)
            }

    def _should_invest(self, row, config, i: int, data: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
        """Determine if we should invest based on strategy conditions"""
        current_date = row['timestamp']

        # Check if it's a regular SIP date (fallback day of month with holiday handling)
        is_sip_date, sip_reason = self._check_sip_date_with_holiday_handling(
            current_date, config, i, data, symbol
        )

        # Check for drawdown opportunities
        drawdown_100 = row.get('Drawdown_100', 0)
        drawdown_50 = row.get('Drawdown_50', 0)
        drawdown_20 = row.get('Drawdown_20', 0)

        is_extreme_drawdown = drawdown_100 <= config.extreme_drawdown_threshold
        is_drawdown_opportunity = (
                drawdown_100 <= config.major_drawdown_threshold or
                drawdown_100 <= config.minor_drawdown_threshold or
                drawdown_50 <= config.major_drawdown_threshold or
                drawdown_50 <= config.minor_drawdown_threshold or
                drawdown_20 <= config.major_drawdown_threshold or
                drawdown_20 <= config.minor_drawdown_threshold
        )

        if is_extreme_drawdown:
            return True, f"extreme_drawdown_{abs(drawdown_100):.1f}%"
        if is_sip_date:
            return True, "regular_sip"
        elif is_drawdown_opportunity:
            return True, f"drawdown_opportunity_{abs(drawdown_100):.1f}%"
        else:
            return False, "no_signal"

    def _check_sip_date_with_holiday_handling(self, current_date: datetime,
                                              config: SIPConfig, i: int,
                                              data: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
        """
        ENHANCEMENT 2: Check SIP date with holiday handling
        """
        # Check if it's exactly the fallback day
        if current_date.day == config.fallback_day:
            return True, "regular_sip_fallback_day"

        # Check if fallback day was a holiday and this is the next available date
        # Look back up to 5 days to see if we missed the fallback day
        for days_back in range(1, 6):
            if i >= days_back:
                check_date = data.iloc[i - days_back]['timestamp']
                if (check_date.day == config.fallback_day and
                        check_date.month == current_date.month and
                        check_date.year == current_date.year):
                    # Found the fallback day in recent history, this could be makeup
                    return True, f"sip_makeup_after_{days_back}_days"

        # Check if this is the last trading day of the month and we haven't invested yet
        if self._is_last_trading_day_of_month(current_date, i, data):
            month_key = f"{current_date.year}-{current_date.month:02d}"

            # Check if we have any investments this month
            if (self.monthly_tracker and
                    symbol in self.monthly_tracker.monthly_investments and
                    month_key in self.monthly_tracker.monthly_investments[symbol]):
                monthly_data = self.monthly_tracker.monthly_investments[symbol][month_key]
                if monthly_data['investment_count'] == 0:
                    return True, "last_trading_day_makeup"

        return False, "not_sip_date"

    def _is_last_trading_day_of_month(self, current_date: datetime,
                                      i: int, data: pd.DataFrame) -> bool:
        """Check if this is the last trading day of the month"""
        # Look ahead to see if next trading day is in different month
        if i < len(data) - 1:
            next_date = data.iloc[i + 1]['timestamp']
            return current_date.month != next_date.month

        # If this is the last day in dataset, check calendar
        return current_date.month != (current_date + timedelta(days=1)).month

    def _is_month_end_or_fallback_approaching(self, current_date: datetime,
                                              config: SIPConfig, i: int,
                                              data: pd.DataFrame) -> bool:
        """
        ENHANCEMENT 1: Check if we should force remaining investment
        """
        # Check if we're within 3 trading days of month end
        days_to_check = 3
        for days_ahead in range(1, days_to_check + 1):
            if i + days_ahead < len(data):
                future_date = data.iloc[i + days_ahead]['timestamp']
                if future_date.month != current_date.month:
                    return True

        # Also check if we're past fallback day and approaching month end
        if current_date.day > config.fallback_day:
            if current_date.month == 12:
                first_next = datetime(current_date.year + 1, 1, 1)
            else:
                first_next = datetime(current_date.year, current_date.month + 1, 1)
            days_remaining_in_month = (first_next - current_date).days
            return days_remaining_in_month <= 5

        return False

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def _calculate_monthly_utilization(self, monthly_summary: Dict) -> Dict:
        """Calculate monthly budget utilization statistics"""
        if not monthly_summary:
            return {}

        utilizations = []
        for month_data in monthly_summary.values():
            utilization = month_data.get('budget_utilization_percent', 0)
            utilizations.append(utilization)

        if not utilizations:
            return {}

        return {
            'average_utilization_percent': sum(utilizations) / len(utilizations),
            'max_utilization_percent': max(utilizations),
            'min_utilization_percent': min(utilizations),
            'months_with_full_utilization': sum(1 for u in utilizations if u >= 95),
            'months_tracked': len(utilizations)
        }

    def _analyze_skips(self, skipped_investments: List[Dict]) -> Dict:
        """Analyze patterns in skipped investments"""
        if not skipped_investments:
            return {'total_skips': 0}

        skip_reasons = {}
        skip_amounts = []

        for skip in skipped_investments:
            reason = skip.get('reason', 'unknown')
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            skip_amounts.append(skip.get('intended_amount', 0))

        return {
            'total_skips': len(skipped_investments),
            'skip_reasons': skip_reasons,
            'total_skipped_amount': sum(skip_amounts),
            'average_skipped_amount': sum(skip_amounts) / len(skip_amounts) if skip_amounts else 0,
            'most_common_skip_reason': max(skip_reasons.items(), key=lambda x: x[1])[0] if skip_reasons else 'none'
        }


# Export key classes and functions
__all__ = [
    'EnhancedSIPStrategy',
    'SIPConfig',
    'Trade',
    'SIPResults',
    'SIPPortfolioTracker'
]
