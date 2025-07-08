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
    drawdown_threshold_1: float = -10.0
    drawdown_threshold_2: float = -4.0
    investment_multiplier_1: float = 2.0
    investment_multiplier_2: float = 3.0
    investment_multiplier_3: float = 5.0
    rolling_window: int = 100
    fallback_day: int = 22
    min_investment_gap_days: int = 5
    max_amount_in_a_month: Optional[float] = None  # Monthly investment limit
    price_reduction_threshold: float = 4.0  # Price reduction threshold percentage
    enable_monthly_limits: bool = True  # Enable/disable monthly tracking
    reset_on_month_change: bool = True  # Reset monthly tracker on new month

    def __post_init__(self):
        """Set default monthly limit if not provided"""
        if self.max_amount_in_a_month is None:
            # Default to 4x fixed investment (assuming weekly SIPs)
            self.max_amount_in_a_month = self.fixed_investment * 4

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
    total_return_percent: Optional[float] = None
    num_trades: Optional[int] = None
    num_skipped: Optional[int] = None
    # Monthly investment tracking fields
    monthly_investment_summary: Optional[Dict[str, Dict[str, Any]]] = None
    monthly_limit_exceeded: int = 0
    price_threshold_skipped: int = 0
    max_amount_in_a_month: float = 0.0
    price_reduction_threshold: Optional[float] = None
    average_monthly_investment: float = 0.0
    months_at_limit: int = 0
    skipped_investments: Optional[List[Dict]] = None

    config_used: Optional[Dict[str, Any]] = None


class SIPPortfolioTracker:
    """Enhanced portfolio tracker with monthly investment limits and price thresholds"""

    def __init__(self, min_gap_days: int = 5, monthly_limit: Optional[float] = None,
                 enable_monthly_limits: bool = True, price_threshold: float = 3.0):
        # Existing fields
        self.trades: List[Trade] = []
        self.total_investment = 0.0
        self.total_units = 0.0
        self.max_drawdown = 0.0
        self.min_gap_days = min_gap_days
        self.last_investment_date: Optional[datetime] = None

        # NEW: Monthly investment tracking
        self.monthly_limit = monthly_limit
        self.enable_monthly_limits = enable_monthly_limits
        self.price_threshold = price_threshold
        self.monthly_investment_tracker = {}  # {year-month: amount_invested}

        # NEW: Additional tracking for analytics
        self.monthly_limit_exceeded_count = 0
        self.price_threshold_skipped_count = 0
        self.max_amount_in_a_month = 0.0
        self.skipped_investments: List[Dict] = []

    def get_monthly_invested(self, date: datetime) -> float:
        """Get total investment for the month"""
        month_key = date.strftime('%Y-%m')
        return self.monthly_investment_tracker.get(month_key, 0.0)

    def can_invest_within_monthly_limit(self, date: datetime, amount: float) -> bool:
        """Check if investment is within monthly limit"""
        if not self.enable_monthly_limits or not self.monthly_limit:
            return True

        current_month_investment = self.get_monthly_invested(date)
        return (current_month_investment + amount) <= self.monthly_limit

    def check_price_threshold(self, current_price: float, data: pd.DataFrame, index: int) -> bool:
        """Check if current price meets the reduction threshold"""
        if index < 20:  # Need some history for comparison
            return True

        # Calculate recent average price (last 20 days)
        recent_prices = data.iloc[max(0, index - 20):index]['close']
        if len(recent_prices) == 0:
            return True

        avg_recent_price = recent_prices.mean()
        price_reduction = ((avg_recent_price - current_price) / avg_recent_price) * 100

        return price_reduction >= self.price_threshold

    def execute_investment(self, price: float, amount: float, timestamp: datetime,
                           drawdown: Optional[float] = None, symbol: Optional[str] = None,
                           force: bool = False, data: Optional[pd.DataFrame] = None,
                           data_index: Optional[int] = None) -> Optional[Trade]:
        """Execute investment with gap enforcement, monthly limits, and price thresholds"""

        # Check minimum gap unless forced
        if not force and not self.can_invest(timestamp):
            logger.debug(f"Investment skipped due to minimum gap requirement")
            return None

        # Check monthly limit unless forced
        if not force and not self.can_invest_within_monthly_limit(timestamp, amount):
            monthly_invested = self.get_monthly_invested(timestamp)
            logger.debug(f"Investment skipped: Monthly limit would be exceeded. "
                         f"Current: ₹{monthly_invested:,.2f}, Limit: ₹{self.monthly_limit:,.2f}")
            self.monthly_limit_exceeded_count += 1
            self._record_skipped_investment(timestamp, amount, price, "Monthly limit exceeded")
            return None

        # Check price threshold unless forced
        if (not force and data is not None and data_index is not None and
                not self.check_price_threshold(price, data, data_index)):
            logger.debug(f"Investment skipped: Price threshold not met. "
                         f"Current price: ₹{price:.2f}, Threshold: {self.price_threshold}%")
            self.price_threshold_skipped_count += 1
            self._record_skipped_investment(timestamp, amount, price, "Price threshold not met")
            return None

        # Execute investment
        units = amount / price
        self.total_investment += amount
        self.total_units += units

        # Update monthly tracking
        month_key = timestamp.strftime('%Y-%m')
        current_monthly = self.monthly_investment_tracker.get(month_key, 0)
        new_monthly_total = current_monthly + amount
        self.monthly_investment_tracker[month_key] = new_monthly_total

        # Update max monthly amount
        if new_monthly_total > self.max_amount_in_a_month:
            self.max_amount_in_a_month = new_monthly_total

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

    def _record_skipped_investment(self, timestamp: datetime, amount: float,
                                   price: float, reason: str):
        """Record a skipped investment for analytics"""
        skip_record = {
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'intended_amount': amount,
            'price': price,
            'reason': reason,
            'month': timestamp.strftime('%Y-%m')
        }
        self.skipped_investments.append(skip_record)

    def can_invest(self, current_date: datetime) -> bool:
        """Check if investment is allowed based on minimum gap"""
        if not self.last_investment_date:
            return True

        days_since_last = (current_date.date() - self.last_investment_date.date()).days
        return days_since_last >= self.min_gap_days

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

    def get_monthly_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get monthly investment summary"""
        summary = {}
        for month, amount in self.monthly_investment_tracker.items():
            trades_in_month = [t for t in self.trades if t.timestamp.strftime('%Y-%m') == month]
            summary[month] = {
                'total_invested': amount,
                'number_of_investments': len(trades_in_month),
                'average_investment': amount / len(trades_in_month) if trades_in_month else 0,
                'utilization_percent': (amount / self.monthly_limit * 100) if self.monthly_limit else 100
            }
        return summary


class EnhancedSIPStrategy:
    """Base SIP strategy for compatibility"""

    def __init__(self, nsedata_session: AsyncSession = None, trading_session: AsyncSession = None):
        self.nsedata_session = nsedata_session
        self.trading_session = trading_session

    async def fetch_data_from_db_async(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """FIXED: Fetch data using proper table name handling and error recovery"""
        try:
            if not self.nsedata_session:
                logger.warning("No nsedata session provided, falling back to sync method")
                return await self._fetch_data_sync_fallback(symbol, start_date, end_date)

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
                return await self._fetch_data_sync_fallback(symbol, start_date, end_date)

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
                return await self._fetch_data_sync_fallback(symbol, start_date, end_date)

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
            return await self._fetch_data_sync_fallback(symbol, start_date, end_date)

    async def _fetch_data_sync_fallback(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """FIXED: Fallback to synchronous data fetching with proper error handling"""
        try:
            logger.info(f"Using sync fallback for {symbol}")

            # Try to get database URL from environment or use default
            nsedata_url = os.getenv("NSEDATA_URL")
            if not nsedata_url:
                # Try to construct from trading DB URL
                trading_url = os.getenv("DATABASE_URL",
                                        "postgresql://trading_user:password123@localhost:5432/trading_db")
                nsedata_url = trading_url.replace("/trading_db", "/nsedata")

            # Convert async URL to sync URL if needed
            if "asyncpg" in nsedata_url:
                nsedata_url = nsedata_url.replace("postgresql+asyncpg://", "postgresql://")

            def fetch_sync():
                try:
                    # Create synchronous engine
                    engine = create_engine(nsedata_url)

                    # Check if table exists first
                    with engine.connect() as conn:
                        table_exists_query = text("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_schema = 'public' 
                                AND table_name = :table_name
                            )
                        """)

                        exists_result = conn.execute(table_exists_query, {"table_name": symbol})
                        table_exists = exists_result.scalar()

                        if not table_exists:
                            logger.warning(f"Table {symbol} does not exist in sync database either")
                            return pd.DataFrame()

                        # Fetch data with proper quoting
                        query = text(f"""
                            SELECT timestamp, open, high, low, close, volume 
                            FROM "{symbol}"
                            WHERE timestamp BETWEEN :start_date AND :end_date
                            ORDER BY timestamp ASC
                        """)

                        df = pd.read_sql(
                            query,
                            conn,
                            params={
                                'start_date': start_date,
                                'end_date': end_date
                            }
                        )

                        if not df.empty:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])

                            # Ensure numeric columns
                            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                            for col in numeric_columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')

                            # Remove any rows with NaN values
                            df = df.dropna()

                            logger.info(f"✅ Sync fallback successful: {len(df)} records for {symbol}")
                            return df.sort_values('timestamp').reset_index(drop=True)

                        return pd.DataFrame()

                except Exception as e:
                    logger.error(f"Sync fallback failed for {symbol}: {e}")
                    return pd.DataFrame()

            # Run sync operation in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, fetch_sync)

        except Exception as e:
            logger.error(f"Fallback method completely failed: {e}")
            return pd.DataFrame()

    async def run_backtest(self, symbol: str, start_date: str, end_date: str,
                           config: SIPConfig) -> Optional[SIPResults]:
        """Run enhanced SIP backtest with monthly limits and price thresholds"""
        try:
            logger.info(f"🚀 Starting enhanced SIP backtest for {symbol} from {start_date} to {end_date}")

            # Fetch data
            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)

            # Initialize enhanced portfolio tracker with monthly limits
            portfolio = SIPPortfolioTracker(
                min_gap_days=config.min_investment_gap_days,
                monthly_limit=config.max_amount_in_a_month,
                enable_monthly_limits=config.enable_monthly_limits,
                price_threshold=config.price_reduction_threshold
            )

            # Track investment opportunities and skipped investments
            total_opportunities = 0
            skipped_due_to_gap = 0
            skipped_due_to_monthly_limit = 0
            skipped_due_to_price_threshold = 0

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
                        drawdown_100 <= config.drawdown_threshold_2  # -4% or worse
                )

                should_invest = is_sip_date or is_drawdown_opportunity

                if should_invest:
                    total_opportunities += 1
                    current_price = row['close']
                    investment_amount = self.determine_investment_amount(
                        current_price, data, config, i
                    )

                    # Attempt to execute investment (all checks are done inside execute_investment)
                    trade = portfolio.execute_investment(
                        price=current_price,
                        amount=investment_amount,
                        timestamp=current_date,
                        drawdown=drawdown_100,
                        symbol=symbol,
                        data=data,
                        data_index=i
                    )

                    if trade:
                        logger.debug(
                            f"Investment: ₹{trade.amount:,.2f} at ₹{trade.price:.2f} on {trade.timestamp.date()}")
                    else:
                        # Investment was skipped - determine reason
                        if not portfolio.can_invest(current_date):
                            skipped_due_to_gap += 1
                        elif not portfolio.can_invest_within_monthly_limit(current_date, investment_amount):
                            skipped_due_to_monthly_limit += 1
                        elif not portfolio.check_price_threshold(current_price, data, i):
                            skipped_due_to_price_threshold += 1

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
                sharpe_ratio = (cagr - 0.05) / volatility if volatility > 0 else 0
            else:
                volatility = None
                sharpe_ratio = None

            # Get monthly summary
            monthly_summary = portfolio.get_monthly_summary()
            avg_monthly_investment = sum(portfolio.monthly_investment_tracker.values()) / len(
                portfolio.monthly_investment_tracker) if portfolio.monthly_investment_tracker else 0
            months_at_limit = sum(1 for summary in monthly_summary.values() if
                                  summary['utilization_percent'] >= 95) if config.enable_monthly_limits else 0

            results = SIPResults(
                strategy_name="Enhanced SIP Strategy",
                total_investment=portfolio.total_investment,
                final_portfolio_value=final_portfolio_value,
                total_units=portfolio.total_units,
                average_buy_price=portfolio.get_average_buy_price(),
                cagr=cagr * 100,
                trades=portfolio.trades,
                max_drawdown=portfolio.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                start_date=start_date,
                end_date=end_date,
                symbol=symbol,
                # NEW: Monthly tracking results
                total_return_percent=((final_portfolio_value / portfolio.total_investment) - 1) * 100,
                num_trades=len(portfolio.trades),
                num_skipped=len(portfolio.skipped_investments),
                monthly_investment_summary=monthly_summary,
                monthly_limit_exceeded=portfolio.monthly_limit_exceeded_count,
                price_threshold_skipped=portfolio.price_threshold_skipped_count,
                max_amount_in_a_month=portfolio.max_amount_in_a_month,
                average_monthly_investment=avg_monthly_investment,
                price_reduction_threshold=config.price_reduction_threshold,
                months_at_limit=months_at_limit,
                skipped_investments=portfolio.skipped_investments,
                config_used={
                'fixed_investment': config.fixed_investment,
                'max_amount_in_a_month': config.max_amount_in_a_month,
                'price_reduction_threshold': config.price_reduction_threshold,
                'drawdown_threshold_1': config.drawdown_threshold_1,
                'drawdown_threshold_2': config.drawdown_threshold_2,
                'investment_multiplier_1': config.investment_multiplier_1,
                'investment_multiplier_2': config.investment_multiplier_2,
                'investment_multiplier_3': config.investment_multiplier_3,
                'rolling_window': config.rolling_window,
                'fallback_day': config.fallback_day,
                'min_investment_gap_days': config.min_investment_gap_days,
                'enable_monthly_limits': config.enable_monthly_limits
            }
            )

            logger.info(f"✅ Enhanced backtest completed for {symbol}:")
            logger.info(f"   📊 Investment: ₹{portfolio.total_investment:,.2f}")
            logger.info(f"   💰 Final Value: ₹{final_portfolio_value:,.2f}")
            logger.info(f"   📈 CAGR: {cagr :.2f}%")
            logger.info(f"   🔄 Total Trades: {len(portfolio.trades)}")
            logger.info(
                f"   ⏭️  Opportunities: {total_opportunities}, Skipped - Gap: {skipped_due_to_gap}, Monthly: {skipped_due_to_monthly_limit}, Price: {skipped_due_to_price_threshold}")
            logger.info(f"   💸 Monthly limit exceeded: {portfolio.monthly_limit_exceeded_count} times")

            return results

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

                # Use the enhanced strategy with limits
                enhanced_strategy = EnhancedSIPStrategyWithLimits(
                    nsedata_session=self.nsedata_session,
                    trading_session=self.trading_session
                )

                # Convert SIPConfig to request-like object
                config_request = type('obj', (object,), {
                    'fixed_investment': config.fixed_investment,
                    'max_amount_in_a_month': config.max_amount_in_a_month or (config.fixed_investment * 4),
                    'price_reduction_threshold': config.price_reduction_threshold,
                    'drawdown_threshold_1': config.drawdown_threshold_1,
                    'drawdown_threshold_2': config.drawdown_threshold_2,
                    'investment_multiplier_1': config.investment_multiplier_1,
                    'investment_multiplier_2': config.investment_multiplier_2,
                    'investment_multiplier_3': config.investment_multiplier_3,
                    'rolling_window': config.rolling_window,
                    'fallback_day': config.fallback_day,
                    'min_investment_gap_days': config.min_investment_gap_days,
                    'enable_monthly_limits': config.enable_monthly_limits,
                    'reset_on_month_change': config.reset_on_month_change
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

    def get_next_investment_signals(self, data: pd.DataFrame, config: SIPConfig,
                                    current_monthly_invested: float = 0) -> Dict[str, Any]:
        """Generate enhanced investment signals with monthly limit considerations"""
        try:
            if data.empty:
                return {
                    "signal": "NO_DATA",
                    "confidence": 0,
                    "message": "No data available"
                }

            logger.debug(f"Generating signals for {len(data)} data points")

            # Calculate indicators
            try:
                data_with_indicators = self.calculate_technical_indicators(data)
                if data_with_indicators.empty:
                    raise ValueError("Technical indicators calculation returned empty DataFrame")
            except Exception as indicator_error:
                logger.error(f"Technical indicators calculation failed: {indicator_error}")
                return {
                    "signal": "ERROR",
                    "confidence": 0,
                    "message": f"Technical indicators calculation failed: {str(indicator_error)}"
                }

            # Get latest data point
            try:
                latest = data_with_indicators.iloc[-1]
                current_price = float(latest['close'])
            except Exception as latest_error:
                logger.error(f"Error getting latest data point: {latest_error}")
                return {
                    "signal": "ERROR",
                    "confidence": 0,
                    "message": f"Error accessing latest data: {str(latest_error)}"
                }

            # Extract indicators safely with defaults
            try:
                drawdown_100 = float(latest.get('Drawdown_100', 0))
                drawdown_50 = float(latest.get('Drawdown_50', 0))
                rsi = float(latest.get('RSI', 50))
                volatility = float(latest.get('Volatility_20', 0))
                bb_position = float(latest.get('BB_Position', 0.5))
                macd_histogram = float(latest.get('MACD_Histogram', 0))
                williams_r = float(latest.get('Williams_R', -50))
                cci = float(latest.get('CCI', 0))

                logger.debug(
                    f"Extracted indicators - RSI: {rsi}, Drawdown_100: {drawdown_100}, BB_Position: {bb_position}")

            except Exception as extract_error:
                logger.error(f"Error extracting indicators: {extract_error}")
                return {
                    "signal": "ERROR",
                    "confidence": 0,
                    "message": f"Error extracting technical indicators: {str(extract_error)}"
                }

            # Determine signal strength and type
            signal_type = "NORMAL"
            confidence = 0.5
            recommended_multiplier = 1.0
            recommended_amount = config.fixed_investment

            # Analyze market conditions
            market_conditions = []

            try:
                # Drawdown analysis
                if drawdown_100 <= config.drawdown_threshold_1:  # Severe drawdown
                    signal_type = "STRONG_BUY"
                    confidence = 0.9
                    recommended_multiplier = config.investment_multiplier_3
                    market_conditions.append(f"Severe drawdown: {drawdown_100:.2f}%")
                elif drawdown_100 <= config.drawdown_threshold_2:  # Moderate drawdown
                    signal_type = "BUY"
                    confidence = 0.8
                    recommended_multiplier = config.investment_multiplier_2
                    market_conditions.append(f"Moderate drawdown: {drawdown_100:.2f}%")
                elif drawdown_100 <= -1.0:  # Minor drawdown
                    signal_type = "WEAK_BUY"
                    confidence = 0.7
                    recommended_multiplier = config.investment_multiplier_1
                    market_conditions.append(f"Minor drawdown: {drawdown_100:.2f}%")

                # Calculate recommended amount
                recommended_amount = config.fixed_investment * recommended_multiplier

                # NEW: Monthly limit validation
                if config.enable_monthly_limits and config.max_amount_in_a_month:
                    remaining_monthly_budget = config.max_amount_in_a_month - current_monthly_invested

                    if remaining_monthly_budget <= 0:
                        signal_type = "MONTHLY_LIMIT_REACHED"
                        recommended_amount = 0
                        confidence = 0
                        market_conditions.append("Monthly investment limit reached")
                    elif recommended_amount > remaining_monthly_budget:
                        original_amount = recommended_amount
                        recommended_amount = remaining_monthly_budget
                        market_conditions.append(
                            f"Investment capped by monthly limit: ₹{remaining_monthly_budget:,.2f} remaining (was ₹{original_amount:,.2f})")
                        confidence *= 0.8  # Reduce confidence when capped
                    else:
                        utilization = ((
                                                   current_monthly_invested + recommended_amount) / config.max_amount_in_a_month) * 100
                        market_conditions.append(f"Monthly limit utilization: {utilization:.1f}%")

                # Technical momentum analysis (existing logic)
                momentum_score = 0

                # RSI analysis
                if rsi < 30:
                    momentum_score += 1
                    market_conditions.append(f"Oversold conditions (RSI: {rsi:.1f})")
                elif rsi > 70:
                    momentum_score -= 1
                    market_conditions.append(f"Overbought conditions (RSI: {rsi:.1f})")

                # Williams %R analysis
                if williams_r < -80:
                    momentum_score += 0.5
                elif williams_r > -20:
                    momentum_score -= 0.5

                # MACD analysis
                if macd_histogram > 0:
                    momentum_score += 0.5
                    market_conditions.append("Positive momentum trend")
                else:
                    momentum_score -= 0.5
                    market_conditions.append("Negative momentum - potential reversal")

                # Adjust signal based on momentum
                if momentum_score >= 1.5 and signal_type in ["NORMAL", "WEAK_BUY"]:
                    if signal_type == "NORMAL":
                        signal_type = "WEAK_BUY"
                        confidence = max(confidence, 0.6)
                    elif signal_type == "WEAK_BUY":
                        signal_type = "BUY"
                        confidence = max(confidence, 0.75)

                # Cap confidence
                confidence = min(confidence, 1.0)

                # Generate next fallback date
                next_fallback_date = (datetime.now() + timedelta(days=30)).replace(day=config.fallback_day).strftime(
                    '%Y-%m-%d')

                # Determine trade type description
                if signal_type == "STRONG_BUY":
                    trade_type = "Aggressive Dip Purchase"
                    message = "Excellent opportunity. Strong buy signal detected."
                elif signal_type == "BUY":
                    trade_type = "Strategic Dip Purchase"
                    message = "Good opportunity. Consider increased investment."
                elif signal_type == "WEAK_BUY":
                    trade_type = "Minor Dip Purchase"
                    message = "Moderate opportunity. Consider slightly increased investment."
                elif signal_type == "MONTHLY_LIMIT_REACHED":
                    trade_type = "Limit Reached"
                    message = "Monthly investment limit reached. Wait for next month."
                else:
                    trade_type = "Regular SIP"
                    message = "Normal market conditions. Continue regular SIP."

                return {
                    "signal": signal_type,
                    "confidence": confidence,
                    "current_price": current_price,
                    "recommended_amount": recommended_amount,
                    "investment_multiplier": recommended_multiplier,
                    "drawdown_100": drawdown_100,
                    "drawdown_50": drawdown_50,
                    "rsi": rsi,
                    "williams_r": williams_r,
                    "cci": cci,
                    "volatility": volatility,
                    "bb_position": bb_position,
                    "macd_histogram": macd_histogram,
                    "market_conditions": market_conditions,
                    "next_fallback_date": next_fallback_date,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "trade_type": trade_type,
                    "message": message,
                    # NEW: Monthly tracking info
                    "monthly_info": {
                        "current_monthly_invested": current_monthly_invested,
                        "monthly_limit": config.max_amount_in_a_month,
                        "remaining_budget": config.max_amount_in_a_month - current_monthly_invested if config.max_amount_in_a_month else None,
                        "utilization_percent": (
                                    current_monthly_invested / config.max_amount_in_a_month * 100) if config.max_amount_in_a_month else None
                    }
                }

            except Exception as analysis_error:
                logger.error(f"Error in signal analysis: {analysis_error}")
                return {
                    "signal": "ERROR",
                    "confidence": 0,
                    "message": f"Signal analysis failed: {str(analysis_error)}",
                    "current_price": current_price,
                    "analysis_timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Critical error in get_next_investment_signals: {e}")
            return {
                "signal": "ERROR",
                "confidence": 0,
                "message": f"Signal generation failed: {str(e)}",
                "analysis_timestamp": datetime.now().isoformat()
            }

    def _calculate_next_fallback_date(self, current_date: datetime.date, fallback_day: int) -> datetime.date:
        """Calculate next fallback investment date"""
        try:
            # If current day is before fallback day this month, use this month
            if current_date.day < fallback_day:
                try:
                    return current_date.replace(day=fallback_day)
                except ValueError:
                    # Handle months with fewer days
                    last_day = (current_date.replace(month=current_date.month % 12 + 1, day=1) - timedelta(days=1)).day
                    return current_date.replace(day=min(fallback_day, last_day))
            else:
                # Use next month
                next_month = current_date.replace(day=1) + timedelta(days=32)
                next_month = next_month.replace(day=1)
                try:
                    return next_month.replace(day=fallback_day)
                except ValueError:
                    # Handle months with fewer days
                    last_day = (next_month.replace(month=next_month.month % 12 + 1, day=1) - timedelta(days=1)).day
                    return next_month.replace(day=min(fallback_day, last_day))
        except Exception as e:
            logger.error(f"Error calculating next fallback date: {e}")
            return current_date + timedelta(days=30)

    def _determine_trade_type(self, signal_type: str, drawdown: float) -> str:
        """Determine trade type based on signal and conditions"""
        if signal_type == "STRONG_BUY":
            if drawdown <= -15:
                return "Crisis Opportunity"
            elif drawdown <= -10:
                return "Major Dip Purchase"
            else:
                return "Strong Buy Signal"
        elif signal_type == "BUY":
            return "Drawdown Opportunity"
        elif signal_type == "WEAK_BUY":
            return "Minor Dip Purchase"
        elif signal_type == "AVOID":
            return "Market Peak - Avoid"
        else:
            return "Regular SIP Investment"

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators with proper error handling and column naming"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame passed to calculate_technical_indicators")
                return df

            # Create a copy to avoid modifying original data
            data = df.copy()

            logger.debug(f"Calculating technical indicators for {len(data)} data points")

            try:
                # Simple Moving Averages
                data['SMA_20'] = data['close'].rolling(window=20, min_periods=1).mean()
                data['SMA_50'] = data['close'].rolling(window=50, min_periods=1).mean()
                data['SMA_200'] = data['close'].rolling(window=200, min_periods=1).mean()
                logger.debug("✅ SMAs calculated")

            except Exception as sma_error:
                logger.error(f"Error calculating SMAs: {sma_error}")
                data['SMA_20'] = data['close']
                data['SMA_50'] = data['close']
                data['SMA_200'] = data['close']

            try:
                # Exponential Moving Averages
                data['EMA_12'] = data['close'].ewm(span=12, min_periods=1).mean()
                data['EMA_26'] = data['close'].ewm(span=26, min_periods=1).mean()
                logger.debug("✅ EMAs calculated")

            except Exception as ema_error:
                logger.error(f"Error calculating EMAs: {ema_error}")
                data['EMA_12'] = data['close']
                data['EMA_26'] = data['close']

            try:
                # Drawdown calculations (rolling high-based)
                for window in [20, 50, 100]:
                    rolling_high = data['close'].rolling(window=window, min_periods=1).max()
                    drawdown = ((data['close'] - rolling_high) / rolling_high) * 100
                    data[f'Drawdown_{window}'] = drawdown

                logger.debug("✅ Drawdowns calculated")

            except Exception as dd_error:
                logger.error(f"Error calculating drawdowns: {dd_error}")
                data['Drawdown_20'] = 0
                data['Drawdown_50'] = 0
                data['Drawdown_100'] = 0

            try:
                # Volatility (20-day rolling standard deviation of returns)
                returns = data['close'].pct_change()
                data['Volatility_20'] = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100
                logger.debug("✅ Volatility calculated")

            except Exception as vol_error:
                logger.error(f"Error calculating volatility: {vol_error}")
                data['Volatility_20'] = 0

            try:
                # RSI (Relative Strength Index)
                def calculate_rsi(prices, window=14):
                    if len(prices) < window:
                        return pd.Series([50] * len(prices), index=prices.index)

                    delta = prices.diff()
                    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

                    # Handle division by zero
                    rs = gain / loss.replace(0, np.nan)
                    rsi = 100 - (100 / (1 + rs))

                    return rsi.fillna(50)

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
                # Commodity Channel Index (CCI)
                def calculate_cci(high, low, close, window=20):
                    if len(close) < window:
                        return pd.Series([0] * len(close), index=close.index)

                    # Typical Price
                    tp = (high + low + close) / 3

                    # Simple Moving Average of Typical Price
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
                                    config: SIPConfig, index: int) -> float:
        """Enhanced investment amount determination with price threshold consideration"""
        try:
            base_amount = config.fixed_investment

            if index >= len(data):
                return base_amount

            row = data.iloc[index]

            # Get technical indicators
            drawdown_100 = row.get('Drawdown_100', 0)
            drawdown_50 = row.get('Drawdown_50', 0)
            rsi = row.get('RSI', 50)
            volatility = row.get('Volatility_20', 0)
            bb_position = row.get('BB_Position', 0.5)
            macd_histogram = row.get('MACD_Histogram', 0)
            williams_r = row.get('Williams_R', -50)
            cci = row.get('CCI', 0)

            investment_amount = base_amount

            # Check price reduction threshold first
            if index >= 20:  # Need some history
                recent_prices = data.iloc[max(0, index - 20):index]['close']
                if len(recent_prices) > 0:
                    avg_recent_price = recent_prices.mean()
                    price_reduction = ((avg_recent_price - current_price) / avg_recent_price) * 100

                    if price_reduction < config.price_reduction_threshold:
                        # Price hasn't reduced enough - reduce investment amount
                        investment_amount *= 0.5
                        logger.info(
                            f"Price reduction insufficient ({price_reduction:.2f}% < {config.price_reduction_threshold}%), reducing investment")

            # Primary drawdown-based adjustments
            if drawdown_100 <= config.drawdown_threshold_1:  # Severe drawdown (< -10%)
                investment_amount *= config.investment_multiplier_3
                logger.info(f"Severe drawdown detected ({drawdown_100:.2f}%), increasing to {investment_amount}")
            elif drawdown_100 <= config.drawdown_threshold_2:  # Moderate drawdown (-4% to -10%)
                investment_amount *= config.investment_multiplier_2
                logger.info(f"Moderate drawdown detected ({drawdown_100:.2f}%), increasing to {investment_amount}")
            elif drawdown_50 <= -2:  # Minor drawdown
                investment_amount *= config.investment_multiplier_1
                logger.info(f"Minor drawdown detected ({drawdown_50:.2f}%), increasing to {investment_amount}")

            # RSI-based adjustments (oversold/overbought conditions)
            if 0 < rsi < 25:  # Extremely oversold - major opportunity
                investment_amount *= 1.5
                logger.info(
                    f"Extremely oversold RSI ({rsi:.2f}), increasing investment by 50% to {investment_amount}")
            elif 0 < rsi < 30:  # Oversold
                investment_amount *= 1.2
                logger.info(f"Oversold RSI ({rsi:.2f}), increasing investment by 20% to {investment_amount}")
            elif rsi > 75:  # Extremely overbought - reduce investment
                investment_amount *= 0.6
                logger.info(
                    f"Extremely overbought RSI ({rsi:.2f}), reducing investment by 40% to {investment_amount}")
            elif rsi > 70:  # Overbought - reduce investment
                investment_amount *= 0.8
                logger.info(f"Overbought RSI ({rsi:.2f}), reducing investment by 20% to {investment_amount}")

            # Williams %R adjustments
            if williams_r < -80:  # Oversold
                investment_amount *= 1.1
                logger.info(
                    f"Oversold Williams %R ({williams_r:.2f}), increasing investment by 10% to {investment_amount}")
            elif williams_r > -20:  # Overbought
                investment_amount *= 0.9
                logger.info(
                    f"Overbought Williams %R ({williams_r:.2f}), reducing investment by 10% to {investment_amount}")

            # CCI adjustments
            if cci < -100:  # Oversold
                investment_amount *= 1.1
                logger.info(f"Oversold CCI ({cci:.2f}), increasing investment by 10% to {investment_amount}")
            elif cci > 100:  # Overbought
                investment_amount *= 0.9
                logger.info(f"Overbought CCI ({cci:.2f}), reducing investment by 10% to {investment_amount}")

            # Bollinger Bands position adjustments
            if bb_position < 0.1:  # Near lower band - opportunity
                investment_amount *= 1.15
                logger.info(f"Near Bollinger lower band, increasing investment by 15% to {investment_amount}")
            elif bb_position > 0.9:  # Near upper band - caution
                investment_amount *= 0.85
                logger.info(f"Near Bollinger upper band, reducing investment by 15% to {investment_amount}")

            # MACD momentum adjustments
            if macd_histogram > 0:  # Positive momentum
                investment_amount *= 1.05  # Slight increase
                logger.info(
                    f"Positive MACD momentum ({macd_histogram:.2f}), increasing investment by 5% to {investment_amount}")
            elif macd_histogram < -0.5:  # Strong negative momentum - opportunity
                investment_amount *= 1.1
                logger.info(
                    f"Strong negative MACD momentum ({macd_histogram:.2f}), increasing investment by 10% to {investment_amount}")

            # Volatility-based adjustments
            if volatility > 0.35:  # High volatility - opportunity but with caution
                investment_amount *= 1.1
                logger.info(
                    f"High volatility ({volatility:.2f}), increasing investment by 10% to {investment_amount}")
            elif volatility < 0.15:  # Low volatility - normal conditions
                investment_amount *= 0.95
                logger.info(f"Low volatility ({volatility:.2f}), reducing investment by 5% to {investment_amount}")

            # Cap the maximum multiplier based on monthly limits
            if config.max_amount_in_a_month:
                max_allowed = min(base_amount * 5, config.max_amount_in_a_month)
            else:
                max_allowed = base_amount * 5

            if investment_amount > max_allowed:
                investment_amount = max_allowed
                logger.warning(f"Capping investment at maximum allowed: {max_allowed}")

            # Ensure minimum investment
            min_allowed = base_amount * 0.5  # Minimum 10% of base amount
            if investment_amount < min_allowed:
                investment_amount = min_allowed

            return round(investment_amount, 2)

        except Exception as e:
            logger.error(f"Error determining investment amount: {e}")
            return config.fixed_investment

    def validate_portfolio_allocation(self, symbols_config: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate portfolio allocation percentages"""
        errors = []

        try:
            total_allocation = sum(symbol.get('allocation_percentage', 0) for symbol in symbols_config)

            if abs(total_allocation - 100.0) > 0.01:
                errors.append(f"Total allocation is {total_allocation}%, must equal 100%")

            for symbol_config in symbols_config:
                allocation = symbol_config.get('allocation_percentage', 0)
                if allocation <= 0 or allocation > 100:
                    errors.append(f"Invalid allocation {allocation}% for {symbol_config.get('symbol', 'unknown')}")

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    def calculate_portfolio_performance(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate comprehensive portfolio performance metrics"""
        try:
            total_invested = portfolio_data.get('total_invested', 0)
            current_value = portfolio_data.get('current_value', 0)

            if total_invested > 0:
                total_return = ((current_value - total_invested) / total_invested) * 100

                # Calculate CAGR if we have creation date
                created_at = portfolio_data.get('created_at')
                if created_at:
                    if isinstance(created_at, str):
                        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        created_date = created_at

                    days_invested = (datetime.now() - created_date).days
                    years_invested = max(days_invested / 365.25, 1 / 365.25)  # Minimum 1 day

                    cagr = ((current_value / total_invested) ** (1 / years_invested) - 1) * 100
                else:
                    cagr = 0

                return {
                    'total_return_percent': total_return,
                    'cagr_percent': cagr,
                    'total_invested': total_invested,
                    'current_value': current_value,
                    'absolute_profit': current_value - total_invested
                }
            else:
                return {
                    'total_return_percent': 0,
                    'cagr_percent': 0,
                    'total_invested': 0,
                    'current_value': 0,
                    'absolute_profit': 0
                }

        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {
                'total_return_percent': 0,
                'cagr_percent': 0,
                'total_invested': total_invested,
                'current_value': current_value,
                'absolute_profit': 0,
                'error': str(e)
            }

    async def get_current_portfolio_value(self, portfolio_id: str) -> Dict[str, Any]:
        """Get current portfolio value with real-time prices"""
        try:
            # This would typically fetch real-time prices from market data
            # For now, using placeholder logic
            return {
                'portfolio_id': portfolio_id,
                'current_value': 0.0,
                'last_updated': datetime.now().isoformat(),
                'message': 'Real-time pricing not implemented'
            }

        except Exception as e:
            logger.error(f"Error getting current portfolio value: {e}")
            return {
                'portfolio_id': portfolio_id,
                'current_value': 0.0,
                'error': str(e)
            }
        """Generate enhanced investment signals with robust error handling"""
        try:
            if data.empty:
                return {
                    "signal": "NO_DATA",
                    "confidence": 0,
                    "message": "No data available"
                }

            logger.debug(f"Generating signals for {len(data)} data points")

            # Calculate indicators - this is where the previous error occurred
            try:
                data_with_indicators = self.calculate_technical_indicators(data)
                if data_with_indicators.empty:
                    raise ValueError("Technical indicators calculation returned empty DataFrame")
            except Exception as indicator_error:
                logger.error(f"Technical indicators calculation failed: {indicator_error}")
                return {
                    "signal": "ERROR",
                    "confidence": 0,
                    "message": f"Technical indicators calculation failed: {str(indicator_error)}"
                }

            # Get latest data point
            try:
                latest = data_with_indicators.iloc[-1]
                current_price = float(latest['close'])
            except Exception as latest_error:
                logger.error(f"Error getting latest data point: {latest_error}")
                return {
                    "signal": "ERROR",
                    "confidence": 0,
                    "message": f"Error accessing latest data: {str(latest_error)}"
                }

            # Generate signals based on multiple factors
            signals = []
            confidence_scores = []

            try:
                # 1. Drawdown-based signals
                drawdown_100 = latest.get('Drawdown_100', 0)
                if drawdown_100 <= config.drawdown_threshold_1:
                    signals.append("STRONG_BUY")
                    confidence_scores.append(0.9)
                elif drawdown_100 <= config.drawdown_threshold_2:
                    signals.append("BUY")
                    confidence_scores.append(0.7)

                # 2. RSI signals
                rsi = latest.get('RSI', 50)
                if rsi <= 30:
                    signals.append("BUY")
                    confidence_scores.append(0.6)
                elif rsi >= 70:
                    signals.append("AVOID")
                    confidence_scores.append(0.5)

                # 3. Bollinger Band signals
                bb_position = latest.get('BB_Position', 0.5)
                if bb_position <= 0.1:  # Near lower band
                    signals.append("BUY")
                    confidence_scores.append(0.5)
                elif bb_position >= 0.9:  # Near upper band
                    signals.append("AVOID")
                    confidence_scores.append(0.4)

                # 4. MACD signals
                macd = latest.get('MACD', 0)
                macd_signal = latest.get('MACD_Signal', 0)
                if macd > macd_signal and macd > 0:
                    signals.append("BUY")
                    confidence_scores.append(0.4)

                # Determine overall signal
                if "STRONG_BUY" in signals:
                    overall_signal = "STRONG_BUY"
                    overall_confidence = max(confidence_scores)
                elif signals.count("BUY") >= 2:
                    overall_signal = "BUY"
                    overall_confidence = sum(confidence_scores) / len(confidence_scores)
                elif "AVOID" in signals and len([s for s in signals if s in ["BUY", "STRONG_BUY"]]) == 0:
                    overall_signal = "AVOID"
                    overall_confidence = max([c for i, c in enumerate(confidence_scores) if signals[i] == "AVOID"])
                else:
                    overall_signal = "HOLD"
                    overall_confidence = 0.3

                return {
                    "signal": overall_signal,
                    "confidence": round(overall_confidence, 2),
                    "individual_signals": signals,
                    "confidence_scores": confidence_scores,
                    "current_price": current_price,
                    "drawdown_100": drawdown_100,
                    "rsi": rsi,
                    "bb_position": bb_position,
                    "recommended_amount": self.determine_investment_amount(current_price, data_with_indicators, config,
                                                                           len(data_with_indicators) - 1),
                    "message": f"{overall_signal} signal with {overall_confidence:.1%} confidence"
                }

            except Exception as signal_error:
                logger.error(f"Error generating signals: {signal_error}")
                return {
                    "signal": "HOLD",
                    "confidence": 0,
                    "message": f"Signal generation error: {str(signal_error)}"
                }

        except Exception as e:
            logger.error(f"Critical error in signal generation: {e}")
            return {
                "signal": "ERROR",
                "confidence": 0,
                "message": f"Critical signal generation error: {str(e)}"
            }

    async def generate_investment_report(self, symbols: List[str], config: SIPConfig) -> Dict[str, Any]:
        """Generate comprehensive investment report for multiple symbols - COMPLETE IMPLEMENTATION"""
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
                            "data_quality": data_quality
                        }
                        overall_metrics["error_count"] += 1
                        continue

                    overall_metrics["analyzed_symbols"] += 1

                    # Get comprehensive symbol analysis
                    try:
                        symbol_stats = await asyncio.wait_for(
                            self.get_symbol_statistics(symbol, days=365),
                            timeout=45.0
                        )

                        investment_signals = await asyncio.wait_for(
                            self.get_investment_signals(symbol, config),
                            timeout=30.0
                        )

                        support_resistance = await asyncio.wait_for(
                            self.calculate_support_resistance(symbol),
                            timeout=30.0
                        )

                        # Generate recommendations
                        recommendation = self._generate_symbol_recommendation(
                            symbol_stats, investment_signals, support_resistance
                        )

                        symbol_reports[symbol] = {
                            "status": "SUCCESS",
                            "data_quality": data_quality,
                            "statistics": symbol_stats,
                            "investment_signals": investment_signals,
                            "support_resistance": support_resistance,
                            "recommendation": recommendation
                        }

                        # Update overall metrics
                        overall_metrics["successful_analyses"] += 1

                        signal_type = recommendation.get('action', 'HOLD')
                        confidence = recommendation.get('confidence', 0)

                        if signal_type == 'STRONG_BUY':
                            overall_metrics["strong_buy_signals"] += 1
                        elif signal_type in ['BUY', 'WEAK_BUY']:
                            overall_metrics["buy_signals"] += 1
                        elif signal_type in ['AVOID', 'SELL']:
                            overall_metrics["avoid_signals"] += 1

                        if confidence > 0:
                            total_confidence += confidence
                            successful_recommendations += 1

                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout during analysis for {symbol}")
                        symbol_reports[symbol] = {
                            "status": "TIMEOUT",
                            "message": "Analysis timeout - symbol requires manual review",
                            "data_quality": data_quality
                        }
                        overall_metrics["error_count"] += 1

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
                "status": "ERROR",
                "message": f"Report generation failed: {str(e)}",
                "report_generated": datetime.now().isoformat(),
                "error": str(e)
            }

    def generate_monthly_investment_report(self, results: SIPResults) -> Dict[str, Any]:
        """Generate monthly investment breakdown and analysis"""
        try:
            if not results.monthly_investment_summary:
                return {
                    "status": "NO_DATA",
                    "message": "No monthly investment data available"
                }

            monthly_data = []
            total_months = len(results.monthly_investment_summary)
            total_invested = sum(results.monthly_investment_summary.values())

            for month, amount in sorted(results.monthly_investment_summary.items()):
                month_data = {
                    "month": month,
                    "amount_invested": amount,
                    "number_of_investments": 0,  # Will be calculated from trades
                    "average_investment_per_trade": 0
                }

                # Add percentage of limit if monthly limits are enabled
                if hasattr(results, 'max_amount_in_a_month') and results.max_amount_in_a_month:
                    month_data["percentage_of_limit"] = round((amount / results.max_amount_in_a_month * 100), 2)
                    month_data["remaining_limit"] = results.max_amount_in_a_month - amount

                # Count trades in this month
                month_trades = [t for t in results.trades if t.timestamp.strftime('%Y-%m') == month]
                month_data["number_of_investments"] = len(month_trades)
                month_data["average_investment_per_trade"] = round(amount / len(month_trades), 2) if month_trades else 0

                monthly_data.append(month_data)

            # Calculate statistics
            amounts = list(results.monthly_investment_summary.values())
            avg_monthly = sum(amounts) / len(amounts) if amounts else 0
            max_monthly = max(amounts) if amounts else 0
            min_monthly = min(amounts) if amounts else 0

            # Count months at limit (95% or more of limit)
            months_at_limit = 0
            if hasattr(results, 'max_amount_in_a_month') and results.max_amount_in_a_month:
                months_at_limit = sum(1 for amount in amounts if amount >= results.max_amount_in_a_month * 0.95)

            return {
                "status": "SUCCESS",
                "monthly_breakdown": monthly_data,
                "summary_statistics": {
                    "total_months": total_months,
                    "total_invested": round(total_invested, 2),
                    "average_monthly_investment": round(avg_monthly, 2),
                    "max_monthly_investment": round(max_monthly, 2),
                    "min_monthly_investment": round(min_monthly, 2),
                    "months_at_limit": months_at_limit,
                    "limit_utilization_rate": round((months_at_limit / total_months * 100),
                                                    2) if total_months > 0 else 0
                },
                "insights": self._generate_monthly_insights(monthly_data, results),
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating monthly investment report: {e}")
            return {
                "status": "ERROR",
                "message": f"Failed to generate monthly report: {str(e)}",
                "error": str(e)
            }

    def _generate_monthly_insights(self, monthly_data: List[Dict], results: SIPResults) -> List[str]:
        """Generate insights from monthly investment patterns"""
        insights = []

        try:
            if not monthly_data:
                return ["No monthly data available for analysis"]

            amounts = [month['amount_invested'] for month in monthly_data]

            # Consistency analysis
            if len(set(amounts)) == 1:
                insights.append("💰 Consistent monthly investment pattern maintained")
            else:
                std_dev = pd.Series(amounts).std()
                mean_amount = pd.Series(amounts).mean()
                cv = (std_dev / mean_amount) * 100 if mean_amount > 0 else 0

                if cv < 20:
                    insights.append("📊 Relatively consistent investment amounts across months")
                elif cv > 50:
                    insights.append("⚠️ High variability in monthly investment amounts")

            # Utilization analysis
            if hasattr(results, 'max_amount_in_a_month') and results.max_amount_in_a_month:
                avg_utilization = sum(month.get('percentage_of_limit', 0) for month in monthly_data) / len(monthly_data)

                if avg_utilization > 90:
                    insights.append("🎯 Excellent monthly limit utilization - maximizing investment potential")
                elif avg_utilization > 70:
                    insights.append("✅ Good monthly limit utilization - room for increased investment")
                elif avg_utilization < 50:
                    insights.append("💡 Low monthly limit utilization - consider increasing investment frequency")

            # Frequency analysis
            total_investments = sum(month['number_of_investments'] for month in monthly_data)
            avg_investments_per_month = total_investments / len(monthly_data)

            if avg_investments_per_month > 4:
                insights.append("🔄 High investment frequency - good market timing discipline")
            elif avg_investments_per_month < 2:
                insights.append("📅 Low investment frequency - consider more regular investments")

            # Trend analysis
            if len(monthly_data) >= 3:
                recent_amounts = amounts[-3:]
                earlier_amounts = amounts[:-3] if len(amounts) > 3 else amounts[:1]

                if recent_amounts and earlier_amounts:
                    recent_avg = sum(recent_amounts) / len(recent_amounts)
                    earlier_avg = sum(earlier_amounts) / len(earlier_amounts)

                    change_pct = ((recent_avg - earlier_avg) / earlier_avg) * 100 if earlier_avg > 0 else 0

                    if change_pct > 20:
                        insights.append("📈 Investment amounts increasing over time - scaling up strategy")
                    elif change_pct < -20:
                        insights.append("📉 Investment amounts decreasing over time - may need strategy review")

            # Performance correlation
            if hasattr(results, 'cagr') and results.cagr:
                if results.cagr > 0.15:  # 15% CAGR
                    insights.append("🏆 Strong performance with current investment pattern")
                elif results.cagr > 0.08:  # 8% CAGR
                    insights.append("📊 Decent performance - consider optimizing investment timing")

            return insights[:5]  # Limit to 5 key insights

        except Exception as e:
            logger.error(f"Error generating monthly insights: {e}")
            return [f"Error analyzing monthly patterns: {str(e)}"]

    async def validate_symbol_data_quality(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Validate data quality for a symbol"""
        try:
            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                return {
                    "symbol": symbol,
                    "status": "NO_DATA",
                    "message": "No data available for the specified period",
                    "data_points": 0,
                    "coverage_percent": 0
                }

            # Calculate expected vs actual data points
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            expected_trading_days = (end_dt - start_dt).days * 0.7  # Approximate trading days

            actual_data_points = len(data)
            coverage_percent = (actual_data_points / expected_trading_days) * 100 if expected_trading_days > 0 else 0

            # Check for data gaps
            data_gaps = []
            for i in range(1, min(len(data), 100)):  # Check first 100 rows
                prev_date = data.iloc[i - 1]['timestamp']
                curr_date = data.iloc[i]['timestamp']
                gap_days = (curr_date - prev_date).days
                if gap_days > 3:  # More than 3 days gap
                    data_gaps.append({
                        "from": prev_date.strftime('%Y-%m-%d'),
                        "to": curr_date.strftime('%Y-%m-%d'),
                        "gap_days": gap_days
                    })

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
                "data_gaps": data_gaps[:5],  # Limit to first 5 gaps
                "price_range": {
                    "min": float(data['close'].min()),
                    "max": float(data['close'].max()),
                    "avg": float(data['close'].mean())
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

    async def get_symbol_statistics(self, symbol: str, days: int = 365) -> Dict[str, Any]:
        """Get comprehensive statistics for a symbol"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                return {
                    "symbol": symbol,
                    "status": "NO_DATA",
                    "message": "No data available"
                }

            # Calculate basic statistics
            returns = data['close'].pct_change().dropna()

            stats = {
                "symbol": symbol,
                "status": "SUCCESS",
                "price_statistics": {
                    "current_price": float(data['close'].iloc[-1]),
                    "period_high": float(data['close'].max()),
                    "period_low": float(data['close'].min()),
                    "period_return": float(((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100),
                    "volatility_annualized": float(returns.std() * np.sqrt(252) * 100),
                    "avg_volume": int(data['volume'].mean()) if 'volume' in data.columns else 0
                },
                "risk_metrics": {
                    "max_drawdown": self._calculate_max_drawdown(data['close']),
                    "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                    "sortino_ratio": self._calculate_sortino_ratio(returns)
                },
                "trend_analysis": {
                    "trend_direction": "BULLISH" if data['close'].iloc[-1] > data['close'].iloc[0] else "BEARISH",
                    "trend_strength": abs(((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100)
                }
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics for {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "ERROR",
                "message": f"Error calculating statistics: {str(e)}"
            }

    async def get_investment_signals(self, symbol: str, config: SIPConfig) -> Dict[str, Any]:
        """Get investment signals for a symbol"""
        try:
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
                                                                  0) <= config.drawdown_threshold_1 else "WEAK",
                        "recommended_action": "BUY" if latest.get('Drawdown_100',
                                                                  0) <= config.drawdown_threshold_2 else "HOLD"
                    },
                    "price_position": {
                        "current_price": float(latest['close']),
                        "vs_20_sma": float(((latest['close'] / latest.get('SMA_20', latest['close'])) - 1) * 100),
                        "vs_50_sma": float(((latest['close'] / latest.get('SMA_50', latest['close'])) - 1) * 100)
                    }
                },
                "overall_signal": "BUY" if latest.get('Drawdown_100', 0) <= config.drawdown_threshold_2 else "HOLD"
            }

            return signals

        except Exception as e:
            logger.error(f"Error getting investment signals for {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "ERROR",
                "message": f"Error calculating signals: {str(e)}"
            }

    async def calculate_support_resistance(self, symbol: str) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                return {
                    "symbol": symbol,
                    "status": "NO_DATA"
                }

            current_price = data['close'].iloc[-1]

            # Simple support/resistance calculation
            highs = data['high'].rolling(window=20).max()
            lows = data['low'].rolling(window=20).min()

            resistance = highs.iloc[-1] if not pd.isna(highs.iloc[-1]) else current_price * 1.05
            support = lows.iloc[-1] if not pd.isna(lows.iloc[-1]) else current_price * 0.95

            return {
                "symbol": symbol,
                "status": "SUCCESS",
                "levels": {
                    "resistance": float(resistance) if resistance else None,
                    "support": float(support) if support else None
                },
                "current_price": float(current_price)
            }

        except Exception as e:
            logger.error(f"Error calculating support/resistance for {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "ERROR",
                "message": f"Error calculating support/resistance: {str(e)}"
            }

    def _generate_symbol_recommendation(self, stats: Dict, signals: Dict, support_resistance: Dict) -> Dict[str, Any]:
        """Generate recommendation for a symbol"""
        try:
            if stats.get('status') != 'SUCCESS' or signals.get('status') != 'SUCCESS':
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reasons": ["Insufficient data for analysis"]
                }

            # Simple recommendation logic
            drawdown = signals.get('signals', {}).get('drawdown_signal', {}).get('current_drawdown', 0)
            overall_signal = signals.get('overall_signal', 'HOLD')

            if drawdown <= -10:
                action = "STRONG_BUY"
                confidence = 0.8
                reasons = [f"Significant drawdown of {abs(drawdown):.1f}% presents buying opportunity"]
            elif drawdown <= -5:
                action = "BUY"
                confidence = 0.6
                reasons = [f"Moderate drawdown of {abs(drawdown):.1f}% suggests accumulation"]
            elif overall_signal == "BUY":
                action = "WEAK_BUY"
                confidence = 0.4
                reasons = ["Technical indicators suggest buying opportunity"]
            else:
                action = "HOLD"
                confidence = 0.3
                reasons = ["No clear signals - maintain current position"]

            return {
                "action": action,
                "confidence": confidence,
                "reasons": reasons,
                "priority": "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.5 else "LOW"
            }

        except Exception as e:
            logger.error(f"Error generating symbol recommendation: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasons": [f"Error generating recommendation: {str(e)}"]
            }

    def _generate_portfolio_recommendation(self, overall_metrics: Dict, symbol_reports: Dict) -> Dict[str, Any]:
        """Generate overall portfolio recommendation"""
        try:
            total_symbols = overall_metrics["total_symbols"]
            strong_buy_signals = overall_metrics["strong_buy_signals"]
            buy_signals = overall_metrics["buy_signals"]
            avoid_signals = overall_metrics["avoid_signals"]

            if strong_buy_signals >= total_symbols * 0.5:
                action = "AGGRESSIVE_INVESTMENT"
                recommendations = [
                    "Strong buy signals across majority of portfolio",
                    "Consider increasing investment amounts",
                    "Focus on symbols with highest confidence scores"
                ]
            elif buy_signals >= total_symbols * 0.3:
                action = "MODERATE_INVESTMENT"
                recommendations = [
                    "Good investment opportunities available",
                    "Maintain regular SIP schedule",
                    "Consider additional investments in strong performers"
                ]
            elif avoid_signals >= total_symbols * 0.5:
                action = "DEFENSIVE"
                recommendations = [
                    "Market conditions suggest caution",
                    "Reduce investment amounts temporarily",
                    "Focus on defensive stocks only"
                ]
            else:
                action = "MAINTAIN_COURSE"
                recommendations = [
                    "Mixed signals - continue with current strategy",
                    "Monitor market conditions closely",
                    "Regular SIP investments recommended"
                ]

            return {
                "portfolio_action": action,
                "recommendations": recommendations,
                "confidence": overall_metrics["avg_confidence"]
            }

        except Exception as e:
            logger.error(f"Error generating portfolio recommendation: {e}")
            return {
                "portfolio_action": "MANUAL_REVIEW",
                "recommendations": ["Portfolio analysis failed - manual review required"],
                "error": str(e)
            }

    def _generate_risk_assessment(self, symbol_reports: Dict, overall_metrics: Dict) -> Dict[str, Any]:
        """Generate risk assessment"""
        try:
            risk_levels = []

            for symbol, report in symbol_reports.items():
                if report.get('status') == 'SUCCESS':
                    stats = report.get('statistics', {})
                    volatility = stats.get('price_statistics', {}).get('volatility_annualized', 0)

                    if volatility > 40:
                        risk_levels.append("HIGH")
                    elif volatility > 25:
                        risk_levels.append("MEDIUM")
                    else:
                        risk_levels.append("LOW")

            if not risk_levels:
                overall_risk = "UNKNOWN"
            elif risk_levels.count("HIGH") > len(risk_levels) * 0.5:
                overall_risk = "HIGH"
            elif risk_levels.count("MEDIUM") > len(risk_levels) * 0.3:
                overall_risk = "MEDIUM"
            else:
                overall_risk = "LOW"

            return {
                "overall_risk_level": overall_risk,
                "risk_factors": [
                    "Market volatility",
                    "Individual stock concentration",
                    "Sector concentration risk"
                ],
                "mitigation_strategies": [
                    "Regular portfolio rebalancing",
                    "Diversification across sectors",
                    "Systematic investment approach"
                ]
            }

        except Exception as e:
            logger.error(f"Error generating risk assessment: {e}")
            return {
                "overall_risk_level": "UNKNOWN",
                "risk_factors": ["Risk assessment failed"],
                "mitigation_strategies": ["Manual risk review required"],
                "error": str(e)
            }

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

    async def get_portfolio_recommendations(self, symbols: List[str], config: SIPConfig) -> Dict[str, Any]:
        """Get portfolio recommendations for quick reports"""
        try:
            recommendations = {}

            for symbol in symbols:
                try:
                    signals = await self.get_investment_signals(symbol, config)
                    overall_signal = signals.get('overall_signal', 'HOLD')

                    if overall_signal == 'BUY':
                        recommendations[symbol] = {
                            "action": "BUY",
                            "confidence": 0.7,
                            "reason": "Technical indicators suggest buying opportunity"
                        }
                    else:
                        recommendations[symbol] = {
                            "action": "HOLD",
                            "confidence": 0.5,
                            "reason": "No clear signals - maintain position"
                        }

                except Exception as e:
                    recommendations[symbol] = {
                        "action": "ERROR",
                        "confidence": 0.0,
                        "reason": f"Analysis failed: {str(e)}"
                    }

            return recommendations

        except Exception as e:
            logger.error(f"Error getting portfolio recommendations: {e}")
            return {}

    async def quick_symbol_check(self, symbol: str) -> Dict[str, Any]:
        """Quick symbol availability and basic stats check"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            data = await self.fetch_data_from_db_async(symbol, start_date, end_date)

            if data.empty:
                return {
                    "available": False,
                    "message": "No recent data available"
                }

            return {
                "available": True,
                "current_price": float(data['close'].iloc[-1]),
                "recent_return": float(((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100),
                "data_points": len(data),
                "last_updated": data['timestamp'].iloc[-1].strftime('%Y-%m-%d')
            }

        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy"""
        return {
            "strategy_name": "Enhanced SIP Strategy",
            "version": "4.0.0",
            "description": "Advanced SIP strategy with dynamic investment amounts, technical analysis, and comprehensive portfolio management",
            "features": [
                "Fixed data fetching with proper PostgreSQL table handling",
                "Enhanced technical analysis with 15+ indicators",
                "Dynamic investment amount calculation based on market conditions",
                "Minimum investment gap enforcement",
                "Multi-symbol portfolio support",
                "Comprehensive signal generation with confidence scoring",
                "Advanced risk assessment and portfolio analytics",
                "Data quality validation and reporting",
                "Support and resistance level calculation",
                "Trend analysis across multiple timeframes",
                "Automated investment recommendations",
                "Benchmark comparison with regular SIP",
                "Monthly investment limits and price thresholds"
            ],
            "technical_indicators": [
                "Simple Moving Averages (20, 50, 200)",
                "Exponential Moving Averages (12, 26)",
                "Relative Strength Index (RSI)",
                "Williams %R",
                "Commodity Channel Index (CCI)",
                "Moving Average Convergence Divergence (MACD)",
                "Bollinger Bands with position calculation",
                "Average True Range (ATR)",
                "Drawdown calculations (20, 50, 100 day)",
                "Volatility measures (20-day rolling)",
                "Support and Resistance levels"
            ],
            "risk_management": [
                "Minimum investment gap enforcement",
                "Dynamic position sizing based on market conditions",
                "Risk-adjusted allocation suggestions",
                "Portfolio diversification scoring",
                "Comprehensive risk assessment",
                "Volatility-based adjustments",
                "Monthly investment limits",
                "Price reduction thresholds"
            ],
            "created_by": "Enhanced SIP Strategy System v4.0",
            "last_updated": datetime.now().isoformat()
        }


class MonthlyInvestmentTracker:
    """
    Tracks monthly investments with limits and price threshold logic
    Implements the core requirements:
    1. Monthly investment limits (default 4x fixed_investment)
    2. 4% price reduction threshold for multiple signals in same month
    """

    def __init__(self, max_monthly_amount: float, price_reduction_threshold: float = 4.0):
        self.max_monthly_amount = max_monthly_amount
        self.price_reduction_threshold = price_reduction_threshold / 100  # Convert to decimal
        self.monthly_investments = {}  # symbol -> {month_key -> {total_invested, investments, last_price}}
        self.skipped_investments = []

    def get_month_key(self, date: datetime) -> str:
        """Generate month key for tracking"""
        return f"{date.year}-{date.month:02d}"

    def can_invest(self, symbol: str, current_date: datetime,
                   intended_amount: float, current_price: float) -> Dict:
        """
        Check if investment is allowed based on monthly limits and price thresholds

        Returns:
        {
            'can_invest': bool,
            'suggested_amount': float,
            'reason': str,
            'remaining_budget': float
        }
        """
        month_key = self.get_month_key(current_date)

        # Initialize symbol tracking if needed
        if symbol not in self.monthly_investments:
            self.monthly_investments[symbol] = {}

        if month_key not in self.monthly_investments[symbol]:
            self.monthly_investments[symbol][month_key] = {
                'total_invested': 0.0,
                'investments': [],
                'last_price': None,
                'investment_count': 0
            }

        month_data = self.monthly_investments[symbol][month_key]
        current_invested = month_data['total_invested']
        remaining_budget = self.max_monthly_amount - current_invested

        # Check 1: Monthly limit
        if current_invested >= self.max_monthly_amount:
            self.record_skip(symbol, current_date, intended_amount,
                             "Monthly investment limit reached", current_price)
            return {
                'can_invest': False,
                'suggested_amount': 0.0,
                'reason': 'Monthly limit exceeded',
                'remaining_budget': 0.0
            }

        # Check 2: If this would exceed monthly limit, adjust amount
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

    def record_investment(self, symbol: str, date: datetime, amount: float,
                          price: float, trade_id: str) -> None:
        """Record a successful investment"""
        month_key = self.get_month_key(date)

        if symbol not in self.monthly_investments:
            self.monthly_investments[symbol] = {}
        if month_key not in self.monthly_investments[symbol]:
            self.monthly_investments[symbol][month_key] = {
                'total_invested': 0.0,
                'investments': [],
                'last_price': None,
                'investment_count': 0
            }

        month_data = self.monthly_investments[symbol][month_key]
        month_data['total_invested'] += amount
        month_data['last_price'] = price
        month_data['investment_count'] += 1
        month_data['investments'].append({
            'trade_id': trade_id,
            'date': date.strftime('%Y-%m-%d'),
            'amount': amount,
            'price': price,
            'units': amount / price
        })

        logger.debug(f"💰 Recorded investment: {symbol} ₹{amount:,.2f} @ ₹{price:.2f} "
                     f"(Month total: ₹{month_data['total_invested']:,.2f})")

    def record_skip(self, symbol: str, date: datetime, intended_amount: float,
                    reason: str, current_price: float, additional_info: Dict = None) -> None:
        """Record a skipped investment with reason"""
        skip_record = {
            'symbol': symbol,
            'date': date.strftime('%Y-%m-%d'),
            'intended_amount': intended_amount,
            'reason': reason,
            'current_price': current_price,
            'month_key': self.get_month_key(date)
        }

        if additional_info:
            skip_record.update(additional_info)

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


class EnhancedSIPStrategyWithLimits:
    """Enhanced SIP Strategy with monthly investment limits and price thresholds"""

    def __init__(self, nsedata_session: AsyncSession = None, trading_session: AsyncSession = None):
        self.nsedata_session = nsedata_session
        self.trading_session = trading_session
        self.monthly_tracker = None

    async def run_backtest(self, symbol: str, start_date: str, end_date: str,
                           config) -> Optional[Dict]:
        """
        Enhanced backtest with monthly limits and price thresholds

        This method implements both requirements:
        1. Monthly investment limits with default 4x fixed_investment
        2. 4% price reduction threshold for multiple signals in same month
        """
        try:
            logger.info(f"🚀 Starting enhanced SIP backtest for {symbol}")
            logger.info(f"📅 Period: {start_date} to {end_date}")
            logger.info(f"💰 Fixed investment: ₹{config.fixed_investment:,.2f}")
            logger.info(f"📊 Monthly limit: ₹{config.max_amount_in_a_month:,.2f}")
            logger.info(f"📉 Price reduction threshold: {config.price_reduction_threshold}%")

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
                should_invest, investment_reason = self._should_invest(row, config, i, data)

                if should_invest:
                    # Calculate investment amount based on market conditions
                    base_investment_amount = self._calculate_investment_amount(
                        current_price, data, config, i
                    )

                    # APPLY MONTHLY LIMITS AND PRICE THRESHOLD LOGIC
                    investment_check = self.monthly_tracker.can_invest(
                        symbol, current_date, base_investment_amount, current_price
                    )

                    if investment_check['can_invest']:
                        # Execute investment with approved/adjusted amount
                        final_amount = investment_check['suggested_amount']
                        units_bought = final_amount / current_price

                        total_investment += final_amount
                        total_units += units_bought

                        # Record the investment in tracker
                        trade_id = str(uuid.uuid4())
                        self.monthly_tracker.record_investment(
                            symbol, current_date, final_amount, current_price, trade_id
                        )

                        # Record trade
                        trade = {
                            'trade_id': trade_id,
                            'date': current_date.strftime('%Y-%m-%d'),
                            'price': float(current_price),
                            'units': float(units_bought),
                            'amount': float(final_amount),
                            'total_investment': float(total_investment),
                            'total_units': float(total_units),
                            'portfolio_value': float(total_units * current_price),
                            'trade_type': investment_reason,
                            'drawdown': float(row.get('Drawdown_100', 0)),
                            'original_intended_amount': float(base_investment_amount),
                            'amount_adjusted': final_amount != base_investment_amount
                        }
                        trades.append(trade)

                        logger.debug(f"💰 Investment executed: {symbol} ₹{final_amount:,.2f} @ ₹{current_price:.2f}")

                    else:
                        # Track skip reasons
                        skip_reason = investment_check['reason']
                        if 'Monthly limit' in skip_reason:
                            monthly_exceeded_count += 1
                        elif 'Price reduction' in skip_reason:
                            price_threshold_skipped += 1

                        logger.debug(f"⏭️ Investment skipped: {symbol} - {skip_reason}")

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

    def _should_invest(self, row, config, i: int, data: pd.DataFrame) -> Tuple[bool, str]:
        """Determine if we should invest based on strategy conditions"""
        current_date = row['timestamp']

        # Check if it's a regular SIP date (fallback day of month)
        is_sip_date = (
                current_date.day == config.fallback_day or
                (i > 0 and current_date.month != data.iloc[i - 1]['timestamp'].month)
        )

        # Check for drawdown opportunities
        drawdown_100 = row.get('Drawdown_100', 0)
        is_drawdown_opportunity = (
                drawdown_100 <= config.drawdown_threshold_1 or
                drawdown_100 <= config.drawdown_threshold_2
        )

        if is_sip_date:
            return True, "regular_sip"
        elif is_drawdown_opportunity:
            return True, f"drawdown_opportunity_{abs(drawdown_100):.1f}%"
        else:
            return False, "no_signal"

    def _calculate_investment_amount(self, current_price: float, data: pd.DataFrame,
                                     config, i: int) -> float:
        """Calculate investment amount based on market conditions"""
        try:
            base_amount = config.fixed_investment
            row = data.iloc[i]
            drawdown_100 = row.get('Drawdown_100', 0)

            # Determine multiplier based on drawdown
            if drawdown_100 <= config.drawdown_threshold_1:
                multiplier = config.investment_multiplier_3  # Highest multiplier for deepest drawdown
            elif drawdown_100 <= config.drawdown_threshold_2:
                multiplier = config.investment_multiplier_2  # Medium multiplier
            else:
                multiplier = config.investment_multiplier_1  # Base multiplier

            investment_amount = base_amount * multiplier

            # Apply safety caps to prevent excessive investments
            max_allowed = base_amount * 5  # Maximum 5x investment
            if investment_amount > max_allowed:
                investment_amount = max_allowed
                logger.warning(f"Capping investment at maximum allowed: {max_allowed}")

            # Ensure minimum investment
            min_allowed = base_amount * 0.5  # Minimum 50% of base amount
            if investment_amount < min_allowed:
                investment_amount = min_allowed

            return round(investment_amount, 2)

        except Exception as e:
            logger.error(f"Error determining investment amount: {e}")
            return config.fixed_investment

    def _calculate_enhanced_results(self, symbol: str, total_investment: float,
                                    final_portfolio_value: float, total_units: float,
                                    trades: List[Dict], config,
                                    start_date: str, end_date: str, data,
                                    monthly_exceeded_count: int, price_threshold_skipped: int) -> Dict:
        """Calculate comprehensive performance metrics"""

        # Basic performance metrics
        total_return_percent = ((final_portfolio_value / total_investment) - 1) * 100 if total_investment > 0 else 0

        # Calculate CAGR
        start_timestamp = data.iloc[0]['timestamp']
        end_timestamp = data.iloc[-1]['timestamp']
        years = (end_timestamp - start_timestamp).days / 365.25
        cagr_percent = ((final_portfolio_value / total_investment) ** (
                    1 / years) - 1) * 100 if years > 0 and total_investment > 0 else 0

        # Get monthly summary and skipped investments
        monthly_summary = self.monthly_tracker.get_monthly_summary(symbol)
        skipped_investments = self.monthly_tracker.skipped_investments

        results = {
            'symbol': symbol,
            'strategy_name': 'Enhanced SIP with Monthly Limits',
            'period': f"{start_date} to {end_date}",
            'total_investment': float(total_investment),
            'final_portfolio_value': float(final_portfolio_value),
            'total_units': float(total_units),
            'average_buy_price': float(total_investment / total_units) if total_units > 0 else 0,
            'total_return_percent': float(total_return_percent),
            'cagr_percent': float(cagr_percent),
            'num_trades': len(trades),
            'num_skipped': len(skipped_investments),
            'monthly_limit_exceeded': monthly_exceeded_count,
            'price_threshold_skipped': price_threshold_skipped,
            'config_used': {
                'fixed_investment': float(config.fixed_investment),
                'max_amount_in_a_month': float(config.max_amount_in_a_month),
                'price_reduction_threshold': float(config.price_reduction_threshold),
                'drawdown_threshold_1': float(config.drawdown_threshold_1),
                'drawdown_threshold_2': float(config.drawdown_threshold_2),
                'investment_multiplier_1': float(config.investment_multiplier_1),
                'investment_multiplier_2': float(config.investment_multiplier_2),
                'investment_multiplier_3': float(config.investment_multiplier_3)
            },
            'trades': self._convert_numpy_types(trades),
            'skipped_investments': self._convert_numpy_types(skipped_investments),
            'monthly_summary': self._convert_numpy_types(monthly_summary),
            'final_price': float(data.iloc[-1]['close']),
            'analytics': {
                'monthly_utilization': self._calculate_monthly_utilization(monthly_summary),
                'skip_analysis': self._analyze_skips(skipped_investments),
                'trade_frequency': len(trades) / max(1, years),
                'avg_monthly_investment': total_investment / max(1, years * 12)
            }
        }

        logger.info(f"✅ Enhanced backtest completed for {symbol}:")
        logger.info(f"   📊 Investment: ₹{total_investment:,.2f}")
        logger.info(f"   💰 Final Value: ₹{final_portfolio_value:,.2f}")
        logger.info(f"   📈 CAGR: {cagr_percent:.2f}%")
        logger.info(f"   🔄 Total Trades: {len(trades)}")
        logger.info(f"   ⏭️ Skipped: {len(skipped_investments)}")

        return results

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

    async def fetch_data_from_db_async(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch market data from database"""
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

            result = await self.nsedata_session.execute(query, {
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
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators including rolling high and drawdown"""
        try:
            if data.empty:
                return data

            # Calculate 100-period rolling high for drawdown calculation
            data['Rolling_High_100'] = data['close'].rolling(window=100, min_periods=1).max()

            # Calculate drawdown from rolling high
            data['Drawdown_100'] = ((data['close'] - data['Rolling_High_100']) / data['Rolling_High_100']) * 100

            # Calculate simple moving averages
            data['SMA_20'] = data['close'].rolling(window=20, min_periods=1).mean()
            data['SMA_50'] = data['close'].rolling(window=50, min_periods=1).mean()

            # Calculate daily returns
            data['Daily_Return'] = data['close'].pct_change()

            # Calculate volatility (20-day rolling standard deviation)
            data['Volatility_20'] = data['Daily_Return'].rolling(window=20, min_periods=1).std() * np.sqrt(252)

            return data

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data


# ============================================================================
# STRATEGY FACTORY AND INITIALIZATION
# ============================================================================

def create_enhanced_sip_strategy(nsedata_session: AsyncSession = None,
                                 trading_session: AsyncSession = None) -> EnhancedSIPStrategy:
    """Factory function to create Enhanced SIP Strategy instance"""
    return EnhancedSIPStrategy(nsedata_session=nsedata_session, trading_session=trading_session)


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_sip_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate SIP configuration parameters"""
    errors = []

    try:
        # Required fields
        required_fields = [
            'fixed_investment', 'drawdown_threshold_1', 'drawdown_threshold_2',
            'investment_multiplier_1', 'investment_multiplier_2', 'investment_multiplier_3',
            'rolling_window', 'fallback_day', 'min_investment_gap_days'
        ]

        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validation rules
        if config.get('fixed_investment', 0) <= 0:
            errors.append("Fixed investment must be positive")

        if config.get('drawdown_threshold_1', 0) >= config.get('drawdown_threshold_2', 0):
            errors.append("Drawdown threshold 1 must be more negative than threshold 2")

        if config.get('investment_multiplier_1', 0) <= 0:
            errors.append("Investment multipliers must be positive")

        if config.get('fallback_day', 0) < 1 or config.get('fallback_day', 32) > 31:
            errors.append("Fallback day must be between 1 and 31")

        if config.get('min_investment_gap_days', 0) < 1:
            errors.append("Minimum investment gap must be at least 1 day")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"Configuration validation error: {str(e)}"]


# Export key classes and functions
__all__ = [
    'EnhancedSIPStrategy',
    'SIPConfig',
    'Trade',
    'SIPResults',
    'SIPPortfolioTracker',
    'create_enhanced_sip_strategy',
    'validate_sip_config'
]
