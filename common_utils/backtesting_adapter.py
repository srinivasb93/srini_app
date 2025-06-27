# common_utils/backtesting_adapter.py - FINAL FIXED VERSION
"""
Fixed adapter for backtesting.py library integration
"""

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)


class RSIStrategy(Strategy):
    """RSI Strategy for backtesting.py - FIXED LOGIC"""
    # Optimizable parameters
    rsi_period = 14
    oversold = 30
    overbought = 70

    def init(self):
        def rsi_calc(close):
            try:
                rsi_values = ta.rsi(pd.Series(close), length=self.rsi_period)
                rsi_array = rsi_values.fillna(50).to_numpy()

                if len(rsi_array) != len(close):
                    padded_array = np.full(len(close), 50.0)
                    valid_length = min(len(rsi_array), len(close))
                    padded_array[-valid_length:] = rsi_array[-valid_length:]
                    return padded_array

                return rsi_array
            except Exception as e:
                logger.error(f"Error calculating RSI: {e}")
                return np.full(len(close), 50.0)

        self.rsi = self.I(rsi_calc, self.data.Close)

    def next(self):
        # Only trade if we have valid RSI values and enough data
        if len(self.rsi) < self.rsi_period + 1:
            return

        current_rsi = self.rsi[-1]

        # 🔥 FIXED LOGIC: Traditional RSI Strategy
        # Buy when RSI is below oversold threshold (stock is oversold)
        if current_rsi <= self.oversold and not self.position:
            self.buy()

        # Sell when RSI is above overbought threshold (stock is overbought)
        elif current_rsi >= self.overbought and self.position:
            self.sell()


class MACDStrategy(Strategy):
    """MACD Strategy for backtesting.py - FIXED VERSION"""
    # Optimizable parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        def macd_line(close):
            """Fixed MACD line calculation"""
            try:
                macd_data = ta.macd(pd.Series(close), fast=self.fast_period, slow=self.slow_period,
                                    signal=self.signal_period)
                macd_col = f'MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}'

                if macd_col in macd_data.columns:
                    macd_values = macd_data[macd_col].fillna(0).to_numpy()

                    # Ensure correct length
                    if len(macd_values) != len(close):
                        padded_array = np.zeros(len(close))
                        valid_length = min(len(macd_values), len(close))
                        padded_array[-valid_length:] = macd_values[-valid_length:]
                        return padded_array

                    return macd_values
                else:
                    return np.zeros(len(close))
            except Exception as e:
                logger.error(f"Error calculating MACD line: {e}")
                return np.zeros(len(close))

        def macd_signal(close):
            """Fixed MACD signal calculation"""
            try:
                macd_data = ta.macd(pd.Series(close), fast=self.fast_period, slow=self.slow_period,
                                    signal=self.signal_period)
                signal_col = f'MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}'

                if signal_col in macd_data.columns:
                    signal_values = macd_data[signal_col].fillna(0).to_numpy()

                    # Ensure correct length
                    if len(signal_values) != len(close):
                        padded_array = np.zeros(len(close))
                        valid_length = min(len(signal_values), len(close))
                        padded_array[-valid_length:] = signal_values[-valid_length:]
                        return padded_array

                    return signal_values
                else:
                    return np.zeros(len(close))
            except Exception as e:
                logger.error(f"Error calculating MACD signal: {e}")
                return np.zeros(len(close))

        self.macd = self.I(macd_line, self.data.Close)
        self.macd_signal = self.I(macd_signal, self.data.Close)

    def next(self):
        # Only trade if we have enough data
        min_periods = max(self.slow_period, self.signal_period) + 1
        if len(self.macd) < min_periods:
            return

        # Buy signal: MACD crosses above signal line
        if crossover(self.macd, self.macd_signal):
            self.buy()

        # Sell signal: MACD crosses below signal line
        elif crossover(self.macd_signal, self.macd):
            self.sell()


class BollingerBandsStrategy(Strategy):
    """Bollinger Bands Strategy for backtesting.py - FIXED VERSION"""
    # Optimizable parameters
    bb_period = 20
    bb_std = 2.0

    def init(self):
        def bb_upper(close):
            """Fixed Bollinger Bands upper calculation"""
            try:
                bb_data = ta.bbands(pd.Series(close), length=self.bb_period, std=self.bb_std)
                upper_col = f'BBU_{self.bb_period}_{self.bb_std}'

                if upper_col in bb_data.columns:
                    upper_values = bb_data[upper_col].ffill().bfill().to_numpy()

                    # Ensure correct length
                    if len(upper_values) != len(close):
                        # Use close price as fallback for missing values
                        close_array = np.array(close)
                        padded_array = close_array * 1.02  # 2% above close as fallback
                        valid_length = min(len(upper_values), len(close))
                        padded_array[-valid_length:] = upper_values[-valid_length:]
                        return padded_array

                    return upper_values
                else:
                    return np.array(close) * 1.02  # Fallback
            except Exception as e:
                logger.error(f"Error calculating BB upper: {e}")
                return np.array(close) * 1.02

        def bb_lower(close):
            """Fixed Bollinger Bands lower calculation"""
            try:
                bb_data = ta.bbands(pd.Series(close), length=self.bb_period, std=self.bb_std)
                lower_col = f'BBL_{self.bb_period}_{self.bb_std}'

                if lower_col in bb_data.columns:
                    lower_values = bb_data[lower_col].ffill().bfill().to_numpy()

                    # Ensure correct length
                    if len(lower_values) != len(close):
                        # Use close price as fallback for missing values
                        close_array = np.array(close)
                        padded_array = close_array * 0.98  # 2% below close as fallback
                        valid_length = min(len(lower_values), len(close))
                        padded_array[-valid_length:] = lower_values[-valid_length:]
                        return padded_array

                    return lower_values
                else:
                    return np.array(close) * 0.98  # Fallback
            except Exception as e:
                logger.error(f"Error calculating BB lower: {e}")
                return np.array(close) * 0.98

        self.bb_upper = self.I(bb_upper, self.data.Close)
        self.bb_lower = self.I(bb_lower, self.data.Close)

    def next(self):
        # Only trade if we have enough data
        if len(self.bb_upper) < self.bb_period + 1:
            return

        current_price = self.data.Close[-1]

        # Buy when price touches lower band
        if current_price <= self.bb_lower[-1] and not self.position:
            self.buy()

        # Sell when price touches upper band
        elif current_price >= self.bb_upper[-1] and self.position:
            self.sell()


def prepare_data_for_backtesting(df: pd.DataFrame) -> pd.DataFrame:
    """Convert your data format to backtesting.py format - ENHANCED VERSION"""
    bt_data = df.copy()

    # Rename columns to match backtesting.py expectations
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }

    for old_col, new_col in column_mapping.items():
        if old_col in bt_data.columns:
            bt_data = bt_data.rename(columns={old_col: new_col})

    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in bt_data.columns:
            raise ValueError(f"Required column {col} not found in data")

    # Add Volume if missing
    if 'Volume' not in bt_data.columns:
        bt_data['Volume'] = 1000000  # Default volume

    # Clean data - remove any invalid values
    for col in required_cols:
        bt_data[col] = pd.to_numeric(bt_data[col], errors='coerce')

    # FIXED: Use new pandas methods instead of deprecated fillna
    bt_data = bt_data.ffill().bfill()

    # Ensure we have valid OHLC data
    bt_data = bt_data[(bt_data['High'] >= bt_data['Low']) &
                      (bt_data['High'] >= bt_data['Open']) &
                      (bt_data['High'] >= bt_data['Close']) &
                      (bt_data['Low'] <= bt_data['Open']) &
                      (bt_data['Low'] <= bt_data['Close'])]

    logger.info(f"Prepared {len(bt_data)} rows of clean OHLC data for backtesting")

    return bt_data[['Open', 'High', 'Low', 'Close', 'Volume']]


def convert_backtesting_result(result, opt_params=None) -> Dict:
    """Convert backtesting.py result to your application format - FINAL FIXED VERSION"""

    # Get trades data
    trades = result._trades if hasattr(result, '_trades') else pd.DataFrame()

    # 🔥 DEBUG LOGGING
    logger.info(f"=== DEBUGGING BACKTESTING RESULT ===")
    logger.info(f"Result keys: {list(result.keys())}")
    logger.info(f"Result type: {type(result)}")
    logger.info(f"Return %: {result.get('Return [%]', 'NOT FOUND')}")
    logger.info(f"# Trades: {result.get('# Trades', 'NOT FOUND')}")
    logger.info(f"Trades DataFrame shape: {trades.shape if not trades.empty else 'EMPTY'}")

    if not trades.empty:
        logger.info(f"Trades columns: {trades.columns.tolist()}")
        logger.info(f"Sample trade: {trades.iloc[0].to_dict()}")
    else:
        logger.warning("NO TRADES FOUND IN BACKTESTING RESULT!")

    # Calculate metrics
    total_trades = len(trades)
    if total_trades > 0:
        winning_trades = len(trades[trades['PnL'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100

        # Create tradebook in your format
        tradebook = []
        for _, trade in trades.iterrows():
            entry_time = trade.get('EntryTime', '')
            exit_time = trade.get('ExitTime', '')

            # 🔥 FIX: Safely convert timestamps to strings
            if hasattr(entry_time, 'strftime'):
                entry_time = entry_time.strftime('%Y-%m-%d')
            elif hasattr(entry_time, 'isoformat'):
                entry_time = entry_time.isoformat()
            else:
                entry_time = str(entry_time)

            if hasattr(exit_time, 'strftime'):
                exit_time = exit_time.strftime('%Y-%m-%d')
            elif hasattr(exit_time, 'isoformat'):
                exit_time = exit_time.isoformat()
            else:
                exit_time = str(exit_time)

            tradebook.append({
                'Date': exit_time if exit_time else entry_time,
                'Action': 'BUY' if trade['Size'] > 0 else 'SELL',
                'Price': float(trade.get('ExitPrice', trade.get('EntryPrice', 0))),
                'Quantity': abs(trade['Size']),
                'Profit': float(trade['PnL']),
                'PortfolioValue': float(result['Equity Final [$]']),  # 🔥 FIXED: Use correct key
                'EntryPrice': float(trade.get('EntryPrice', 0)),
                'ExitPrice': float(trade.get('ExitPrice', 0)),
                'ExitReason': 'Signal' if trade.get('ExitPrice') else 'Ongoing',
                'Reason': 'RSI Signal'  # Add reason for your frontend
            })
    else:
        winning_trades = losing_trades = 0
        win_rate = 0
        tradebook = []

    # 🔥 FIXED: Properly extract numeric values from result
    def safe_extract_value(key, default=0.0):
        """Safely extract numeric values from backtesting result"""
        try:
            value = result.get(key, default)

            # Handle different value types
            if hasattr(value, 'value'):  # Pandas/numpy scalar
                return float(value.value)
            elif hasattr(value, 'item'):  # Numpy scalar
                return float(value.item())
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                # For timestamps, try to extract a meaningful numeric value
                if 'Duration' in key:
                    return float(value.days) if hasattr(value, 'days') else 0.0
                else:
                    return 0.0  # Default for timestamps that shouldn't be numeric
            else:
                return float(value)
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning(f"Could not convert {key}={value} to float: {e}")
            return default

    # 🔥 FIXED: Safe value extraction
    start_value = safe_extract_value('Start', 100000.0)
    end_value = safe_extract_value('Equity Final [$]', start_value)
    return_pct = safe_extract_value('Return [%]', 0.0)

    # Calculate total profit correctly
    total_profit = (return_pct / 100) * start_value

    # Build result dictionary with safe value extraction
    converted_result = {
        'TotalProfit': total_profit,
        'TotalReturn': return_pct,
        'WinRate': float(win_rate),
        'TotalTrades': int(total_trades),
        'WinningTrades': int(winning_trades),
        'LosingTrades': int(losing_trades),
        'SharpeRatio': safe_extract_value('Sharpe Ratio', 0),
        'MaxDrawdown': abs(safe_extract_value('Max. Drawdown [%]', 0)),
        'CalmarRatio': safe_extract_value('Calmar Ratio', 0),
        'StartValue': start_value,
        'EndValue': end_value,
        'BuyHoldReturn': safe_extract_value('Buy & Hold Return [%]', 0),
        'Tradebook': tradebook,
        'StrategyName': 'Optimized Strategy',
        'optimization_enabled': opt_params is not None,

        # 🔥 ADDITIONAL KEYS FOR FRONTEND COMPATIBILITY
        'FinalPortfolioValue': end_value,
        'ProfitFactor': safe_extract_value('Profit Factor', 1.0),
        'AverageWin': sum(t['Profit'] for t in tradebook if t['Profit'] > 0) / max(winning_trades,
                                                                                   1) if winning_trades > 0 else 0,
        'AverageLoss': sum(t['Profit'] for t in tradebook if t['Profit'] < 0) / max(losing_trades,
                                                                                    1) if losing_trades > 0 else 0,
        'LargestWin': max((t['Profit'] for t in tradebook if t['Profit'] > 0), default=0),
        'LargestLoss': min((t['Profit'] for t in tradebook if t['Profit'] < 0), default=0),
        'WinningStreak': 0,  # You can implement streak calculation if needed
        'LosingStreak': 0,  # You can implement streak calculation if needed
    }

    if opt_params:
        # Get optimized parameters safely
        try:
            optimized_params = result._strategy._params if hasattr(result, '_strategy') else {}
        except AttributeError:
            optimized_params = {}

        converted_result['OptimizedParameters'] = optimized_params
        converted_result['all_runs'] = [{
            'parameters': optimized_params,
            'TotalPNL': total_profit,
            'WinRate': win_rate,
            'TotalTrades': total_trades,
            'SharpeRatio': converted_result['SharpeRatio'],
            'MaxDrawdown': converted_result['MaxDrawdown']
        }]

    return converted_result


# Simple test function to verify indicators work
def test_indicators():
    """Test function to verify indicators are working properly"""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    test_data = pd.DataFrame({
        'Open': prices + np.random.randn(100) * 0.1,
        'High': prices + abs(np.random.randn(100)) * 0.2,
        'Low': prices - abs(np.random.randn(100)) * 0.2,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    try:
        # Test RSI Strategy
        bt_rsi = Backtest(test_data, RSIStrategy, cash=10000)
        result_rsi = bt_rsi.run()
        print(f"RSI Strategy test passed: {result_rsi['Return [%]']:.2f}% return")

        # Test MACD Strategy
        bt_macd = Backtest(test_data, MACDStrategy, cash=10000)
        result_macd = bt_macd.run()
        print(f"MACD Strategy test passed: {result_macd['Return [%]']:.2f}% return")

        # Test Bollinger Bands Strategy
        bt_bb = Backtest(test_data, BollingerBandsStrategy, cash=10000)
        result_bb = bt_bb.run()
        print(f"Bollinger Bands Strategy test passed: {result_bb['Return [%]']:.2f}% return")

        return True
    except Exception as e:
        print(f"Indicator test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    test_indicators()