
"""
Enhanced Analytics Module for Algo Trading Application  
Advanced ECharts integration with comprehensive technical analysis
Features: Support/Resistance, Pattern Recognition, Multiple Indicators, Trading Signals
"""

import json
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nicegui import ui, app
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from cache_manager import frontend_cache, FrontendCacheConfig
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Enhanced theme configuration
THEME_CONFIG = {
    "dark": {
        "bg": "#0a0a0a",
        "card_bg": "#1a1a1a",
        "text": "#FFFFFF",
        "text_secondary": "#B0B0B0",
        "accent": "#3b82f6",
        "grid": "#2a2a2a",
        "watermark": "rgba(255, 255, 255, 0.05)",
        "candle_up": "#00C851",
        "candle_down": "#FF4444",
        "volume": "rgba(38, 166, 154, 0.6)",
        "sma": "#FF9800",
        "ema": "#9C27B0",
        "rsi": "#2196F3",
        "macd": "#4CAF50",
        "macd_signal": "#FF5722",
        "macd_hist": "#FC107E",
        "bb_upper": "#FF5722",
        "bb_middle": "#00BCD4",
        "bb_lower": "#4CAF50",
        "support": "#4CAF50",
        "resistance": "#FF5722"
    },
    "light": {
        "bg": "#FFFFFF",
        "card_bg": "#F5F5F5",
        "text": "#000000",
        "text_secondary": "#666666",
        "accent": "#3b82f6",
        "grid": "#E0E0E0",
        "watermark": "rgba(0, 0, 0, 0.05)",
        "candle_up": "#00C851",
        "candle_down": "#FF4444",
        "volume": "rgba(38, 166, 154, 0.6)",
        "sma": "#FF9800",
        "ema": "#9C27B0",
        "rsi": "#2196F3",
        "macd": "#4CAF50",
        "macd_signal": "#FF5722",
        "macd_hist": "#FFC107",
        "bb_upper": "#FF5722",
        "bb_middle": "#00BCD4",
        "bb_lower": "#4CAF50",
        "support": "#4CAF50",
        "resistance": "#FF5722"
    }
}


@dataclass
class TechnicalSignal:
    """Data class for technical analysis signals"""
    type: str
    strength: str  # 'strong', 'medium', 'weak'
    direction: str  # 'bullish', 'bearish', 'neutral'
    description: str
    price_level: Optional[float] = None
    confidence: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class ChartState:
    """Manages chart state and real-time updates"""
    chart_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_live: bool = False
    last_update: Optional[datetime] = None
    update_interval: int = 5  # seconds
    auto_refresh: bool = True
    markers: List[Dict] = field(default_factory=list)
    drawings: List[Dict] = field(default_factory=list)


class ChartDataManager:
    """Manages chart data and real-time updates"""

    def __init__(self):
        self.data_cache = {}
        self.update_callbacks = []
        self.executor = ThreadPoolExecutor(max_workers=2)

    def add_update_callback(self, callback):
        """Add callback for data updates"""
        self.update_callbacks.append(callback)

    async def fetch_live_data(self, symbol: str, interval: str, fetch_api):
        """Fetch live data for real-time updates"""
        try:
            # Implementation for live data fetching
            # This would integrate with your existing fetch_api
            pass
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            return None

    def notify_updates(self, symbol: str, data):
        """Notify all callbacks of data updates"""
        for callback in self.update_callbacks:
            try:
                callback(symbol, data)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")


class TechnicalAnalyzer:
    """Advanced technical analysis engine"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.signals = []

    def calculate_sma(self, series: pd.Series, period: int) -> list:
        """Calculate Simple Moving Average"""
        sma = series.rolling(window=period).mean()
        return sma.tolist()

    def calculate_ema(self, series: pd.Series, period: int) -> list:
        """Calculate Exponential Moving Average"""
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.tolist()

    def calculate_rsi(self, series: pd.Series, period: int) -> list:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
        rsi = 100 - (100 / (1 + rs))
        return rsi.tolist()

    def calculate_bbands(self, series: pd.Series, period: int, std: float) -> dict:
        """Calculate Bollinger Bands"""
        mean = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper = mean + (std_dev * std)
        lower = mean - (std_dev * std)
        return {
            "upper": upper.tolist(),
            "middle": mean.tolist(),
            "lower": lower.tolist()
        }

    def calculate_macd(self, series: pd.Series, fast: int, slow: int, signal: int) -> dict:
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return {
            "macd": macd.tolist(),
            "signal": signal_line.tolist(),
            "hist": histogram.tolist()
        }

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> dict:
        """Calculate Average Directional Index (ADX)"""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

        # Calculate smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()

        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)

        # Calculate DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()

        return {
            "adx": adx.tolist(),
            "di_plus": di_plus.tolist(),
            "di_minus": di_minus.tolist()
        }

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> list:
        """Calculate Average True Range (ATR)"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.tolist()

    def calculate_wma(self, series: pd.Series, period: int) -> list:
        """Calculate Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        wma = series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        return wma.tolist()

    def calculate_linear_regression(self, series: pd.Series, period: int) -> list:
        """Calculate Linear Regression"""

        def slope(ser, n):
            x = np.array(range(len(ser)))
            slopes = [0] * (n - 1)
            reg_prices = [0] * (n - 1)
            for i in range(n, len(ser) + 1):
                y_scaled = ser[i - n:i]
                x_scaled = sm.add_constant(x[i - n:i])
                model = sm.OLS(y_scaled, x_scaled)
                results = model.fit()
                slopes.append(results.params[-1])
                reg_prices.append(model.predict(results.params)[-1])
            return reg_prices

        return slope(series, period)

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                             k_period: int = 14, d_period: int = 3) -> dict:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {
            "k": k_percent.tolist(),
            "d": d_percent.tolist()
        }

    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> list:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r.tolist()

    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> list:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_dev = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))

        cci = (typical_price - sma_tp) / (0.015 * mean_dev)
        return cci.tolist()

    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> list:
        """Calculate On-Balance Volume"""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return obv

    def calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> list:
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap.tolist()

    def calculate_pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        """Calculate Pivot Points"""
        if len(high) < 2:
            return {"pivot": [], "r1": [], "r2": [], "s1": [], "s2": []}

        # Use previous day's data for pivot calculation
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)

        return {
            "pivot": pivot.tolist(),
            "r1": r1.tolist(),
            "r2": r2.tolist(),
            "s1": s1.tolist(),
            "s2": s2.tolist()
        }

    def calculate_fibonacci_retracements(self, high_price: float, low_price: float) -> dict:
        """Calculate Fibonacci Retracement Levels"""
        price_range = high_price - low_price

        return {
            "level_0": high_price,
            "level_236": high_price - (price_range * 0.236),
            "level_382": high_price - (price_range * 0.382),
            "level_500": high_price - (price_range * 0.500),
            "level_618": high_price - (price_range * 0.618),
            "level_786": high_price - (price_range * 0.786),
            "level_100": low_price
        }

    def calculate_all_indicators(self) -> Dict:
        """Calculate all technical indicators"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        indicators = {}

        # Moving Averages
        indicators['sma_20'] = self.calculate_sma(close, 20)
        indicators['sma_50'] = self.calculate_sma(close, 50)
        indicators['sma_200'] = self.calculate_sma(close, 200)
        indicators['ema_12'] = self.calculate_ema(close, 12)
        indicators['ema_26'] = self.calculate_ema(close, 26)
        indicators['wma_20'] = self.calculate_wma(close, 20)

        # RSI
        indicators['rsi_14'] = self.calculate_rsi(close, 14)

        # MACD
        macd_data = self.calculate_macd(close, 12, 26, 9)
        indicators['macd'] = macd_data['macd']
        indicators['macd_signal'] = macd_data['signal']
        indicators['macd_hist'] = macd_data['hist']

        # Bollinger Bands
        bb_data = self.calculate_bbands(close, 20, 2)
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_middle'] = bb_data['middle']
        indicators['bb_lower'] = bb_data['lower']

        # Advanced Indicators
        adx_data = self.calculate_adx(high, low, close, 14)
        indicators['adx_14'] = adx_data['adx']
        indicators['di_plus_14'] = adx_data['di_plus']
        indicators['di_minus_14'] = adx_data['di_minus']

        indicators['atr_14'] = self.calculate_atr(high, low, close, 14)
        indicators['linreg_20'] = self.calculate_linear_regression(close, 20)

        # Additional Oscillators
        stoch_data = self.calculate_stochastic(high, low, close, 14, 3)
        indicators['stoch_k'] = stoch_data['k']
        indicators['stoch_d'] = stoch_data['d']

        indicators['williams_r'] = self.calculate_williams_r(high, low, close, 14)
        indicators['cci'] = self.calculate_cci(high, low, close, 20)

        # Volume Indicators
        volume = self.df['volume'] if 'volume' in self.df else pd.Series([0] * len(close))
        indicators['obv'] = self.calculate_obv(close, volume)
        indicators['vwap'] = self.calculate_vwap(high, low, close, volume)
        indicators['volume_sma'] = self.calculate_sma(volume, 20)

        # Pivot Points
        pivot_data = self.calculate_pivot_points(high, low, close)
        indicators['pivot'] = pivot_data['pivot']
        indicators['resistance_1'] = pivot_data['r1']
        indicators['resistance_2'] = pivot_data['r2']
        indicators['support_1'] = pivot_data['s1']
        indicators['support_2'] = pivot_data['s2']

        return indicators

    def find_support_resistance(self, window: int = 20) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels using pivot points"""
        supports = []
        resistances = []

        for i in range(window, len(self.df) - window):
            # Check for support (local minimum)
            if all(self.df['low'].iloc[i] <= self.df['low'].iloc[i - j] for j in range(1, window + 1)) and \
                    all(self.df['low'].iloc[i] <= self.df['low'].iloc[i + j] for j in range(1, window + 1)):
                supports.append(self.df['low'].iloc[i])

            # Check for resistance (local maximum)
            if all(self.df['high'].iloc[i] >= self.df['high'].iloc[i - j] for j in range(1, window + 1)) and \
                    all(self.df['high'].iloc[i] >= self.df['high'].iloc[i + j] for j in range(1, window + 1)):
                resistances.append(self.df['high'].iloc[i])

        # Remove duplicates and round
        supports = list(set([round(s, 2) for s in supports if s > 0]))
        resistances = list(set([round(r, 2) for r in resistances if r > 0]))

        return supports, resistances

    def remove_close_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """Remove levels that are too close to each other"""
        levels = sorted(levels)
        return [level for i, level in enumerate(levels) if
                i == 0 or all(abs(level - r) / r > threshold for r in levels[:i])]

    def detect_breakouts(self, current_price: float, supports: List[float], resistances: List[float],
                         threshold: float = 0.01) -> Dict[str, List[float]]:
        """Detect breakouts from support and resistance levels"""
        breakouts = {"bullish": [], "bearish": []}

        for support in supports:
            if abs(current_price - support) / support < threshold and current_price > support:
                breakouts["bullish"].append(support)

        for resistance in resistances:
            if abs(current_price - resistance) / resistance < threshold and current_price < resistance:
                breakouts["bearish"].append(resistance)

        return breakouts

    def detect_momentum_divergence(self, period: int = 14) -> Dict[str, bool]:
        """Detect RSI momentum divergence"""
        if len(self.df) < period * 2:
            return {"bullish": False, "bearish": False}

        recent_data = self.df.tail(period * 2).copy()
        recent_data['rsi'] = self.calculate_rsi(recent_data['close'], 14)

        # Find peaks and troughs
        price_highs = argrelextrema(recent_data["close"].values, np.greater, order=5)[0]
        price_lows = argrelextrema(recent_data["close"].values, np.less, order=5)[0]
        rsi_highs = argrelextrema(recent_data["rsi"].values, np.greater, order=5)[0]
        rsi_lows = argrelextrema(recent_data["rsi"].values, np.less, order=5)[0]

        bullish_divergence = False
        bearish_divergence = False

        if len(price_highs) > 1 and len(rsi_highs) > 1:
            if (recent_data["close"].iloc[price_highs[-1]] > recent_data["close"].iloc[price_highs[-2]] and
                    recent_data["rsi"].iloc[rsi_highs[-1]] < recent_data["rsi"].iloc[rsi_highs[-2]]):
                bearish_divergence = True

        if len(price_lows) > 1 and len(rsi_lows) > 1:
            if (recent_data["close"].iloc[price_lows[-1]] < recent_data["close"].iloc[price_lows[-2]] and
                    recent_data["rsi"].iloc[rsi_lows[-1]] > recent_data["rsi"].iloc[rsi_lows[-2]]):
                bullish_divergence = True

        return {"bullish": bullish_divergence, "bearish": bearish_divergence}

    def generate_trade_setup(self, risk_percentage: float = 2.0) -> Dict:
        """Generate trade setup with entry, stop loss, and targets"""
        if len(self.df) < 20:
            return {}

        current_price = self.df['close'].iloc[-1]
        atr = self.calculate_atr(self.df['high'], self.df['low'], self.df['close'], 14)
        if not atr or len(atr) == 0:
            return {}

        current_atr = atr[-1]

        # Get support and resistance levels
        supports = self.find_support_resistance()[0]
        resistances = self.find_support_resistance()[1]

        # Get technical signals
        signals = self.generate_signals()
        overall_signal = signals[0].direction if signals else 'neutral'

        # Calculate entry, stop loss, and targets based on signal
        if overall_signal == 'bullish':
            # Bullish setup
            entry = current_price
            stop_loss = entry - (current_atr * 2)  # 2 ATR below entry
            target1 = entry + (current_atr * 2)  # 2 ATR above entry
            target2 = entry + (current_atr * 4)  # 4 ATR above entry
            target3 = entry + (current_atr * 6)  # 6 ATR above entry

            # Adjust based on nearest support
            if supports:
                nearest_support = max([s for s in supports if s < current_price], default=stop_loss)
                stop_loss = max(stop_loss, nearest_support)

            # Adjust targets based on nearest resistance
            if resistances:
                nearest_resistance = min([r for r in resistances if r > current_price], default=target1)
                target1 = min(target1, nearest_resistance)

        elif overall_signal == 'bearish':
            # Bearish setup
            entry = current_price
            stop_loss = entry + (current_atr * 2)  # 2 ATR above entry
            target1 = entry - (current_atr * 2)  # 2 ATR below entry
            target2 = entry - (current_atr * 4)  # 4 ATR below entry
            target3 = entry - (current_atr * 6)  # 6 ATR below entry

            # Adjust based on nearest resistance
            if resistances:
                nearest_resistance = min([r for r in resistances if r > current_price], default=stop_loss)
                stop_loss = min(stop_loss, nearest_resistance)

            # Adjust targets based on nearest support
            if supports:
                nearest_support = max([s for s in supports if s < current_price], default=target1)
                target1 = max(target1, nearest_support)

        else:
            # Neutral - no trade setup
            return {}

        # Calculate position size based on risk
        risk_amount = current_price * (risk_percentage / 100)
        stop_distance = abs(entry - stop_loss)
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0

        # Calculate risk-reward ratios
        rr1 = abs(target1 - entry) / stop_distance if stop_distance > 0 else 0
        rr2 = abs(target2 - entry) / stop_distance if stop_distance > 0 else 0
        rr3 = abs(target3 - entry) / stop_distance if stop_distance > 0 else 0

        return {
            'signal': overall_signal,
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target1': round(target1, 2),
            'target2': round(target2, 2),
            'target3': round(target3, 2),
            'position_size': round(position_size, 2),
            'risk_reward_1': round(rr1, 2),
            'risk_reward_2': round(rr2, 2),
            'risk_reward_3': round(rr3, 2),
            'atr': round(current_atr, 2),
            'risk_percentage': risk_percentage
        }

    def calculate_position_size(self, account_size: float, risk_percentage: float,
                                entry_price: float, stop_loss: float) -> Dict:
        """Calculate position size based on risk management"""
        if entry_price <= 0 or stop_loss <= 0:
            return {}

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return {}

        # Calculate total risk amount
        risk_amount = account_size * (risk_percentage / 100)

        # Calculate position size
        position_size = risk_amount / risk_per_share

        # Calculate position value
        position_value = position_size * entry_price

        # Calculate margin requirement (assuming 20% margin)
        margin_required = position_value * 0.2

        # Calculate potential profit/loss scenarios
        scenarios = {
            'risk_amount': round(risk_amount, 2),
            'position_size': round(position_size, 0),
            'position_value': round(position_value, 2),
            'margin_required': round(margin_required, 2),
            'risk_per_share': round(risk_per_share, 2),
            'max_loss': round(risk_amount, 2),
            'max_loss_percentage': round((risk_amount / account_size) * 100, 2)
        }

        return scenarios

    def calculate_monthly_returns(self) -> Dict:
        """Calculate monthly returns for the past year"""
        if len(self.df) < 30:
            return {}

        # Ensure we have datetime index
        df_copy = self.df.copy()
        if 'time' in df_copy.columns:
            df_copy['datetime'] = pd.to_datetime(df_copy['time'], unit='s')
            df_copy.set_index('datetime', inplace=True)

        # Calculate daily returns
        df_copy['returns'] = df_copy['close'].pct_change()

        # Resample to monthly returns
        monthly_returns = df_copy['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create monthly returns dictionary
        monthly_data = {}
        for date, return_val in monthly_returns.items():
            if not pd.isna(return_val):
                month_key = date.strftime('%Y-%m')
                monthly_data[month_key] = round(return_val * 100, 2)

        # Calculate statistics
        if monthly_data:
            returns_list = list(monthly_data.values())
            stats = {
                'monthly_returns': monthly_data,
                'avg_monthly_return': round(np.mean(returns_list), 2),
                'best_month': max(monthly_data.items(), key=lambda x: x[1]),
                'worst_month': min(monthly_data.items(), key=lambda x: x[1]),
                'volatility': round(np.std(returns_list), 2),
                'positive_months': len([r for r in returns_list if r > 0]),
                'negative_months': len([r for r in returns_list if r < 0])
            }
        else:
            stats = {}

        return stats

    def detect_patterns(self) -> List[Dict]:
        """Detect enhanced candlestick patterns"""
        patterns = []

        # Only check recent candles (last 20) to avoid too many patterns
        start_idx = max(1, len(self.df) - 20)

        for i in range(start_idx, len(self.df)):
            open_price = self.df['open'].iloc[i]
            close_price = self.df['close'].iloc[i]
            high_price = self.df['high'].iloc[i]
            low_price = self.df['low'].iloc[i]

            # Calculate body and shadows
            body = abs(open_price - close_price)
            total_range = high_price - low_price

            # Skip if total range is too small (avoid noise)
            if total_range < 0.01:
                continue

            # Doji pattern
            if body <= total_range * 0.05 and total_range > 0.02:
                patterns.append({
                    'name': 'DOJI',
                    'signal': 'neutral',
                    'strength': 1,
                    'index': i
                })
                continue

            # Hammer pattern
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)

            if (lower_shadow > body * 2.5 and
                    upper_shadow < body * 0.3 and
                    body > 0.01):
                patterns.append({
                    'name': 'HAMMER',
                    'signal': 'bullish',
                    'strength': 2,
                    'index': i
                })
                continue

            # Shooting Star pattern
            if (upper_shadow > body * 2.5 and
                    lower_shadow < body * 0.3 and
                    body > 0.01):
                patterns.append({
                    'name': 'SHOOTING_STAR',
                    'signal': 'bearish',
                    'strength': 2,
                    'index': i
                })
                continue

            # Engulfing patterns (need previous candle)
            if i > 0:
                prev_open = self.df['open'].iloc[i - 1]
                prev_close = self.df['close'].iloc[i - 1]
                prev_body = abs(prev_open - prev_close)

                # Bullish Engulfing
                if (close_price > open_price and  # Current candle is bullish
                        prev_close < prev_open and  # Previous candle is bearish
                        open_price < prev_close and  # Current open below prev close
                        close_price > prev_open):  # Current close above prev open
                    patterns.append({
                        'name': 'BULLISH_ENGULFING',
                        'signal': 'bullish',
                        'strength': 3,
                        'index': i
                    })
                    continue

                # Bearish Engulfing
                if (close_price < open_price and  # Current candle is bearish
                        prev_close > prev_open and  # Previous candle is bullish
                        open_price > prev_close and  # Current open above prev close
                        close_price < prev_open):  # Current close below prev open
                    patterns.append({
                        'name': 'BEARISH_ENGULFING',
                        'signal': 'bearish',
                        'strength': 3,
                        'index': i
                    })
                    continue

            # Marubozu (strong trend candle)
            if body > total_range * 0.8:
                if close_price > open_price:
                    patterns.append({
                        'name': 'BULLISH_MARUBOZU',
                        'signal': 'bullish',
                        'strength': 3,
                        'index': i
                    })
                else:
                    patterns.append({
                        'name': 'BEARISH_MARUBOZU',
                        'signal': 'bearish',
                        'strength': 3,
                        'index': i
                    })

        # Limit to most recent 5 patterns to avoid overwhelming display
        return patterns[-5:] if len(patterns) > 5 else patterns

    def generate_signals(self) -> List[TechnicalSignal]:
        """Generate comprehensive trading signals"""
        signals = []
        indicators = self.calculate_all_indicators()

        current_price = self.df['close'].iloc[-1]
        current_rsi = indicators['rsi_14'][-1] if indicators['rsi_14'][-1] is not None else 50
        current_macd = indicators['macd'][-1] if indicators['macd'][-1] is not None else 0
        current_macd_signal = indicators['macd_signal'][-1] if indicators['macd_signal'][-1] is not None else 0
        current_bb_upper = indicators['bb_upper'][-1] if indicators['bb_upper'][-1] is not None else current_price
        current_bb_lower = indicators['bb_lower'][-1] if indicators['bb_lower'][-1] is not None else current_price
        current_sma_20 = indicators['sma_20'][-1] if indicators['sma_20'][-1] is not None else current_price
        current_sma_50 = indicators['sma_50'][-1] if indicators['sma_50'][-1] is not None else current_price
        current_sma_200 = indicators['sma_200'][-1] if indicators['sma_200'][-1] is not None else current_price

        # RSI Signals
        if current_rsi < 30:
            signals.append(TechnicalSignal(
                type="RSI",
                strength="strong",
                direction="bullish",
                description="RSI indicates oversold conditions",
                confidence=0.8
            ))
        elif current_rsi > 70:
            signals.append(TechnicalSignal(
                type="RSI",
                strength="strong",
                direction="bearish",
                description="RSI indicates overbought conditions",
                confidence=0.8
            ))

        # MACD Signals
        if current_macd > current_macd_signal and len(indicators['macd']) > 1:
            prev_macd = indicators['macd'][-2] if indicators['macd'][-2] is not None else 0
            prev_signal = indicators['macd_signal'][-2] if indicators['macd_signal'][-2] is not None else 0
            if prev_macd <= prev_signal:
                signals.append(TechnicalSignal(
                    type="MACD",
                    strength="medium",
                    direction="bullish",
                    description="MACD crossed above signal line",
                    confidence=0.7
                ))
        elif current_macd < current_macd_signal and len(indicators['macd']) > 1:
            prev_macd = indicators['macd'][-2] if indicators['macd'][-2] is not None else 0
            prev_signal = indicators['macd_signal'][-2] if indicators['macd_signal'][-2] is not None else 0
            if prev_macd >= prev_signal:
                signals.append(TechnicalSignal(
                    type="MACD",
                    strength="medium",
                    direction="bearish",
                    description="MACD crossed below signal line",
                    confidence=0.7
                ))

        # Bollinger Bands Signals
        if current_price <= current_bb_lower:
            signals.append(TechnicalSignal(
                type="Bollinger Bands",
                strength="medium",
                direction="bullish",
                description="Price at lower Bollinger Band",
                confidence=0.6
            ))
        elif current_price >= current_bb_upper:
            signals.append(TechnicalSignal(
                type="Bollinger Bands",
                strength="medium",
                direction="bearish",
                description="Price at upper Bollinger Band",
                confidence=0.6
            ))

        # Moving Average Signals
        if current_price > current_sma_20 > current_sma_50:
            signals.append(TechnicalSignal(
                type="Moving Averages",
                strength="strong",
                direction="bullish",
                description="Price above 20 and 50 SMA",
                confidence=0.75
            ))
        elif current_price < current_sma_20 < current_sma_50:
            signals.append(TechnicalSignal(
                type="Moving Averages",
                strength="strong",
                direction="bearish",
                description="Price below 20 and 50 SMA",
                confidence=0.75
            ))

        # Golden/Death Cross
        if current_sma_50 > current_sma_200 and len(indicators['sma_50']) > 1 and len(indicators['sma_200']) > 1:
            prev_sma_50 = indicators['sma_50'][-2] if indicators['sma_50'][-2] is not None else 0
            prev_sma_200 = indicators['sma_200'][-2] if indicators['sma_200'][-2] is not None else 0
            if prev_sma_50 <= prev_sma_200:
                signals.append(TechnicalSignal(
                    type="Golden Cross",
                    strength="strong",
                    direction="bullish",
                    description="50 SMA crossed above 200 SMA",
                    confidence=0.9
                ))
        elif current_sma_50 < current_sma_200 and len(indicators['sma_50']) > 1 and len(indicators['sma_200']) > 1:
            prev_sma_50 = indicators['sma_50'][-2] if indicators['sma_50'][-2] is not None else 0
            prev_sma_200 = indicators['sma_200'][-2] if indicators['sma_200'][-2] is not None else 0
            if prev_sma_50 >= prev_sma_200:
                signals.append(TechnicalSignal(
                    type="Death Cross",
                    strength="strong",
                    direction="bearish",
                    description="50 SMA crossed below 200 SMA",
                    confidence=0.9
                ))

        # Add enhanced signals based on new indicators
        if 'stoch_k' in indicators and 'stoch_d' in indicators:
            current_stoch_k = indicators['stoch_k'][-1] if indicators['stoch_k'][-1] is not None else 50
            current_stoch_d = indicators['stoch_d'][-1] if indicators['stoch_d'][-1] is not None else 50

            if current_stoch_k < 20 and current_stoch_d < 20:
                signals.append(TechnicalSignal(
                    type="Stochastic",
                    strength="medium",
                    direction="bullish",
                    description="Stochastic indicates oversold conditions",
                    confidence=0.6
                ))
            elif current_stoch_k > 80 and current_stoch_d > 80:
                signals.append(TechnicalSignal(
                    type="Stochastic",
                    strength="medium",
                    direction="bearish",
                    description="Stochastic indicates overbought conditions",
                    confidence=0.6
                ))

        # Williams %R signals
        if 'williams_r' in indicators:
            current_williams = indicators['williams_r'][-1] if indicators['williams_r'][-1] is not None else -50

            if current_williams < -80:
                signals.append(TechnicalSignal(
                    type="Williams %R",
                    strength="medium",
                    direction="bullish",
                    description="Williams %R indicates oversold conditions",
                    confidence=0.6
                ))
            elif current_williams > -20:
                signals.append(TechnicalSignal(
                    type="Williams %R",
                    strength="medium",
                    direction="bearish",
                    description="Williams %R indicates overbought conditions",
                    confidence=0.6
                ))

        # CCI signals
        if 'cci' in indicators:
            current_cci = indicators['cci'][-1] if indicators['cci'][-1] is not None else 0

            if current_cci < -100:
                signals.append(TechnicalSignal(
                    type="CCI",
                    strength="medium",
                    direction="bullish",
                    description="CCI indicates oversold conditions",
                    confidence=0.6
                ))
            elif current_cci > 100:
                signals.append(TechnicalSignal(
                    type="CCI",
                    strength="medium",
                    direction="bearish",
                    description="CCI indicates overbought conditions",
                    confidence=0.6
                ))

        return signals


def merge_state(stored_state, default_state):
    """Merge stored state with default state"""
    merged = default_state.copy()
    if stored_state:
        for key, value in stored_state.items():
            if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = merge_state(value, merged[key])
            else:
                merged[key] = value
    return merged


# Global chart data manager instance
chart_data_manager = ChartDataManager()


async def render_analytics_page(fetch_api, user_storage, instruments):
    """Render the enhanced analytics page with professional design matching the HTML reference"""
    broker = user_storage.get("broker", "Zerodha")
    theme = user_storage.get("theme", "dark")
    theme_config = THEME_CONFIG.get(theme.lower(), THEME_CONFIG["dark"])

    # Debug: Check instruments structure
    logger.info(f"Analytics page called with {len(instruments) if instruments else 0} instruments")
    if instruments:
        logger.debug(f"Instruments type: {type(instruments)}")
        sample_keys = list(instruments.keys())[:3]
        logger.debug(f"Sample instrument keys: {sample_keys}")

    # Ensure instruments is not None
    if not instruments:
        instruments = {}
        logger.warning("No instruments provided to analytics page")

    # Professional header matching the HTML reference
    with ui.row().classes("w-full items-center justify-between p-4").style(
            f"background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(10px); border-bottom: 1px solid #334155;"):
        with ui.row().classes("items-center"):
            ui.label("📈 AlgoTrader Analytics").classes("text-h5 font-bold").style("color: #3b82f6;")

        with ui.row().classes("items-center space-x-3"):
            if not instruments:
                ui.label("No instruments available").classes("text-red-500 text-caption")
                instrument_select = ui.select(
                    options=[],
                    with_input=True,
                    value=None,
                    label="Symbol"
                ).classes("w-32").props("disabled")
            else:
                instrument_select = ui.select(
                    options=sorted(list(instruments.keys())),
                    with_input=True,
                    value=list(instruments.keys())[0] if instruments else None,
                    label="Symbol"
                ).classes("w-32").style(f"background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;")

                # Auto-update when instrument changes
                instrument_select.on("update:model-value", lambda: ui.timer(0.1, lambda: update_analysis(), once=True))

            # Date range inputs
            from_date = ui.input("From Date", value="2020-01-01").props("dense type=date").classes("w-32").style(
                "background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;")
            to_date = ui.input("To Date", value=datetime.now().strftime("%Y-%m-%d")).props(
                "dense type=date").classes("w-32").style(
                "background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;")

            # Data frequency selection
            frequency_select = ui.select(
                options=["minute", "day", "week", "month"],
                value="day",
                label="Frequency"
            ).classes("w-32").style("background: #1e293b; border: 1px solid #475569; border-radius: 0.5rem;")

            ui.button("Load Chart", icon="refresh",
                      on_click=lambda: ui.timer(0.1, lambda: update_analysis(), once=True)).classes(
                "bg-blue-600 hover:bg-blue-700").style(
                "background: #3b82f6; border-radius: 0.5rem; padding: 0.5rem 1rem;")

    # Helper function for async indicator updates
    def trigger_indicator_update():
        """Trigger async indicator update"""
        async def update_wrapper():
            try:
                logger.info("Triggering indicator update...")
                await update_chart_indicators()
                logger.info("Indicator update completed")
            except Exception as e:
                logger.error(f"Error in indicator update: {e}")
                ui.notify("Error updating indicators", type="warning")
        
        ui.timer(0.1, update_wrapper, once=True)
    
    # Define dialog functions after UI elements are created
    def show_price_alert_dialog():
        """Show price alert creation dialog"""
        selected_symbol = instrument_select.value
        if not selected_symbol:
            ui.notify("Please select a symbol first", type="warning")
            return
            
        current_price = None
        if not df.empty:
            current_price = float(df.iloc[-1]['close'])
        
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label(f"Set Price Alert for {selected_symbol}").classes("text-h6 font-bold mb-4")
            
            if current_price:
                ui.label(f"Current Price: ₹{current_price:.2f}").classes("text-sm text-gray-400 mb-2")
            
            with ui.column().classes("gap-4 w-full"):
                alert_price = ui.number("Alert Price (₹)", 
                                      value=current_price if current_price else 100, 
                                      min=0.01, step=0.01, format="%.2f").classes("w-full")
                
                alert_type = ui.select(["Above", "Below"], value="Above", label="Alert When Price").classes("w-full")
                
                notification_method = ui.select(["Popup", "Email", "SMS"], value="Popup", 
                                              label="Notification Method").classes("w-full")
                
                with ui.row().classes("justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("outline")
                    ui.button("Set Alert", on_click=lambda: create_price_alert(
                        selected_symbol, alert_price.value, alert_type.value, 
                        notification_method.value, dialog
                    )).props("color=primary")
        
        dialog.open()

    def create_price_alert(symbol, price, alert_type, method, dialog):
        """Create a price alert"""
        try:
            # Here you would integrate with your alert system
            alert_message = f"Alert set for {symbol}: notify when price goes {alert_type.lower()} ₹{price:.2f} via {method}"
            ui.notify(alert_message, type="positive")
            dialog.close()
            
            # In a real implementation, you would save this alert to database/alert service
            logger.info(f"Price alert created: {symbol} {alert_type} ₹{price:.2f} via {method}")
        except Exception as e:
            logger.error(f"Error creating price alert: {e}")
            ui.notify("Error creating price alert", type="negative")

    def show_create_order_dialog():
        """Show order creation dialog"""
        selected_symbol = instrument_select.value
        if not selected_symbol:
            ui.notify("Please select a symbol first", type="warning")
            return
            
        current_price = None
        if not df.empty:
            current_price = float(df.iloc[-1]['close'])
        
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label(f"Create Order for {selected_symbol}").classes("text-h6 font-bold mb-4")
            
            if current_price:
                ui.label(f"Current Price: ₹{current_price:.2f}").classes("text-sm text-gray-400 mb-2")
            
            with ui.column().classes("gap-4 w-full"):
                order_type = ui.select(["Market", "Limit", "Stop Loss"], value="Market", 
                                     label="Order Type").classes("w-full")
                
                side = ui.select(["Buy", "Sell"], value="Buy", label="Side").classes("w-full")
                
                quantity = ui.number("Quantity", value=1, min=1, step=1).classes("w-full")
                
                price_input = ui.number("Price (₹)", 
                                      value=current_price if current_price else 100, 
                                      min=0.01, step=0.01, format="%.2f").classes("w-full")
                
                with ui.row().classes("justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close).props("outline")
                    ui.button("Place Order", on_click=lambda: place_order(
                        selected_symbol, order_type.value, side.value, 
                        quantity.value, price_input.value, dialog
                    )).props("color=primary")
        
        dialog.open()

    def place_order(symbol, order_type, side, quantity, price, dialog):
        """Place an order (simulation)"""
        try:
            # This is a simulation - in real implementation, integrate with broker API
            order_value = quantity * price
            order_message = f"{side} {quantity} shares of {symbol} at ₹{price:.2f} ({order_type}) - Total: ₹{order_value:.2f}"
            ui.notify(f"Order Simulation: {order_message}", type="info")
            dialog.close()
            
            logger.info(f"Simulated order: {order_message}")
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            ui.notify("Error placing order", type="negative")

    # Main analysis function - needs to be defined before being used in UI callbacks
    async def update_analysis():
        """Update the chart and analysis"""
        nonlocal df, analyzer

        selected_symbol = instrument_select.value
        if not selected_symbol:
            ui.notify("Please select an instrument", type="negative")
            return

        try:
            # Update chart title
            chart_title.text = f"{selected_symbol} - Stock Analysis"

            # Show enhanced loading message
            chart_container.clear()
            with chart_container:
                ui.html(f"""
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #94a3b8; font-size: 1.1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 48px; margin-bottom: 20px; animation: pulse 2s infinite;">📊</div>
                        <div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">Loading chart data...</div>
                        <div style="font-size: 14px; color: #64748b; margin-bottom: 20px;">Fetching {selected_symbol} data</div>
                        <div style="width: 200px; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; margin: 0 auto;">
                            <div style="width: 60%; height: 100%; background: linear-gradient(90deg, #3b82f6, #2563eb); border-radius: 2px; animation: loading 2s infinite;"></div>
                        </div>
                    </div>
                </div>
                <style>
                    @keyframes pulse {{
                        0% {{ transform: scale(1); }}
                        50% {{ transform: scale(1.05); }}
                        100% {{ transform: scale(1); }}
                    }}
                    @keyframes loading {{
                        0% {{ transform: translateX(-100%); }}
                        100% {{ transform: translateX(300%); }}
                    }}
                </style>
                """)

            # Enhanced data fetching with error handling
            logger.info(f"Fetching data for symbol: {selected_symbol}")

            # Fetch OHLC data from API
            try:
                # Use date range from UI inputs
                start_date = datetime.strptime(from_date.value, "%Y-%m-%d")
                end_date = datetime.strptime(to_date.value, "%Y-%m-%d")
                
                # Use frequency from UI selection
                interval = frequency_select.value

                # Get instrument token from symbol
                if not instruments or selected_symbol not in instruments:
                    ui.notify(f"Instrument '{selected_symbol}' not found in available instruments", type="negative")
                    return

                instrument_token = instruments[selected_symbol]
                logger.info(f"Selected symbol: {selected_symbol}, token: {instrument_token}")

                # Format parameters for API call
                params = {
                    "instrument": instrument_token,
                    "from_date": start_date.strftime("%Y-%m-%d"),
                    "to_date": end_date.strftime("%Y-%m-%d"),
                    "interval": 1 if interval != "minute" else 30,
                    "unit": interval,
                    "source": "default"  # Can be "default", "db", "upstox", or "openchart"
                }

                response = await fetch_api("/historical-data/Upstox", params=params)
                
                if response and not response.get("error"):
                    candles = response.get("data", [])
                    if not candles:
                        show_empty_chart("No data available for the selected parameters")
                        return
                else:
                    error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                    logger.warning(f"Failed to fetch OHLC data: {error_msg}")
                    show_empty_chart(f"Failed to fetch data: {error_msg}")
                    return
                    
                # Convert to DataFrame
                df = pd.DataFrame(candles)
                
                if df.empty:
                    show_empty_chart("No data available")
                    return
                    
                # Data preprocessing - ensure all required columns exist
                required_columns = ["open", "high", "low", "close", "volume"]
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = 0.0 if col != "volume" else 0

                # Convert timestamp to datetime if it exists
                if "timestamp" in df.columns:
                    try:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df["time"] = df["timestamp"].astype(np.int64) // 10**9
                    except Exception as e:
                        logger.error(f"Error converting timestamps: {e}")
                        show_empty_chart("Error processing timestamp data")
                        return
                else:
                    logger.error("Invalid data format: missing timestamp")
                    show_empty_chart("Invalid data format: missing timestamp")
                    return
                
                # Sort by time and clean data
                df = df.sort_values("time")

                # Validate data has numeric values
                numeric_columns = ["open", "high", "low", "close", "volume"]
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Remove rows with NaN values
                df = df.dropna(subset=["open", "high", "low", "close"])

                if df.empty or len(df) < 10:
                    show_empty_chart("Insufficient valid data points")
                    return
                    
                logger.info(f"Successfully loaded {len(df)} data points for {selected_symbol}")
                
            except Exception as e:
                logger.error(f"Data fetching failed: {e}")
                show_empty_chart("Data fetching failed")
                return

            # Initialize technical analyzer
            try:
                analyzer = TechnicalAnalyzer(df)
                logger.info("Technical analyzer initialized successfully")
            except Exception as e:
                logger.error(f"Analyzer initialization failed: {e}")
                show_empty_chart("Technical analysis initialization failed")
                return

            # Generate analysis data
            try:
                indicators = analyzer.calculate_all_indicators() if analyzer else {}
                signals = analyzer.generate_signals() if analyzer else []
                supports, resistances = analyzer.find_support_resistance() if analyzer else ([], [])
                patterns = analyzer.detect_patterns() if analyzer else []
            except Exception as e:
                logger.error(f"Error in technical analysis: {e}")
                ui.notify("Error calculating technical indicators", type="warning")
                indicators = {}
                signals = []
                supports, resistances = [], []
                patterns = []

            # Get current indicator configurations
            try:
                indicators_config = {
                    'sma_20': sma_20.value,
                    'sma_50': sma_50.value,
                    'sma_200': sma_200.value,
                    'ema_12': ema_12.value,
                    'ema_26': ema_26.value,
                    'bb': bb.value,
                    'heikin_ashi': heikin_ashi.value,
                    'rsi_14': rsi_14.value,
                    'macd': macd.value,
                    'stoch': stoch.value,
                    'williams_r': williams_r.value,
                    'cci': cci.value,
                    'obv': obv.value,
                    'volume_sma': volume_sma.value,
                    'overlay_volume': overlay_volume.value,
                    'support_resistance': support_resistance.value,
                    'pattern_recognition': pattern_recognition.value,
                    'fibonacci': fibonacci.value,
                    'pivot_points': pivot_points.value,
                    'vwap': vwap_switch.value
                }
                await render_echart(df, analyzer, indicators_config)
            except Exception as e:
                logger.error(f"Error rendering chart: {e}")
                ui.notify("Error rendering chart", type="warning")
                show_empty_chart("Chart rendering failed")
                return

            # Update right sidebar widgets with analysis data
            try:
                # Update trading signals
                update_signals_display(signals)
                
                # Update patterns (only if enabled)
                if pattern_recognition.value:
                    update_patterns_display(patterns)
                else:
                    patterns_container.clear()
                    with patterns_container:
                        ui.html('<div style="text-align: center; padding: 20px; color: #64748b;"><div style="font-size: 24px; margin-bottom: 10px;">📊</div><div style="font-size: 14px;">Pattern recognition disabled</div></div>')
                
                # Update technical summary
                update_summary_display(df, indicators, signals)
                
                # Update support/resistance levels (only if enabled)
                if support_resistance.value:
                    update_levels_display(supports, resistances)
                else:
                    levels_container.clear()
                    with levels_container:
                        ui.html('<div style="text-align: center; padding: 20px; color: #64748b;"><div style="font-size: 24px; margin-bottom: 10px;">📊</div><div style="font-size: 14px;">Support/Resistance disabled</div></div>')
                
                # Update Trade Setup display
                try:
                    trade_setup = analyzer.generate_trade_setup() if analyzer else {}
                    update_trade_setup_display(trade_setup)
                except Exception as e:
                    logger.error(f"Error updating trade setup: {e}")
                    trade_setup_container.clear()
                    with trade_setup_container:
                        ui.html('<div style="text-align: center; padding: 20px; color: #64748b;"><div style="font-size: 24px; margin-bottom: 10px;">💰</div><div style="font-size: 14px;">Trade setup calculation failed</div></div>')
                
                # Update Breakouts display
                try:
                    if analyzer and supports and resistances:
                        current_price = df['close'].iloc[-1]
                        breakouts = analyzer.detect_breakouts(current_price, supports, resistances)
                        update_breakouts_display(breakouts)
                    else:
                        breakouts_container.clear()
                        with breakouts_container:
                            ui.html('<div style="text-align: center; padding: 20px; color: #64748b;"><div style="font-size: 24px; margin-bottom: 10px;">🚀</div><div style="font-size: 14px;">No breakout data available</div></div>')
                except Exception as e:
                    logger.error(f"Error updating breakouts: {e}")
                    breakouts_container.clear()
                    with breakouts_container:
                        ui.html('<div style="text-align: center; padding: 20px; color: #64748b;"><div style="font-size: 24px; margin-bottom: 10px;">🚀</div><div style="font-size: 14px;">Breakout calculation failed</div></div>')
                
                # Update Monthly Returns display
                try:
                    monthly_returns = analyzer.calculate_monthly_returns() if analyzer else {}
                    update_monthly_returns_display(monthly_returns)
                except Exception as e:
                    logger.error(f"Error updating monthly returns: {e}")
                    monthly_returns_container.clear()
                    with monthly_returns_container:
                        ui.html('<div style="text-align: center; padding: 20px; color: #64748b;"><div style="font-size: 24px; margin-bottom: 10px;">📅</div><div style="font-size: 14px;">Monthly returns calculation failed</div></div>')
                
                logger.info("Right sidebar widgets updated successfully")
            except Exception as e:
                logger.error(f"Error updating analysis displays: {e}")
                ui.notify("Error updating analysis displays", type="warning")

            # Update chart state
            chart_state.last_update = datetime.now()
            logger.info(f"Chart updated successfully at {chart_state.last_update}")

        except Exception as e:
            logger.error(f"Comprehensive error in update_analysis: {e}")
            ui.notify("Error updating analysis", type="negative")
            show_empty_chart("Analysis update failed")

    # Add CSS overrides to fix NiceGUI flex conflicts
    ui.add_head_html("""
    <style>
    .chart-column.nicegui-column {
        display: block !important;
        align-items: unset !important;
    }
    .chart-area.nicegui-column {
        align-items: stretch !important;
    }
    .chart-container.nicegui-column {
        align-items: stretch !important;
    }
    .chart-container .nicegui-echart {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }
    </style>
    """)

    # Main container using CSS Grid for precise layout control
    with ui.element('div').classes("w-full").style("height: calc(100vh - 80px); display: grid; grid-template-columns: 320px minmax(0, 1fr) 320px; gap: 8px; padding: 8px;"):
        # Left sidebar - Technical Indicators (controlled by grid)
        with ui.column().classes("space-y-4").style(
                "background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); border-radius: 1rem; border: 1px solid #334155; padding: 1.5rem; overflow-y: auto;"):
            ui.label("📊 Technical Indicators").classes("text-h6 font-bold").style(
                "color: #3b82f6; margin-bottom: 1rem;")

            # Trend Indicators
            with ui.expansion("📈 Trend Indicators", icon="trending_up").classes("w-full").style(
                    "background: rgba(51, 65, 85, 0.3); border-radius: 0.5rem; margin-bottom: 0.5rem;"):
                with ui.column().classes("space-y-2 p-2"):
                    def handle_sma_20_change():
                        ui.notify("SMA 20 toggled!", type="info")
                        logger.info(f"SMA 20 switch changed to: {sma_20.value}")
                        trigger_indicator_update()
                    
                    sma_20 = ui.switch("SMA (20)", value=False, on_change=handle_sma_20_change).classes("w-full")

                    sma_50 = ui.switch("SMA (50)", value=True, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    sma_200 = ui.switch("SMA (200)", value=True, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    ema_12 = ui.switch("EMA (12)", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    ema_26 = ui.switch("EMA (26)", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    bb = ui.switch("Bollinger Bands", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    heikin_ashi = ui.switch("Heikin Ashi", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

            # Momentum Indicators
            with ui.expansion("⚡ Momentum", icon="speed").classes("w-full").style(
                    "background: rgba(51, 65, 85, 0.3); border-radius: 0.5rem; margin-bottom: 0.5rem;"):
                with ui.column().classes("space-y-2 p-2"):
                    rsi_14 = ui.switch("RSI (14)", value=True, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    macd = ui.switch("MACD", value=True, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    stoch = ui.switch("Stochastic", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    williams_r = ui.switch("Williams %R", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    cci = ui.switch("CCI", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

            # Volume Indicators
            with ui.expansion("📊 Volume", icon="bar_chart").classes("w-full").style(
                    "background: rgba(51, 65, 85, 0.3); border-radius: 0.5rem; margin-bottom: 0.5rem;"):
                with ui.column().classes("space-y-2 p-2"):
                    overlay_volume = ui.switch("Overlay Volume on Price", value=True, on_change=lambda: trigger_indicator_update()).classes("w-full")
                    obv = ui.switch("OBV", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    volume_sma = ui.switch("Volume SMA", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

            # Analysis Tools
            with ui.expansion("🔧 Analysis Tools", icon="build").classes("w-full").style(
                    "background: rgba(51, 65, 85, 0.3); border-radius: 0.5rem; margin-bottom: 0.5rem;"):
                with ui.column().classes("space-y-2 p-2"):
                    support_resistance = ui.switch("Support/Resistance", value=True, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    pattern_recognition = ui.switch("Pattern Recognition", value=True, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    fibonacci = ui.switch("Fibonacci Retracements", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    pivot_points = ui.switch("Pivot Points", value=True, on_change=lambda: trigger_indicator_update()).classes("w-full")

                    vwap_switch = ui.switch("VWAP", value=False, on_change=lambda: trigger_indicator_update()).classes("w-full")

        # Main chart and quick actions area (controlled by grid) - override flex properties
        with ui.column().classes("space-y-4 chart-column").style("width: 100%; height: 100%; display: block !important; align-items: unset !important;"):
            # Main chart area - override flex alignment for full width
            with ui.column().classes("space-y-4 chart-area").style(
                    "background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); border-radius: 1rem; border: 1px solid #334155; padding: 1rem; align-items: stretch !important;"):
                # Chart header with timeframe buttons
                with ui.row().classes("items-center justify-between mb-4"):
                    chart_title = ui.label("Select Symbol - Stock Analysis").classes("text-h6 font-bold").style(
                        "color: #e2e8f0;")

                    with ui.row().classes("space-x-2"):
                        timeframe_buttons = {}
                        for tf in ["1m", "5m", "1h", "1d", "1w"]:
                            btn = ui.button(tf, on_click=lambda t=tf: set_timeframe(t)).classes("text-xs").style(
                                "background: rgba(51, 65, 85, 0.8); color: #94a3b8; border: 1px solid #475569; border-radius: 0.25rem; padding: 0.25rem 0.75rem;")
                            timeframe_buttons[tf] = btn
                            if tf == "1d":
                                btn.style("background: #3b82f6; color: white; border-color: #3b82f6;")
                        
                        # Theme toggle button
                        def toggle_theme():
                            current_theme = user_storage.get("theme", "dark")
                            new_theme = "light" if current_theme == "dark" else "dark"
                            user_storage["theme"] = new_theme
                            ui.notify(f"Switched to {new_theme} theme", type="info")
                            if not df.empty and analyzer:
                                ui.timer(0.1, lambda: update_chart_indicators(), once=True)
                        
                        ui.button("🌓", on_click=toggle_theme).classes("text-xs").style(
                            "background: rgba(51, 65, 85, 0.8); color: #94a3b8; border: 1px solid #475569; border-radius: 0.25rem; padding: 0.25rem 0.5rem;").tooltip("Toggle Dark/Light Theme")

                # Chart container with live update controls
                # with ui.row().classes("items-center gap-2 mb-2"):
                    live_toggle = ui.switch("Live Updates", value=False).classes("text-sm")
                    live_toggle.on("change", lambda: toggle_live_updates(live_toggle.value))

                    refresh_btn = ui.button("Refresh", icon="refresh",
                                            on_click=lambda: ui.timer(0.1, lambda: update_analysis(), once=True)).classes(
                        "text-xs").style("background: #3b82f6; border-radius: 0.25rem; padding: 0.25rem 0.75rem;")

                    # ui.label("Auto-refresh every 30s when live").classes("text-xs text-gray-400")

                chart_container = ui.column().classes("w-full chart-container").style(
                    "background: #0f172a; border-radius: 0.5rem; border: 1px solid #334155; min-height: 600px; width: 100%; max-width: none; flex: 1 1 auto; position: relative; overflow: hidden; align-items: stretch !important;")

                # Initialize with professional placeholder
                with chart_container:
                    ui.html(f"""
                    <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #94a3b8; font-size: 1.1rem;">
                        <div style="text-align: center;">
                            <div style="font-size: 48px; margin-bottom: 20px;">📈</div>
                            <div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">Select a symbol to start analysis</div>
                            <div style="font-size: 14px; color: #64748b; margin-bottom: 20px;">Professional trading chart with real-time indicators</div>
                            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
                                <div style="text-align: center;">
                                    <div style="font-size: 24px; margin-bottom: 5px;">🎯</div>
                                    <div style="font-size: 12px;">Technical Analysis</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 24px; margin-bottom: 5px;">📈</div>
                                    <div style="font-size: 12px;">Real-time Data</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 24px; margin-bottom: 5px;">⚡</div>
                                    <div style="font-size: 12px;">Fast Performance</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """)

            # Quick Actions - Same width as chart widget
            with ui.column().classes("w-full").style(
                    "background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); border-radius: 1rem; border: 1px solid #334155; padding: 1rem;"):
                with ui.row().classes("items-center justify-between w-full"):
                    with ui.row().classes("items-center"):
                        ui.icon("flash_on").classes("text-lg mr-2").style("color: #3b82f6;")
                        ui.label("⚡ Quick Actions").classes("text-h6 font-bold").style("color: #3b82f6;")

                    with ui.row().classes("gap-2"):
                        ui.button("Set Price Alert", icon="notifications", 
                                on_click=show_price_alert_dialog).classes(
                            "text-sm").style("background: #3b82f6; border-radius: 0.5rem; padding: 0.5rem 1rem;")
                        ui.button("Create Order", icon="add_shopping_cart",
                                  on_click=show_create_order_dialog).classes("text-sm").style(
                            "background: #10b981; border-radius: 0.5rem; padding: 0.5rem 1rem;")
                        ui.button("Position Calculator", icon="calculate",
                                  on_click=lambda: show_position_calculator()).classes("text-sm").style(
                            "background: #f59e0b; border-radius: 0.5rem; padding: 0.5rem 1rem;")

        # Right sidebar - Analysis Panel (controlled by grid)
        with ui.column().classes("space-y-4").style(
                "background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); border-radius: 1rem; border: 1px solid #334155; padding: 1.5rem; overflow-y: auto;"):
            # Trading Signals
            with ui.column().classes("space-y-3"):
                with ui.row().classes("items-center"):
                    ui.icon("signal_cellular_alt").classes("text-lg mr-2").style("color: #3b82f6;")
                    ui.label("🎯 Trading Signals").classes("text-h6 font-bold").style("color: #3b82f6;")
                signals_container = ui.column().classes("space-y-3")

            # Pattern Recognition
            with ui.column().classes("space-y-3"):
                with ui.row().classes("items-center"):
                    ui.icon("pattern").classes("text-lg mr-2").style("color: #3b82f6;")
                    ui.label("📈 Pattern Recognition").classes("text-h6 font-bold").style("color: #3b82f6;")
                patterns_container = ui.column().classes("space-y-3")

            # Technical Summary
            with ui.column().classes("space-y-3"):
                with ui.row().classes("items-center"):
                    ui.icon("analytics").classes("text-lg mr-2").style("color: #3b82f6;")
                    ui.label("📊 Technical Summary").classes("text-h6 font-bold").style("color: #3b82f6;")
                summary_container = ui.column().classes("space-y-3")

            # Support/Resistance Levels
            with ui.column().classes("space-y-3"):
                with ui.row().classes("items-center"):
                    ui.icon("show_chart").classes("text-lg mr-2").style("color: #3b82f6;")
                    ui.label("📊 Support/Resistance").classes("text-h6 font-bold").style("color: #3b82f6;")
                levels_container = ui.column().classes("space-y-3")

            # Trade Setup
            with ui.column().classes("space-y-3"):
                with ui.row().classes("items-center"):
                    ui.icon("trending_up").classes("text-lg mr-2").style("color: #3b82f6;")
                    ui.label("💰 Trade Setup").classes("text-h6 font-bold").style("color: #3b82f6;")
                trade_setup_container = ui.column().classes("space-y-3")

            # Breakout Detection
            with ui.column().classes("space-y-3"):
                with ui.row().classes("items-center"):
                    ui.icon("candlestick_chart").classes("text-lg mr-2").style("color: #3b82f6;")
                    ui.label("🚀 Breakouts").classes("text-h6 font-bold").style("color: #3b82f6;")
                breakouts_container = ui.column().classes("space-y-3")

            # Monthly Returns
            with ui.column().classes("space-y-3"):
                with ui.row().classes("items-center"):
                    ui.icon("calendar_month").classes("text-lg mr-2").style("color: #3b82f6;")
                    ui.label("📅 Monthly Returns").classes("text-h6 font-bold").style("color: #3b82f6;")
                monthly_returns_container = ui.column().classes("space-y-3")

    # Global variables and chart state
    df = pd.DataFrame()
    analyzer = None
    current_timeframe = "1d"
    current_period = "1Y"
    current_source = "default"
    chart_state = ChartState()
    data_manager = ChartDataManager()
    live_update_timer = None
    is_dark_theme = True


    def toggle_live_updates(enabled: bool):
        """Toggle live data updates"""
        nonlocal live_update_timer
        chart_state.is_live = enabled

        if enabled:
            # Start live updates every 30 seconds
            live_update_timer = ui.timer(30.0, lambda: update_analysis_live(), active=True)
            ui.notify("Live updates enabled (30s interval)", type="positive")
        else:
            # Stop live updates
            if live_update_timer:
                live_update_timer.cancel()
                live_update_timer = None
            ui.notify("Live updates disabled", type="info")

    async def update_analysis_live():
        """Update analysis for live mode"""
        if chart_state.is_live:
            try:
                await update_analysis()
                chart_state.last_update = datetime.now()
                logger.info(f"Live update completed at {chart_state.last_update}")
            except Exception as e:
                logger.error(f"Live update failed: {e}")
                ui.notify("Live update failed", type="warning")


    def set_timeframe(timeframe):
        """Set the current timeframe and update the chart"""
        nonlocal current_timeframe
        current_timeframe = timeframe

        # Update button styles
        for tf, btn in timeframe_buttons.items():
            if tf == timeframe:
                btn.style("background: #3b82f6; color: white; border-color: #3b82f6;")
            else:
                btn.style("background: rgba(51, 65, 85, 0.8); color: #94a3b8; border: 1px solid #475569;")

        # Update chart
        ui.timer(0.1, lambda: update_analysis(), once=True)

    async def update_chart_indicators():
        """Update chart indicators without refetching data"""
        nonlocal df, analyzer
        
        if df is None or df.empty:
            ui.notify("Please select a symbol and load data first", type="warning")
            return
            
        if analyzer is None:
            ui.notify("Chart data not ready for indicator updates", type="warning") 
            return
            
        if not df.empty and analyzer:
            try:
                # Get current indicator states with debug logging
                indicators_config = {
                    'sma_20': sma_20.value,
                    'sma_50': sma_50.value,
                    'sma_200': sma_200.value,
                    'ema_12': ema_12.value,
                    'ema_26': ema_26.value,
                    'bb': bb.value,
                    'heikin_ashi': heikin_ashi.value,
                    'rsi_14': rsi_14.value,
                    'macd': macd.value,
                    'stoch': stoch.value,
                    'williams_r': williams_r.value,
                    'cci': cci.value,
                    'obv': obv.value,
                    'volume_sma': volume_sma.value,
                    'overlay_volume': overlay_volume.value,
                    'support_resistance': support_resistance.value,
                    'pattern_recognition': pattern_recognition.value,
                    'fibonacci': fibonacci.value,
                    'pivot_points': pivot_points.value,
                    'vwap': vwap_switch.value
                }
                
                logger.info(f"Current indicator config: SMA_20={sma_20.value}, SMA_50={sma_50.value}, EMA_12={ema_12.value}")

                # Update chart with new indicator configuration
                await render_echart(df, analyzer, indicators_config)
                
                # Also update the sidebar analysis displays
                try:
                    # Re-calculate analysis with current data
                    indicators = analyzer.calculate_all_indicators() if analyzer else {}
                    signals = analyzer.generate_signals() if analyzer else []
                    supports, resistances = analyzer.find_support_resistance() if analyzer else ([], [])
                    patterns = analyzer.detect_patterns() if analyzer else []
                    
                    # Update sidebar widgets
                    update_signals_display(signals)
                    
                    if pattern_recognition.value:
                        update_patterns_display(patterns)
                    else:
                        patterns_container.clear()
                        with patterns_container:
                            ui.html('<div style="text-align: center; padding: 20px; color: #64748b;"><div style="font-size: 24px; margin-bottom: 10px;">📊</div><div style="font-size: 14px;">Pattern recognition disabled</div></div>')
                    
                    update_summary_display(df, indicators, signals)
                    
                    if support_resistance.value:
                        update_levels_display(supports, resistances)
                    else:
                        levels_container.clear()
                        with levels_container:
                            ui.html('<div style="text-align: center; padding: 20px; color: #64748b;"><div style="font-size: 24px; margin-bottom: 10px;">📊</div><div style="font-size: 14px;">Support/Resistance disabled</div></div>')
                    
                except Exception as e:
                    logger.error(f"Error updating sidebar displays: {e}")
                    
            except Exception as e:
                logger.error(f"Error updating chart indicators: {e}")
                ui.notify("Error updating indicators", type="negative")

    async def render_echart(df, analyzer, indicators_config):
        """Render the chart using ECharts"""
        logger.info(f"Rendering chart with indicators: {indicators_config}")
        
        # Get current theme
        current_theme = user_storage.get("theme", "dark")
        theme_config = THEME_CONFIG.get(current_theme.lower(), THEME_CONFIG["dark"])
        
        try:
            chart_container.clear()
            
            if df.empty:
                show_empty_chart("No data available")
                return
            
            # Prepare data with error handling
            ohlc_data = df[['open', 'close', 'low', 'high']].values.tolist()
            dates = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            volumes = df['volume'].values.tolist()
            
            # Ensure all data is in proper format
            ohlc_data = [[float(val) if pd.notna(val) else 0 for val in row] for row in ohlc_data]
            volumes = [float(vol) if pd.notna(vol) else 0 for vol in volumes]
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")
            show_empty_chart("Error preparing chart data")
            return

        # Determine chart rendering options from UI
        overlay_volume = indicators_config.get('overlay_volume', True)
        use_heikin_ashi = indicators_config.get('heikin_ashi', False)

        # Optionally transform to Heikin Ashi candles
        ohlc_data_main = ohlc_data
        if use_heikin_ashi:
            try:
                ha_open = []
                ha_close = []
                ha_high = []
                ha_low = []
                for idx, row in enumerate(df[['open', 'high', 'low', 'close']].itertuples(index=False, name=None)):
                    o, h, l, c = row
                    cur_close = (o + h + l + c) / 4.0
                    if idx == 0:
                        cur_open = (o + c) / 2.0
                    else:
                        cur_open = (ha_open[-1] + ha_close[-1]) / 2.0
                    ha_open.append(cur_open)
                    ha_close.append(cur_close)
                    ha_high.append(max(h, cur_open, cur_close))
                    ha_low.append(min(l, cur_open, cur_close))
                ohlc_data_main = [[float(o), float(c), float(l), float(h)] for o, c, l, h in zip(ha_open, ha_close, ha_low, ha_high)]
            except Exception as e:
                logger.error(f"Error computing Heikin Ashi: {e}")
                ohlc_data_main = ohlc_data

        # Build base series with optional last price line
        last_close = df['close'].iloc[-1] if not df.empty else None
        series = [
            {
                'name': 'Heikin Ashi' if use_heikin_ashi else 'Candlestick',
                'type': 'candlestick',
                'data': ohlc_data_main,
                'itemStyle': {
                    'color': theme_config['candle_up'],
                    'color0': theme_config['candle_down'],
                    'borderColor': theme_config['candle_up'],
                    'borderColor0': theme_config['candle_down']
                },
                'markLine': ({
                    'symbol': 'none',
                    'label': {'show': True, 'formatter': f"Last: {last_close:.2f}"},
                    'lineStyle': {'type': 'dashed', 'color': '#94a3b8', 'opacity': 0.6},
                    'data': [{'yAxis': float(last_close)}]
                } if last_close is not None else {})
            }
        ]

        # Decide indices for volume series based on overlay setting
        # Base yAxis count below is 4; when overlay is enabled we append a 5th yAxis for volume overlay
        volume_x_idx = 0 if overlay_volume else 1
        volume_y_idx = 4 if overlay_volume else 1

        # Volume bars colored by candle direction; softened opacity when overlaid
        volume_bar_data = [
            {
                'value': vol,
                'itemStyle': {
                    'color': (theme_config['candle_up'] if i == 0 or ohlc_data_main[i][1] >= ohlc_data_main[i][0] else theme_config['candle_down']),
                    'opacity': 0.4 if overlay_volume else 0.9
                }
            } for i, vol in enumerate(volumes)
        ]

        series.append({
            'name': 'Volume',
            'type': 'bar',
            'data': volume_bar_data,
            'xAxisIndex': volume_x_idx,
            'yAxisIndex': volume_y_idx,
            'barWidth': '60%'
        })

        legend_data = ['Heikin Ashi' if use_heikin_ashi else 'Candlestick', 'Volume']

        if indicators_config.get('sma_20'):
            logger.info("Adding SMA (20) to chart")
            legend_data.append('SMA (20)')
            series.append({'name': 'SMA (20)', 'type': 'line', 'data': analyzer.calculate_sma(df['close'], 20), 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': theme_config['sma']}})
        else:
            logger.info("SMA (20) is disabled - not adding to chart")
        if indicators_config.get('sma_50'):
            legend_data.append('SMA (50)')
            series.append({'name': 'SMA (50)', 'type': 'line', 'data': analyzer.calculate_sma(df['close'], 50), 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#3b82f6'}})
        if indicators_config.get('sma_200'):
            legend_data.append('SMA (200)')
            series.append({'name': 'SMA (200)', 'type': 'line', 'data': analyzer.calculate_sma(df['close'], 200), 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#f59e0b'}})
        if indicators_config.get('ema_12'):
            legend_data.append('EMA (12)')
            series.append({'name': 'EMA (12)', 'type': 'line', 'data': analyzer.calculate_ema(df['close'], 12), 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': theme_config['ema']}})
        if indicators_config.get('ema_26'):
            legend_data.append('EMA (26)')
            series.append({'name': 'EMA (26)', 'type': 'line', 'data': analyzer.calculate_ema(df['close'], 26), 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#8b5cf6'}})

        if indicators_config.get('bb'):
            legend_data.extend(['BB Upper', 'BB Middle', 'BB Lower'])
            bb_data = analyzer.calculate_bbands(df['close'], 20, 2)
            series.append({'name': 'BB Upper', 'type': 'line', 'data': bb_data['upper'], 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': theme_config['bb_upper']}})
            series.append({'name': 'BB Middle', 'type': 'line', 'data': bb_data['middle'], 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': theme_config['bb_middle']}})
            series.append({'name': 'BB Lower', 'type': 'line', 'data': bb_data['lower'], 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': theme_config['bb_lower']}})

        if indicators_config.get('rsi_14'):
            legend_data.append('RSI (14)')
            rsi_data = analyzer.calculate_rsi(df['close'], 14)
            series.append({'name': 'RSI (14)', 'type': 'line', 'data': rsi_data, 'xAxisIndex': 2, 'yAxisIndex': 2, 'smooth': True, 'showSymbol': False, 'lineStyle': {'opacity': .5, 'color': theme_config['rsi']}})

        if indicators_config.get('macd'):
            legend_data.extend(['MACD', 'Signal', 'Hist'])
            macd_data = analyzer.calculate_macd(df['close'], 12, 26, 9)
            series.append({'name': 'MACD', 'type': 'line', 'data': macd_data['macd'], 'xAxisIndex': 3, 'yAxisIndex': 3, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': theme_config['macd']}})
            series.append({'name': 'Signal', 'type': 'line', 'data': macd_data['signal'], 'xAxisIndex': 3, 'yAxisIndex': 3, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': theme_config['macd_signal']}})
            series.append({'name': 'Hist', 'type': 'bar', 'data': macd_data['hist'], 'xAxisIndex': 3, 'yAxisIndex': 3, 'itemStyle': {'color': '#FC107E'}})

        # Add missing momentum indicators
        if indicators_config.get('stoch'):
            legend_data.extend(['Stoch %K', 'Stoch %D'])
            stoch_data = analyzer.calculate_stochastic(df['high'], df['low'], df['close'], 14, 3)
            if stoch_data:
                series.append({'name': 'Stoch %K', 'type': 'line', 'data': stoch_data.get('k', []), 'xAxisIndex': 2, 'yAxisIndex': 2, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#ff9800'}})
                series.append({'name': 'Stoch %D', 'type': 'line', 'data': stoch_data.get('d', []), 'xAxisIndex': 2, 'yAxisIndex': 2, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#4caf50'}})

        if indicators_config.get('williams_r'):
            legend_data.append('Williams %R')
            williams_r_data = analyzer.calculate_williams_r(df['high'], df['low'], df['close'], 14)
            if williams_r_data:
                series.append({'name': 'Williams %R', 'type': 'line', 'data': williams_r_data, 'xAxisIndex': 2, 'yAxisIndex': 2, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#e91e63'}})

        if indicators_config.get('cci'):
            legend_data.append('CCI')
            cci_data = analyzer.calculate_cci(df['high'], df['low'], df['close'], 20)
            if cci_data:
                series.append({'name': 'CCI', 'type': 'line', 'data': cci_data, 'xAxisIndex': 2, 'yAxisIndex': 2, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#9c27b0'}})

        # Volume indicators
        if indicators_config.get('obv'):
            legend_data.append('OBV')
            obv_data = analyzer.calculate_obv(df['close'], df['volume'])
            if obv_data:
                if overlay_volume:
                    # Move OBV to momentum pane when volume is overlaid
                    series.append({'name': 'OBV', 'type': 'line', 'data': obv_data, 'xAxisIndex': 2, 'yAxisIndex': 2, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#795548'}})
                else:
                    series.append({'name': 'OBV', 'type': 'line', 'data': obv_data, 'xAxisIndex': 1, 'yAxisIndex': 1, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#795548'}})

        if indicators_config.get('volume_sma'):
            legend_data.append('Volume SMA')
        volume_sma_data = analyzer.calculate_sma(df['volume'], 20)
        if volume_sma_data:
            series.append({'name': 'Volume SMA', 'type': 'line', 'data': volume_sma_data, 'xAxisIndex': volume_x_idx, 'yAxisIndex': volume_y_idx, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#607d8b', 'opacity': 0.6 if overlay_volume else 1}})

        # VWAP indicator (overlaid on main chart)
        if indicators_config.get('vwap'):
            legend_data.append('VWAP')
            vwap_data = analyzer.calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
            if vwap_data:
                series.append({'name': 'VWAP', 'type': 'line', 'data': vwap_data, 'smooth': True, 'showSymbol': False, 'lineStyle': {'width': 2, 'color': '#ff5722'}})

        # Analysis Tools indicators
        if indicators_config.get('support_resistance'):
            try:
                supports, resistances = analyzer.find_support_resistance()
                if supports and len(supports) > 0:
                    # Add support lines
                    for i, support in enumerate(supports[:3]):  # Show top 3 support levels
                        if support is not None and not pd.isna(support):
                            legend_data.append(f'Support {i+1}')
                            support_line = [float(support)] * len(dates)
                            series.append({'name': f'Support {i+1}', 'type': 'line', 'data': support_line, 'smooth': False, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#4caf50', 'type': 'dashed'}})
                
                if resistances and len(resistances) > 0:
                    # Add resistance lines
                    for i, resistance in enumerate(resistances[:3]):  # Show top 3 resistance levels
                        if resistance is not None and not pd.isna(resistance):
                            legend_data.append(f'Resistance {i+1}')
                            resistance_line = [float(resistance)] * len(dates)
                            series.append({'name': f'Resistance {i+1}', 'type': 'line', 'data': resistance_line, 'smooth': False, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#f44336', 'type': 'dashed'}})
            except Exception as e:
                logger.error(f"Error adding support/resistance lines: {e}")

        if indicators_config.get('pivot_points'):
            try:
                pivot_data = analyzer.calculate_pivot_points(df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1])
                if pivot_data and isinstance(pivot_data, dict):
                    # Add pivot point lines
                    for name, value in pivot_data.items():
                        if value is not None and not pd.isna(value):
                            legend_data.append(name.upper())
                            pivot_line = [float(value)] * len(dates)
                            color = '#2196f3' if 'pivot' in name.lower() else ('#ff9800' if 'r' in name.lower() else '#9c27b0')
                            series.append({'name': name.upper(), 'type': 'line', 'data': pivot_line, 'smooth': False, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': color, 'type': 'dotted'}})
            except Exception as e:
                logger.error(f"Error adding pivot points: {e}")

        if indicators_config.get('fibonacci'):
            try:
                # Get recent high and low for Fibonacci retracement
                recent_data = df.tail(50)  # Last 50 periods
                if len(recent_data) > 10:
                    high_price = float(recent_data['high'].max())
                    low_price = float(recent_data['low'].min())
                    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                    
                    if high_price > low_price:  # Ensure valid range
                        for level in fib_levels:
                            fib_price = high_price - (high_price - low_price) * level
                            legend_data.append(f'Fib {level:.1%}')
                            fib_line = [float(fib_price)] * len(dates)
                            series.append({'name': f'Fib {level:.1%}', 'type': 'line', 'data': fib_line, 'smooth': False, 'showSymbol': False, 'lineStyle': {'width': 1, 'color': '#673ab7', 'type': 'dotted', 'opacity': 0.7}})
            except Exception as e:
                logger.error(f"Error adding fibonacci retracements: {e}")

        try:
            with chart_container:
                # Create chart with proper width detection
                # Compute dynamic grids based on volume overlay setting
                if overlay_volume:
                    grids = [
                        {'left': '3%', 'right': '2%', 'top': '8%', 'height': '58%'},  # Expanded main chart
                        {'left': '3%', 'right': '2%', 'top': '58%', 'height': '0%'},   # Hidden volume pane
                        {'left': '3%', 'right': '2%', 'top': '70%', 'height': '15%'},  # Momentum
                        {'left': '3%', 'right': '2%', 'top': '87%', 'height': '10%'}   # MACD
                    ]
                else:
                    grids = [
                        {'left': '3%', 'right': '2%', 'top': '8%', 'height': '45%'},  # Main candlestick chart
                        {'left': '3%', 'right': '2%', 'top': '58%', 'height': '12%'},  # Volume chart
                        {'left': '3%', 'right': '2%', 'top': '75%', 'height': '12%'},  # Momentum indicators (RSI, Stochastic, etc.)
                        {'left': '3%', 'right': '2%', 'top': '90%', 'height': '8%'}    # MACD
                    ]

                chart_config = {
                'backgroundColor': theme_config['bg'],
                'tooltip': {
                    'trigger': 'axis',
                    'axisPointer': {
                        'type': 'cross',
                        'link': [{'xAxisIndex': 'all'}],  # Sync crosshair across all grids
                        'crossStyle': {'color': theme_config['text']}
                    }
                },
                'legend': {
                    'data': legend_data, 
                    'textStyle': {'color': theme_config['text']},
                    'top': '1%'
                },
                'grid': grids,
                'xAxis': [
                    {'type': 'category', 'data': dates, 'scale': True, 'boundaryGap': False, 'axisLine': {'onZero': False}, 'splitLine': {'show': False}, 'axisLabel': {'show': False}, 'min': 'dataMin', 'max': 'dataMax'},
                    {'type': 'category', 'data': dates, 'gridIndex': 1, 'scale': True, 'boundaryGap': False, 'axisLine': {'onZero': False}, 'axisTick': {'show': True}, 'splitLine': {'show': False}, 'axisLabel': {'show': False}, 'min': 'dataMin', 'max': 'dataMax'},
                    {'type': 'category', 'data': dates, 'gridIndex': 2, 'scale': True, 'boundaryGap': False, 'axisLine': {'onZero': False}, 'axisTick': {'show': True}, 'splitLine': {'show': False}, 'axisLabel': {'show': False}, 'min': 'dataMin', 'max': 'dataMax'},
                    {'type': 'category', 'data': dates, 'gridIndex': 3, 'scale': True, 'boundaryGap': False, 'axisLine': {'onZero': False}, 'axisTick': {'show': True}, 'splitLine': {'show': False}, 'axisLabel': {'show': True, 'textStyle': {'color': theme_config['text']}}, 'min': 'dataMin', 'max': 'dataMax'}
                ],
                'yAxis': (
                    [
                        {'scale': True, 'splitArea': {'show': True}, 'axisLabel': {'textStyle': {'color': theme_config['text']}}},
                        {'scale': True, 'gridIndex': 1, 'splitNumber': 3, 'axisLabel': {'show': True, 'textStyle': {'color': theme_config['text']}}, 'axisLine': {'show': True}, 'axisTick': {'show': True}, 'splitLine': {'show': True}},
                        {'scale': True, 'gridIndex': 2, 'splitNumber': 3, 'axisLabel': {'show': True, 'textStyle': {'color': theme_config['text']}}, 'axisLine': {'show': True}, 'axisTick': {'show': True}, 'splitLine': {'show': True}},
                        {'scale': True, 'gridIndex': 3, 'splitNumber': 2, 'axisLabel': {'show': True, 'textStyle': {'color': theme_config['text']}}, 'axisLine': {'show': True}, 'axisTick': {'show': True}, 'splitLine': {'show': True}},
                    ]
                    + (
                        [
                            # Additional y-axis for overlaid volume on main grid
                            {
                                'scale': True,
                                'gridIndex': 0,
                                'position': 'right',
                                'axisLabel': {'show': False},
                                'axisLine': {'show': False},
                                'axisTick': {'show': False},
                                'splitLine': {'show': False},
                                'min': 0,
                                'max': (max(volumes) * 4 if len(volumes) > 0 else 1)
                            }
                        ] if overlay_volume else []
                    )
                ),
                'dataZoom': [
                    {'type': 'inside', 'xAxisIndex': [0, 1, 2, 3], 'start': 85, 'end': 100},
                    {'show': True, 'xAxisIndex': [0, 1, 2, 3], 'type': 'slider', 'top': '97%', 'start': 85, 'end': 100}
                ],
                'series': series
            }
            
                chart = ui.echart(chart_config).style('height: 800px; width: 100%; max-width: 100%; min-width: 0; display: block;').classes('w-full').props('responsive=true autoResize=true')
                
                # Force ECharts to resize properly after layout settles
                ui.add_head_html('<script>setTimeout(function() { if (window.echarts) { window.echarts.getInstanceByDom && window.echarts.getInstanceByDom(document.querySelector(".nicegui-echart")) && window.echarts.getInstanceByDom(document.querySelector(".nicegui-echart")).resize(); } }, 300);</script>')
                
                # Also try the update method
                ui.timer(0.2, lambda: chart.update(), once=True)
                
        except Exception as e:
            logger.error(f"Error rendering chart: {e}")
            show_empty_chart("Chart rendering failed")

    def show_empty_chart(message):
        """Show empty chart state"""
        chart_container.clear()
        with chart_container:
            ui.html(f"""
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #94a3b8; font-size: 1.1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">{message}</div>
                    <div style="font-size: 14px; color: #64748b;">Try selecting different parameters</div>
                </div>
            </div>
            """)

    def show_position_calculator():
        """Show position sizing calculator dialog"""
        with ui.dialog() as dialog, ui.card().style("min-width: 400px; padding: 20px;"):
            ui.label("Position Size Calculator").classes("text-h6 font-bold mb-4")

            with ui.column().classes("space-y-4"):
                with ui.row().classes("space-x-4"):
                    account_size = ui.number("Account Size (₹)", value=100000, min=1000, step=1000).classes("flex-1")
                    risk_percentage = ui.number("Risk %", value=2.0, min=0.1, max=10, step=0.1).classes("flex-1")

                with ui.row().classes("space-x-4"):
                    entry_price = ui.number("Entry Price (₹)", value=100, min=1, step=0.01).classes("flex-1")
                    stop_loss = ui.number("Stop Loss (₹)", value=95, min=1, step=0.01).classes("flex-1")

                # Results container
                results_container = ui.column().classes("mt-4")

                with ui.row().classes("space-x-4 mt-4"):
                    ui.button("Calculate", icon="calculate", 
                             on_click=lambda: calculate_position(account_size.value, risk_percentage.value,
                                                               entry_price.value, stop_loss.value, results_container)).classes("bg-blue-600")
                    ui.button("Close", icon="close", on_click=dialog.close).classes("bg-gray-600")
        
        dialog.open()

    def calculate_position(account_size, risk_percentage, entry_price, stop_loss, results_container):
        """Calculate position size and show results"""
        if entry_price <= 0 or stop_loss <= 0:
            ui.notify("Invalid prices", type="negative")
            return

        # Calculate position size
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            ui.notify("Invalid stop loss", type="negative")
            return

        risk_amount = account_size * (risk_percentage / 100)
        position_size = risk_amount / risk_per_share
        position_value = position_size * entry_price
        margin_required = position_value * 0.2

        # Clear previous results and show new results
        results_container.clear()
        with results_container:
            ui.separator().classes("my-3")
            ui.label("Position Size Results").classes("text-h6 font-bold text-blue-400 mb-3")

            with ui.column().classes("space-y-2"):
                with ui.row().classes("justify-between"):
                    ui.label("Position Size:").classes("text-sm font-medium")
                    ui.label(f"{position_size:.0f} shares").classes("text-sm font-bold text-green-400")
                
                with ui.row().classes("justify-between"):
                    ui.label("Position Value:").classes("text-sm font-medium")
                    ui.label(f"₹{position_value:.2f}").classes("text-sm font-bold")
                
                with ui.row().classes("justify-between"):
                    ui.label("Risk Amount:").classes("text-sm font-medium")
                    ui.label(f"₹{risk_amount:.2f}").classes("text-sm font-bold text-red-400")
                
                with ui.row().classes("justify-between"):
                    ui.label("Margin Required:").classes("text-sm font-medium")
                    ui.label(f"₹{margin_required:.2f}").classes("text-sm font-bold text-yellow-400")
                
                with ui.row().classes("justify-between"):
                    ui.label("Risk per Share:").classes("text-sm font-medium")
                    ui.label(f"₹{risk_per_share:.2f}").classes("text-sm font-bold")
                
                with ui.row().classes("justify-between"):
                    ui.label("Max Loss:").classes("text-sm font-medium")
                    ui.label(f"{risk_percentage}%").classes("text-sm font-bold text-red-400")
        
        ui.notify("Position calculated successfully!", type="positive")

    def update_signals_display(signals):
        """Update the trading signals display with professional styling"""
        signals_container.clear()

        if not signals:
            with signals_container:
                ui.html(f"""
                <div style="text-align: center; padding: 20px; color: #64748b;">
                    <div style="font-size: 24px; margin-bottom: 10px;">📊</div>
                    <div style="font-size: 14px;">No signals detected</div>
                </div>
                """)
            return

        for signal in signals:
            color_map = {
                "bullish": "#10b981",
                "bearish": "#ef4444",
                "neutral": "#f59e0b"
            }

            with signals_container:
                ui.html(f"""
                <div style="background: rgba(51, 65, 85, 0.5); border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem; border-left: 4px solid {color_map.get(signal.direction, '#3b82f6')};">
                    <div style="font-weight: 600; margin-bottom: 0.25rem; color: #e2e8f0;">{signal.type}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">{signal.description}</div>
                </div>
                """)

    def update_patterns_display(patterns):
        """Update the pattern recognition display"""
        patterns_container.clear()

        if not patterns:
            with patterns_container:
                ui.html(f"""
                <div style="text-align: center; padding: 20px; color: #64748b;">
                    <div style="font-size: 24px; margin-bottom: 10px;">📊</div>
                    <div style="font-size: 14px;">No patterns detected</div>
                </div>
                """)
            return

        for pattern in patterns:
            with patterns_container:
                ui.html(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: rgba(51, 65, 85, 0.3); border-radius: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="color: #e2e8f0;">{pattern['name'].replace('_', ' ')}</span>
                    <span style="background: #3b82f6; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;">{pattern['strength'] * 50}%</span>
                </div>
                """)

    def update_summary_display(df, indicators, signals):
        """Update the technical summary display"""
        summary_container.clear()

        if df.empty:
            return

        current_price = df['close'].iloc[-1]
        bullish_signals = len([s for s in signals if s.direction == "bullish"])
        bearish_signals = len([s for s in signals if s.direction == "bearish"])
        total_signals = len(signals)

        # Determine overall sentiment
        if bullish_signals > bearish_signals:
            sentiment = "BULLISH"
            sentiment_color = "#10b981"
        elif bearish_signals > bullish_signals:
            sentiment = "BEARISH"
            sentiment_color = "#ef4444"
        else:
            sentiment = "NEUTRAL"
            sentiment_color = "#f59e0b"

        with summary_container:
            ui.html(f"""
            <div style="background: rgba(51, 65, 85, 0.5); border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;">
                <div style="font-weight: 600; margin-bottom: 0.25rem; color: {sentiment_color};">Overall Sentiment: {sentiment}</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">{bullish_signals} of {total_signals} indicators suggest {sentiment.lower()} momentum.</div>
            </div>
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #94a3b8;">Current Price:</span>
                    <span style="color: #e2e8f0; font-weight: 600;">₹{current_price:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #94a3b8;">Signals:</span>
                    <span style="color: #e2e8f0;">{total_signals}</span>
                </div>
            </div>
            """)

    def update_levels_display(supports, resistances):
        """Update the support/resistance levels display"""
        levels_container.clear()

        with levels_container:
            if supports:
                ui.html(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="font-weight: 600; color: #10b981; margin-bottom: 0.5rem;">Support Levels</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                        {''.join([f'<span style="background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;">₹{level:.2f}</span>' for level in sorted(supports, reverse=True)[:3]])}
                    </div>
                </div>
                """)

            if resistances:
                ui.html(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="font-weight: 600; color: #ef4444; margin-bottom: 0.5rem;">Resistance Levels</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                        {''.join([f'<span style="background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;">₹{level:.2f}</span>' for level in sorted(resistances)[:3]])}
                    </div>
                </div>
                """)

            if not supports and not resistances:
                ui.html(f"""
                <div style="text-align: center; padding: 20px; color: #64748b;">
                    <div style="font-size: 24px; margin-bottom: 10px;">📊</div>
                    <div style="font-size: 14px;">No levels detected</div>
                </div>
                """)

    def update_trade_setup_display(trade_setup):
        """Update the trade setup display"""
        trade_setup_container.clear()

        if not trade_setup:
            with trade_setup_container:
                ui.html(f"""
                <div style="text-align: center; padding: 20px; color: #64748b;">
                    <div style="font-size: 24px; margin-bottom: 10px;">💰</div>
                    <div style="font-size: 14px;">No trade setup available</div>
                </div>
                """)
            return

        signal_color = "#10b981" if trade_setup['signal'] == 'bullish' else "#ef4444"

        with trade_setup_container:
            ui.html(f"""
            <div style="background: rgba(51, 65, 85, 0.5); border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;">
                <div style="font-weight: 600; margin-bottom: 0.5rem; color: {signal_color}; text-transform: uppercase;">{trade_setup['signal']} Setup</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.85rem;">
                    <div><span style="color: #94a3b8;">Entry:</span> <span style="color: #e2e8f0;">₹{trade_setup['entry']}</span></div>
                    <div><span style="color: #94a3b8;">Stop Loss:</span> <span style="color: #ef4444;">₹{trade_setup['stop_loss']}</span></div>
                    <div><span style="color: #94a3b8;">Target 1:</span> <span style="color: #10b981;">₹{trade_setup['target1']}</span></div>
                    <div><span style="color: #94a3b8;">Target 2:</span> <span style="color: #10b981;">₹{trade_setup['target2']}</span></div>
                    <div><span style="color: #94a3b8;">Target 3:</span> <span style="color: #10b981;">₹{trade_setup['target3']}</span></div>
                    <div><span style="color: #94a3b8;">RR Ratio:</span> <span style="color: #e2e8f0;">{trade_setup['risk_reward_1']}:1</span></div>
                </div>
            </div>
            """)

    def update_breakouts_display(breakouts):
        """Update the breakout detection display"""
        breakouts_container.clear()

        if not breakouts or (not breakouts['bullish'] and not breakouts['bearish']):
            with breakouts_container:
                ui.html(f"""
                <div style="text-align: center; padding: 20px; color: #64748b;">
                    <div style="font-size: 24px; margin-bottom: 10px;">🚀</div>
                    <div style="font-size: 14px;">No breakouts detected</div>
                </div>
                """)
            return

        with breakouts_container:
            if breakouts['bullish']:
                ui.html(f"""
                <div style="background: rgba(16, 185, 129, 0.2); border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="font-weight: 600; color: #10b981; margin-bottom: 0.25rem;">Bullish Breakout</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Breaking above resistance at ₹{breakouts['bullish'][0]:.2f}</div>
                </div>
                """)

            if breakouts['bearish']:
                ui.html(f"""
                <div style="background: rgba(239, 68, 68, 0.2); border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="font-weight: 600; color: #ef4444; margin-bottom: 0.25rem;">Bearish Breakout</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Breaking below support at ₹{breakouts['bearish'][0]:.2f}</div>
                </div>
                """)

    def update_monthly_returns_display(monthly_returns):
        """Update the monthly returns display"""
        monthly_returns_container.clear()

        if not monthly_returns or not monthly_returns.get('monthly_returns'):
            with monthly_returns_container:
                ui.html(f"""
                <div style="text-align: center; padding: 20px; color: #64748b;">
                    <div style="font-size: 24px; margin-bottom: 10px;">📅</div>
                    <div style="font-size: 14px;">No monthly data available</div>
                </div>
                """)
            return

        with monthly_returns_container:
            # Show key statistics
            ui.html(f"""
            <div style="background: rgba(51, 65, 85, 0.5); border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;">
                <div style="font-weight: 600; margin-bottom: 0.5rem; color: #e2e8f0;">Monthly Performance</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.85rem;">
                    <div><span style="color: #94a3b8;">Avg Return:</span> <span style="color: #e2e8f0;">{monthly_returns['avg_monthly_return']}%</span></div>
                    <div><span style="color: #94a3b8;">Volatility:</span> <span style="color: #e2e8f0;">{monthly_returns.get('volatility', 'N/A')}%</span></div>
                    <div><span style="color: #94a3b8;">Best Month:</span> <span style="color: #10b981;">{monthly_returns['best_month'][1]}%</span></div>
                    <div><span style="color: #94a3b8;">Worst Month:</span> <span style="color: #ef4444;">{monthly_returns['worst_month'][1]}%</span></div>
                </div>
            </div>
            """)

            # Create enhanced heatmap visualization
            monthly_data = monthly_returns['monthly_returns']
            if monthly_data:
                # Organize data by year and month
                years_months = {}
                for month_str, return_val in monthly_data.items():
                    year, month = month_str.split('-')
                    year = int(year)
                    month = int(month)
                    if year not in years_months:
                        years_months[year] = {}
                    years_months[year][month] = return_val

                # Calculate color scale bounds
                all_returns = list(monthly_data.values())
                max_return = max(all_returns) if all_returns else 1
                min_return = min(all_returns) if all_returns else -1

                # Generate heatmap HTML
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                heatmap_html = f"""
                <div style="background: rgba(51, 65, 85, 0.3); border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;">
                    <div style="font-weight: 600; margin-bottom: 0.75rem; color: #e2e8f0;">Returns Heatmap</div>
                    <div style="font-size: 0.7rem; color: #94a3b8; margin-bottom: 0.5rem;">
                        <span style="color: #10b981;">■ Positive</span>
                        <span style="margin-left: 1rem; color: #ef4444;">■ Negative</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 40px repeat(12, 1fr); gap: 1px; margin-bottom: 2px;">
                        <div></div>
                """

                # Add month headers
                for month_name in month_names:
                    heatmap_html += f'<div style="text-align: center; font-size: 0.6rem; color: #94a3b8; padding: 2px;">{month_name}</div>'

                heatmap_html += "</div>"

                # Add year rows
                for year in sorted(years_months.keys(), reverse=True):
                    heatmap_html += f'<div style="display: grid; grid-template-columns: 40px repeat(12, 1fr); gap: 1px; margin-bottom: 1px;"><div style="text-align: right; font-size: 0.7rem; color: #94a3b8; padding: 4px 8px 4px 0;">{year}</div>'

                    for month in range(1, 13):
                        if month in years_months[year]:
                            return_val = years_months[year][month]
                            # Calculate color intensity based on return value
                            if return_val > 0:
                                intensity = min(return_val / max(max_return, 1), 1)
                                color = f"rgba(16, 185, 129, {0.3 + intensity * 0.7})"
                            else:
                                intensity = min(abs(return_val) / max(abs(min_return), 1), 1)
                                color = f"rgba(239, 68, 68, {0.3 + intensity * 0.7})"

                            heatmap_html += f'<div style="background: {color}; color: #fff; text-align: center; padding: 4px 2px; border-radius: 2px; font-size: 0.6rem; font-weight: 600; cursor: pointer;" title="{month_names[month - 1]} {year}: {return_val:+.1f}%">{return_val:+.1f}</div>'
                        else:
                            heatmap_html += '<div style="background: rgba(100, 116, 139, 0.2); border-radius: 2px; padding: 4px 2px;"></div>'

                    heatmap_html += "</div>"

                heatmap_html += "</div>"

                ui.html(heatmap_html)

            # Show recent months
            recent_months = list(monthly_returns['monthly_returns'].items())[-6:]
            for month, return_val in recent_months:
                color = "#10b981" if return_val > 0 else "#ef4444"
                with monthly_returns_container:
                    ui.html(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: rgba(51, 65, 85, 0.3); border-radius: 0.25rem; margin-bottom: 0.25rem;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">{month}</span>
                        <span style="color: {color}; font-weight: 600; font-size: 0.85rem;">{return_val}%</span>
                    </div>
                    """)

# TradingView function removed - using echarts instead

    async def update_analysis_display():
        """Update the analysis display with current data"""
        if not df.empty and analyzer:
            try:
                # Update technical analysis
                # await update_technical_analysis()  # Function will be implemented in next phase

                # Update monthly returns
                # await update_monthly_returns()  # Function will be implemented in next phase

                logger.info("Analysis display updated successfully")
            except Exception as e:
                logger.error(f"Error updating analysis display: {e}")
                ui.notify("Error updating analysis display", type="negative")

    # Chart is now handled by render_echart function

    # Initial chart update
    await update_analysis()

    # Setup data manager callbacks
    data_manager.add_update_callback(lambda symbol, data: ui.timer(0.1, lambda: update_analysis_live(), once=True))
