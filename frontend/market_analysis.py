"""
Market Analysis Utilities - market_analysis.py
Advanced market analysis functions and pattern recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import talib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PatternResult:
    """Result of pattern recognition analysis"""
    pattern_name: str
    confidence: float
    signal: str  # BUY, SELL, HOLD
    description: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: str = "1d"


@dataclass
class SupportResistance:
    """Support and resistance level identification"""
    level: float
    strength: int  # 1-5 scale
    type: str  # support, resistance
    touch_count: int
    last_touch: datetime
    distance_from_price: float


@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # trending_up, trending_down, sideways, volatile
    strength: float  # 0-1 scale
    duration_days: int
    volatility: float
    description: str


class PatternRecognition:
    """Advanced pattern recognition for technical analysis"""

    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> List[PatternResult]:
        """Detect various candlestick patterns"""
        patterns = []

        try:
            # Ensure we have required columns
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                return patterns

            open_prices = df['Open'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            close_prices = df['Close'].values

            # Bullish patterns
            hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            morning_star = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            piercing = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)

            # Bearish patterns
            hanging_man = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
            dark_cloud = talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)
            evening_star = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            shooting_star = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)

            # Neutral patterns
            doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            spinning_top = talib.CDLSPINNINGTOP(open_prices, high_prices, low_prices, close_prices)

            # Process patterns from recent data
            recent_idx = -5  # Check last 5 candles

            pattern_checks = [
                (hammer[recent_idx:], "Hammer", "BUY", "Bullish reversal pattern at support"),
                (engulfing[recent_idx:], "Bullish Engulfing", "BUY", "Strong bullish reversal signal"),
                (morning_star[recent_idx:], "Morning Star", "BUY", "Three-candle bullish reversal"),
                (piercing[recent_idx:], "Piercing Pattern", "BUY", "Bullish reversal after downtrend"),
                (hanging_man[recent_idx:], "Hanging Man", "SELL", "Bearish reversal at resistance"),
                (dark_cloud[recent_idx:], "Dark Cloud Cover", "SELL", "Bearish reversal pattern"),
                (evening_star[recent_idx:], "Evening Star", "SELL", "Three-candle bearish reversal"),
                (shooting_star[recent_idx:], "Shooting Star", "SELL", "Bearish reversal at top"),
                (doji[recent_idx:], "Doji", "HOLD", "Indecision in the market"),
                (spinning_top[recent_idx:], "Spinning Top", "HOLD", "Market indecision pattern")
            ]

            for pattern_data, name, signal, desc in pattern_checks:
                if len(pattern_data) > 0 and any(pattern_data != 0):
                    # Calculate confidence based on pattern strength
                    max_val = max(abs(x) for x in pattern_data if x != 0)
                    confidence = min(abs(max_val) / 100.0, 1.0)

                    patterns.append(PatternResult(
                        pattern_name=name,
                        confidence=confidence,
                        signal=signal,
                        description=desc
                    ))

            return patterns

        except Exception as e:
            print(f"Error detecting candlestick patterns: {e}")
            return patterns

    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> List[PatternResult]:
        """Detect chart patterns like triangles, flags, head and shoulders"""
        patterns = []

        try:
            if len(df) < 50:
                return patterns

            close_prices = df['Close'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values

            # Head and Shoulders pattern detection
            patterns.extend(PatternRecognition._detect_head_and_shoulders(high_prices, close_prices))

            # Triangle patterns
            patterns.extend(PatternRecognition._detect_triangles(high_prices, low_prices, close_prices))

            # Flag and pennant patterns
            patterns.extend(PatternRecognition._detect_flags_pennants(close_prices))

            # Double top/bottom patterns
            patterns.extend(PatternRecognition._detect_double_patterns(high_prices, low_prices, close_prices))

            return patterns

        except Exception as e:
            print(f"Error detecting chart patterns: {e}")
            return patterns

    @staticmethod
    def _detect_head_and_shoulders(highs: np.ndarray, closes: np.ndarray) -> List[PatternResult]:
        """Detect head and shoulders pattern"""
        patterns = []

        try:
            if len(highs) < 20:
                return patterns

            # Find recent peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(highs[-50:], distance=5, prominence=highs[-50:].std())

            if len(peaks) >= 3:
                # Check for head and shoulders formation
                recent_peaks = peaks[-3:]
                peak_heights = highs[-50:][recent_peaks]

                # Head and shoulders: middle peak higher than shoulders
                if (len(peak_heights) == 3 and
                        peak_heights[1] > peak_heights[0] and
                        peak_heights[1] > peak_heights[2] and
                        abs(peak_heights[0] - peak_heights[2]) < peak_heights[1] * 0.05):
                    patterns.append(PatternResult(
                        pattern_name="Head and Shoulders",
                        confidence=0.75,
                        signal="SELL",
                        description="Bearish reversal pattern - potential trend change",
                        target_price=closes[-1] * 0.95,
                        stop_loss=closes[-1] * 1.02
                    ))

            return patterns

        except Exception as e:
            print(f"Error in head and shoulders detection: {e}")
            return patterns

    @staticmethod
    def _detect_triangles(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[PatternResult]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []

        try:
            if len(closes) < 30:
                return patterns

            recent_data = 30
            recent_highs = highs[-recent_data:]
            recent_lows = lows[-recent_data:]
            recent_closes = closes[-recent_data:]

            # Calculate trend lines
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]

            # Ascending triangle (flat resistance, rising support)
            if abs(high_slope) < recent_highs.std() * 0.01 and low_slope > 0:
                patterns.append(PatternResult(
                    pattern_name="Ascending Triangle",
                    confidence=0.7,
                    signal="BUY",
                    description="Bullish continuation pattern",
                    target_price=recent_closes[-1] * 1.05,
                    stop_loss=recent_closes[-1] * 0.97
                ))

            # Descending triangle (flat support, falling resistance)
            elif abs(low_slope) < recent_lows.std() * 0.01 and high_slope < 0:
                patterns.append(PatternResult(
                    pattern_name="Descending Triangle",
                    confidence=0.7,
                    signal="SELL",
                    description="Bearish continuation pattern",
                    target_price=recent_closes[-1] * 0.95,
                    stop_loss=recent_closes[-1] * 1.03
                ))

            # Symmetrical triangle (converging trend lines)
            elif high_slope < 0 and low_slope > 0 and abs(high_slope + low_slope) < 0.01:
                patterns.append(PatternResult(
                    pattern_name="Symmetrical Triangle",
                    confidence=0.6,
                    signal="HOLD",
                    description="Consolidation pattern - breakout expected",
                    target_price=None,
                    stop_loss=None
                ))

            return patterns

        except Exception as e:
            print(f"Error in triangle detection: {e}")
            return patterns

    @staticmethod
    def _detect_flags_pennants(closes: np.ndarray) -> List[PatternResult]:
        """Detect flag and pennant patterns"""
        patterns = []

        try:
            if len(closes) < 20:
                return patterns

            # Look for strong move followed by consolidation
            recent_closes = closes[-20:]

            # Check for strong initial move (pole)
            initial_move = (recent_closes[5] - recent_closes[0]) / recent_closes[0]

            if abs(initial_move) > 0.05:  # 5% move
                # Check for consolidation after the move
                consolidation_period = recent_closes[5:15]
                consolidation_range = (
                                                  consolidation_period.max() - consolidation_period.min()) / consolidation_period.mean()

                if consolidation_range < 0.03:  # Tight consolidation
                    if initial_move > 0:
                        patterns.append(PatternResult(
                            pattern_name="Bull Flag",
                            confidence=0.65,
                            signal="BUY",
                            description="Bullish continuation pattern after strong rally",
                            target_price=recent_closes[-1] * (1 + abs(initial_move)),
                            stop_loss=recent_closes[-1] * 0.98
                        ))
                    else:
                        patterns.append(PatternResult(
                            pattern_name="Bear Flag",
                            confidence=0.65,
                            signal="SELL",
                            description="Bearish continuation pattern after strong decline",
                            target_price=recent_closes[-1] * (1 - abs(initial_move)),
                            stop_loss=recent_closes[-1] * 1.02
                        ))

            return patterns

        except Exception as e:
            print(f"Error in flag/pennant detection: {e}")
            return patterns

    @staticmethod
    def _detect_double_patterns(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[PatternResult]:
        """Detect double top and double bottom patterns"""
        patterns = []

        try:
            if len(closes) < 40:
                return patterns

            from scipy.signal import find_peaks

            # Find peaks and troughs
            peaks, _ = find_peaks(highs[-40:], distance=10, prominence=highs[-40:].std())
            troughs, _ = find_peaks(-lows[-40:], distance=10, prominence=lows[-40:].std())

            # Double top detection
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                peak_heights = highs[-40:][last_two_peaks]

                if abs(peak_heights[0] - peak_heights[1]) < peak_heights.mean() * 0.02:
                    patterns.append(PatternResult(
                        pattern_name="Double Top",
                        confidence=0.7,
                        signal="SELL",
                        description="Bearish reversal pattern at resistance",
                        target_price=closes[-1] * 0.93,
                        stop_loss=closes[-1] * 1.03
                    ))

            # Double bottom detection
            if len(troughs) >= 2:
                last_two_troughs = troughs[-2:]
                trough_lows = lows[-40:][last_two_troughs]

                if abs(trough_lows[0] - trough_lows[1]) < trough_lows.mean() * 0.02:
                    patterns.append(PatternResult(
                        pattern_name="Double Bottom",
                        confidence=0.7,
                        signal="BUY",
                        description="Bullish reversal pattern at support",
                        target_price=closes[-1] * 1.07,
                        stop_loss=closes[-1] * 0.97
                    ))

            return patterns

        except Exception as e:
            print(f"Error in double pattern detection: {e}")
            return patterns


class SupportResistanceAnalysis:
    """Support and resistance level identification"""

    @staticmethod
    def find_support_resistance_levels(df: pd.DataFrame, lookback: int = 50) -> List[SupportResistance]:
        """Find key support and resistance levels"""
        levels = []

        try:
            if len(df) < lookback:
                return levels

            recent_data = df.tail(lookback).copy()
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            closes = recent_data['Close'].values
            current_price = closes[-1]

            from scipy.signal import find_peaks

            # Find resistance levels (peaks)
            resistance_peaks, peak_properties = find_peaks(
                highs,
                distance=5,
                prominence=np.std(highs) * 0.5
            )

            # Find support levels (troughs)
            support_troughs, trough_properties = find_peaks(
                -lows,
                distance=5,
                prominence=np.std(lows) * 0.5
            )

            # Process resistance levels
            for peak_idx in resistance_peaks:
                level_price = highs[peak_idx]

                # Count how many times price touched this level
                touch_count = SupportResistanceAnalysis._count_touches(
                    highs, level_price, tolerance=0.01
                )

                # Calculate strength based on touches and recency
                strength = min(5, max(1, touch_count))
                distance_from_price = abs(level_price - current_price) / current_price

                levels.append(SupportResistance(
                    level=level_price,
                    strength=strength,
                    type="resistance",
                    touch_count=touch_count,
                    last_touch=recent_data.index[peak_idx],
                    distance_from_price=distance_from_price
                ))

            # Process support levels
            for trough_idx in support_troughs:
                level_price = lows[trough_idx]

                touch_count = SupportResistanceAnalysis._count_touches(
                    lows, level_price, tolerance=0.01
                )

                strength = min(5, max(1, touch_count))
                distance_from_price = abs(level_price - current_price) / current_price

                levels.append(SupportResistance(
                    level=level_price,
                    strength=strength,
                    type="support",
                    touch_count=touch_count,
                    last_touch=recent_data.index[trough_idx],
                    distance_from_price=distance_from_price
                ))

            # Sort by strength and proximity to current price
            levels.sort(key=lambda x: (x.strength, -x.distance_from_price), reverse=True)

            # Return top 10 levels
            return levels[:10]

        except Exception as e:
            print(f"Error finding support/resistance levels: {e}")
            return levels

    @staticmethod
    def _count_touches(price_array: np.ndarray, level: float, tolerance: float = 0.01) -> int:
        """Count how many times price touched a specific level"""
        try:
            touch_threshold = level * tolerance
            touches = np.sum(np.abs(price_array - level) <= touch_threshold)
            return int(touches)
        except:
            return 1


class MarketRegimeAnalysis:
    """Market regime and trend analysis"""

    @staticmethod
    def classify_market_regime(df: pd.DataFrame, lookback: int = 30) -> MarketRegime:
        """Classify current market regime"""
        try:
            if len(df) < lookback:
                return MarketRegime("unknown", 0.0, 0, 0.0, "Insufficient data")

            recent_data = df.tail(lookback).copy()
            closes = recent_data['Close'].values
            highs = recent_data['High'].values
            lows = recent_data['Low'].values

            # Calculate trend strength
            trend_slope = np.polyfit(range(len(closes)), closes, 1)[0]
            trend_strength = abs(trend_slope) / np.mean(closes)

            # Calculate volatility
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            # Calculate ADX for trend strength
            try:
                adx = talib.ADX(highs, lows, closes, timeperiod=14)
                adx_current = adx[-1] if not np.isnan(adx[-1]) else 25
            except:
                adx_current = 25

            # Determine regime type
            if adx_current > 40 and trend_slope > 0:
                regime_type = "trending_up"
                description = "Strong uptrend with high momentum"
                strength = min(1.0, adx_current / 60)

            elif adx_current > 40 and trend_slope < 0:
                regime_type = "trending_down"
                description = "Strong downtrend with high momentum"
                strength = min(1.0, adx_current / 60)

            elif volatility > 0.3:
                regime_type = "volatile"
                description = "High volatility, choppy price action"
                strength = min(1.0, volatility / 0.5)

            elif adx_current < 25:
                regime_type = "sideways"
                description = "Range-bound market, low trend strength"
                strength = 1.0 - (adx_current / 25)

            else:
                regime_type = "transitional"
                description = "Market in transition between regimes"
                strength = 0.5

            return MarketRegime(
                regime_type=regime_type,
                strength=strength,
                duration_days=lookback,
                volatility=volatility,
                description=description
            )

        except Exception as e:
            print(f"Error classifying market regime: {e}")
            return MarketRegime("unknown", 0.0, 0, 0.0, "Analysis error")


class VolumeAnalysis:
    """Advanced volume analysis techniques"""

    @staticmethod
    def analyze_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
        """Create volume profile analysis"""
        try:
            if len(df) < 20:
                return {}

            # Calculate price bins
            price_min = df['Low'].min()
            price_max = df['High'].max()
            price_bins = np.linspace(price_min, price_max, bins + 1)

            # Calculate volume at each price level
            volume_profile = np.zeros(bins)

            for _, row in df.iterrows():
                # Find which bin this candle's range covers
                low_bin = np.digitize(row['Low'], price_bins) - 1
                high_bin = np.digitize(row['High'], price_bins) - 1

                # Distribute volume across the price range
                bins_covered = max(1, high_bin - low_bin + 1)
                volume_per_bin = row['Volume'] / bins_covered

                for bin_idx in range(max(0, low_bin), min(bins, high_bin + 1)):
                    volume_profile[bin_idx] += volume_per_bin

            # Find Point of Control (POC) - highest volume price level
            poc_bin = np.argmax(volume_profile)
            poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2

            # Find Value Area (70% of volume)
            total_volume = np.sum(volume_profile)
            value_area_volume = total_volume * 0.7

            # Find value area high and low
            sorted_bins = np.argsort(volume_profile)[::-1]
            cumulative_volume = 0
            value_area_bins = []

            for bin_idx in sorted_bins:
                cumulative_volume += volume_profile[bin_idx]
                value_area_bins.append(bin_idx)
                if cumulative_volume >= value_area_volume:
                    break

            value_area_low = price_bins[min(value_area_bins)]
            value_area_high = price_bins[max(value_area_bins) + 1]

            return {
                "poc_price": poc_price,
                "value_area_low": value_area_low,
                "value_area_high": value_area_high,
                "volume_profile": volume_profile.tolist(),
                "price_bins": price_bins.tolist(),
                "total_volume": total_volume
            }

        except Exception as e:
            print(f"Error in volume profile analysis: {e}")
            return {}

    @staticmethod
    def detect_volume_anomalies(df: pd.DataFrame, lookback: int = 20) -> List[Dict]:
        """Detect volume anomalies and their implications"""
        anomalies = []

        try:
            if len(df) < lookback * 2:
                return anomalies

            recent_data = df.tail(lookback).copy()
            volumes = recent_data['Volume'].values
            closes = recent_data['Close'].values

            # Calculate volume moving average and standard deviation
            volume_ma = np.mean(volumes[:-5])  # Exclude last 5 days from baseline
            volume_std = np.std(volumes[:-5])

            # Check recent volume spikes
            for i in range(-5, 0):  # Check last 5 days
                current_volume = volumes[i]
                current_close = closes[i]
                previous_close = closes[i - 1] if i > -len(closes) else closes[i]

                price_change = (current_close - previous_close) / previous_close * 100

                # Volume spike detection
                if current_volume > volume_ma + (2 * volume_std):
                    spike_magnitude = (current_volume - volume_ma) / volume_ma

                    anomaly_type = "accumulation" if price_change > 0 else "distribution" if price_change < 0 else "neutral"

                    anomalies.append({
                        "date": recent_data.index[i],
                        "type": f"volume_spike_{anomaly_type}",
                        "magnitude": spike_magnitude,
                        "volume": current_volume,
                        "price_change": price_change,
                        "significance": "high" if spike_magnitude > 1.0 else "medium"
                    })

                # Volume dry-up detection
                elif current_volume < volume_ma - (1.5 * volume_std):
                    dryup_magnitude = (volume_ma - current_volume) / volume_ma

                    anomalies.append({
                        "date": recent_data.index[i],
                        "type": "volume_dryup",
                        "magnitude": dryup_magnitude,
                        "volume": current_volume,
                        "price_change": price_change,
                        "significance": "medium" if abs(price_change) < 1 else "high"
                    })

            return anomalies

        except Exception as e:
            print(f"Error detecting volume anomalies: {e}")
            return anomalies


class SentimentAnalysis:
    """Market sentiment analysis using price action and indicators"""

    @staticmethod
    def calculate_sentiment_score(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall market sentiment score"""
        try:
            if len(df) < 20:
                return {"sentiment_score": 0.5, "sentiment_label": "Neutral"}

            sentiment_factors = []

            # 1. Price momentum (20%)
            closes = df['Close'].values
            price_momentum = (closes[-1] - closes[-10]) / closes[-10]
            momentum_score = max(0, min(1, 0.5 + price_momentum * 5))
            sentiment_factors.append(("price_momentum", momentum_score, 0.20))

            # 2. Volume trend (15%)
            volumes = df['Volume'].values
            volume_trend = np.polyfit(range(len(volumes[-10:])), volumes[-10:], 1)[0]
            volume_score = 0.5 + np.tanh(volume_trend / np.mean(volumes)) * 0.5
            sentiment_factors.append(("volume_trend", volume_score, 0.15))

            # 3. RSI sentiment (15%)
            rsi = talib.RSI(closes, timeperiod=14)
            rsi_current = rsi[-1] if not np.isnan(rsi[-1]) else 50
            rsi_score = rsi_current / 100
            sentiment_factors.append(("rsi_sentiment", rsi_score, 0.15))

            # 4. MACD sentiment (15%)
            macd, macd_signal, _ = talib.MACD(closes)
            if not (np.isnan(macd[-1]) or np.isnan(macd_signal[-1])):
                macd_score = 0.6 if macd[-1] > macd_signal[-1] else 0.4
            else:
                macd_score = 0.5
            sentiment_factors.append(("macd_sentiment", macd_score, 0.15))

            # 5. Bollinger Bands position (10%)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(closes)
            if not np.isnan(bb_upper[-1]):
                bb_position = (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                bb_score = max(0, min(1, bb_position))
            else:
                bb_score = 0.5
            sentiment_factors.append(("bollinger_position", bb_score, 0.10))

            # 6. Volatility sentiment (10%)
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns[-10:])
            # Lower volatility = more positive sentiment
            vol_score = max(0, min(1, 1 - volatility * 20))
            sentiment_factors.append(("volatility_sentiment", vol_score, 0.10))

            # 7. Support/Resistance proximity (15%)
            current_price = closes[-1]
            recent_high = np.max(closes[-20:])
            recent_low = np.min(closes[-20:])

            if recent_high != recent_low:
                price_position = (current_price - recent_low) / (recent_high - recent_low)
                sr_score = price_position
            else:
                sr_score = 0.5
            sentiment_factors.append(("support_resistance", sr_score, 0.15))

            # Calculate weighted sentiment score
            total_score = sum(score * weight for _, score, weight in sentiment_factors)

            # Determine sentiment label
            if total_score >= 0.7:
                sentiment_label = "Very Bullish"
            elif total_score >= 0.6:
                sentiment_label = "Bullish"
            elif total_score >= 0.4:
                sentiment_label = "Neutral"
            elif total_score >= 0.3:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Very Bearish"

            return {
                "sentiment_score": total_score,
                "sentiment_label": sentiment_label,
                "factors": {name: {"score": score, "weight": weight}
                            for name, score, weight in sentiment_factors},
                "confidence": min(1.0, len(df) / 50)  # Confidence based on data availability
            }

        except Exception as e:
            print(f"Error calculating sentiment score: {e}")
            return {"sentiment_score": 0.5, "sentiment_label": "Neutral", "error": str(e)}


class RiskMetrics:
    """Risk assessment and metrics calculation"""

    @staticmethod
    def calculate_risk_metrics(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            if len(df) < 30:
                return {}

            closes = df['Close'].values
            returns = np.diff(closes) / closes[:-1]

            # Basic risk metrics
            volatility = np.std(returns) * np.sqrt(252)
            downside_deviation = np.std([r for r in returns if r < 0]) * np.sqrt(252)

            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)

            # Expected Shortfall (CVaR)
            cvar_95 = np.mean([r for r in returns if r <= var_95])
            cvar_99 = np.mean([r for r in returns if r <= var_99])

            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)

            # Sharpe Ratio (assuming risk-free rate = 6%)
            risk_free_rate = 0.06
            excess_returns = np.mean(returns) * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0

            # Sortino Ratio
            sortino_ratio = excess_returns / downside_deviation if downside_deviation > 0 else 0

            # Calmar Ratio
            annual_return = np.mean(returns) * 252
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

            risk_metrics = {
                "volatility": volatility,
                "downside_deviation": downside_deviation,
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "cvar_99": cvar_99,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "annual_return": annual_return
            }

            # Beta calculation if benchmark provided
            if benchmark_df is not None and len(benchmark_df) >= len(df):
                benchmark_closes = benchmark_df['Close'].values[-len(closes):]
                benchmark_returns = np.diff(benchmark_closes) / benchmark_closes[:-1]

                if len(benchmark_returns) == len(returns):
                    covariance = np.cov(returns, benchmark_returns)[0, 1]
                    benchmark_variance = np.var(benchmark_returns)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

                    # Alpha calculation
                    benchmark_annual_return = np.mean(benchmark_returns) * 252
                    alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))

                    risk_metrics.update({
                        "beta": beta,
                        "alpha": alpha,
                        "correlation": np.corrcoef(returns, benchmark_returns)[0, 1]
                    })

            return risk_metrics

        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {}

    @staticmethod
    def calculate_position_sizing(df: pd.DataFrame, risk_percent: float = 2.0,
                                  stop_loss_percent: float = 3.0) -> Dict[str, float]:
        """Calculate optimal position sizing based on risk management"""
        try:
            current_price = df['Close'].iloc[-1]

            # Calculate ATR for dynamic stop loss
            atr = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else current_price * 0.02

            # Dynamic stop loss based on ATR
            atr_stop_loss = 2 * current_atr
            percent_stop_loss = current_price * (stop_loss_percent / 100)

            # Use the wider of the two stop losses
            stop_loss_amount = max(atr_stop_loss, percent_stop_loss)
            stop_loss_price = current_price - stop_loss_amount

            # Position sizing calculations
            risk_per_share = stop_loss_amount

            # Assuming portfolio value (this should come from actual portfolio data)
            portfolio_value = 1000000  # 10 lakhs default
            risk_amount = portfolio_value * (risk_percent / 100)

            # Calculate position size
            position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
            max_investment = position_size * current_price

            return {
                "position_size_shares": int(position_size),
                "max_investment_amount": max_investment,
                "stop_loss_price": stop_loss_price,
                "risk_per_share": risk_per_share,
                "risk_amount": risk_amount,
                "risk_reward_ratio": 3.0,  # Default 1:3
                "target_price": current_price + (risk_per_share * 3),
                "atr_value": current_atr
            }

        except Exception as e:
            print(f"Error calculating position sizing: {e}")
            return {}


# Utility functions for integration with main analytics

def perform_comprehensive_analysis(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Perform comprehensive technical analysis"""
    try:
        analysis_results = {
            "symbol": symbol,
            "analysis_timestamp": datetime.now().isoformat(),
            "data_points": len(df)
        }

        # Pattern recognition
        candlestick_patterns = PatternRecognition.detect_candlestick_patterns(df)
        chart_patterns = PatternRecognition.detect_chart_patterns(df)

        analysis_results["patterns"] = {
            "candlestick": [p.__dict__ for p in candlestick_patterns],
            "chart": [p.__dict__ for p in chart_patterns]
        }

        # Support and resistance
        sr_levels = SupportResistanceAnalysis.find_support_resistance_levels(df)
        analysis_results["support_resistance"] = [sr.__dict__ for sr in sr_levels]

        # Market regime
        market_regime = MarketRegimeAnalysis.classify_market_regime(df)
        analysis_results["market_regime"] = market_regime.__dict__

        # Volume analysis
        volume_profile = VolumeAnalysis.analyze_volume_profile(df)
        volume_anomalies = VolumeAnalysis.detect_volume_anomalies(df)

        analysis_results["volume_analysis"] = {
            "profile": volume_profile,
            "anomalies": volume_anomalies
        }

        # Sentiment analysis
        sentiment = SentimentAnalysis.calculate_sentiment_score(df)
        analysis_results["sentiment"] = sentiment

        # Risk metrics
        risk_metrics = RiskMetrics.calculate_risk_metrics(df)
        position_sizing = RiskMetrics.calculate_position_sizing(df)

        analysis_results["risk_analysis"] = {
            "metrics": risk_metrics,
            "position_sizing": position_sizing
        }

        return analysis_results

    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
            "analysis_timestamp": datetime.now().isoformat()
        }


def generate_trading_signals(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate trading signals based on comprehensive analysis"""
    signals = []

    try:
        # Pattern-based signals
        if "patterns" in analysis:
            for pattern in analysis["patterns"]["candlestick"]:
                if pattern["confidence"] > 0.6:
                    signals.append({
                        "type": "pattern",
                        "signal": pattern["signal"],
                        "source": pattern["pattern_name"],
                        "confidence": pattern["confidence"],
                        "timeframe": "short_term"
                    })

            for pattern in analysis["patterns"]["chart"]:
                if pattern["confidence"] > 0.6:
                    signals.append({
                        "type": "pattern",
                        "signal": pattern["signal"],
                        "source": pattern["pattern_name"],
                        "confidence": pattern["confidence"],
                        "timeframe": "medium_term"
                    })

        # Sentiment-based signals
        if "sentiment" in analysis:
            sentiment_score = analysis["sentiment"]["sentiment_score"]
            if sentiment_score > 0.7:
                signals.append({
                    "type": "sentiment",
                    "signal": "BUY",
                    "source": "Market Sentiment",
                    "confidence": sentiment_score,
                    "timeframe": "short_term"
                })
            elif sentiment_score < 0.3:
                signals.append({
                    "type": "sentiment",
                    "signal": "SELL",
                    "source": "Market Sentiment",
                    "confidence": 1 - sentiment_score,
                    "timeframe": "short_term"
                })

        # Volume-based signals
        if "volume_analysis" in analysis and "anomalies" in analysis["volume_analysis"]:
            for anomaly in analysis["volume_analysis"]["anomalies"]:
                if anomaly["significance"] == "high":
                    signal_type = "BUY" if "accumulation" in anomaly["type"] else "SELL" if "distribution" in anomaly[
                        "type"] else "HOLD"
                    signals.append({
                        "type": "volume",
                        "signal": signal_type,
                        "source": "Volume Anomaly",
                        "confidence": min(0.8, anomaly["magnitude"]),
                        "timeframe": "short_term"
                    })

        # Market regime signals
        if "market_regime" in analysis:
            regime = analysis["market_regime"]
            if regime["regime_type"] == "trending_up" and regime["strength"] > 0.7:
                signals.append({
                    "type": "trend",
                    "signal": "BUY",
                    "source": "Strong Uptrend",
                    "confidence": regime["strength"],
                    "timeframe": "long_term"
                })
            elif regime["regime_type"] == "trending_down" and regime["strength"] > 0.7:
                signals.append({
                    "type": "trend",
                    "signal": "SELL",
                    "source": "Strong Downtrend",
                    "confidence": regime["strength"],
                    "timeframe": "long_term"
                })

        # Consolidate and rank signals
        signals.sort(key=lambda x: x["confidence"], reverse=True)

        return signals[:10]  # Return top 10 signals

    except Exception as e:
        print(f"Error generating trading signals: {e}")
        return signals