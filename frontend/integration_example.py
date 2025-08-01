"""
Analytics Integration Example - integration_example.py
Shows how to integrate enhanced analytics with existing trading app
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import talib

# Import your existing modules
from analytics import render_analytics_page, chart_state, update_chart_and_metrics
from analytics_config import get_config, AnalyticsConfig
from market_analysis import (
    perform_comprehensive_analysis,
    generate_trading_signals,
    PatternRecognition,
    SupportResistanceAnalysis,
    MarketRegimeAnalysis,
    VolumeAnalysis,
    SentimentAnalysis,
    RiskMetrics
)

logger = logging.getLogger(__name__)


class EnhancedAnalyticsManager:
    """Main class to manage enhanced analytics features"""

    def __init__(self, config: AnalyticsConfig = None):
        self.config = config or get_config()
        self.cache = {}
        self.analysis_results = {}

    async def initialize(self):
        """Initialize the analytics manager"""
        try:
            logger.info("Initializing Enhanced Analytics Manager")

            # Validate configuration
            if not self.config:
                raise ValueError("Configuration not loaded")

            # Initialize data sources
            await self._initialize_data_sources()

            # Setup caching
            self._setup_cache()

            logger.info("✓ Enhanced Analytics Manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize analytics manager: {e}")
            return False

    async def _initialize_data_sources(self):
        """Initialize data source connections"""
        try:
            # Test primary data source
            if self.config.PRIMARY_DATA_SOURCE.value == "yfinance":
                import yfinance as yf
                # Test connection with a sample symbol
                test_ticker = yf.Ticker("RELIANCE.NS")
                test_data = test_ticker.history(period="5d")
                if test_data.empty:
                    logger.warning("YFinance connection test failed")
                else:
                    logger.info("✓ YFinance data source connected")

            # Initialize fallback sources
            for source in self.config.FALLBACK_DATA_SOURCES:
                logger.info(f"Fallback source available: {source.value}")

        except Exception as e:
            logger.error(f"Error initializing data sources: {e}")

    def _setup_cache(self):
        """Setup caching mechanism"""
        try:
            self.cache = {
                "data_cache": {},
                "analysis_cache": {},
                "pattern_cache": {},
                "last_update": {}
            }
            logger.info("✓ Cache initialized")
        except Exception as e:
            logger.error(f"Error setting up cache: {e}")

    async def get_enhanced_stock_analysis(self, symbol: str, timeframe: str = "1d") -> Dict:
        """Get comprehensive stock analysis with caching"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            current_time = datetime.now()

            # Check cache validity
            if (cache_key in self.cache["analysis_cache"] and
                    cache_key in self.cache["last_update"]):

                last_update = self.cache["last_update"][cache_key]
                cache_timeout = timedelta(seconds=self.config.PERFORMANCE_CONFIG["cache_timeout"])

                if current_time - last_update < cache_timeout:
                    logger.info(f"Returning cached analysis for {symbol}")
                    return self.cache["analysis_cache"][cache_key]

            # Fetch fresh data
            logger.info(f"Fetching fresh analysis for {symbol}")
            df = await self._fetch_stock_data(symbol, timeframe)

            if df.empty:
                return {"error": f"No data available for {symbol}"}

            # Perform comprehensive analysis
            analysis = perform_comprehensive_analysis(df, symbol)

            # Generate trading signals
            signals = generate_trading_signals(analysis)
            analysis["trading_signals"] = signals

            # Calculate additional metrics
            analysis["enhanced_metrics"] = await self._calculate_enhanced_metrics(df, symbol)

            # Cache results
            self.cache["analysis_cache"][cache_key] = analysis
            self.cache["last_update"][cache_key] = current_time

            return analysis

        except Exception as e:
            logger.error(f"Error getting enhanced analysis for {symbol}: {e}")
            return {"error": str(e)}

    async def _fetch_stock_data(self, symbol: str, timeframe: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock data with fallback sources"""
        try:
            # Try primary data source first
            if self.config.PRIMARY_DATA_SOURCE.value == "yfinance":
                import yfinance as yf

                yf_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period=period, interval=self._convert_timeframe(timeframe))

                if not df.empty:
                    return self._standardize_dataframe(df)

            # Try fallback sources
            for source in self.config.FALLBACK_DATA_SOURCES:
                try:
                    if source.value == "nsepython":
                        df = await self._fetch_from_nsepython(symbol, period)
                        if not df.empty:
                            return self._standardize_dataframe(df)

                except Exception as e:
                    logger.warning(f"Fallback source {source.value} failed: {e}")
                    continue

            logger.error(f"All data sources failed for {symbol}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert internal timeframe to yfinance format"""
        timeframe_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "1d": "1d", "1w": "1wk", "1M": "1mo"
        }
        return timeframe_map.get(timeframe, "1d")

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dataframe columns"""
        try:
            # Ensure standard column names
            column_mapping = {
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume"
            }

            # Rename columns if needed
            df = df.rename(columns=column_mapping)

            # Remove any timezone info for consistency
            if hasattr(df.index, 'tz_localize'):
                df.index = df.index.tz_localize(None)

            return df

        except Exception as e:
            logger.error(f"Error standardizing dataframe: {e}")
            return df

    async def _fetch_from_nsepython(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from NSE Python (fallback)"""
        try:
            from nsepython import nsefetch
            # Implementation depends on nsepython API
            # This is a placeholder - implement based on actual nsepython usage
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching from nsepython: {e}")
            return pd.DataFrame()

    async def _calculate_enhanced_metrics(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate additional enhanced metrics"""
        try:
            enhanced_metrics = {}

            # Market strength indicators
            enhanced_metrics["market_strength"] = self._calculate_market_strength(df)

            # Trend quality metrics
            enhanced_metrics["trend_quality"] = self._calculate_trend_quality(df)

            # Volatility analysis
            enhanced_metrics["volatility_analysis"] = self._calculate_volatility_metrics(df)

            # Momentum indicators
            enhanced_metrics["momentum_analysis"] = self._calculate_momentum_indicators(df)

            # Risk-adjusted returns
            enhanced_metrics["risk_adjusted_metrics"] = self._calculate_risk_adjusted_returns(df)

            return enhanced_metrics

        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {e}")
            return {}

    def _calculate_market_strength(self, df: pd.DataFrame) -> Dict:
        """Calculate market strength indicators"""
        try:
            closes = df['Close'].values
            volumes = df['Volume'].values

            # Price-Volume Trend
            pvt = []
            for i in range(1, len(closes)):
                change_ratio = (closes[i] - closes[i - 1]) / closes[i - 1]
                pvt_value = volumes[i] * change_ratio
                pvt.append(pvt_value)

            pvt_trend = sum(pvt[-10:]) / len(pvt[-10:]) if len(pvt) >= 10 else 0

            # Accumulation/Distribution Line
            ad_line = []
            for i in range(len(df)):
                if df['High'].iloc[i] != df['Low'].iloc[i]:
                    clv = ((df['Close'].iloc[i] - df['Low'].iloc[i]) -
                           (df['High'].iloc[i] - df['Close'].iloc[i])) / (df['High'].iloc[i] - df['Low'].iloc[i])
                else:
                    clv = 0
                ad_value = clv * df['Volume'].iloc[i]
                ad_line.append(ad_value)

            ad_trend = sum(ad_line[-10:]) / len(ad_line[-10:]) if len(ad_line) >= 10 else 0

            return {
                "pvt_trend": pvt_trend,
                "ad_trend": ad_trend,
                "strength_score": (pvt_trend + ad_trend) / 2,
                "interpretation": "Strong" if (pvt_trend + ad_trend) > 0 else "Weak"
            }

        except Exception as e:
            logger.error(f"Error calculating market strength: {e}")
            return {}

    def _calculate_trend_quality(self, df: pd.DataFrame) -> Dict:
        """Calculate trend quality metrics"""
        try:
            closes = df['Close'].values

            # R-squared of linear regression (trend strength)
            x = np.arange(len(closes))
            slope, intercept = np.polyfit(x, closes, 1)
            predicted = slope * x + intercept

            ss_res = np.sum((closes - predicted) ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Trend consistency (percentage of periods following trend)
            trend_direction = 1 if slope > 0 else -1
            consistent_periods = 0

            for i in range(1, len(closes)):
                period_direction = 1 if closes[i] > closes[i - 1] else -1
                if period_direction == trend_direction:
                    consistent_periods += 1

            consistency = consistent_periods / (len(closes) - 1) if len(closes) > 1 else 0

            return {
                "trend_strength": r_squared,
                "trend_consistency": consistency,
                "trend_direction": "Up" if slope > 0 else "Down",
                "quality_score": (r_squared + consistency) / 2,
                "interpretation": self._interpret_trend_quality((r_squared + consistency) / 2)
            }

        except Exception as e:
            logger.error(f"Error calculating trend quality: {e}")
            return {}

    def _interpret_trend_quality(self, score: float) -> str:
        """Interpret trend quality score"""
        if score >= 0.8:
            return "Excellent trend quality"
        elif score >= 0.6:
            return "Good trend quality"
        elif score >= 0.4:
            return "Moderate trend quality"
        else:
            return "Poor trend quality"

    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive volatility metrics"""
        try:
            closes = df['Close'].values
            highs = df['High'].values
            lows = df['Low'].values

            # Historical Volatility
            returns = np.diff(closes) / closes[:-1]
            hist_vol = np.std(returns) * np.sqrt(252)  # Annualized

            # Parkinson Volatility (High-Low)
            parkinson_vol = np.sqrt(np.mean(np.log(highs / lows) ** 2) / (4 * np.log(2))) * np.sqrt(252)

            # Garman-Klass Volatility
            try:
                opens = df['Open'].values
                gk_vol = np.sqrt(np.mean(
                    np.log(highs / closes) * np.log(highs / opens) +
                    np.log(lows / closes) * np.log(lows / opens)
                )) * np.sqrt(252)
            except:
                gk_vol = hist_vol

            # Volatility trend
            recent_vol = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else hist_vol
            vol_trend = "Increasing" if recent_vol > hist_vol else "Decreasing"

            return {
                "historical_volatility": hist_vol,
                "parkinson_volatility": parkinson_vol,
                "garman_klass_volatility": gk_vol,
                "recent_volatility": recent_vol,
                "volatility_trend": vol_trend,
                "volatility_regime": self._classify_volatility_regime(hist_vol)
            }

        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {}

    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility < 0.15:
            return "Low Volatility"
        elif volatility < 0.25:
            return "Normal Volatility"
        elif volatility < 0.35:
            return "High Volatility"
        else:
            return "Extreme Volatility"

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive momentum indicators"""
        try:
            closes = df['Close'].values

            # Rate of Change (ROC) for different periods
            roc_5 = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
            roc_10 = (closes[-1] / closes[-11] - 1) * 100 if len(closes) >= 11 else 0
            roc_20 = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0

            # Price Rate of Change Oscillator
            try:
                proc = talib.ROCP(closes, timeperiod=10)
                current_proc = proc[-1] if not np.isnan(proc[-1]) else 0
            except:
                current_proc = 0

            # Momentum (10-period)
            try:
                momentum = talib.MOM(closes, timeperiod=10)
                current_momentum = momentum[-1] if not np.isnan(momentum[-1]) else 0
            except:
                current_momentum = 0

            # Ultimate Oscillator
            try:
                ult_osc = talib.ULTOSC(df['High'].values, df['Low'].values, closes)
                current_ult_osc = ult_osc[-1] if not np.isnan(ult_osc[-1]) else 50
            except:
                current_ult_osc = 50

            # Momentum score (composite)
            momentum_score = (
                                     (roc_10 / 10) +  # Normalize ROC
                                     (current_proc * 100) +  # Scale PROC
                                     (current_momentum / closes[-1] * 100) +  # Normalize momentum
                                     ((current_ult_osc - 50) / 50)  # Center Ultimate Oscillator
                             ) / 4

            return {
                "roc_5_day": roc_5,
                "roc_10_day": roc_10,
                "roc_20_day": roc_20,
                "price_roc": current_proc,
                "momentum_10": current_momentum,
                "ultimate_oscillator": current_ult_osc,
                "composite_momentum_score": momentum_score,
                "momentum_interpretation": self._interpret_momentum(momentum_score)
            }

        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {}

    def _interpret_momentum(self, score: float) -> str:
        """Interpret momentum score"""
        if score > 2:
            return "Very Strong Positive Momentum"
        elif score > 1:
            return "Strong Positive Momentum"
        elif score > 0:
            return "Positive Momentum"
        elif score > -1:
            return "Negative Momentum"
        elif score > -2:
            return "Strong Negative Momentum"
        else:
            return "Very Strong Negative Momentum"

    def _calculate_risk_adjusted_returns(self, df: pd.DataFrame) -> Dict:
        """Calculate risk-adjusted return metrics"""
        try:
            closes = df['Close'].values
            returns = np.diff(closes) / closes[:-1]

            if len(returns) < 30:
                return {}

            # Basic metrics
            annual_return = np.mean(returns) * 252
            annual_vol = np.std(returns) * np.sqrt(252)

            # Information Ratio (assuming benchmark return of 12%)
            benchmark_return = 0.12
            excess_return = annual_return - benchmark_return
            tracking_error = annual_vol  # Simplified
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

            # Treynor Ratio (assuming beta of 1 for simplicity)
            risk_free_rate = 0.06
            beta = 1.0  # Simplified - should be calculated against market index
            treynor_ratio = (annual_return - risk_free_rate) / beta

            # Omega Ratio (gain/loss ratio above risk-free rate)
            threshold = risk_free_rate / 252  # Daily risk-free rate
            gains = sum([r - threshold for r in returns if r > threshold])
            losses = abs(sum([threshold - r for r in returns if r < threshold]))
            omega_ratio = gains / losses if losses > 0 else float('inf')

            return {
                "annual_return": annual_return * 100,
                "annual_volatility": annual_vol * 100,
                "information_ratio": information_ratio,
                "treynor_ratio": treynor_ratio,
                "omega_ratio": min(omega_ratio, 10),  # Cap for display
                "risk_adjusted_score": self._calculate_risk_adjusted_score(
                    annual_return, annual_vol, information_ratio
                )
            }

        except Exception as e:
            logger.error(f"Error calculating risk-adjusted returns: {e}")
            return {}

    def _calculate_risk_adjusted_score(self, ret: float, vol: float, info_ratio: float) -> float:
        """Calculate composite risk-adjusted score"""
        try:
            # Normalize components
            return_score = min(max(ret * 5, -1), 1)  # Scale return
            vol_penalty = min(vol * 2, 1)  # Penalize high volatility
            info_score = min(max(info_ratio, -1), 1)  # Information ratio component

            # Composite score
            score = (return_score - vol_penalty + info_score) / 2
            return max(-1, min(1, score))  # Bound between -1 and 1

        except Exception as e:
            return 0

    async def run_stock_screener(self, criteria: Dict) -> List[Dict]:
        """Run stock screener with given criteria"""
        try:
            logger.info("Running stock screener with criteria")

            symbols = self.config.DEFAULT_SYMBOLS
            screener_results = []

            for symbol in symbols[:10]:  # Limit for demo
                try:
                    # Get analysis for each symbol
                    analysis = await self.get_enhanced_stock_analysis(symbol)

                    if "error" in analysis:
                        continue

                    # Apply screening criteria
                    if self._meets_screening_criteria(analysis, criteria):
                        screener_results.append({
                            "symbol": symbol,
                            "analysis": analysis,
                            "score": self._calculate_screening_score(analysis, criteria)
                        })

                except Exception as e:
                    logger.warning(f"Error screening {symbol}: {e}")
                    continue

            # Sort by score
            screener_results.sort(key=lambda x: x["score"], reverse=True)

            return screener_results[:20]  # Return top 20

        except Exception as e:
            logger.error(f"Error running stock screener: {e}")
            return []

    def _meets_screening_criteria(self, analysis: Dict, criteria: Dict) -> bool:
        """Check if analysis meets screening criteria"""
        try:
            # Example criteria checking
            if "sentiment" in analysis and "min_sentiment" in criteria:
                if analysis["sentiment"]["sentiment_score"] < criteria["min_sentiment"]:
                    return False

            if "risk_analysis" in analysis and "max_volatility" in criteria:
                risk_metrics = analysis["risk_analysis"].get("metrics", {})
                if risk_metrics.get("volatility", 0) > criteria["max_volatility"]:
                    return False

            # Add more criteria as needed
            return True

        except Exception as e:
            logger.error(f"Error checking screening criteria: {e}")
            return False

    def _calculate_screening_score(self, analysis: Dict, criteria: Dict) -> float:
        """Calculate screening score for ranking"""
        try:
            score = 0

            # Sentiment component
            if "sentiment" in analysis:
                score += analysis["sentiment"]["sentiment_score"] * 30

            # Pattern component
            if "patterns" in analysis:
                pattern_score = 0
                for pattern in analysis["patterns"]["candlestick"]:
                    if pattern["signal"] == "BUY":
                        pattern_score += pattern["confidence"] * 10
                score += min(pattern_score, 20)

            # Trading signals component
            if "trading_signals" in analysis:
                signal_score = 0
                for signal in analysis["trading_signals"]:
                    if signal["signal"] == "BUY":
                        signal_score += signal["confidence"] * 10
                score += min(signal_score, 30)

            # Risk-adjusted component
            if "enhanced_metrics" in analysis:
                risk_metrics = analysis["enhanced_metrics"].get("risk_adjusted_metrics", {})
                risk_score = risk_metrics.get("risk_adjusted_score", 0)
                score += risk_score * 20

            return max(0, min(100, score))  # Bound between 0-100

        except Exception as e:
            logger.error(f"Error calculating screening score: {e}")
            return 0

    async def generate_market_report(self, symbols: List[str]) -> Dict:
        """Generate comprehensive market report"""
        try:
            logger.info(f"Generating market report for {len(symbols)} symbols")

            report = {
                "report_date": datetime.now().isoformat(),
                "symbols_analyzed": len(symbols),
                "market_overview": {},
                "symbol_analysis": {},
                "sector_analysis": {},
                "risk_assessment": {},
                "recommendations": []
            }

            all_analyses = []

            # Analyze each symbol
            for symbol in symbols:
                analysis = await self.get_enhanced_stock_analysis(symbol)
                if "error" not in analysis:
                    all_analyses.append(analysis)
                    report["symbol_analysis"][symbol] = analysis

            # Generate market overview
            report["market_overview"] = self._generate_market_overview(all_analyses)

            # Generate recommendations
            report["recommendations"] = self._generate_market_recommendations(all_analyses)

            return report

        except Exception as e:
            logger.error(f"Error generating market report: {e}")
            return {"error": str(e)}

    def _generate_market_overview(self, analyses: List[Dict]) -> Dict:
        """Generate market overview from analyses"""
        try:
            if not analyses:
                return {}

            # Aggregate sentiment
            sentiments = []
            for analysis in analyses:
                if "sentiment" in analysis:
                    sentiments.append(analysis["sentiment"]["sentiment_score"])

            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5

            # Aggregate signals
            buy_signals = sell_signals = hold_signals = 0
            for analysis in analyses:
                if "trading_signals" in analysis:
                    for signal in analysis["trading_signals"]:
                        if signal["signal"] == "BUY":
                            buy_signals += 1
                        elif signal["signal"] == "SELL":
                            sell_signals += 1
                        else:
                            hold_signals += 1

            return {
                "average_market_sentiment": avg_sentiment,
                "sentiment_interpretation": self._interpret_sentiment(avg_sentiment),
                "signal_distribution": {
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                    "hold_signals": hold_signals
                },
                "market_bias": "Bullish" if buy_signals > sell_signals else "Bearish" if sell_signals > buy_signals else "Neutral"
            }

        except Exception as e:
            logger.error(f"Error generating market overview: {e}")
            return {}

    def _interpret_sentiment(self, sentiment: float) -> str:
        """Interpret sentiment score"""
        if sentiment >= 0.7:
            return "Very Bullish"
        elif sentiment >= 0.6:
            return "Bullish"
        elif sentiment >= 0.4:
            return "Neutral"
        elif sentiment >= 0.3:
            return "Bearish"
        else:
            return "Very Bearish"

    def _generate_market_recommendations(self, analyses: List[Dict]) -> List[str]:
        """Generate market recommendations"""
        recommendations = []

        try:
            if not analyses:
                return recommendations

            # Top performers
            top_performers = []
            for analysis in analyses:
                if "trading_signals" in analysis:
                    buy_signals = [s for s in analysis["trading_signals"] if s["signal"] == "BUY"]
                    if len(buy_signals) >= 2:  # Multiple buy signals
                        top_performers.append(analysis["symbol"])

            if top_performers:
                recommendations.append(f"Top buy candidates: {', '.join(top_performers[:3])}")

            # Risk warnings
            high_risk = []
            for analysis in analyses:
                if "risk_analysis" in analysis:
                    metrics = analysis["risk_analysis"].get("metrics", {})
                    if metrics.get("volatility", 0) > 0.3:
                        high_risk.append(analysis["symbol"])

            if high_risk:
                recommendations.append(f"High volatility stocks (exercise caution): {', '.join(high_risk[:3])}")

            # Market regime recommendations
            trending_up = trending_down = 0
            for analysis in analyses:
                if "market_regime" in analysis:
                    if analysis["market_regime"]["regime_type"] == "trending_up":
                        trending_up += 1
                    elif analysis["market_regime"]["regime_type"] == "trending_down":
                        trending_down += 1

            if trending_up > trending_down:
                recommendations.append("Market showing upward bias - consider momentum strategies")
            elif trending_down > trending_up:
                recommendations.append("Market showing downward bias - focus on defensive positions")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return recommendations


# Usage example and integration functions

async def main_integration_example():
    """Example of how to use the enhanced analytics"""

    # Initialize the enhanced analytics manager
    analytics_manager = EnhancedAnalyticsManager()

    if not await analytics_manager.initialize():
        print("Failed to initialize analytics manager")
        return

    # Example 1: Analyze a single stock
    print("\n=== Single Stock Analysis ===")
    analysis = await analytics_manager.get_enhanced_stock_analysis("RELIANCE")

    if "error" not in analysis:
        print(f"✓ Analysis completed for RELIANCE")
        print(f"  Sentiment: {analysis.get('sentiment', {}).get('sentiment_label', 'N/A')}")
        print(f"  Trading Signals: {len(analysis.get('trading_signals', []))}")
        print(f"  Patterns Found: {len(analysis.get('patterns', {}).get('candlestick', []))}")

    # Example 2: Run stock screener
    print("\n=== Stock Screener ===")
    screening_criteria = {
        "min_sentiment": 0.6,
        "max_volatility": 0.25
    }

    screener_results = await analytics_manager.run_stock_screener(screening_criteria)
    print(f"✓ Screener found {len(screener_results)} matching stocks")

    for result in screener_results[:3]:
        print(f"  {result['symbol']}: Score {result['score']:.1f}")

    # Example 3: Generate market report
    print("\n=== Market Report ===")
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
    report = await analytics_manager.generate_market_report(symbols)

    if "error" not in report:
        overview = report.get("market_overview", {})
        print(f"✓ Market Report Generated")
        print(f"  Market Sentiment: {overview.get('sentiment_interpretation', 'N/A')}")
        print(f"  Market Bias: {overview.get('market_bias', 'N/A')}")
        print(f"  Recommendations: {len(report.get('recommendations', []))}")


# Integration with existing NiceGUI app
async def integrate_with_existing_app(fetch_api, user_storage, get_cached_instruments, broker):
    """Integration function for existing NiceGUI app"""

    try:
        # Initialize enhanced analytics
        analytics_manager = EnhancedAnalyticsManager()
        await analytics_manager.initialize()

        # Enhance existing chart state with new capabilities
        chart_state["enhanced_features"] = {
            "pattern_recognition": True,
            "advanced_indicators": True,
            "risk_metrics": True,
            "sentiment_analysis": True
        }

        logger.info("✓ Enhanced analytics integrated with existing app")
        return True

    except Exception as e:
        logger.error(f"Failed to integrate enhanced analytics: {e}")
        return False


if __name__ == "__main__":
    # Run the integration example
    asyncio.run(main_integration_example())