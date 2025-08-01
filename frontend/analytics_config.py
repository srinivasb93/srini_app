"""
Analytics Configuration - analytics_config.py
Configuration settings for enhanced analytics features
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
from enum import Enum


class DataSource(Enum):
    """Available data sources for market data"""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    UPSTOX = "upstox"
    ZERODHA = "zerodha"
    NSE_PYTHON = "nsepython"
    FINNHUB = "finnhub"
    POLYGON = "polygon"


class ChartProvider(Enum):
    """Chart providers"""
    TRADINGVIEW = "tradingview"
    LIGHTWEIGHT_CHARTS = "lightweight_charts"
    PLOTLY = "plotly"
    CUSTOM = "custom"


@dataclass
class TechnicalIndicatorConfig:
    """Configuration for technical indicators"""
    name: str
    enabled: bool = False
    parameters: Dict = None
    color: str = "#ffffff"
    line_width: int = 1
    chart_pane: str = "main"  # main, separate, overlay


@dataclass
class ChartConfig:
    """Chart appearance and behavior configuration"""
    theme: str = "dark"
    chart_type: str = "candlestick"  # candlestick, line, area, heikin_ashi
    height: int = 600
    width: str = "100%"
    timezone: str = "Asia/Kolkata"
    grid_lines: bool = True
    volume_panel: bool = True
    crosshair: bool = True
    price_scale_position: str = "right"
    time_scale_visible: bool = True
    watermark_visible: bool = True
    watermark_text: str = "Srini's Trading App"


@dataclass
class DataAPIConfig:
    """Configuration for data API endpoints"""
    # Alpha Vantage
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"

    # Finnhub
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    finnhub_base_url: str = "https://finnhub.io/api/v1"

    # Polygon
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")
    polygon_base_url: str = "https://api.polygon.io"

    # News APIs
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    news_api_base_url: str = "https://newsapi.org/v2"


class AnalyticsConfig:
    """Main analytics configuration class"""

    # Data Sources
    PRIMARY_DATA_SOURCE = DataSource.YFINANCE
    FALLBACK_DATA_SOURCES = [DataSource.NSE_PYTHON, DataSource.ALPHA_VANTAGE]

    # Chart Settings
    CHART_PROVIDER = ChartProvider.TRADINGVIEW
    CHART_CONFIG = ChartConfig()

    # API Configuration
    API_CONFIG = DataAPIConfig()

    # Default symbols for quick access
    DEFAULT_SYMBOLS = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "ITC", "LT", "WIPRO", "ONGC", "BHARTIARTL",
        "KOTAKBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE"
    ]

    # Index symbols
    INDEX_SYMBOLS = [
        "NIFTY50", "BANKNIFTY", "FINNIFTY", "NIFTYMIDCAP50",
        "NIFTYSMALLCAP50", "NIFTYIT", "NIFTYPHARMA", "NIFTYAUTO"
    ]

    # Technical Indicators Configuration
    TECHNICAL_INDICATORS = {
        "sma": TechnicalIndicatorConfig(
            name="Simple Moving Average",
            parameters={"periods": [20, 50, 200]},
            color="#ff6b6b",
            chart_pane="main"
        ),
        "ema": TechnicalIndicatorConfig(
            name="Exponential Moving Average",
            parameters={"periods": [12, 26, 50]},
            color="#4ecdc4",
            chart_pane="main"
        ),
        "rsi": TechnicalIndicatorConfig(
            name="Relative Strength Index",
            parameters={"period": 14, "overbought": 70, "oversold": 30},
            color="#ffd93d",
            chart_pane="separate"
        ),
        "macd": TechnicalIndicatorConfig(
            name="MACD",
            parameters={"fast": 12, "slow": 26, "signal": 9},
            color="#6c5ce7",
            chart_pane="separate"
        ),
        "bollinger": TechnicalIndicatorConfig(
            name="Bollinger Bands",
            parameters={"period": 20, "std_dev": 2},
            color="#a29bfe",
            chart_pane="main"
        ),
        "stochastic": TechnicalIndicatorConfig(
            name="Stochastic Oscillator",
            parameters={"k_period": 14, "d_period": 3, "smooth": 3},
            color="#fd79a8",
            chart_pane="separate"
        ),
        "williams_r": TechnicalIndicatorConfig(
            name="Williams %R",
            parameters={"period": 14},
            color="#fdcb6e",
            chart_pane="separate"
        ),
        "cci": TechnicalIndicatorConfig(
            name="Commodity Channel Index",
            parameters={"period": 20},
            color="#e17055",
            chart_pane="separate"
        ),
        "atr": TechnicalIndicatorConfig(
            name="Average True Range",
            parameters={"period": 14},
            color="#81ecec",
            chart_pane="separate"
        ),
        "adx": TechnicalIndicatorConfig(
            name="Average Directional Index",
            parameters={"period": 14},
            color="#fab1a0",
            chart_pane="separate"
        ),
        "obv": TechnicalIndicatorConfig(
            name="On Balance Volume",
            parameters={},
            color="#00b894",
            chart_pane="separate"
        ),
        "vwap": TechnicalIndicatorConfig(
            name="Volume Weighted Average Price",
            parameters={},
            color="#0984e3",
            chart_pane="main"
        )
    }

    # Timeframes configuration
    TIMEFRAMES = {
        "1m": {"label": "1 Min", "seconds": 60},
        "5m": {"label": "5 Min", "seconds": 300},
        "15m": {"label": "15 Min", "seconds": 900},
        "30m": {"label": "30 Min", "seconds": 1800},
        "1h": {"label": "1 Hour", "seconds": 3600},
        "4h": {"label": "4 Hour", "seconds": 14400},
        "1d": {"label": "1 Day", "seconds": 86400},
        "1w": {"label": "1 Week", "seconds": 604800},
        "1M": {"label": "1 Month", "seconds": 2629746}
    }

    # Chart types
    CHART_TYPES = {
        "candlestick": "Candlestick",
        "line": "Line Chart",
        "area": "Area Chart",
        "heikin_ashi": "Heikin Ashi",
        "renko": "Renko",
        "point_figure": "Point & Figure"
    }

    # Drawing tools
    DRAWING_TOOLS = {
        "line": {"name": "Trend Line", "icon": "trending_up"},
        "horizontal_line": {"name": "Horizontal Line", "icon": "horizontal_rule"},
        "vertical_line": {"name": "Vertical Line", "icon": "more_vert"},
        "rectangle": {"name": "Rectangle", "icon": "crop_square"},
        "circle": {"name": "Circle", "icon": "circle"},
        "fibonacci": {"name": "Fibonacci Retracement", "icon": "timeline"},
        "pitchfork": {"name": "Andrews Pitchfork", "icon": "call_split"},
        "gann_fan": {"name": "Gann Fan", "icon": "grain"},
        "text": {"name": "Text Label", "icon": "text_fields"}
    }

    # Alert configuration
    ALERT_CONFIG = {
        "max_alerts_per_user": 50,
        "alert_check_interval": 60,  # seconds
        "supported_conditions": [
            "price_above", "price_below", "price_crosses_above", "price_crosses_below",
            "rsi_above", "rsi_below", "volume_above_average", "macd_bullish_crossover",
            "macd_bearish_crossover", "bollinger_squeeze", "bollinger_breakout"
        ],
        "notification_methods": ["email", "sms", "push", "webhook"]
    }

    # Screener configuration
    SCREENER_CONFIG = {
        "max_results": 100,
        "available_criteria": {
            "technical": [
                "rsi_range", "macd_signal", "sma_position", "ema_position",
                "bollinger_position", "volume_ratio", "price_change",
                "volatility_range", "momentum_score"
            ],
            "fundamental": [
                "pe_ratio", "pb_ratio", "market_cap", "debt_to_equity",
                "roe", "roce", "dividend_yield", "revenue_growth",
                "profit_growth", "current_ratio"
            ],
            "price_volume": [
                "price_range", "volume_range", "avg_volume_ratio",
                "price_near_52w_high", "price_near_52w_low",
                "breakout_pattern", "breakdown_pattern"
            ]
        },
        "preset_screens": {
            "momentum_stocks": {
                "rsi_range": (50, 70),
                "volume_ratio": (1.5, 5.0),
                "price_change": (2, 10)
            },
            "oversold_stocks": {
                "rsi_range": (20, 35),
                "pe_ratio": (0, 25),
                "volume_ratio": (1.2, 3.0)
            },
            "breakout_candidates": {
                "volume_ratio": (2.0, 10.0),
                "price_near_52w_high": True,
                "rsi_range": (50, 80)
            },
            "value_stocks": {
                "pe_ratio": (0, 15),
                "pb_ratio": (0, 2),
                "debt_to_equity": (0, 0.5),
                "roe": (15, 100)
            }
        }
    }

    # Performance and caching
    PERFORMANCE_CONFIG = {
        "cache_timeout": 300,  # 5 minutes
        "max_cache_size": 1000,
        "data_refresh_interval": 60,  # seconds
        "batch_size": 50,
        "concurrent_requests": 10,
        "request_timeout": 30
    }

    # Risk management defaults
    RISK_CONFIG = {
        "default_stop_loss_percent": 2.0,
        "default_target_percent": 6.0,
        "max_position_size_percent": 10.0,
        "risk_reward_ratio": 1.5,
        "atr_multiplier_stop_loss": 2.0,
        "atr_multiplier_target": 3.0
    }

    # Export and reporting
    EXPORT_CONFIG = {
        "supported_formats": ["json", "csv", "excel", "pdf"],
        "report_templates": [
            "daily_analysis", "weekly_summary", "monthly_performance",
            "stock_analysis", "portfolio_review", "trade_journal"
        ],
        "chart_export_formats": ["png", "svg", "pdf"],
        "max_export_rows": 10000
    }

    # UI Configuration
    UI_CONFIG = {
        "theme": "dark",
        "primary_color": "#3b82f6",
        "success_color": "#22c55e",
        "danger_color": "#ef4444",
        "warning_color": "#f59e0b",
        "info_color": "#06b6d4",
        "font_family": "Inter, Segoe UI, sans-serif",
        "animation_duration": 300,
        "refresh_intervals": {
            "live_data": 5000,  # 5 seconds
            "portfolio": 30000,  # 30 seconds
            "news": 300000  # 5 minutes
        }
    }

    # Logging configuration
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_path": "logs/analytics.log",
        "max_file_size": "10MB",
        "backup_count": 5,
        "log_to_console": True
    }


# Environment-specific configurations
class DevelopmentConfig(AnalyticsConfig):
    """Development environment configuration"""
    LOGGING_CONFIG = {
        **AnalyticsConfig.LOGGING_CONFIG,
        "level": "DEBUG"
    }
    PERFORMANCE_CONFIG = {
        **AnalyticsConfig.PERFORMANCE_CONFIG,
        "cache_timeout": 60,
        "data_refresh_interval": 10
    }


class ProductionConfig(AnalyticsConfig):
    """Production environment configuration"""
    PERFORMANCE_CONFIG = {
        **AnalyticsConfig.PERFORMANCE_CONFIG,
        "cache_timeout": 600,
        "data_refresh_interval": 30,
        "concurrent_requests": 20
    }
    LOGGING_CONFIG = {
        **AnalyticsConfig.LOGGING_CONFIG,
        "level": "WARNING"
    }


class TestingConfig(AnalyticsConfig):
    """Testing environment configuration"""
    PRIMARY_DATA_SOURCE = DataSource.YFINANCE
    FALLBACK_DATA_SOURCES = []

    PERFORMANCE_CONFIG = {
        **AnalyticsConfig.PERFORMANCE_CONFIG,
        "cache_timeout": 0,  # No caching in tests
        "data_refresh_interval": 1
    }


def get_config():
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


# Validation functions
def validate_config(config: AnalyticsConfig) -> bool:
    """Validate configuration settings"""
    try:
        # Check required API keys if using premium data sources
        if config.PRIMARY_DATA_SOURCE in [DataSource.ALPHA_VANTAGE, DataSource.FINNHUB, DataSource.POLYGON]:
            api_key_map = {
                DataSource.ALPHA_VANTAGE: config.API_CONFIG.alpha_vantage_api_key,
                DataSource.FINNHUB: config.API_CONFIG.finnhub_api_key,
                DataSource.POLYGON: config.API_CONFIG.polygon_api_key
            }

            if not api_key_map.get(config.PRIMARY_DATA_SOURCE):
                print(f"Warning: No API key found for {config.PRIMARY_DATA_SOURCE.value}")
                return False

        # Validate timeframes
        if not config.TIMEFRAMES:
            print("Error: No timeframes configured")
            return False

        # Validate symbols
        if not config.DEFAULT_SYMBOLS:
            print("Warning: No default symbols configured")

        return True

    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False


# Usage example
if __name__ == "__main__":
    config = get_config()

    if validate_config(config):
        print("✓ Configuration validated successfully")
        print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        print(f"Primary data source: {config.PRIMARY_DATA_SOURCE.value}")
        print(f"Chart provider: {config.CHART_PROVIDER.value}")
        print(f"Available indicators: {len(config.TECHNICAL_INDICATORS)}")
    else:
        print("✗ Configuration validation failed")