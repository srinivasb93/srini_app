"""
Pydantic models for SIP Strategy API Routes
Extracted from sip_routes.py for better code organization
"""
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, validator
from enum import Enum


class JobTriggerType(str, Enum):
    CRON = "cron"
    INTERVAL = "interval"
    DATE = "date"


class SchedulerConfigRequest(BaseModel):
    job_id: str
    trigger_type: JobTriggerType
    # For CRON triggers
    cron_expression: Optional[str] = None
    # For interval triggers
    interval_seconds: Optional[int] = None
    interval_minutes: Optional[int] = None
    interval_hours: Optional[int] = None
    # For date triggers
    run_date: Optional[datetime] = None
    # Common job settings
    timezone: str = "Asia/Kolkata"
    max_instances: int = 1
    coalesce: bool = True
    misfire_grace_time: int = 300
    replace_existing: bool = True


class SIPSymbolConfig(BaseModel):
    """Configuration for individual symbol in multi-symbol portfolio"""
    symbol: str
    allocation_percentage: float = 100.0
    config: Optional[Dict] = None

    @validator('allocation_percentage')
    def validate_allocation(cls, v):
        if v <= 0 or v > 100:
            raise ValueError('Allocation percentage must be between 0 and 100')
        return v


class SIPConfigRequest(BaseModel):
    """Enhanced SIP Configuration with monthly investment limits"""
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

    @validator('fixed_investment')
    def validate_investment(cls, v):
        if v <= 0:
            raise ValueError('Investment amount must be positive')
        return v

    @validator('max_amount_in_a_month', always=True)
    def validate_monthly_limit(cls, v, values):
        if v is None:
            # Default to 4 times the fixed investment
            fixed_investment = values.get('fixed_investment', 5000)
            return fixed_investment * 4
        if v <= 0:
            raise ValueError('Monthly limit must be positive')
        fixed_investment = values.get('fixed_investment', 5000)
        if v < fixed_investment:
            raise ValueError('Monthly limit cannot be less than fixed investment')
        return v

    @validator('price_reduction_threshold')
    def validate_price_threshold(cls, v):
        if v <= 0:
            raise ValueError('Price reduction threshold must be positive')
        return v


class SIPBacktestRequest(BaseModel):
    """Enhanced backtest request with monthly limits"""
    symbols: List[str]
    start_date: str
    end_date: str
    config: SIPConfigRequest
    enable_monthly_limits: bool = True

    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError('At least one symbol required')
        return v


class SIPMultiPortfolioRequest(BaseModel):
    """Multi-symbol portfolio creation"""
    portfolio_name: str
    symbols: List[SIPSymbolConfig]
    default_config: SIPConfigRequest
    auto_rebalance: bool = False
    rebalance_frequency_days: int = 30

    @validator('symbols')
    def validate_symbols_allocation(cls, v):
        if not v:
            raise ValueError('At least one symbol required')

        total_allocation = sum(symbol.allocation_percentage for symbol in v)
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError('Total allocation must equal 100%')

        return v


class SIPPortfolioRequest(BaseModel):
    symbol: str
    portfolio_name: Optional[str] = None
    config: SIPConfigRequest


class InvestmentReportRequest(BaseModel):
    symbols: List[str]
    config: Optional[SIPConfigRequest] = None
    report_type: str = "comprehensive"  # "quick", "comprehensive", "detailed"
    include_risk_assessment: bool = True
    include_allocation_suggestions: bool = True

    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError('At least one symbol required')
        if len(v) > 20:
            raise ValueError('Maximum 20 symbols allowed per report')
        return v


class InvestmentReportResponse(BaseModel):
    report_id: str
    report_generated: str
    analysis_period: str
    overall_metrics: Dict[str, Any]
    portfolio_recommendation: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    symbol_reports: Dict[str, Any]
    disclaimer: str


class OrderType(str, Enum):
    GTT = "GTT"
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class SIPExecutionRequest(BaseModel):
    """Request model for SIP execution"""
    amount: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None  # Required for LIMIT orders
    gtt_trigger_price: Optional[float] = None  # Required for GTT orders
    gtt_limit_price: Optional[float] = None  # Required for GTT orders


class PortfolioUpdateRequest(BaseModel):
    """Request model for portfolio updates"""
    portfolio_name: Optional[str] = None
    config: Optional[SIPConfigRequest] = None
    status: Optional[str] = None  # 'active', 'paused', 'cancelled'
    auto_rebalance: Optional[bool] = None

    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['active', 'paused', 'cancelled']:
            raise ValueError('Status must be one of: active, paused, cancelled')
        return v


class MultiPortfolioUpdateRequest(BaseModel):
    """Request model for multi-portfolio updates"""
    portfolio_name: Optional[str] = None
    symbols: Optional[List[SIPSymbolConfig]] = None
    default_config: Optional[SIPConfigRequest] = None
    status: Optional[str] = None  # 'active', 'paused', 'cancelled'
    auto_rebalance: Optional[bool] = None
    rebalance_frequency_days: Optional[int] = None

    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['active', 'paused', 'cancelled']:
            raise ValueError('Status must be one of: active, paused, cancelled')
        return v

    @validator('rebalance_frequency_days')
    def validate_rebalance_frequency(cls, v):
        if v is not None and (v < 7 or v > 365):
            raise ValueError('Rebalance frequency must be between 7 and 365 days')
        return v
