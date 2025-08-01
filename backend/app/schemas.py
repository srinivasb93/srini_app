from pydantic import BaseModel, EmailStr, validator, field_validator
from datetime import datetime
from typing import Optional, List, Dict, Any

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str
    upstox_api_key: Optional[str] = None
    upstox_api_secret: Optional[str] = None
    zerodha_api_key: Optional[str] = None
    zerodha_api_secret: Optional[str] = None


class UserResponse(UserBase):
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True

class Order(BaseModel):
    order_id: str
    broker: Optional[str] = None
    trading_symbol: Optional[str] = None
    instrument_token: Optional[str] = None
    transaction_type: Optional[str] = None
    quantity: Optional[int] = None
    order_type: Optional[str] = None
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    product_type: Optional[str] = None
    status: Optional[str] = None
    remarks: Optional[str] = None
    order_timestamp: Optional[datetime] = None
    user_id: Optional[str] = None

    class Config:
        from_attributes = True

class ScheduledOrder(BaseModel):
    scheduled_order_id: str
    broker: Optional[str] = None
    instrument_token: Optional[str] = None
    transaction_type: Optional[str] = None
    quantity: Optional[int] = None
    order_type: Optional[str] = None
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    product_type: Optional[str] = None
    schedule_datetime: Optional[datetime] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    status: Optional[str] = None
    is_amo: Optional[bool] = None
    user_id: str

    class Config:
        from_attributes = True

class ScheduledOrderRequest(BaseModel):
    broker: str
    instrument_token: str
    trading_symbol: str
    transaction_type: str
    quantity: int
    order_type: str
    price: float
    trigger_price: Optional[float] = None
    product_type: str
    schedule_datetime: datetime
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    is_amo: bool = False

    class Config:
        from_attributes = True

class AutoOrder(BaseModel):
    auto_order_id: str
    instrument_token: Optional[str] = None
    transaction_type: Optional[str] = None
    risk_per_trade: Optional[float] = None
    stop_loss_type: Optional[str] = None
    stop_loss_value: Optional[float] = None
    target_value: Optional[float] = None
    atr_period: Optional[int] = None
    product_type: Optional[str] = None
    order_type: Optional[str] = None
    limit_price: Optional[float] = None
    user_id: Optional[str] = None
    broker: Optional[str] = None

    class Config:
        from_attributes = True

class AutoOrderRequest(BaseModel):
    trading_symbol: str
    instrument_token: str
    transaction_type: str
    risk_per_trade: float
    stop_loss_type: str
    stop_loss_value: float
    target_value: float
    atr_period: int
    product_type: str
    order_type: str
    limit_price: Optional[float] = None
    broker: str

    class Config:
        from_attributes = True

class GTTOrder(BaseModel):
    gtt_order_id: str
    instrument_token: Optional[str] = None
    trading_symbol: Optional[str] = None
    transaction_type: Optional[str] = None
    quantity: Optional[int] = None
    trigger_type: Optional[str] = None
    trigger_price: Optional[float] = None
    limit_price: Optional[float] = None
    last_price: Optional[float] = None
    second_trigger_price: Optional[float] = None
    second_limit_price: Optional[float] = None
    status: Optional[str] = None
    broker: Optional[str] = None
    created_at: Optional[datetime] = None
    user_id: Optional[str] = None

    class Config:
        from_attributes = True

class GTTOrderRequest(BaseModel):
    instrument_token: str
    trading_symbol: str
    transaction_type: str
    quantity: int
    trigger_type: str
    trigger_price: float
    limit_price: float
    last_price: float
    second_trigger_price: Optional[float] = None
    second_limit_price: Optional[float] = None
    broker: str

class TradeHistory(BaseModel):
    trade_id: str
    instrument_token: Optional[str] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    quantity: Optional[int] = None
    pnl: Optional[float] = None
    user_id: Optional[str] = None
    broker: Optional[str] = None

    class Config:
        from_attributes = True

class PlaceOrderRequest(BaseModel):
    trading_symbol: str
    instrument_token: str
    quantity: int
    order_type: str
    transaction_type: str
    product_type: str
    is_amo: bool
    price: float = 0.0
    trigger_price: float = 0.0
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    validity: str = "DAY"
    broker: str
    schedule_datetime: Optional[str] = None

    @field_validator("schedule_datetime")
    def validate_schedule_datetime(cls, v):
        if v:
            try:
                datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError("schedule_datetime must be in format 'YYYY-MM-DD HH:MM:SS'")
        return v

class ModifyOrderRequest(BaseModel):
    quantity: Optional[int] = None
    order_type: Optional[str] = None
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    validity: Optional[str] = "DAY"

class ProfileResponse(BaseModel):
    user_id: str
    email: EmailStr
    name: Optional[str] = None
    broker: str

class MarginResponse(BaseModel):
    equity: Optional[Dict[str, float]] = None  # e.g., {"available": 10000, "used": 5000}
    commodity: Optional[Dict[str, float]] = None
    broker: str

class QuoteResponse(BaseModel):
    trading_symbol: str
    instrument_token: str
    last_price: float
    net_change: float
    pct_change: float
    volume: int
    average_price: Optional[float] = None
    ohlc: Optional[Dict[str, float]] = None
    depth: Optional[dict] = None

class OHLCResponse(BaseModel):
    instrument_token: str
    trading_symbol: str
    open: float
    high: float
    low: float
    close: float
    previous_close: Optional[float] = None
    volume: Optional[int] = None

class LTPResponse(BaseModel):
    instrument_token: str
    trading_symbol: str
    last_price: float
    volume: Optional[int] = None
    previous_close: Optional[float] = None

class HistoricalDataPoint(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None

class HistoricalDataResponse(BaseModel):
    instrument_token: str
    data: List[HistoricalDataPoint]

class Instrument(BaseModel):
    instrument_token: str
    trading_symbol: str
    exchange: str
    instrument_type: Optional[str] = None
    segment: Optional[str] = None

class OrderHistory(BaseModel):
    order_id: str
    status: str
    timestamp: datetime
    price: Optional[float] = None
    quantity: Optional[int] = None
    remarks: Optional[str] = None

class Trade(BaseModel):
    trade_id: str
    order_id: str
    instrument_token: str
    quantity: int
    price: float
    timestamp: datetime

class StrategyRequest(BaseModel):
    broker: str
    name: str
    description: Optional[str] = None
    entry_conditions: List[Dict[str, Any]]
    exit_conditions: List[Dict[str, Any]]
    parameters: Dict[str, Any]

class StrategyResponse(BaseModel):
    strategy_id: str
    user_id: str
    broker: str
    name: str
    description: Optional[str] = None
    entry_conditions: List[Dict[str, Any]]
    exit_conditions: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class BacktestRequest(BaseModel):
    instrument_token: str
    timeframe: str
    strategy: str
    params: Dict[str, Any]
    start_date: str
    end_date: str

    @validator("params")
    def validate_params(cls, v):
        if "initial_investment" in v and v["initial_investment"] <= 0:
            raise ValueError("Initial investment must be positive")
        if "stop_loss_range" in v:
            if not isinstance(v["stop_loss_range"], list) or len(v["stop_loss_range"]) != 2:
                raise ValueError("Stop loss range must be a list of [min, max]")
            if v["stop_loss_range"][0] >= v["stop_loss_range"][1]:
                raise ValueError("Stop loss min must be less than max")
        if "target_range" in v:
            if not isinstance(v["target_range"], list) or len(v["target_range"]) != 2:
                raise ValueError("Target range must be a list of [min, max]")
            if v["target_range"][0] >= v["target_range"][1]:
                raise ValueError("Target min must be less than max")
        return v

class MFSIPRequest(BaseModel):
    scheme_code: str
    amount: float
    frequency: str
    start_date: datetime

class MFSIPResponse(BaseModel):
    sip_id: str
    scheme_code: str
    amount: float
    frequency: str
    start_date: datetime
    status: str
    user_id: str
    created_at: datetime