from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List, Dict

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str
    upstox_api_key: Optional[str] = None
    upstox_api_secret: Optional[str] = None
    upstox_username: Optional[str] = None
    upstox_password: Optional[str] = None
    upstox_totp_token: Optional[str] = None
    zerodha_api_key: Optional[str] = None
    zerodha_api_secret: Optional[str] = None
    zerodha_username: Optional[str] = None
    zerodha_password: Optional[str] = None
    zerodha_totp_token: Optional[str] = None

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
    instrument_token: str
    last_price: float
    volume: int
    average_price: Optional[float] = None
    ohlc: Optional[Dict[str, float]] = None  # e.g., {"open": 100, "high": 105, "low": 98, "close": 102}

class OHLCResponse(BaseModel):
    instrument_token: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None

class LTPResponse(BaseModel):
    instrument_token: str
    last_price: float

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
    exchange: str
    trading_symbol: str
    name: Optional[str] = None
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