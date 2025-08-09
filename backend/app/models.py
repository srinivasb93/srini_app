from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, func, JSON
from database import Base
from datetime import datetime
from uuid import uuid4

class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    upstox_api_key = Column(String)
    upstox_api_secret = Column(String)
    upstox_access_token = Column(String)
    upstox_access_token_expiry = Column(DateTime)
    zerodha_api_key = Column(String)
    zerodha_api_secret = Column(String)
    zerodha_access_token = Column(String)
    zerodha_access_token_expiry = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

class Order(Base):
    __tablename__ = "orders"
    order_id = Column(String, primary_key=True)
    broker = Column(String)
    trading_symbol = Column(String)
    instrument_token = Column(String)
    transaction_type = Column(String)
    quantity = Column(Integer)
    order_type = Column(String)
    price = Column(Float)
    trigger_price = Column(Float)
    product_type = Column(String)
    status = Column(String)
    remarks = Column(String)
    order_timestamp = Column(DateTime)
    user_id = Column(String)
    # Add stop loss and target fields
    stop_loss = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    is_amo = Column(Boolean, default=False)
    exit_order_id = Column(String, nullable=True)  # Track exit order for this position
    # Add trailing stop loss fields
    is_trailing_stop_loss = Column(Boolean, default=False)
    trailing_stop_loss_percent = Column(Float, nullable=True)
    trail_start_target_percent = Column(Float, nullable=True)
    trailing_activated = Column(Boolean, default=False)
    highest_price_achieved = Column(Float, nullable=True)
    lowest_price_achieved = Column(Float, nullable=True)  # For SELL orders trailing stop
    trailing_stop_price = Column(Float, nullable=True)

class ScheduledOrder(Base):
    __tablename__ = "scheduled_orders"
    scheduled_order_id = Column(String, primary_key=True, index=True)
    broker = Column(String)
    instrument_token = Column(String)
    trading_symbol = Column(String)
    transaction_type = Column(String)
    quantity = Column(Integer)
    order_type = Column(String)
    price = Column(Float, nullable=True)
    trigger_price = Column(Float, nullable=True)
    product_type = Column(String)
    schedule_datetime = Column(DateTime)
    stop_loss = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    status = Column(String, default="PENDING")
    is_amo = Column(Boolean, default=False)  # Ensure this exists
    user_id = Column(String)  # Likely added for user-specific orders
    # Add missing columns for execution tracking
    executed_at = Column(DateTime, nullable=True)
    executed_order_id = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

class AutoOrder(Base):
    __tablename__ = "auto_orders"
    auto_order_id = Column(String, primary_key=True)
    instrument_token = Column(String)
    trading_symbol = Column(String)
    transaction_type = Column(String)
    risk_per_trade = Column(Float)
    stop_loss_type = Column(String)
    stop_loss_value = Column(Float)
    target_value = Column(Float)
    atr_period = Column(Integer)
    product_type = Column(String)
    order_type = Column(String)
    limit_price = Column(Float)
    user_id = Column(String)
    broker = Column(String)
    created_at = Column(DateTime)

class GTTOrder(Base):
    __tablename__ = "gtt_orders"
    gtt_order_id = Column(String, primary_key=True)
    instrument_token = Column(String)
    trading_symbol = Column(String)
    transaction_type = Column(String)
    quantity = Column(Integer)
    trigger_type = Column(String)
    trigger_price = Column(Float)
    limit_price = Column(Float)
    last_price = Column(Float)
    second_trigger_price = Column(Float, nullable=True)
    second_limit_price = Column(Float, nullable=True)
    status = Column(String)
    broker = Column(String)
    user_id = Column(String)
    created_at = Column(DateTime)

class TradeHistory(Base):
    __tablename__ = "trade_history"
    trade_id = Column(String, primary_key=True)
    instrument_token = Column(String)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Integer)
    pnl = Column(Float)
    user_id = Column(String)
    broker = Column(String)

class QueuedOrder(Base):
    __tablename__ = "queued_orders"
    queued_order_id = Column(String, primary_key=True)
    parent_order_id = Column(String)
    instrument_token = Column(String)
    trading_symbol = Column(String)
    transaction_type = Column(String)
    quantity = Column(Integer)
    order_type = Column(String)
    price = Column(Float)
    trigger_price = Column(Float)
    product_type = Column(String)
    validity = Column(String)
    is_gtt = Column(String)
    status = Column(String)
    broker = Column(String)
    user_id = Column(String)

class MFSIP(Base):
    __tablename__ = "mf_sips"
    sip_id = Column(String, primary_key=True)
    scheme_code = Column(String)
    amount = Column(Float)
    frequency = Column(String)
    start_date = Column(DateTime)
    status = Column(String)
    user_id = Column(String)
    created_at = Column(DateTime)

class Instrument(Base):
    __tablename__ = "instruments"

    instrument_token = Column(String(50), primary_key=True)
    trading_symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(20), nullable=False)
    instrument_type = Column(String(20), nullable=False)
    segment = Column(String(20))

class Strategy(Base):
    __tablename__ = "strategies"
    strategy_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, nullable=False)
    broker = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    entry_conditions = Column(JSON, nullable=False)
    exit_conditions = Column(JSON, nullable=False)
    parameters = Column(JSON, nullable=False)
    status = Column(String, default="inactive")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)