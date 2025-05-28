from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, func
from database import Base
from datetime import datetime

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
