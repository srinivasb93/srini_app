from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, func, JSON, Text
from .database import Base
from datetime import datetime
from uuid import uuid4

class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

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
    preferences = Column(JSON, nullable=True)  # Store user preferences including risk settings
    created_at = Column(DateTime, server_default=func.now())

class Order(Base):
    __tablename__ = "orders"
    __table_args__ = {'extend_existing': True}
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
    # Essential tracking columns
    status_updated_at = Column(DateTime, nullable=True)
    broker_message = Column(Text, nullable=True)  # All broker responses (success/failure)
    trailing_activated_at = Column(DateTime, nullable=True)
    trailing_stop_triggered_at = Column(DateTime, nullable=True)
    stop_loss_triggered_at = Column(DateTime, nullable=True)
    exit_triggered_at = Column(DateTime, nullable=True)

class SIPActualTrade(Base):
    __tablename__ = "sip_actual_trades"
    __table_args__ = {'extend_existing': True}
    trade_id = Column(String, primary_key=True)
    portfolio_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    price = Column(Float, nullable=False)
    units = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    trade_type = Column(String, default='BUY')
    execution_status = Column(String, default='PENDING')  # PENDING, EXECUTED, FAILED
    order_status = Column(String, default='PLACED')  # PLACED, FAILED
    order_id = Column(String, nullable=True)  # For regular orders
    gtt_order_id = Column(String, nullable=True)  # For GTT orders
    order_type = Column(String, nullable=True)
    broker = Column(String, nullable=True)
    # Essential tracking columns
    order_executed_at = Column(DateTime, nullable=True)  # When order was executed (NULL = failed)
    broker_message = Column(Text, nullable=True)  # All broker responses
    portfolio_updated_at = Column(DateTime, nullable=True)  # When portfolio was updated

class ScheduledOrder(Base):
    __tablename__ = "scheduled_orders"
    __table_args__ = {'extend_existing': True}
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
    __table_args__ = {'extend_existing': True}
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
    __table_args__ = {'extend_existing': True}
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
    # Additional fields for richer GTT context
    gtt_type = Column(String, nullable=True)  # SINGLE or MULTIPLE
    rules = Column(JSON, nullable=True)       # Store full rules array when available

class TradeHistory(Base):
    __tablename__ = "trade_history"
    __table_args__ = {'extend_existing': True}
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
    __table_args__ = {'extend_existing': True}
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
    __table_args__ = {'extend_existing': True}
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
    __table_args__ = {'extend_existing': True}

    instrument_token = Column(String(50), primary_key=True)
    trading_symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(20), nullable=False)
    instrument_type = Column(String(20), nullable=False)
    segment = Column(String(20))

class Strategy(Base):
    __tablename__ = "strategies"
    __table_args__ = {'extend_existing': True}
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

class Position(Base):
    __tablename__ = "positions"
    __table_args__ = {'extend_existing': True}
    position_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, nullable=False)
    broker = Column(String, nullable=False)
    instrument_token = Column(String, nullable=False)
    trading_symbol = Column(String, nullable=False)
    quantity = Column(Integer, default=0)
    average_price = Column(Float, default=0.0)
    last_price = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class StrategyExecution(Base):
    __tablename__ = "strategy_executions"
    __table_args__ = {'extend_existing': True}
    execution_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    strategy_id = Column(String, nullable=False)  # References strategies table
    user_id = Column(String, nullable=False)
    broker = Column(String, nullable=False)
    instrument_token = Column(String, nullable=False)
    trading_symbol = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    risk_per_trade = Column(Float, default=2.0)  # Percentage
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    status = Column(String, default="running")  # running, stopped, completed, failed
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0)
    signals_generated = Column(Integer, default=0)
    trades_executed = Column(Integer, default=0)
    execution_config = Column(JSON, nullable=True)  # Store execution-specific config
    started_at = Column(DateTime, default=datetime.now)
    stopped_at = Column(DateTime, nullable=True)
    last_signal_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
