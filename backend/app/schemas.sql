-- Ensure trading_user exists and set the password
  DO $$
  BEGIN
      IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading_user') THEN
          CREATE ROLE trading_user WITH LOGIN PASSWORD 'password123';
      END IF;
  END $$;

  -- Alter the password to ensure it matches
  ALTER ROLE trading_user WITH PASSWORD 'password123';

  -- Grant privileges to trading_user
  GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;

  -- Create tables
  CREATE TABLE IF NOT EXISTS orders (
      order_id VARCHAR PRIMARY KEY,
      broker VARCHAR,
      trading_symbol VARCHAR,
      instrument_token VARCHAR,
      transaction_type VARCHAR,
      quantity INTEGER,
      order_type VARCHAR,
      price FLOAT,
      trigger_price FLOAT,
      product_type VARCHAR,
      status VARCHAR,
      remarks TEXT,
      user_id VARCHAR,
      order_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS scheduled_orders (
      scheduled_order_id VARCHAR PRIMARY KEY,
      broker VARCHAR,
      instrument_token VARCHAR,
      transaction_type VARCHAR,
      quantity INTEGER,
      order_type VARCHAR,
      price FLOAT,
      trigger_price FLOAT,
      product_type VARCHAR,
      schedule_datetime TIMESTAMP,
      stop_loss FLOAT,
      target FLOAT,
      status VARCHAR,
      is_amo BOOLEAN
  );

  CREATE TABLE IF NOT EXISTS auto_orders (
      auto_order_id VARCHAR PRIMARY KEY,
      instrument_token VARCHAR,
      transaction_type VARCHAR,
      risk_per_trade FLOAT,
      stop_loss_type VARCHAR,
      stop_loss_value FLOAT,
      target_value FLOAT,
      atr_period INTEGER,
      product_type VARCHAR,
      order_type VARCHAR,
      limit_price FLOAT
  );

  CREATE TABLE IF NOT EXISTS gtt_orders (
      gtt_order_id VARCHAR PRIMARY KEY,
      instrument_token VARCHAR,
      trading_symbol VARCHAR,
      transaction_type VARCHAR,
      quantity INTEGER,
      trigger_type VARCHAR,
      trigger_price FLOAT,
      limit_price FLOAT,
      second_trigger_price FLOAT,
      second_limit_price FLOAT,
      status VARCHAR,
      broker VARCHAR,
      created_at TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS trade_history (
      trade_id VARCHAR PRIMARY KEY,
      instrument_token VARCHAR,
      entry_time TIMESTAMP,
      exit_time TIMESTAMP,
      entry_price FLOAT,
      exit_price FLOAT,
      quantity INTEGER,
      pnl FLOAT
  );

  CREATE TABLE IF NOT EXISTS queued_orders (
      queued_order_id SERIAL PRIMARY KEY,
      parent_order_id VARCHAR,
      instrument_token VARCHAR,
      trading_symbol VARCHAR,
      transaction_type VARCHAR,
      quantity INTEGER,
      order_type VARCHAR,
      price FLOAT,
      trigger_price FLOAT,
      product_type VARCHAR,
      validity VARCHAR,
      is_gtt BOOLEAN,
      status VARCHAR
  );

  CREATE TABLE IF NOT EXISTS instruments (
      instrument_token VARCHAR PRIMARY KEY,
      exchange VARCHAR,
      symbol VARCHAR,
      name VARCHAR,
      last_price FLOAT
  );

  -- Grant table privileges to trading_user
  GRANT ALL ON ALL TABLES IN SCHEMA public TO trading_user;