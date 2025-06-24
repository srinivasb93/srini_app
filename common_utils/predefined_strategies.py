import pandas as pd


def check_macd_crossover(macd_line, signal_line):
    if len(macd_line) < 2 or len(signal_line) < 2:
        return None
    if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        return "BUY"
    elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        return "SELL"
    return None


def check_bollinger_band_signals(df, upper_band, lower_band):
    if len(df) < 2:
        return None
    if df['close'].iloc[-2] <= upper_band.iloc[-2] and df['close'].iloc[-1] > upper_band.iloc[-1]:
        return "SELL"
    elif df['close'].iloc[-2] >= lower_band.iloc[-2] and df['close'].iloc[-1] < lower_band.iloc[-1]:
        return "BUY"
    return None


def check_stochastic_signals(k, d, overbought=80, oversold=20):
    if len(k) < 2 or len(d) < 2:
        return None
    if (k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1]) and k.iloc[-1] < oversold:
        return "BUY"
    elif (k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1]) and k.iloc[-1] > overbought:
        return "SELL"
    return None


def check_support_resistance_breakout(df, lookback=20):
    if len(df) < lookback + 2:
        return None
    recent_high = df['high'].iloc[-lookback:-2].max()
    recent_low = df['low'].iloc[-lookback:-2].min()
    if df['close'].iloc[-2] < recent_high and df['close'].iloc[-1] > recent_high:
        return "BUY"
    elif df['close'].iloc[-2] > recent_low and df['close'].iloc[-1] < recent_low:
        return "SELL"
    return None


def backtest_strategy(df, strategy_func, **kwargs):
    df_copy = df.copy()
    signals = strategy_func(df_copy, **kwargs)
    df_copy['signal'] = signals
    df_copy['position'] = 0
    df_copy['pnl'] = 0
    position = 0
    buy_price = 0

    for i, row in df_copy.iterrows():
        if row['signal'] == "BUY" and position == 0:
            position = 1
            buy_price = row['close']
            df_copy.at[i, 'position'] = position
        elif row['signal'] == "SELL" and position == 1:
            position = 0
            sell_price = row['close']
            df_copy.at[i, 'position'] = position
            df_copy.at[i, 'pnl'] = sell_price - buy_price

    df_copy['cumulative_pnl'] = df_copy['pnl'].cumsum()
    return df_copy


def macd_strategy(df, fast_period=12, slow_period=26, signal_period=9):
    # Construct column names based on pandas-ta naming conventions
    macd_col = f'MACD_{fast_period}_{slow_period}_{signal_period}'
    signal_col = f'MACDs_{fast_period}_{slow_period}_{signal_period}'

    # Ensure required columns from pre-calculation exist
    if macd_col not in df.columns or signal_col not in df.columns:
        raise ValueError(f"Required MACD columns not found in DataFrame: {macd_col}, {signal_col}")

    # Use existing indicator columns
    macd_line = df[macd_col]
    signal_line = df[signal_col]

    # Generate signals using efficient vectorized operations
    signals = pd.Series(index=df.index, dtype='object')
    buy_signals = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line > signal_line)
    sell_signals = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line < signal_line)

    signals[buy_signals] = "BUY"
    signals[sell_signals] = "SELL"

    return signals


def bollinger_band_strategy(df, period=20, num_std=2):
    # Construct column names based on pandas-ta naming conventions
    upper_band_col = f'BBU_{period}_{float(num_std)}'
    lower_band_col = f'BBL_{period}_{float(num_std)}'

    # Ensure required columns exist
    if upper_band_col not in df.columns or lower_band_col not in df.columns:
        raise ValueError(f"Required Bollinger Bands columns not found: {upper_band_col}, {lower_band_col}")

    # Generate signals using vectorized operations
    signals = pd.Series(index=df.index, dtype='object')
    signals[df['close'] < df[lower_band_col]] = "BUY"
    signals[df['close'] > df[upper_band_col]] = "SELL"

    return signals


def rsi_strategy(df, period=14, overbought=70, oversold=30):
    # Construct column name based on pandas-ta naming conventions
    rsi_col = f'RSI_{period}'
    if rsi_col not in df.columns:
        raise ValueError(f"Required RSI column not found: {rsi_col}")

    rsi = df[rsi_col]
    signals = pd.Series(index=df.index, dtype='object')

    # Generate signals for crossover events using vectorized operations
    buy_signals = (rsi.shift(1) < oversold) & (rsi >= oversold)
    sell_signals = (rsi.shift(1) > overbought) & (rsi <= overbought)

    signals[buy_signals] = "BUY"
    signals[sell_signals] = "SELL"

    return signals