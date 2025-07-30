import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from backend.app.services import place_order, execute_strategy, schedule_strategy_execution, stop_strategy_execution
from backend.app.schemas import MFSIPRequest
from datetime import datetime

@pytest.mark.asyncio
async def test_place_order():
    mock_api = MagicMock()
    mock_api.place_order.return_value = MagicMock(data=MagicMock(order_id="12345"))
    mock_db = AsyncMock()
    mock_db.execute.return_value = AsyncMock(scalars=MagicMock(first=MagicMock(email="test@example.com")))
    response = await place_order(
        api=mock_api,
        instrument_token="NSE:INFY",
        trading_symbol="INFY",
        transaction_type="BUY",
        quantity=10,
        price=0,
        order_type="MARKET",
        broker="Upstox",
        db=mock_db,
        user_id="test_user"
    )
    assert response.data.order_id == "12345"
    mock_api.place_order.assert_called_once()

@pytest.mark.asyncio
async def test_execute_strategy():
    mock_api = MagicMock()
    mock_db = AsyncMock()
    mock_db.execute.return_value = AsyncMock(scalars=MagicMock(first=MagicMock(email="test@example.com")))
    result = await execute_strategy(
        api=mock_api,
        strategy="RSI Oversold/Overbought",
        instrument_token="NSE:INFY",
        quantity=10,
        stop_loss=1.0,
        take_profit=2.0,
        broker="Upstox",
        db=mock_db,
        user_id="test_user"
    )
    assert isinstance(result, str)
    mock_db.execute.assert_called()

@pytest.mark.asyncio
async def test_schedule_strategy_execution():
    mock_api = MagicMock()
    mock_db = AsyncMock()
    mock_db.execute.return_value = AsyncMock(scalars=MagicMock(first=MagicMock(email="test@example.com")))
    result = await schedule_strategy_execution(
        api=mock_api,
        strategy="MACD Crossover",
        instrument_token="NSE:INFY",
        quantity=10,
        stop_loss=1.0,
        take_profit=2.0,
        interval_minutes=5,
        run_hours=[(9, 15), (15, 30)],
        broker="Upstox",
        db=mock_db,
        user_id="test_user"
    )
    assert result["status"] == "success"
    assert "scheduled" in result["message"]
    mock_db.execute.assert_called()