from sqlalchemy.ext.asyncio import create_async_engine
import asyncio
from sqlalchemy.sql import text

async def test_db():
    try:
        engine = create_async_engine(
            "postgresql+asyncpg://trading_user:password123@localhost:5432/trading_db",
            echo=True
        )
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print("Connection successful:", result.scalar())
        await engine.dispose()
    except Exception as e:
        print("Connection failed:", str(e))

asyncio.run(test_db())