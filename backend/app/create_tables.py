from sqlalchemy.ext.asyncio import AsyncSession
from database import init_engine, Base
import asyncio
import logging
from models import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def create_tables():
    session_factory, engine = await init_engine()
    logger.info(f"Registered tables: {list(Base.metadata.tables.keys())}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        created_tables = Base.metadata.tables.keys()
        logger.info(f"Created tables: {', '.join(created_tables)}")
    await engine.dispose()
    logger.info("Database tables created successfully")

if __name__ == "__main__":
    asyncio.run(create_tables())