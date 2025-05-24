import os
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
from sqlalchemy.testing import future

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://trading_user:password123@localhost:5432/trading_db")
logger.info(f"Using DATABASE_URL in database.py: {DATABASE_URL}")

Base = declarative_base()

# Global session factory (set during startup)
session_factory = None
engine = None

async def init_engine():
    try:
        engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        global session_factory
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        logger.info("Database engine and session factory initialized")
        return session_factory, engine
    except Exception as e:
        logger.error(f"Failed to initialize database engine: {str(e)}")
        raise

async def get_db():
    global session_factory
    if session_factory is None:
        raise RuntimeError("Session factory not initialized")
    async with session_factory() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()