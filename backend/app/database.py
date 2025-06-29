import os
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://trading_user:password123@localhost:5432/trading_db")
logger.info(f"Using DATABASE_URL in database.py: {DATABASE_URL}")

Base = declarative_base()

# SIMPLE GLOBAL VARIABLES - no singleton complexity
session_factory = None
engine = None

async def init_engine():
    """Initialize the database engine and session factory"""
    global session_factory, engine

    logger.info("🔧 init_engine() called")

    try:
        logger.info("🔧 Creating async engine...")
        engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )
        logger.info("✅ Async engine created successfully")

        logger.info("🔧 Creating session factory...")
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        logger.info("✅ Session factory created successfully")

        # Test the connection
        logger.info("🔧 Testing database connection...")
        async with session_factory() as test_session:
            from sqlalchemy import text
            await test_session.execute(text("SELECT 1"))
            await test_session.commit()
        logger.info("✅ Database connection test passed")

        logger.info(f"🎉 Database initialization completed. Session factory: {session_factory}")
        return session_factory, engine

    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        session_factory = None
        engine = None
        raise

async def get_db():
    """Get database session - simple version"""
    global session_factory

    logger.debug(f"get_db() called. Session factory: {session_factory is not None}")

    if session_factory is None:
        logger.error(f"❌ Session factory is None in get_db() - module: {__name__}")

        # CRITICAL FIX: Try to initialize if not already done
        logger.warning("⚠️ Attempting emergency database initialization...")
        try:
            await init_engine()
            logger.info("✅ Emergency initialization successful")
        except Exception as e:
            logger.error(f"❌ Emergency initialization failed: {e}")
            raise RuntimeError(f"Session factory not initialized: {e}")

    logger.debug("✅ Session factory is available, creating session")

    async with session_factory() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"❌ Database session error: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()

# Simple function to manually set the session factory
def set_global_session_factory(sf):
    """Set the global session factory manually"""
    global session_factory
    session_factory = sf
    logger.info(f"✅ Global session factory set manually: {sf}")

# Simple function to check if initialized
def is_database_initialized():
    """Check if database is initialized"""
    return session_factory is not None

# Module-level log
logger.info(f"📦 Database module loaded: {__name__}")