# backend/app/database.py - Enhanced Multi-Database Support
import os
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
from typing import Dict, Optional

load_dotenv()

logger = logging.getLogger(__name__)

Base = declarative_base()


# Multiple database configurations
class DatabaseManager:
    """Manages multiple database connections for trading application"""

    def __init__(self):
        self.engines: Dict[str, any] = {}
        self.session_factories: Dict[str, any] = {}

        # Database configurations
        self.db_configs = {
            'trading_db': {
                'url': os.getenv("DATABASE_URL",
                                 "postgresql+asyncpg://trading_user:password123@localhost:5432/trading_db"),
                'description': 'Main trading application database'
            },
            'nsedata': {
                'url': os.getenv("NSEDATA_URL", "postgresql+asyncpg://trading_user:password123@localhost:5432/nsedata"),
                'description': 'Stock market data database'
            }
        }

    async def init_engine(self, database_name: str = 'trading_db'):
        """Initialize engine for specific database"""
        if database_name in self.engines:
            return self.engines[database_name], self.session_factories[database_name]

        config = self.db_configs.get(database_name)
        if not config:
            raise ValueError(f"Database {database_name} not configured")

        logger.info(f"🔧 Initializing {database_name} database engine...")

        try:
            engine = create_async_engine(
                config['url'],
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_pre_ping=True
            )

            session_factory = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Test connection
            async with session_factory() as test_session:
                from sqlalchemy import text
                await test_session.execute(text("SELECT 1"))
                await test_session.commit()

            self.engines[database_name] = engine
            self.session_factories[database_name] = session_factory

            logger.info(f"✅ {database_name} database initialized successfully")
            return engine, session_factory

        except Exception as e:
            logger.error(f"❌ Failed to initialize {database_name}: {e}")
            raise

    async def init_all_databases(self):
        """Initialize all configured databases"""
        for db_name in self.db_configs.keys():
            try:
                await self.init_engine(db_name)
            except Exception as e:
                logger.error(f"Failed to initialize {db_name}: {e}")

    def get_session_factory(self, database_name: str = 'trading_db'):
        """Get session factory for specific database"""
        return self.session_factories.get(database_name)


# Global database manager instance
db_manager = DatabaseManager()


# Dependency functions for FastAPI
async def get_db(database_name: str = 'trading_db'):
    """Get database session for trading_db (default)"""
    session_factory = db_manager.get_session_factory(database_name)
    if not session_factory:
        await db_manager.init_engine(database_name)
        session_factory = db_manager.get_session_factory(database_name)

    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_nsedata_db():
    """Get database session specifically for nsedata"""
    session_factory = db_manager.get_session_factory('nsedata')
    if not session_factory:
        await db_manager.init_engine('nsedata')
        session_factory = db_manager.get_session_factory('nsedata')

    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


# Convenience function for getting any database session
async def get_database_session(database_name: str):
    """Generic function to get session for any database"""
    session_factory = db_manager.get_session_factory(database_name)
    if not session_factory:
        await db_manager.init_engine(database_name)
        session_factory = db_manager.get_session_factory(database_name)

    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


# Initialize function for app startup
async def init_databases():
    """Initialize all databases - call this in main.py startup"""
    await db_manager.init_all_databases()
    logger.info("🎉 All databases initialized successfully")