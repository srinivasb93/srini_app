# backend/app/database.py - Fixed Database Manager with Proper Connection Handling
import os
import logging
import asyncio
import weakref
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text, event
from dotenv import load_dotenv
from typing import Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import atexit

load_dotenv()

logger = logging.getLogger(__name__)

Base = declarative_base()


class DatabaseManager:
    """Enhanced database manager with proper connection lifecycle management"""

    def __init__(self):
        self.engines: Dict[str, any] = {}
        self.session_factories: Dict[str, any] = {}
        self._active_sessions: weakref.WeakSet = weakref.WeakSet()
        self._shutdown_event = asyncio.Event()
        self._is_shutting_down = False

        # Enhanced database configurations
        self.db_configs = {
            'trading_db': {
                'url': os.getenv("DATABASE_URL",
                                 "postgresql+asyncpg://trading_user:password123@localhost:5432/trading_db"),
                'description': 'Main trading application database',
                'pool_settings': {
                    'pool_size': 8,  # Reduced for stability
                    'max_overflow': 12,  # Conservative overflow
                    'pool_timeout': 30,  # Reasonable timeout
                    'pool_recycle': 3600,  # Recycle every hour
                    'pool_pre_ping': True,  # Health checks
                    'pool_reset_on_return': 'rollback',  # Clean state
                    'echo': False,  # Set to True for debugging
                    'future': True
                }
            },
            'nsedata': {
                'url': os.getenv("NSEDATA_URL", "postgresql+asyncpg://trading_user:password123@localhost:5432/nsedata"),
                'description': 'Stock market data database',
                'pool_settings': {
                    'pool_size': 6,  # Smaller for data operations
                    'max_overflow': 8,
                    'pool_timeout': 20,
                    'pool_recycle': 1800,  # More frequent recycling
                    'pool_pre_ping': True,
                    'pool_reset_on_return': 'rollback',
                    'echo': False,
                    'future': True
                }
            }
        }

        # Register cleanup on exit
        atexit.register(self._sync_cleanup)

    async def init_engine(self, database_name: str = 'trading_db'):
        """Initialize engine with enhanced error handling and proper cleanup"""
        if self._is_shutting_down:
            raise RuntimeError("Database manager is shutting down")

        if database_name in self.engines:
            return self.engines[database_name], self.session_factories[database_name]

        config = self.db_configs.get(database_name)
        if not config:
            raise ValueError(f"Database {database_name} not configured")

        logger.info(f"Initializing {database_name} database engine...")

        try:
            # Enhanced engine configuration
            pool_settings = config.get('pool_settings', {})

            engine = create_async_engine(
                config['url'],
                **pool_settings
            )

            # Add event listeners for connection management
            @event.listens_for(engine.sync_engine, "connect")
            def set_connection_pragmas(dbapi_connection, connection_record):
                """Set connection-level settings"""
                logger.debug(f"New connection established for {database_name}")

            @event.listens_for(engine.sync_engine, "close")
            def receive_close(dbapi_connection, connection_record):
                """Handle connection close events"""
                logger.debug(f"Connection closed for {database_name}")

            session_factory = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,  # Manual control
                autocommit=False
            )

            # Enhanced connection test with proper cleanup
            await self._test_connection(session_factory, database_name)

            self.engines[database_name] = engine
            self.session_factories[database_name] = session_factory

            logger.info(f"{database_name} database initialized successfully")
            return engine, session_factory

        except Exception as e:
            logger.error(f"❌ Failed to initialize {database_name}: {e}")
            raise

    async def _test_connection(self, session_factory, database_name: str, max_retries: int = 3):
        """Test database connection with proper cleanup"""
        for attempt in range(max_retries):
            session = None
            try:
                session = session_factory()
                result = await session.execute(text("SELECT 1, NOW() as server_time, version()"))
                row = result.fetchone()
                await session.commit()

                logger.info(
                    f"Database {database_name} connected successfully\n"
                    f"   Server time: {row[1]}\n"
                    f"   PostgreSQL version: {row[2][:50]}..."
                )
                return

            except Exception as test_error:
                if session:
                    try:
                        await session.rollback()
                        await session.close()
                    except:
                        pass

                if attempt == max_retries - 1:
                    raise test_error

                wait_time = 2 ** attempt
                logger.warning(
                    f"Connection test attempt {attempt + 1} failed for {database_name}: {test_error}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            finally:
                if session:
                    try:
                        await session.close()
                    except Exception as close_error:
                        logger.warning(f"Error closing test session: {close_error}")

    async def get_session(self, database_name: str = 'trading_db') -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with proper lifecycle management"""
        if self._is_shutting_down:
            raise RuntimeError("Database manager is shutting down")

        session_factory = self.session_factories.get(database_name)
        if not session_factory:
            await self.init_engine(database_name)
            session_factory = self.session_factories.get(database_name)

        session = None
        try:
            session = session_factory()
            self._active_sessions.add(session)

            yield session

        except Exception as e:
            if session:
                try:
                    await session.rollback()
                    logger.warning(f"Session rolled back due to error: {e}")
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
            raise
        finally:
            if session:
                try:
                    await session.close()
                    logger.debug(f"Session closed for {database_name}")
                except Exception as close_error:
                    logger.error(f"Error closing session: {close_error}")

    async def health_check(self, database_name: str = None) -> dict:
        """Check database health with proper async context handling"""
        databases_to_check = [database_name] if database_name else list(self.db_configs.keys())
        health_status = {}

        for db_name in databases_to_check:
            try:
                if self._is_shutting_down:
                    health_status[db_name] = {'status': 'shutting_down'}
                    continue

                # Get session factory directly instead of using async context manager
                session_factory = self.session_factories.get(db_name)
                if not session_factory:
                    health_status[db_name] = {
                        'status': 'not_initialized',
                        'error': 'Session factory not found'
                    }
                    continue

                # Create session manually for health check
                session = None
                try:
                    session = session_factory()
                    start_time = asyncio.get_event_loop().time()

                    result = await session.execute(text("SELECT 1, NOW(), version()"))
                    row = result.fetchone()

                    response_time = (asyncio.get_event_loop().time() - start_time) * 1000

                    health_status[db_name] = {
                        'status': 'healthy',
                        'response_time_ms': round(response_time, 2),
                        'server_time': row[1].isoformat(),
                        'description': self.db_configs[db_name]['description']
                    }

                except Exception as session_error:
                    health_status[db_name] = {
                        'status': 'unhealthy',
                        'error': f"Session error: {str(session_error)}"
                    }
                finally:
                    if session:
                        try:
                            await session.close()
                        except Exception as close_error:
                            logger.warning(f"Error closing health check session for {db_name}: {close_error}")

            except Exception as e:
                health_status[db_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }

        return health_status

    async def close_all_sessions(self):
        """Close all active sessions gracefully"""
        logger.info("Closing all active database sessions...")

        # Copy the set to avoid modification during iteration
        active_sessions = list(self._active_sessions)

        if not active_sessions:
            logger.info("No active sessions to close")
            return

        logger.info(f"Found {len(active_sessions)} active sessions to close")

        close_tasks = []
        for session in active_sessions:
            if session and not session.is_active:
                continue

            async def close_session(s):
                try:
                    if s.in_transaction():
                        await s.rollback()
                    await s.close()
                    logger.debug("Session closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")

            close_tasks.append(close_session(session))

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Clear the weak set
        self._active_sessions.clear()
        logger.info("All sessions closed")

    async def close_all(self):
        """Gracefully close all database connections and engines"""
        if self._is_shutting_down:
            logger.warning("Database manager already shutting down")
            return

        self._is_shutting_down = True
        logger.info("Starting database manager shutdown...")

        try:
            # First, close all active sessions
            await self.close_all_sessions()

            # Then dispose of all engines
            dispose_tasks = []
            for db_name, engine in self.engines.items():
                if engine:
                    async def dispose_engine(name, eng):
                        try:
                            # Give ongoing operations a chance to complete
                            await asyncio.sleep(0.1)

                            # Dispose the engine
                            await eng.dispose()
                            logger.info(f"Engine {name} disposed successfully")
                        except Exception as e:
                            logger.error(f"❌ Error disposing engine {name}: {e}")

                    dispose_tasks.append(dispose_engine(db_name, engine))

            if dispose_tasks:
                # Wait for all engines to dispose, but don't wait forever
                await asyncio.wait_for(
                    asyncio.gather(*dispose_tasks, return_exceptions=True),
                    timeout=10.0
                )

            # Clear all references
            self.engines.clear()
            self.session_factories.clear()

            self._shutdown_event.set()
            logger.info("Database manager shutdown completed")

        except asyncio.TimeoutError:
            logger.warning("⚠️ Database shutdown timed out, forcing cleanup")
            self.engines.clear()
            self.session_factories.clear()
        except Exception as e:
            logger.error(f"❌ Error during database shutdown: {e}")
            # Force cleanup even on error
            self.engines.clear()
            self.session_factories.clear()

    def _sync_cleanup(self):
        """Synchronous cleanup for atexit"""
        if self._is_shutting_down:
            return

        logger.info("Performing synchronous database cleanup...")
        try:
            # Try to run async cleanup if event loop is available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup as a task
                    asyncio.create_task(self.close_all())
                else:
                    # Run cleanup in the loop
                    loop.run_until_complete(self.close_all())
            except RuntimeError:
                # No event loop available, just clear references
                logger.warning("No event loop available for cleanup, clearing references")
                self.engines.clear()
                self.session_factories.clear()
                self._is_shutting_down = True
        except Exception as e:
            logger.error(f"Error in sync cleanup: {e}")

    async def init_all_databases(self):
        """Initialize all configured databases"""
        logger.info("Initializing all databases...")

        initialization_results = {}

        for db_name in self.db_configs.keys():
            try:
                await self.init_engine(db_name)
                initialization_results[db_name] = "success"
                logger.info(f"{db_name} initialized successfully")
            except Exception as e:
                initialization_results[db_name] = f"failed: {str(e)}"
                logger.error(f"❌ Failed to initialize {db_name}: {e}")
                # Continue with other databases instead of failing completely

        # Check if at least one database was initialized successfully
        successful_dbs = [db for db, result in initialization_results.items() if result == "success"]

        if not successful_dbs:
            raise Exception("Failed to initialize any databases")

        logger.info(f"Database initialization completed. Success: {len(successful_dbs)}/{len(self.db_configs)}")
        return initialization_results

    def get_session_factory(self, database_name: str = 'trading_db'):
        """Get session factory for specific database"""
        if self._is_shutting_down:
            return None
        return self.session_factories.get(database_name)


# Global database manager instance
db_manager = DatabaseManager()


# Enhanced dependency functions
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for trading_db with proper lifecycle management"""
    # FIXED: Don't wrap the generator in async with - just delegate to it
    async for session in db_manager.get_session('trading_db'):
        yield session

async def get_nsedata_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for nsedata with proper lifecycle management"""
    # FIXED: Don't wrap the generator in async with - just delegate to it
    async for session in db_manager.get_session('nsedata'):
        yield session

async def get_database_session(database_name: str) -> AsyncGenerator[AsyncSession, None]:
    """Generic function to get session for any database"""
    # FIXED: Don't wrap the generator in async with - just delegate to it
    async for session in db_manager.get_session(database_name):
        yield session


# Initialize function for app startup
async def init_databases():
    """Initialize all databases - call this in main.py startup"""
    try:
        await db_manager.init_all_databases()
        logger.info("All databases initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


# Cleanup function for app shutdown
async def cleanup_databases():
    """Cleanup all database connections - call this in main.py shutdown"""
    await db_manager.close_all()