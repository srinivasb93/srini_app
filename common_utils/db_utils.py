import logging
from sqlalchemy.ext.asyncio import AsyncSession


# Removed basicConfig to not interfere with main app logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def async_fetch_query(db: AsyncSession, query: str, params: dict) -> list:
    """
    Fetch data from the database using the provided SQL query.
    Args:
        db: AsyncSession dependency
        query: SQL query string
        params: Parameters for the query
    Returns:
        List of dictionaries containing the query results
    """
    try:
        result = await db.execute(query, params)
        rows = result.fetchall()
        columns = result.keys()
        data = [dict(zip(columns, row)) for row in rows]
        logger.info(f"Fetched {len(data)} rows from query: {query}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

async def async_execute_query(db: AsyncSession, query: str, params: dict) -> None:
    """
    Execute a write query (INSERT, UPDATE, DELETE) on the database.
    Args:
        db: AsyncSession dependency
        query: SQL query string
        params: Parameters for the query
    """
    try:
        await db.execute(query, params)
        await db.commit()
        logger.info(f"Executed query: {query}")
    except Exception as e:
        await db.rollback()
        logger.error(f"Error executing query: {str(e)}")
        raise