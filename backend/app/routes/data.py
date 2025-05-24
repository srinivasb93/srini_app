from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from backend.app.database import get_db
import logging
from typing import List, Dict, Optional
from backend.app.auth import oauth2_scheme
import re

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])


def sanitize_input(value: str) -> str:
    """Sanitize input to prevent SQL injection."""
    # Remove dangerous characters and limit length
    if not isinstance(value, str):
        raise ValueError("Input must be a string")
    value = value[:100]  # Limit length
    # Allow alphanumeric, underscores, colons, and basic punctuation
    if not re.match(r'^[\w:.=\- ]+$', value):
        raise ValueError("Invalid characters in input")
    return value


def validate_columns(columns: List[str], table_columns: List[str]) -> List[str]:
    """Validate requested columns against table schema."""
    if not columns:
        return table_columns
    invalid_columns = [col for col in columns if col not in table_columns]
    if invalid_columns:
        raise HTTPException(status_code=400, detail=f"Invalid columns: {invalid_columns}")
    return columns


async def get_table_columns(db: AsyncSession, table_name: str) -> List[str]:
    """Fetch column names for a table."""
    try:
        query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = :table_name
        """)
        result = await db.execute(query, {"table_name": table_name})
        columns = [row[0] for row in result.fetchall()]
        if not columns:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found")
        return columns
    except Exception as e:
        logger.error(f"Error fetching columns for table {table_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching table schema")


def serialize_row(row: tuple, columns: List[str]) -> Dict:
    """Convert SQL row to JSON-serializable dictionary."""
    result = {}
    for key, value in zip(columns, row):
        if hasattr(value, 'isoformat'):  # Handle datetime
            result[key] = value.isoformat()
        elif isinstance(value, (int, float, str, bool)) or value is None:
            result[key] = value
        else:  # Handle Decimal, etc.
            result[key] = str(value)
    return result


@router.get("/fetch-table-data/", response_model=Dict[str, List[Dict]])
async def fetch_table_data(
        table_name: str,
        columns: Optional[List[str]] = Query(None, description="Columns to select"),
        filters: Optional[str] = Query(None, description="SQL WHERE clause, e.g., instrument='NSE:RELIANCE'"),
        limit: Optional[int] = Query(1000, ge=1, le=10000, description="Max rows to return"),
        offset: Optional[int] = Query(0, ge=0, description="Row offset for pagination"),
        db: AsyncSession = Depends(get_db),
        token: str = Depends(oauth2_scheme)
):
    """
    Fetch data from any table in the database.

    Parameters:
    - table_name: Name of the table to query
    - columns: Optional list of column names to select (default: all columns)
    - filters: Optional SQL WHERE clause (e.g., "instrument='NSE:RELIANCE' AND timestamp >= '2025-05-20'")
    - limit: Maximum number of rows to return (default: 1000)
    - offset: Number of rows to skip for pagination (default: 0)

    Returns:
    - data: List of records as dictionaries
    """
    try:
        # Sanitize table name
        table_name = sanitize_input(table_name)

        # Get table columns
        table_columns = await get_table_columns(db, table_name)
        selected_columns = validate_columns(columns, table_columns)

        # Build SQL query
        columns_str = ", ".join(selected_columns)
        query_str = f"SELECT {columns_str} FROM {table_name}"

        params = {}
        if filters:
            # Basic filter sanitization (limited to prevent injection)
            filters = sanitize_input(filters)
            query_str += f" WHERE {filters}"

        query_str += " ORDER BY 1"  # Order by first column for consistency
        if limit:
            query_str += " LIMIT :limit"
            params["limit"] = limit
        if offset:
            query_str += " OFFSET :offset"
            params["offset"] = offset

        # Execute query
        query = text(query_str)
        result = await db.execute(query, params)
        data = result.fetchall()

        # Serialize results
        formatted_data = [serialize_row(row, selected_columns) for row in data]

        logger.info(f"Fetched {len(formatted_data)} rows from table {table_name}")
        return {"data": formatted_data}

    except ValueError as ve:
        logger.error(f"Invalid input: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        logger.error(f"Error fetching data from table {table_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")