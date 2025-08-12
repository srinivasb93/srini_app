from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging
from typing import List, Dict, Optional
import re
import os, sys
# Add the project_root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)
from backend.app.database import get_db, get_nsedata_db
from backend.app.auth import oauth2_scheme

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])


def sanitize_input(value: str) -> str:
    """Sanitize input to prevent SQL injection."""
    # Remove dangerous characters and limit length
    if not isinstance(value, str):
        raise ValueError("Input must be a string")
    value = value[:500]  # Increased limit for complex WHERE clauses
    # Allow alphanumeric, underscores, colons, basic punctuation, quotes, and SQL operators
    # Updated regex to allow single quotes, parentheses, and common SQL operators
    if not re.match(r"^[\w:.=\-\s'\"()><,|&%]+$", value):
        raise ValueError("Invalid characters in input")
    return value


def sanitize_filter(filter_str: str) -> str:
    """Enhanced sanitization specifically for SQL WHERE clauses."""
    if not isinstance(filter_str, str):
        raise ValueError("Filter must be a string")

    # Limit length
    filter_str = filter_str[:500]

    # Allow more SQL-specific characters for WHERE clauses
    # Includes: letters, numbers, spaces, quotes, operators, parentheses, etc.
    allowed_pattern = r"^[\w\s'\"()=<>!.,\-+*/%|&]+$"

    if not re.match(allowed_pattern, filter_str):
        raise ValueError("Invalid characters in filter clause")

    # Basic protection against common SQL injection patterns
    dangerous_patterns = [
        r';\s*(drop|delete|insert|update|create|alter)\s+',
        r'--.*',
        r'/\*.*\*/',
        r'\b(union|exec|execute)\b'
    ]

    filter_lower = filter_str.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, filter_lower, re.IGNORECASE):
            raise ValueError("Potentially dangerous SQL pattern detected")

    return filter_str


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
        limit: Optional[int] = Query(3000, ge=1, le=10000, description="Max rows to return"),
        offset: Optional[int] = Query(0, ge=0, description="Row offset for pagination"),
        required_db: str = Query("trading", description="Database to query, e.g., 'trading' or 'nsedata'"),
        trading_db: AsyncSession = Depends(get_db),
        nsedata_db: AsyncSession = Depends(get_nsedata_db),
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
        db = trading_db if required_db == "trading" else nsedata_db
        # Sanitize table name
        table_name = sanitize_input(table_name)

        # FIXED: Handle Query objects when called directly (not through FastAPI route)
        # Extract actual values from Query objects if present
        from fastapi.params import Query as QueryParam

        if isinstance(columns, QueryParam):
            columns = None
        if isinstance(filters, QueryParam):
            filters = None
        if isinstance(limit, QueryParam):
            limit = 1000
        if isinstance(offset, QueryParam):
            offset = 0
        if isinstance(required_db, QueryParam):
            required_db = "trading"

        # DEBUG: Add logging to see what values are actually received
        # logger.info(f"Received parameters - columns: {columns}, filters: {filters}")
        # logger.info(f"columns type: {type(columns)}, filters type: {type(filters)}")
        # logger.info(f"columns bool: {bool(columns)}, filters bool: {bool(filters)}")

        # Get table columns
        table_columns = await get_table_columns(db, table_name)
        selected_columns = validate_columns(columns, table_columns)

        # Build SQL query
        columns_str = ", ".join([f'"{col}"' for col in selected_columns])  # Quote each column name
        query_str = f"SELECT {columns_str} FROM \"{table_name}\""  # Always quote table name for consistency

        params = {}

        # FIXED: Properly check for filters - handle empty strings and None
        if filters and isinstance(filters, str) and filters.strip():  # Check for non-empty string after stripping whitespace
            # Basic filter sanitization (limited to prevent injection)
            filters = sanitize_filter(filters)
            query_str += f" WHERE {filters}"

        query_str += " ORDER BY 1"  # Order by first column for consistency
        if limit:
            query_str += " LIMIT :limit"
            params["limit"] = limit
        if offset:
            query_str += " OFFSET :offset"
            params["offset"] = offset

        logger.debug(f"Executing query: {query_str}")
        logger.debug(f"Query parameters: {params}")

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