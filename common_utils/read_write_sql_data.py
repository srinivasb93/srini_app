import logging
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_sql_data(df: pd.DataFrame, table_name: str, load_type: str, index_required: bool, db: AsyncSession) -> None:
    """
    Load data from a DataFrame into the database using a bulk INSERT statement.

    Args:
        df: DataFrame containing the data
        table_name: Name of the table to load data into
        load_type: Type of load ('append' or 'replace')
        index_required: Whether to include the DataFrame index
        db: AsyncSession dependency
        database: Database name (not used in async context but kept for compatibility)
    """
    try:
        if df.empty:
            logger.info(f"No data to load into {table_name}")
            return

        if index_required:
            df = df.reset_index()

        # Convert DataFrame to list of dictionaries
        data = df.to_dict(orient="records")

        # Generate the INSERT query dynamically based on the DataFrame columns
        columns = list(data[0].keys())
        columns_str = ", ".join(columns)

        # Create a list of parameter sets for bulk insert
        # Each parameter set is a dictionary with unique parameter names
        param_sets = []
        for i, record in enumerate(data):
            # Create a new dictionary with unique parameter names for each row
            param_set = {f"{col}_{i}": value for col, value in record.items()}
            param_sets.append(param_set)

        # Generate the VALUES clause for all rows
        values_clauses = []
        for i in range(len(data)):
            placeholders = ", ".join([f":{col}_{i}" for col in columns])
            values_clauses.append(f"({placeholders})")
        values_str = ", ".join(values_clauses)

        # Construct the full INSERT query
        query = f"INSERT INTO {table_name} ({columns_str}) VALUES {values_str}"

        # If load_type is "replace", truncate the table first
        if load_type == "replace":
            await db.execute(text(f"TRUNCATE TABLE {table_name}"))
            logger.info(f"Truncated table {table_name} before loading")

        # Combine all parameters into a single dictionary
        all_params = {}
        for param_set in param_sets:
            all_params.update(param_set)

        # Execute the bulk insert
        await db.execute(text(query), all_params)
        await db.commit()
        logger.info(f"Loaded {len(data)} rows into {table_name} with load_type={load_type}")

    except Exception as e:
        await db.rollback()
        logger.error(f"Error loading data into {table_name}: {str(e)}")
        raise
