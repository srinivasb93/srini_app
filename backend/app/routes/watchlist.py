from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel
import os, sys

# Add the project_root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)
from backend.app.database import get_db
from backend.app.auth import oauth2_scheme, get_current_user

logger = logging.getLogger(__name__)

watchlist_router = APIRouter(prefix="/watchlist", tags=["watchlist"])

# Pydantic models
class WatchlistCreate(BaseModel):
    name: str

class WatchlistSymbol(BaseModel):
    symbol: str
    watchlist_name: Optional[str] = "Default"

class WatchlistSymbols(BaseModel):
    symbols: List[str]
    watchlist_name: Optional[str] = "Default"

# Initialize watchlist tables
async def init_watchlist_tables(db: AsyncSession):
    """Create watchlist tables if they don't exist"""
    try:
        # Create user_watchlists table
        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS user_watchlists (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, name)
            )
        """))

        # Create watchlist_symbols table
        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS watchlist_symbols (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                watchlist_name VARCHAR(255) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, watchlist_name, symbol)
            )
        """))

        await db.commit()
        logger.info("Watchlist tables created successfully")
    except Exception as e:
        logger.error(f"Error creating watchlist tables: {e}")
        await db.rollback()

@watchlist_router.post("/init-tables/")
async def init_tables(db: AsyncSession = Depends(get_db), token: str = Depends(oauth2_scheme)):
    """Initialize watchlist tables"""
    await init_watchlist_tables(db)
    return {"message": "Watchlist tables initialized"}

@watchlist_router.get("/")
async def get_user_watchlists(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all watchlists for the current user"""
    try:
        user_id = current_user.get("sub", "anonymous")

        # Get user's watchlists
        result = await db.execute(text("""
            SELECT name, created_at, updated_at,
                   (SELECT COUNT(*) FROM watchlist_symbols ws 
                    WHERE ws.user_id = uw.user_id AND ws.watchlist_name = uw.name) as symbol_count
            FROM user_watchlists uw
            WHERE user_id = :user_id
            ORDER BY name
        """), {"user_id": user_id})

        watchlists = [
            {
                "name": row[0],
                "created_at": row[1].isoformat() if row[1] else None,
                "updated_at": row[2].isoformat() if row[2] else None,
                "symbol_count": row[3]
            }
            for row in result.fetchall()
        ]

        # Ensure Default watchlist exists
        if not any(wl["name"] == "Default" for wl in watchlists):
            await create_watchlist_internal(db, user_id, "Default")
            watchlists.insert(0, {
                "name": "Default",
                "created_at": None,
                "updated_at": None,
                "symbol_count": 0
            })

        return {"watchlists": watchlists}

    except Exception as e:
        logger.error(f"Error fetching user watchlists: {e}")
        raise HTTPException(status_code=500, detail="Error fetching watchlists")

@watchlist_router.post("/create/")
async def create_watchlist(
    watchlist: WatchlistCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new watchlist for the current user"""
    try:
        user_id = current_user.get("sub", "anonymous")
        await create_watchlist_internal(db, user_id, watchlist.name)
        return {"message": f"Watchlist '{watchlist.name}' created successfully"}

    except Exception as e:
        if "unique constraint" in str(e).lower():
            raise HTTPException(status_code=400, detail="Watchlist with this name already exists")
        logger.error(f"Error creating watchlist: {e}")
        raise HTTPException(status_code=500, detail="Error creating watchlist")

async def create_watchlist_internal(db: AsyncSession, user_id: str, name: str):
    """Internal function to create a watchlist"""
    await db.execute(text("""
        INSERT INTO user_watchlists (user_id, name) 
        VALUES (:user_id, :name)
        ON CONFLICT (user_id, name) DO NOTHING
    """), {"user_id": user_id, "name": name})
    await db.commit()

@watchlist_router.delete("/{watchlist_name}/")
async def delete_watchlist(
    watchlist_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a watchlist and all its symbols"""
    if watchlist_name == "Default":
        raise HTTPException(status_code=400, detail="Cannot delete Default watchlist")

    try:
        user_id = current_user.get("sub", "anonymous")

        # Delete symbols first
        await db.execute(text("""
            DELETE FROM watchlist_symbols 
            WHERE user_id = :user_id AND watchlist_name = :watchlist_name
        """), {"user_id": user_id, "watchlist_name": watchlist_name})

        # Delete watchlist
        result = await db.execute(text("""
            DELETE FROM user_watchlists 
            WHERE user_id = :user_id AND name = :watchlist_name
        """), {"user_id": user_id, "watchlist_name": watchlist_name})

        await db.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Watchlist not found")

        return {"message": f"Watchlist '{watchlist_name}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting watchlist: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error deleting watchlist")

@watchlist_router.get("/{watchlist_name}/symbols/")
async def get_watchlist_symbols(
    watchlist_name: str,
    page: int = 0,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get symbols in a watchlist with pagination"""
    try:
        user_id = current_user.get("sub", "anonymous")

        # Get total count
        count_result = await db.execute(text("""
            SELECT COUNT(*) FROM watchlist_symbols 
            WHERE user_id = :user_id AND watchlist_name = :watchlist_name
        """), {"user_id": user_id, "watchlist_name": watchlist_name})

        total_count = count_result.scalar()

        # Get symbols with pagination
        offset = page * page_size
        result = await db.execute(text("""
            SELECT symbol, added_at FROM watchlist_symbols 
            WHERE user_id = :user_id AND watchlist_name = :watchlist_name
            ORDER BY added_at DESC
            LIMIT :page_size OFFSET :offset
        """), {
            "user_id": user_id,
            "watchlist_name": watchlist_name,
            "page_size": page_size,
            "offset": offset
        })

        symbols = [
            {
                "symbol": row[0],
                "added_at": row[1].isoformat() if row[1] else None
            }
            for row in result.fetchall()
        ]

        total_pages = (total_count + page_size - 1) // page_size

        return {
            "symbols": symbols,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_count": total_count,
                "page_size": page_size
            }
        }

    except Exception as e:
        logger.error(f"Error fetching watchlist symbols: {e}")
        raise HTTPException(status_code=500, detail="Error fetching symbols")

@watchlist_router.post("/{watchlist_name}/symbols/")
async def add_symbols_to_watchlist(
    watchlist_name: str,
    symbols_data: WatchlistSymbols,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Add multiple symbols to a watchlist"""
    try:
        user_id = current_user.get("sub", "anonymous")

        # Ensure watchlist exists
        await create_watchlist_internal(db, user_id, watchlist_name)

        added_count = 0
        skipped_count = 0

        for symbol in symbols_data.symbols:
            try:
                await db.execute(text("""
                    INSERT INTO watchlist_symbols (user_id, watchlist_name, symbol) 
                    VALUES (:user_id, :watchlist_name, :symbol)
                """), {
                    "user_id": user_id,
                    "watchlist_name": watchlist_name,
                    "symbol": symbol.strip().upper()
                })
                added_count += 1
            except Exception:
                # Symbol already exists, skip it
                skipped_count += 1

        await db.commit()

        return {
            "message": f"Added {added_count} symbol(s) to watchlist",
            "added_count": added_count,
            "skipped_count": skipped_count
        }

    except Exception as e:
        logger.error(f"Error adding symbols to watchlist: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error adding symbols")

@watchlist_router.delete("/{watchlist_name}/symbols/{symbol}/")
async def remove_symbol_from_watchlist(
    watchlist_name: str,
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Remove a symbol from a watchlist"""
    try:
        user_id = current_user.get("sub", "anonymous")

        result = await db.execute(text("""
            DELETE FROM watchlist_symbols 
            WHERE user_id = :user_id AND watchlist_name = :watchlist_name AND symbol = :symbol
        """), {
            "user_id": user_id,
            "watchlist_name": watchlist_name,
            "symbol": symbol.strip().upper()
        })

        await db.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Symbol not found in watchlist")

        return {"message": f"Symbol '{symbol}' removed from watchlist"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing symbol from watchlist: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error removing symbol")

@watchlist_router.delete("/{watchlist_name}/symbols/")
async def clear_watchlist(
    watchlist_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Clear all symbols from a watchlist"""
    try:
        user_id = current_user.get("sub", "anonymous")

        await db.execute(text("""
            DELETE FROM watchlist_symbols 
            WHERE user_id = :user_id AND watchlist_name = :watchlist_name
        """), {"user_id": user_id, "watchlist_name": watchlist_name})

        await db.commit()

        return {"message": f"Watchlist '{watchlist_name}' cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing watchlist: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error clearing watchlist")
