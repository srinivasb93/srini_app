import logging
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException

from backend.app.services import init_upstox_api, init_zerodha_api
from backend.app.services import TokenExpiredError
from models import User

logger = logging.getLogger(__name__)

# Global user APIs cache
user_apis = {}

async def initialize_user_apis(user_id: str, db: AsyncSession, force_reinitialize: bool = False):
    """Initialize trading APIs for user"""
    global user_apis

    if user_id in user_apis and not force_reinitialize:
        return user_apis[user_id]

    try:
        logger.info(f"Initializing APIs for user {user_id}")

        # Get user record with all tokens
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Initialize Upstox APIs
        upstox_apis = await init_upstox_api(db, user_id)

        # Initialize Zerodha APIs with proper token handling
        kite_apis = await init_zerodha_api(db, user_id)

        # Store in global user_apis with access tokens
        user_apis[user_id] = {
            "upstox": upstox_apis,
            "zerodha": kite_apis,
            "access_token": user.upstox_access_token,  # Add this
            "zerodha_access_token": user.zerodha_access_token  # Add this
        }

        logger.info(f"APIs initialized successfully for user {user_id}")

    except TokenExpiredError as e:
        logger.info(f"Token expired for user {user_id}: {e.message}")
        user_apis[user_id] = {
            "upstox": {"user": None, "order": None, "portfolio": None, "market_data": None, "history": None},
            "zerodha": {"kite": None},
            "access_token": None,
            "zerodha_access_token": None
        }
    except Exception as e:
        logger.error(f"Failed to initialize APIs for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"API initialization failed: {str(e)}")

    return user_apis[user_id]

async def get_cached_user_token(user_id: str, db: AsyncSession, broker: str = "upstox") -> str:
    """Get user token from cache or database"""
    if user_id not in user_apis:
        await initialize_user_apis(user_id, db)

    # Return appropriate token based on broker
    if broker.lower() == "zerodha":
        return user_apis[user_id].get("zerodha_access_token")
    else:
        return user_apis[user_id].get("access_token")