import logging
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException

from backend.app.services import init_upstox_api, init_zerodha_api
from backend.app.services import TokenExpiredError
from .models import User

logger = logging.getLogger(__name__)

# Global user APIs cache
user_apis = {}

async def initialize_user_apis(user_id: str, db: AsyncSession, force_reinitialize: bool = False):
    """Initialize trading APIs for user"""
    global user_apis

    try:
        # Always load the latest user tokens so cached entries can be validated
        result = await db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalars().first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        cached_entry = user_apis.get(user_id)
        now = datetime.now()

        def _token_valid(token: Optional[str], expiry: Optional[datetime]) -> bool:
            return bool(token and expiry and now < expiry)

        should_refresh = force_reinitialize or cached_entry is None

        if cached_entry and not should_refresh:
            upstox_token_valid = _token_valid(user.upstox_access_token, user.upstox_access_token_expiry)
            zerodha_token_valid = _token_valid(user.zerodha_access_token, user.zerodha_access_token_expiry)

            if upstox_token_valid:
                if (
                    not cached_entry["upstox"].get("order")
                    or cached_entry.get("access_token") != user.upstox_access_token
                ):
                    should_refresh = True
            else:
                # Tokens are no longer valid; ensure cached APIs reflect the disconnect
                cached_entry["upstox"] = {
                    "user": None,
                    "order": None,
                    "portfolio": None,
                    "market_data": None,
                    "history": None,
                }
                cached_entry["access_token"] = None

            if zerodha_token_valid:
                if (
                    not cached_entry["zerodha"].get("kite")
                    or cached_entry.get("zerodha_access_token") != user.zerodha_access_token
                ):
                    should_refresh = True
            else:
                cached_entry["zerodha"] = {"kite": None}
                cached_entry["zerodha_access_token"] = None

            if not should_refresh:
                return cached_entry

        logger.info(f"Initializing APIs for user {user_id}")

        # Initialize Upstox APIs
        upstox_apis = await init_upstox_api(db, user_id)

        # Initialize Zerodha APIs with proper token handling
        kite_apis = await init_zerodha_api(db, user_id)

        # Store in global user_apis with access tokens
        user_apis[user_id] = {
            "upstox": upstox_apis,
            "zerodha": kite_apis,
            "access_token": user.upstox_access_token,
            "zerodha_access_token": user.zerodha_access_token,
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
