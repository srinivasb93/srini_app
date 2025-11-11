import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from .models import User
from .schemas import UserCreate, UserResponse
from .database import get_db
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secure-jwt-secret-key")
ALGORITHM = "HS256"
# Increased to 8 hours for better user experience
# Can be configured via environment variable JWT_ACCESS_TOKEN_EXPIRE_MINUTES
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Token Response Model
class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None  # seconds until expiration

# User Manager
class UserManager:
    @staticmethod
    async def create_user(user_create: UserCreate, db: AsyncSession):
        hashed_password = pwd_context.hash(user_create.password)
        db_user = User(
            user_id=str(user_create.user_id),
            email=user_create.email,
            hashed_password=hashed_password,
            upstox_api_key=user_create.upstox_api_key,
            upstox_api_secret=user_create.upstox_api_secret,
            zerodha_api_key=user_create.zerodha_api_key,
            zerodha_api_secret=user_create.zerodha_api_secret,
            created_at=datetime.utcnow()
        )
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        logger.info(f"User {db_user.user_id} registered")
        return db_user

    @staticmethod
    async def authenticate_user(email: str, password: str, db: AsyncSession):
        result = await db.execute(select(User).filter(User.email == email))
        user = result.scalars().first()
        if not user or not pwd_context.verify(password, user.hashed_password):
            logger.warning(f"Failed login attempt for email: {email}")
            return None
        return user

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def create_refresh_token(data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[str]:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            token_type_in_payload: str = payload.get("type")
            
            # Verify token type matches expected type
            if token_type_in_payload != token_type:
                logger.warning(f"Token type mismatch. Expected: {token_type}, Got: {token_type_in_payload}")
                raise credentials_exception
                
            if user_id is None:
                raise credentials_exception
            return user_id
        except JWTError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            raise credentials_exception

    @staticmethod
    async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise credentials_exception
            
            # Fetch user from database
            result = await db.execute(select(User).filter(User.user_id == user_id))
            user = result.scalars().first()
            if user is None:
                raise credentials_exception
            
            return {
                "sub": user.user_id,
                "email": user.email,
                "user_id": user.user_id
            }
        except JWTError:
            raise credentials_exception

# Create a dependency function for easier import
async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    """Dependency function to get current user from JWT token"""
    return await UserManager.get_current_user(token, db)
