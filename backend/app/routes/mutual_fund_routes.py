"""
Enhanced Mutual Fund Data Routes Module
Comprehensive mutual fund data fetching using mftool library
Optimized for algo trading applications with SIP integration
"""


import logging
import asyncio
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union, Any
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import time
from functools import lru_cache
import re
from backend.app.routes.sip_routes import get_current_user

# Import your existing dependencies
from backend.app.database import get_db

# MFTool import
try:
    from mftool import Mftool

    MFTOOL_AVAILABLE = True
except ImportError:
    MFTOOL_AVAILABLE = False
    logging.warning("MFTool library not available. Install with: pip install mftool")

logger = logging.getLogger(__name__)

# Create router
mf_router = APIRouter(prefix="/api/mutual-funds", tags=["mutual-funds"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class MutualFundRequest(BaseModel):
    """Request model for mutual fund data"""
    scheme_code: str = Field(..., description="Mutual fund scheme code")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")

    @validator('scheme_code')
    def validate_scheme_code(cls, v):
        return v.strip()

    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        if v:
            try:
                datetime.strptime(v, '%Y-%m-%d')
                return v
            except ValueError:
                raise ValueError('Date must be in YYYY-MM-DD format')


class NAVData(BaseModel):
    """NAV data point"""
    date: str
    nav: float


class MutualFundInfo(BaseModel):
    """Mutual fund basic information"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scheme_code: str
    scheme_name: str
    fund_house: str
    scheme_type: str
    scheme_category: str
    current_nav: float
    nav_date: str


class MutualFundHistoryResponse(BaseModel):
    """Mutual fund historical data response"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scheme_code: str
    scheme_name: str
    fund_house: str
    nav_data: List[NAVData]
    total_points: int
    start_date: str
    end_date: str
    last_updated: datetime


class MutualFundSearchRequest(BaseModel):
    """Search request for mutual funds"""
    query: str = Field(..., description="Search term")
    fund_house: Optional[str] = Field(None, description="Filter by fund house")
    scheme_type: Optional[str] = Field(None, description="Filter by scheme type")
    limit: int = Field(default=20, description="Number of results")


class SIPCalculationRequest(BaseModel):
    """SIP calculation request"""
    scheme_code: str
    monthly_amount: float = Field(..., gt=0, description="Monthly SIP amount")
    duration_years: int = Field(..., gt=0, description="Investment duration in years")
    start_date: Optional[str] = Field(None, description="SIP start date")


class SIPCalculationResponse(BaseModel):
    """SIP calculation response"""
    scheme_code: str
    scheme_name: str
    monthly_amount: float
    duration_years: int
    total_investment: float
    final_value: float
    absolute_return: float
    absolute_return_percent: float
    annualized_return: float
    calculations: List[Dict[str, Any]]


class MutualFundPerformance(BaseModel):
    """Mutual fund performance metrics"""
    scheme_code: str
    scheme_name: str
    current_nav: float
    returns_1m: Optional[float] = None
    returns_3m: Optional[float] = None
    returns_6m: Optional[float] = None
    returns_1y: Optional[float] = None
    returns_3y: Optional[float] = None
    returns_5y: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None


class FundHouseResponse(BaseModel):
    """Fund house information"""
    fund_house: str
    total_schemes: int
    aum: Optional[float] = None


class SchemeTypeResponse(BaseModel):
    """Scheme type information"""
    scheme_type: str
    total_schemes: int
    categories: List[str]


# =============================================================================
# MUTUAL FUND DATA MANAGER
# =============================================================================

class MutualFundDataManager:
    """Enhanced mutual fund data manager with caching and performance optimization"""

    def __init__(self):
        self.mf = None
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour cache for MF data
        self.scheme_cache = {}
        self.initialize_mftool()

    def initialize_mftool(self):
        """Initialize MFTool instance"""
        if MFTOOL_AVAILABLE:
            try:
                self.mf = Mftool()
                logger.info("MFTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MFTool: {e}")
                self.mf = None
        else:
            logger.warning("MFTool not available")

    def _get_cache_key(self, *args) -> str:
        """Generate cache key"""
        return "_".join(str(arg) for arg in args)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is valid"""
        if cache_key not in self.cache:
            return False
        return time.time() - self.cache[cache_key]['timestamp'] < self.cache_expiry

    async def get_scheme_details(self, scheme_code: str) -> Dict[str, Any]:
        """Get mutual fund scheme details"""
        if not self.mf:
            raise ValueError("MFTool not available")

        cache_key = self._get_cache_key("scheme_details", scheme_code)

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            scheme_details = await loop.run_in_executor(
                None,
                self.mf.get_scheme_details,
                scheme_code
            )

            if not scheme_details:
                raise ValueError(f"Scheme {scheme_code} not found")

            # Cache the result
            self.cache[cache_key] = {
                'data': scheme_details,
                'timestamp': time.time()
            }

            return scheme_details

        except Exception as e:
            logger.error(f"Error fetching scheme details for {scheme_code}: {e}")
            raise

    async def get_scheme_historical_nav(self, scheme_code: str, start_date: str = None, end_date: str = None) -> List[
        Dict]:
        """Get historical NAV data for a scheme"""
        if not self.mf:
            raise ValueError("MFTool not available")

        cache_key = self._get_cache_key("nav_history", scheme_code, start_date, end_date)

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        try:
            loop = asyncio.get_event_loop()

            if start_date and end_date:
                # Get historical data for date range
                nav_data = await loop.run_in_executor(
                    None,
                    self.mf.get_scheme_historical_nav_year,
                    scheme_code,
                    int(start_date[:4])  # Extract year
                )
            else:
                # Get all available historical data
                nav_data = await loop.run_in_executor(
                    None,
                    self.mf.get_scheme_historical_nav,
                    scheme_code
                )

            if not nav_data:
                raise ValueError(f"No NAV data found for scheme {scheme_code}")

            # Filter by date range if specified
            if start_date and end_date:
                nav_data = self._filter_nav_by_date_range(nav_data, start_date, end_date)

            # Cache the result
            self.cache[cache_key] = {
                'data': nav_data,
                'timestamp': time.time()
            }

            return nav_data

        except Exception as e:
            logger.error(f"Error fetching NAV history for {scheme_code}: {e}")
            raise

    def _filter_nav_by_date_range(self, nav_data: List[Dict], start_date: str, end_date: str) -> List[Dict]:
        """Filter NAV data by date range"""
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            filtered_data = []
            for nav_point in nav_data:
                nav_date = datetime.strptime(nav_point['date'], '%d-%m-%Y')
                if start_dt <= nav_date <= end_dt:
                    filtered_data.append(nav_point)

            return filtered_data

        except Exception as e:
            logger.error(f"Error filtering NAV data: {e}")
            return nav_data

    async def search_schemes(self, query: str, fund_house: str = None, scheme_type: str = None, limit: int = 20) -> \
    List[Dict]:
        """Search mutual fund schemes"""
        if not self.mf:
            raise ValueError("MFTool not available")

        cache_key = self._get_cache_key("search", query, fund_house, scheme_type, limit)

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        try:
            # Get all schemes
            loop = asyncio.get_event_loop()
            all_schemes = await loop.run_in_executor(None, self.mf.get_scheme_codes)

            if not all_schemes:
                return []

            # Filter schemes based on search criteria
            filtered_schemes = []
            query_lower = query.lower()

            for scheme_code, scheme_name in all_schemes.items():
                scheme_name_lower = scheme_name.lower()

                # Check if query matches scheme name
                if query_lower in scheme_name_lower:
                    # Additional filtering can be added here
                    filtered_schemes.append({
                        'scheme_code': scheme_code,
                        'scheme_name': scheme_name
                    })

                    if len(filtered_schemes) >= limit:
                        break

            # Cache the result
            self.cache[cache_key] = {
                'data': filtered_schemes,
                'timestamp': time.time()
            }

            return filtered_schemes

        except Exception as e:
            logger.error(f"Error searching schemes: {e}")
            raise

    async def calculate_sip_returns(self, scheme_code: str, monthly_amount: float,
                                    duration_years: int, start_date: str = None) -> Dict[str, Any]:
        """Calculate SIP returns for a mutual fund"""
        try:
            # Get historical NAV data
            end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=duration_years * 365)).strftime('%Y-%m-%d')

            nav_data = await self.get_scheme_historical_nav(scheme_code, start_date, end_date)
            scheme_details = await self.get_scheme_details(scheme_code)

            if not nav_data:
                raise ValueError("Insufficient NAV data for SIP calculation")

            # Convert NAV data to DataFrame for easier processing
            df = pd.DataFrame(nav_data)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'])
            df = df.sort_values('date')

            # Calculate SIP returns
            total_investment = 0
            total_units = 0
            calculations = []

            # Simulate monthly SIP investments
            current_date = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)

            while current_date <= end_date_dt:
                # Find NAV for the current month
                month_navs = df[df['date'].dt.to_period('M') == current_date.to_period('M')]

                if not month_navs.empty:
                    # Use first available NAV of the month
                    nav_value = month_navs.iloc[0]['nav']
                    units_purchased = monthly_amount / nav_value

                    total_investment += monthly_amount
                    total_units += units_purchased

                    calculations.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'nav': nav_value,
                        'amount_invested': monthly_amount,
                        'units_purchased': units_purchased,
                        'cumulative_units': total_units,
                        'cumulative_investment': total_investment
                    })

                # Move to next month
                current_date = current_date + pd.DateOffset(months=1)

            # Calculate final value
            final_nav = df.iloc[-1]['nav']
            final_value = total_units * final_nav
            absolute_return = final_value - total_investment
            absolute_return_percent = (absolute_return / total_investment) * 100 if total_investment > 0 else 0

            # Calculate annualized return
            years_invested = len(calculations) / 12
            annualized_return = ((final_value / total_investment) ** (
                        1 / years_invested) - 1) * 100 if years_invested > 0 and total_investment > 0 else 0

            return {
                'scheme_code': scheme_code,
                'scheme_name': scheme_details.get('scheme_name', ''),
                'monthly_amount': monthly_amount,
                'duration_years': duration_years,
                'total_investment': total_investment,
                'final_value': final_value,
                'absolute_return': absolute_return,
                'absolute_return_percent': absolute_return_percent,
                'annualized_return': annualized_return,
                'calculations': calculations
            }

        except Exception as e:
            logger.error(f"Error calculating SIP returns: {e}")
            raise

    async def calculate_fund_performance(self, scheme_code: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for a fund"""
        try:
            # Get 5 years of data for comprehensive analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5 * 365)

            nav_data = await self.get_scheme_historical_nav(
                scheme_code,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            scheme_details = await self.get_scheme_details(scheme_code)

            if not nav_data:
                raise ValueError("Insufficient data for performance calculation")

            # Convert to DataFrame
            df = pd.DataFrame(nav_data)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'])
            df = df.sort_values('date').reset_index(drop=True)

            # Calculate returns
            df['daily_return'] = df['nav'].pct_change()

            current_nav = df.iloc[-1]['nav']

            # Calculate period returns
            def get_period_return(days):
                try:
                    target_date = end_date - timedelta(days=days)
                    closest_data = df[df['date'] >= target_date].iloc[0]
                    past_nav = closest_data['nav']
                    return ((current_nav / past_nav) - 1) * 100
                except:
                    return None

            returns_1m = get_period_return(30)
            returns_3m = get_period_return(90)
            returns_6m = get_period_return(180)
            returns_1y = get_period_return(365)
            returns_3y = get_period_return(3 * 365)
            returns_5y = get_period_return(5 * 365)

            # Calculate volatility (annualized)
            volatility = df['daily_return'].std() * np.sqrt(252) * 100 if len(df) > 1 else None

            # Calculate max drawdown
            df['cumulative'] = (1 + df['daily_return']).cumprod()
            df['running_max'] = df['cumulative'].cummax()
            df['drawdown'] = (df['cumulative'] / df['running_max']) - 1
            max_drawdown = df['drawdown'].min() * 100 if len(df) > 1 else None

            # Simple Sharpe ratio (assuming 6% risk-free rate)
            risk_free_rate = 0.06
            if returns_1y and volatility:
                sharpe_ratio = (returns_1y / 100 - risk_free_rate) / (volatility / 100)
            else:
                sharpe_ratio = None

            return {
                'scheme_code': scheme_code,
                'scheme_name': scheme_details.get('scheme_name', ''),
                'current_nav': current_nav,
                'returns_1m': returns_1m,
                'returns_3m': returns_3m,
                'returns_6m': returns_6m,
                'returns_1y': returns_1y,
                'returns_3y': returns_3y,
                'returns_5y': returns_5y,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'alpha': None,  # Would need benchmark data
                'beta': None  # Would need benchmark data
            }

        except Exception as e:
            logger.error(f"Error calculating fund performance: {e}")
            raise


# Global instance
mf_data_manager = MutualFundDataManager()


# =============================================================================
# API ENDPOINTS
# =============================================================================

@mf_router.get("/schemes/{scheme_code}/details")
async def get_mutual_fund_details(
        scheme_code: str,
        user_id: str = Depends(get_current_user)
):
    """Get detailed information about a mutual fund scheme"""
    try:
        scheme_details = await mf_data_manager.get_scheme_details(scheme_code)

        return MutualFundInfo(
            scheme_code=scheme_code,
            scheme_name=scheme_details.get('scheme_name', ''),
            fund_house=scheme_details.get('fund_house', ''),
            scheme_type=scheme_details.get('scheme_type', ''),
            scheme_category=scheme_details.get('scheme_category', ''),
            current_nav=float(scheme_details.get('nav', 0)),
            nav_date=scheme_details.get('last_updated', '')
        )

    except Exception as e:
        logger.error(f"Error fetching scheme details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mf_router.post("/schemes/historical", response_model=MutualFundHistoryResponse)
async def get_mutual_fund_history(
        request: MutualFundRequest,
        background_tasks: BackgroundTasks,
        user_id: str = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """Get historical NAV data for a mutual fund scheme"""
    try:
        # Get scheme details and historical data
        scheme_details = await mf_data_manager.get_scheme_details(request.scheme_code)
        nav_data = await mf_data_manager.get_scheme_historical_nav(
            request.scheme_code,
            request.start_date,
            request.end_date
        )

        # Convert NAV data to response format
        nav_points = [
            NAVData(date=point['date'], nav=float(point['nav']))
            for point in nav_data
        ]

        # Store in database in background
        background_tasks.add_task(
            store_mutual_fund_data,
            db,
            request.scheme_code,
            nav_data
        )

        return MutualFundHistoryResponse(
            scheme_code=request.scheme_code,
            scheme_name=scheme_details.get('scheme_name', ''),
            fund_house=scheme_details.get('fund_house', ''),
            nav_data=nav_points,
            total_points=len(nav_points),
            start_date=request.start_date or nav_points[0].date if nav_points else '',
            end_date=request.end_date or nav_points[-1].date if nav_points else '',
            last_updated=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error fetching mutual fund history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mf_router.post("/search")
async def search_mutual_funds(
        request: MutualFundSearchRequest,
        user_id: str = Depends(get_current_user)
):
    """Search for mutual fund schemes"""
    try:
        results = await mf_data_manager.search_schemes(
            query=request.query,
            fund_house=request.fund_house,
            scheme_type=request.scheme_type,
            limit=request.limit
        )

        return {
            'query': request.query,
            'results': results,
            'total_found': len(results),
            'filters': {
                'fund_house': request.fund_house,
                'scheme_type': request.scheme_type
            }
        }

    except Exception as e:
        logger.error(f"Error searching mutual funds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mf_router.post("/sip/calculate", response_model=SIPCalculationResponse)
async def calculate_sip_returns(
        request: SIPCalculationRequest,
        user_id: str = Depends(get_current_user)
):
    """Calculate SIP returns for a mutual fund scheme"""
    try:
        result = await mf_data_manager.calculate_sip_returns(
            scheme_code=request.scheme_code,
            monthly_amount=request.monthly_amount,
            duration_years=request.duration_years,
            start_date=request.start_date
        )

        return SIPCalculationResponse(**result)

    except Exception as e:
        logger.error(f"Error calculating SIP returns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mf_router.get("/schemes/{scheme_code}/performance")
async def get_fund_performance(
        scheme_code: str,
        user_id: str = Depends(get_current_user)
):
    """Get comprehensive performance metrics for a mutual fund"""
    try:
        performance = await mf_data_manager.calculate_fund_performance(scheme_code)
        return MutualFundPerformance(**performance)

    except Exception as e:
        logger.error(f"Error calculating fund performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mf_router.get("/fund-houses")
async def get_fund_houses(
        user_id: str = Depends(get_current_user)
):
    """Get list of all fund houses"""
    try:
        if not mf_data_manager.mf:
            raise HTTPException(status_code=503, detail="MFTool not available")

        # This would need to be implemented by iterating through all schemes
        # For now, return a basic response
        return {
            'message': 'Fund houses endpoint - implementation depends on mftool capabilities',
            'available': MFTOOL_AVAILABLE
        }

    except Exception as e:
        logger.error(f"Error fetching fund houses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mf_router.get("/categories")
async def get_scheme_categories(
        user_id: str = Depends(get_current_user)
):
    """Get list of mutual fund categories"""
    try:
        # Common mutual fund categories
        categories = [
            "Equity Funds",
            "Debt Funds",
            "Hybrid Funds",
            "Solution Oriented Funds",
            "Other Funds",
            "Large Cap Funds",
            "Mid Cap Funds",
            "Small Cap Funds",
            "Multi Cap Funds",
            "Sectoral/Thematic Funds"
        ]

        return {
            'categories': categories,
            'total_categories': len(categories),
            'last_updated': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mf_router.get("/top-performing")
async def get_top_performing_funds(
        category: Optional[str] = Query(None, description="Filter by category"),
        period: str = Query(default="1y", description="Performance period (1m, 3m, 6m, 1y, 3y, 5y)"),
        limit: int = Query(default=10, description="Number of results"),
        user_id: str = Depends(get_current_user)
):
    """Get top performing mutual funds"""
    try:
        # This would require analyzing performance data for multiple funds
        # For now, return a placeholder response
        return {
            'message': 'Top performing funds endpoint - requires comprehensive fund analysis',
            'filters': {
                'category': category,
                'period': period,
                'limit': limit
            },
            'note': 'Implementation requires iterating through multiple schemes and calculating performance'
        }

    except Exception as e:
        logger.error(f"Error fetching top performing funds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def store_mutual_fund_data(db: AsyncSession, scheme_code: str, nav_data: List[Dict]):
    """Store mutual fund NAV data in database (background task)"""
    try:
        # Create table name
        table_name = f"mf_{scheme_code}"

        # Create table if not exists
        create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date DATE PRIMARY KEY,
                nav REAL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """

        await db.execute(text(create_query))

        # Insert data
        for point in nav_data:
            # Convert date format from DD-MM-YYYY to YYYY-MM-DD
            date_parts = point['date'].split('-')
            formatted_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"

            insert_query = f"""
                INSERT INTO {table_name} (date, nav)
                VALUES (:date, :nav)
                ON CONFLICT (date) DO UPDATE SET
                    nav = EXCLUDED.nav
            """

            await db.execute(text(insert_query), {
                'date': formatted_date,
                'nav': float(point['nav'])
            })

        await db.commit()
        logger.info(f"Stored {len(nav_data)} NAV points for scheme {scheme_code}")

    except Exception as e:
        logger.error(f"Error storing mutual fund data for {scheme_code}: {e}")
        await db.rollback()


# =============================================================================
# HEALTH CHECK
# =============================================================================

@mf_router.get("/health")
async def health_check():
    """Health check endpoint for mutual funds module"""
    return {
        'status': 'healthy',
        'mftool_available': MFTOOL_AVAILABLE,
        'cache_size': len(mf_data_manager.cache),
        'timestamp': datetime.now().isoformat()
    }