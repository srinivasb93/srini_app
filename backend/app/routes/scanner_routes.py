"""
Comprehensive Stock Scanner & Analysis API
Advanced scanning, analytics, and individual stock analysis
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel, Field
from datetime import datetime
import math
import logging

from ..database import get_nsedata_db
from ..auth import get_current_user
from ..models import User

router = APIRouter(prefix="/scanner", tags=["Market Scanner"])

logger = logging.getLogger(__name__)

# Data Science Helper Functions
def clean_financial_value(value):
    """Clean financial data to prevent JSON serialization issues"""
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        # Handle string representations of infinity/NaN
        if str(value).lower() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
            return None
            
        # Handle actual infinity/NaN values
        try:
            if math.isnan(value) or math.isinf(value):
                return None
        except (TypeError, ValueError):
            return None
            
        return float(value)
    
    return value


def clean_financial_data(data):
    """Recursively clean financial data structure"""
    if isinstance(data, dict):
        return {k: clean_financial_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_financial_data(item) for item in data]
    else:
        return clean_financial_value(data)


def calculate_signal_strength(row):
    """Calculate trading signal strength based on multiple indicators"""
    score = 0
    
    # Volume-based signals (0-3 points)
    vol_ratio = getattr(row, 'Vol_Abv_Avg20', 0) or 0
    if vol_ratio > 3:
        score += 3
    elif vol_ratio > 2:
        score += 2
    elif vol_ratio > 1.5:
        score += 1
    
    # Momentum signals (0-3 points)
    reg_signal = getattr(row, 'Reg_Cross_Sig', '') or ''
    if 'Cross_Up' in reg_signal:
        score += 3
    elif 'Reg6_Abv_Reg18' in reg_signal:
        score += 2
    elif 'Cross_Down' in reg_signal:
        score -= 2
    elif 'Reg6_Blw_Reg18' in reg_signal:
        score -= 1
    
    # EMA signals (0-2 points)
    ema_signal = getattr(row, 'EMA20_Sig', '') or ''
    if 'Cross_Abv_20EMA' in ema_signal:
        score += 2
    elif 'Close_GT_20EMA' in ema_signal:
        score += 1
    elif 'Cross_Blw_20EMA' in ema_signal:
        score -= 2
    elif 'Close_LT_20EMA' in ema_signal:
        score -= 1
        
    # Breakout signals (0-2 points)
    breakout = getattr(row, 'Break_Sup_Res', '') or ''
    if 'Res_Broken_Up' in breakout:
        score += 2
    elif 'Sup_Broken_Down' in breakout:
        score -= 2
    
    return max(0, min(10, score))  # Clamp between 0-10


def get_trading_recommendation(signal_strength, vol_ratio, range_pct):
    """Generate trading recommendation based on analytics"""
    if signal_strength >= 7 and vol_ratio > 1.5:
        return "Strong Buy"
    elif signal_strength >= 5 and vol_ratio > 1.2:
        return "Buy"
    elif signal_strength <= 3 and vol_ratio > 1.5 and range_pct > 3:
        return "Strong Sell"
    elif signal_strength <= 4:
        return "Sell"
    else:
        return "Hold"

# Enhanced Scan Request Model based on actual data analysis
class ScannerCriteria(BaseModel):
    # Support & Resistance Analysis (based on EOD_analysis.py logic)
    support_resistance_signals: bool = Field(False, description="Include support/resistance signals")
    breakout_patterns: bool = Field(False, description="Look for breakout patterns")
    
    # EMA & Regression Analysis (based on actual EOD data)
    ema_alignment: bool = Field(False, description="Price above EMA alignment (20>60>200)")
    ema_cross_signals: bool = Field(False, description="EMA crossover signals")
    regression_trend: Optional[str] = Field(None, description="Regression trend: 'Cross_Up', 'Cross_Down', 'Reg6_Abv_Reg18', 'Reg6_Blw_Reg18'")
    
    # Volume Analysis (Most Reliable Indicator)
    min_volume_surge: float = Field(1.0, ge=0.5, le=10.0, description="Minimum volume surge vs 20-day average")
    volume_confirmation: bool = Field(False, description="Volume confirms price movement")
    
    # Price Action
    min_price: Optional[float] = Field(None, ge=1.0, description="Minimum stock price")
    max_price: Optional[float] = Field(None, description="Maximum stock price")
    min_daily_change: Optional[float] = Field(None, description="Minimum daily percentage change")
    max_daily_change: Optional[float] = Field(None, description="Maximum daily percentage change")
    
    # Volatility & Risk (based on Range_Pct and ATR from EOD_Summary)
    max_volatility: Optional[float] = Field(None, ge=0.1, description="Maximum volatility (Range_Pct)")
    min_atr: Optional[float] = Field(None, description="Minimum ATR value")
    narrow_range_breakout: bool = Field(False, description="Narrow range breakout candidates")
    narrow_range: Optional[bool] = Field(None, description="Narrow range filter")
    max_range_pct: Optional[float] = Field(None, description="Maximum range percentage")
    
    # 20-Day Breakout Analysis (based on Breakout_20 column)
    breakout_20_up: bool = Field(False, description="20-day high breakouts")
    breakout_20_down: bool = Field(False, description="20-day low breakdowns")
    
    # Specific Signal Types (based on actual Support/Resistance columns)
    support_signals: List[str] = Field(default=[], description="Specific support signals: 'Price_Abv_Supp', 'Price_Crs_Abv_Supp'")
    resistance_signals: List[str] = Field(default=[], description="Specific resistance signals: 'Price_Blw_Res', 'Price_Crs_Blw_Res'")
    
    # Legacy support for backward compatibility
    above_ema_20: Optional[bool] = Field(None, description="Legacy: Price above EMA20")
    above_ema_60: Optional[bool] = Field(None, description="Legacy: Price above EMA60")
    above_ema_200: Optional[bool] = Field(None, description="Legacy: Price above EMA200")
    
    # Output Controls
    sort_by: str = Field('close', description="Sort by: 'close', 'Pct_Chg_D', 'Vol_Abv_Avg20', 'Symbol'")
    sort_order: str = Field('desc', description="Sort order: 'asc' or 'desc'")
    limit: int = Field(50, ge=1, le=200, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Starting record offset")
    
    # Enhanced filters based on actual data
    min_volume: Optional[float] = Field(None, description="Minimum volume filter")
    volume_surge_threshold: Optional[float] = Field(None, description="Minimum volume surge vs average")
    breakout_signal: Optional[str] = Field(None, description="Breakout type: 'up', 'down', 'any'")
    near_52w_high_pct: Optional[float] = Field(None, description="Within X% of 52-week high")
    near_52w_low_pct: Optional[float] = Field(None, description="Within X% of 52-week low")

class StockAnalysis(BaseModel):
    symbol: str
    timestamp: datetime
    
    # Price data
    current_price: float
    open: float
    high: float
    low: float
    volume: int
    
    # Performance metrics
    daily_change: float
    weekly_change: float
    monthly_change: float
    yearly_change: float
    
    # Technical indicators
    ema_20: float
    ema_60: float
    ema_200: float
    atr: float
    
    # Volume analysis
    volume_vs_avg: float
    volume_signal: str
    
    # Trend analysis
    ema_signals: Dict[str, str]
    breakout_signal: Optional[str]
    
    # Risk metrics
    volatility: float
    range_pct: float
    narrow_range: bool
    
    # 52-week analysis
    high_52w: float
    low_52w: float
    from_52w_high_pct: float
    from_52w_low_pct: float
    
    # Trading signals
    signals: List[str]
    signal_strength: int
    recommendation: str

class MarketOverview(BaseModel):
    total_stocks: int
    market_sentiment: str
    avg_daily_change: float
    
    # Market breadth
    gainers: int
    losers: int
    unchanged: int
    
    # Technical overview
    above_200ema: int
    below_200ema: int
    breakouts_up: int
    breakouts_down: int
    
    # Volume analysis
    high_volume_stocks: int
    normal_volume_stocks: int
    low_volume_stocks: int
    
    # Volatility overview
    high_volatility: int
    normal_volatility: int
    low_volatility: int

class TopMover(BaseModel):
    symbol: str
    current_price: float
    daily_change: float
    weekly_change: float
    volume: int
    volume_vs_avg: float
    market_cap_category: Optional[str]
    signals: List[str]

@router.get("/full-data/agg")
async def get_agg_data_table(
    sort_by: str = Query("Symbol", description="Column to sort by"),
    sort_order: str = Query("asc", description="asc or desc"),
    limit: int = Query(100, description="Number of records"),
    offset: int = Query(0, description="Starting record"),
    symbol: Optional[str] = Query(None, description="Filter by specific symbol"),
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get complete AGG_DATA table with sorting and pagination"""
    try:
        # Build query with dynamic sorting
        sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
        
        # Build WHERE clause for symbol filter
        where_clause = ""
        params = {"limit": limit, "offset": offset}
        
        if symbol:
            where_clause = 'WHERE "Symbol" = :symbol'
            params["symbol"] = symbol.upper()
        
        query = f"""
        SELECT *,
               CASE 
                   WHEN "Pct_Chg_D" > 2 THEN 'strong_positive'
                   WHEN "Pct_Chg_D" > 0 THEN 'positive'
                   WHEN "Pct_Chg_D" < -2 THEN 'strong_negative'
                   WHEN "Pct_Chg_D" < 0 THEN 'negative'
                   ELSE 'neutral'
               END as daily_trend,
               CASE 
                   WHEN "Percent_Chg_W" > 5 THEN 'strong_positive'
                   WHEN "Percent_Chg_W" > 0 THEN 'positive'
                   WHEN "Percent_Chg_W" < -5 THEN 'strong_negative'
                   WHEN "Percent_Chg_W" < 0 THEN 'negative'
                   ELSE 'neutral'
               END as weekly_trend,
               CASE 
                   WHEN "close" / "high_52W" > 0.95 THEN 'near_52w_high'
                   WHEN "close" / "low_52W" < 1.1 THEN 'near_52w_low'
                   ELSE 'mid_range'
               END as range_position
        FROM "AGG_DATA"
        {where_clause}
        ORDER BY "{sort_by}" {sort_direction}
        LIMIT :limit OFFSET :offset
        """
        
        result = await db.execute(text(query), params)
        rows = result.fetchall()
        
        # Convert to list of dicts with NaN/infinity handling
        columns = result.keys()
        data = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                # Handle NaN and infinity values
                if isinstance(value, (int, float)):
                    if math.isnan(value) or math.isinf(value):
                        value = None
                row_dict[col] = value
            data.append(row_dict)
        
        # Get total count
        count_query = 'SELECT COUNT(*) as total FROM "AGG_DATA"'
        count_result = await db.execute(text(count_query))
        total_count = count_result.scalar()
        
        return {
            "data": data,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching AGG_DATA: {str(e)}")

@router.get("/full-data/eod")
async def get_eod_summary_table(
    sort_by: str = Query("Symbol", description="Column to sort by"),
    sort_order: str = Query("asc", description="asc or desc"),
    limit: int = Query(100, description="Number of records"),
    offset: int = Query(0, description="Starting record"),
    symbol: Optional[str] = Query(None, description="Filter by specific symbol"),
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get complete EOD_Summary table with sorting and pagination"""
    try:
        sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
        
        # Build WHERE clause for symbol filter
        where_clause = ""
        params = {"limit": limit, "offset": offset}
        
        if symbol:
            where_clause = 'WHERE "Symbol" = :symbol'
            params["symbol"] = symbol.upper()
        
        query = f"""
        SELECT *,
               CASE 
                   WHEN "Pct_Chg" > 2 THEN 'strong_bullish'
                   WHEN "Pct_Chg" > 0 THEN 'bullish'
                   WHEN "Pct_Chg" < -2 THEN 'strong_bearish'
                   WHEN "Pct_Chg" < 0 THEN 'bearish'
                   ELSE 'neutral'
               END as trend_status,
               CASE 
                   WHEN "Vol_Abv_Avg20" > 2 THEN 'super_high'
                   WHEN "Vol_Abv_Avg20" > 1.5 THEN 'high'
                   WHEN "Vol_Abv_Avg20" > 1 THEN 'normal'
                   ELSE 'low'
               END as volume_status,
               CASE 
                   WHEN "ATR" > 0 AND "Range" / "ATR" > 2 THEN 'high_volatility'
                   WHEN "ATR" > 0 AND "Range" / "ATR" > 1 THEN 'normal_volatility'
                   ELSE 'low_volatility'
               END as volatility_status
        FROM "EOD_Summary"
        {where_clause}
        ORDER BY "{sort_by}" {sort_direction}
        LIMIT :limit OFFSET :offset
        """
        
        result = await db.execute(text(query), params)
        rows = result.fetchall()
        
        # Convert to list of dicts with NaN/infinity handling
        columns = result.keys()
        data = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                # Handle NaN and infinity values
                if isinstance(value, (int, float)):
                    if math.isnan(value) or math.isinf(value):
                        value = None
                row_dict[col] = value
            data.append(row_dict)
        
        # Get total count
        count_query = 'SELECT COUNT(*) as total FROM "EOD_Summary"'
        count_result = await db.execute(text(count_query))
        total_count = count_result.scalar()
        
        return {
            "data": data,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching EOD_Summary: {str(e)}")

@router.get("/stocks/list")
async def get_all_stocks_list(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get list of all available stocks for dropdown"""
    try:
        query = """
        SELECT DISTINCT a."Symbol",
               a."close" as current_price,
               a."Pct_Chg_D" as daily_change,
               CASE 
                   WHEN a."volume" > 20000000 THEN 'High Liquidity'
                   WHEN a."volume" > 5000000 THEN 'Medium Liquidity'
                   ELSE 'Low Liquidity'
               END as liquidity_category
        FROM "AGG_DATA" a
        ORDER BY a."Symbol"
        """
        
        result = await db.execute(text(query))
        rows = result.fetchall()
        
        stocks = []
        for row in rows:
            stocks.append({
                "symbol": row[0],
                "current_price": row[1],
                "daily_change": row[2],
                "category": row[3]
            })
        
        return {"stocks": stocks, "total": len(stocks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stocks list: {str(e)}")

@router.get("/stock/{symbol}/analysis")
async def get_individual_stock_analysis(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get comprehensive analysis for individual stock"""
    try:
        query = """
        SELECT 
            a.*,
            e."Pct_Chg_5D", e."Pct_Chg_20D", e."Pct_Chg_365D",
            e."EMA_20", e."EMA_60", e."EMA_200",
            e."ATR", e."Range_ATR", e."Vol_Abv_Avg20",
            e."EMA20_Sig", e."EMA60_Sig", e."EMA200_Sig",
            e."Vol20_Sig", e."Breakout_20", e."Break_Sup_Res",
            e."Narrow_Range", e."Range_Pct", e."Avg_Range_Pct_20"
        FROM "AGG_DATA" a
        JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
        WHERE a."Symbol" = :symbol
        """
        
        result = await db.execute(text(query), {"symbol": symbol})
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Build comprehensive analysis
        analysis = {
            "basic_info": {
                "symbol": row.Symbol,
                "timestamp": row.timestamp,
                "current_price": row.close,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "volume": row.volume,
                "daily_range": row.Range_D
            },
            
            "performance": {
                "daily_change": row.Pct_Chg_D,
                "weekly_change": row.Percent_Chg_W,
                "monthly_change": row.Percent_Chg_M,
                "yearly_change": row.Percent_Chg_Y,
                "5d_change": row.Pct_Chg_5D,
                "20d_change": row.Pct_Chg_20D,
                "365d_change": row.Pct_Chg_365D,
                "3_year_returns": getattr(row, '3_Year_Returns', None),
                "5_year_returns": getattr(row, '5_Year_Returns', None),
                "max_returns": row.Max_Returns
            },
            
            "technical_indicators": {
                "ema_20": row.EMA_20,
                "ema_60": row.EMA_60,
                "ema_200": row.EMA_200,
                "weekly_ema_13": row.Wk_EMA_13,
                "weekly_ema_52": row.Wk_EMA_52,
                "monthly_ema_20": row.Mth_EMA_20,
                "atr": row.ATR,
                "range_atr_ratio": row.Range_ATR
            },
            
            "volume_analysis": {
                "current_volume": row.volume,
                "volume_vs_avg": row.Vol_Abv_Avg20,
                "volume_signal": row.Vol20_Sig,
                "weekly_volume": row.volume_W,
                "monthly_volume": row.volume_M,
                "yearly_volume": row.volume_Y,
                "max_weekly_volume": row.Max_Vol_W,
                "max_monthly_volume": row.Max_Vol_M
            },
            
            "range_analysis": {
                "52w_high": row.high_52W,
                "52w_low": row.low_52W,
                "52w_high_date": row.high_52W_Date,
                "52w_low_date": row.low_52W_Date,
                "from_52w_high_pct": ((row.high_52W - row.close) / row.high_52W * 100) if row.high_52W else 0,
                "from_52w_low_pct": ((row.close - row.low_52W) / row.low_52W * 100) if row.low_52W else 0,
                "ath_date": row.ATH_Date,
                "atl_date": row.ATL_Date,
                "6w_high": row.high_6W,
                "6w_low": row.low_6W,
                "6m_high": row.high_6M,
                "6m_low": row.low_6M
            },
            
            "volatility_metrics": {
                "range_pct": row.Range_Pct,
                "avg_range_pct_20": row.Avg_Range_Pct_20,
                "narrow_range": row.Narrow_Range,
                "daily_range": row.Range_D,
                "weekly_range": row.Range_W,
                "monthly_range": row.Range_M,
                "max_weekly_change": row.Max_Chg_W,
                "max_monthly_change": row.Max_Chg_M
            },
            
            "signals_and_trends": {
                "ema_20_signal": row.EMA20_Sig,
                "ema_60_signal": row.EMA60_Sig,
                "ema_200_signal": row.EMA200_Sig,
                "breakout_20": row.Breakout_20,
                "support_resistance_break": row.Break_Sup_Res,
                "volume_signal": row.Vol20_Sig
            },
            
            "linear_regression": {
                "6w_lr": row.LR_6_W,
                "6w_lr_low": row.low_LR_6_W,
                "6w_lr_high": row.high_LR_6_W,
                "6m_lr": row.LR_6_M,
                "6m_lr_low": row.low_LR_6_M,
                "6m_lr_high": row.high_LR_6_M
            }
        }
        
        # Generate trading signals
        signals = []
        signal_strength = 0
        
        # EMA signals
        if row.EMA20_Sig == "Close_GT_20EMA":
            signals.append("Above EMA 20")
            signal_strength += 1
        elif row.EMA20_Sig == "Close_LT_20EMA":
            signals.append("Below EMA 20")
            signal_strength -= 1
            
        if row.EMA200_Sig == "Close_GT_200EMA":
            signals.append("Above EMA 200 (Long-term Bullish)")
            signal_strength += 2
        elif row.EMA200_Sig == "Close_LT_200EMA":
            signals.append("Below EMA 200 (Long-term Bearish)")
            signal_strength -= 2
        
        # Volume signals
        if row.Vol_Abv_Avg20 > 2:
            signals.append("Super High Volume")
            signal_strength += 1
        elif row.Vol_Abv_Avg20 > 1.5:
            signals.append("High Volume")
            signal_strength += 1
        
        # Breakout signals
        if row.Breakout_20 == "Breakout_20_Up":
            signals.append("20-Day Breakout (Up)")
            signal_strength += 2
        elif row.Breakout_20 == "Breakout_20_Down":
            signals.append("20-Day Breakdown")
            signal_strength -= 2
        
        # Range analysis
        from_52w_high = ((row.high_52W - row.close) / row.high_52W * 100) if row.high_52W else 0
        from_52w_low = ((row.close - row.low_52W) / row.low_52W * 100) if row.low_52W else 0
        
        if from_52w_high < 5:
            signals.append("Near 52-Week High")
            signal_strength += 1
        elif from_52w_low < 10:
            signals.append("Near 52-Week Low")
            signal_strength += 1
        
        # Performance signals
        if row.Pct_Chg_D > 3:
            signals.append("Strong Daily Gain")
            signal_strength += 1
        elif row.Pct_Chg_D < -3:
            signals.append("Strong Daily Loss")
            signal_strength -= 1
        
        # Generate recommendation
        if signal_strength >= 4:
            recommendation = "Strong Buy"
        elif signal_strength >= 2:
            recommendation = "Buy"
        elif signal_strength >= 0:
            recommendation = "Hold"
        elif signal_strength >= -2:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"
        
        analysis["trading_summary"] = {
            "signals": signals,
            "signal_strength": signal_strength,
            "recommendation": recommendation,
            "risk_level": "High" if row.Range_Pct > 3 else "Medium" if row.Range_Pct > 1.5 else "Low"
        }
        
        # Clean all data using our helper function
        return clean_financial_data(analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock analysis: {str(e)}")

@router.get("/market-overview")
async def get_comprehensive_market_overview(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get comprehensive market overview with detailed analytics"""
    try:
        overview_query = """
        SELECT 
            COUNT(*) as total_stocks,
            AVG(a."Pct_Chg_D") as avg_daily_change,
            COUNT(CASE WHEN a."Pct_Chg_D" > 0 THEN 1 END) as gainers,
            COUNT(CASE WHEN a."Pct_Chg_D" < 0 THEN 1 END) as losers,
            COUNT(CASE WHEN a."Pct_Chg_D" = 0 THEN 1 END) as unchanged,
            COUNT(CASE WHEN e."Cls_Abv_EMA200" > 0 THEN 1 END) as above_200ema,
            COUNT(CASE WHEN e."Cls_Abv_EMA200" < 0 THEN 1 END) as below_200ema,
            COUNT(CASE WHEN e."Breakout_20" = 'Breakout_20_Up' THEN 1 END) as breakouts_up,
            COUNT(CASE WHEN e."Breakout_20" = 'Breakout_20_Down' THEN 1 END) as breakouts_down,
            COUNT(CASE WHEN e."Vol_Abv_Avg20" > 1.5 THEN 1 END) as high_volume,
            COUNT(CASE WHEN e."Vol_Abv_Avg20" > 1 AND e."Vol_Abv_Avg20" <= 1.5 THEN 1 END) as normal_volume,
            COUNT(CASE WHEN e."Vol_Abv_Avg20" <= 1 THEN 1 END) as low_volume,
            COUNT(CASE WHEN e."Range_Pct" > 3 THEN 1 END) as high_volatility,
            COUNT(CASE WHEN e."Range_Pct" > 1.5 AND e."Range_Pct" <= 3 THEN 1 END) as normal_volatility,
            COUNT(CASE WHEN e."Range_Pct" <= 1.5 THEN 1 END) as low_volatility,
            AVG(e."Vol_Abv_Avg20") as avg_volume_ratio,
            AVG(e."Range_Pct") as avg_volatility
        FROM "AGG_DATA" a
        JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
        """
        
        result = await db.execute(text(overview_query))
        row = result.fetchone()
        
        # Market sentiment calculation
        sentiment_score = 0
        if row[1]:  # avg_daily_change
            sentiment_score += row[1] * 10  # Weight daily change
        
        bullish_ratio = row[2] / row[0] if row[0] > 0 else 0  # gainers ratio
        sentiment_score += (bullish_ratio - 0.5) * 20  # Weight bullish ratio
        
        ema_bullish_ratio = row[5] / row[0] if row[0] > 0 else 0  # above 200 EMA ratio
        sentiment_score += (ema_bullish_ratio - 0.5) * 15  # Weight EMA trend
        
        if sentiment_score > 10:
            market_sentiment = "Very Bullish"
        elif sentiment_score > 5:
            market_sentiment = "Bullish"
        elif sentiment_score > -5:
            market_sentiment = "Neutral"
        elif sentiment_score > -10:
            market_sentiment = "Bearish"
        else:
            market_sentiment = "Very Bearish"
        
        overview = {
            "total_stocks": row[0],
            "market_sentiment": market_sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "avg_daily_change": round(row[1] or 0, 2),
            
            "market_breadth": {
                "gainers": row[2],
                "losers": row[3],
                "unchanged": row[4],
                "gainers_pct": round((row[2] / row[0]) * 100, 1) if row[0] > 0 else 0,
                "losers_pct": round((row[3] / row[0]) * 100, 1) if row[0] > 0 else 0
            },
            
            "technical_overview": {
                "above_200ema": row[5],
                "below_200ema": row[6],
                "above_200ema_pct": round((row[5] / row[0]) * 100, 1) if row[0] > 0 else 0,
                "breakouts_up": row[7],
                "breakouts_down": row[8]
            },
            
            "volume_analysis": {
                "high_volume_stocks": row[9],
                "normal_volume_stocks": row[10],
                "low_volume_stocks": row[11],
                "avg_volume_ratio": round(row[15] or 0, 2)
            },
            
            "volatility_overview": {
                "high_volatility": row[12],
                "normal_volatility": row[13],
                "low_volatility": row[14],
                "avg_volatility": round(row[16] or 0, 2)
            }
        }
        
        # Clean all data using our helper function
        return clean_financial_data(overview)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting market overview: {str(e)}")

@router.get("/top-movers")
async def get_comprehensive_top_movers(
    move_type: str = Query(..., description="gainers, losers, volume, breakouts_up, breakouts_down"),
    limit: int = Query(20, description="Number of results"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get comprehensive top movers with detailed analysis"""
    try:
        base_query = """
        SELECT 
            a."Symbol",
            a."close" as current_price,
            a."volume",
            a."Pct_Chg_D" as daily_change,
            a."Percent_Chg_W" as weekly_change,
            a."Percent_Chg_M" as monthly_change,
            e."Vol_Abv_Avg20" as volume_vs_avg,
            e."EMA20_Sig",
            e."EMA200_Sig",
            e."Breakout_20",
            e."Vol20_Sig",
            e."Range_Pct",
            a."high_52W",
            a."low_52W",
            CASE 
                WHEN a."volume" > 20000000 THEN 'High Liquidity'
                WHEN a."volume" > 5000000 THEN 'Medium Liquidity'
                ELSE 'Low Liquidity'
            END as liquidity_category
        FROM "AGG_DATA" a
        JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
        WHERE 1=1
        """
        
        if move_type == "gainers":
            query = base_query + 'ORDER BY a."Pct_Chg_D" DESC LIMIT :limit'
        elif move_type == "losers":
            query = base_query + 'ORDER BY a."Pct_Chg_D" ASC LIMIT :limit'
        elif move_type == "volume":
            query = base_query + 'ORDER BY e."Vol_Abv_Avg20" DESC LIMIT :limit'
        elif move_type == "breakouts_up":
            query = base_query + 'AND e."Breakout_20" = \'Breakout_20_Up\' ORDER BY a."Pct_Chg_D" DESC LIMIT :limit'
        elif move_type == "breakouts_down":
            query = base_query + 'AND e."Breakout_20" = \'Breakout_20_Down\' ORDER BY a."Pct_Chg_D" ASC LIMIT :limit'
        else:
            raise HTTPException(status_code=400, detail="Invalid move_type")
        
        result = await db.execute(text(query), {"limit": limit})
        rows = result.fetchall()
        
        movers = []
        for row in rows:
            # Generate signals for each stock
            signals = []
            if row[7]:  # EMA20_Sig
                if "GT" in row[7]:
                    signals.append("Above EMA 20")
                elif "LT" in row[7]:
                    signals.append("Below EMA 20")
                    
            if row[8]:  # EMA200_Sig
                if "GT" in row[8]:
                    signals.append("Above 200 EMA")
                elif "LT" in row[8]:
                    signals.append("Below 200 EMA")
            
            if row[10]:  # Vol20_Sig
                if "Super" in row[10]:
                    signals.append("Super Volume")
                elif "High" in row[10]:
                    signals.append("High Volume")
            
            if row[9]:  # Breakout_20
                if "Up" in row[9]:
                    signals.append("Breakout Up")
                elif "Down" in row[9]:
                    signals.append("Breakdown")
            
            # Calculate 52-week position
            from_52w_high = ((row[12] - row[1]) / row[12] * 100) if row[12] else 0
            from_52w_low = ((row[1] - row[13]) / row[13] * 100) if row[13] else 0
            
            mover = {
                "symbol": row[0],
                "current_price": row[1],
                "volume": row[2],
                "daily_change": row[3],
                "weekly_change": row[4],
                "monthly_change": row[5],
                "volume_vs_avg": row[6],
                "volatility": row[11],
                "market_cap_category": row[14],
                "from_52w_high": round(from_52w_high, 1),
                "from_52w_low": round(from_52w_low, 1),
                "signals": signals,
                "trend_strength": len([s for s in signals if "Above" in s or "Breakout" in s]) - len([s for s in signals if "Below" in s or "Breakdown" in s])
            }
            movers.append(mover)
        
        return {"movers": movers, "type": move_type, "count": len(movers)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting top movers: {str(e)}")

@router.post("/advanced-scan")
async def advanced_stock_scan_enhanced(
    criteria: ScannerCriteria,
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Enhanced stock scanning based on actual data analysis logic from EOD_analysis.py and AGG_DATA"""
    try:
        # Build dynamic query based on criteria
        conditions = []
        params = {}
        
        base_query = """
        SELECT 
            a."Symbol",
            a."close",
            a."volume",
            a."Pct_Chg_D",
            a."Percent_Chg_W",
            a."Percent_Chg_M",
            a."Percent_Chg_Y",
            a."high_52W",
            a."low_52W",
            e."EMA_20",
            e."EMA_60", 
            e."EMA_200",
            e."ATR",
            e."Vol_Abv_Avg20",
            e."Range_Pct",
            e."Narrow_Range",
            e."EMA20_Sig",
            e."EMA60_Sig",
            e."EMA200_Sig",
            e."Breakout_20",
            e."Vol20_Sig",
            e."Curr_Supp",
            e."Curr_Res",
            e."Break_Sup_Res",
            e."Reg_Cross_Sig",
            e."Reg_6",
            e."Reg_18",
            e."Cls_Abv_EMA20",
            e."Cls_Abv_EMA60",
            e."Cls_Abv_EMA200"
        FROM "AGG_DATA" a
        JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
        WHERE 1=1
        """
        
        # Apply filters based on enhanced criteria
        if criteria.min_price:
            conditions.append('a."close" >= :min_price')
            params['min_price'] = criteria.min_price
            
        if criteria.max_price:
            conditions.append('a."close" <= :max_price')
            params['max_price'] = criteria.max_price
            
        if criteria.min_volume:
            conditions.append('a."volume" >= :min_volume')
            params['min_volume'] = criteria.min_volume
            
        # Enhanced volume surge filter (based on Vol_Abv_Avg20)
        if criteria.min_volume_surge and criteria.min_volume_surge > 1.0:
            conditions.append('e."Vol_Abv_Avg20" >= :volume_surge')
            params['volume_surge'] = criteria.min_volume_surge
            
        if criteria.volume_surge_threshold:
            conditions.append('e."Vol_Abv_Avg20" >= :volume_surge_alt')
            params['volume_surge_alt'] = criteria.volume_surge_threshold
            
        # Support/Resistance signal filters (based on actual EOD_analysis.py logic)
        if criteria.support_resistance_signals:
            # Curr_Supp/Curr_Res are numeric levels; avoid string comparisons
            conditions.append('(e."Curr_Supp" IS NOT NULL OR e."Curr_Res" IS NOT NULL)')
            
        if criteria.support_signals:
            # Match textual support/resistance events against Break_Sup_Res signal column
            support_conditions = []
            for i, signal in enumerate(criteria.support_signals):
                param_name = f'support_signal_{i}'
                support_conditions.append(f'e."Break_Sup_Res" LIKE :{param_name}')
                params[param_name] = f'%{signal}%'
            if support_conditions:
                conditions.append('(' + ' OR '.join(support_conditions) + ')')
                
        if criteria.resistance_signals:
            # Match textual support/resistance events against Break_Sup_Res signal column
            resistance_conditions = []
            for i, signal in enumerate(criteria.resistance_signals):
                param_name = f'resistance_signal_{i}'
                resistance_conditions.append(f'e."Break_Sup_Res" LIKE :{param_name}')
                params[param_name] = f'%{signal}%'
            if resistance_conditions:
                conditions.append('(' + ' OR '.join(resistance_conditions) + ')')
                
        # Breakout pattern filters (based on Break_Sup_Res column)
        if criteria.breakout_patterns:
            conditions.append('e."Break_Sup_Res" IS NOT NULL AND e."Break_Sup_Res" != \'\'')
            
        # 20-day breakout filters (based on Breakout_20 column)
        if criteria.breakout_20_up:
            conditions.append('e."Breakout_20" = \'Breakout_20_Up\'')
            
        if criteria.breakout_20_down:
            conditions.append('e."Breakout_20" = \'Breakout_20_Down\'')
            
        # EMA alignment filters (based on actual EMA signal columns)
        if criteria.ema_alignment:
            conditions.append('e."Cls_Abv_EMA20" > 0 AND e."Cls_Abv_EMA60" > 0 AND e."Cls_Abv_EMA200" > 0')
            
        if criteria.ema_cross_signals:
            conditions.append('e."EMA20_Sig" LIKE \'%Cross%\' OR e."EMA60_Sig" LIKE \'%Cross%\' OR e."EMA200_Sig" LIKE \'%Cross%\'')
            
        # Regression trend filters (based on Reg_Cross_Sig column)
        if criteria.regression_trend:
            conditions.append('e."Reg_Cross_Sig" LIKE :regression_trend')
            params['regression_trend'] = f'%{criteria.regression_trend}%'
            
        # Narrow range breakout (based on Narrow_Range column)
        if criteria.narrow_range_breakout:
            conditions.append('e."Narrow_Range" = true AND e."Vol_Abv_Avg20" > 1.2')
            
        # Volatility filters (based on Range_Pct)
        if criteria.max_volatility:
            conditions.append('e."Range_Pct" <= :max_volatility')
            params['max_volatility'] = criteria.max_volatility
            
        # Volume confirmation (volume supports price movement)
        if criteria.volume_confirmation:
            # Group OR clause to preserve precedence with other ANDed conditions
            conditions.append('((a."Pct_Chg_D" > 0 AND e."Vol_Abv_Avg20" > 1.2) OR (a."Pct_Chg_D" < 0 AND e."Vol_Abv_Avg20" > 1.2))')
        
        # Performance filters
        if criteria.min_daily_change is not None:
            conditions.append('a."Pct_Chg_D" >= :min_daily')
            params['min_daily'] = criteria.min_daily_change
            
        if criteria.max_daily_change is not None:
            conditions.append('a."Pct_Chg_D" <= :max_daily')
            params['max_daily'] = criteria.max_daily_change
        
        # Technical filters
        if criteria.above_ema_20 is not None:
            if criteria.above_ema_20:
                conditions.append('e."Cls_Abv_EMA20" > 0')
            else:
                conditions.append('e."Cls_Abv_EMA20" < 0')
                
        if criteria.above_ema_200 is not None:
            if criteria.above_ema_200:
                conditions.append('e."Cls_Abv_EMA200" > 0')
            else:
                conditions.append('e."Cls_Abv_EMA200" < 0')
        
        # Enhanced breakout filters
        if criteria.breakout_signal:
            if criteria.breakout_signal == "up":
                conditions.append('e."Breakout_20" = \'Breakout_20_Up\'')
            elif criteria.breakout_signal == "down":
                conditions.append('e."Breakout_20" = \'Breakout_20_Down\'')
            elif criteria.breakout_signal == "any":
                conditions.append('e."Breakout_20" IS NOT NULL AND e."Breakout_20" != \'\'')
        
        # 52-week position filters
        if criteria.near_52w_high_pct:
            conditions.append('(a."close" / a."high_52W") >= :near_52w_high')
            params['near_52w_high'] = (100 - criteria.near_52w_high_pct) / 100
            
        if criteria.near_52w_low_pct:
            conditions.append('(a."close" / a."low_52W") <= :near_52w_low')
            params['near_52w_low'] = 1 + (criteria.near_52w_low_pct / 100)
        
        # ATR filters
        if criteria.min_atr:
            conditions.append('e."ATR" >= :min_atr')
            params['min_atr'] = criteria.min_atr
            
        # Narrow range filter
        if criteria.narrow_range is not None:
            conditions.append('e."Narrow_Range" = :narrow_range')
            params['narrow_range'] = criteria.narrow_range
        
        # Maximum range percentage filter
        if criteria.max_range_pct:
            conditions.append('e."Range_Pct" <= :max_range_pct')
            params['max_range_pct'] = criteria.max_range_pct
        
        # Add conditions to query
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        # Handle sorting - if signal_strength is requested, add it to SELECT and use it for ORDER BY
        if criteria.sort_by == 'signal_strength':
            # Add signal strength calculation to the query
            signal_strength_calc = """
            (
                CASE WHEN e."Vol_Abv_Avg20" > 3 THEN 3
                     WHEN e."Vol_Abv_Avg20" > 2 THEN 2  
                     WHEN e."Vol_Abv_Avg20" > 1.5 THEN 1 ELSE 0 END +
                CASE WHEN e."Reg_Cross_Sig" LIKE '%Cross_Up%' THEN 3
                     WHEN e."Reg_Cross_Sig" LIKE '%Reg6_Abv_Reg18%' THEN 2
                     WHEN e."Reg_Cross_Sig" LIKE '%Cross_Down%' THEN -2
                     WHEN e."Reg_Cross_Sig" LIKE '%Reg6_Blw_Reg18%' THEN -1 ELSE 0 END +
                CASE WHEN e."EMA20_Sig" LIKE '%Cross_Abv%' THEN 2
                     WHEN e."EMA20_Sig" LIKE '%GT%' THEN 1
                     WHEN e."EMA20_Sig" LIKE '%Cross_Blw%' THEN -2
                     WHEN e."EMA20_Sig" LIKE '%LT%' THEN -1 ELSE 0 END +
                CASE WHEN e."Break_Sup_Res" LIKE '%Res_Broken_Up%' THEN 2
                     WHEN e."Break_Sup_Res" LIKE '%Sup_Broken_Down%' THEN -2 ELSE 0 END
            ) as signal_strength"""

            # Replace the last column in SELECT with the signal strength calculation
            base_query = base_query.replace(
                'e."Cls_Abv_EMA200"',
                f'e."Cls_Abv_EMA200", {signal_strength_calc}',
                1
            )
            sort_column = "signal_strength"
        else:
            # Standard column sorting
            agg_columns = ['close', 'volume', 'Symbol', 'Pct_Chg_D', 'Percent_Chg_W', 'Percent_Chg_M', 'Percent_Chg_Y', 'high_52W', 'low_52W']
            eod_columns = ['ATR', 'Vol_Abv_Avg20', 'Range_Pct', 'EMA_20', 'EMA_60', 'EMA_200']
            
            if criteria.sort_by in agg_columns:
                sort_column = f'a."{criteria.sort_by}"'
            elif criteria.sort_by in eod_columns:
                sort_column = f'e."{criteria.sort_by}"'
            else:
                sort_column = 'a."close"'  # Default to close price
                
        sort_direction = "DESC" if criteria.sort_order.lower() == "desc" else "ASC"
        base_query += f" ORDER BY {sort_column} {sort_direction}"
        
        # Add pagination
        base_query += " LIMIT :limit OFFSET :offset"
        params['limit'] = criteria.limit
        params['offset'] = criteria.offset
        
        # Execute query
        result = await db.execute(text(base_query), params)
        rows = result.fetchall()
        
        # Process results with comprehensive analytical data
        stocks = []
        for row in rows:
            # Generate signals with enhanced logic
            signals = []
            signal_strength = 0
            
            # Technical signals
            if row[16] and "GT" in row[16]:  # EMA20_Sig
                signals.append("Above EMA 20")
                signal_strength += 1
            if row[17] and "GT" in row[17]:  # EMA60_Sig
                signals.append("Above EMA 60")
                signal_strength += 1
            if row[18] and "GT" in row[18]:  # EMA200_Sig
                signals.append("Above EMA 200")
                signal_strength += 2
                
            # Volume signals
            if row[13] and row[13] > 2:  # Vol_Abv_Avg20
                signals.append("Super High Volume")
                signal_strength += 2
            elif row[13] and row[13] > 1.5:
                signals.append("High Volume")
                signal_strength += 1
                
            # Breakout signals
            if row[19] and "Up" in str(row[19]):  # Breakout_20
                signals.append("Breakout Up")
                signal_strength += 2
            elif row[19] and "Down" in str(row[19]):
                signals.append("Breakdown")
                signal_strength -= 2
            
            # Support/Resistance signals
            if row[23] and "Support" in str(row[23]):  # Break_Sup_Res
                signals.append("Support Break")
                if "Broken_Up" in str(row[23]):
                    signal_strength += 1
                else:
                    signal_strength -= 1
            elif row[23] and "Resistance" in str(row[23]):
                signals.append("Resistance Break")
                signal_strength += 1
            
            # Regression signals
            if row[24] and "Reg6_Abv_Reg18" in str(row[24]):  # Reg_Cross_Sig
                signals.append("Bullish Regression Cross")
                signal_strength += 1
            elif row[24] and "Reg6_Blw_Reg18" in str(row[24]):
                signals.append("Bearish Regression Cross")
                signal_strength -= 1
            
            # Calculate position metrics with safety checks
            from_52w_high = 0
            from_52w_low = 0
            if row[7] and row[1] and row[7] > 0:  # high_52W and close
                from_52w_high = ((row[7] - row[1]) / row[7] * 100)
            if row[8] and row[1] and row[8] > 0:  # low_52W and close
                from_52w_low = ((row[1] - row[8]) / row[8] * 100)
            
            # Handle None values safely
            def safe_value(val, default=0):
                return val if val is not None else default
            
            stock = {
                "symbol": row[0],
                "current_price": safe_value(row[1]),
                "volume": safe_value(row[2]),
                "daily_change": safe_value(row[3]),
                "weekly_change": safe_value(row[4]),
                "monthly_change": safe_value(row[5]),
                "yearly_change": safe_value(row[6]),
                "high_52w": safe_value(row[7]),
                "low_52w": safe_value(row[8]),
                "ema_20": safe_value(row[9]),
                "ema_60": safe_value(row[10]),
                "ema_200": safe_value(row[11]),
                "atr": safe_value(row[12]),
                "volume_vs_avg": safe_value(row[13], 1.0),
                "volatility": safe_value(row[14]),
                "narrow_range": bool(row[15]) if row[15] is not None else False,
                "ema_20_signal": str(row[16]) if row[16] else "",
                "ema_60_signal": str(row[17]) if row[17] else "",
                "ema_200_signal": str(row[18]) if row[18] else "",
                "breakout_20": str(row[19]) if row[19] else "",
                "volume_signal": str(row[20]) if row[20] else "",
                "support": safe_value(row[21]),
                "resistance": safe_value(row[22]),
                "support_resistance_break": str(row[23]) if row[23] else "",
                "regression_cross": str(row[24]) if row[24] else "",
                "reg_6": safe_value(row[25]),
                "reg_18": safe_value(row[26]),
                "cls_abv_ema_20": safe_value(row[27]),
                "cls_abv_ema_60": safe_value(row[28]),
                "cls_abv_ema_200": safe_value(row[29]),
                "from_52w_high": round(from_52w_high, 1) if from_52w_high else 0,
                "from_52w_low": round(from_52w_low, 1) if from_52w_low else 0,
                "signals": signals,
                "signal_strength": signal_strength,
                "recommendation": "Strong Buy" if signal_strength >= 4 else "Buy" if signal_strength >= 2 else "Sell" if signal_strength <= -2 else "Strong Sell" if signal_strength <= -4 else "Hold"
            }
            stocks.append(stock)
        
        # Calculate additional analytics
        total_signal_strength = sum(stock['signal_strength'] for stock in stocks)
        avg_signal_strength = total_signal_strength / len(stocks) if stocks else 0
        strong_buy_count = len([s for s in stocks if s['recommendation'] == 'Strong Buy'])
        buy_count = len([s for s in stocks if s['recommendation'] == 'Buy'])
        
        # Count specific signal types found
        signal_summary = {
            'support_resistance_signals': len([s for s in stocks if s['support_resistance_break']]),
            'breakout_signals': len([s for s in stocks if s['breakout_20']]),
            'volume_surge_stocks': len([s for s in stocks if s['volume_vs_avg'] > 1.5]),
            'ema_aligned_stocks': len([s for s in stocks if s['cls_abv_ema_20'] > 0 and s['cls_abv_ema_200'] > 0])
        }
        
        return {
            "stocks": clean_financial_data(stocks),
            "total_found": len(stocks),
            "criteria_applied": len([k for k, v in criteria.dict().items() if v is not None and v != [] and k not in ['sort_by', 'sort_order', 'limit', 'offset']]),
            "analytics": {
                "avg_signal_strength": round(avg_signal_strength, 2),
                "strong_buy_count": strong_buy_count,
                "buy_count": buy_count,
                "signal_summary": signal_summary
            },
            "scan_metadata": {
                "sorted_by": criteria.sort_by,
                "sort_order": criteria.sort_order,
                "filters_applied": {
                    "price_range": bool(criteria.min_price or criteria.max_price),
                    "volume_filters": bool(criteria.min_volume_surge or criteria.volume_surge_threshold),
                    "technical_filters": bool(criteria.ema_alignment or criteria.ema_cross_signals),
                    "breakout_filters": bool(criteria.breakout_patterns or criteria.breakout_20_up or criteria.breakout_20_down),
                    "support_resistance": bool(criteria.support_resistance_signals or criteria.support_signals or criteria.resistance_signals)
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing advanced scan: {str(e)}")

@router.get("/market-stats")
async def get_market_stats(
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get market overview statistics for scanner dashboard"""
    try:
        # Get total active stocks
        total_stocks_result = await db.execute(
            text("SELECT COUNT(*) as total FROM \"EOD_Summary\"")
        )
        total_stocks = total_stocks_result.scalar()

        # Get volume surge count (stocks with >1.5x average volume)
        volume_surge_result = await db.execute(
            text("""
                SELECT COUNT(*) as count 
                FROM \"EOD_Summary\" e 
                WHERE e."Vol_Abv_Avg20" > 1.5
            """)
        )
        volume_surge_count = volume_surge_result.scalar()

        # Get breakout count (stocks with positive breakout signals)
        breakout_result = await db.execute(
            text("""
                SELECT COUNT(*) as count 
                FROM \"EOD_Summary\" e 
                WHERE e."Break_Sup_Res" LIKE '%Up%' OR e."Breakout_20" LIKE 'Breakout%'
            """)
        )
        breakout_count = breakout_result.scalar()

        # Get active signals count (EMA alignment + volume)
        signals_result = await db.execute(
            text("""
                SELECT COUNT(*) as count 
                FROM \"EOD_Summary\" e 
                WHERE e."close" > e."EMA_20" 
                AND e."EMA_20" > e."EMA_60" 
                AND e."Vol_Abv_Avg20" > 1.2
            """)
        )
        active_signals = signals_result.scalar()

        return {
            "total_stocks": total_stocks or 0,
            "volume_surge_count": volume_surge_count or 0,
            "breakout_count": breakout_count or 0,
            "active_signals": active_signals or 0,
            "opportunities": min(active_signals or 0, breakout_count or 0),
            "market_status": "Open",
            "last_update": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching market stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching market stats: {str(e)}")

@router.get("/presets")
async def get_enhanced_scan_presets(
    current_user: User = Depends(get_current_user)
):
    """Get enhanced scanning presets with detailed configurations"""
    presets = {
        "bullish_momentum": {
            "name": "Bullish Momentum",
            "description": "Strong momentum stocks above all EMAs with volume surge",
            "category": "Growth",
            "criteria": {
                "above_ema_20": True,
                "above_ema_60": True,
                "above_ema_200": True,
                "min_daily_change": 1.0,
                "volume_surge_threshold": 1.5,
                "breakout_signal": "up",
                "sort_by": "Pct_Chg_D",
                "sort_order": "desc"
            }
        },
        "breakout_candidates": {
            "name": "Breakout Candidates", 
            "description": "Stocks breaking out of consolidation with volume",
            "category": "Breakout",
            "criteria": {
                "narrow_range": True,
                "volume_surge_threshold": 1.3,
                "above_ema_200": True,
                "min_daily_change": 0.5,
                "sort_by": "Vol_Abv_Avg20",
                "sort_order": "desc"
            }
        },
        "value_opportunities": {
            "name": "Value Opportunities",
            "description": "Quality stocks near 52-week lows with reversal signs",
            "category": "Value",
            "criteria": {
                "near_52w_low_pct": 15,
                "above_ema_200": True,
                "min_daily_change": 0,
                "volume_surge_threshold": 1.0,
                "sort_by": "from_52w_low",
                "sort_order": "asc"
            }
        },
        "high_volume_movers": {
            "name": "High Volume Movers",
            "description": "Stocks with unusual volume activity",
            "category": "Volume",
            "criteria": {
                "volume_surge_threshold": 2.0,
                "min_daily_change": 1.0,
                "sort_by": "Vol_Abv_Avg20", 
                "sort_order": "desc"
            }
        },
        "swing_trade_setup": {
            "name": "Swing Trade Setup",
            "description": "Ideal swing trading candidates with good risk-reward",
            "category": "Swing",
            "criteria": {
                "above_ema_20": True,
                "min_atr": 1.0,
                "volume_surge_threshold": 1.2,
                "min_daily_change": 0.5,
                "sort_by": "signal_strength",
                "sort_order": "desc"
            }
        },
        "oversold_bounce": {
            "name": "Oversold Bounce",
            "description": "Oversold stocks showing reversal signs",
            "category": "Reversal",
            "criteria": {
                "max_daily_change": -2.0,
                "above_ema_200": True,
                "volume_surge_threshold": 1.5,
                "sort_by": "Pct_Chg_D",
                "sort_order": "asc"
            }
        }
    }
    
    return {"presets": presets, "total": len(presets)}

@router.get("/analytics/sector-wise")
async def get_sector_wise_analytics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get sector-wise performance analytics (based on available data)"""
    try:
        # Since we don't have sector data, we'll categorize by market cap and performance
        query = """
        SELECT 
            CASE 
                WHEN a."volume" > 20000000 THEN 'High Volume'
                WHEN a."volume" > 5000000 THEN 'Medium Volume'
                ELSE 'Low Volume'
            END as volume_category,
            COUNT(*) as stock_count,
            AVG(a."Pct_Chg_D") as avg_daily_change,
            AVG(a."Percent_Chg_W") as avg_weekly_change,
            AVG(a."Percent_Chg_M") as avg_monthly_change,
            AVG(e."Vol_Abv_Avg20") as avg_volume_ratio,
            COUNT(CASE WHEN a."Pct_Chg_D" > 0 THEN 1 END) as gainers,
            COUNT(CASE WHEN a."Pct_Chg_D" < 0 THEN 1 END) as losers,
            COUNT(CASE WHEN e."Cls_Abv_EMA200" > 0 THEN 1 END) as above_200ema
        FROM "AGG_DATA" a
        JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
        GROUP BY volume_category
        ORDER BY volume_category
        """
        
        result = await db.execute(text(query))
        rows = result.fetchall()
        
        categories = []
        for row in rows:
            category = {
                "name": row[0],
                "stock_count": row[1],
                "avg_daily_change": round(row[2] or 0, 2),
                "avg_weekly_change": round(row[3] or 0, 2),
                "avg_monthly_change": round(row[4] or 0, 2),
                "avg_volume_ratio": round(row[5] or 0, 2),
                "gainers": row[6],
                "losers": row[7],
                "above_200ema": row[8],
                "bullish_pct": round((row[8] / row[1]) * 100, 1) if row[1] > 0 else 0
            }
            categories.append(category)
        
        return {"categories": categories, "total_categories": len(categories)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sector analytics: {str(e)}")

@router.get("/signals/dashboard")
async def get_signals_dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get comprehensive signals dashboard with categorized stocks"""
    try:
        # Get stocks by different signal categories
        signals_query = """
        WITH signal_analysis AS (
            SELECT 
                a."Symbol",
                a."close",
                a."Pct_Chg_D",
                a."volume",
                e."Vol_Abv_Avg20",
                e."EMA20_Sig",
                e."EMA200_Sig", 
                e."Breakout_20",
                e."Break_Sup_Res",
                e."Reg_Cross_Sig",
                e."Narrow_Range",
                e."Range_Pct",
                e."ATR",
                CASE 
                    WHEN e."EMA20_Sig" LIKE '%Cross_Abv%' OR e."EMA20_Sig" LIKE '%GT%' THEN 'bullish_ema'
                    WHEN e."EMA20_Sig" LIKE '%Cross_Blw%' OR e."EMA20_Sig" LIKE '%LT%' THEN 'bearish_ema'
                    ELSE 'neutral_ema'
                END as ema_signal_type,
                CASE 
                    WHEN e."Vol_Abv_Avg20" > 2 THEN 'super_volume'
                    WHEN e."Vol_Abv_Avg20" > 1.5 THEN 'high_volume'
                    WHEN e."Vol_Abv_Avg20" > 1 THEN 'normal_volume'
                    ELSE 'low_volume'
                END as volume_signal_type,
                CASE 
                    WHEN e."Breakout_20" LIKE '%Up%' THEN 'breakout_up'
                    WHEN e."Breakout_20" LIKE '%Down%' THEN 'breakdown'
                    ELSE 'no_breakout'
                END as breakout_signal_type,
                CASE 
                    WHEN e."Break_Sup_Res" LIKE '%Res_Broken_Up%' THEN 'resistance_break'
                    WHEN e."Break_Sup_Res" LIKE '%Sup_Broken_Down%' THEN 'support_break'
                    ELSE 'no_sr_break'
                END as support_resistance_type
            FROM "AGG_DATA" a
            JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
            WHERE a."close" > 0 AND e."Vol_Abv_Avg20" > 0
        )
        SELECT 
            ema_signal_type,
            volume_signal_type,
            breakout_signal_type,
            support_resistance_type,
            COUNT(*) as count,
            AVG("Pct_Chg_D") as avg_change,
            AVG("Vol_Abv_Avg20") as avg_volume_ratio
        FROM signal_analysis
        GROUP BY ema_signal_type, volume_signal_type, breakout_signal_type, support_resistance_type
        ORDER BY count DESC
        """
        
        result = await db.execute(text(signals_query))
        signal_combinations = result.fetchall()
        
        # Get top stocks in each major signal category based on proper EOD_analysis logic
        categories = {
            "bullish_momentum": {
                "name": "Bullish Momentum",
                "description": "EMA alignment + Volume + Regression trend bullish",
                "query": """
                SELECT a."Symbol", a."close", a."Pct_Chg_D", e."Vol_Abv_Avg20", 
                       e."EMA20_Sig", e."EMA200_Sig", e."Reg_Cross_Sig"
                FROM "AGG_DATA" a JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
                WHERE (e."EMA20_Sig" LIKE '%GT%' OR e."EMA20_Sig" LIKE '%Cross_Abv%')
                AND (e."EMA200_Sig" LIKE '%GT%' OR e."EMA200_Sig" LIKE '%Cross_Abv%')
                AND (e."Reg_Cross_Sig" = 'Reg6_Abv_Reg18' OR e."Reg_Cross_Sig" = 'Cross_Up')
                AND e."Vol_Abv_Avg20" > 1.2
                ORDER BY a."Pct_Chg_D" DESC LIMIT 15
                """
            },
            "volume_breakouts": {
                "name": "Volume Breakouts", 
                "description": "20-day breakouts with volume confirmation",
                "query": """
                SELECT a."Symbol", a."close", a."Pct_Chg_D", e."Vol_Abv_Avg20", e."Breakout_20"
                FROM "AGG_DATA" a JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
                WHERE e."Breakout_20" = 'Breakout_20_Up' AND e."Vol_Abv_Avg20" > 1.5
                ORDER BY e."Vol_Abv_Avg20" DESC LIMIT 15
                """
            },
            "support_resistance_signals": {
                "name": "Support/Resistance Signals",
                "description": "Support bounce or resistance break signals", 
                "query": """
                SELECT a."Symbol", a."close", a."Pct_Chg_D", e."Vol_Abv_Avg20", 
                       e."Support", e."Resistance", e."Break_Sup_Res"
                FROM "AGG_DATA" a JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
                WHERE (e."Support" IN ('Price_Abv_Supp', 'Price_Crs_Abv_Supp', 'Price_Abv_Cur_Res')
                    OR e."Resistance" IN ('Price_Blw_Res', 'Price_Crs_Blw_Res', 'Price_Blw_Cur_Sup')
                    OR e."Break_Sup_Res" LIKE '%Res_Broken_Up%'
                    OR e."Break_Sup_Res" LIKE '%Sup_Broken_Down%')
                AND e."Vol_Abv_Avg20" > 1.0
                ORDER BY a."Pct_Chg_D" DESC LIMIT 15
                """
            },
            "narrow_range_breakouts": {
                "name": "Narrow Range Breakouts",
                "description": "Breaking out of consolidation with volume",
                "query": """
                SELECT a."Symbol", a."close", a."Pct_Chg_D", e."Vol_Abv_Avg20", 
                       e."Narrow_Range_Breakout", e."Range_Pct"
                FROM "AGG_DATA" a JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
                WHERE e."Narrow_Range_Breakout" = true AND e."Vol_Abv_Avg20" > 1.2
                ORDER BY e."Vol_Abv_Avg20" DESC LIMIT 15
                """
            },
            "regression_signals": {
                "name": "Regression Cross Signals",
                "description": "Reg6 vs Reg18 momentum signals",
                "query": """
                SELECT a."Symbol", a."close", a."Pct_Chg_D", e."Vol_Abv_Avg20", e."Reg_Cross_Sig"
                FROM "AGG_DATA" a JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
                WHERE e."Reg_Cross_Sig" IN ('Cross_Up', 'Cross_Down') AND e."Vol_Abv_Avg20" > 1.1
                ORDER BY a."Pct_Chg_D" DESC LIMIT 15
                """
            }
        }
        
        dashboard_data = {
            "signal_combinations": [],
            "categories": {}
        }
        
        # Process signal combinations
        for combo in signal_combinations:
            dashboard_data["signal_combinations"].append({
                "ema_signal": combo[0],
                "volume_signal": combo[1], 
                "breakout_signal": combo[2],
                "support_resistance": combo[3],
                "count": combo[4],
                "avg_change": round(combo[5] or 0, 2),
                "avg_volume_ratio": round(combo[6] or 0, 2)
            })
        
        # Get stocks for each category
        for category_key, category_info in categories.items():
            cat_result = await db.execute(text(category_info["query"]))
            cat_rows = cat_result.fetchall()
            
            stocks = []
            for row in cat_rows:
                stocks.append({
                    "symbol": row[0],
                    "price": row[1],
                    "change": row[2],
                    "volume_ratio": row[3],
                    "signal": str(row[4]) if len(row) > 4 else ""
                })
            
            dashboard_data["categories"][category_key] = {
                "name": category_info["name"],
                "description": category_info["description"],
                "stocks": stocks,
                "count": len(stocks)
            }
        
        return clean_financial_data(dashboard_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting signals dashboard: {str(e)}")

@router.get("/analytics/comprehensive")
async def get_comprehensive_analytics(
    timeframe: str = Query("daily", description="daily, weekly, monthly"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_nsedata_db)
):
    """Get comprehensive market analytics with multiple timeframes"""
    try:
        # Determine the change column based on timeframe
        change_col = {
            "daily": "Pct_Chg_D",
            "weekly": "Percent_Chg_W", 
            "monthly": "Percent_Chg_M"
        }.get(timeframe, "Pct_Chg_D")
        
        analytics_query = f"""
        WITH market_analysis AS (
            SELECT 
                a."Symbol",
                a."close",
                a."{change_col}" as period_change,
                a."volume",
                e."Vol_Abv_Avg20",
                e."EMA_20",
                e."EMA_200", 
                e."ATR",
                e."Range_Pct",
                e."Breakout_20",
                e."Break_Sup_Res",
                CASE WHEN a."close" > e."EMA_200" THEN 1 ELSE 0 END as above_ema200,
                CASE WHEN e."Vol_Abv_Avg20" > 1.5 THEN 1 ELSE 0 END as high_volume,
                CASE WHEN e."Range_Pct" > 3 THEN 1 ELSE 0 END as high_volatility
            FROM "AGG_DATA" a
            JOIN "EOD_Summary" e ON a."Symbol" = e."Symbol"
            WHERE a."close" > 0
        )
        SELECT 
            -- Market breadth
            COUNT(*) as total_stocks,
            COUNT(CASE WHEN period_change > 0 THEN 1 END) as gainers,
            COUNT(CASE WHEN period_change < 0 THEN 1 END) as losers,
            COUNT(CASE WHEN period_change = 0 THEN 1 END) as unchanged,
            
            -- Technical analysis
            SUM(above_ema200) as above_ema200_count,
            SUM(high_volume) as high_volume_count,
            SUM(high_volatility) as high_volatility_count,
            
            -- Performance metrics
            AVG(period_change) as avg_change,
            STDDEV(period_change) as volatility,
            MAX(period_change) as max_gain,
            MIN(period_change) as max_loss,
            
            -- Volume metrics
            AVG("Vol_Abv_Avg20") as avg_volume_ratio,
            AVG("ATR") as avg_atr,
            AVG("Range_Pct") as avg_range_pct,
            
            -- Signal counts
            COUNT(CASE WHEN "Breakout_20" LIKE '%Up%' THEN 1 END) as breakout_up_count,
            COUNT(CASE WHEN "Breakout_20" LIKE '%Down%' THEN 1 END) as breakdown_count,
            COUNT(CASE WHEN "Break_Sup_Res" LIKE '%Res_Broken%' THEN 1 END) as resistance_breaks,
            COUNT(CASE WHEN "Break_Sup_Res" LIKE '%Sup_Broken%' THEN 1 END) as support_breaks
            
        FROM market_analysis
        """
        
        result = await db.execute(text(analytics_query))
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="No market data found")
        
        # Calculate additional metrics
        total_stocks = row[0] or 0
        gainers = row[1] or 0
        losers = row[2] or 0
        
        analytics = {
            "timeframe": timeframe,
            "market_breadth": {
                "total_stocks": total_stocks,
                "gainers": gainers,
                "losers": losers,
                "unchanged": row[3] or 0,
                "gainers_pct": round((gainers / total_stocks) * 100, 1) if total_stocks > 0 else 0,
                "losers_pct": round((losers / total_stocks) * 100, 1) if total_stocks > 0 else 0,
                "advance_decline_ratio": round(gainers / losers, 2) if losers > 0 else 0
            },
            "technical_summary": {
                "above_ema200": row[4] or 0,
                "above_ema200_pct": round(((row[4] or 0) / total_stocks) * 100, 1) if total_stocks > 0 else 0,
                "high_volume_stocks": row[5] or 0,
                "high_volatility_stocks": row[6] or 0
            },
            "performance_metrics": {
                "avg_change": round(row[7] or 0, 2),
                "market_volatility": round(row[8] or 0, 2),
                "max_gain": round(row[9] or 0, 2),
                "max_loss": round(row[10] or 0, 2),
                "avg_volume_ratio": round(row[11] or 0, 2),
                "avg_atr": round(row[12] or 0, 2),
                "avg_range_pct": round(row[13] or 0, 2)
            },
            "signal_summary": {
                "breakouts_up": row[14] or 0,
                "breakdowns": row[15] or 0,
                "resistance_breaks": row[16] or 0,
                "support_breaks": row[17] or 0,
                "net_breakouts": (row[14] or 0) - (row[15] or 0)
            }
        }
        
        # Market sentiment calculation
        sentiment_score = 0
        if analytics["market_breadth"]["gainers_pct"] > 60:
            sentiment_score += 2
        elif analytics["market_breadth"]["gainers_pct"] > 50:
            sentiment_score += 1
        
        if analytics["technical_summary"]["above_ema200_pct"] > 60:
            sentiment_score += 2
        elif analytics["technical_summary"]["above_ema200_pct"] > 50:
            sentiment_score += 1
            
        if analytics["signal_summary"]["net_breakouts"] > 0:
            sentiment_score += 1
        elif analytics["signal_summary"]["net_breakouts"] < 0:
            sentiment_score -= 1
        
        sentiment_map = {
            5: "Very Bullish", 4: "Bullish", 3: "Moderately Bullish",
            2: "Neutral", 1: "Cautious", 0: "Bearish", -1: "Very Bearish"
        }
        
        analytics["market_sentiment"] = {
            "sentiment": sentiment_map.get(sentiment_score, "Neutral"),
            "score": sentiment_score,
            "confidence": "High" if abs(sentiment_score) >= 3 else "Medium" if abs(sentiment_score) >= 2 else "Low"
        }
        
        return clean_financial_data(analytics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting comprehensive analytics: {str(e)}")