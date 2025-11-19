"""
Main API Service for TradeWiser Pricing Tool
Production-ready FastAPI application with JWT authentication and Supabase integration
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import our modules
from auth import get_current_user, get_current_user_optional, require_role
from database_supabase import (
    get_commodity_data, get_regions, get_all_commodities,
    get_price_history, get_average_market_price,
    store_quality_price_vector, find_similar_quality_vectors,
    log_api_request, initialize_database
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', '/var/log/tradewiser-pricing/api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting TradeWiser Pricing API...")
    initialize_database()
    yield
    # Shutdown
    logger.info("Shutting down TradeWiser Pricing API...")


# Initialize FastAPI app
app = FastAPI(
    title="TradeWiser Pricing API",
    description="Agricultural commodity pricing tool with quality-based pricing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "https://tradewiser.in,https://pricing.tradewiser.in").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Get user from token if available
    user_id = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            from auth import verify_token
            token = auth_header.replace("Bearer ", "")
            payload = verify_token(token)
            user_id = payload.get("user_id") or payload.get("sub")
        except:
            pass
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time_ms = int((time.time() - start_time) * 1000)
    
    # Log request
    log_api_request(
        user_id=user_id,
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        response_time_ms=response_time_ms,
        ip_address=request.client.host if request.client else None
    )
    
    return response


# ==================== PYDANTIC MODELS ====================

class ErrorResponse(BaseModel):
    detail: str


class CommodityInfo(BaseModel):
    name: str
    description: Optional[str] = None
    trading_unit: Optional[str] = None
    quality_parameters: Dict[str, Dict[str, Any]]
    regions: List[str]


class PriceRequest(BaseModel):
    commodity: str
    quality_params: Dict[str, float]
    region: str


class PriceResponse(BaseModel):
    commodity: str
    region: str
    base_price: float
    final_price: float
    quality_adjustment: float
    confidence: float
    currency: str = "INR"
    unit: str
    timestamp: datetime


class PriceHistoryEntry(BaseModel):
    date: str
    price: float
    volume: Optional[float] = None
    market_name: Optional[str] = None


class PriceHistory(BaseModel):
    commodity: str
    region: str
    currency: str = "INR"
    unit: str
    data: List[PriceHistoryEntry]


# ==================== API ROUTES ====================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "api": "TradeWiser Pricing API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "tradewiser-pricing-api"
    }


# ==================== COMMODITIES API ====================

@app.get("/api/commodities", tags=["Commodities"], response_model=List[str])
async def list_commodities(user: Dict = Depends(get_current_user_optional)):
    """Get list of all available commodities."""
    try:
        commodities = get_all_commodities()
        return commodities
    except Exception as e:
        logger.error(f"Error retrieving commodities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/commodities/{commodity}", tags=["Commodities"], response_model=CommodityInfo)
async def get_commodity(
    commodity: str,
    user: Dict = Depends(get_current_user)
):
    """Get detailed information about a specific commodity."""
    try:
        commodity_data = get_commodity_data(commodity)
        if not commodity_data:
            raise HTTPException(status_code=404, detail=f"Commodity {commodity} not found")
        
        return {
            "name": commodity,
            "description": commodity_data.get("description", ""),
            "trading_unit": commodity_data.get("trading_unit", "kg"),
            "quality_parameters": commodity_data.get("quality_parameters", {}),
            "regions": commodity_data.get("regions", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving commodity data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/commodities/{commodity}/regions", tags=["Commodities"], response_model=List[str])
async def get_commodity_regions(
    commodity: str,
    user: Dict = Depends(get_current_user)
):
    """Get all regions available for a specific commodity."""
    try:
        regions = get_regions(commodity)
        if not regions:
            raise HTTPException(status_code=404, detail=f"No regions found for commodity {commodity}")
        
        return regions
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving regions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PRICING API ====================

@app.post("/api/price/calculate", tags=["Pricing"], response_model=PriceResponse)
async def calculate_commodity_price(
    request: PriceRequest,
    user: Dict = Depends(get_current_user)
):
    """Calculate price for a commodity based on quality parameters and region."""
    try:
        # Get commodity data
        commodity_data = get_commodity_data(request.commodity)
        if not commodity_data:
            raise HTTPException(status_code=404, detail=f"Commodity {request.commodity} not found")
        
        # Get base price from market data
        base_price = get_average_market_price(request.commodity, request.region, days_back=30)
        
        if base_price == 0:
            # Use price range midpoint as fallback
            price_range = commodity_data.get('price_range', {})
            base_price = (price_range.get('min', 1000) + price_range.get('max', 10000)) / 2
        
        # Calculate quality adjustment
        quality_impact = commodity_data.get('quality_impact', {})
        quality_adjustment = 0
        
        for param_name, param_value in request.quality_params.items():
            if param_name in quality_impact:
                impact = quality_impact[param_name]
                standard_value = commodity_data.get('quality_parameters', {}).get(param_name, {}).get('standard_value', 50)
                
                # Calculate deviation from standard
                deviation = param_value - standard_value
                
                # Apply impact factor
                if deviation > 0:
                    quality_adjustment += deviation * impact.get('premium_factor', 0)
                else:
                    quality_adjustment += deviation * impact.get('discount_factor', 0)
        
        # Calculate final price
        final_price = base_price + quality_adjustment
        
        # Store this quality-price vector for future ML
        store_quality_price_vector(
            commodity=request.commodity,
            region=request.region,
            quality_params=request.quality_params,
            price=final_price,
            source='calculated',
            confidence=0.8
        )
        
        return {
            "commodity": request.commodity,
            "region": request.region,
            "base_price": round(base_price, 2),
            "final_price": round(final_price, 2),
            "quality_adjustment": round(quality_adjustment, 2),
            "confidence": 0.8,
            "currency": "INR",
            "unit": commodity_data.get("trading_unit", "kg"),
            "timestamp": datetime.now()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating price: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/price/history/{commodity}/{region}", tags=["Pricing"], response_model=PriceHistory)
async def get_commodity_price_history(
    commodity: str,
    region: str,
    days: int = Query(30, description="Number of days of history to retrieve"),
    user: Dict = Depends(get_current_user)
):
    """Get historical price data for a specific commodity and region."""
    try:
        # Get commodity data
        commodity_data = get_commodity_data(commodity)
        if not commodity_data:
            raise HTTPException(status_code=404, detail=f"Commodity {commodity} not found")
        
        # Get price history
        history = get_price_history(commodity, region, days)
        
        if not history:
            raise HTTPException(status_code=404, detail="No price history available")
        
        # Format response
        history_entries = []
        for entry in history:
            history_entries.append({
                "date": entry.get("date"),
                "price": float(entry.get("price")),
                "volume": float(entry.get("volume")) if entry.get("volume") else None,
                "market_name": entry.get("market_name")
            })
        
        return {
            "commodity": commodity,
            "region": region,
            "currency": "INR",
            "unit": commodity_data.get("trading_unit", "kg"),
            "data": history_entries
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving price history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ADMIN API ====================

@app.get("/api/admin/stats", tags=["Admin"])
async def get_admin_stats(user: Dict = Depends(require_role("admin"))):
    """Get admin statistics (admin only)."""
    return {
        "message": "Admin statistics endpoint",
        "user": user
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("ENVIRONMENT") != "production"
    )
