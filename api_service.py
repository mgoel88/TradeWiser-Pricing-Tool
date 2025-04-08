"""
API Service module for WIZX Agricultural Commodity Pricing Platform.

This module provides RESTful API endpoints for integrating WIZX pricing engine 
with external applications. The API follows OpenAPI specification.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# FastAPI for modern API development
from fastapi import FastAPI, HTTPException, Query, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Import our application modules
from database_sql import (
    get_commodity_data, get_regions, get_all_commodities,
    get_price_history as get_sql_price_history,
    calculate_wizx_index, get_wizx_indices
)
from pricing_engine import calculate_price, get_price_history
from quality_analyzer import analyze_quality_from_image, analyze_report
from models import predict_price_trend
from data_crawler import fetch_latest_agmarknet_data, fetch_commodity_list
from ai_vision import analyze_commodity_image, save_uploaded_image
from utils import format_date, validate_date_range

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Initialize FastAPI app
app = FastAPI(
    title="WIZX Agricultural Commodity Pricing API",
    description="API for agricultural commodity pricing data and analytics",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models for request/response validation
class ErrorResponse(BaseModel):
    detail: str

class CommodityInfo(BaseModel):
    name: str
    description: Optional[str] = None
    trading_units: Optional[str] = None
    quality_parameters: Dict[str, Dict[str, Any]]

class PriceRequest(BaseModel):
    commodity: str
    quality_params: Dict[str, float]
    region: str

class PriceResponse(BaseModel):
    commodity: str
    region: str
    base_price: float
    final_price: float
    quality_delta: float
    location_delta: float
    market_delta: float
    currency: str = "INR"
    timestamp: datetime
    unit: str
    confidence: Optional[float] = None

class QualityAnalysisResponse(BaseModel):
    commodity: str
    quality_params: Dict[str, float]
    quality_score: float
    quality_grade: str
    confidence: Optional[float] = None
    analysis_summary: Optional[str] = None
    timestamp: datetime

class PriceHistoryEntry(BaseModel):
    date: datetime
    price: float
    volume: Optional[float] = None
    
class PriceHistory(BaseModel):
    commodity: str
    region: str
    currency: str = "INR"
    unit: str
    data: List[PriceHistoryEntry]

class IndexValue(BaseModel):
    date: datetime
    value: float
    
class CommodityIndex(BaseModel):
    commodity: str
    current_value: float
    change_percentage: float
    history: List[IndexValue]

class PriceForecast(BaseModel):
    commodity: str
    region: str
    forecast_data: List[Dict[str, Any]]
    model_confidence: float
    created_at: datetime

# ----- API Authentication -----
async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Verify the API key."""
    # In production, use a secure method to store and validate API keys
    # For now, use a simple check (this is just for demo purposes)
    valid_api_keys = ["test_key", "demo_key"]  # In production, store securely
    
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
    return api_key

# ----- API Routes -----

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "api": "WIZX Agricultural Commodity Pricing API",
        "version": "1.0.0",
        "documentation": "/docs",
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint for API status monitoring."""
    return {"status": "healthy", "timestamp": datetime.now()}

# Commodities API
@app.get("/commodities", tags=["Commodities"], response_model=List[str])
async def list_commodities(api_key: str = Depends(verify_api_key)):
    """Get a list of all available commodities."""
    try:
        commodities = get_all_commodities()
        return commodities
    except Exception as e:
        logger.error(f"Error retrieving commodities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/commodities/{commodity}", tags=["Commodities"], response_model=CommodityInfo)
async def get_commodity(
    commodity: str,
    api_key: str = Depends(verify_api_key)
):
    """Get detailed information about a specific commodity."""
    try:
        commodity_data = get_commodity_data(commodity)
        if not commodity_data:
            raise HTTPException(status_code=404, detail=f"Commodity {commodity} not found")
        
        return {
            "name": commodity,
            "description": commodity_data.get("description", ""),
            "trading_units": commodity_data.get("trading_unit", "kg"),
            "quality_parameters": commodity_data.get("quality_parameters", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving commodity data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/commodities/{commodity}/regions", tags=["Commodities"], response_model=List[str])
async def get_commodity_regions(
    commodity: str,
    api_key: str = Depends(verify_api_key)
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

# Pricing API
@app.post("/price", tags=["Pricing"], response_model=PriceResponse)
async def calculate_commodity_price(
    request: PriceRequest,
    api_key: str = Depends(verify_api_key)
):
    """Calculate price for a commodity based on quality parameters and region."""
    try:
        # Get commodity data
        commodity_data = get_commodity_data(request.commodity)
        if not commodity_data:
            raise HTTPException(status_code=404, detail=f"Commodity {request.commodity} not found")
        
        # Calculate price
        price_result = calculate_price(
            request.commodity,
            request.quality_params,
            request.region
        )
        
        if not price_result:
            raise HTTPException(status_code=404, detail="Could not calculate price")
        
        # Format response
        response = {
            "commodity": request.commodity,
            "region": request.region,
            "base_price": price_result.get("base_price", 0),
            "final_price": price_result.get("final_price", 0),
            "quality_delta": price_result.get("quality_delta", 0),
            "location_delta": price_result.get("location_delta", 0),
            "market_delta": price_result.get("market_delta", 0),
            "currency": "INR",
            "timestamp": datetime.now(),
            "unit": commodity_data.get("trading_unit", "kg"),
            "confidence": price_result.get("confidence", 0.8)
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating price: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/price-history/{commodity}/{region}",
    tags=["Pricing"],
    response_model=PriceHistory
)
async def get_commodity_price_history(
    commodity: str,
    region: str,
    days: int = Query(30, description="Number of days of history to retrieve"),
    api_key: str = Depends(verify_api_key)
):
    """Get historical price data for a specific commodity and region."""
    try:
        # Get commodity data
        commodity_data = get_commodity_data(commodity)
        if not commodity_data:
            raise HTTPException(status_code=404, detail=f"Commodity {commodity} not found")
        
        # Get price history
        history = get_sql_price_history(commodity, region, days)
        
        if not history:
            raise HTTPException(status_code=404, detail="No price history available")
        
        # Format response
        history_entries = []
        for entry in history:
            history_entries.append({
                "date": entry.get("date"),
                "price": entry.get("price"),
                "volume": entry.get("volume", None)
            })
        
        response = {
            "commodity": commodity,
            "region": region,
            "currency": "INR",
            "unit": commodity_data.get("trading_unit", "kg"),
            "data": history_entries
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving price history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Quality Analysis API
@app.post(
    "/quality-analysis/image",
    tags=["Quality Analysis"],
    response_model=QualityAnalysisResponse
)
async def analyze_commodity_quality_from_image(
    commodity: str = Query(..., description="The commodity to analyze"),
    analysis_type: str = Query("detailed", description="Type of analysis to perform"),
    image: UploadFile = File(..., description="Image of the commodity to analyze"),
    api_key: str = Depends(verify_api_key)
):
    """Analyze quality of a commodity from an uploaded image."""
    try:
        # Save uploaded image
        image_path = await save_uploaded_image(image)
        
        # Analyze commodity image
        analysis_result = analyze_commodity_image(image_path, commodity, analysis_type)
        
        if not analysis_result or "error" in analysis_result:
            raise HTTPException(status_code=422, detail="Failed to analyze image")
        
        # Extract quality parameters
        quality_params = analysis_result.get("quality_params", {})
        
        # Format response
        response = {
            "commodity": commodity,
            "quality_params": quality_params,
            "quality_score": quality_params.get("quality_score", 0),
            "quality_grade": quality_params.get("quality_grade", "Unknown"),
            "confidence": quality_params.get("confidence", 0.8),
            "analysis_summary": quality_params.get("ai_summary", ""),
            "timestamp": datetime.now()
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing commodity image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Index API
@app.get(
    "/index/{commodity}",
    tags=["Indices"],
    response_model=CommodityIndex
)
async def get_commodity_index(
    commodity: str,
    days: int = Query(30, description="Number of days of index data to retrieve"),
    api_key: str = Depends(verify_api_key)
):
    """Get index data for a specific commodity."""
    try:
        # Calculate the index
        index_data = get_wizx_indices(commodity, days)
        
        if not index_data:
            raise HTTPException(status_code=404, detail=f"No index data available for {commodity}")
        
        # Extract latest value and history
        current_value = index_data[-1]["value"] if index_data else 0
        previous_value = index_data[-2]["value"] if len(index_data) > 1 else current_value
        
        # Calculate change percentage
        if previous_value > 0:
            change_percentage = ((current_value - previous_value) / previous_value) * 100
        else:
            change_percentage = 0
        
        # Format history entries
        history = []
        for entry in index_data:
            history.append({
                "date": entry.get("date"),
                "value": entry.get("value")
            })
        
        # Build response
        response = {
            "commodity": commodity,
            "current_value": current_value,
            "change_percentage": round(change_percentage, 2),
            "history": history
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving index data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Forecasting API
@app.get(
    "/forecast/{commodity}/{region}",
    tags=["Forecasting"],
    response_model=PriceForecast
)
async def get_price_forecast(
    commodity: str,
    region: str,
    days: int = Query(30, description="Number of days to forecast"),
    api_key: str = Depends(verify_api_key)
):
    """Get price forecast for a specific commodity and region."""
    try:
        # Get forecast data
        forecast = predict_price_trend(commodity, region, days)
        
        if not forecast:
            raise HTTPException(status_code=404, detail="Could not generate forecast")
        
        # Format response
        response = {
            "commodity": commodity,
            "region": region,
            "forecast_data": forecast.get("data", []),
            "model_confidence": forecast.get("confidence", 0.7),
            "created_at": datetime.now()
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Start the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)