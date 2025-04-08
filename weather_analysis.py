
"""
Weather analysis module for assessing weather impact on commodity prices and quality
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from models import predict_price_trend
from database import get_commodity_data

logger = logging.getLogger(__name__)

def analyze_weather_impact(commodity, region, weather_data=None):
    """Analyze weather impact on commodity prices and quality"""
    # In real implementation, this would use actual weather data
    # For demo, using synthetic data with seasonal patterns
    
    commodity_data = get_commodity_data(commodity)
    if not commodity_data:
        return None
        
    # Simulate weather patterns
    weather_patterns = {
        "Wheat": {
            "optimal_temp": (20, 25),
            "optimal_rainfall": (500, 700),
            "growing_season": (11, 3),  # Nov-Mar
            "harvest_month": 4  # April
        },
        "Rice": {
            "optimal_temp": (24, 30),
            "optimal_rainfall": (1000, 1500),
            "growing_season": (6, 10),  # Jun-Oct
            "harvest_month": 11  # November
        },
        "Tur Dal": {
            "optimal_temp": (25, 30),
            "optimal_rainfall": (600, 1000),
            "growing_season": (6, 9),  # Jun-Sep
            "harvest_month": 10  # October
        }
    }
    
    pattern = weather_patterns.get(commodity, {
        "optimal_temp": (25, 30),
        "optimal_rainfall": (800, 1200),
        "growing_season": (6, 9),
        "harvest_month": 10
    })
    
    current_month = datetime.now().month
    months_since_harvest = (current_month - pattern["harvest_month"]) % 12
    
    # Calculate storage impact
    storage_impact = calculate_storage_impact(months_since_harvest)
    
    # Calculate seasonal risk
    seasonal_risk = calculate_seasonal_risk(commodity, current_month, pattern)
    
    return {
        "weather_pattern": pattern,
        "storage_impact": storage_impact,
        "seasonal_risk": seasonal_risk,
        "months_since_harvest": months_since_harvest,
        "next_harvest_in": (pattern["harvest_month"] - current_month) % 12
    }

def calculate_storage_impact(months):
    """Calculate price impact from storage duration"""
    # Base storage cost per month (% of base price)
    base_storage_cost = 0.015  # 1.5% per month
    
    # Progressive increase in storage cost
    storage_factor = 1 + (base_storage_cost * months * (1 + months * 0.1))
    
    return storage_factor

def calculate_seasonal_risk(commodity, current_month, pattern):
    """Calculate seasonal risk score based on growing season"""
    growing_start, growing_end = pattern["growing_season"]
    
    # Check if currently in growing season
    in_growing_season = (
        (growing_start <= current_month <= growing_end) if growing_start <= growing_end
        else (current_month >= growing_start or current_month <= growing_end)
    )
    
    risk_factors = {
        "in_growing_season": in_growing_season,
        "risk_level": "low" if not in_growing_season else "medium",
        "price_volatility": "low" if not in_growing_season else "high"
    }
    
    return risk_factors
