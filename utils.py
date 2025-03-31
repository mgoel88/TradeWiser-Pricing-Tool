"""
Utility functions for the agricultural commodity pricing engine.
"""

import os
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def generate_cache_key(prefix, *args, **kwargs):
    """
    Generate a cache key based on arguments.
    
    Args:
        prefix (str): Cache key prefix
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        str: Cache key
    """
    # Convert arguments to strings
    args_str = '_'.join(str(arg) for arg in args)
    kwargs_str = '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    
    # Combine and hash
    combined = f"{prefix}_{args_str}_{kwargs_str}"
    hashed = hashlib.md5(combined.encode()).hexdigest()
    
    return f"{prefix}_{hashed}"

def get_cached_data(cache_key, expiry=3600):
    """
    Retrieve data from cache if available and not expired.
    
    Args:
        cache_key (str): Cache key
        expiry (int): Cache expiry time in seconds
        
    Returns:
        dict or None: Cached data if available and not expired
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    if not os.path.exists(cache_path):
        return None
    
    # Check if cache has expired
    modified_time = os.path.getmtime(cache_path)
    current_time = time.time()
    
    if current_time - modified_time > expiry:
        logger.debug(f"Cache expired for {cache_key}")
        return None
    
    # Load cached data
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading cache: {e}")
        return None

def save_cached_data(cache_key, data):
    """
    Save data to cache.
    
    Args:
        cache_key (str): Cache key
        data: Data to cache
        
    Returns:
        bool: True if successful
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        return False

def cache_result(prefix, expiry=3600):
    """
    Decorator to cache function results.
    
    Args:
        prefix (str): Cache key prefix
        expiry (int): Cache expiry time in seconds
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            # Check cache
            cached_data = get_cached_data(cache_key, expiry)
            
            if cached_data is not None:
                logger.debug(f"Using cached data for {func.__name__}")
                return cached_data
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            save_cached_data(cache_key, result)
            
            return result
        
        return wrapper
    
    return decorator

def format_currency(value, currency="₹"):
    """
    Format a currency value.
    
    Args:
        value (float): The value to format
        currency (str): Currency symbol
        
    Returns:
        str: Formatted currency string
    """
    return f"{currency}{value:.2f}"

def format_percentage(value):
    """
    Format a percentage value.
    
    Args:
        value (float): The value to format
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value:.2f}%"

def convert_to_quintal(value, unit):
    """
    Convert a value to quintals.
    
    Args:
        value (float): The value to convert
        unit (str): The unit to convert from
        
    Returns:
        float: Value in quintals
    """
    # Conversion factors to quintals
    conversion_factors = {
        "kg": 0.01,
        "ton": 10,
        "mt": 10,  # Metric ton
        "quintal": 1,
        "g": 0.00001,
        "lb": 0.00453592  # Pounds
    }
    
    unit_lower = unit.lower()
    
    if unit_lower in conversion_factors:
        return value * conversion_factors[unit_lower]
    
    logger.warning(f"Unknown unit for conversion: {unit}")
    return value

def normalize_commodity_name(name):
    """
    Normalize a commodity name for consistent reference.
    
    Args:
        name (str): Commodity name
        
    Returns:
        str: Normalized name
    """
    # Common commodity name variations
    name_mapping = {
        "wheat": "Wheat",
        "gehu": "Wheat",
        "atta": "Wheat",
        
        "rice": "Rice",
        "chawal": "Rice",
        "paddy": "Rice",
        
        "tur": "Tur Dal",
        "tur dal": "Tur Dal",
        "arhar": "Tur Dal",
        "red gram": "Tur Dal",
        
        "soyabean": "Soyabean",
        "soybean": "Soyabean",
        "soya": "Soyabean",
        
        "mustard": "Mustard",
        "sarson": "Mustard",
        "rai": "Mustard"
    }
    
    # Normalize to lowercase
    name_lower = name.lower()
    
    # Check for direct match
    if name_lower in name_mapping:
        return name_mapping[name_lower]
    
    # Check for partial match
    for key, value in name_mapping.items():
        if key in name_lower or name_lower in key:
            return value
    
    # If no match found, capitalize first letter of each word
    return ' '.join(word.capitalize() for word in name_lower.split())

def get_seasonal_index(commodity, month):
    """
    Get seasonal index for a commodity in a specific month.
    
    Args:
        commodity (str): The commodity
        month (int): Month (1-12)
        
    Returns:
        float: Seasonal index (1.0 = average)
    """
    # Seasonal patterns for different commodities
    # These are simplified patterns for demonstration
    # In a real implementation, this would be calculated from historical data
    seasonal_patterns = {
        "Wheat": {
            1: 0.95, 2: 0.92, 3: 0.90, 4: 0.95, 5: 1.05, 6: 1.15,
            7: 1.10, 8: 1.05, 9: 1.00, 10: 0.98, 11: 0.97, 12: 0.95
        },
        "Rice": {
            1: 0.98, 2: 0.97, 3: 0.95, 4: 0.93, 5: 0.92, 6: 0.95,
            7: 1.00, 8: 1.05, 9: 1.10, 10: 1.07, 11: 1.05, 12: 1.00
        },
        "Tur Dal": {
            1: 1.05, 2: 1.03, 3: 1.00, 4: 0.97, 5: 0.95, 6: 0.93,
            7: 0.95, 8: 0.98, 9: 1.00, 10: 1.03, 11: 1.05, 12: 1.05
        },
        "Soyabean": {
            1: 1.00, 2: 1.02, 3: 1.05, 4: 1.07, 5: 1.05, 6: 1.00,
            7: 0.95, 8: 0.90, 9: 0.90, 10: 0.95, 11: 0.98, 12: 1.00
        },
        "Mustard": {
            1: 1.00, 2: 0.98, 3: 0.95, 4: 0.93, 5: 0.95, 6: 0.98,
            7: 1.00, 8: 1.02, 9: 1.05, 10: 1.07, 11: 1.05, 12: 1.02
        }
    }
    
    # Default pattern if commodity not found
    default_pattern = {
        1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00, 5: 1.00, 6: 1.00,
        7: 1.00, 8: 1.00, 9: 1.00, 10: 1.00, 11: 1.00, 12: 1.00
    }
    
    # Get pattern for this commodity
    pattern = seasonal_patterns.get(commodity, default_pattern)
    
    # Get index for this month
    return pattern.get(month, 1.0)

def calculate_market_sentiment(commodity, days=7):
    """
    Calculate market sentiment for a commodity.
    
    Args:
        commodity (str): The commodity
        days (int): Number of days to analyze
        
    Returns:
        dict: Sentiment analysis results
    """
    # In a real implementation, this would analyze news, social media, and other sources
    # For demonstration, generate a random sentiment with bias
    
    # Random sentiment score (-1 to 1)
    base_sentiment = np.random.normal(0, 0.5)
    base_sentiment = max(-1, min(1, base_sentiment))
    
    # Add some commodity-specific bias
    commodity_bias = {
        "Wheat": 0.2,
        "Rice": 0.1,
        "Tur Dal": -0.1,
        "Soyabean": 0.0,
        "Mustard": -0.2
    }
    
    bias = commodity_bias.get(commodity, 0.0)
    
    # Calculate final sentiment
    sentiment_score = base_sentiment + bias
    sentiment_score = max(-1, min(1, sentiment_score))
    
    # Determine sentiment category
    if sentiment_score >= 0.5:
        sentiment = "Bullish"
    elif sentiment_score >= 0.2:
        sentiment = "Moderately Bullish"
    elif sentiment_score >= -0.2:
        sentiment = "Neutral"
    elif sentiment_score >= -0.5:
        sentiment = "Moderately Bearish"
    else:
        sentiment = "Bearish"
    
    # Generate price impact estimate
    price_impact = sentiment_score * 0.05  # 5% max impact
    
    return {
        'commodity': commodity,
        'sentiment_score': sentiment_score,
        'sentiment': sentiment,
        'price_impact': price_impact,
        'days_analyzed': days
    }

def generate_date_range(start_date, end_date=None, days=None):
    """
    Generate a date range.
    
    Args:
        start_date (str or datetime): Start date
        end_date (str or datetime, optional): End date
        days (int, optional): Number of days
        
    Returns:
        list: List of dates
    """
    # Convert to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    if end_date is None and days is not None:
        end_date = start_date + timedelta(days=days)
    
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date)
    
    return dates

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
