"""
Pricing engine module for agricultural commodities.

This module provides functions for price calculation based on quality parameters,
market conditions and regional factors.
"""

import logging
import json
import os
from datetime import datetime, date, timedelta
import random
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import database functionality
from database_sql import (
    get_commodity_data,
    get_price_quality_parameters,
    get_regions,
    get_price_history,
    get_quality_impact,
    get_commodity_prices
)


def calculate_price(commodity, quality_params=None, region=None):
    """
    Calculate price for a commodity based on quality parameters and region.
    
    Args:
        commodity (str): The commodity name
        quality_params (dict, optional): Quality parameters
        region (str, optional): Region name
        
    Returns:
        dict: Price calculation results
    """
    try:
        # Default parameters
        if quality_params is None:
            quality_params = {}
        
        if region is None:
            region = "National"
        
        # Get commodity data first
        commodity_data = get_commodity_data(commodity)
        
        if not commodity_data:
            logger.warning(f"Commodity {commodity} not found")
            return None
        
        # Get price quality parameters for this commodity
        std_quality_params = get_price_quality_parameters(commodity)
        
        # Get commodity prices from the database
        price_data = get_commodity_prices(commodity, region)
        
        if not price_data:
            logger.warning(f"No price data for {commodity} in {region}")
            return {
                "commodity": commodity,
                "region": region,
                "base_price": 0,
                "final_price": 0,
                "quality_delta": 0,
                "location_delta": 0,
                "market_delta": 0,
                "currency": "INR",
                "unit": "kg",
                "timestamp": datetime.now(),
                "confidence": 0.5,
                "price_rating": "Unknown"
            }
        
        # Base values from the database
        base_price = price_data.get("base_price", 0)
        location_delta = price_data.get("location_delta", 0)
        
        # Calculate quality delta based on provided parameters
        quality_delta = calculate_quality_delta(commodity, quality_params, base_price)
        
        # Get market delta from database or calculate a reasonable value
        market_delta = price_data.get("market_delta", 0)
        
        # Calculate final price
        final_price = base_price + quality_delta + location_delta + market_delta
        
        # Ensure final price is non-negative
        final_price = max(final_price, 0)
        
        # Determine price rating based on quality and market factors
        price_rating = calculate_price_rating(quality_delta, location_delta, market_delta, base_price)
        
        # Return complete price structure
        return {
            "commodity": commodity,
            "region": region,
            "base_price": base_price,
            "final_price": final_price,
            "quality_delta": quality_delta,
            "location_delta": location_delta,
            "market_delta": market_delta,
            "currency": "INR",
            "unit": "kg",
            "timestamp": datetime.now(),
            "confidence": calculate_confidence(commodity, quality_params, region),
            "price_rating": price_rating
        }
        
    except Exception as e:
        logger.error(f"Error calculating price: {e}")
        return None


def calculate_quality_delta(commodity, quality_params, base_price):
    """
    Calculate price adjustment based on quality parameters.
    
    Args:
        commodity (str): The commodity name
        quality_params (dict): Quality parameters
        base_price (float): Base price for reference
        
    Returns:
        float: Quality-based price adjustment
    """
    try:
        if not quality_params:
            return 0
        
        # Get standard quality parameters for this commodity
        std_quality_params = get_price_quality_parameters(commodity)
        
        if not std_quality_params:
            logger.warning(f"No standard quality parameters found for {commodity}")
            
            # Use a basic calculation when no standards available
            quality_delta = 0
            
            # Common quality factors
            if "moisture_content" in quality_params:
                moisture = quality_params["moisture_content"]
                # Lower moisture generally increases value (within limits)
                if moisture < 12:
                    quality_delta += base_price * 0.02  # 2% premium
                elif moisture > 16:
                    quality_delta -= base_price * 0.03  # 3% discount
            
            if "foreign_matter" in quality_params:
                foreign = quality_params["foreign_matter"]
                # Lower foreign matter increases value
                if foreign < 0.5:
                    quality_delta += base_price * 0.015  # 1.5% premium
                elif foreign > 2:
                    quality_delta -= base_price * 0.025 * (foreign / 2)  # Discount proportional to excess
            
            return quality_delta
        
        # Calculate quality delta using impact factors
        quality_delta = 0
        
        for param_name, param_value in quality_params.items():
            if param_name in std_quality_params:
                std_param = std_quality_params[param_name]
                
                # Skip if necessary fields are missing
                if "min" not in std_param or "max" not in std_param or "impact_factor" not in std_param:
                    continue
                
                # Get parameter range and impact
                param_min = std_param.get("min", 0)
                param_max = std_param.get("max", 100)
                param_std = std_param.get("standard_value", (param_min + param_max) / 2)
                impact_factor = std_param.get("impact_factor", 1.0)
                
                # Apply customized logic by commodity type
                impact = get_quality_impact(commodity, param_name)
                
                if impact:
                    # Use the lookup function result if available
                    quality_delta += impact * base_price
                else:
                    # Calculate where this value falls in the range
                    if param_max == param_min:
                        continue  # Avoid division by zero
                    
                    # Normalize the parameter to 0-1 range
                    param_range = param_max - param_min
                    normalized_value = (param_value - param_min) / param_range
                    normalized_std = (param_std - param_min) / param_range
                    
                    # Calculate the price adjustment factor
                    if impact_factor > 0:
                        # For positive impact, higher than standard is good
                        adjustment = (normalized_value - normalized_std) * impact_factor
                    else:
                        # For negative impact, lower than standard is good
                        adjustment = (normalized_std - normalized_value) * abs(impact_factor)
                    
                    # Convert adjustment factor to price change
                    param_delta = adjustment * base_price * 0.05  # 5% max impact per parameter
                    quality_delta += param_delta
        
        # Cap the quality delta to reasonable limits
        max_delta = base_price * 0.25  # 25% max adjustment
        return max(-max_delta, min(quality_delta, max_delta))
        
    except Exception as e:
        logger.error(f"Error calculating quality delta: {e}")
        return 0


def calculate_confidence(commodity, quality_params, region):
    """
    Calculate confidence level for price calculation.
    
    Args:
        commodity (str): The commodity name
        quality_params (dict): Quality parameters
        region (str): Region name
        
    Returns:
        float: Confidence level (0-1)
    """
    try:
        # Base confidence
        confidence = 0.85
        
        # Adjust based on data completeness
        if not quality_params:
            confidence -= 0.15  # Significant reduction for missing quality data
        
        # Check parameter coverage
        std_quality_params = get_price_quality_parameters(commodity)
        if std_quality_params:
            # Count how many standard parameters are provided
            param_coverage = sum(1 for param in std_quality_params if param in quality_params)
            total_params = len(std_quality_params)
            
            if total_params > 0:
                param_ratio = param_coverage / total_params
                if param_ratio < 0.5:
                    confidence -= 0.1  # Reduce confidence for low parameter coverage
                elif param_ratio > 0.8:
                    confidence += 0.05  # Increase confidence for high parameter coverage
        
        # Check price history data
        price_history = get_price_history(commodity, region, days=30)
        if not price_history:
            confidence -= 0.1  # Reduce confidence for missing price history
        elif len(price_history) < 5:
            confidence -= 0.05  # Reduce confidence for limited price history
        
        # Cap confidence between 0.5 and 1.0
        return max(0.5, min(confidence, 1.0))
    
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        return 0.75  # Return a reasonable default


def calculate_price_rating(quality_delta, location_delta, market_delta, base_price):
    """
    Calculate price rating based on adjustment factors.
    
    Args:
        quality_delta (float): Quality adjustment
        location_delta (float): Location adjustment
        market_delta (float): Market adjustment
        base_price (float): Base price
        
    Returns:
        str: Price rating
    """
    try:
        # Calculate total delta as percentage of base price
        if base_price == 0:
            return "Unknown"
        
        total_delta = quality_delta + location_delta + market_delta
        delta_percent = (total_delta / base_price) * 100
        
        # Determine rating based on delta percentage
        if delta_percent > 15:
            return "Premium"
        elif delta_percent > 5:
            return "Above Market"
        elif delta_percent > -5:
            return "Fair Market"
        elif delta_percent > -15:
            return "Below Market"
        else:
            return "Discount"
    
    except Exception as e:
        logger.error(f"Error calculating price rating: {e}")
        return "Fair"


def calculate_price_forecast(commodity, region, days=30):
    """
    Calculate price forecast for a commodity in a region.
    
    Args:
        commodity (str): The commodity name
        region (str): Region name
        days (int): Number of days to forecast
        
    Returns:
        list: Forecasted price data
    """
    try:
        # Get historical price data
        history = get_price_history(commodity, region, days=90)
        
        if not history or len(history) < 7:
            logger.warning(f"Insufficient historical data for {commodity} in {region}")
            return []
        
        # Get current price
        price_data = get_commodity_prices(commodity, region)
        if not price_data:
            logger.warning(f"No current price data for {commodity} in {region}")
            return []
        
        current_price = price_data.get("final_price", 0)
        
        # In a real system, this would use time series forecasting models
        # For this implementation, we'll create a reasonable trend-based forecast
        
        # Extract just the prices from history
        prices = [entry.get("price", 0) for entry in history]
        
        # Calculate a simple trend
        if len(prices) >= 30:
            # Use the last 30 days to calculate trend
            recent_prices = prices[-30:]
            start_avg = sum(recent_prices[:10]) / 10
            end_avg = sum(recent_prices[-10:]) / 10
            daily_trend = (end_avg - start_avg) / 20  # Average daily change
        else:
            # Use whatever history we have
            if len(prices) < 2:
                daily_trend = 0
            else:
                daily_trend = (prices[-1] - prices[0]) / (len(prices) - 1)
        
        # Calculate basic volatility
        if len(prices) >= 7:
            diffs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            avg_daily_change = sum(diffs) / len(diffs)
            volatility = avg_daily_change / (sum(prices) / len(prices)) if sum(prices) > 0 else 0.01
        else:
            volatility = 0.01  # Default 1% daily volatility
        
        # Generate forecast
        forecast = []
        current_date = datetime.now().date()
        forecast_price = current_price
        
        for i in range(days):
            forecast_date = current_date + timedelta(days=i+1)
            
            # Add some randomness to the trend based on volatility
            random_factor = random.uniform(-1, 1) * volatility * forecast_price
            forecast_price = max(0, forecast_price + daily_trend + random_factor)
            
            # Add seasonal pattern (if appropriate)
            # This is a simplified example - real forecasts would use proper time series models
            seasonal_factor = 0.02 * forecast_price * math.sin(2 * math.pi * (i % 30) / 30)
            forecast_price += seasonal_factor
            
            forecast.append({
                "date": forecast_date,
                "price": forecast_price,
                "lower_bound": max(0, forecast_price * (1 - volatility * (i+1)/days * 2)),
                "upper_bound": forecast_price * (1 + volatility * (i+1)/days * 2)
            })
        
        return forecast
    
    except Exception as e:
        logger.error(f"Error calculating price forecast: {e}")
        return []


def get_price_comparison(commodity, regions=None):
    """
    Get price comparison across different regions.
    
    Args:
        commodity (str): The commodity name
        regions (list, optional): List of regions to compare
        
    Returns:
        dict: Price comparison data
    """
    try:
        # Get regions if not provided
        if not regions:
            regions = get_regions(commodity)
            
            if not regions:
                logger.warning(f"No regions found for {commodity}")
                return {}
        
        # Get prices for each region
        prices = {}
        for region in regions:
            price_data = get_commodity_prices(commodity, region)
            if price_data:
                prices[region] = price_data
        
        if not prices:
            logger.warning(f"No price data found for {commodity}")
            return {}
        
        # Calculate statistics
        avg_price = sum(p.get("final_price", 0) for p in prices.values()) / len(prices) if prices else 0
        min_region = min(prices.keys(), key=lambda r: prices[r].get("final_price", 0)) if prices else None
        max_region = max(prices.keys(), key=lambda r: prices[r].get("final_price", 0)) if prices else None
        
        min_price = prices[min_region].get("final_price", 0) if min_region else 0
        max_price = prices[max_region].get("final_price", 0) if max_region else 0
        
        # Calculate price variance
        if len(prices) > 1:
            variance = sum((p.get("final_price", 0) - avg_price) ** 2 for p in prices.values()) / len(prices)
            std_dev = variance ** 0.5
            coefficient_of_variation = (std_dev / avg_price) * 100 if avg_price > 0 else 0
        else:
            variance = 0
            std_dev = 0
            coefficient_of_variation = 0
        
        return {
            "commodity": commodity,
            "average_price": avg_price,
            "min_price": min_price,
            "min_region": min_region,
            "max_price": max_price,
            "max_region": max_region,
            "price_range": max_price - min_price,
            "price_range_percent": ((max_price - min_price) / avg_price) * 100 if avg_price > 0 else 0,
            "standard_deviation": std_dev,
            "coefficient_of_variation": coefficient_of_variation,
            "prices": prices
        }
        
    except Exception as e:
        logger.error(f"Error getting price comparison: {e}")
        return {}