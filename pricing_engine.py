"""
Pricing engine module for calculating standardized commodity prices with quality adjustments.
"""

import os
import json
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data_crawler import fetch_agmarknet_data, fetch_commodity_list
from database import get_commodity_data, get_regions, get_quality_impact

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
PRICING_DIR = os.path.join(DATA_DIR, "pricing")

# Ensure directory exists
os.makedirs(PRICING_DIR, exist_ok=True)

def calculate_base_price(commodity, region, date=None):
    """
    Calculate the baseline price for a commodity in a specific region.
    
    Args:
        commodity (str): The commodity
        region (str): The region
        date (datetime, optional): The date for price calculation
        
    Returns:
        float: Base price for the commodity
    """
    logger.info(f"Calculating base price for {commodity} in {region}")
    
    if not date:
        date = datetime.now()
    
    # Get recent price data from the database
    # For now, we're simulating this with random generation
    
    # Get commodity master data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data:
        logger.warning(f"No data found for commodity: {commodity}")
        return 0.0
    
    # Get price range for this commodity
    price_range = commodity_data.get('price_range', {})
    min_price = price_range.get('min', 2000)
    max_price = price_range.get('max', 5000)
    
    # Get historical prices
    # In reality, this would fetch from a database of historical prices
    # For now, generate a reasonable price within the expected range
    
    # Different regions have different base prices
    region_factors = {
        "North India": 1.02,
        "South India": 0.98,
        "East India": 0.95,
        "West India": 1.05,
        "Central India": 1.0
    }
    
    region_factor = region_factors.get(region, 1.0)
    
    # Calculate a base price using a weighted random approach
    # to simulate real-world pricing
    base_price = random.uniform(min_price, max_price) * region_factor
    
    logger.info(f"Base price for {commodity} in {region}: ₹{base_price:.2f}")
    
    return base_price

def calculate_quality_delta(commodity, quality_params, benchmark_params=None):
    """
    Calculate the price adjustment based on quality parameters.
    
    Args:
        commodity (str): The commodity
        quality_params (dict): Quality parameters of the commodity
        benchmark_params (dict, optional): Benchmark quality parameters
        
    Returns:
        float: Quality adjustment delta
    """
    logger.info(f"Calculating quality delta for {commodity}")
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for commodity: {commodity}")
        return 0.0
    
    # Get standard quality parameters if benchmark not provided
    if not benchmark_params:
        benchmark_params = {
            param: details.get('standard_value', (details.get('min', 0) + details.get('max', 100)) / 2)
            for param, details in commodity_data['quality_parameters'].items()
        }
    
    # Calculate quality delta
    delta = 0.0
    
    for param, value in quality_params.items():
        if param in commodity_data['quality_parameters']:
            # Get parameter impact from database
            impact = get_quality_impact(commodity, param)
            
            if impact:
                # Get the benchmark value
                benchmark = benchmark_params.get(param, commodity_data['quality_parameters'][param].get('standard_value', 0))
                
                # Calculate deviation from benchmark
                deviation = value - benchmark
                
                # Calculate impact based on parameter type
                param_type = commodity_data['quality_parameters'][param].get('impact_type', 'linear')
                
                if param_type == 'linear':
                    # Linear impact: value directly affects price
                    param_delta = deviation * impact.get('factor', 0)
                elif param_type == 'threshold':
                    # Threshold impact: price changes beyond thresholds
                    if deviation > 0:
                        param_delta = deviation * impact.get('premium_factor', 0)
                    else:
                        param_delta = deviation * impact.get('discount_factor', 0)
                else:
                    # Default to no impact
                    param_delta = 0
                
                # Apply the parameter delta
                delta += param_delta
    
    logger.info(f"Quality delta for {commodity}: ₹{delta:.2f}")
    
    return delta

def calculate_location_delta(commodity, region):
    """
    Calculate the price adjustment based on location.
    
    Args:
        commodity (str): The commodity
        region (str): The region
        
    Returns:
        float: Location adjustment delta
    """
    logger.info(f"Calculating location delta for {commodity} in {region}")
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data:
        logger.warning(f"No data found for commodity: {commodity}")
        return 0.0
    
    # Get regional factors
    # This would normally come from a database of regional price factors
    
    # For now, use simple regional adjustments
    region_deltas = {
        "North India": {"Wheat": 50, "Rice": -20, "Tur Dal": 30, "Mustard": 40},
        "South India": {"Wheat": -30, "Rice": 40, "Tur Dal": 20, "Groundnut": 50},
        "East India": {"Rice": 30, "Jute": 40, "Masur Dal": 20},
        "West India": {"Wheat": 20, "Cotton": 50, "Groundnut": 30},
        "Central India": {"Wheat": 10, "Soyabean": 40, "Chana Dal": 20}
    }
    
    # Get the delta for this commodity and region
    delta = region_deltas.get(region, {}).get(commodity, 0)
    
    logger.info(f"Location delta for {commodity} in {region}: ₹{delta:.2f}")
    
    return delta

def calculate_market_delta(commodity, date=None):
    """
    Calculate the price adjustment based on current market conditions.
    
    Args:
        commodity (str): The commodity
        date (datetime, optional): The date for calculation
        
    Returns:
        float: Market adjustment delta
    """
    logger.info(f"Calculating market delta for {commodity}")
    
    if not date:
        date = datetime.now()
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data:
        logger.warning(f"No data found for commodity: {commodity}")
        return 0.0
    
    # Calculate seasonal impact
    day_of_year = date.timetuple().tm_yday
    
    # Get seasonal pattern for this commodity
    # This should come from historical analysis
    # For now, use a sinusoidal pattern with commodity-specific parameters
    
    commodity_seasonality = {
        "Wheat": {"amplitude": 0.05, "phase": 0, "period": 365},
        "Rice": {"amplitude": 0.03, "phase": 30, "period": 365},
        "Maize": {"amplitude": 0.06, "phase": 15, "period": 365},
        "Tur Dal": {"amplitude": 0.04, "phase": 45, "period": 365},
        "Soyabean": {"amplitude": 0.07, "phase": 60, "period": 365}
    }
    
    seasonality = commodity_seasonality.get(commodity, {"amplitude": 0.03, "phase": 0, "period": 365})
    
    # Calculate seasonal factor
    amplitude = seasonality["amplitude"]
    phase = seasonality["phase"]
    period = seasonality["period"]
    
    seasonal_factor = amplitude * np.sin(2 * np.pi * (day_of_year - phase) / period)
    
    # Get base price for scaling
    base_price = 4000  # Default value
    
    if commodity_data and 'price_range' in commodity_data:
        base_price = (commodity_data['price_range'].get('min', 2000) + 
                     commodity_data['price_range'].get('max', 5000)) / 2
    
    # Calculate market delta
    delta = base_price * seasonal_factor
    
    # Add some current market adjustment (random for now)
    # In reality, this would be based on recent price movements and other factors
    market_adjustment = random.uniform(-0.02, 0.02) * base_price
    
    delta += market_adjustment
    
    logger.info(f"Market delta for {commodity}: ₹{delta:.2f}")
    
    return delta

def calculate_price(commodity, quality_params, region, date=None):
    """
    Calculate the final price using the formula:
    P_final = P_baseline + Σ(Δ_quality + Δ_location + Δ_market)
    
    Args:
        commodity (str): The commodity
        quality_params (dict): Quality parameters
        region (str): The region
        date (datetime, optional): The date for calculation
        
    Returns:
        tuple: (final_price, base_price, quality_delta, location_delta, market_delta)
    """
    logger.info(f"Calculating price for {commodity} in {region}")
    
    if not date:
        date = datetime.now()
    
    # Calculate components
    base_price = calculate_base_price(commodity, region, date)
    quality_delta = calculate_quality_delta(commodity, quality_params)
    location_delta = calculate_location_delta(commodity, region)
    market_delta = calculate_market_delta(commodity, date)
    
    # Calculate final price
    final_price = base_price + quality_delta + location_delta + market_delta
    
    # Ensure price doesn't go below zero
    final_price = max(0, final_price)
    
    logger.info(f"Final price for {commodity} in {region}: ₹{final_price:.2f}")
    
    return (final_price, base_price, quality_delta, location_delta, market_delta)

def get_price_history(commodity, region, days, start_date=None):
    """
    Get historical price data for a commodity in a region.
    
    Args:
        commodity (str): The commodity
        region (str): The region
        days (int): Number of days of history
        start_date (datetime, optional): Start date for history
        
    Returns:
        list: Historical price data
    """
    logger.info(f"Getting price history for {commodity} in {region} for {days} days")
    
    if not start_date:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    else:
        end_date = start_date + timedelta(days=days)
    
    # In a real implementation, this would fetch from a database
    # For now, generate sample data
    
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Get commodity data for price range
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data:
        logger.warning(f"No data found for commodity: {commodity}")
        return []
    
    # Get price range for this commodity
    price_range = commodity_data.get('price_range', {})
    min_price = price_range.get('min', 2000)
    max_price = price_range.get('max', 5000)
    
    # Different regions have different base prices
    region_factors = {
        "North India": 1.02,
        "South India": 0.98,
        "East India": 0.95,
        "West India": 1.05,
        "Central India": 1.0
    }
    
    region_factor = region_factors.get(region, 1.0)
    
    # Generate price history
    history = []
    
    # Set a starting price in the middle of the range
    price = (min_price + max_price) / 2 * region_factor
    
    # Add some seasonal component
    for date in date_range:
        # Add some random walk to the price (with mean reversion)
        random_change = random.normalvariate(0, 1) * (max_price - min_price) * 0.005
        mean_reversion = ((min_price + max_price) / 2 * region_factor - price) * 0.05
        
        # Add seasonality
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 0.03 * np.sin(2 * np.pi * day_of_year / 365)
        seasonal_change = price * seasonal_factor * 0.1
        
        # Update price
        price += random_change + mean_reversion + seasonal_change
        
        # Ensure price stays within reasonable bounds
        price = max(min_price * 0.8, min(max_price * 1.2, price))
        
        history.append({
            "date": date,
            "price": round(price, 2)
        })
    
    return history

def calculate_price_curve(commodity, quality_param, region, min_value=None, max_value=None, points=20):
    """
    Calculate a price curve for a commodity by varying a quality parameter.
    
    Args:
        commodity (str): The commodity
        quality_param (str): The quality parameter to vary
        region (str): The region
        min_value (float, optional): Minimum parameter value
        max_value (float, optional): Maximum parameter value
        points (int): Number of points in the curve
        
    Returns:
        dict: Price curve data
    """
    logger.info(f"Calculating price curve for {commodity} in {region} by varying {quality_param}")
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for commodity: {commodity}")
        return {"status": "error", "message": "Commodity data not found"}
    
    # Check if the quality parameter exists
    if quality_param not in commodity_data['quality_parameters']:
        logger.warning(f"Quality parameter {quality_param} not found for {commodity}")
        return {"status": "error", "message": "Quality parameter not found"}
    
    # Get parameter range
    param_data = commodity_data['quality_parameters'][quality_param]
    
    if min_value is None:
        min_value = param_data.get('min', 0)
    
    if max_value is None:
        max_value = param_data.get('max', 100)
    
    # Generate parameter values
    param_values = np.linspace(min_value, max_value, points)
    
    # Calculate prices for each parameter value
    curve_data = []
    
    # Get standard quality parameters
    standard_params = {
        param: details.get('standard_value', (details.get('min', 0) + details.get('max', 100)) / 2)
        for param, details in commodity_data['quality_parameters'].items()
    }
    
    for value in param_values:
        # Update the varied parameter
        test_params = standard_params.copy()
        test_params[quality_param] = value
        
        # Calculate price
        final_price, base_price, quality_delta, location_delta, market_delta = calculate_price(
            commodity, test_params, region
        )
        
        curve_data.append({
            "parameter_value": value,
            "final_price": final_price,
            "base_price": base_price,
            "quality_delta": quality_delta,
            "location_delta": location_delta,
            "market_delta": market_delta
        })
    
    return {
        "status": "success",
        "commodity": commodity,
        "quality_parameter": quality_param,
        "region": region,
        "unit": param_data.get('unit', ''),
        "standard_value": param_data.get('standard_value', (param_data.get('min', 0) + param_data.get('max', 100)) / 2),
        "curve": curve_data
    }

def export_pricing_model(commodity, filename=None):
    """
    Export a pricing model for a commodity.
    
    Args:
        commodity (str): The commodity
        filename (str, optional): Output filename
        
    Returns:
        str: Path to the exported file
    """
    logger.info(f"Exporting pricing model for {commodity}")
    
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{commodity.lower().replace(' ', '_')}_pricing_model_{timestamp}.json"
    
    filepath = os.path.join(PRICING_DIR, filename)
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data:
        logger.warning(f"No data found for commodity: {commodity}")
        return None
    
    # Generate model data
    model_data = {
        "commodity": commodity,
        "model_generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "quality_parameters": commodity_data.get('quality_parameters', {}),
        "price_range": commodity_data.get('price_range', {}),
        "regions": get_regions(commodity)
    }
    
    # Generate curves for each quality parameter
    curves = {}
    
    for param in commodity_data.get('quality_parameters', {}):
        # Generate curve for each region
        region_curves = {}
        
        for region in get_regions(commodity):
            curve_data = calculate_price_curve(commodity, param, region)
            if curve_data['status'] == 'success':
                region_curves[region] = curve_data['curve']
        
        curves[param] = region_curves
    
    model_data['curves'] = curves
    
    # Export model
    with open(filepath, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    logger.info(f"Exported pricing model to {filepath}")
    
    return filepath

if __name__ == "__main__":
    # Test the pricing engine
    print("Testing pricing engine...")
    
    commodity = "Wheat"
    region = "North India"
    
    # Test quality parameters
    quality_params = {
        "moisture": 13.0,
        "foreign_matter": 1.5,
        "damaged_grains": 3.0
    }
    
    final_price, base_price, quality_delta, location_delta, market_delta = calculate_price(
        commodity, quality_params, region
    )
    
    print(f"Price components for {commodity} in {region}:")
    print(f"Base Price: ₹{base_price:.2f}")
    print(f"Quality Delta: ₹{quality_delta:.2f}")
    print(f"Location Delta: ₹{location_delta:.2f}")
    print(f"Market Delta: ₹{market_delta:.2f}")
    print(f"Final Price: ₹{final_price:.2f}")
