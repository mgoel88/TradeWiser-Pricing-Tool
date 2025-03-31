"""
Data crawler module for fetching agricultural commodity pricing data from various sources.
"""

import os
import time
import random
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urljoin
import trafilatura
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
AGMARKNET_BASE_URL = "https://agmarknet.gov.in/"
ENAM_BASE_URL = "https://enam.gov.in/web/"
DATA_DIR = "./data"
CACHE_EXPIRY = 24 * 60 * 60  # 24 hours in seconds

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def get_cached_data(filename, expiry=CACHE_EXPIRY):
    """
    Retrieve cached data if available and not expired.
    
    Args:
        filename (str): The name of the cache file
        expiry (int): Cache expiry time in seconds
        
    Returns:
        dict or None: The cached data if available and not expired, otherwise None
    """
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        return None
    
    # Check if cache has expired
    modified_time = os.path.getmtime(filepath)
    current_time = time.time()
    
    if current_time - modified_time > expiry:
        logger.info(f"Cache expired for {filename}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading cache file {filename}: {e}")
        return None

def save_cached_data(data, filename):
    """
    Save data to cache file.
    
    Args:
        data (dict): The data to cache
        filename (str): The name of the cache file
    """
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f)
        logger.info(f"Data cached to {filename}")
    except Exception as e:
        logger.error(f"Error caching data to {filename}: {e}")

def fetch_agmarknet_data(commodity=None, start_date=None, end_date=None, state=None):
    """
    Fetch commodity data from Agmarknet portal.
    
    Args:
        commodity (str, optional): Specific commodity to fetch
        start_date (datetime, optional): Start date for data
        end_date (datetime, optional): End date for data
        state (str, optional): State filter
        
    Returns:
        list: List of daily price data
    """
    logger.info(f"Fetching Agmarknet data for commodity: {commodity}, state: {state}")
    
    # Prepare cache key
    cache_params = []
    if commodity:
        cache_params.append(commodity)
    if state:
        cache_params.append(state)
    if start_date:
        cache_params.append(start_date.strftime('%Y%m%d'))
    if end_date:
        cache_params.append(end_date.strftime('%Y%m%d'))
    
    cache_key = f"agmarknet_{'_'.join(cache_params) if cache_params else 'all'}.json"
    
    # Check cache first
    cached_data = get_cached_data(cache_key)
    if cached_data:
        logger.info(f"Using cached Agmarknet data: {cache_key}")
        return cached_data
    
    # If not in cache, would normally scrape the website
    # For this implementation, we'll generate mock data that resembles real data
    
    # Mock data generation
    commodities = fetch_commodity_list() if not commodity else [commodity]
    states = ["Maharashtra", "Gujarat", "Punjab", "Uttar Pradesh", "Karnataka"] if not state else [state]
    
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()
    
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Generate mock data
    data = []
    
    for curr_date in date_range:
        for curr_commodity in commodities:
            for curr_state in states:
                # Get markets for the state
                markets = get_markets_for_state(curr_state)
                
                for market in markets[:3]:  # Limit to 3 markets per state for mock data
                    price_data = {
                        "date": curr_date.strftime('%Y-%m-%d'),
                        "commodity": curr_commodity,
                        "state": curr_state,
                        "market": market,
                        "min_price": round(random.uniform(2000, 3500), 2),
                        "max_price": round(random.uniform(3500, 5000), 2),
                        "modal_price": round(random.uniform(3000, 4500), 2),
                        "arrival_qty": round(random.uniform(10, 500), 2)
                    }
                    
                    data.append(price_data)
    
    # Cache the data
    save_cached_data(data, cache_key)
    
    return data

def fetch_enam_data(commodity=None, market=None, date=None):
    """
    Fetch commodity data from eNAM portal.
    
    Args:
        commodity (str, optional): Specific commodity to fetch
        market (str, optional): Specific market to fetch from
        date (datetime, optional): Date for which to fetch data
        
    Returns:
        list: List of price data
    """
    logger.info(f"Fetching eNAM data for commodity: {commodity}, market: {market}")
    
    # Prepare cache key
    cache_params = []
    if commodity:
        cache_params.append(commodity)
    if market:
        cache_params.append(market)
    if date:
        cache_params.append(date.strftime('%Y%m%d'))
    
    cache_key = f"enam_{'_'.join(cache_params) if cache_params else 'all'}.json"
    
    # Check cache first
    cached_data = get_cached_data(cache_key)
    if cached_data:
        logger.info(f"Using cached eNAM data: {cache_key}")
        return cached_data
    
    # If not in cache, would normally scrape the website
    # For this implementation, we'll generate mock data that resembles real data
    
    # Mock data generation
    commodities = fetch_commodity_list() if not commodity else [commodity]
    markets = ["Ahmednagar", "Pune", "Nashik", "Nagpur", "Amritsar"] if not market else [market]
    
    if not date:
        date = datetime.now() - timedelta(days=1)
    
    # Generate mock data
    data = []
    
    for curr_commodity in commodities:
        for curr_market in markets:
            price_data = {
                "date": date.strftime('%Y-%m-%d'),
                "commodity": curr_commodity,
                "market": curr_market,
                "min_price": round(random.uniform(2000, 3500), 2),
                "max_price": round(random.uniform(3500, 5000), 2),
                "avg_price": round(random.uniform(3000, 4500), 2),
                "traded_qty": round(random.uniform(10, 500), 2)
            }
            
            data.append(price_data)
    
    # Cache the data
    save_cached_data(data, cache_key)
    
    return data

def fetch_commodity_list(source="agmarknet", region=None):
    """
    Fetch list of available commodities from the specified source.
    
    Args:
        source (str): Data source (agmarknet, enam, etc.)
        region (str, optional): Region filter
        
    Returns:
        list: List of commodity names
    """
    cache_key = f"{source}_commodities.json"
    
    # Check cache first
    cached_data = get_cached_data(cache_key)
    if cached_data:
        # If region filter is provided, filter the cached data
        if region and 'by_region' in cached_data:
            return cached_data.get('by_region', {}).get(region, [])
        return cached_data.get('all', [])
    
    # Mock commodity list
    all_commodities = [
        "Wheat", "Rice", "Maize", "Jowar", "Bajra",
        "Tur Dal", "Moong Dal", "Urad Dal", "Masur Dal", "Chana Dal",
        "Soyabean", "Groundnut", "Mustard", "Sunflower",
        "Cotton", "Jute", "Onion", "Potato", "Tomato"
    ]
    
    # Region-specific commodities for mock data
    by_region = {
        "North India": ["Wheat", "Rice", "Maize", "Bajra", "Chana Dal", "Mustard"],
        "South India": ["Rice", "Jowar", "Tur Dal", "Urad Dal", "Groundnut", "Tomato"],
        "East India": ["Rice", "Jute", "Masur Dal", "Mustard", "Potato"],
        "West India": ["Wheat", "Jowar", "Tur Dal", "Groundnut", "Cotton", "Onion"],
        "Central India": ["Wheat", "Soyabean", "Chana Dal", "Cotton", "Onion", "Potato"]
    }
    
    # Structure the data
    data = {
        'all': all_commodities,
        'by_region': by_region
    }
    
    # Cache the data
    save_cached_data(data, cache_key)
    
    # Return based on region filter
    if region and region in by_region:
        return by_region[region]
    
    return all_commodities

def get_markets_for_state(state):
    """
    Get list of markets for a given state.
    
    Args:
        state (str): The state name
        
    Returns:
        list: List of market names
    """
    # Mock market data by state
    markets_by_state = {
        "Maharashtra": ["Pune", "Mumbai", "Nagpur", "Nashik", "Ahmednagar", "Kolhapur"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
        "Punjab": ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Meerut", "Allahabad"],
        "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum"]
    }
    
    return markets_by_state.get(state, [])

def fetch_latest_agmarknet_data(commodity=None, days=7, state=None):
    """
    Fetch the latest data from Agmarknet for a specific commodity.
    
    Args:
        commodity (str, optional): Specific commodity to fetch
        days (int): Number of days of data to fetch
        state (str, optional): State filter
        
    Returns:
        list: List of daily price data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return fetch_agmarknet_data(commodity, start_date, end_date, state)

def crawl_global_price_indices(commodity=None):
    """
    Crawl global price indices for agricultural commodities.
    
    Args:
        commodity (str, optional): Specific commodity to fetch
        
    Returns:
        dict: Global price index data
    """
    logger.info(f"Fetching global price indices for commodity: {commodity}")
    
    cache_key = f"global_indices_{commodity if commodity else 'all'}.json"
    
    # Check cache first
    cached_data = get_cached_data(cache_key)
    if cached_data:
        logger.info(f"Using cached global indices data: {cache_key}")
        return cached_data
    
    # Mock global price indices
    # This would normally involve crawling global commodity exchanges and indices
    
    # Generate example data
    data = {
        "fao_index": {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "overall_index": round(random.uniform(95, 105), 2),
            "cereals_index": round(random.uniform(90, 110), 2),
            "oils_index": round(random.uniform(85, 115), 2)
        },
        "cme_futures": {
            "wheat": round(random.uniform(600, 800), 2),
            "corn": round(random.uniform(500, 700), 2),
            "soybean": round(random.uniform(1200, 1600), 2),
            "rice": round(random.uniform(12, 18), 2)
        },
        "platts_indices": {
            "rice_index": round(random.uniform(350, 450), 2),
            "wheat_index": round(random.uniform(250, 350), 2)
        }
    }
    
    # Cache the data
    save_cached_data(data, cache_key)
    
    return data

def schedule_data_crawl(frequency="daily"):
    """
    Schedule regular data crawl based on specified frequency.
    This would normally involve setting up a cron job or scheduler.
    
    Args:
        frequency (str): Crawl frequency (daily, weekly, etc.)
    """
    logger.info(f"Scheduling data crawl with frequency: {frequency}")
    # This would set up a scheduler in a production environment
    # For now, just log the request
    
    # Return success message
    return {
        "status": "scheduled",
        "frequency": frequency,
        "next_run": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    }

def get_website_text_content(url):
    """
    Extract the main text content from a website.
    
    Args:
        url (str): The website URL
        
    Returns:
        str: Extracted text content
    """
    try:
        # Get the webpage content
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text
        else:
            logger.error(f"Failed to download content from {url}")
            return None
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return None

def fetch_historic_price_data(commodity, region=None, years_back=5):
    """
    Fetch historic price data for trend analysis.
    
    Args:
        commodity (str): The commodity
        region (str, optional): Region filter
        years_back (int): Number of years of historical data
        
    Returns:
        list: Historical price data
    """
    logger.info(f"Fetching historic data for {commodity} in {region}, going back {years_back} years")
    
    cache_key = f"historic_{commodity}_{region}_{years_back}years.json"
    
    # Check cache first
    cached_data = get_cached_data(cache_key)
    if cached_data:
        logger.info(f"Using cached historic data: {cache_key}")
        return cached_data
    
    # Mock historical data generation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')  # Weekly data
    
    # Base price trend with seasonality and long-term growth
    base_price = 3000  # Starting price
    annual_growth = 0.05  # 5% annual growth
    seasonal_amplitude = 0.15  # 15% seasonal variation
    
    historic_data = []
    
    for i, curr_date in enumerate(date_range):
        # Calculate days since start
        days_since_start = (curr_date - start_date).days
        years_since_start = days_since_start / 365
        
        # Apply long-term growth
        growth_factor = (1 + annual_growth) ** years_since_start
        
        # Apply seasonality (using sine wave with 1-year period)
        day_of_year = curr_date.dayofyear
        seasonal_factor = 1 + seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365)
        
        # Add some random noise
        noise = random.uniform(0.95, 1.05)
        
        # Calculate price
        price = base_price * growth_factor * seasonal_factor * noise
        
        # Add data point
        historic_data.append({
            "date": curr_date.strftime('%Y-%m-%d'),
            "commodity": commodity,
            "region": region if region else "All India",
            "price": round(price, 2)
        })
    
    # Cache the data
    save_cached_data(historic_data, cache_key)
    
    return historic_data

if __name__ == "__main__":
    # Test the functions
    print("Testing commodity list fetch...")
    commodities = fetch_commodity_list()
    print(f"Found {len(commodities)} commodities")
    
    print("\nTesting Agmarknet data fetch...")
    data = fetch_latest_agmarknet_data(commodities[0], days=3)
    print(f"Fetched {len(data)} records")
    
    print("\nTesting global indices fetch...")
    indices = crawl_global_price_indices()
    print(f"Fetched global indices: {indices.keys()}")
