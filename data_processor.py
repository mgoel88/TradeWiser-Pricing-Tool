"""
Data processor module for ETL operations on agricultural commodity pricing data.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_crawler import fetch_agmarknet_data, fetch_enam_data, fetch_commodity_list

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_data(data, source="agmarknet"):
    """
    Process and clean raw data from various sources.
    
    Args:
        data (list): Raw data from source
        source (str): Data source name
        
    Returns:
        pd.DataFrame: Processed and cleaned data
    """
    logger.info(f"Processing data from {source} with {len(data)} records")
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(data)
    
    # Basic data cleaning
    if source == "agmarknet":
        # Handle dates
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate average price if needed
        if 'modal_price' in df.columns:
            df['price'] = df['modal_price']
        else:
            df['price'] = (df['min_price'] + df['max_price']) / 2
        
        # Drop rows with missing prices
        df = df.dropna(subset=['price'])
        
        # Remove outliers (prices that are too high or too low)
        for commodity in df['commodity'].unique():
            commodity_df = df[df['commodity'] == commodity]
            q1 = commodity_df['price'].quantile(0.25)
            q3 = commodity_df['price'].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds for outliers
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            # Filter out outliers
            df = df[~((df['commodity'] == commodity) & 
                      ((df['price'] < lower_bound) | (df['price'] > upper_bound)))]
    
    elif source == "enam":
        # Handle dates
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate average price if needed
        if 'avg_price' in df.columns:
            df['price'] = df['avg_price']
        else:
            df['price'] = (df['min_price'] + df['max_price']) / 2
        
        # Drop rows with missing prices
        df = df.dropna(subset=['price'])
    
    logger.info(f"Processed data: {len(df)} records after cleaning")
    
    return df

def standardize_commodity(commodity_data, reference_grade=None):
    """
    Standardize commodity data to a reference grade.
    
    Args:
        commodity_data (dict): Raw commodity data
        reference_grade (dict, optional): Reference grade specifications
        
    Returns:
        dict: Standardized commodity data
    """
    logger.info(f"Standardizing commodity data for {commodity_data.get('commodity', 'unknown')}")
    
    # If reference grade not provided, use defaults
    if not reference_grade:
        # Get default reference grades
        reference_grades = {
            "Wheat": {"moisture": 12.0, "foreign_matter": 1.0, "damaged_grains": 2.0},
            "Rice": {"moisture": 14.0, "foreign_matter": 0.5, "broken_grains": 5.0},
            "Tur Dal": {"moisture": 10.0, "foreign_matter": 1.0, "damaged_grains": 3.0},
            "Soyabean": {"moisture": 12.0, "foreign_matter": 1.0, "oil_content": 18.0},
            "Mustard": {"moisture": 8.0, "foreign_matter": 2.0, "oil_content": 40.0}
        }
        
        commodity = commodity_data.get('commodity')
        reference_grade = reference_grades.get(commodity, {})
    
    # Create a copy of the data for standardization
    standardized_data = commodity_data.copy()
    
    # Apply quality adjustments based on reference grade
    # This would involve more complex logic in a real implementation
    # Here we're just adding placeholder functionality
    
    # Add standardization info
    standardized_data['reference_grade'] = reference_grade
    standardized_data['standardized'] = True
    
    return standardized_data

def aggregate_daily_prices(data, level="state"):
    """
    Aggregate daily prices to different geographical levels.
    
    Args:
        data (pd.DataFrame): Raw price data
        level (str): Aggregation level (market, district, state, national)
        
    Returns:
        pd.DataFrame: Aggregated price data
    """
    logger.info(f"Aggregating daily prices to {level} level")
    
    if level == "market":
        # Already at market level, just group by date, commodity, and market
        grouped = data.groupby(['date', 'commodity', 'market'])
    elif level == "state":
        # Group by date, commodity, and state
        grouped = data.groupby(['date', 'commodity', 'state'])
    elif level == "national":
        # Group by date and commodity only
        grouped = data.groupby(['date', 'commodity'])
    else:
        logger.warning(f"Unknown aggregation level: {level}")
        return data
    
    # Calculate aggregated metrics
    aggregated = grouped.agg({
        'price': ['mean', 'min', 'max', 'std', 'count'],
        'arrival_qty': ['sum', 'mean'] if 'arrival_qty' in data.columns else []
    }).reset_index()
    
    # Flatten multi-level columns
    aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
    
    return aggregated

def calculate_price_indices(data, base_year=None, base_period=None):
    """
    Calculate price indices relative to a base period.
    
    Args:
        data (pd.DataFrame): Historical price data
        base_year (int, optional): Base year for index
        base_period (tuple, optional): Base period (start_date, end_date)
        
    Returns:
        pd.DataFrame: Price indices
    """
    logger.info(f"Calculating price indices with base year: {base_year}")
    
    # Convert to DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Ensure date column is datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Extract year
    data['year'] = data['date'].dt.year
    
    # If base year not provided, use the earliest year in the data
    if not base_year:
        base_year = data['year'].min()
    
    # If base period not provided, use the entire base year
    if not base_period:
        base_period = (
            datetime(base_year, 1, 1),
            datetime(base_year, 12, 31)
        )
    
    # Calculate base prices
    base_prices = {}
    
    for commodity in data['commodity'].unique():
        # Filter data for base period and commodity
        base_data = data[
            (data['commodity'] == commodity) &
            (data['date'] >= base_period[0]) &
            (data['date'] <= base_period[1])
        ]
        
        if not base_data.empty:
            # Calculate average price for base period
            base_prices[commodity] = base_data['price'].mean()
    
    # Calculate indices
    indices = []
    
    for commodity in data['commodity'].unique():
        if commodity not in base_prices:
            continue
        
        # Get base price
        base_price = base_prices[commodity]
        
        # Calculate monthly averages
        monthly_data = data[data['commodity'] == commodity].copy()
        monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
        monthly_avg = monthly_data.groupby('year_month')['price'].mean().reset_index()
        
        # Calculate index
        for _, row in monthly_avg.iterrows():
            index_value = (row['price'] / base_price) * 100
            
            indices.append({
                'commodity': commodity,
                'year_month': row['year_month'],
                'index_value': index_value,
                'base_year': base_year
            })
    
    # Convert to DataFrame
    indices_df = pd.DataFrame(indices)
    
    return indices_df

def merge_data_sources(agmarknet_data, enam_data):
    """
    Merge data from multiple sources.
    
    Args:
        agmarknet_data (pd.DataFrame): Data from Agmarknet
        enam_data (pd.DataFrame): Data from eNAM
        
    Returns:
        pd.DataFrame: Merged data
    """
    logger.info("Merging data from multiple sources")
    
    # Process data from each source
    agmarknet_df = process_data(agmarknet_data, source="agmarknet")
    enam_df = process_data(enam_data, source="enam")
    
    # Ensure common structure for merging
    common_columns = ['date', 'commodity', 'market', 'price']
    
    agmarknet_df = agmarknet_df[common_columns].copy()
    enam_df = enam_df[common_columns].copy()
    
    # Add source identifier
    agmarknet_df['source'] = 'agmarknet'
    enam_df['source'] = 'enam'
    
    # Concatenate the data
    merged_df = pd.concat([agmarknet_df, enam_df], ignore_index=True)
    
    # Handle duplicates (same date, commodity, market)
    # Take average if there are duplicates
    merged_df = merged_df.groupby(['date', 'commodity', 'market']).agg({
        'price': 'mean',
        'source': lambda x: '+'.join(x.unique())
    }).reset_index()
    
    return merged_df

def export_processed_data(data, filename):
    """
    Export processed data to a file.
    
    Args:
        data (pd.DataFrame): Processed data
        filename (str): Output filename
        
    Returns:
        str: Path to the exported file
    """
    filepath = os.path.join(PROCESSED_DIR, filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if isinstance(data, pd.DataFrame):
        # Export as CSV
        data.to_csv(filepath, index=False)
    else:
        # Export as JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    logger.info(f"Exported processed data to {filepath}")
    
    return filepath

def run_etl_pipeline(commodity=None, start_date=None, end_date=None, state=None):
    """
    Run the complete ETL pipeline.
    
    Args:
        commodity (str, optional): Specific commodity to process
        start_date (datetime, optional): Start date for data
        end_date (datetime, optional): End date for data
        state (str, optional): State filter
        
    Returns:
        dict: Processing results and statistics
    """
    logger.info(f"Running ETL pipeline for {commodity}, {state}")
    
    # Step 1: Fetch data from sources
    agmarknet_data = fetch_agmarknet_data(commodity, start_date, end_date, state)
    enam_data = fetch_enam_data(commodity)
    
    # Step 2: Process and clean the data
    processed_agmarknet = process_data(agmarknet_data, source="agmarknet")
    processed_enam = process_data(enam_data, source="enam")
    
    # Step 3: Merge data from different sources
    merged_data = merge_data_sources(processed_agmarknet, processed_enam)
    
    # Step 4: Aggregate data to different levels
    state_level_data = aggregate_daily_prices(merged_data, level="state")
    national_level_data = aggregate_daily_prices(merged_data, level="national")
    
    # Step 5: Calculate price indices
    indices = calculate_price_indices(merged_data)
    
    # Step 6: Export processed data
    timestamp = datetime.now().strftime('%Y%m%d')
    
    export_processed_data(merged_data, f"merged_data_{timestamp}.csv")
    export_processed_data(state_level_data, f"state_level_data_{timestamp}.csv")
    export_processed_data(national_level_data, f"national_level_data_{timestamp}.csv")
    export_processed_data(indices, f"price_indices_{timestamp}.csv")
    
    # Return processing statistics
    return {
        "status": "success",
        "records_processed": len(merged_data),
        "commodities_processed": len(merged_data['commodity'].unique()),
        "markets_processed": len(merged_data['market'].unique()),
        "date_range": [merged_data['date'].min().strftime('%Y-%m-%d'), 
                       merged_data['date'].max().strftime('%Y-%m-%d')],
        "output_files": [
            f"merged_data_{timestamp}.csv",
            f"state_level_data_{timestamp}.csv",
            f"national_level_data_{timestamp}.csv",
            f"price_indices_{timestamp}.csv"
        ]
    }

if __name__ == "__main__":
    # Test the ETL pipeline
    print("Running ETL pipeline test...")
    results = run_etl_pipeline()
    print(f"ETL Results: {results}")
