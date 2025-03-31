"""
WIZX Index module for calculating and analyzing commodity price indices.

The WIZX Index is a standardized commodity price index similar to price
indices like SENSEX, NYMEX, etc. It provides a reliable benchmark for
tracking agricultural commodity prices across different regions and qualities.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

from database_sql import (
    Session, Commodity, Region, PricePoint, WIZXIndex,
    get_commodity_data, calculate_wizx_index, get_wizx_indices
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
WIZX_DATA_DIR = os.path.join(DATA_DIR, "wizx_indices")

# Ensure directories exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(WIZX_DATA_DIR):
    os.makedirs(WIZX_DATA_DIR)


def calculate_all_indices(date_val=None):
    """
    Calculate WIZX indices for all commodities.
    
    Args:
        date_val (date, optional): The date for calculation
        
    Returns:
        dict: Results of index calculations
    """
    if date_val is None:
        date_val = date.today()
    
    session = Session()
    
    try:
        # Get all commodities
        commodities = session.query(Commodity).all()
        
        if not commodities:
            logger.warning("No commodities found in database")
            return {"success": False, "message": "No commodities found"}
        
        results = {
            "date": date_val,
            "indices": {},
            "success_count": 0,
            "failed_count": 0
        }
        
        for commodity in commodities:
            try:
                # Calculate index
                index_result = calculate_wizx_index(commodity.name, date_val)
                
                if index_result:
                    results["indices"][commodity.name] = index_result
                    results["success_count"] += 1
                else:
                    logger.warning(f"Failed to calculate WIZX index for {commodity.name}")
                    results["failed_count"] += 1
                    results["indices"][commodity.name] = {"error": "Insufficient data"}
            except Exception as e:
                logger.error(f"Error calculating WIZX index for {commodity.name}: {e}")
                results["failed_count"] += 1
                results["indices"][commodity.name] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating all indices: {e}")
        return {"success": False, "message": str(e)}
    finally:
        session.close()


def calculate_composite_index(commodity_weights=None, date_val=None, name="WIZX-Composite"):
    """
    Calculate a composite WIZX index across multiple commodities.
    
    Args:
        commodity_weights (dict, optional): Commodity weights {commodity_name: weight}
        date_val (date, optional): The date for calculation
        name (str): Name of the composite index
        
    Returns:
        dict: Composite index data
    """
    if date_val is None:
        date_val = date.today()
    
    session = Session()
    
    try:
        # Get all indices for the date
        indices = session.query(WIZXIndex).\
            join(Commodity).\
            filter(WIZXIndex.date == date_val).\
            all()
        
        if not indices:
            logger.warning(f"No indices found for date {date_val}")
            return {"success": False, "message": f"No indices found for date {date_val}"}
        
        # If weights not provided, use equal weighting
        if not commodity_weights:
            commodity_weights = {}
            for idx in indices:
                commodity_name = session.query(Commodity.name).filter(Commodity.id == idx.commodity_id).scalar()
                commodity_weights[commodity_name] = 1.0 / len(indices)
        
        # Validate weights
        valid_commodities = [
            session.query(Commodity.name).filter(Commodity.id == idx.commodity_id).scalar()
            for idx in indices
        ]
        
        for commodity in list(commodity_weights.keys()):
            if commodity not in valid_commodities:
                del commodity_weights[commodity]
                logger.warning(f"Removed invalid commodity from weights: {commodity}")
        
        if not commodity_weights:
            logger.warning("No valid commodity weights")
            return {"success": False, "message": "No valid commodity weights"}
        
        # Normalize weights
        total_weight = sum(commodity_weights.values())
        normalized_weights = {k: v / total_weight for k, v in commodity_weights.items()}
        
        # Calculate weighted sum
        composite_value = 0
        components = {}
        
        for idx in indices:
            commodity_name = session.query(Commodity.name).filter(Commodity.id == idx.commodity_id).scalar()
            
            if commodity_name in normalized_weights:
                weight = normalized_weights[commodity_name]
                composite_value += idx.index_value * weight
                
                components[commodity_name] = {
                    "index_value": idx.index_value,
                    "weight": weight,
                    "weighted_value": idx.index_value * weight
                }
        
        # Get previous composite value
        previous_date = date_val - timedelta(days=1)
        previous_composite = session.query(pd.DataFrame).filter(
            pd.DataFrame.name == name,
            pd.DataFrame.date == previous_date
        ).first()
        
        previous_value = 1000.0
        if previous_composite:
            previous_value = previous_composite.value
        
        # Calculate change
        change = composite_value - previous_value
        change_percentage = (change / previous_value) * 100 if previous_value else 0
        
        # Save to file
        file_path = os.path.join(WIZX_DATA_DIR, f"{name}_{date_val}.json")
        
        pd.DataFrame({
            "name": name,
            "date": date_val,
            "value": composite_value,
            "previous_value": previous_value,
            "change": change,
            "change_percentage": change_percentage,
            "components": components,
            "weights": normalized_weights
        }).to_json(file_path, orient="records", lines=True)
        
        return {
            "success": True,
            "name": name,
            "date": date_val,
            "value": composite_value,
            "previous_value": previous_value,
            "change": change,
            "change_percentage": change_percentage,
            "components": components
        }
        
    except Exception as e:
        logger.error(f"Error calculating composite index: {e}")
        return {"success": False, "message": str(e)}
    finally:
        session.close()


def get_sector_indices(sector_definitions=None, date_val=None):
    """
    Calculate sector-specific WIZX indices.
    
    Args:
        sector_definitions (dict, optional): Sector definitions {sector_name: [commodity_list]}
        date_val (date, optional): The date for calculation
        
    Returns:
        dict: Sector indices
    """
    if date_val is None:
        date_val = date.today()
    
    # Default sectors if not provided
    if not sector_definitions:
        sector_definitions = {
            "WIZX-Cereals": ["Wheat", "Rice", "Maize"],
            "WIZX-Pulses": ["Tur Dal", "Moong Dal", "Urad Dal", "Chana Dal"],
            "WIZX-Oilseeds": ["Soyabean", "Mustard", "Groundnut", "Sunflower"]
        }
    
    results = {}
    
    for sector_name, commodities in sector_definitions.items():
        # Create weight dictionary with equal weights
        weights = {commodity: 1.0 / len(commodities) for commodity in commodities}
        
        # Calculate composite index for this sector
        sector_index = calculate_composite_index(
            commodity_weights=weights,
            date_val=date_val,
            name=sector_name
        )
        
        results[sector_name] = sector_index
    
    return {
        "date": date_val,
        "sector_indices": results
    }


def historical_index_performance(commodity, days=365):
    """
    Get historical performance of a WIZX index.
    
    Args:
        commodity (str): Commodity name or 'Composite' for composite index
        days (int): Number of days of history
        
    Returns:
        dict: Historical performance data
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    # Get indices data
    if commodity == "Composite" or commodity.startswith("WIZX-"):
        # Load composite index data from files
        index_files = [f for f in os.listdir(WIZX_DATA_DIR) if f.startswith(commodity) and f.endswith(".json")]
        
        indices = []
        for file_name in index_files:
            file_path = os.path.join(WIZX_DATA_DIR, file_name)
            try:
                data = pd.read_json(file_path, lines=True)
                if data:
                    indices.append(data)
            except Exception as e:
                logger.error(f"Error reading index file {file_name}: {e}")
        
        if not indices:
            return {"success": False, "message": f"No historical data found for {commodity}"}
        
        # Convert to DataFrame
        df = pd.concat(indices)
        
        # Filter by date range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        if df.empty:
            return {"success": False, "message": f"No data found for {commodity} in date range"}
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate periodic returns
        periodic_returns = {
            "last_day": df['change_percentage'].iloc[-1] if len(df) > 1 else None,
            "last_week": calculate_return(df, 7),
            "last_month": calculate_return(df, 30),
            "last_quarter": calculate_return(df, 90),
            "last_year": calculate_return(df, 365)
        }
        
        # Calculate volatility
        if len(df) >= 30:
            volatility = df['change_percentage'].std()
        else:
            volatility = None
        
        return {
            "success": True,
            "commodity": commodity,
            "start_date": start_date,
            "end_date": end_date,
            "current_value": df['value'].iloc[-1],
            "periodic_returns": periodic_returns,
            "volatility": volatility,
            "min_value": df['value'].min(),
            "max_value": df['value'].max(),
            "values": df[['date', 'value', 'change_percentage']].to_dict('records')
        }
        
    else:
        # Get commodity WIZX indices
        indices = get_wizx_indices(commodity, start_date, end_date)
        
        if not indices or commodity not in indices:
            return {"success": False, "message": f"No historical data found for {commodity}"}
        
        # Convert to DataFrame
        df = pd.DataFrame(indices[commodity])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate periodic returns
        periodic_returns = {
            "last_day": df['change_percentage'].iloc[-1] if len(df) > 1 else None,
            "last_week": calculate_return(df, 7),
            "last_month": calculate_return(df, 30),
            "last_quarter": calculate_return(df, 90),
            "last_year": calculate_return(df, 365)
        }
        
        # Calculate volatility
        if len(df) >= 30:
            volatility = df['change_percentage'].std()
        else:
            volatility = None
        
        return {
            "success": True,
            "commodity": commodity,
            "start_date": start_date,
            "end_date": end_date,
            "current_value": df['index_value'].iloc[-1],
            "periodic_returns": periodic_returns,
            "volatility": volatility,
            "min_value": df['index_value'].min(),
            "max_value": df['index_value'].max(),
            "values": df.to_dict('records')
        }


def calculate_return(df, days):
    """
    Calculate return over a period.
    
    Args:
        df (DataFrame): DataFrame with price data
        days (int): Number of days
        
    Returns:
        float: Return percentage
    """
    if len(df) <= 1:
        return None
    
    last_value = df['value'].iloc[-1]
    
    # Find closest value 'days' days ago
    if len(df) >= days:
        previous_value = df['value'].iloc[-days]
    else:
        previous_value = df['value'].iloc[0]
    
    if previous_value == 0:
        return None
    
    return (last_value - previous_value) / previous_value * 100


def compare_indices(commodities, days=30):
    """
    Compare multiple WIZX indices.
    
    Args:
        commodities (list): List of commodity names to compare
        days (int): Number of days to compare
        
    Returns:
        dict: Comparison data
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    results = {
        "start_date": start_date,
        "end_date": end_date,
        "commodities": {},
        "correlation_matrix": None
    }
    
    # Get data for each commodity
    all_data = {}
    
    for commodity in commodities:
        performance = historical_index_performance(commodity, days)
        
        if performance["success"]:
            results["commodities"][commodity] = {
                "current_value": performance["current_value"],
                "periodic_returns": performance["periodic_returns"],
                "volatility": performance["volatility"]
            }
            
            # Extract time series for correlation calculation
            if 'values' in performance:
                if commodity.startswith("WIZX-") or commodity == "Composite":
                    df = pd.DataFrame(performance["values"])
                    all_data[commodity] = df.set_index('date')['value']
                else:
                    df = pd.DataFrame(performance["values"])
                    all_data[commodity] = df.set_index('date')['index_value']
        else:
            results["commodities"][commodity] = {"error": performance["message"]}
    
    # Calculate correlation matrix
    if len(all_data) >= 2:
        df = pd.DataFrame(all_data)
        results["correlation_matrix"] = df.corr().to_dict()
    
    return results


def export_indices(start_date=None, end_date=None, commodities=None, file_path=None):
    """
    Export WIZX indices to a CSV file.
    
    Args:
        start_date (date, optional): Start date
        end_date (date, optional): End date
        commodities (list, optional): List of commodities to export
        file_path (str, optional): Output file path
        
    Returns:
        dict: Export results
    """
    if end_date is None:
        end_date = date.today()
    
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    # Get data
    if commodities:
        # Get specific commodities
        all_indices = {}
        for commodity in commodities:
            if commodity.startswith("WIZX-") or commodity == "Composite":
                performance = historical_index_performance(commodity, (end_date - start_date).days)
                if performance["success"]:
                    all_indices[commodity] = performance["values"]
            else:
                indices = get_wizx_indices(commodity, start_date, end_date)
                if indices and commodity in indices:
                    all_indices[commodity] = indices[commodity]
    else:
        # Get all commodities
        indices = get_wizx_indices(None, start_date, end_date)
        all_indices = indices
        
        # Add composite indices
        for index_name in ["WIZX-Composite", "WIZX-Cereals", "WIZX-Pulses", "WIZX-Oilseeds"]:
            performance = historical_index_performance(index_name, (end_date - start_date).days)
            if performance["success"]:
                all_indices[index_name] = performance["values"]
    
    if not all_indices:
        return {"success": False, "message": "No indices found for the specified parameters"}
    
    # Convert to DataFrames and merge
    merged_data = None
    
    for commodity, values in all_indices.items():
        df = pd.DataFrame(values)
        
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Rename columns to include commodity
            value_column = 'value' if 'value' in df.columns else 'index_value'
            change_column = 'change_percentage' if 'change_percentage' in df.columns else None
            
            if value_column in df.columns:
                df.rename(columns={value_column: f"{commodity}_{value_column}"}, inplace=True)
            
            if change_column and change_column in df.columns:
                df.rename(columns={change_column: f"{commodity}_{change_column}"}, inplace=True)
            
            # Set date as index
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
            
            # Merge with existing data
            if merged_data is None:
                merged_data = df
            else:
                merged_data = pd.merge(merged_data, df, left_index=True, right_index=True, how='outer')
    
    if merged_data is None or merged_data.empty:
        return {"success": False, "message": "No valid data to export"}
    
    # Reset index to get date as column
    merged_data.reset_index(inplace=True)
    
    # Sort by date
    merged_data.sort_values('date', inplace=True)
    
    # Generate file path if not provided
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(WIZX_DATA_DIR, f"wizx_indices_{timestamp}.csv")
    
    # Export to CSV
    merged_data.to_csv(file_path, index=False)
    
    return {
        "success": True,
        "file_path": file_path,
        "commodities": list(all_indices.keys()),
        "rows": len(merged_data),
        "columns": len(merged_data.columns),
        "date_range": [start_date, end_date]
    }