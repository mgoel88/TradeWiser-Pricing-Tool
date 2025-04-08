"""
WIZX Index module for agricultural commodity price index calculation.

The WIZX Index is a composite measure of agricultural commodity prices 
that provides a consistent basis for tracking market trends.
"""

import os
import json
import logging
from datetime import datetime, date, timedelta

# Database imports
from database_sql import calculate_wizx_index, get_commodity_index_data, get_all_commodities

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_composite_index(days=30):
    """
    Calculate the composite index value for all commodities.
    
    Args:
        days (int): Number of days of historical data to include
        
    Returns:
        dict: Composite index information
    """
    try:
        # Get the composite index data
        index_data = get_wizx_index()
        
        if not index_data:
            logger.warning("Failed to retrieve composite index data")
            return {"success": False, "error": "No data available"}
        
        current_value = index_data.get("current_value", 1000)
        change_percentage = index_data.get("change_percentage", 0)
        
        # Get previous value
        history = index_data.get("history", [])
        previous_value = current_value
        
        if len(history) > 1:
            previous_value = history[-2].get("value", current_value)
        
        return {
            "success": True,
            "value": current_value,
            "previous_value": previous_value,
            "change_percentage": change_percentage
        }
        
    except Exception as e:
        logger.error(f"Error calculating composite index: {e}")
        return {"success": False, "error": str(e)}


def calculate_all_indices():
    """
    Calculate indices for all commodities.
    
    Returns:
        dict: Dictionary of commodity indices
    """
    try:
        # Get all commodities
        commodities = get_all_commodities()
        results = {}
        
        # Calculate index for each commodity
        for commodity in commodities:
            index_data = get_wizx_index(commodity)
            
            if index_data:
                results[commodity] = {
                    "current_value": index_data.get("current_value", 1000),
                    "change_percentage": index_data.get("change_percentage", 0)
                }
        
        return results
    
    except Exception as e:
        logger.error(f"Error calculating all indices: {e}")
        return {}


def get_wizx_index(commodity=None):
    """
    Get the WIZX index data for a specific commodity or the composite index.
    
    Args:
        commodity (str, optional): The commodity name. If None, returns the composite index.
        
    Returns:
        dict: WIZX index data
    """
    try:
        # If a specific commodity is requested
        if commodity:
            return get_commodity_index_data(commodity)
        
        # Otherwise, construct a composite index
        # Get data for all commodities
        commodities = get_all_commodities()
        all_indices = []
        
        for comm in commodities:
            index_data = get_commodity_index_data(comm)
            if index_data:
                all_indices.append(index_data)
        
        # If we have no data, return a default structure
        if not all_indices:
            return {
                "commodity": "Composite",
                "current_value": 1000,
                "change_percentage": 0,
                "history": [
                    {"date": date.today(), "value": 1000}
                ]
            }
        
        # Calculate composite index
        # Start with an empty history dict
        history_by_date = {}
        
        # For each commodity index
        for idx in all_indices:
            # Process each history point
            for point in idx.get("history", []):
                date_val = point.get("date")
                value = point.get("value")
                
                if date_val and value:
                    # Initialize if first time seeing this date
                    if date_val not in history_by_date:
                        history_by_date[date_val] = {
                            "sum": 0,
                            "count": 0
                        }
                    
                    # Add this value to the sum for this date
                    history_by_date[date_val]["sum"] += value
                    history_by_date[date_val]["count"] += 1
        
        # Convert to list and calculate averages
        composite_history = []
        for date_val, data in sorted(history_by_date.items()):
            if data["count"] > 0:
                avg_value = data["sum"] / data["count"]
                composite_history.append({
                    "date": date_val,
                    "value": avg_value
                })
        
        # Calculate current value and change percentage
        if composite_history:
            current_value = composite_history[-1]["value"]
            
            # Calculate change percentage
            if len(composite_history) > 1:
                prev_value = composite_history[-2]["value"]
                change_percentage = ((current_value - prev_value) / prev_value) * 100
            else:
                change_percentage = 0
        else:
            current_value = 1000
            change_percentage = 0
        
        # Return the composite index
        return {
            "commodity": "Composite",
            "current_value": current_value,
            "change_percentage": change_percentage,
            "history": composite_history
        }
        
    except Exception as e:
        logger.error(f"Error retrieving WIZX index: {e}")
        
        # Return a default structure in case of error
        return {
            "commodity": commodity or "Composite",
            "current_value": 1000,
            "change_percentage": 0,
            "history": [
                {"date": date.today(), "value": 1000}
            ]
        }


def update_wizx_indices(date_val=None):
    """
    Update the WIZX indices for all commodities.
    
    Args:
        date_val (date, optional): The date to calculate indices for. Defaults to today.
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Default to today if no date provided
        if date_val is None:
            date_val = date.today()
        
        logger.info(f"Updating WIZX indices for {date_val}")
        
        # Get all commodities
        commodities = get_all_commodities()
        
        # Calculate index for each commodity
        success_count = 0
        
        for commodity in commodities:
            result = calculate_wizx_index(commodity, date_val)
            
            if result:
                success_count += 1
                logger.info(f"Updated WIZX index for {commodity}: {result['index_value']:.2f}")
            else:
                logger.warning(f"Failed to update WIZX index for {commodity}")
        
        logger.info(f"WIZX index update complete. Updated {success_count} out of {len(commodities)} commodities.")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error updating WIZX indices: {e}")
        return False


def backfill_wizx_indices(start_date=None, end_date=None):
    """
    Backfill WIZX indices for a date range.
    
    Args:
        start_date (date, optional): Start date for backfill. Defaults to 30 days ago.
        end_date (date, optional): End date for backfill. Defaults to yesterday.
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Default dates if not provided
        if end_date is None:
            end_date = date.today() - timedelta(days=1)
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        logger.info(f"Backfilling WIZX indices from {start_date} to {end_date}")
        
        # Get all commodities
        commodities = get_all_commodities()
        
        # Calculate indices for each date and commodity
        current_date = start_date
        total_updates = 0
        
        while current_date <= end_date:
            day_updates = 0
            
            for commodity in commodities:
                result = calculate_wizx_index(commodity, current_date)
                
                if result:
                    day_updates += 1
            
            logger.info(f"Backfilled {day_updates} indices for {current_date}")
            total_updates += day_updates
            
            # Move to next day
            current_date += timedelta(days=1)
        
        logger.info(f"WIZX index backfill complete. Updated {total_updates} indices.")
        
        return total_updates > 0
        
    except Exception as e:
        logger.error(f"Error backfilling WIZX indices: {e}")
        return False


def get_wizx_index_statistics(commodity=None, days=30):
    """
    Get statistics for the WIZX index.
    
    Args:
        commodity (str, optional): The commodity to get statistics for. If None, returns composite index stats.
        days (int): Number of days of historical data to analyze
        
    Returns:
        dict: Index statistics
    """
    try:
        # Get index data
        index_data = get_wizx_index(commodity)
        
        if not index_data:
            return None
        
        # Filter history to requested number of days
        history = index_data.get("history", [])
        
        if len(history) == 0:
            return {
                "commodity": commodity or "Composite",
                "current_value": index_data.get("current_value", 1000),
                "change_percentage": index_data.get("change_percentage", 0),
                "min_value": index_data.get("current_value", 1000),
                "max_value": index_data.get("current_value", 1000),
                "avg_value": index_data.get("current_value", 1000),
                "volatility": 0
            }
        
        # Filter to last 'days' days
        if len(history) > days:
            history = history[-days:]
        
        # Extract values
        values = [point.get("value", 0) for point in history]
        
        # Calculate statistics
        min_value = min(values)
        max_value = max(values)
        avg_value = sum(values) / len(values)
        
        # Calculate volatility (standard deviation of daily percentage changes)
        volatility = 0
        if len(values) > 1:
            daily_changes = []
            for i in range(1, len(values)):
                if values[i-1] > 0:
                    daily_change = (values[i] - values[i-1]) / values[i-1] * 100
                    daily_changes.append(daily_change)
            
            if daily_changes:
                # Standard deviation calculation
                mean = sum(daily_changes) / len(daily_changes)
                variance = sum((x - mean) ** 2 for x in daily_changes) / len(daily_changes)
                volatility = variance ** 0.5
        
        # Return statistics
        return {
            "commodity": commodity or "Composite",
            "current_value": index_data.get("current_value", 1000),
            "change_percentage": index_data.get("change_percentage", 0),
            "min_value": min_value,
            "max_value": max_value,
            "avg_value": avg_value,
            "volatility": volatility  # Daily percentage volatility
        }
        
    except Exception as e:
        logger.error(f"Error getting WIZX index statistics: {e}")
        return None


if __name__ == "__main__":
    # Use this for testing or standalone operation
    # Example: python wizx_index.py
    try:
        # Update today's indices
        update_wizx_indices()
        
        # Print composite index
        index_data = get_wizx_index()
        if index_data:
            print(f"Composite WIZX Index: {index_data['current_value']:.2f} ({index_data['change_percentage']:+.2f}%)")
            
            # Print the last 5 days
            print("\nLast 5 days:")
            for point in index_data.get("history", [])[-5:]:
                print(f"{point['date']}: {point['value']:.2f}")
        
    except Exception as e:
        logger.error(f"Error running WIZX index module: {e}")