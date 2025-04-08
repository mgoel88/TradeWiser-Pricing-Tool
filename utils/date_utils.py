"""
Date utilities for WIZX agricultural platform.
"""

from datetime import datetime, timedelta


def format_date(date_obj, format_string="%Y-%m-%d"):
    """
    Format a date object to a string.
    
    Args:
        date_obj: The date object to format
        format_string: The format string to use
        
    Returns:
        str: Formatted date string
    """
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d")
        except ValueError:
            try:
                date_obj = datetime.strptime(date_obj, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                return date_obj
    
    if isinstance(date_obj, datetime):
        return date_obj.strftime(format_string)
    
    return str(date_obj)


def validate_date_range(start_date, end_date):
    """
    Validate a date range and convert to datetime objects.
    
    Args:
        start_date: The start date (string or datetime)
        end_date: The end date (string or datetime)
        
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    if isinstance(start_date, str):
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                start_date = datetime.now() - timedelta(days=30)
    
    if isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                end_date = datetime.now()
    
    # Default dates if invalid
    if not isinstance(start_date, datetime):
        start_date = datetime.now() - timedelta(days=30)
    
    if not isinstance(end_date, datetime):
        end_date = datetime.now()
    
    # Ensure start date is before end date
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    
    return start_date, end_date


def get_date_range_from_days(days, end_date=None):
    """
    Get a date range from a number of days.
    
    Args:
        days: Number of days in the range
        end_date: Optional end date (defaults to today)
        
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    if end_date is None:
        end_date = datetime.now()
    
    if isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                end_date = datetime.now()
    
    start_date = end_date - timedelta(days=days)
    
    return start_date, end_date