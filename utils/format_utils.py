"""
Formatting utilities for the WIZX platform.
"""


def format_price(value, currency_symbol="₹", decimal_places=2, default="N/A"):
    """
    Format a price value with currency symbol.
    
    Args:
        value (float or int): The price value to format
        currency_symbol (str): Currency symbol to prepend
        decimal_places (int): Number of decimal places to show
        default (str): Default value if price is None
        
    Returns:
        str: Formatted price string
    """
    if value is None:
        return default
    
    try:
        return f"{currency_symbol}{float(value):.{decimal_places}f}"
    except (ValueError, TypeError):
        return default


def format_delta(value, decimal_places=1, include_sign=True, default="0"):
    """
    Format a delta value (change).
    
    Args:
        value (float): The delta value to format
        decimal_places (int): Number of decimal places to show
        include_sign (bool): Whether to include + sign for positive values
        default (str): Default value if delta is None
        
    Returns:
        str: Formatted delta string
    """
    if value is None:
        return default
    
    try:
        value = float(value)
        sign = "+" if value > 0 and include_sign else ""
        return f"{sign}{value:.{decimal_places}f}"
    except (ValueError, TypeError):
        return default


def format_percent(value, decimal_places=1, include_sign=True, default="0%"):
    """
    Format a percentage value.
    
    Args:
        value (float): The percentage value to format
        decimal_places (int): Number of decimal places to show
        include_sign (bool): Whether to include + sign for positive values
        default (str): Default value if percentage is None
        
    Returns:
        str: Formatted percentage string
    """
    if value is None:
        return default
    
    try:
        value = float(value)
        sign = "+" if value > 0 and include_sign else ""
        return f"{sign}{value:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return default


def format_large_number(value, format_type="compact", default="N/A"):
    """
    Format large numbers with appropriate suffixes or thousand separators.
    
    Args:
        value (int or float): The number to format
        format_type (str): Format type: 'compact' for K/M/B suffixes, 'comma' for comma separators
        default (str): Default value if number is None
        
    Returns:
        str: Formatted number string
    """
    if value is None:
        return default
    
    try:
        value = float(value)
        
        if format_type == "compact":
            if value >= 1_000_000_000:
                return f"{value/1_000_000_000:.1f}B"
            elif value >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif value >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return f"{value:.0f}"
        elif format_type == "comma":
            return f"{value:,.0f}"
        else:
            return str(value)
    except (ValueError, TypeError):
        return default


def format_date(date_obj, date_format="%d %b %Y", default="N/A"):
    """
    Format a date object to string.
    
    Args:
        date_obj (datetime): The date to format
        date_format (str): Date format string
        default (str): Default value if date is None
        
    Returns:
        str: Formatted date string
    """
    if date_obj is None:
        return default
    
    try:
        return date_obj.strftime(date_format)
    except (ValueError, TypeError, AttributeError):
        return default