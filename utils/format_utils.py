"""
Formatting utilities for WIZX agricultural platform.
"""


def format_price(value, currency="₹", decimal_places=2):
    """
    Format a price value with currency symbol.
    
    Args:
        value: The price value to format
        currency: The currency symbol to use
        decimal_places: Number of decimal places
        
    Returns:
        str: Formatted price string
    """
    try:
        value = float(value)
        return f"{currency}{value:.{decimal_places}f}"
    except (ValueError, TypeError):
        return f"{currency}0.00"


def format_delta(value, prefix=True, decimal_places=2):
    """
    Format a delta value with sign and optionally with prefix.
    
    Args:
        value: The delta value to format
        prefix: Whether to include a '+' prefix for positive values
        decimal_places: Number of decimal places
        
    Returns:
        str: Formatted delta string
    """
    try:
        value = float(value)
        sign = "+" if value > 0 and prefix else ""
        return f"{sign}{value:.{decimal_places}f}"
    except (ValueError, TypeError):
        return "0.00"


def format_percent(value, decimal_places=1, with_symbol=True):
    """
    Format a value as a percentage.
    
    Args:
        value: The value to format (0.1 = 10%)
        decimal_places: Number of decimal places
        with_symbol: Whether to include the % symbol
        
    Returns:
        str: Formatted percentage string
    """
    try:
        value = float(value)
        if -1 <= value <= 1:  # Assuming decimal value
            value = value * 100
        
        formatted = f"{value:.{decimal_places}f}"
        if with_symbol:
            formatted += "%"
        
        return formatted
    except (ValueError, TypeError):
        return "0.0%" if with_symbol else "0.0"