"""
Pages for WIZX agricultural platform.
"""

from .dashboard import render as render_dashboard
from .price_calculator import render as render_price_calculator
from .quality_analysis import render as render_quality_analysis

__all__ = [
    'render_dashboard',
    'render_price_calculator',
    'render_quality_analysis'
]