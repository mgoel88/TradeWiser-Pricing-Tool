"""
Data models for the WIZX SDK.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class CommodityInfo:
    """Information about a commodity."""
    name: str
    description: Optional[str] = None
    trading_units: Optional[str] = None
    quality_parameters: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.quality_parameters is None:
            self.quality_parameters = {}


@dataclass
class PriceRequest:
    """Request parameters for calculating a commodity price."""
    commodity: str
    quality_params: Dict[str, float]
    region: str


@dataclass
class PriceResponse:
    """Response containing price information for a commodity."""
    commodity: str
    region: str
    base_price: float
    final_price: float
    quality_delta: float
    location_delta: float
    market_delta: float
    currency: str = "INR"
    timestamp: datetime = None
    unit: str = "kg"
    confidence: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def total_delta(self) -> float:
        """Calculate the total delta applied to the base price."""
        return self.quality_delta + self.location_delta + self.market_delta


@dataclass
class QualityAnalysisResponse:
    """Response containing quality analysis results for a commodity."""
    commodity: str
    quality_params: Dict[str, float]
    quality_score: float
    quality_grade: str
    confidence: Optional[float] = None
    analysis_summary: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PriceHistoryEntry:
    """A single entry in price history data."""
    date: datetime
    price: float
    volume: Optional[float] = None


@dataclass
class PriceHistory:
    """Historical price data for a commodity and region."""
    commodity: str
    region: str
    currency: str = "INR"
    unit: str = "kg"
    data: List[PriceHistoryEntry] = None

    def __post_init__(self):
        if self.data is None:
            self.data = []


@dataclass
class IndexValue:
    """A single index value entry."""
    date: datetime
    value: float


@dataclass
class CommodityIndex:
    """Index data for a commodity."""
    commodity: str
    current_value: float
    change_percentage: float
    history: List[IndexValue] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


@dataclass
class PriceForecast:
    """Price forecast data for a commodity and region."""
    commodity: str
    region: str
    forecast_data: List[Dict[str, Any]]
    model_confidence: float
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()