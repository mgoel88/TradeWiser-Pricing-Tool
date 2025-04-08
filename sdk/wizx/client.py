"""
WIZX API Client for the Python SDK.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException

from .models import (
    CommodityInfo, PriceRequest, PriceResponse, QualityAnalysisResponse,
    PriceHistory, PriceHistoryEntry, CommodityIndex, IndexValue, PriceForecast
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WizxClient:
    """Client for interacting with the WIZX agricultural commodity pricing API."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "http://localhost:8000",
        timeout: int = 30
    ):
        """
        Initialize the WIZX client.

        Args:
            api_key: API key for authentication. If not provided, checks WIZX_API_KEY env var.
            base_url: Base URL for the API. Defaults to localhost for development.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("WIZX_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Some requests may fail.")

        self.base_url = base_url
        self.timeout = timeout

    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers with API key.

        Returns:
            Dictionary of request headers.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        files: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path.
            params: Query parameters.
            data: Request body data.
            files: Multipart files to upload.

        Returns:
            Response data as a dictionary.

        Raises:
            WizxApiError: If the API returns an error.
            ConnectionError: If there's a network error.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()

        try:
            if files:
                # Remove Content-Type for multipart/form-data
                if "Content-Type" in headers:
                    del headers["Content-Type"]
                    
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data,
                files=files,
                timeout=self.timeout
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # If response is empty or not JSON, return empty dict
            if not response.text:
                return {}
                
            return response.json()
        
        except RequestException as e:
            logger.error(f"API request error: {str(e)}")
            
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                try:
                    error_data = e.response.json()
                    error_message = error_data.get('detail', str(e))
                except ValueError:
                    error_message = e.response.text or str(e)
                    
                raise WizxApiError(error_message, status_code) from e
            else:
                raise ConnectionError(f"Connection error: {str(e)}") from e

    # Commodity endpoints
    def list_commodities(self) -> List[str]:
        """
        Get a list of all available commodities.

        Returns:
            List of commodity names.
        
        Raises:
            WizxApiError: If the API returns an error.
        """
        response = self._request("GET", "/commodities")
        return response

    def get_commodity(self, commodity: str) -> CommodityInfo:
        """
        Get detailed information about a specific commodity.

        Args:
            commodity: Name of the commodity.

        Returns:
            CommodityInfo object with commodity details.
            
        Raises:
            WizxApiError: If the API returns an error.
        """
        response = self._request("GET", f"/commodities/{commodity}")
        
        return CommodityInfo(
            name=response["name"],
            description=response.get("description"),
            trading_units=response.get("trading_units"),
            quality_parameters=response.get("quality_parameters", {})
        )

    def get_commodity_regions(self, commodity: str) -> List[str]:
        """
        Get all regions available for a specific commodity.

        Args:
            commodity: Name of the commodity.

        Returns:
            List of region names.
            
        Raises:
            WizxApiError: If the API returns an error.
        """
        response = self._request("GET", f"/commodities/{commodity}/regions")
        return response

    # Pricing endpoints
    def calculate_price(
        self,
        commodity: str,
        quality_params: Dict[str, float],
        region: str
    ) -> PriceResponse:
        """
        Calculate price for a commodity based on quality parameters and region.

        Args:
            commodity: Name of the commodity.
            quality_params: Dictionary of quality parameters.
            region: Region name.

        Returns:
            PriceResponse object with calculated price details.
            
        Raises:
            WizxApiError: If the API returns an error.
        """
        request_data = {
            "commodity": commodity,
            "quality_params": quality_params,
            "region": region
        }
        
        response = self._request("POST", "/price", data=request_data)
        
        return PriceResponse(
            commodity=response["commodity"],
            region=response["region"],
            base_price=response["base_price"],
            final_price=response["final_price"],
            quality_delta=response["quality_delta"],
            location_delta=response["location_delta"],
            market_delta=response["market_delta"],
            currency=response.get("currency", "INR"),
            timestamp=datetime.fromisoformat(response["timestamp"]) if isinstance(response["timestamp"], str) else response["timestamp"],
            unit=response.get("unit", "kg"),
            confidence=response.get("confidence")
        )

    def get_price_history(
        self,
        commodity: str,
        region: str,
        days: int = 30
    ) -> PriceHistory:
        """
        Get historical price data for a specific commodity and region.

        Args:
            commodity: Name of the commodity.
            region: Region name.
            days: Number of days of history to retrieve.

        Returns:
            PriceHistory object with historical price data.
            
        Raises:
            WizxApiError: If the API returns an error.
        """
        params = {"days": days}
        response = self._request("GET", f"/price-history/{commodity}/{region}", params=params)
        
        entries = []
        for entry in response["data"]:
            entries.append(PriceHistoryEntry(
                date=datetime.fromisoformat(entry["date"]) if isinstance(entry["date"], str) else entry["date"],
                price=entry["price"],
                volume=entry.get("volume")
            ))
            
        return PriceHistory(
            commodity=response["commodity"],
            region=response["region"],
            currency=response.get("currency", "INR"),
            unit=response.get("unit", "kg"),
            data=entries
        )

    # Quality Analysis endpoints
    def analyze_image(
        self,
        image_path: str,
        commodity: str,
        analysis_type: str = "detailed"
    ) -> QualityAnalysisResponse:
        """
        Analyze quality of a commodity from an image.

        Args:
            image_path: Path to the image file.
            commodity: Name of the commodity.
            analysis_type: Type of analysis to perform.

        Returns:
            QualityAnalysisResponse object with analysis results.
            
        Raises:
            WizxApiError: If the API returns an error.
            FileNotFoundError: If the image file doesn't exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        params = {
            "commodity": commodity,
            "analysis_type": analysis_type
        }
        
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            response = self._request("POST", "/quality-analysis/image", params=params, files=files)
        
        return QualityAnalysisResponse(
            commodity=response["commodity"],
            quality_params=response["quality_params"],
            quality_score=response["quality_score"],
            quality_grade=response["quality_grade"],
            confidence=response.get("confidence"),
            analysis_summary=response.get("analysis_summary"),
            timestamp=datetime.fromisoformat(response["timestamp"]) if isinstance(response["timestamp"], str) else response["timestamp"]
        )

    # Index endpoints
    def get_commodity_index(
        self,
        commodity: str,
        days: int = 30
    ) -> CommodityIndex:
        """
        Get index data for a specific commodity.

        Args:
            commodity: Name of the commodity.
            days: Number of days of index data to retrieve.

        Returns:
            CommodityIndex object with index data.
            
        Raises:
            WizxApiError: If the API returns an error.
        """
        params = {"days": days}
        response = self._request("GET", f"/index/{commodity}", params=params)
        
        history = []
        for entry in response["history"]:
            history.append(IndexValue(
                date=datetime.fromisoformat(entry["date"]) if isinstance(entry["date"], str) else entry["date"],
                value=entry["value"]
            ))
            
        return CommodityIndex(
            commodity=response["commodity"],
            current_value=response["current_value"],
            change_percentage=response["change_percentage"],
            history=history
        )

    # Forecasting endpoints
    def get_price_forecast(
        self,
        commodity: str,
        region: str,
        days: int = 30
    ) -> PriceForecast:
        """
        Get price forecast for a specific commodity and region.

        Args:
            commodity: Name of the commodity.
            region: Region name.
            days: Number of days to forecast.

        Returns:
            PriceForecast object with forecast data.
            
        Raises:
            WizxApiError: If the API returns an error.
        """
        params = {"days": days}
        response = self._request("GET", f"/forecast/{commodity}/{region}", params=params)
        
        return PriceForecast(
            commodity=response["commodity"],
            region=response["region"],
            forecast_data=response["forecast_data"],
            model_confidence=response["model_confidence"],
            created_at=datetime.fromisoformat(response["created_at"]) if isinstance(response["created_at"], str) else response["created_at"]
        )


class WizxApiError(Exception):
    """Exception raised for API errors."""
    
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)