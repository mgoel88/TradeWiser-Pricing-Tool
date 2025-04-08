# WIZX Agricultural Commodity Pricing SDK

A Python SDK for integrating with the WIZX Agricultural Commodity Pricing platform. This SDK makes it easy to access commodity pricing data, analyze quality, track indices, and forecast prices.

## Installation

```bash
pip install wizx-sdk
```

## Quick Start

```python
from wizx import WizxClient

# Initialize the client
client = WizxClient(api_key="your_api_key")

# Get a list of available commodities
commodities = client.list_commodities()
print(f"Available commodities: {', '.join(commodities)}")

# Get information about a specific commodity
rice_info = client.get_commodity("Rice")
print(f"Rice details: {rice_info.description}")

# Get regions where a commodity is traded
regions = client.get_commodity_regions("Rice")
print(f"Rice regions: {', '.join(regions)}")

# Calculate price based on quality parameters
price = client.calculate_price(
    commodity="Rice",
    quality_params={
        "moisture_content": 14.0,
        "broken_percentage": 5.0,
        "foreign_matter": 0.5,
        "chalkiness": 2.0
    },
    region="Karnataka"
)
print(f"Final price: ₹{price.final_price:.2f} per {price.unit}")
print(f"Base price: ₹{price.base_price:.2f}, Quality adjustment: ₹{price.quality_delta:.2f}")

# Get price history
history = client.get_price_history("Rice", "Karnataka", days=60)
for entry in history.data[-5:]:
    print(f"{entry.date.strftime('%Y-%m-%d')}: ₹{entry.price:.2f}")

# Analyze quality from an image
analysis = client.analyze_image(
    image_path="path/to/rice_sample.jpg",
    commodity="Rice",
    analysis_type="detailed"
)
print(f"Quality score: {analysis.quality_score}/100")
print(f"Quality grade: {analysis.quality_grade}")
print(f"Analysis summary: {analysis.analysis_summary}")

# Get commodity index
index = client.get_commodity_index("Rice", days=30)
print(f"Current Rice index: {index.current_value:.2f} ({index.change_percentage:+.2f}%)")

# Get price forecast
forecast = client.get_price_forecast("Rice", "Karnataka", days=14)
print(f"14-day forecast:")
for day in forecast.forecast_data:
    print(f"{day['date']}: ₹{day['price']:.2f}")
print(f"Model confidence: {forecast.model_confidence:.1%}")
```

## Authentication

The SDK requires an API key for authentication. You can provide it in one of two ways:

1. Pass it directly to the WizxClient constructor:
   ```python
   client = WizxClient(api_key="your_api_key")
   ```

2. Set the `WIZX_API_KEY` environment variable:
   ```python
   # API key will be automatically loaded from the WIZX_API_KEY env var
   client = WizxClient()
   ```

## API Documentation

For complete API documentation, visit our online documentation:
https://docs.wizx.io/api

## Error Handling

The SDK uses custom exceptions to handle errors:

```python
from wizx import WizxClient
from wizx.client import WizxApiError

client = WizxClient(api_key="your_api_key")

try:
    # Make an API call
    rice_info = client.get_commodity("NonExistentCommodity")
except WizxApiError as e:
    print(f"API Error: {e.message} (Status code: {e.status_code})")
except ConnectionError as e:
    print(f"Connection Error: {str(e)}")
```

## License

This SDK is released under the MIT License. See the LICENSE file for details.