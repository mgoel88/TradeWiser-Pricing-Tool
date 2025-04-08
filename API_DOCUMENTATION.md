# WIZX Agricultural Platform API Documentation

The WIZX Agricultural Platform provides RESTful APIs for accessing agricultural commodity pricing data, quality assessment, and market insights. This document provides information on how to use these APIs.

## Authentication

All API endpoints require authentication using an API key. Include the API key in the request header as follows:

```
X-API-Key: your_api_key
```

## Base URL

Development: `http://localhost:8000`
Production: `https://api.wizx.io`

## API Endpoints

### General

#### Health Check
```
GET /health
```

Returns the current status of the API.

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2025-04-08T12:34:56.789Z"
}
```

### Commodities

#### List Commodities
```
GET /commodities
```

Returns a list of all available commodities.

**Response**
```json
[
  "Rice",
  "Wheat",
  "Maize",
  "Soybean",
  "Cotton"
]
```

#### Get Commodity Details
```
GET /commodities/{commodity}
```

Returns detailed information about a specific commodity.

**Parameters**
- `commodity` (path): The name of the commodity

**Response**
```json
{
  "name": "Rice",
  "description": "Rice is a staple food in India and widely cultivated.",
  "trading_units": "kg",
  "quality_parameters": {
    "moisture_content": {
      "min": 8,
      "max": 20,
      "standard_value": 14,
      "impact_factor": 0.8
    },
    "broken_percentage": {
      "min": 0,
      "max": 30,
      "standard_value": 5,
      "impact_factor": 1.2
    },
    "foreign_matter": {
      "min": 0,
      "max": 5,
      "standard_value": 0.5,
      "impact_factor": 1.5
    }
  }
}
```

#### Get Commodity Regions
```
GET /commodities/{commodity}/regions
```

Returns all regions available for a specific commodity.

**Parameters**
- `commodity` (path): The name of the commodity

**Response**
```json
[
  "Karnataka",
  "Maharashtra",
  "Punjab",
  "Uttar Pradesh"
]
```

### Pricing

#### Calculate Price
```
POST /price
```

Calculate price for a commodity based on quality parameters and region.

**Request Body**
```json
{
  "commodity": "Rice",
  "quality_params": {
    "moisture_content": 14.5,
    "broken_percentage": 6.2,
    "foreign_matter": 0.8
  },
  "region": "Karnataka"
}
```

**Response**
```json
{
  "commodity": "Rice",
  "region": "Karnataka",
  "base_price": 2400.00,
  "final_price": 2356.80,
  "quality_delta": -72.00,
  "location_delta": 28.80,
  "market_delta": 0.00,
  "currency": "INR",
  "timestamp": "2025-04-08T12:34:56.789Z",
  "unit": "kg",
  "confidence": 0.92
}
```

#### Get Price History
```
GET /price-history/{commodity}/{region}
```

Get historical price data for a specific commodity and region.

**Parameters**
- `commodity` (path): The name of the commodity
- `region` (path): The name of the region
- `days` (query, optional): Number of days of history to retrieve (default: 30)

**Response**
```json
{
  "commodity": "Rice",
  "region": "Karnataka",
  "currency": "INR",
  "unit": "kg",
  "data": [
    {
      "date": "2025-04-08T00:00:00.000Z",
      "price": 2400.00,
      "volume": 1250.0
    },
    {
      "date": "2025-04-07T00:00:00.000Z",
      "price": 2389.50,
      "volume": 1450.0
    },
    ...
  ]
}
```

### Quality Analysis

#### Analyze Commodity Image
```
POST /quality-analysis/image
```

Analyze quality of a commodity from an uploaded image.

**Parameters**
- `commodity` (query): The commodity to analyze
- `analysis_type` (query, optional): Type of analysis to perform (default: "detailed")
- `image` (form, file): Image of the commodity to analyze

**Response**
```json
{
  "commodity": "Rice",
  "quality_params": {
    "moisture_content": 14.8,
    "broken_percentage": 7.2,
    "foreign_matter": 0.9,
    "discoloration": 3.5,
    "quality_score": 82.5,
    "quality_grade": "A"
  },
  "quality_score": 82.5,
  "quality_grade": "A",
  "confidence": 0.88,
  "analysis_summary": "The rice sample shows good quality with slightly elevated moisture content. Minor broken grains and foreign matter present. Overall premium grade suitable for export markets.",
  "timestamp": "2025-04-08T12:34:56.789Z"
}
```

### Indices

#### Get Commodity Index
```
GET /index/{commodity}
```

Get index data for a specific commodity.

**Parameters**
- `commodity` (path): The name of the commodity
- `days` (query, optional): Number of days of index data to retrieve (default: 30)

**Response**
```json
{
  "commodity": "Rice",
  "current_value": 124.8,
  "change_percentage": 1.2,
  "history": [
    {
      "date": "2025-04-08T00:00:00.000Z",
      "value": 124.8
    },
    {
      "date": "2025-04-07T00:00:00.000Z",
      "value": 123.3
    },
    ...
  ]
}
```

### Forecasting

#### Get Price Forecast
```
GET /forecast/{commodity}/{region}
```

Get price forecast for a specific commodity and region.

**Parameters**
- `commodity` (path): The name of the commodity
- `region` (path): The name of the region
- `days` (query, optional): Number of days to forecast (default: 30)

**Response**
```json
{
  "commodity": "Rice",
  "region": "Karnataka",
  "forecast_data": [
    {
      "date": "2025-04-09T00:00:00.000Z",
      "price": 2412.50,
      "lower": 2380.20,
      "upper": 2445.80
    },
    {
      "date": "2025-04-10T00:00:00.000Z",
      "price": 2425.10,
      "lower": 2385.50,
      "upper": 2465.70
    },
    ...
  ],
  "model_confidence": 0.85,
  "created_at": "2025-04-08T12:34:56.789Z"
}
```

## Error Handling

The API returns standard HTTP status codes to indicate the success or failure of a request.

**Common Error Codes**
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Valid request but unable to process
- `500 Internal Server Error`: Server error

**Error Response Format**
```json
{
  "detail": "Error message describing the issue"
}
```

## Rate Limiting

API requests are limited to 100 requests per minute per API key. If you exceed this limit, you will receive a `429 Too Many Requests` response.

## SDK Integration

For easier integration, we provide SDKs for various programming languages:

- [Python SDK](https://github.com/wizx/wizx-python-sdk)
- [JavaScript SDK](https://github.com/wizx/wizx-js-sdk)
- [Java SDK](https://github.com/wizx/wizx-java-sdk)

## Support

For API support or to report issues, please contact:
- Email: api-support@wizx.io
- Documentation: https://docs.wizx.io