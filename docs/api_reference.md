
# AgriPrice Engine API Reference

## Price Calculation APIs

### `GET /api/v1/calculate_price`
Calculate price for a commodity based on quality parameters.

**Parameters:**
- `commodity` (string): Commodity name 
- `region` (string): Region name
- `quality_params` (object): Quality parameters

**Response:**
```json
{
  "final_price": 2500.0,
  "base_price": 2000.0,
  "quality_delta": 300.0,
  "location_delta": 100.0,
  "market_delta": 100.0
}
```

### `GET /api/v1/price_history`
Get historical prices for a commodity.

**Parameters:**
- `commodity` (string): Commodity name
- `region` (string): Region name 
- `days` (integer): Number of days of history

### `GET /api/v1/wizx_index`
Get WIZX index values for commodities.

**Parameters:**
- `commodity` (string, optional): Specific commodity
- `start_date` (string): Start date (YYYY-MM-DD)
- `end_date` (string): End date (YYYY-MM-DD)

## Quality Analysis APIs

### `POST /api/v1/analyze_quality`
Analyze quality parameters from images or lab reports.

**Parameters:**
- `commodity` (string): Commodity name
- `image_data` (file, optional): Image file
- `lab_report` (file, optional): Lab report file

## Data Submission APIs

### `POST /api/v1/submit_price`
Submit price data for verification.

**Parameters:**
- `commodity` (string): Commodity name
- `region` (string): Region name
- `price` (number): Price value
- `quality_params` (object): Quality parameters
- `source_details` (object): Source information
