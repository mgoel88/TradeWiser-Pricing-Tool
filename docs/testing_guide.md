
# Testing Guide

## Setting Up Test Environment

1. The project uses Python's unittest framework
2. Tests are organized in the `tests` directory
3. Test data is stored in `tests/fixtures`

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests/test_pricing_engine.py

# Run with coverage
coverage run -m unittest discover tests
coverage report
```

## Test Categories

### Unit Tests
- `test_pricing_engine.py`: Tests for price calculation logic
- `test_quality_analyzer.py`: Tests for quality analysis functions
- `test_data_processor.py`: Tests for data processing functions

### Integration Tests
- `test_api_endpoints.py`: Tests for API endpoints
- `test_database.py`: Tests for database operations

### Performance Tests
- `test_performance.py`: Tests for system performance under load

## Writing Tests

Example test case:
```python
def test_price_calculation():
    result = calculate_price(
        commodity="Wheat",
        quality_params={"moisture": 12.0},
        region="North India"
    )
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert all(isinstance(x, float) for x in result)
```
