# Tests Directory

Comprehensive test suite for the parking optimization system.

## Test Files

### Core Testing

- **`test_comprehensive.py`** - Main test suite with unit and integration tests
  - ParkingZone functionality tests
  - DynamicPricing algorithm validation
  - RouteOptimizer testing
  - Performance benchmarks

- **`test_system.py`** - System integration tests
  - End-to-end workflow validation
  - Component interaction testing

### Framework Validation

- **`test_framework.py`** - CIS 505 Framework Validation framework
  - Algorithm correctness verification
  - Complexity analysis validation
  - Performance benchmarking
  - Generates  Report data

- **`test_algorithm_validation.py`** - Detailed algorithm testing
  - Mathematical proof validation
  - Edge case testing
  - Stress testing scenarios

### API Integration

- **`test_apis.py`** - API integration testing
  - Google Maps API validation
  - Mapbox API testing
  - Fallback mode verification

### Supporting Files

- **`TESTING_GUIDE.md`** - Comprehensive testing documentation
- **`validation_data/`** - Generated test results and benchmarks

## Running Tests

```bash
# Run all tests
make test

# Generate test coverage
make test-coverage
```

## Analysis & Reporting

For comprehensive testing and analysis:

```bash
# Generate comprehensive report with analysis and visualizations
make report

# List all simulation runs
make list-runs

# Show details of latest run
make show-run
```

Results are saved to `tests/validation_data/` directory.
