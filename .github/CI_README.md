# CI/CD Setup for optimalgiv

This document describes the continuous integration setup for the optimalgiv package.

## Overview

The CI pipeline tests the Python wrapper against the Julia package across multiple versions and platforms to ensure compatibility and correctness.

## Workflow Structure

### Main Test Job (`test`)
- **Matrix Testing**: Tests across:
  - Python versions: 3.9, 3.10, 3.11, 3.12
  - Julia versions: 1.9, 1.10
  - OS: Ubuntu (all combinations), macOS and Windows (latest versions only)
- **Environment**: Sets `JULIA_NUM_THREADS=1` to prevent segfaults
- **Caching**: Caches both Julia packages and Python dependencies
- **Coverage**: Generates coverage reports and uploads to Codecov

### Fresh Install Test (`fresh-install-test`)
- Tests the first-time installation experience
- Runs on all three OS platforms
- Forces full Julia setup with `OPTIMALGIV_FORCE_SETUP=1`
- Validates that the package can install Julia and all dependencies from scratch

## Test Configuration

### Dependencies
Test dependencies are specified in `pyproject.toml`:
```toml
[project.optional-dependencies]
test = [
    "pytest >= 7.0",
    "pytest-cov >= 4.0",
    "pytest-xdist >= 3.0",
]
```

### pytest Configuration
See `pytest.ini` for test discovery and execution settings.

### Coverage Configuration
See `.coveragerc` for coverage measurement settings.

## Local Testing

Before pushing changes, you can test the CI setup locally:

```bash
# Run the local CI test script
./scripts/test_ci_locally.sh

# Or manually:
export JULIA_NUM_THREADS=1
pip install -e ".[test]"
pytest tests/ -v --cov=optimalgiv
```

## Triggering CI

The CI runs on:
- Push to `main` or `internalpc` branches
- Pull requests to `main`
- Manual workflow dispatch

## Test Data

Tests use `examples/simdata1.csv` which contains synthetic data for testing the GIV estimation algorithms.

## Troubleshooting

### Common Issues

1. **Julia Installation Timeout**: The first-time Julia setup can take 2-4 minutes. The CI has appropriate timeouts configured.

2. **Segmentation Faults**: Always ensure `JULIA_NUM_THREADS=1` is set.

3. **Cache Issues**: If tests fail due to corrupted cache, manually clear the cache in GitHub Actions settings.

### Adding New Tests

1. Add test files to `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures for common setup
4. Mark slow tests with `@pytest.mark.slow`

## Future Improvements

- [ ] Add benchmarking tests
- [ ] Add documentation build tests
- [ ] Add integration tests with real Julia package updates
- [ ] Add release automation