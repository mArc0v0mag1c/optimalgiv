#!/bin/bash
# Script to test CI setup locally
# This mimics what the GitHub Actions CI will do

set -e  # Exit on error

echo "=== Local CI Test Script ==="
echo "This script tests the CI setup locally before pushing to GitHub"
echo ""

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Must run this script from the project root directory"
    exit 1
fi

# Set environment variables like CI
export JULIA_NUM_THREADS=1
export PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=no

echo "1. Installing package and test dependencies..."
uv sync --dev

echo ""
echo "2. Checking Julia setup..."
uv run python -c "
import sys
print(f'Python version: {sys.version}')
import optimalgiv as og
print(f'optimalgiv version: {og.__version__}')
print('Julia initialized successfully!')
"

echo ""
echo "3. Running tests with coverage..."
uv run pytest tests/ -v --cov=optimalgiv --cov-report=term --cov-report=html

echo ""
echo "4. Testing fresh install scenario (optional - requires clean environment)..."
echo "   To test fresh install, run in a new virtual environment:"
echo "   OPTIMALGIV_FORCE_SETUP=1 python -c 'import optimalgiv'"

echo ""
echo "=== Local CI test completed successfully! ==="
echo "Coverage report available in htmlcov/index.html"