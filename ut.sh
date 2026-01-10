#!/bin/sh
#
# Unit Test Script for CI/CD
# Used by: .github/workflows/unit_test.yml
# For user installation, use: ./install.sh directly
#

# Install dependencies using install.sh (CPU version)
# This ensures consistent installation logic between user and CI
bash install.sh cpu

# Install optional dependencies for testing
uv sync --extra pythtb

# Run tests
uv run pytest ./dptb/tests/