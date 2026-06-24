#!/bin/sh
#
# Unit Test Script for CI/CD
# Used by: .github/workflows/unit_test.yml
# For user installation, use: ./install.sh directly
#
set -e

# Install dependencies using the same tested installer path used by users.
bash install.sh cpu --extra pythtb --test

# Run tests in the environment prepared by install.sh.
.venv/bin/python -m pytest ./dptb/tests/
