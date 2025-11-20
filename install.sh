#!/bin/bash
# DeePTB Installation Script with CPU/GPU Selection
#
# Usage:
#   ./install.sh           # Install CPU version (default)
#   ./install.sh cpu       # Install CPU version
#   ./install.sh cu118     # Install CUDA 11.8 version
#   ./install.sh cu121     # Install CUDA 12.1 version
#   ./install.sh cu124     # Install CUDA 12.4 version

set -e  # Exit on error

# Default to CPU
VARIANT="${1:-cpu}"

# Detect torch version from pyproject.toml (use 2.5.0 as default)
TORCH_VERSION="2.5.0"

# Set the find-links URL based on variant
FIND_LINKS_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${VARIANT}.html"

echo "======================================"
echo "DeePTB Installation Script"
echo "======================================"
echo "PyTorch variant: $VARIANT"
echo "Find-links URL: $FIND_LINKS_URL"
echo "======================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    pip install uv
fi

# Sync dependencies with the specified find-links
echo "Installing DeePTB with torch_scatter ($VARIANT version)..."
uv sync --find-links "$FIND_LINKS_URL"

echo ""
echo "✅ Installation complete!"
echo ""
echo "To run DeePTB:"
echo "  uv run dptb --help"
echo ""
