#!/bin/bash
# Julia Installation and Setup Script for DeePTB Pardiso Backend
# Supported Platform: Linux only

set -e  # Exit on error

echo "========================================="
echo "DeePTB Pardiso Backend Setup"
echo "========================================="

# Check platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Error: macOS is not supported for Pardiso backend."
    echo "Pardiso requires Intel MKL which has limited support on macOS."
    echo ""
    echo "Alternative: Use the dense solver backend or run on Linux."
    exit 1
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo "Error: Windows is not currently supported for Pardiso backend."
    echo ""
    echo "Alternative: Use WSL2 (Windows Subsystem for Linux) or run on a Linux machine."
    exit 1
elif [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: Unsupported platform: $OSTYPE"
    echo "This script is designed for Linux only."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Platform: Linux (supported)"
echo ""

# Check if Julia is already installed
if command -v julia &> /dev/null; then
    JULIA_VERSION=$(julia --version | grep -oP '\d+\.\d+\.\d+' || julia --version)
    echo "Julia is already installed: $JULIA_VERSION"
    read -p "Do you want to reinstall Julia? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping Julia installation."
        SKIP_JULIA_INSTALL=true
    fi
fi

# Install Julia if needed
if [ "$SKIP_JULIA_INSTALL" != "true" ]; then
    echo ""
    echo "Installing Julia..."
    echo "========================================="
    
    # Use official Julia installer
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    
    # Add Julia to PATH for current session
    export PATH="$HOME/.juliaup/bin:$PATH"
    
    echo "Julia installation complete!"
fi

# Verify Julia installation
echo ""
echo "Verifying Julia installation..."
julia --version

# Install Julia packages
echo ""
echo "Installing required Julia packages..."
echo "========================================="

julia install_julia_packages.jl

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "You can now use the DeePTB Pardiso backend:"
echo "  dptb pdso band.json -i model.pth -stu structure.vasp -o ./output"
echo ""
echo "For more information, see:"
echo "  - examples/To_pardiso/README.md"
echo "  - dptb/postprocess/pardiso/README.md"
echo ""
