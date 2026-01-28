# Julia Package Dependencies for DeePTB Pardiso Backend

# This file lists all Julia packages required for the Pardiso backend.
# You can install them by running:
#   julia install_julia_packages.jl

using Pkg

println("Installing DeePTB Pardiso backend dependencies...")
println("=" ^ 60)

# Core packages
packages = [
    "JSON",           # JSON parsing for structure.json
    "HDF5",           # HDF5 file I/O for Hamiltonian/Overlap matrices
    "ArgParse",       # Command-line argument parsing
    "Pardiso",        # Intel MKL Pardiso solver
    "Arpack",         # Eigenvalue solver (ARPACK)
    "LinearMaps",     # Linear map abstractions
    "JLD",            # Julia data serialization (for caching)
    "SparseArrays",   # Sparse matrix support
    "LinearAlgebra",  # Linear algebra operations
    "Printf",         # Formatted printing
    "Dates",          # Date/time utilities
    "DelimitedFiles"  # Text file I/O (for legacy .dat files)
]

# Install packages
for pkg in packages
    println("\nInstalling $pkg...")
    try
        Pkg.add(pkg)
        println("✓ $pkg installed successfully")
    catch e
        println("✗ Failed to install $pkg")
        println("  Error: $e")
    end
end

println("\n" * "=" ^ 60)
println("Precompiling packages...")
Pkg.precompile()

println("\n" * "=" ^ 60)
println("Installation complete!")
println("\nVerifying Pardiso installation...")

try
    using Pardiso
    println("✓ Pardiso.jl loaded successfully")
    println("  MKL Pardiso available: ", Pardiso.MKL_PARDISO_LOADED[])
catch e
    println("✗ Pardiso verification failed")
    println("  Error: $e")
    println("\nNote: Pardiso requires Intel MKL to be installed.")
    println("See: https://github.com/JuliaSparse/Pardiso.jl")
end

println("\nSetup complete! You can now use the Pardiso backend.")
