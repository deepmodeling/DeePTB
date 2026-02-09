"""
Band structure calculation task module.

This module provides functions for computing band structures along
specified k-paths using the Pardiso solver.
"""

module BandCalculation

using Printf
using SparseArrays
using LinearAlgebra
using JSON
using HDF5
using Dates

# Import shared modules from parent scope (main.jl)
using ..Hamiltonian: HR2HK
using ..KPoints: parse_kpath

export run_band_calculation

# Solver function is now passed as argument

"""
    save_bandstructure_h5(klist, xlist, eigenvalues, e_fermi, high_sym, labels, output_dir)

Save band structure data to HDF5 format.
"""
function save_bandstructure_h5(klist, xlist, eigenvalues, e_fermi, high_sym, labels, output_dir)
    h5_path = joinpath(output_dir, "bandstructure.h5")
    try
        h5open(h5_path, "w") do file
            # Convert Vector{Vector} to Matrix for HDF5
            # hcat(eigenvalues...) -> [nb, nk] matrix
            write(file, "eigenvalues", hcat(eigenvalues...))
            write(file, "klist", hcat(klist...))
            write(file, "xlist", xlist)
            write(file, "E_fermi", e_fermi)
            write(file, "high_sym_kpoints", high_sym)
            write(file, "labels", labels)
        end
        @info "Generated bandstructure.h5"
    catch e
        @warn "Failed to generate bandstructure.h5: $e"
    end
end

"""
    run_band_calculation(config, H_R, S_R, structure, output_dir, solver_opts, log_path, solver_func)

Main function to run band structure calculation.

# Arguments
- `solver_func::Function`: Function to solve eigenvalue problem at k
"""
function run_band_calculation(config, H_R, S_R, structure, output_dir, solver_opts, log_path, solver_func)
    # Helper for logging
    function log_message(msg)
        @info msg
        try
            open(log_path, "a") do f
                println(f, "[$(Dates.now())] $msg")
            end
        catch e
            # Ignore file errors to prevent crash
        end
    end

    # Extract parameters
    kline_type = get(config, "kline_type", "abacus")
    num_band = get(config, "num_band", 8)
    fermi_level = get(config, "E_fermi", 0.0)
    max_iter = get(config, "max_iter", 300)
    out_wfc = get(config, "out_wfc", "false") == "true"

    lat = structure["cell"]
    norbits = structure["norbits"]

    # Parse k-path
    if kline_type == "abacus"
        kpath_cfg = config["kpath"]
        klabels_input = string.(get(config, "klabels", String[]))
        klist, xlist, high_sym, klabels = parse_kpath(kpath_cfg, lat, klabels_input)
    else
        error("Only ABACUS kline_type supported in modular version")
    end

    log_message("Starting Band Structure Calculation")
    log_message("Total K-points: $(length(klist)), Bands: $num_band, Fermi: $fermi_level eV")

    # Initialize text output (bands.dat)
    txt_path = joinpath(output_dir, "bands.dat")
    open(txt_path, "w") do f
        @printf(f, "# %4s %10s %10s %10s %12s %s\n", "Idx", "Kx", "Ky", "Kz", "Dist", "Eigenvalues(eV, shifted by Fermi)")
    end

    all_egvals = Vector{Vector{Float64}}()
    start_time = time()

    # Main calculation loop
    for (ik, kpt) in enumerate(klist)
        # Construct H(k) and S(k)
        H_k, S_k = HR2HK(kpt, H_R, S_R, norbits)

        # Solve eigenvalue problem using provided solver
        egvals, _, _ = solver_func(H_k, S_k, fermi_level, num_band, max_iter,
                                        false, solver_opts.ill_project, solver_opts.ill_threshold)
        push!(all_egvals, egvals)

        # Append to text file incrementally
        open(txt_path, "a") do f
            # Write K-point info
            @printf(f, "%6d %10.6f %10.6f %10.6f %12.6f", ik, kpt[1], kpt[2], kpt[3], xlist[ik])
            # Write eigenvalues (shifted by Fermi level for consistency with plot)
            for e in egvals
                @printf(f, " %12.6f", e - fermi_level)
            end
            @printf(f, "\n")
        end

        # Progress logging
        if ik % 10 == 0 || ik == length(klist)
            elapsed = (time() - start_time) / 60
            log_message(@sprintf("K-point %4d/%d done | Elapsed: %.2f min", ik, length(klist), elapsed))
        end
    end

    # Save final HDF5 format
    save_bandstructure_h5(klist, xlist, all_egvals, fermi_level, high_sym, klabels, output_dir)

    log_message("Band structure calculation completed")
end

end # module
