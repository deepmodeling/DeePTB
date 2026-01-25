"""
DOS calculation task module.

This module provides functions for computing Density of States (DOS)
using the Pardiso solver.
"""

module DosCalculation

using Printf
using SparseArrays
using LinearAlgebra
using DelimitedFiles
using Dates

# Import shared modules from parent scope (main.jl)
using ..Hamiltonian: HR2HK
using ..KPoints: gen_kmesh

export run_dos_calculation

"""
    run_dos_calculation(config, H_R, S_R, structure, output_dir, solver_opts, log_path, solver_func)

Main function to run DOS calculation.
"""
function run_dos_calculation(config, H_R, S_R, structure, output_dir, solver_opts, log_path, solver_func)
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
    nkmesh = Int64.(config["kmesh"])
    num_band = get(config, "num_band", 8)
    fermi_level = get(config, "fermi_level", 0.0)
    max_iter = get(config, "max_iter", 300)
    norbits = structure["norbits"]

    log_message("Starting Density of States Calculation")
    log_message("K-Grid: $nkmesh")

    # Generate K-grid
    ks, nk_total = gen_kmesh(nkmesh)
    
    log_message("Total K-points: $nk_total, Bands: $num_band, Fermi: $fermi_level eV")

    egvals_all = zeros(num_band, nk_total)
    start_time = time()

    # Main loop
    for ik in 1:nk_total
        kpt = ks[:,ik]
        H_k, S_k = HR2HK(kpt, H_R, S_R, norbits)

        # Solve eigenvalues
        # Note: DOS usually doesn't need eigenvectors, so out_wfc=false
        egval, _, _ = solver_func(H_k, S_k, fermi_level, num_band, max_iter,
                                  false, solver_opts.ill_project, solver_opts.ill_threshold)
        
        egvals_all[:,ik] = egval

        if ik % 50 == 0 || ik == nk_total
            elapsed = (time() - start_time) / 60
            log_message(@sprintf("DOS K-point %5d/%d done | Elapsed: %.2f min", ik, nk_total, elapsed))
        end
    end

    # Save eigenvalues
    writedlm(joinpath(output_dir, "egvals.dat"), egvals_all)

    # Compute DOS if config present
    if haskey(config, "epsilon") && haskey(config, "omegas")
        ϵ = config["epsilon"]
        ω_cfg = config["omegas"]
        ωlist = collect(range(ω_cfg[1], ω_cfg[2], length = Int(ω_cfg[3])))
        dos = zeros(length(ωlist))
        
        # Gaussian broadening calculation
        factor = 1.0 / (ϵ * sqrt(pi) * nk_total)
        
        for ik in 1:nk_total, ib in 1:num_band, iω in eachindex(ωlist)
            diff = egvals_all[ib,ik] - ωlist[iω] - fermi_level
            dos[iω] += exp(-(diff^2 / ϵ^2)) * factor
        end
        
        open(joinpath(output_dir, "dos.dat"), "w") do f
            for (ω, d) in zip(ωlist, dos)
                @printf(f, "%12.6f  %12.6f\\n", ω - fermi_level, d)
            end
        end
        log_message("Generated dos.dat")
    end
end

end # module
