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

export run_dos_calculation

"""
    construct_hk(kpt, H_R, S_R, norbits)

Construct H(k) and S(k) from H(R) and S(R) via Fourier transform.
"""
function construct_hk(kpt, H_R, S_R, norbits)
    default_dtype = Complex{Float64}
    H_k = spzeros(default_dtype, norbits, norbits)
    S_k = spzeros(default_dtype, norbits, norbits)

    for R in keys(H_R)
        phase = exp(im * 2π * dot(kpt, R))
        H_k += H_R[R] * phase
        S_k += S_R[R] * phase
    end

    H_k = (H_k + H_k') / 2
    S_k = (S_k + S_k') / 2

    return H_k, S_k
end

"""
    run_dos_calculation(config, H_R, S_R, structure, output_dir, solver_opts, log_path, solver_func)

Main function to run DOS calculation.
"""
function run_dos_calculation(config, H_R, S_R, structure, output_dir, solver_opts, log_path, solver_func)
    # Extract parameters
    nkmesh = Int64.(config["kmesh"])
    nk_total = prod(nkmesh)
    num_band = get(config, "num_band", 8)
    fermi_level = get(config, "fermi_level", 0.0)
    max_iter = get(config, "max_iter", 300)
    norbits = structure["norbits"]

    @info "Starting Density of States Calculation"
    @info "K-Grid: $nkmesh (Total: $nk_total), Bands: $num_band, Fermi: $fermi_level eV"

    # Generate K-grid
    ks = zeros(3, nk_total)
    ik = 1
    for ix in 1:nkmesh[1], iy in 1:nkmesh[2], iz in 1:nkmesh[3]
        ks[:,ik] .= [(ix-1)/nkmesh[1], (iy-1)/nkmesh[2], (iz-1)/nkmesh[3]]
        ik += 1
    end

    egvals_all = zeros(num_band, nk_total)
    start_time = time()

    # Main loop
    for ik in 1:nk_total
        kpt = ks[:,ik]
        H_k, S_k = construct_hk(kpt, H_R, S_R, norbits)

        # Solve eigenvalues
        # Note: DOS usually doesn't need eigenvectors, so out_wfc=false
        egval, _, _ = solver_func(H_k, S_k, fermi_level, num_band, max_iter,
                                  false, solver_opts.ill_project, solver_opts.ill_threshold)
        
        egvals_all[:,ik] = egval

        if ik % 50 == 0 || ik == nk_total
            elapsed = (time() - start_time) / 60
            @info @sprintf("DOS K-point %5d/%d done | Elapsed: %.2f min", ik, nk_total, elapsed)
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
        @info "Generated dos.dat"
    end
end

end # module
