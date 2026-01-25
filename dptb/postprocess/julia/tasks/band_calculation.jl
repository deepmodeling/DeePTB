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

export run_band_calculation

# Solver function is now passed as argument

"""
    construct_hk(kpt, H_R, S_R, norbits)

Construct H(k) and S(k) from H(R) and S(R) via Fourier transform.

# Arguments
- `kpt::Vector{Float64}`: k-point in fractional coordinates
- `H_R::Dict`: Hamiltonian in real space
- `S_R::Dict`: Overlap in real space
- `norbits::Int`: Total number of orbitals

# Returns
- `H_k::SparseMatrixCSC`: Hamiltonian at k
- `S_k::SparseMatrixCSC`: Overlap at k
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

    # Hermitianize
    H_k = (H_k + H_k') / 2
    S_k = (S_k + S_k') / 2

    return H_k, S_k
end

"""
    parse_kpath_abacus(kpath_config, lat, labels)

Parse k-path in ABACUS format.

# Returns
- `klist::Vector{Vector{Float64}}`: List of k-points
- `xlist::Vector{Float64}`: Cumulative distances
- `high_sym_kpts::Vector{Float64}`: High-symmetry point positions
- `klabels::Vector{String}`: K-point labels
"""
function parse_kpath_abacus(kpath_config, lat, labels=String[])
    klist_vec = Vector{Vector{Float64}}()
    xlist = Float64[]
    high_sym_kpts = Float64[0.0]
    klabels_vec = isempty(labels) ? String[] : copy(labels)

    total_dist = 0.0
    kpoints = [Float64.(row[1:3]) for row in kpath_config]
    recip_metric_inv = 2π * inv(lat)

    for i in 1:(length(kpath_config)-1)
        k_start = kpoints[i]
        k_end = kpoints[i+1]
        n_segment = Int(kpath_config[i][4])

        dk = k_end .- k_start
        dk_cart = dk' * recip_metric_inv
        dist_segment = norm(dk_cart)

        if n_segment > 0
            for j in 0:(n_segment-1)
                t = j / n_segment
                kpt = k_start .+ t .* dk
                push!(klist_vec, kpt)
                push!(xlist, total_dist + t * dist_segment)
            end
        end

        total_dist += dist_segment
        push!(high_sym_kpts, total_dist)
    end

    push!(klist_vec, kpoints[end])
    push!(xlist, total_dist)

    return klist_vec, xlist, high_sym_kpts, klabels_vec
end

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
        klist, xlist, high_sym, klabels = parse_kpath_abacus(kpath_cfg, lat, klabels_input)
    else
        error("Only ABACUS kline_type supported in modular version")
    end

    @info "Starting Band Structure Calculation"
    @info "Total K-points: $(length(klist)), Bands: $num_band, Fermi: $fermi_level eV"

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
        H_k, S_k = construct_hk(kpt, H_R, S_R, norbits)

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
            @info @sprintf("K-point %4d/%d done | Elapsed: %.2f min", ik, length(klist), elapsed)
        end
    end

    # Save final HDF5 format
    save_bandstructure_h5(klist, xlist, all_egvals, fermi_level, high_sym, klabels, output_dir)

    @info "Band structure calculation completed"
end

end # module
