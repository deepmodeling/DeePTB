module Hamiltonian

using SparseArrays
using LinearAlgebra
using Serialization

export HR2HK, blocks_to_csc, get_HR_SR_sparse

"""
    get_HR_SR_sparse(input_dir, structure, matrix_loader, use_cache)

Main function to get sparse Hamiltonian and Overlap matrices, with caching.
"""
function get_HR_SR_sparse(input_dir::String, structure::Dict, matrix_loader::Function, use_cache::Bool=true)
    cache_file = joinpath(input_dir, "sparse_matrices.jld")

    if use_cache && isfile(cache_file)
        println("Loading H/S matrices from cache: $cache_file")
        try
            data = deserialize(cache_file)
            if isa(data, Tuple) && length(data) == 2
                 return data[1], data[2]
            else
                 println("Cache format invalid. Rebuilding...")
            end
        catch e
            println("Failed to load cache: $e. Rebuilding...")
        end
    end

    println("Loading H/S matrices from HDF5 files...")
    h_blocks = matrix_loader(joinpath(input_dir, "predicted_hamiltonians.h5"))
    s_blocks = matrix_loader(joinpath(input_dir, "predicted_overlaps.h5"))
    
    println("Building sparse Hamiltonian matrices...")
    H_R, S_R = blocks_to_csc(h_blocks, s_blocks, structure["norbits"], structure["site_norbits"])

    if use_cache
        println("Saving sparse H/S matrices to cache: $cache_file")
        serialize(cache_file, (H_R, S_R))
    end

    return H_R, S_R
end

"""
    HR2HK(kpt, H_R, S_R, norbits)

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
function HR2HK(kpt, H_R, S_R, norbits)
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
    blocks_to_csc(h_blocks, s_blocks, norbits, site_norbits)

Construct sparse matrices H(R) and S(R) from raw blocks.

# Arguments
- `h_blocks::Dict`: Raw Hamiltonian blocks from HDF5
- `s_blocks::Dict`: Raw Overlap blocks from HDF5
- `norbits::Int`: Total number of orbitals
- `site_norbits::Vector{Int}`: Number of orbitals per site

# Returns
- `H_R`: Dict{Vector{Int}, SparseMatrixCSC}
- `S_R`: Dict{Vector{Int}, SparseMatrixCSC}
"""
function blocks_to_csc(h_blocks::Dict{String, Any}, s_blocks::Dict{String, Any}, 
                                 norbits::Int, site_norbits::Vector{Int})
    
    # Precompute orbital start indices for each atom (1-based for Julia)
    atom_orb_starts = cumsum(vcat([0], site_norbits))

    # We need to collect triplets (I, J, V) for each R vector
    H_data = Dict{Vector{Int}, Tuple{Vector{Int}, Vector{Int}, Vector{ComplexF64}}}()
    S_data = Dict{Vector{Int}, Tuple{Vector{Int}, Vector{Int}, Vector{ComplexF64}}}()
    
    # Function to process a set of blocks (H or S)
    function process_blocks!(blocks, data_dict)
        for (key, block) in blocks
            # Parse key "src_dst_rx_ry_rz"
            parts = split(key, '_')
            if length(parts) != 5
                @warn "Skipping malformed key: $key"
                continue
            end
            
            src = parse(Int, parts[1]) + 1
            dst = parse(Int, parts[2]) + 1
            rx = parse(Int, parts[3])
            ry = parse(Int, parts[4])
            rz = parse(Int, parts[5])
            
            R = [rx, ry, rz]
            
            if !haskey(data_dict, R)
                data_dict[R] = (Int[], Int[], ComplexF64[])
            end
            
            rows, cols, vals = data_dict[R]
            
            orb_start_src = atom_orb_starts[src]
            orb_start_dst = atom_orb_starts[dst]
            norb_src = site_norbits[src]
            norb_dst = site_norbits[dst]
            
            # Apply permutedims to match legacy behavior (transpose from Python orientation)
            block_perm = permutedims(block)
            
            nrows, ncols = size(block_perm)
            
            if nrows != norb_src || ncols != norb_dst
                @warn "Block dimension mismatch for $key: expected ($norb_src, $norb_dst), got ($nrows, $ncols)"
                continue
            end

            for c in 1:ncols
                for r in 1:nrows
                    val = block_perm[r, c]
                    if abs(val) > 1e-12 
                        push!(rows, orb_start_src + r)
                        push!(cols, orb_start_dst + c)
                        push!(vals, val)
                    end
                end
            end
        end
    end

    process_blocks!(h_blocks, H_data)
    process_blocks!(s_blocks, S_data)

    H_R = Dict{Vector{Int}, SparseMatrixCSC{ComplexF64, Int}}()
    S_R = Dict{Vector{Int}, SparseMatrixCSC{ComplexF64, Int}}()

    for (R, (rows, cols, vals)) in H_data
        H_R[R] = sparse(rows, cols, vals, norbits, norbits)
    end
    
    for (R, (rows, cols, vals)) in S_data
        S_R[R] = sparse(rows, cols, vals, norbits, norbits)
    end

    return H_R, S_R
end

end # module
