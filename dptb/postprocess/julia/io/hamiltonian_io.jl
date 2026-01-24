module HamiltonianIO

export load_hamiltonian_hdf5, load_overlap_hdf5, construct_sparse_matrices, get_hamiltonian_and_overlap

using HDF5
using SparseArrays
using Serialization

"""
Load Hamiltonian or overlap matrix blocks from an HDF5 file.
"""
function load_matrix_hdf5(filename::String)
    if !isfile(filename)
        # If file doesn't exist (e.g. overlap), return empty dict
        return Dict{String, Any}()
    end

    h5open(filename, "r") do file
        blocks = Dict{String, Any}()
        # The python script saves blocks inside group "0" (for first model)
        if "0" in keys(file)
            grp = file["0"]
            for key in keys(grp)
                blocks[key] = read(grp, key)
            end
        else
            # Reading keys from root if "0" group not present
            for key in keys(file)
                if isa(file[key], HDF5.Dataset)
                    blocks[key] = read(file, key)
                end
            end
        end
        return blocks
    end
end

load_hamiltonian_hdf5(input_dir::String) = load_matrix_hdf5(joinpath(input_dir, "predicted_hamiltonians.h5"))
load_overlap_hdf5(input_dir::String) = load_matrix_hdf5(joinpath(input_dir, "predicted_overlaps.h5"))

"""
Construct sparse matrices H(R) and S(R) from raw blocks.

Returns:
    H_R: Dict{Vector{Int}, SparseMatrixCSC}
    S_R: Dict{Vector{Int}, SparseMatrixCSC}
"""
function construct_sparse_matrices(h_blocks::Dict{String, Any}, s_blocks::Dict{String, Any}, 
                                 norbits::Int, site_norbits::Vector{Int})
    
    # Precompute orbital start indices for each atom (1-based for Julia)
    # site_norbits is [norb_atom1, norb_atom2, ...]
    # atom_orb_starts[i] is the starting orbital index for atom i
    atom_orb_starts = cumsum(vcat([0], site_norbits))

    # We need to collect triplets (I, J, V) for each R vector
    # H_data[R] = (rows, cols, vals)
    H_data = Dict{Vector{Int}, Tuple{Vector{Int}, Vector{Int}, Vector{ComplexF64}}}()
    S_data = Dict{Vector{Int}, Tuple{Vector{Int}, Vector{Int}, Vector{ComplexF64}}}()
    
    # Function to process a set of blocks (H or S)
    function process_blocks!(blocks, data_dict)
        for (key, block) in blocks
            # Parse key "src_dst_rx_ry_rz"
            # Python indices 0-based, Julia 1-based
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
            
            # Block dimensions: (norb_src, norb_dst) in Python?
            # DeePTB stores blocks as [orb_src, orb_dst] (usually)
            # We need to verify if transposition is needed. 
            # Assuming standard DeePTB: block[i, j] corresponds to <src_i | H | dst_j>
            # which maps to matrix element H[global_src_i, global_dst_j]
            
            # Since Julia is column-major and Python is row-major, 'read' might transpose?
            # HDF5.jl read usually preserves shape. 
            # If Python block is (N_src, N_dst), Julia reads it as (N_src, N_dst) or (N_dst, N_src)?
            # HDF5 in Julia reads arrays in column-major order.
            # A (rows, cols) array in Python (C-order) becomes (cols, rows) in Julia (F-order) if not carefully handled.
            # However, `dptb_to_pardiso.ipynb` suggests `permutedims` in old script.
            
            # Old script: d_out[nk] = permutedims(data)
            # Let's apply permutedims to match legacy behavior
            
            block_perm = permutedims(block) # Transpose to match Julia's expected orientation
            
            # Iterate over the block
            # dims of block_perm should be (norb_src, norb_dst)
            nrows, ncols = size(block_perm)
            
            if nrows != norb_src || ncols != norb_dst
                @warn "Block dimension mismatch for $key: expected ($norb_src, $norb_dst), got ($nrows, $ncols)"
                continue
            end

            for c in 1:ncols       # dst orbital index (local)
                for r in 1:nrows   # src orbital index (local)
                    val = block_perm[r, c]
                    if abs(val) > 1e-12 # Optional sparsity filter
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

    # Convert to SparseMatrixCSC
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

"""
Main function to get Hamiltonian and Overlap matrices, with caching.
"""
function get_hamiltonian_and_overlap(input_dir::String, structure::Dict, use_cache::Bool=true)
    cache_file = joinpath(input_dir, "sparse_matrices.jld")

    if use_cache && isfile(cache_file)
        println("Loading H/S matrices from cache: $cache_file")
        try
            data = deserialize(cache_file)
            # data is expected to be (H_R, S_R)
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
    h_blocks = load_hamiltonian_hdf5(input_dir)
    s_blocks = load_overlap_hdf5(input_dir)
    
    println("Building sparse Hamiltonian matrices...")
    H_R, S_R = construct_sparse_matrices(h_blocks, s_blocks, structure["norbits"], structure["site_norbits"])

    if use_cache
        println("Saving sparse H/S matrices to cache: $cache_file")
        serialize(cache_file, (H_R, S_R))
    end

    return H_R, S_R
end

end # module