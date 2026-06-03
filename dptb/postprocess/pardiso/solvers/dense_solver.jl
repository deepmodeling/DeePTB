"""
Dense linear algebra solver using standard LAPACK (via Julia's LinearAlgebra).

This solver is robust for small systems and serves as a fallback when
Pardiso is not available or efficient.
"""

module DenseSolver

using LinearAlgebra
using SparseArrays

export solve_eigen_k_dense

const default_dtype = Complex{Float64}

"""
    solve_eigen_k_dense(H_k, S_k, fermi_level, num_band, max_iter, out_wfc, ill_project, ill_threshold)

Solve eigenvalue problem using dense diagonalization (ZHEGV).

# Arguments
- `H_k`: Hamiltonian matrix (sparse or dense)
- `S_k`: Overlap matrix (sparse or dense)
- `fermi_level`: Center energy (used for sorting)
- `num_band`: Number of bands to return around fermi_level

# Returns
- `evals`: Eigenvalues
- `evecs`: Eigenvectors (if requested, currently empty for band calc)
- `residual`: 0.0
"""
function solve_eigen_k_dense(H_k, S_k, fermi_level, num_band, max_iter, out_wfc, ill_project, ill_threshold)
    # Convert to dense matrices
    H_dense = Matrix(H_k)
    S_dense = Matrix(S_k)
    
    # Ensure Hermitian
    H_dense = Hermitian(H_dense)
    S_dense = Hermitian(S_dense)
    
    if ill_project
        # 1. Diagonalize S first to find good subspace
        egval_S, egvec_S = eigen(S_dense)
        
        # 2. Filter good states
        project_index = abs.(egval_S) .> ill_threshold
        n_good = sum(project_index)
        
        if n_good < size(H_k, 1)
            # @warn "Ill-conditioned eigenvalues detected, projecting out $(size(H_k, 1) - n_good) states"
            egvec_S_good = egvec_S[:, project_index]
            
            # 3. Project H and S to the good subspace
            # H_k_proj = V' * H * V
            H_k_proj = egvec_S_good' * H_dense * egvec_S_good
            S_k_proj = egvec_S_good' * S_dense * egvec_S_good
            
            # 4. Solve the projected (smaller) problem
            # Using Hermitian wrapper to ensure real eigenvalues
            full_vals_proj, full_vecs_proj = eigen(Hermitian(H_k_proj), Hermitian(S_k_proj))
            
            # 5. Reconstruct full eigenvectors if needed
            # The "bad" states are set to high energy (fermi + 1e4)
            full_vals = vcat(full_vals_proj, fill(1e4 + fermi_level, size(H_k, 1) - n_good))
            
            if out_wfc
                # Recover eigenvectors in original basis: V_full = V_good * V_proj
                egvec_good = egvec_S_good * full_vecs_proj
                # Pad with zeros for projected-out states
                full_vecs = hcat(egvec_good, zeros(default_dtype, size(H_k, 1), size(H_k, 1) - n_good))
            else
                 # Placeholder if not needed
                 full_vecs = zeros(default_dtype, size(H_k, 1), 0)
            end
        else
            # No projection needed
            full_vals, full_vecs = eigen(H_dense, S_dense)
        end
    else
        full_vals, full_vecs = eigen(H_dense, S_dense)
    end
    
    # Select bands closest to Fermi level
    diff = abs.(full_vals .- fermi_level)
    
    # Get indices of 'num_band' closest eigenvalues
    perm = sortperm(diff)
    closest_indices = perm[1:min(length(perm), num_band)]
    
    # Sort indices to keep bands ordered by energy
    sort!(closest_indices)
    
    # Extract eigenvalues
    evals = full_vals[closest_indices]
    
    # Check if we have enough bands and pad if necessary
    if length(evals) < num_band
        n_pad = num_band - length(evals)
        pad_vals = fill(1e4 + fermi_level, n_pad)
        evals = vcat(evals, pad_vals)
    end
    
    # Handle eigenvectors output
    if out_wfc
        # Extract corresponding eigenvectors
        # Note: full_vecs might be empty if we skipped calculation in projection branch (though currently we calc or placeholder)
        if size(full_vecs, 2) > 0
             evecs = full_vecs[:, closest_indices]
             
             # Pad evecs if necessary
             if size(evecs, 2) < num_band
                 evecs = hcat(evecs, zeros(default_dtype, size(H_k, 1), num_band - size(evecs, 2)))
             end
        else
             evecs = zeros(default_dtype, size(H_k, 1), 0)
        end
    else
        evecs = zeros(default_dtype, size(H_k, 1), 0)
    end
    
    return evals, evecs, 0.0
end

end # module
