"""
Pardiso-based eigenvalue solver for large-scale tight-binding systems.

This module provides efficient eigenvalue solvers using Intel MKL Pardiso
with shift-invert technique for better convergence.
"""
module PardisoSolver

using Pardiso
using Arpack
using LinearMaps
using LinearAlgebra

const default_dtype = Complex{Float64}

"""
    make_shift_invert_map(H::AbstractMatrix, S::AbstractMatrix)

Construct a linear map for shift-invert eigenvalue solver.
"""
function make_shift_invert_map(H::AbstractMatrix, S::AbstractMatrix)
    # Architecture check for MKL compatibility
    if Sys.isapple() && Sys.ARCH == :aarch64
        @warn "MKL Pardiso is not natively supported on Apple Silicon (M1/M2/etc)."
        @warn "For local testing on Mac, please set 'eig_solver' to 'numpy' (DenseSolver) in your config."
        error("MKLPardisoSolver failed: Architecture mismatch (Apple Silicon). Use 'numpy' solver or run on Intel/Linux machine.")
    end

    # Enforce MKL Pardiso Solver
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.COMPLEX_HERM_INDEF)
    pardisoinit(ps)
    fix_iparm!(ps, :N)

    H_pardiso = get_matrix(ps, H, :N)
    b = rand(ComplexF64, size(H, 1))

    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, H_pardiso, b)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, H_pardiso, b)

    lm = LinearMap{ComplexF64}(
        (y, x) -> begin
            set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
            pardiso(ps, y, H_pardiso, S * x)
        end,
        size(H, 1);
        ismutating=true
    )

    return lm, ps
end

"""
    solve_eigen_k_pardiso(H_k, S_k, fermi_level, num_band, max_iter, out_wfc, ill_project, ill_threshold)

Solve generalized eigenvalue problem H|ψ⟩ = E S|ψ⟩ at a single k-point using Pardiso shift-invert.
"""
function solve_eigen_k_pardiso(H_k, S_k, fermi_level, num_band, max_iter, out_wfc, ill_project, ill_threshold)
    if ill_project
        lm, ps = make_shift_invert_map(Hermitian(H_k) - fermi_level * Hermitian(S_k), Hermitian(S_k))

        if out_wfc
            egval_inv, egvec_sub = eigs(lm, nev=num_band, which=:LM, ritzvec=true, maxiter=max_iter)
        else
            egval_inv = eigs(lm, nev=num_band, which=:LM, ritzvec=false, maxiter=max_iter)[1]
            egvec_sub = zeros(default_dtype, size(H_k, 1), 0)
        end

        set_phase!(ps, Pardiso.RELEASE_ALL)
        pardiso(ps)

        egval = real(1 ./ egval_inv) .+ fermi_level

        if out_wfc && size(egvec_sub, 2) > 0
            egvec_sub = Matrix{default_dtype}(qr(egvec_sub).Q)
            S_k_sub = egvec_sub' * S_k * egvec_sub
            egval_S, egvec_S = eigen(Hermitian(Matrix(S_k_sub)))
            project_index = abs.(egval_S) .> ill_threshold
            n_good = sum(project_index)

            if n_good < num_band
                @warn "Ill-conditioned eigenvalues detected, projecting out $(num_band - n_good) states"
                H_k_sub = egvec_sub' * H_k * egvec_sub
                egvec_S_good = egvec_S[:, project_index]

                H_k_proj = egvec_S_good' * H_k_sub * egvec_S_good
                S_k_proj = egvec_S_good' * S_k_sub * egvec_S_good
                egval_proj, egvec_proj = eigen(Hermitian(Matrix(H_k_proj)), Hermitian(Matrix(S_k_proj)))

                egval = vcat(egval_proj, fill(1e4, num_band - n_good))
                egvec_good = egvec_sub * egvec_S_good * egvec_proj
                egvec = hcat(egvec_good, zeros(default_dtype, size(H_k, 1), num_band - n_good))
            else
                egvec = egvec_sub
            end
        else
            egvec = zeros(default_dtype, size(H_k, 1), 0)
        end
    else
        lm, ps = make_shift_invert_map(Hermitian(H_k) - fermi_level * Hermitian(S_k), Hermitian(S_k))

        if out_wfc
            egval_inv, egvec = eigs(lm, nev=num_band, which=:LM, ritzvec=true, maxiter=max_iter)
            egval = real(1 ./ egval_inv) .+ fermi_level
        else
            egval_inv = eigs(lm, nev=num_band, which=:LM, ritzvec=false, maxiter=max_iter)[1]
            egval = real(1 ./ egval_inv) .+ fermi_level
            egvec = zeros(default_dtype, size(H_k, 1), 0)
        end

        set_phase!(ps, Pardiso.RELEASE_ALL)
        pardiso(ps)
    end

    perm = sortperm(egval)
    egval_sorted = egval[perm]

    if out_wfc && size(egvec, 2) > 0
        egvec_sorted = egvec[:, perm]
        return egval_sorted, egvec_sorted, 0.0
    else
        return egval_sorted, zeros(default_dtype, size(H_k, 1), 0), 0.0
    end
end

export make_shift_invert_map, solve_eigen_k_pardiso

end # module