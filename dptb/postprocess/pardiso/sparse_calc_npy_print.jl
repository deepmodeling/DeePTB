#!/usr/bin/env julia
using Printf
using DelimitedFiles, LinearAlgebra, JSON
using HDF5
using ArgParse
using SparseArrays
using Pardiso, Arpack, LinearMaps
using JLD
using FileIO
using Dates

const ev2Ry = 0.07349864879716558
const default_dtype = Complex{Float64}

# ==================== Argument Parsing ====================
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--output_dir", "-o"
            help = "Output directory for results"
            arg_type = String
            default = "./results"
        "--config"
            help = "Task configuration file (JSON)"
            arg_type = String
            default = "./band.json"
        "--input_dir", "-i"
            help = "Input directory containing data files"
            arg_type = String
            default = "./input_data"
        "--ill_project"
            help = "Enable ill-conditioned projection"
            arg_type = Bool
            default = true
        "--ill_threshold"
            help = "Threshold for ill-conditioned projection"
            arg_type = Float64
            default = 5e-4
    end
    return parse_args(s)
end

# ==================== Logging Helper ====================
function tee_log(msg::String, log_path::String)
    # Print to console (stdout)
    println(msg)
    # Append to log file
    open(log_path, "a") do io
        println(io, msg)
    end
end

function tee_info(msg::String, log_path::String)
    # Log as info to console
    @info msg
    # Log plain text to file
    open(log_path, "a") do io
        println(io, "[Info] $msg")
    end
end

# ==================== Helper Functions ====================
function _load_h5_to_dict(filename::String, log_path::String)
    tee_info("Reading HDF5 file: $filename", log_path)
    fid = h5open(filename, "r")
    d_out = Dict{Array{Int64,1}, Array{default_dtype, 2}}()
    
    function parse_key(k::AbstractString)
        if occursin("_", k) && !occursin("(", k)
             return parse.(Int64, split(k, "_"))
        elseif occursin("(", k)
             return map(x -> parse(Int64, convert(String, x)), split(k[2:length(k)-1], ','))
        else
             return Int64[]
        end
    end

    ks = keys(fid)
    if "0" in ks
        src = read(fid["0"])
        for (k_str, data) in src
             nk = parse_key(k_str)
             if !isempty(nk)
                 d_out[nk] = permutedims(data)
             end
        end
    else
        for key in ks
            data = read(fid[key])
            nk = parse_key(key)
            if !isempty(nk)
                d_out[nk] = permutedims(data)
            end
        end
    end

    close(fid)
    @info "Successfully loaded $(length(d_out)) matrix blocks"
    # Note: caller will log succes count if needed, or we tee_info here but keeping it simple
    return d_out
end

function k_data2num_ks(kdata::AbstractString)
    return parse(Int64, split(kdata)[1])
end

function k_data2kpath(kdata::AbstractString)
    return map(x->parse(Float64,x), split(kdata)[2:7])
end

function k_data2weightkpt(kdata::AbstractString)
    return parse(Float64, split(kdata)[8])
end

function k_data2labels(kdata::AbstractString)
    parts = split(kdata)
    if length(parts) >= 10
        return String(parts[9]), String(parts[10])
    else
        return "", ""
    end
end

function std_out_array(a::AbstractArray)
    return join([@sprintf("%.10f ", x) for x in a])
end

function reciprocal_2_cartessian(lat, kx, ky, kz)
    ax, bx, cx = lat[1,1], lat[1,2], lat[1,3]
    ay, by, cy = lat[2,1], lat[2,2], lat[2,3]
    az, bz, cz = lat[3,1], lat[3,2], lat[3,3]
    v1 = ax*(by*cz - bz*cy) + ay*(bz*cx - bx*cz) + az*(bx*cy - by*cx)
    v2 = 2π / v1
    a1 = v2*(by*cz - bz*cy); b1 = v2*(cy*az - cz*ay); c1 = v2*(ay*bz - az*by)
    a2 = v2*(bz*cx - bx*cz); b2 = v2*(cz*ax - cx*az); c2 = v2*(az*bx - ax*bz)
    a3 = v2*(bx*cy - by*cx); b3 = v2*(cx*ay - cy*ax); c3 = v2*(ax*by - ay*bx)
    return [kx*a1 + ky*b1 + kz*c1, kx*a2 + ky*b2 + kz*c2, kx*a3 + ky*b3 + kz*c3] ./ 2π
end

# ==================== K-Path Parsing ====================
function parse_kpath_abacus(kpath_config, lat, labels=String[])
    klist_vec = Vector{Vector{Float64}}()
    xlist = Float64[]
    high_sym_kpts = Float64[0.0]
    klabels_vec = String[] 
    
    if !isempty(labels)
        append!(klabels_vec, labels)
    end
    
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

function parse_kpath_vasp(k_data, lat)
    num_ks = k_data2num_ks.(k_data)
    kpaths = k_data2kpath.(k_data)
    
    klist_vec = Vector{Vector{Float64}}()
    xlist_vec = Vector{Float64}()
    high_sym_kpoints_vec = Float64[]
    klabels_vec = String[]
    
    k_length_total = 0.0
    
    for ipath in 1:length(kpaths)
        kpath = kpaths[ipath]
        lbls = k_data2labels(k_data[ipath])
        
        if ipath == 1
            push!(high_sym_kpoints_vec, k_length_total)
            push!(klabels_vec, lbls[1])
        elseif !isempty(klabels_vec) && klabels_vec[end] != lbls[1]
             push!(high_sym_kpoints_vec, k_length_total)
             push!(klabels_vec, lbls[1])
        end
        
        npts = num_ks[ipath]
        ks_rs = reciprocal_2_cartessian(lat, kpath[1:3]...)
        ke_rs = reciprocal_2_cartessian(lat, kpath[4:6]...)
        seg_length = norm(ke_rs - ks_rs)
        delta_k = seg_length / npts * 2π 
        
        kxs = LinRange(kpath[1], kpath[4], npts)
        kys = LinRange(kpath[2], kpath[5], npts)
        kzs = LinRange(kpath[3], kpath[6], npts)
        
        for ipt in 1:npts
            kpt = [kxs[ipt], kys[ipt], kzs[ipt]]
            push!(klist_vec, kpt)
            push!(xlist_vec, k_length_total) 
            k_length_total += delta_k
        end
        
        push!(high_sym_kpoints_vec, k_length_total)
        push!(klabels_vec, lbls[2])
    end
    
    return klist_vec, xlist_vec, high_sym_kpoints_vec, klabels_vec
end

# ==================== Eigensolver Implementation ====================
function construct_linear_map(H, S)
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
    return (
        LinearMap{ComplexF64}(
            (y, x) -> begin
                set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
                pardiso(ps, y, H_pardiso, S * x)
            end,
            size(H, 1);
            ismutating = true
        ),
        ps
    )
end

function solve_eigen_at_k(H_k, S_k, fermi_level, num_band, max_iter, out_wfc, ill_project, ill_threshold)
    if ill_project
        lm, ps = construct_linear_map(Hermitian(H_k) - fermi_level * Hermitian(S_k), Hermitian(S_k))
        
        if out_wfc
            egval_inv, egvec_sub = eigs(lm, nev = num_band, which = :LM, ritzvec = true, maxiter = max_iter)
        else
            egval_inv = eigs(lm, nev = num_band, which = :LM, ritzvec = false, maxiter = max_iter)[1]
            egvec_sub = zeros(default_dtype, size(H_k,1), 0)
        end
        set_phase!(ps, Pardiso.RELEASE_ALL); pardiso(ps)
        
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
                egvec = hcat(egvec_good, zeros(default_dtype, size(H_k,1), num_band - n_good))
            else
                egvec = egvec_sub
            end
        else
            egvec = zeros(default_dtype, size(H_k,1), 0)
        end
    else
        lm, ps = construct_linear_map(Hermitian(H_k) - fermi_level * Hermitian(S_k), Hermitian(S_k))
        if out_wfc
            egval_inv, egvec = eigs(lm, nev = num_band, which = :LM, ritzvec = true, maxiter = max_iter)
            egval = real(1 ./ egval_inv) .+ fermi_level
        else
            egval_inv = eigs(lm, nev = num_band, which = :LM, ritzvec = false, maxiter = max_iter)[1]
            egval = real(1 ./ egval_inv) .+ fermi_level
            egvec = zeros(default_dtype, size(H_k,1), 0)
        end
        set_phase!(ps, Pardiso.RELEASE_ALL); pardiso(ps)
    end
    
    perm = sortperm(egval)
    egval_sorted = egval[perm]
    
    if out_wfc && size(egvec, 2) > 0
        egvec_sorted = egvec[:, perm]
        return egval_sorted, egvec_sorted, 0.0
    else
        return egval_sorted, zeros(default_dtype, size(H_k,1), 0), 0.0
    end
end

# ==================== Modular Loading Functions ====================
function load_system_info(input_dir, spinful, log_path)
    site_positions = readdlm(joinpath(input_dir, "positions.dat")) |> permutedims
    atomic_numbers = readdlm(joinpath(input_dir, "atomic_numbers.dat"), Int)[:]
    
    basis_str = replace(read(joinpath(input_dir, "basis.dat"), String), "'" => "\"")
    basis_dict = JSON.parse(basis_str)
    
    z_to_symbol = Dict(
        1=>"H", 2=>"He", 3=>"Li", 4=>"Be", 5=>"B", 6=>"C", 7=>"N", 8=>"O", 9=>"F", 10=>"Ne",
        11=>"Na", 12=>"Mg", 13=>"Al", 14=>"Si", 15=>"P", 16=>"S", 17=>"Cl", 18=>"Ar", 19=>"K", 20=>"Ca",
        21=>"Sc", 22=>"Ti", 23=>"V", 24=>"Cr", 25=>"Mn", 26=>"Fe", 27=>"Co", 28=>"Ni", 29=>"Cu", 30=>"Zn",
        31=>"Ga", 32=>"Ge", 33=>"As", 34=>"Se", 35=>"Br", 36=>"Kr", 37=>"Rb", 38=>"Sr", 39=>"Y", 40=>"Zr",
        41=>"Nb", 42=>"Mo", 43=>"Tc", 44=>"Ru", 45=>"Rh", 46=>"Pd", 47=>"Ag", 48=>"Cd", 49=>"In", 50=>"Sn",
        51=>"Sb", 52=>"Te", 53=>"I", 54=>"Xe", 55=>"Cs", 56=>"Ba", 72=>"Hf", 73=>"Ta", 74=>"W", 75=>"Re", 
        76=>"Os", 77=>"Ir", 78=>"Pt", 79=>"Au", 80=>"Hg", 81=>"Tl", 82=>"Pb", 83=>"Bi", 84=>"Po"
    )

    function count_orbitals(basis)
        l_map = Dict('s'=>0, 'p'=>1, 'd'=>2, 'f'=>3, 'g'=>4)
        total = 0
        for m in eachmatch(r"(\d+)([spdfg])", basis)
            count = parse(Int, m.captures[1])
            l = l_map[m.captures[2][1]]
            total += count * (2l + 1)
        end
        return total
    end

    site_norbits = [count_orbitals(basis_dict[z_to_symbol[z]]) * (1 + spinful) for z in atomic_numbers]
    norbits = sum(site_norbits)
    lat = readdlm(joinpath(input_dir, "cell.dat")) |> permutedims
    
    return site_positions, atomic_numbers, site_norbits, norbits, lat
end

function load_sparse_hamiltonian(input_dir, site_norbits, norbits, spinful, log_path)
    sparse_file = joinpath(input_dir, "sparse_matrix.jld")
    if isfile(sparse_file)
        tee_info("Loading sparse matrices from cache", log_path)
        return load(sparse_file, "H_R"), load(sparse_file, "S_R")
    end

    ham_h5 = _load_h5_to_dict(joinpath(input_dir, "predicted_hamiltonians.h5"), log_path)
    tee_info("Successfully loaded $(length(ham_h5)) matrix blocks", log_path)
    overlap_h5 = _load_h5_to_dict(joinpath(input_dir, "predicted_overlaps.h5"), log_path)
    tee_info("Successfully loaded $(length(overlap_h5)) matrix blocks", log_path)
    
    I_R = Dict{Vector{Int64}, Vector{Int64}}()
    J_R = Dict{Vector{Int64}, Vector{Int64}}()
    H_V_R = Dict{Vector{Int64}, Vector{default_dtype}}()
    S_V_R = Dict{Vector{Int64}, Vector{default_dtype}}()
    
    site_norbits_cumsum = cumsum(site_norbits)
    
    for key in keys(ham_h5)
        Hblock = ham_h5[key]
        Sblock = get(overlap_h5, key, zero(Hblock))
        if spinful
            Sblock = vcat(hcat(Sblock, zeros(size(Sblock))), hcat(zeros(size(Sblock)), Sblock))
        end
        
        i, j = key[1]+1, key[2]+1
        R = key[3:5]
        
        if !haskey(I_R, R)
            I_R[R], J_R[R], H_V_R[R], S_V_R[R] = Int64[], Int64[], default_dtype[], default_dtype[]
        end
        
        for bi in 1:site_norbits[i], bj in 1:site_norbits[j]
            ii = site_norbits_cumsum[i] - site_norbits[i] + bi
            jj = site_norbits_cumsum[j] - site_norbits[j] + bj
            push!(I_R[R], ii); push!(J_R[R], jj)
            push!(H_V_R[R], Hblock[bi,bj]); push!(S_V_R[R], Sblock[bi,bj])
        end
    end
    
    H_R = Dict(R => sparse(I_R[R], J_R[R], H_V_R[R], norbits, norbits) for R in keys(I_R))
    S_R = Dict(R => sparse(I_R[R], J_R[R], S_V_R[R], norbits, norbits) for R in keys(I_R))
    
    save(sparse_file, "H_R", H_R, "S_R", S_R)
    save(sparse_file, "H_R", H_R, "S_R", S_R)
    tee_info("Sparse matrices constructed and cached", log_path)
    return H_R, S_R
end

# ==================== Core Calculation Logic ====================
function run_band_calculation(config, H_R, S_R, lat, norbits, output_dir, solver_opts, log_path)
    kline_type = get(config, "kline_type", "abacus")
    num_band = get(config, "num_band", 8)
    fermi_level = get(config, "E_fermi", 0.0)
    max_iter = get(config, "max_iter", 300)
    out_wfc = get(config, "out_wfc", "false") == "true"
    which_k = get(config, "which_k", 0)
    
    if kline_type == "abacus"
        kpath_cfg = config["kpath"]
        nk_total = sum(Int(row[4]) for row in kpath_cfg[1:end-1]) + 1
        klabels_input = string.(get(config, "klabels", String[]))
        klist, xlist, high_sym, klabels = parse_kpath_abacus(kpath_cfg, lat, klabels_input)
    else
        k_data = config["kpath"]
        nk_total = sum(k_data2num_ks.(k_data))
        klist, xlist, high_sym, klabels = parse_kpath_vasp(k_data, lat)
    end
    
    
    tee_info("Starting Band Structure Calculation", log_path)
    tee_info("Total K-points: $nk_total, Bands: $num_band, Fermi: $fermi_level eV", log_path)


    
    eigenval_file = joinpath(output_dir, "EIGENVAL")
    
    # Initialize EIGENVAL (VASP Format)
    open(eigenval_file, "w") do f
        @printf(f, "%5i%5i%5i%5i\n", 0, 0, 1, 1) # Simple header
        @printf(f, "%15.7E%15.7E%15.7E%15.7E%15.7e\n", 0.0, 0.0, 0.0, 0.0, 0.0)
        @printf(f, "%19.15E\n", 0.0)
        @printf(f, "%5s\n", "CAR")
        @printf(f, "%15s\n", "DeepTB System")
        @printf(f, "%7i%7i%7i\n", 0, nk_total, num_band)
    end

    all_egvals = Vector{Vector{Float64}}()
    
    start_time = time()
    for (ik, kpt) in enumerate(klist)
        if which_k != 0 && which_k != ik; continue; end
        
        H_k, S_k = spzeros(default_dtype, norbits, norbits), spzeros(default_dtype, norbits, norbits)
        for R in keys(H_R)
            phase = exp(im * 2π * dot(kpt, R))
            H_k += H_R[R] * phase
            S_k += S_R[R] * phase
        end
        H_k, S_k = (H_k + H_k') / 2, (S_k + S_k') / 2
        
        egvals, egvecs, _ = solve_eigen_at_k(H_k, S_k, fermi_level, num_band, max_iter, out_wfc, 
                                            solver_opts.ill_project, solver_opts.ill_threshold)
        push!(all_egvals, egvals)
        

        
        # Save to EIGENVAL
        open(eigenval_file, "a") do f
            println(f)
            @printf(f, "%15.7E%15.7E%15.7E%15.7e\n", kpt..., 0.0) 
            for (ib, eb) in enumerate(egvals); @printf(f, "%5i%16.6f%11.6f\n", ib, eb - fermi_level, 0.0); end
        end
        
        if out_wfc && size(egvecs, 2) == num_band
            k_rs = reciprocal_2_cartessian(lat, kpt...)
            open(joinpath(output_dir, "LOWF_K_$ik.dat"), "w") do io
                @printf(io, "%i (index of k points)\n", ik)
                println(io, std_out_array(k_rs))
                @printf(io, "%i (number of bands)\n", num_band)
                @printf(io, "%i (number of orbitals)\n", norbits)
                for ib in 1:num_band
                    @printf(io, "%i (band)\n", ib)
                    @printf(io, "%.23e (Ry)\n", egvals[ib] * ev2Ry)
                    @printf(io, "%.23e (Occupations)\n", 0.0)
                    for iw in 1:norbits
                        @printf(io, "%.23e %.23e ", real(egvecs[iw,ib]), imag(egvecs[iw,ib]))
                        if iw % 5 == 0 && iw < norbits; println(io); end
                    end
                    println(io)
                end
            end
        end
        
        if ik % 10 == 0 || ik == length(klist)
            msg = @sprintf("K-point %4d/%d done | Elapsed: %.2f min", ik, length(klist), (time()-start_time)/60)
            tee_log(msg, log_path)
        end
    end
    # Handle NPY export
    save_bandstructure_npy(klist, xlist, all_egvals, fermi_level, high_sym, klabels, output_dir, log_path)
end

function run_dos_calculation(config, H_R, S_R, norbits, output_dir, solver_opts, log_path)
    nkmesh = Int64.(config["kmesh"])
    nk_total = prod(nkmesh)
    num_band = get(config, "num_band", 8)
    fermi_level = get(config, "fermi_level", 0.0)
    max_iter = get(config, "max_iter", 300)
    
    tee_info("Starting Density of States Calculation", log_path)
    tee_info("K-Grid: $nkmesh (Total: $nk_total), Bands: $num_band, Fermi: $fermi_level eV", log_path)

    ks = zeros(3, nk_total)
    ik = 1
    for ix in 1:nkmesh[1], iy in 1:nkmesh[2], iz in 1:nkmesh[3]
        ks[:,ik] .= [(ix-1)/nkmesh[1], (iy-1)/nkmesh[2], (iz-1)/nkmesh[3]]
        ik += 1
    end
    
    egvals_all = zeros(num_band, nk_total)
    start_time = time()
    for ik in 1:nk_total
        kpt = ks[:,ik]
        H_k, S_k = spzeros(default_dtype, norbits, norbits), spzeros(default_dtype, norbits, norbits)
        for R in keys(H_R)
            phase = exp(im * 2π * dot(kpt, R))
            H_k += H_R[R] * phase
            S_k += S_R[R] * phase
        end
        H_k, S_k = (H_k + H_k')/2, (S_k + S_k')/2
        
        egval, _, _ = solve_eigen_at_k(H_k, S_k, fermi_level, num_band, max_iter, false, 
                                      solver_opts.ill_project, solver_opts.ill_threshold)
        egvals_all[:,ik] = egval
        
        if ik % 50 == 0 || ik == nk_total
            msg = @sprintf("DOS K-point %5d/%d done | Elapsed: %.2f min", ik, nk_total, (time()-start_time)/60)
            tee_log(msg, log_path)
        end
    end
    
    writedlm(joinpath(output_dir, "egvals.dat"), egvals_all)
    
    if haskey(config, "epsilon") && haskey(config, "omegas")
        ϵ = config["epsilon"]
        ω_cfg = config["omegas"]
        ωlist = collect(range(ω_cfg[1], ω_cfg[2], length = Int(ω_cfg[3])))
        dos = zeros(length(ωlist))
        # Gaussian broadening factor
        factor = 1.0 / (ϵ * sqrt(pi) * nk_total)
        for ik in 1:nk_total, ib in 1:num_band, iω in eachindex(ωlist)
            diff = egvals_all[ib,ik] - ωlist[iω] - fermi_level
            dos[iω] += exp(-(diff^2 / ϵ^2)) * factor
        end
        open(joinpath(output_dir, "dos.dat"), "w") do f
            for (ω, d) in zip(ωlist, dos)
                @printf(f, "%12.6f  %12.6f\n", ω - fermi_level, d)
            end
        end
        tee_info("Generated dos.dat", log_path)
    end
end

function save_bandstructure_npy(klist, xlist, eigenvalues, e_fermi, high_sym, labels, output_dir, log_path)
    try
        data = Dict(
            "klist" => klist, "xlist" => xlist, "eigenvalues" => eigenvalues,
            "E_fermi" => e_fermi, "high_sym_kpoints" => high_sym, "labels" => labels,
            "output_path" => joinpath(output_dir, "bandstructure.npy")
        )
        temp_json = joinpath(output_dir, "temp_band_data.json")
        open(temp_json, "w") do f; JSON.print(f, data); end
        
        py_script = """
import json, numpy as np
with open('$temp_json', 'r') as f: data = json.load(f)
out = {k: np.array(v) if k != 'labels' and k != 'output_path' else v for k, v in data.items()}
np.save(data['output_path'], out)
"""
        py_file = joinpath(output_dir, "convert_npy.py")
        write(py_file, py_script)
        run(`python3 $py_file`)
        rm(temp_json; force=true); rm(py_file; force=true)
        tee_info("Generated bandstructure.npy", log_path)
    catch e
        @warn "Failed to generate bandstructure.npy: $e"
        tee_log("[Warn] Failed to generate bandstructure.npy: $e", log_path)
    end
end
function main()
    args = parse_commandline()
    begin_time = time()
    
    # Init logging
    output_dir = args["output_dir"]
    mkpath(output_dir)
    log_path = joinpath(output_dir, "log.dat")
    open(log_path, "w") do io
        println(io, "="^70)
        println(io, "Sparse Hamiltonian Calculation Task Summary")
        println(io, "Start Time: ", Dates.now())
        println(io, "Output Dir: ", output_dir)
        println(io, "="^70)
    end

    # Load configurations
    config = JSON.parsefile(args["config"])
    if haskey(config, "task_options"); merge!(config, config["task_options"]); end
    input_dir = args["input_dir"]
    
    # Set default values
    calc_job = get(config, "calc_job", "band")
    spinful = haskey(config, "isspinful") ? (config["isspinful"] in [true, "true"]) : false
    
    
    # Load system info
    pos, atomic_nums, site_norbs, norbits, lat = load_system_info(input_dir, spinful, log_path)
    
    tee_log("Sites: $(length(atomic_nums)), Orbitals: $norbits, Spinful: $spinful", log_path)

    # Load/Build Hamiltonians
    H_R, S_R = load_sparse_hamiltonian(input_dir, site_norbs, norbits, spinful, log_path)

    solver_opts = (ill_project = args["ill_project"], ill_threshold = args["ill_threshold"])
    
    if calc_job == "band"
        run_band_calculation(config, H_R, S_R, lat, norbits, output_dir, solver_opts, log_path)
    elseif calc_job == "dos"
        run_dos_calculation(config, H_R, S_R, norbits, output_dir, solver_opts, log_path)
    end
    
    total_min = (time() - begin_time) / 60
    msg = @sprintf("Task completed successfully in %.2f minutes", total_min)
    tee_log(msg, log_path)
    println("Task finished. Total time: $(round(total_min, digits=2)) min")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
