"""
K-point generation utilities.
"""
module KPoints

using LinearAlgebra

export parse_kpath, gen_kmesh

"""
    parse_kpath(kpath_config, lat, labels)

Parse k-path in ABACUS format.

# Returns
- `klist::Vector{Vector{Float64}}`: List of k-points
- `xlist::Vector{Float64}`: Cumulative distances
- `high_sym_kpts::Vector{Float64}`: High-symmetry point positions
- `klabels::Vector{String}`: K-point labels
"""
function parse_kpath(kpath_config, lat, labels=String[])
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
    gen_kmesh(nkmesh)

Generate a uniform K-point mesh.

# Arguments
- `nkmesh::Vector{Int}`: Number of k-points in each direction [nx, ny, nz]

# Returns
- `ks::Matrix{Float64}`: Matrix of size [3, nk_total] containing k-points
- `nk_total::Int`: Total number of k-points
"""
function gen_kmesh(nkmesh)
    nk_total = prod(nkmesh)
    ks = zeros(3, nk_total)
    ik = 1
    for ix in 1:nkmesh[1], iy in 1:nkmesh[2], iz in 1:nkmesh[3]
        ks[:,ik] .= [(ix-1)/nkmesh[1], (iy-1)/nkmesh[2], (iz-1)/nkmesh[3]]
        ik += 1
    end
    return ks, nk_total
end

end # module
