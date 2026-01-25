"""
Data I/O utilities for DeePTB Julia backend.
"""
module DataIO

using JSON
using HDF5
using SparseArrays

export load_structure, load_matrix_hdf5

"""
    load_structure(input_dir::String)

Load structure information from structure.json file.

# Arguments
- `input_dir::String`: Directory containing structure.json

# Returns
- `structure::Dict`: Dictionary containing:
  - `cell`: 3x3 lattice matrix
  - `positions`: natoms x 3 position matrix
  - `site_norbits`: Vector of orbital counts per atom
  - `norbits`: Total number of orbitals
  - `symbols`: Vector of element symbols
  - `natoms`: Number of atoms
  - `spinful`: Whether calculation is spin-polarized
  - `basis`: Basis set dictionary

# Example
```julia
structure = load_structure("pardiso_input")
println("Total orbitals: ", structure["norbits"])
```
"""
function load_structure(input_dir::String)
    json_path = joinpath(input_dir, "structure.json")

    if !isfile(json_path)
        error("structure.json not found in $input_dir")
    end

    # Parse JSON (DeePTB Schema v1.0)
    data = JSON.parsefile(json_path)

    # Check version or schema type if needed
    if haskey(data, "structure") && haskey(data, "basis_info")
        # Refined Schema (v1.0)
        struc = data["structure"]
        basis = data["basis_info"]
        
        # Expand symbols
        symbols = expand_species(struc["chemical_formula"])
        
        # Get pre-calculated counts
        orb_counts = basis["orbital_counts"]
        site_norbits = [orb_counts[sym] for sym in symbols]

        structure = Dict{String, Any}(
            "cell" => permutedims(hcat(struc["cell"]...)),
            "positions" => permutedims(hcat(struc["positions"]...)),
            "site_norbits" => site_norbits,
            "norbits" => basis["total_orbitals"],
            "symbols" => symbols,
            "natoms" => struc["nsites"],
            "spinful" => basis["spinful"],
            "basis" => basis["basis"]
        )
    else
        # Fallback to Old Flat Format (for backward compatibility if needed, or error out)
        # Assuming we might still have old files during dev
        @warn "Legacy structure.json format detected. Please regenerate using latest Python backend."
        structure = Dict{String, Any}(
            "cell" => permutedims(hcat(data["cell"]...)),
            "positions" => permutedims(hcat(data["positions"]...)),
            "site_norbits" => data["site_norbits"],
            "norbits" => data["norbits"],
            "symbols" => data["symbols"],
            "natoms" => data["natoms"],
            "spinful" => data["spinful"],
            "basis" => data["basis"]
        )
    end

    @info "Loaded structure: $(structure["natoms"]) atoms, $(structure["norbits"]) orbitals"

    return structure
end

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
"""
Expand chemical formula string (e.g. "C2H2") or list to list of symbols.
"""
function expand_species(species::AbstractString)
    symbols = String[]
    for m in eachmatch(r"([A-Z][a-z]?)(\d*)", species)
        elem = m.captures[1]
        count_str = m.captures[2]
        count = isempty(count_str) ? 1 : parse(Int, count_str)
        for _ in 1:count
            push!(symbols, elem)
        end
    end
    return symbols
end

expand_species(species::AbstractVector) = String.(species)

end # module
