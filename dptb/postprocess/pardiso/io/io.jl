"""
Data I/O utilities for DeePTB Julia backend.
"""
module DataIO

using JSON
using HDF5
using SparseArrays
using DelimitedFiles

export load_structure, load_matrix_hdf5

const ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

function element_symbol(z::Integer)
    if 1 <= z <= length(ELEMENT_SYMBOLS)
        return ELEMENT_SYMBOLS[z]
    end
    error("Unsupported atomic number in legacy Pardiso input: $z")
end

"""
    load_structure(input_dir::String; spinful::Bool=false)

Load structure information from structure.json file (preferred) or legacy .dat files.

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

"""
    load_structure(input_dir::String; spinful::Bool=false)

Load structure information from `structure.json` (preferred) or legacy `.dat` files.
Dispatches to specific loading functions based on file availability.
"""
function load_structure(input_dir::String; spinful::Bool=false)
    if isfile(joinpath(input_dir, "structure.json"))
        return load_structure_json(input_dir)
    elseif isfile(joinpath(input_dir, "positions.dat"))
        return load_structure_dat(input_dir, spinful)
    else
        error("Structure information not found in $input_dir. Expected structure.json or legacy .dat files.")
    end
end

"""
    load_structure_json(input_dir::String)

Load structure from modern `structure.json` format.
"""
function load_structure_json(input_dir::String)
    json_path = joinpath(input_dir, "structure.json")
    data = JSON.parsefile(json_path)

    # We assume valid schema v1.0 as legacy format is no longer supported
    if !haskey(data, "structure") || !haskey(data, "basis_info")
         error("Invalid structure.json format. Expected keys 'structure' and 'basis_info'.")
    end

    struc = data["structure"]
    basis = data["basis_info"]
    
    # Prefer explicit per-site symbols and orbital counts from the JSON schema.
    # Falling back to chemical_formula loses site ordering for mixed-element structures.
    symbols = haskey(struc, "symbols") ? String.(struc["symbols"]) : expand_species(struc["chemical_formula"])
    site_norbits = haskey(basis, "site_norbits") ? Int.(basis["site_norbits"]) : begin
        orb_counts = basis["orbital_counts"]
        spinful = basis["spinful"]
        [orb_counts[sym] * (1 + spinful) for sym in symbols]
    end
    norbits = haskey(basis, "total_orbitals") ? Int(basis["total_orbitals"]) : sum(site_norbits)

    structure = Dict{String, Any}(
        "cell" => permutedims(hcat(struc["cell"]...)),
        "positions" => permutedims(hcat(struc["positions"]...)),
        "site_norbits" => site_norbits,
        "norbits" => norbits,
        "symbols" => symbols,
        "natoms" => struc["nsites"],
        "spinful" => basis["spinful"],
        "basis" => basis["basis"]
    )
    
    @info "Loaded structure from JSON: $(structure["natoms"]) atoms, $(structure["norbits"]) orbitals"
    return structure
end

"""
    load_structure_dat(input_dir::String, spinful::Bool)

Load structure from legacy `.dat` files (positions.dat, basis.dat, etc.).
"""
function load_structure_dat(input_dir::String, spinful::Bool)
    @info "structure.json not found, falling back to legacy .dat files"
    
    # Load data using DelimitedFiles
    # Note: readdlm reads into matrix (rows x cols). 
    # vasp/legacy format usually stores as (natoms x 3).
    # DeepTB structure expects rows = atoms.
    site_positions = readdlm(joinpath(input_dir, "positions.dat"))
    atomic_numbers = readdlm(joinpath(input_dir, "atomic_numbers.dat"), Int)[:]
    lat = readdlm(joinpath(input_dir, "cell.dat"))
    
    # Parse basis info
    basis_str = replace(read(joinpath(input_dir, "basis.dat"), String), "'" => "\"")
    basis_dict = JSON.parse(basis_str)
    
    l_map = Dict('s'=>0, 'p'=>1, 'd'=>2, 'f'=>3, 'g'=>4)
    function count_orbitals(basis)
        total = 0
        for m in eachmatch(r"([0-9]+)([spdfg])", basis)
            count = parse(Int, m.captures[1])
            l = l_map[m.captures[2][1]]
            total += count * (2l + 1)
        end
        return total
    end

    symbols = [element_symbol(z) for z in atomic_numbers]
    site_norbits = [count_orbitals(basis_dict[sym]) * (1 + spinful) for sym in symbols]
    norbits = sum(site_norbits)
    
    structure = Dict{String, Any}(
        "cell" => lat,
        "positions" => site_positions,
        "site_norbits" => site_norbits,
        "norbits" => norbits,
        "symbols" => symbols,
        "natoms" => length(atomic_numbers),
        "spinful" => spinful,
        "basis" => basis_dict
    )

    @info "Loaded structure from legacy .dat files: $(structure["natoms"]) atoms, $(structure["norbits"]) orbitals"
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
