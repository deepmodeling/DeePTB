"""
I/O module for loading structure data from JSON format.

This module provides functions to load structure information exported
by DeePTB's to_pardiso_new() method.
"""

module StructureIO

using JSON

export load_structure_json

"""
    load_structure_json(input_dir::String)

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
structure = load_structure_json("pardiso_input")
println("Total orbitals: ", structure["norbits"])
```
"""
function load_structure_json(input_dir::String)
    json_path = joinpath(input_dir, "structure.json")

    if !isfile(json_path)
        error("structure.json not found in $input_dir")
    end

    data = JSON.parsefile(json_path)

    # Convert to Julia-friendly format
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

    @info "Loaded structure: $(structure["natoms"]) atoms, $(structure["norbits"]) orbitals"

    return structure
end

end # module
