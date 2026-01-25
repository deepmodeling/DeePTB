#!/usr/bin/env julia
"""
Main entry point for DeePTB Julia backend calculations.

This script provides a modular interface for running band structure
and DOS calculations using the Pardiso solver.

Usage:
    julia main.jl --input_dir ./pardiso_input --output_dir ./results --config ./band.json
"""

using Printf
using ArgParse
using JSON
using Dates

# Include modules
include("io/structure_io.jl")
using .StructureIO
include("io/hamiltonian_io.jl")
using .HamiltonianIO
include("solvers/pardiso_solver.jl")
include("solvers/dense_solver.jl")
using .DenseSolver
include("tasks/band_calculation.jl")
include("tasks/dos_calculation.jl")

# Import functions
using .StructureIO: load_structure_json
using .HamiltonianIO: construct_sparse_matrices
using .BandCalculation: run_band_calculation
using .DosCalculation: run_dos_calculation

# Make solver functions available to modules
# (solve_eigen_at_k is already in global scope from pardiso_solver.jl)

"""
    parse_commandline()

Parse command line arguments.
"""
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

"""
    main()

Main execution function.
"""
function main()
    args = parse_commandline()
    begin_time = time()

    # Initialize logging
    output_dir = args["output_dir"]
    mkpath(output_dir)
    log_path = joinpath(output_dir, "log.dat")

    open(log_path, "w") do io
        println(io, "="^70)
        println(io, "DeePTB Julia Backend Calculation")
        println(io, "Start Time: ", Dates.now())
        println(io, "Output Dir: ", output_dir)
        println(io, "="^70)
    end

    # Load configuration
    config = JSON.parsefile(args["config"])
    if haskey(config, "task_options")
        merge!(config, config["task_options"])
    end

    input_dir = args["input_dir"]
    calc_job = get(config, "calc_job", "band")

    # Load structure
    @info "Loading structure from $input_dir"
    structure = load_structure_json(input_dir)
    structure["site_norbits"] = Int.(structure["site_norbits"])

    # Load/construct sparse matrices
    # Load/construct sparse matrices
    @info "Constructing sparse Hamiltonian matrices"
    H_R, S_R = get_hamiltonian_and_overlap(input_dir, structure)

    # Solver options
    solver_opts = (
        ill_project = args["ill_project"],
        ill_threshold = args["ill_threshold"]
    )

    # Select solver
    eig_solver = get(config, "eig_solver", "pardiso")
    solver_func = nothing
    
    if eig_solver == "numpy" || eig_solver == "dense"
        @info "Using Dense Solver (LAPACK)"
        solver_func = solve_eigen_dense_at_k
    else
        @info "Using Pardiso Solver"
        # Check if Pardiso is actually available, otherwise warn/fallback?
        # For now assume user knows what they are doing if they didn't specify 'dense'
        solver_func = solve_eigen_at_k
    end

    # Run calculation
    if calc_job == "band"
        @info "Running band structure calculation"
        run_band_calculation(config, H_R, S_R, structure, output_dir, solver_opts, log_path, solver_func)
    elseif calc_job == "dos"
        @info "Running DOS calculation"
        run_dos_calculation(config, H_R, S_R, structure, output_dir, solver_opts, log_path, solver_func)
    else
        error("Unknown calculation job: $calc_job")
    end

    total_min = (time() - begin_time) / 60
    msg = @sprintf("Task completed successfully in %.2f minutes", total_min)
    @info msg

    open(log_path, "a") do io
        println(io, "="^70)
        println(io, msg)
        println(io, "End Time: ", Dates.now())
        println(io, "="^70)
    end

    println("Task finished. Total time: $(round(total_min, digits=2)) min")
end

# Run main if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
