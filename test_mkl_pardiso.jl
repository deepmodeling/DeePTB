
using Pkg

# Check what is installed
println("Checking dependencies...")
Pkg.status("MKL_jll")
Pkg.status("Pardiso")

# Try to find the MKL library file explicitly
using MKL_jll
using Libdl
# Older JLLs use LIBPATH
mkl_path = joinpath(MKL_jll.LIBPATH[], "libmkl_rt." * Libdl.dlext)
println("MKL JLL Path reported: ", mkl_path)

function mkl_pardiso_loaded()
    if isdefined(Pardiso, :MKL_PARDISO_LOADED)
        return getfield(Pardiso, :MKL_PARDISO_LOADED)[]
    end
    return false
end

# Check if file exists
if isfile(mkl_path)
    println("MKL library file exists.")
else
    println("MKL library file DOES NOT exist at reported path!")
end

# Attempt 1: Just loading Pardiso
println("\n--- Attempting to load Pardiso ---")
using Pardiso

# Check whether Pardiso.jl has loaded MKL Pardiso.
println("MKL Pardiso loaded: ", mkl_pardiso_loaded())

try
    println("Initializing MKLPardisoSolver...")
    ps = MKLPardisoSolver()
    println("SUCCESS: MKLPardisoSolver initialized!")
catch e
    println("FAILURE: ", e)
end

# Attempt 2: Manually loading the library (dlopen) if the above failed
if !mkl_pardiso_loaded()
    println("\n--- Attempting manual dlopen of MKL ---")
    try
        Libdl.dlopen(mkl_path, Libdl.RTLD_GLOBAL)
        println("dlopen success.")
        
        # We might need to tell Pardiso it's loaded? 
        # But Pardiso.jl checks internally.
        # Let's see if it works now.
        ps = MKLPardisoSolver()
        println("SUCCESS: MKLPardisoSolver initialized after dlopen!")
    catch e
        println("FAILURE after dlopen: ", e)
    end
end
