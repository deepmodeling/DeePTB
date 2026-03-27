
using Pkg

# Check what is installed
println("Checking dependencies...")
Pkg.status("MKL_jll")
Pkg.status("Pardiso")

# Try to find the MKL library file explicitly
using MKL_jll
# Older JLLs use LIBPATH
mkl_path = joinpath(MKL_jll.LIBPATH[], "libmkl_rt.dylib")
println("MKL JLL Path reported: ", mkl_path)

# Check if file exists
if isfile(mkl_path)
    println("MKL library file exists.")
else
    println("MKL library file DOES NOT exist at reported path!")
end

# Attempt 1: Just loading Pardiso
println("\n--- Attempting to load Pardiso ---")
using Pardiso

# Check if we can force MKL
println("Current backend: ", Pardiso.get_solver_type())

try
    println("Initializing MKLPardisoSolver...")
    ps = MKLPardisoSolver()
    println("SUCCESS: MKLPardisoSolver initialized!")
catch e
    println("FAILURE: ", e)
end

# Attempt 2: Manually loading the library (dlopen) if the above failed
if !Pardiso.MKL_PARDISO_LOADED
    println("\n--- Attempting manual dlopen of MKL ---")
    using Libdl
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
