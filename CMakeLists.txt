cmake_minimum_required(VERSION 3.17...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

# Search for packages and commands
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

#------------------------------ Compiler options ------------------------------#
# Detect compiler vendor
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  set(VENDOR "gnu")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(VENDOR "intel")
else()
  message(FATAL_ERROR "Unsupported Fortran compiler ${CMAKE_Fortran_COMPILER}")
endif()

# Add compiler options for OpenMP
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
endif()

#----------------------------- Fortran extension ------------------------------#
# Define the fortranobject
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)

# Set the Fortran source file path
set(FORTRAN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/dptb/postprocess/fortran/ac_cond.f90")

# Generate the interface
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ac_condmodule.c"
  DEPENDS "${FORTRAN_SOURCE}"
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py "${FORTRAN_SOURCE}"
          -m ac_cond --lower
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM
)

# Define the python module
python_add_library(ac_cond MODULE
                   "${CMAKE_CURRENT_BINARY_DIR}/ac_condmodule.c"
                   "${FORTRAN_SOURCE}"
                   WITH_SOABI)
target_link_libraries(ac_cond PRIVATE fortranobject)
if(OpenMP_Fortran_FOUND)
  target_link_libraries(ac_cond PRIVATE OpenMP::OpenMP_Fortran)
endif()

install(TARGETS ac_cond DESTINATION ./dptb/postprocess/fortran)