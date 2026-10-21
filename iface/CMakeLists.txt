
set(dreami_files
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/Main.cpp"
)

add_executable(dreami ${dreami_files})
target_link_libraries(dreami PUBLIC dream)
set_target_properties(dreami PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Kinetic-block-only tests: solve A_kk x_k = b_k - A_kf x_f with GMRES +
# hypre/BoomerAMG, the fluid/scalar unknowns x_f taken from a reference
# solution. Same source, built twice: the serial one enforces a single MPI
# rank, the parallel one runs on any rank count.
set(testkinetichypre_source
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/TestKineticHypre.cpp"
)

add_executable(testkinetichypreserial ${testkinetichypre_source})
target_compile_definitions(testkinetichypreserial PRIVATE KINETIC_SERIAL)
target_link_libraries(testkinetichypreserial PUBLIC dream)
set_target_properties(testkinetichypreserial PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

add_executable(testkinetichypreparallel ${testkinetichypre_source})
target_link_libraries(testkinetichypreparallel PUBLIC dream)
set_target_properties(testkinetichypreparallel PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Same kinetic test with the f_hot / f_re field-split preconditioner (-split)
add_executable(testkinetichypresplit
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/TestKineticHypre_split.cpp"
)
target_link_libraries(testkinetichypresplit PUBLIC dream)
set_target_properties(testkinetichypresplit PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Kinetic test for Nr > 1: GMRES + BoomerAMG against GMRES + block Jacobi
# over the radius (one block per radial cell, f_hot + f_re of that radius)
add_executable(testkinetichypreradius
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/TestKineticHypreRadius.cpp"
)
target_link_libraries(testkinetichypreradius PUBLIC dream)
set_target_properties(testkinetichypreradius PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Previous version of the radius test (one block per radius, no ASM scan)
add_executable(oldtestkinetichypreradius
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/old_testKineticHypreRadius.cpp"
)
target_link_libraries(oldtestkinetichypreradius PUBLIC dream)
set_target_properties(oldtestkinetichypreradius PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#if (INTERPROC_OPTIM AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
#    include(CheckIPOSupported)
#    # Clang does not support IPO (although Clang++ does), so we need
#    # to explicitly check both C and C++ (since Clang is used for linking)
#    check_ipo_supported(RESULT result LANGUAGES C CXX)
#    if (result)
#        set_target_properties(dreami PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
#    endif()
#endif ()

