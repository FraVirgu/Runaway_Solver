
set(dreami_files
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/Main.cpp"
)

add_executable(dreami ${dreami_files})
target_link_libraries(dreami PUBLIC dream)
set_target_properties(dreami PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Standalone test for the DistributeMatrixFromRank0 / DistributeVectorFromRank0
# helpers in Distribute.hpp -- does not touch Main.cpp or the dreami target.
set(testdistribute_files
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/TestDistribute.cpp"
)

add_executable(testdistribute ${testdistribute_files})
target_link_libraries(testdistribute PUBLIC dream)
set_target_properties(testdistribute PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Standalone serial (n=1) test comparing PETSc-managed ILU+GMRES against
# explicit M^-1A/M^-1b preconditioning + plain GMRES (MILU_PROVA). Does not
# touch Main.cpp or the dreami target.
set(testiluCompare_files
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/TestILUCompare.cpp"
)

add_executable(testiluCompare ${testiluCompare_files})
target_link_libraries(testiluCompare PUBLIC dream)
set_target_properties(testiluCompare PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Standalone test for distributed GMRES preconditioned by hypre's Euclid
# (a genuine parallel ILU), loading the matrix directly distributed like
# Main.cpp's parallel path. Does not touch Main.cpp or the dreami target.
set(testhypreeuclid_files
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/TestHypreEuclid.cpp"
)

add_executable(testhypreeuclid ${testhypreeuclid_files})
target_link_libraries(testhypreeuclid PUBLIC dream)
set_target_properties(testhypreeuclid PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Standalone test for distributed GMRES preconditioned by hypre's BoomerAMG
# (algebraic multigrid), compared against a rank-0 PCILU baseline. Loads
# the matrix directly distributed like Main.cpp's parallel path. Does not
# touch Main.cpp or the dreami target.
set(testhypre_files
    "${PROJECT_SOURCE_DIR}/../Runaway_Solver/iface/TestHypre.cpp"
)

add_executable(testhypre ${testhypre_files})
target_link_libraries(testhypre PUBLIC dream)
set_target_properties(testhypre PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#if (INTERPROC_OPTIM AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
#    include(CheckIPOSupported)
#    # Clang does not support IPO (although Clang++ does), so we need
#    # to explicitly check both C and C++ (since Clang is used for linking)
#    check_ipo_supported(RESULT result LANGUAGES C CXX)
#    if (result)
#        set_target_properties(dreami PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
#    endif()
#endif ()

