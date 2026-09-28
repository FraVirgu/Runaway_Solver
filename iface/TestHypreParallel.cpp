/**
 * Standalone test comparing three GMRES preconditioning paths on the same
 * linear system, all fully distributed (PETSC_COMM_WORLD, any rank count):
 *
 *   1. hypre/BoomerAMG, built directly against a raw KSP/PC here.
 *   2. Block-Jacobi with ILU sub-solver (PCBJACOBI / PCILU per block).
 *   3. hypre/BoomerAMG again, but this time via MIAMG -- the MatrixInverter
 *      actually wired into DREAM's linear solver (see
 *      OptionConstants::LINEAR_SOLVER_AMG in Solver::ConstructLinearSolver)
 *      -- so its -dream_amg_lag_pc behaviour can be checked against the
 *      same loaded system as paths 1 and 2, not just measured indirectly
 *      through a full DREAM run.
 *
 * Unlike TestHypre.cpp, this test has no rank-0-only serial baseline (no
 * MILU_PROVA Path A) -- there is no my_rank == 0 special-casing anywhere.
 * The matrix is loaded directly in
 * parallel with MatLoad on PETSC_COMM_WORLD (PETSc's own row
 * decomposition), and each preconditioner factors directly from that
 * already-distributed matrix. Every GMRES iteration then applies the
 * preconditioner on the fly (PCApply) -- no dense matrix, no explicit
 * M^{-1}A, no rank-0 special-casing, and no rank-0-only cross-check either.
 *
 *   -pc_type hypre -pc_hypre_type boomeramg
 *
 * is exactly what this test hardcodes for path 1 (overridable from the
 * command line via PETSC_OPTIONS, same convention as everywhere else in
 * this project). Path 2 hardcodes
 *
 *   -pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_levels 0
 *
 * one block per MPI rank by default (PCBJacobi), each block factored with
 * ILU(0) -- also overridable via PETSC_OPTIONS. Path 3 hardcodes the same
 * hypre/boomeramg configuration as path 1 inside MIAMG::ConfigureSolver(),
 * so any difference from path 1 comes from MIAMG's own logic (in
 * particular -dream_amg_lag_pc), not from a different PC setup.
 *
 * Correctness is checked the one way that makes sense on any rank count:
 * the true residual ||Ax - b|| / ||b||, computed independently of KSP's
 * own converged-reason flag.
 *
 * Loads the same petsc_mat_serial_step1_iter1 / petsc_rhs_serial_step1_iter1
 * files as Main.cpp, TestDistribute.cpp and TestHypre.cpp. Does not touch
 * Main.cpp or TestHypre.cpp: separate executable (see iface/CMakeLists.txt,
 * target testhypreparallel).
 */

#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>

#include <petscksp.h>
#include "DREAM/Init.h"
#include "FVM/Matrix.hpp"
#include "FVM/Solvers/MIAMG.hpp"

using namespace std;
using namespace DREAM;

/**
 * Solve Ax = b with GMRES preconditioned by the given PC type (and, for
 * "hypre", the given hypre sub-type; for "bjacobi", each block is set up
 * with ILU(0) as its sub-solver). Prints iteration count/convergence on
 * rank 0 and returns the wall-clock KSPSetUp+KSPSolve time.
 */
static double SolveWithPC(Mat A, Vec b, Vec x, const char *pcType,
                          const char *hypreType, const char *label,
                          const char *stageName)
{
    PetscLogStage stage;
    PetscLogStageRegister(stageName, &stage);
    PetscLogStagePush(stage);

    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, 100);
    KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1000);
    KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
    // Track the TRUE (unpreconditioned) residual for the convergence test,
    // not the preconditioned one. With a preconditioner as aggressive as
    // BoomerAMG, the preconditioned residual can drop below rtol while the
    // true ||Ax-b|| is still large -- KSPConvergedReason would then report
    // "converged" even though the independent residual check below fails.
    // This keeps KSP's own convergence decision consistent with that check.
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);

    KSPGetPC(ksp, &pc);
    PCSetType(pc, pcType);
    if (hypreType)
        PCHYPRESetType(pc, hypreType);
    if (strcmp(pcType, PCBJACOBI) == 0)
    {
        // Default sub-solver for bjacobi (one block per rank): ILU(0),
        // matching PETSC_OPTIONS="-sub_pc_type ilu -sub_pc_factor_levels 0"
        // from the command line. PCSetUp must run first so the per-block
        // sub-KSPs exist to configure.
        PCSetUp(pc);
        KSP *subksp;
        PetscInt nlocal, first;
        PCBJacobiGetSubKSP(pc, &nlocal, &first, &subksp);
        for (PetscInt i = 0; i < nlocal; i++)
        {
            PC subpc;
            KSPGetPC(subksp[i], &subpc);
            PCSetType(subpc, PCILU);
            PCFactorSetLevels(subpc, 0);
        }
    }
    KSPSetFromOptions(ksp); // command line / PETSC_OPTIONS may override any of the above

    MPI_Barrier(PETSC_COMM_WORLD);
    double t0 = MPI_Wtime();
    KSPSetUp(ksp);
    KSPSolve(ksp, b, x);
    MPI_Barrier(PETSC_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    PetscInt its;
    KSPConvergedReason reason;
    KSPGetIterationNumber(ksp, &its);
    KSPGetConvergedReason(ksp, &reason);

    int my_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);
    if (my_rank == 0)
        cout << label << ": its = " << its
             << ", " << (reason > 0 ? "converged" : "DIVERGED")
             << " (reason " << reason << ")"
             << ", time " << elapsed << " s" << endl;

    KSPDestroy(&ksp);
    PetscLogStagePop();
    return elapsed;
}

/**
 * Solve Ax = b through MIAMG -- the same MatrixInverter DREAM's real solver
 * uses for OptionConstants::LINEAR_SOLVER_AMG -- rather than a KSP/PC built
 * directly here. A is wrapped in a bare FVM::Matrix (the
 * (PetscInt, PetscInt, Mat) constructor just holds the existing Mat, it
 * does not copy or reallocate it), since MatrixInverter::Invert takes
 * FVM::Matrix*, not Mat.
 *
 * Exercises exactly the code path a real DREAM run takes on every timestep:
 * a fresh MIAMG instance's first Invert() call always does the full
 * PCSetUp (MIAMG's own callsSinceSetup counter starts at 0), so calling
 * Invert() here more than once demonstrates -dream_amg_lag_pc the same way
 * repeated timesteps would -- see MIAMG::ConfigureSolver() for what the
 * option does and why it exists.
 *
 * Prints iteration count/convergence on rank 0 and returns the wall-clock
 * time of all Invert() calls combined.
 */
static double SolveWithMIAMG(Mat A, PetscInt M, Vec b, Vec x, int nSolves,
                              PetscLogStage stage)
{
    PetscLogStagePush(stage);

    FVM::Matrix matrix(M, M, A);
    FVM::MIAMG inverter(M);

    int my_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);

    MPI_Barrier(PETSC_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < nSolves; i++)
        inverter.Invert(&matrix, &b, &x);
    MPI_Barrier(PETSC_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    if (my_rank == 0)
        cout << "MIAMG GMRES: " << nSolves << " Invert() call"
             << (nSolves == 1 ? "" : "s") << ", time " << elapsed << " s" << endl;

    PetscLogStagePop();
    return elapsed;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int my_rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    PETSC_COMM_WORLD = MPI_COMM_WORLD;
    dream_initialize();

    const int step = 1;
    string matname = "petsc_mat_serial_step" + to_string(step) + "_iter1";
    string rhsname = "petsc_rhs_serial_step" + to_string(step) + "_iter1";

    int haveStep = 0;
    if (my_rank == 0)
        haveStep = (access(matname.c_str(), F_OK) == 0 &&
                    access(rhsname.c_str(), F_OK) == 0);
    MPI_Bcast(&haveStep, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!haveStep)
    {
        if (my_rank == 0)
            cerr << "Could not find " << matname << " / " << rhsname
                 << " in the current directory -- run the serial simulation "
                    "first to produce them."
                 << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }

    PetscViewer viewer;

    // ---- load A, b already distributed: MatLoad/VecLoad on
    // PETSC_COMM_WORLD split the on-disk (serial) data among ranks
    // according to PETSc's own row decomposition. No rank-0 special-casing
    // and no distribution helpers needed. ----
    Mat A;
    Vec b;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, matname.c_str(), FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetType(A, MATAIJ);
    MatLoad(A, viewer);
    PetscViewerDestroy(&viewer);

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, rhsname.c_str(), FILE_MODE_READ, &viewer);
    VecCreate(PETSC_COMM_WORLD, &b);
    VecLoad(b, viewer);
    PetscViewerDestroy(&viewer);

    PetscInt M, N;
    MatGetSize(A, &M, &N);
    if (my_rank == 0)
        cout << "Loaded system: " << M << " x " << N
             << "  (running on " << size << " rank" << (size == 1 ? "" : "s") << ")" << endl;

    // =============== Distributed GMRES + hypre/BoomerAMG ===============
    Vec x_amg;
    VecDuplicate(b, &x_amg);

#ifdef PETSC_HAVE_HYPRE
    SolveWithPC(A, b, x_amg, PCHYPRE, "boomeramg", "hypre/boomeramg GMRES",
                "AMG solve");
#else
    if (my_rank == 0)
        cerr << "This PETSc build has no hypre support (PETSC_HAVE_HYPRE "
                "undefined) -- cannot run the boomeramg preconditioner."
             << endl;
    dream_finalize();
    MPI_Finalize();
    return 1;
#endif

    MPI_Barrier(PETSC_COMM_WORLD);
    // =============== Distributed GMRES + block-Jacobi(ILU) ===============
    Vec x_bjacobi;
    VecDuplicate(b, &x_bjacobi);
    SolveWithPC(A, b, x_bjacobi, PCBJACOBI, nullptr, "bjacobi/ilu GMRES",
                "BJacobi solve");

    MPI_Barrier(PETSC_COMM_WORLD);
    // =============== Distributed GMRES + hypre/BoomerAMG via MIAMG ===============
#ifdef PETSC_HAVE_HYPRE
    Vec x_miamg;
    VecDuplicate(b, &x_miamg);

    PetscLogStage miamgStage;
    PetscLogStageRegister("MIAMG solve", &miamgStage);

    // One Invert() call reproduces path 1 exactly (fresh PCSetUp, since
    // MIAMG's callsSinceSetup counter starts at 0); pass more to see
    // -dream_amg_lag_pc reuse the hierarchy across "timesteps" the same way
    // it would across repeated calls from SolverLinearlyImplicit::Solve.
    // The matrix and rhs are unchanged between calls here (unlike a real
    // run, where both change every timestep), so this only exercises the
    // reuse-vs-rebuild bookkeeping and its timing, not staleness against an
    // evolving system -- for that, compare against real DREAM runs instead.
    PetscInt nMIAMGSolves = 1;
    PetscOptionsGetInt(NULL, NULL, "-test_miamg_nsolves", &nMIAMGSolves, NULL);
    SolveWithMIAMG(A, M, b, x_miamg, nMIAMGSolves, miamgStage);
#endif

    // ---- true residual, independent of KSP's own converged-reason ----
    auto CheckTrueResidual = [&](Vec x, const char *label)
    {
        Vec r;
        VecDuplicate(b, &r);
        MatMult(A, x, r);    // r = A*x
        VecAXPY(r, -1.0, b); // r = A*x - b

        PetscReal resNorm, bNorm;
        VecNorm(r, NORM_2, &resNorm);
        VecNorm(b, NORM_2, &bNorm);
        PetscReal relRes = (bNorm > 0.0) ? resNorm / bNorm : resNorm;

        const double tol = 1e-6;
        if (my_rank == 0)
            cout << "||A*" << label << " - b|| / ||b|| = " << relRes
                 << (relRes <= tol ? "  (PASS)" : "  (FAIL)") << endl;

        VecDestroy(&r);
    };
    CheckTrueResidual(x_amg, "x_amg");
    CheckTrueResidual(x_bjacobi, "x_bjacobi");
#ifdef PETSC_HAVE_HYPRE
    CheckTrueResidual(x_miamg, "x_miamg");
#endif

    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x_amg);
    VecDestroy(&x_bjacobi);
#ifdef PETSC_HAVE_HYPRE
    VecDestroy(&x_miamg);
#endif

    dream_finalize();
    MPI_Finalize();
    return 0;
}
