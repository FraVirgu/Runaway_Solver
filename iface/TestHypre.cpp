/**
 * Standalone test comparing three GMRES preconditioning paths on the same
 * linear system:
 *
 *   1. Path A: dreami's own linear solver, DREAM::FVM::MIILU, called
 *      directly -- not a hand-copied replica of its KSP/PC setup, but the
 *      actual production class (fvm/Solvers/MIILU.cpp), so this path
 *      can never drift out of sync with what dreami itself runs. A_seq is
 *      wrapped in an FVM::Matrix (the Matrix(m, n, Mat) constructor, which
 *      takes ownership of an already-built PETSc Mat without copying it),
 *      and MIILU is constructed with the same Nhot/Nre block sizes
 *      dreami itself used, read back from petsc_block_sizes.txt (written
 *      by SolverLinearlyImplicit::initialize_internal for exactly this
 *      kind of external replay). Serial (rank 0 only, PETSC_COMM_SELF) --
 *      only factors a rank-local piece of the matrix, no parallel
 *      equivalent -- so this always runs first as the serial baseline.
 *      The -dream_split configuration (none/kinetic/populations) is
 *      selected the same way it is in dreami: via PETSC_OPTIONS.
 *   2. hypre/BoomerAMG, distributed (PETSC_COMM_WORLD, any rank count).
 *   3. Block-Jacobi with ILU sub-solver (PCBJACOBI / PCILU per block),
 *      distributed (PETSC_COMM_WORLD, any rank count).
 *
 * Unlike the rank-0-factor-then-distribute machinery in Main.cpp's else
 * branch (DistributeMatrixFromRank0 / DistributeVectorFromRank0 /
 * MIILU::ApplyILUPreconditioning), the BoomerAMG and block-Jacobi
 * paths need none of that: the matrix is loaded directly in parallel with
 * MatLoad on PETSC_COMM_WORLD (PETSc's own row decomposition), and each
 * preconditioner factors directly from that already-distributed matrix.
 * Every GMRES iteration then applies the preconditioner on the fly
 * (PCApply), exactly like PCILU does serially -- no dense matrix, no
 * explicit M^{-1}A, no rank-0 special-casing.
 *
 *   -pc_type hypre -pc_hypre_type boomeramg
 *
 * is exactly what this test hardcodes for path 2 (overridable from the
 * command line via PETSC_OPTIONS, same convention as everywhere else in
 * this project). Path 3 hardcodes
 *
 *   -pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_levels 0
 *
 * one block per MPI rank by default (PCBJacobi), each block factored with
 * ILU(0) -- also overridable via PETSC_OPTIONS.
 *
 * Correctness is checked two ways for each distributed path:
 *   1. The true residual ||Ax - b|| / ||b||, computed independently of
 *      KSP's own converged-reason flag, on any rank count.
 *   2. When run with exactly 1 rank, ALSO compare against the Path A
 *      solution (x_ilu), as a sanity cross-check that the distributed path
 *      is converging to the same answer as the familiar PCILU path. This
 *      second check is skipped for n > 1, since PCILU has no well-defined
 *      distributed behaviour to compare against.
 *
 * Loads the same petsc_mat_serial_step1_iter1 / petsc_rhs_serial_step1_iter1
 * files as Main.cpp and TestDistribute.cpp. Does not touch Main.cpp:
 * separate executable (see iface/CMakeLists.txt, target testhypre).
 */

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include <petscksp.h>
#include "DREAM/Init.h"
#include "FVM/Matrix.hpp"
#include "FVM/Solvers/MIILU.hpp"

using namespace std;

/**
 * Solve Ax = b with GMRES preconditioned by the given PC type (and, for
 * "hypre", the given hypre sub-type; for "bjacobi", each block is set up
 * with ILU(0) as its sub-solver). Prints iteration count/convergence on
 * rank 0 and returns the wall-clock KSPSetUp+KSPSolve time.
 */
static double SolveWithPC(Mat A, Vec b, Vec x, const char *pcType,
                          const char *hypreType, const char *label)
{
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

    // =============== Path A: rank-0-only DREAM::FVM::MIILU =============
    // Runs first, before the parallel hypre path, so its iteration count
    // and timing are available as the serial baseline for the comparison
    // below. Independent of how many MPI ranks the job was launched with:
    // rank 0 loads its own private copy of A, b on PETSC_COMM_SELF (never
    // touching the already-distributed A, b below) and solves entirely by
    // itself, exactly as in TestILUCompare.cpp's Path A. tLoadA times that
    // load, tPathA the MIILU::Invert call, both rank-0 wall-clock.
    //
    // This calls dreami's actual production solver class directly -- not a
    // hand-copied replica of its KSP/PC setup -- so it can never drift out
    // of sync with what dreami itself runs. A_seq is wrapped in an
    // FVM::Matrix via the Matrix(m, n, Mat) constructor (no copy of the
    // underlying PETSc Mat), and MIILU is given the same Nhot/Nre
    // block sizes dreami used, read back from petsc_block_sizes.txt
    // (written by SolverLinearlyImplicit::initialize_internal for exactly
    // this kind of external replay). The -dream_split configuration
    // (none/kinetic/populations) and any PC/KSP overrides are selected the
    // same way they are in dreami: via PETSC_OPTIONS/the command line,
    // since MIILU::ConfigureSolver() calls KSPSetFromOptions() itself.
    Vec x_ilu = nullptr; // only valid on rank 0; used below if size == 1
    MPI_Barrier(PETSC_COMM_WORLD);
    if (my_rank == 0)
    {
        double tLoadA0 = MPI_Wtime();

        Mat A_seq;
        Vec b_seq;
        PetscViewer viewer_seq;

        PetscViewerBinaryOpen(PETSC_COMM_SELF, matname.c_str(), FILE_MODE_READ, &viewer_seq);
        MatCreate(PETSC_COMM_SELF, &A_seq);
        MatSetType(A_seq, MATSEQAIJ);
        MatLoad(A_seq, viewer_seq);
        PetscViewerDestroy(&viewer_seq);

        PetscViewerBinaryOpen(PETSC_COMM_SELF, rhsname.c_str(), FILE_MODE_READ, &viewer_seq);
        VecCreate(PETSC_COMM_SELF, &b_seq);
        VecLoad(b_seq, viewer_seq);
        PetscViewerDestroy(&viewer_seq);

        double tLoadA = MPI_Wtime() - tLoadA0;

        VecDuplicate(b_seq, &x_ilu);

        PetscInt Ntot;
        MatGetSize(A_seq, &Ntot, nullptr);

        // Nhot/Nre as dreami itself computed them for this same matrix,
        // read back from the file SolverLinearlyImplicit::initialize_internal
        // writes for this purpose. MIILU's monolithic configuration
        // (-dream_split none) does not use them, but its constructor and
        // ConfigureSolver() require valid values regardless.
        PetscInt Nhot = 0, Nre = 0;
        {
            ifstream blockSizesFile("petsc_block_sizes.txt");
            string name;
            PetscInt value;
            while (blockSizesFile >> name >> value)
            {
                if (name == "Nhot")
                    Nhot = value;
                else if (name == "Nre")
                    Nre = value;
            }
            if (!blockSizesFile.eof() && blockSizesFile.fail())
                cerr << "Warning: could not parse petsc_block_sizes.txt -- "
                        "Nhot/Nre default to 0."
                     << endl;
        }

        double tA0 = MPI_Wtime();
        {
            DREAM::FVM::Matrix Aw((PetscInt)Ntot, (PetscInt)Ntot, A_seq);
            DREAM::FVM::MIILU inverter((len_t)Ntot, (len_t)Nhot, (len_t)Nre);

            try
            {
                inverter.Invert(&Aw, &b_seq, &x_ilu);
                cout << "Path A (rank-0 MIILU): converged" << endl;
            }
            catch (const DREAM::FVM::FVMException &ex)
            {
                cout << "Path A (rank-0 MIILU): DIVERGED (" << ex.what() << ")" << endl;
            }
            // Aw wraps A_seq via the Matrix(m, n, Mat) constructor, which
            // leaves 'allocated' false; Matrix::~Matrix() -> Destroy() is
            // therefore a no-op here and never touches A_seq, which is
            // MatDestroy'd explicitly below as usual.
        }
        double tPathA = MPI_Wtime() - tA0;

        cout << "Path A load (rank-0 MatLoad+VecLoad): " << tLoadA << " s"
             << "   solve: " << tPathA << " s" << endl;

        MatDestroy(&A_seq);
        VecDestroy(&b_seq);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
    // =============== Distributed GMRES + hypre/BoomerAMG ===============
    Vec x_amg;
    VecDuplicate(b, &x_amg);

#ifdef PETSC_HAVE_HYPRE
    SolveWithPC(A, b, x_amg, PCHYPRE, "boomeramg", "hypre/boomeramg GMRES");
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
    SolveWithPC(A, b, x_bjacobi, PCBJACOBI, nullptr, "bjacobi/ilu GMRES");

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

    // ---- Path A cross-check: only meaningful when size == 1, since x_ilu
    // then has the same layout as x_amg / x_bjacobi. ----
    if (my_rank == 0)
    {
        auto CheckAgainstPathA = [&](Vec x, const char *label)
        {
            if (size != 1)
            {
                cout << "(" << label << " vs x_ilu comparison skipped: running with "
                     << size << " ranks, not 1)" << endl;
                return;
            }

            Vec diff;
            VecDuplicate(x_ilu, &diff);
            VecWAXPY(diff, -1.0, x_ilu, x);

            PetscReal diffNorm, refNorm;
            VecNorm(diff, NORM_2, &diffNorm);
            VecNorm(x_ilu, NORM_2, &refNorm);
            PetscReal relError = (refNorm > 0.0) ? diffNorm / refNorm : diffNorm;

            const double tol = 1e-6;
            cout << "||" << label << " - x_ilu|| / ||x_ilu|| = " << relError
                 << (relError <= tol ? "  (MATCH)" : "  (MISMATCH)") << endl;

            VecDestroy(&diff);
        };

        CheckAgainstPathA(x_amg, "x_amg");
        CheckAgainstPathA(x_bjacobi, "x_bjacobi");

        VecDestroy(&x_ilu);
    }

    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x_amg);
    VecDestroy(&x_bjacobi);

    dream_finalize();
    MPI_Finalize();
    return 0;
}
