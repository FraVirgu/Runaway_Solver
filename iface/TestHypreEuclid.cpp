/**
 * Standalone test comparing three GMRES preconditioning paths on the same
 * linear system:
 *
 *   1. Path A: PETSc's own PCILU, serial (rank 0 only, PETSC_COMM_SELF).
 *      Only factors a rank-local piece of the matrix -- no parallel
 *      equivalent -- so this always runs first as the serial baseline.
 *   2. hypre/BoomerAMG, distributed (PETSC_COMM_WORLD, any rank count).
 *   3. hypre/Euclid, a genuine PARALLEL incomplete factorization,
 *      distributed (PETSC_COMM_WORLD, any rank count).
 *
 * Unlike the rank-0-factor-then-distribute machinery in Main.cpp's else
 * branch (DistributeMatrixFromRank0 / DistributeVectorFromRank0 /
 * MILU_PROVA::ApplyILUPreconditioning), the two hypre paths need none of
 * that: the matrix is loaded directly in parallel with MatLoad on
 * PETSC_COMM_WORLD (PETSc's own row decomposition), and hypre factors its
 * preconditioner directly from that already-distributed matrix. Every
 * GMRES iteration then applies the preconditioner on the fly (PCApply),
 * exactly like PCILU does serially -- no dense matrix, no explicit M^{-1}A,
 * no rank-0 special-casing.
 *
 *   -pc_type hypre -pc_hypre_type boomeramg
 *   -pc_type hypre -pc_hypre_type euclid
 *
 * is exactly what this test hardcodes for paths 2 and 3 respectively
 * (overridable from the command line via PETSC_OPTIONS, same convention as
 * everywhere else in this project).
 *
 * Correctness is checked two ways for each of the two hypre paths:
 *   1. The true residual ||Ax - b|| / ||b||, computed independently of
 *      KSP's own converged-reason flag, on any rank count.
 *   2. When run with exactly 1 rank, ALSO compare against the Path A
 *      solution (x_ilu), as a sanity cross-check that boomeramg/euclid are
 *      converging to the same answer as the familiar PCILU path. This
 *      second check is skipped for n > 1, since PCILU has no well-defined
 *      distributed behaviour to compare against.
 *
 * Loads the same petsc_mat_serial_step1_iter1 / petsc_rhs_serial_step1_iter1
 * files as Main.cpp and TestDistribute.cpp. Does not touch Main.cpp:
 * separate executable (see iface/CMakeLists.txt, target testhypreeuclid).
 */

#include <iostream>
#include <string>
#include <unistd.h>

#include <petscksp.h>
#include "DREAM/Init.h"

using namespace std;

/**
 * Solve Ax = b with GMRES preconditioned by the given PC type (and, for
 * "hypre", the given hypre sub-type). Prints iteration count/convergence
 * on rank 0 and returns the wall-clock KSPSetUp+KSPSolve time.
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

    KSPGetPC(ksp, &pc);
    PCSetType(pc, pcType);
    if (hypreType)
        PCHYPRESetType(pc, hypreType);
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

    // =============== Path A: rank-0-only PETSc PCILU + GMRES =============
    // Runs first, before either parallel hypre path, so its iteration count
    // and timing are available as the serial baseline for both comparisons
    // below. Independent of how many MPI ranks the job was launched with:
    // rank 0 loads its own private copy of A, b on PETSC_COMM_SELF (never
    // touching the already-distributed A, b below) and solves entirely by
    // itself, exactly as in TestILUCompare.cpp's Path A. tLoadA times that
    // load, tPathA the KSPSetUp+KSPSolve, both rank-0 wall-clock.
    Vec x_ilu = nullptr; // only valid on rank 0; used below if size == 1
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

        double tA0 = MPI_Wtime();
        {
            KSP ksp;
            PC pc;
            KSPCreate(PETSC_COMM_SELF, &ksp);
            KSPSetOperators(ksp, A_seq, A_seq);
            KSPSetType(ksp, KSPGMRES);
            KSPGMRESSetRestart(ksp, 100);
            KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1000);
            KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);

            KSPGetPC(ksp, &pc);
            PCSetType(pc, PCILU);
            PCFactorSetLevels(pc, 1);
            KSPSetFromOptions(ksp);

            KSPSetUp(ksp);
            KSPSolve(ksp, b_seq, x_ilu);

            PetscInt its;
            KSPConvergedReason reason;
            KSPGetIterationNumber(ksp, &its);
            KSPGetConvergedReason(ksp, &reason);
            cout << "Path A (rank-0 PCILU + GMRES): its = " << its
                 << ", " << (reason > 0 ? "converged" : "DIVERGED")
                 << " (reason " << reason << ")" << endl;

            KSPDestroy(&ksp);
        }
        double tPathA = MPI_Wtime() - tA0;

        cout << "Path A load (rank-0 MatLoad+VecLoad): " << tLoadA << " s"
             << "   solve: " << tPathA << " s" << endl;

        MatDestroy(&A_seq);
        VecDestroy(&b_seq);
    }

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

    // ---- true residual, independent of KSP's own converged-reason ----
    {
        Vec r;
        VecDuplicate(b, &r);
        MatMult(A, x_amg, r);   // r = A*x_amg
        VecAXPY(r, -1.0, b);    // r = A*x_amg - b

        PetscReal resNorm, bNorm;
        VecNorm(r, NORM_2, &resNorm);
        VecNorm(b, NORM_2, &bNorm);
        PetscReal relRes = (bNorm > 0.0) ? resNorm / bNorm : resNorm;

        const double tol = 1e-6;
        if (my_rank == 0)
            cout << "||A*x_amg - b|| / ||b|| = " << relRes
                 << (relRes <= tol ? "  (PASS)" : "  (FAIL)") << endl;

        VecDestroy(&r);
    }

    // =============== Distributed GMRES + hypre/Euclid ===============
    Vec x_euclid;
    VecDuplicate(b, &x_euclid);

#ifdef PETSC_HAVE_HYPRE
    SolveWithPC(A, b, x_euclid, PCHYPRE, "euclid", "hypre/euclid GMRES");
#else
    if (my_rank == 0)
        cerr << "This PETSc build has no hypre support (PETSC_HAVE_HYPRE "
                "undefined) -- cannot run the euclid preconditioner."
             << endl;
    dream_finalize();
    MPI_Finalize();
    return 1;
#endif

    // ---- true residual, independent of KSP's own converged-reason ----
    {
        Vec r;
        VecDuplicate(b, &r);
        MatMult(A, x_euclid, r);   // r = A*x_euclid
        VecAXPY(r, -1.0, b);       // r = A*x_euclid - b

        PetscReal resNorm, bNorm;
        VecNorm(r, NORM_2, &resNorm);
        VecNorm(b, NORM_2, &bNorm);
        PetscReal relRes = (bNorm > 0.0) ? resNorm / bNorm : resNorm;

        const double tol = 1e-6;
        if (my_rank == 0)
            cout << "||A*x_euclid - b|| / ||b|| = " << relRes
                 << (relRes <= tol ? "  (PASS)" : "  (FAIL)") << endl;

        VecDestroy(&r);
    }

    // ---- Path A cross-checks: only meaningful when size == 1, since x_ilu
    // then has the same layout as x_amg/x_euclid. ----
    if (my_rank == 0)
    {
        if (size == 1)
        {
            auto compare = [&](Vec x, const char *label)
            {
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

            compare(x_amg, "x_amg");
            compare(x_euclid, "x_euclid");
        }
        else
        {
            cout << "(x_amg/x_euclid vs x_ilu comparison skipped: running "
                    "with " << size << " ranks, not 1)" << endl;
        }

        VecDestroy(&x_ilu);
    }

    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x_amg);
    VecDestroy(&x_euclid);

    dream_finalize();
    MPI_Finalize();
    return 0;
}
