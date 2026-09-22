/**
 * Standalone serial (n=1) test comparing two ways of solving Ax = b with
 * ILU(0)/ILU(1)-preconditioned GMRES:
 *
 *   Path A -- "PETSc-managed": a single KSP with PCILU as the
 *             preconditioner. PETSc applies M^{-1} implicitly inside every
 *             GMRES iteration; the preconditioned operator M^{-1}A is never
 *             formed explicitly.
 *
 *   Path B -- "explicit preconditioning": MILU_PROVA::ApplyILUPreconditioning
 *             factorises the same ILU preconditioner M and forms the
 *             explicit sparse operator MA = M^{-1}A and right-hand side
 *             Mb = M^{-1}b, then MILU_PROVA::Solve_GMRES runs *plain*
 *             (PCNONE) GMRES on MA x = Mb.
 *
 * Both paths should converge to (approximately) the same x, since they are
 * solving mathematically equivalent left-preconditioned systems -- one with
 * the preconditioning folded into every iteration, the other with it
 * applied once up front. Any large discrepancy points at a bug in
 * ApplyILUPreconditioning's sparsification (dropTol) or in the ILU
 * configuration mismatch between the two paths.
 *
 * The matrix is a small synthetic 1D discrete-Laplacian-like sparse system
 * (tridiagonal, diagonally dominant), generated in-process rather than
 * loaded from the simulation's dumped petsc_mat_serial_* files. Those can
 * be arbitrarily large (hundreds of thousands of rows), and
 * ApplyILUPreconditioning converts A to a DENSE n-by-n matrix internally to
 * apply M^{-1} column by column -- at production matrix sizes that dense
 * copy alone needs terabytes of RAM. This test exists to check the
 * arithmetic of the explicit-preconditioning path, not to scale to it.
 *
 * Serial only: this test builds/loads the matrix and vector directly with
 * MatSeqAIJ / VecSeq layouts and makes no attempt at, and gives no
 * guarantee about, correctness under mpirun -n > 1. Run it with n=1.
 *
 * Does not touch Main.cpp: separate executable (see iface/CMakeLists.txt,
 * target testiluCompare).
 */

#include <iostream>

#include "DREAM/Init.h"
#include "FVM/Solvers/MILU_prova.hpp"

using namespace std;

/**
 * Build a small n-by-n tridiagonal, diagonally dominant sparse matrix
 * (2 on the diagonal, -1 on the first off-diagonals -- the standard 1D
 * discrete Laplacian) and a right-hand side of all ones. Small and
 * well-conditioned enough that ApplyILUPreconditioning's dense
 * intermediate (n-by-n doubles) stays a few hundred KB at most.
 */
static void BuildTestSystem(PetscInt n, Mat *A, Vec *b)
{
    MatCreate(PETSC_COMM_SELF, A);
    MatSetSizes(*A, n, n, n, n);
    MatSetType(*A, MATSEQAIJ);
    MatSeqAIJSetPreallocation(*A, 3, NULL);

    for (PetscInt i = 0; i < n; i++)
    {
        PetscInt cols[3];
        PetscScalar vals[3];
        PetscInt ncols = 0;

        if (i > 0)
        {
            cols[ncols] = i - 1;
            vals[ncols] = -1.0;
            ncols++;
        }
        cols[ncols] = i;
        vals[ncols] = 2.0;
        ncols++;
        if (i < n - 1)
        {
            cols[ncols] = i + 1;
            vals[ncols] = -1.0;
            ncols++;
        }

        MatSetValues(*A, 1, &i, ncols, cols, vals, INSERT_VALUES);
    }
    MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);

    VecCreate(PETSC_COMM_SELF, b);
    VecSetSizes(*b, n, n);
    VecSetType(*b, VECSEQ);
    VecSet(*b, 1.0);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int my_rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 1)
    {
        if (my_rank == 0)
            cerr << "TestILUCompare is a serial-only test; got " << size
                 << " MPI ranks. Run with mpirun -n 1 (or plain ./testiluCompare)."
                 << endl;
        MPI_Finalize();
        return 1;
    }

    PETSC_COMM_WORLD = MPI_COMM_WORLD;
    dream_initialize();

    const int step = 1;
    const PetscInt n = 200;

    // ---- build A, b (serial: PETSC_COMM_SELF == PETSC_COMM_WORLD here) ----
    Mat A;
    Vec b;
    BuildTestSystem(n, &A, &b);
    cout << "Test system: " << n << " x " << n << " tridiagonal, diagonally dominant" << endl;

    // =============== Path A: PETSc-managed ILU + GMRES ===============
    Vec x_petsc;
    VecDuplicate(b, &x_petsc);

    double tA0 = MPI_Wtime();
    {
        KSP ksp;
        PC pc;
        KSPCreate(PETSC_COMM_SELF, &ksp);
        KSPSetOperators(ksp, A, A);
        KSPSetType(ksp, KSPGMRES);
        KSPGMRESSetRestart(ksp, 100);
        KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1000);
        KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);

        KSPGetPC(ksp, &pc);
        PCSetType(pc, PCILU);
        PCFactorSetLevels(pc, 1);
        KSPSetFromOptions(ksp);

        KSPSetUp(ksp);
        KSPSolve(ksp, b, x_petsc);

        PetscInt its;
        KSPConvergedReason reason;
        KSPGetIterationNumber(ksp, &its);
        KSPGetConvergedReason(ksp, &reason);
        cout << "Path A (PETSc PCILU + GMRES): its = " << its
             << ", " << (reason > 0 ? "converged" : "DIVERGED")
             << " (reason " << reason << ")" << endl;

        KSPDestroy(&ksp);
    }
    double tPathA = MPI_Wtime() - tA0;

    // =============== Path B: explicit M^{-1}A, M^{-1}b + plain GMRES =====
    Vec x_explicit;
    double tB0 = MPI_Wtime();

    Mat MA;
    Vec Mb;
    DREAM::FVM::MILU_PROVA::ApplyILUPreconditioning(A, b, &MA, &Mb);

    VecDuplicate(Mb, &x_explicit);

    PetscLogStage gmresSetupStage, solveStage;
    PetscLogStageRegister("GMRES setup (explicit)", &gmresSetupStage);
    PetscLogStageRegister("Solve (explicit)", &solveStage);

    DREAM::FVM::MILU_PROVA::Solve_GMRES(
        MA, Mb, x_explicit, step, my_rank, gmresSetupStage, solveStage);

    double tPathB = MPI_Wtime() - tB0;

    // =============== Compare x_petsc vs x_explicit ===============
    {
        Vec diff;
        VecDuplicate(x_petsc, &diff);
        VecWAXPY(diff, -1.0, x_explicit, x_petsc);

        PetscReal diffNorm, refNorm;
        VecNorm(diff, NORM_2, &diffNorm);
        VecNorm(x_petsc, NORM_2, &refNorm);
        PetscReal relError = (refNorm > 0.0) ? diffNorm / refNorm : diffNorm;

        const double tol = 1e-6;
        cout << endl;
        cout << "||x_petsc - x_explicit|| / ||x_petsc|| = " << relError
             << (relError <= tol ? "  (MATCH)" : "  (MISMATCH)") << endl;
        cout << "Path A (PETSc-managed) time: " << tPathA << " s" << endl;
        cout << "Path B (explicit M^-1)  time: " << tPathB << " s" << endl;

        VecDestroy(&diff);
    }

    MatDestroy(&A);
    MatDestroy(&MA);
    VecDestroy(&b);
    VecDestroy(&Mb);
    VecDestroy(&x_petsc);
    VecDestroy(&x_explicit);

    dream_finalize();
    MPI_Finalize();
    return 0;
}
