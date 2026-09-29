/**
 * Implementation of matrix inverter using GMRES preconditioned by
 * hypre/BoomerAMG applied to the whole (unsplit) system -- the same
 * configuration profiled in iface/TestHypreParallel.cpp, wired in here so
 * its per-timestep behaviour (in particular, whether PCSetUp is paid again
 * on every timestep) can be measured on real DREAM runs rather than just
 * the standalone loaded-matrix benchmark.
 */

#include <petscvec.h>
#include "FVM/config.h"
#include "FVM/FVMException.hpp"
#include "FVM/Matrix.hpp"
#include "FVM/Solvers/MIAMG.hpp"

using namespace DREAM::FVM;

/**
 * Constructor.
 *
 * n: Number of elements in solution vector.
 */
MIAMG::MIAMG(const len_t n) {
    KSPCreate(PETSC_COMM_WORLD, &this->ksp);
    this->xn = n;
}

/**
 * Destructor.
 */
MIAMG::~MIAMG() {
    KSPDestroy(&this->ksp);
    if (this->bWork) VecDestroy(&this->bWork);
    if (this->xPrev) VecDestroy(&this->xPrev);
}

/**
 * One-time construction of the solver: outer Krylov method and the
 * hypre/BoomerAMG preconditioner, both overridable from the command line /
 * PETSC_OPTIONS via KSPSetFromOptions() below.
 *
 * Separated from Invert() so it runs exactly once, the same way
 * MIILU::ConfigureSolver() does -- the operator matrix is reset on
 * every call to Invert(), but the preconditioner type/configuration is
 * chosen only the first time.
 *
 * -dream_amg_lag_pc controls how many calls to Invert() (one per timestep,
 * from SolverLinearlyImplicit::Solve) share one PCSetUp before the AMG
 * hierarchy is rebuilt from the latest matrix:
 *
 *   1  rebuild on every solve (default; matches KSPSetOperators() being
 *      called with a new matrix on every timestep, so this is a no-op --
 *      the behaviour measured in the -log_view runs against real settings)
 *   N  rebuild only every N-th solve; the intervening N-1 solves are
 *      preconditioned against an increasingly stale AMG hierarchy
 *
 * There is no bare-KSP equivalent of this in PETSc: KSPSetLagPreconditioner
 * only exists on SNES, which this solver does not use (Invert() is called
 * directly from the linearly-implicit timestep loop, with no SNES wrapping
 * it), so the counting is done here, toggling KSPSetReusePreconditioner()
 * every N calls instead.
 *
 * This amortises PCSetUp's cost (see the discussion around
 * TestHypreParallel.cpp: it does not shrink with more ranks, so it is worth
 * not paying on every timestep) at the cost of preconditioner staleness. As
 * with MIILU's now-removed -dream_ksp_reuse_pc experiment (which was
 * the N=infinity case of this and was found to destabilise the solve),
 * whether a finite N is acceptable for a given system is a question for
 * measurement, not assumption.
 */
void MIAMG::ConfigureSolver() {
#ifndef PETSC_HAVE_HYPRE
    throw FVMException(
        "MIAMG: this PETSc build has no hypre support (PETSC_HAVE_HYPRE "
        "undefined) -- cannot use the boomeramg preconditioner.");
#endif

    KSPSetType(this->ksp, KSPGMRES);
    KSPGMRESSetRestart(this->ksp, 100);
    KSPSetTolerances(this->ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1000);
    // Start from the previous timestep's solution (see Invert()).
    KSPSetInitialGuessNonzero(this->ksp, PETSC_TRUE);
    // See the comment on this in TestHypreParallel.cpp: with a
    // preconditioner as aggressive as BoomerAMG, the preconditioned residual
    // can look converged well before the true residual has dropped, so
    // convergence is tested on the unpreconditioned residual instead.
    KSPSetNormType(this->ksp, KSP_NORM_UNPRECONDITIONED);

    PC pc;
    KSPGetPC(this->ksp, &pc);
    PCSetType(pc, PCHYPRE);
    PCHYPRESetType(pc, "boomeramg");

    // Read last, so that anything supplied via PETSC_OPTIONS or the command
    // line overrides the defaults set above.
    KSPSetFromOptions(this->ksp);

    PetscOptionsGetInt(NULL, NULL, "-dream_amg_lag_pc", &this->lagPC, NULL);
    if (this->lagPC < 1)
        throw FVMException(
            "MIAMG: -dream_amg_lag_pc must be >= 1 (got %d).", (int)this->lagPC);

    this->configured = true;
}

/**
 * Solves the linear equation system represented by
 *
 *   Ax = b
 *
 * where A is a matrix, and b and x are vectors.
 *
 * A: Matrix of size n-by-n representing the linear system.
 * b: Right-hand-side vector containing n elements.
 * x: Solution vector. Contains solution on return.
 */
void MIAMG::Invert(Matrix *A, Vec *b, Vec *x) {
    // The matrix entries change every timestep, but KSPSetOperators() is
    // still called every time: left to itself, this is what makes PETSc
    // rebuild the AMG hierarchy (PCSetUp) again on the next KSPSolve, since
    // the matrix values changed. That per-timestep rebuild cost is exactly
    // what this class was first built to measure -- see the -log_view
    // discussion around TestHypreParallel.cpp -- and is now what
    // -dream_amg_lag_pc (see ConfigureSolver()) lets you amortise.
    KSPSetOperators(this->ksp, A->mat(), A->mat());

    if (!this->configured)
        this->ConfigureSolver();

    // Every lagPC-th call gets a fresh PCSetUp from the current matrix;
    // the calls in between reuse whichever hierarchy that call built.
    // callsSinceSetup counts calls made since the last rebuild, so the
    // first call after construction (callsSinceSetup == 0) always rebuilds.
    bool rebuildNow = (this->callsSinceSetup % this->lagPC == 0);
    KSPSetReusePreconditioner(this->ksp, rebuildNow ? PETSC_FALSE : PETSC_TRUE);
    this->callsSinceSetup++;

    // Invert() is called with b == x, so keep a copy of the RHS and seed x
    // with the previous solution (zero on the very first call).
    if (!this->bWork)
        VecDuplicate(*b, &this->bWork);
    VecCopy(*b, this->bWork);
    if (this->xPrev)
        VecCopy(this->xPrev, *x);
    else
        VecSet(*x, 0.0);

    this->errorcode = KSPSolve(this->ksp, this->bWork, *x);

    KSPConvergedReason reason;
    KSPGetConvergedReason(this->ksp, &reason);

    if (reason >= 0) {
        if (!this->xPrev)
            VecDuplicate(*x, &this->xPrev);
        VecCopy(*x, this->xPrev);
    }

    if (reason < 0) {
        PetscInt its;
        KSPGetIterationNumber(this->ksp, &its);

        throw FVMException(
            "MIAMG: linear solve did not converge after %d iterations "
            "(KSPConvergedReason %d: %s).",
            (int)its, (int)reason, KSPConvergedReasons[reason]);
    }
}
