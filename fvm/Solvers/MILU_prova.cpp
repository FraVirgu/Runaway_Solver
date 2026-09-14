/**
 * Implementation of matrix invertor based on LU
 * factorization (i.e. direct inversion).
 */

#include <petscvec.h>
#include "FVM/config.h"
#include "FVM/FVMException.hpp"
#include "FVM/Matrix.hpp"
#include "FVM/Solvers/MILU_prova.hpp"

using namespace DREAM::FVM;

/**
 * Constructor.
 *
 * n: Number of elements in solution vector.
 */
MILU_PROVA::MILU_PROVA(const len_t n, len_t Nf)
{
    KSPCreate(PETSC_COMM_WORLD, &this->ksp);
    this->xn = n;
    this->Nf = Nf;
}

/**
 * Destructor.
 */
MILU_PROVA::~MILU_PROVA()
{
    // KSPDestroy(&this->ksp);
}

/**
 * Solves the linear equation system represented by
 *
 *   Ax = b
 *
 * where A is a matrix, and b and x are vectors. A
 * pointer is returned to the solution, x.
 *
 * A: Matrix of size m-by-n representing the linear system.
 * b: Right-hand-side vector containing n elements.
 * x: Solution vector. Contains solution on return. Must be
 *    of size n at least.
 */
void MILU_PROVA::Invert(Matrix *A, Vec *b, Vec *x)
{
    PC pc;
    IS is_kinetic, is_fluid;

    PetscInt Nf = this->Nf; // f_hot + f_re
    PetscInt Ntot;
    VecGetSize(*b, &Ntot);

    ISCreateStride(PETSC_COMM_WORLD, Nf, 0, 1, &is_kinetic);
    ISCreateStride(PETSC_COMM_WORLD, Ntot - Nf, Nf, 1, &is_fluid);

    KSPSetOperators(this->ksp, A->mat(), A->mat());

    // Flexible GMRES preconditioned by ILU(0). FGMRES rather than GMRES so
    // that the preconditioner may itself iterate (needed if ILU is later
    // replaced by a nested solve, e.g. fieldsplit or multigrid).
    KSPSetType(this->ksp, KSPFGMRES);

    KSPGetPC(this->ksp, &pc);
    PCSetType(pc, PCILU);
    PCFactorSetLevels(pc, 0);

    // Convergence is tested on the unpreconditioned residual: with an
    // ill-conditioned operator the preconditioned norm is optimistic and can
    // report convergence well before the true residual has been reduced.
    KSPSetNormType(this->ksp, KSP_NORM_UNPRECONDITIONED);

    // rtol is set an order above the attainable floor, kappa(A)*eps_mach.
    KSPSetTolerances(this->ksp, 1e-8, 1e-50, PETSC_DEFAULT, 500);

    // Applied last, so that PETSC_OPTIONS can still override the above when
    // experimenting with alternative solvers and preconditioners.
    KSPSetFromOptions(this->ksp);

    this->errorcode = KSPSolve(this->ksp, *b, *x);

    // KSPSolve returns successfully even when the iteration failed to converge,
    // so the solution must be validated explicitly. Without this check a
    // diverged solve silently propagates a non-solution into the next timestep.
    KSPConvergedReason reason;
    KSPGetConvergedReason(this->ksp, &reason);

    if (reason < 0)
    {
        PetscInt its;
        KSPGetIterationNumber(this->ksp, &its);

        const char *reasonName = KSPConvergedReasons[reason];

        throw FVMException(
            "MILU_PROVA: linear solve did not converge after %d iterations "
            "(KSPConvergedReason %d: %s).",
            (int)its, (int)reason, reasonName);
    }

    ISDestroy(&is_kinetic);
    ISDestroy(&is_fluid);
}
