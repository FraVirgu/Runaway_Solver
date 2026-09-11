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
    KSPSetOperators(this->ksp, A->mat(), A->mat());

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
}
