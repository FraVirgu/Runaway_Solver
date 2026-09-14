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
MILU_PROVA::MILU_PROVA(const len_t n, len_t Nhot, len_t Nre)
{
    KSPCreate(PETSC_COMM_WORLD, &this->ksp);
    this->xn = n;
    this->Nhot = Nhot;
    this->Nre = Nre;
}

/**
 * Destructor.
 */
MILU_PROVA::~MILU_PROVA()
{
    // KSPDestroy(&this->ksp);
}

/**
 * Registers an option only if it has not already been supplied through
 * PETSC_OPTIONS or the command line.
 *
 * PetscOptionsSetValue overwrites unconditionally, so calling it directly
 * would make the built-in choices override anything given by the user -- the
 * opposite of the intended precedence. Checking first leaves the terminal in
 * control and reduces these calls to defaults.
 */
static void SetDefaultOption(const char *name, const char *value)
{
    PetscBool set;
    PetscOptionsHasName(NULL, NULL, name, &set);
    if (!set)
        PetscOptionsSetValue(NULL, name, value);
}

/**
 * Outer Krylov method, common to every configuration.
 *
 * Flexible GMRES rather than GMRES, since the preconditioner is permitted to
 * vary between iterations: the sub-block solves may themselves be Krylov
 * methods. Convergence is tested on the unpreconditioned residual, because
 * with an ill-conditioned operator the preconditioned norm is optimistic and
 * can report convergence well before the true residual has been reduced.
 *
 * The tolerance is set an order above the attainable floor, kappa(A)*eps_mach;
 * for this system kappa is of order 1e8 after preconditioning, so 1e-12 or
 * tighter is unreachable and merely exhausts the iteration limit.
 */
void MILU_PROVA::ConfigureOuter()
{
    KSPSetType(this->ksp, KSPFGMRES);
    KSPSetNormType(this->ksp, KSP_NORM_UNPRECONDITIONED);
    KSPSetTolerances(this->ksp, 1e-8, 1e-50, PETSC_DEFAULT, 500);
}

/**
 * Configuration 1: no splitting.
 *
 * The whole system is handed to a single preconditioner. With -pc_type lu
 * this is the direct solve; with -pc_type ilu it is GMRES preconditioned by
 * ILU(0). Retained chiefly as a baseline: the moment rows couple each fluid
 * unknown to an entire kinetic block and are dense, which is what the split
 * configurations below exist to isolate.
 */
void MILU_PROVA::ConfigureMonolithic()
{
    SetDefaultOption("-pc_type", "ilu");
    SetDefaultOption("-pc_factor_levels", "0");
}

/**
 * Configuration 2: kinetic and fluid separated.
 *
 * Rows [0, Nhot+Nre) hold the distribution functions and the remainder the
 * fluid and scalar quantities. The fluid block is O(N_r) in size and is
 * factorised directly; the kinetic block carries essentially the whole system
 * and is where the choice of preconditioner matters.
 *
 * The split is worth making even though solving the fluid block exactly saves
 * little on its own: it keeps the dense moment rows out of the kinetic block,
 * whose coarsening or factorisation would otherwise have to contend with rows
 * connected to every unknown.
 */
void MILU_PROVA::ConfigureSplitKinetic()
{
    PC pc;
    IS is_kinetic, is_fluid;

    PetscInt Nk = this->Nhot + this->Nre;
    PetscInt Ntot = this->xn;

    ISCreateStride(PETSC_COMM_WORLD, Nk, 0, 1, &is_kinetic);
    ISCreateStride(PETSC_COMM_WORLD, Ntot - Nk, Nk, 1, &is_fluid);

    KSPGetPC(this->ksp, &pc);
    PCSetType(pc, PCFIELDSPLIT);
    PCFieldSplitSetIS(pc, "kinetic", is_kinetic);
    PCFieldSplitSetIS(pc, "fluid", is_fluid);

    // PCFieldSplitSetIS retains its own reference, so the local handles are
    // released immediately rather than at the end of the function.
    ISDestroy(&is_kinetic);
    ISDestroy(&is_fluid);

    // Block-diagonal: the coupling blocks are omitted from the preconditioner.
    // The operator itself is unchanged, so the solution is unaffected; only
    // the convergence rate depends on this choice. The couplings are low rank
    // -- each moment couples one fluid unknown to an entire kinetic block --
    // so discarding them displaces comparatively few eigenvalues.
    PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE);

    SetDefaultOption("-fieldsplit_kinetic_ksp_type", "preonly");
    SetDefaultOption("-fieldsplit_kinetic_pc_type", "ilu");
    SetDefaultOption("-fieldsplit_kinetic_pc_factor_levels", "0");

    SetDefaultOption("-fieldsplit_fluid_ksp_type", "preonly");
    SetDefaultOption("-fieldsplit_fluid_pc_type", "lu");
}

/**
 * Configuration 3: the two kinetic populations separated as well.
 *
 * The operator has a different character on each: on f_hot the collision
 * frequencies scale as p^-3 and are large at thermal momenta, so the block is
 * diffusion-dominated and elliptic-like, which is the regime algebraic
 * multigrid is built for. On f_re those frequencies are negligible and the
 * equation is essentially pure advection, where error is not smooth along the
 * flow direction and coarsening has little to remove; incomplete LU is both
 * cheaper and better suited there.
 *
 * The defaults below therefore differ between the two blocks. Whether the
 * distinction earns its keep is a question for measurement -- the control is
 * to exchange the two preconditioners from the command line and confirm that
 * the result is worse.
 */
void MILU_PROVA::ConfigureSplitPopulations()
{
    PC pc;
    IS is_fhot, is_fre, is_fluid;

    PetscInt Nhot = this->Nhot;
    PetscInt Nre = this->Nre;
    PetscInt Ntot = this->xn;

    ISCreateStride(PETSC_COMM_WORLD, Nhot, 0, 1, &is_fhot);
    ISCreateStride(PETSC_COMM_WORLD, Nre, Nhot, 1, &is_fre);
    ISCreateStride(PETSC_COMM_WORLD, Ntot - Nhot - Nre, Nhot + Nre, 1, &is_fluid);

    KSPGetPC(this->ksp, &pc);
    PCSetType(pc, PCFIELDSPLIT);
    PCFieldSplitSetIS(pc, "fhot", is_fhot);
    PCFieldSplitSetIS(pc, "fre", is_fre);
    PCFieldSplitSetIS(pc, "fluid", is_fluid);

    ISDestroy(&is_fhot);
    ISDestroy(&is_fre);
    ISDestroy(&is_fluid);

    PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE);

    SetDefaultOption("-fieldsplit_fhot_ksp_type", "preonly");
#ifdef PETSC_HAVE_HYPRE
    SetDefaultOption("-fieldsplit_fhot_pc_type", "hypre");
    SetDefaultOption("-fieldsplit_fhot_pc_hypre_type", "boomeramg");
    // The default strength threshold of 0.25 coarsens in every direction
    // alike. The operator is anisotropic -- the couplings in p and in xi0
    // differ by orders of magnitude, and which dominates varies across the
    // domain -- so a more selective threshold is used.
    SetDefaultOption("-fieldsplit_fhot_pc_hypre_boomeramg_strong_threshold", "0.5");
#else
    // hypre unavailable: fall back on PETSc's smoothed-aggregation multigrid.
    SetDefaultOption("-fieldsplit_fhot_pc_type", "gamg");
#endif

    SetDefaultOption("-fieldsplit_fre_ksp_type", "preonly");
    SetDefaultOption("-fieldsplit_fre_pc_type", "ilu");
    SetDefaultOption("-fieldsplit_fre_pc_factor_levels", "0");

    SetDefaultOption("-fieldsplit_fluid_ksp_type", "preonly");
    SetDefaultOption("-fieldsplit_fluid_pc_type", "lu");
}

/**
 * One-time construction of the solver.
 *
 * Separated from Invert() because the splits must be built exactly once:
 * calling PCFieldSplitSetIS on a preconditioner that has already been set up
 * appends further splits rather than replacing them, and the resulting
 * inconsistent state faults on the second solve. The sparsity pattern of the
 * matrix is fixed at construction, so nothing here depends on the timestep.
 *
 * The configuration is chosen by -dream_split:
 *
 *   none         no splitting; -pc_type governs the whole system
 *   kinetic      kinetic block separated from fluid and scalar
 *   populations  f_hot, f_re and fluid separated  (default)
 */
void MILU_PROVA::ConfigureSolver()
{
    PetscInt Nhot = this->Nhot;
    PetscInt Nre = this->Nre;
    PetscInt Ntot = this->xn;

    char split[32] = "populations";
    PetscBool set;
    PetscOptionsGetString(NULL, NULL, "-dream_split", split, sizeof(split), &set);

    this->ConfigureOuter();

    if (std::strcmp(split, "none") == 0)
    {
        this->ConfigureMonolithic();
    }
    else
    {
        // The stride-based index sets below assume the distribution functions
        // occupy the leading rows, in the order f_hot then f_re. This is
        // arranged by the block ordering in SolverLinearlyImplicit; the check
        // here catches the case where those sizes were never set correctly.
        if (Nhot <= 0 || Nhot + Nre >= Ntot)
            throw FVMException(
                "MILU_PROVA: invalid block sizes Nhot=%d Nre=%d (Ntot=%d).",
                (int)Nhot, (int)Nre, (int)Ntot);

        if (std::strcmp(split, "kinetic") == 0)
            this->ConfigureSplitKinetic();
        else if (std::strcmp(split, "populations") == 0)
            this->ConfigureSplitPopulations();
        else
            throw FVMException(
                "MILU_PROVA: unrecognised -dream_split '%s' "
                "(expected none, kinetic or populations).",
                split);
    }

    // Read last, so that the defaults registered above are picked up, while
    // anything the user supplied has already been left untouched by
    // SetDefaultOption.
    KSPSetFromOptions(this->ksp);

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
void MILU_PROVA::Invert(Matrix *A, Vec *b, Vec *x)
{
    // The matrix entries change every timestep, but its sparsity pattern does
    // not, so the operators are reset while the preconditioner structure is
    // built only once.
    KSPSetOperators(this->ksp, A->mat(), A->mat());

    if (!this->configured)
        this->ConfigureSolver();

    this->errorcode = KSPSolve(this->ksp, *b, *x);

    // KSPSolve returns successfully even when the iteration failed to
    // converge, so the solution must be validated explicitly. Without this
    // check a diverged solve silently propagates a non-solution into the
    // next timestep.
    KSPConvergedReason reason;
    KSPGetConvergedReason(this->ksp, &reason);

    if (reason < 0)
    {
        PetscInt its;
        KSPGetIterationNumber(this->ksp, &its);

        throw FVMException(
            "MILU_PROVA: linear solve did not converge after %d iterations "
            "(KSPConvergedReason %d: %s).",
            (int)its, (int)reason, KSPConvergedReasons[reason]);
    }
}