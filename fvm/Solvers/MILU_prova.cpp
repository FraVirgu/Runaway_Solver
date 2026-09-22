/**
 * Implementation of matrix invertor based on LU
 * factorization (i.e. direct inversion).
 */

#include <algorithm>
#include <iostream>
#include <vector>
#include <petscvec.h>
#include <petscmat.h>
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

    // ISCreateStride(PETSC_COMM_WORLD, n, ...) interprets 'n' as THIS RANK'S
    // local contribution, not a global count to be auto-split -- passing the
    // same global field size on every rank silently builds an index set
    // whose layout has no relation to how the operator matrix is actually
    // partitioned (MatLoad, or any other assembly path, may split rows
    // anywhere; the field boundaries need not land on a rank boundary).
    // PCFieldSplitSetIS() / the MatCreateSubMatrix() it triggers then walks
    // off into invalid memory instead of erroring cleanly. Each rank must
    // therefore claim only the part of each field that falls within its own
    // locally-owned row range of the operator matrix.
    Mat A;
    KSPGetOperators(this->ksp, &A, nullptr);
    PetscInt rstart, rend;
    MatGetOwnershipRange(A, &rstart, &rend);

    PetscInt kinetic_lo = std::max(rstart, (PetscInt)0);
    PetscInt kinetic_hi = std::min(rend, Nk);
    PetscInt n_kinetic_local = std::max(kinetic_hi - kinetic_lo, (PetscInt)0);

    PetscInt fluid_lo = std::max(rstart, Nk);
    PetscInt fluid_hi = std::min(rend, Ntot);
    PetscInt n_fluid_local = std::max(fluid_hi - fluid_lo, (PetscInt)0);

    ISCreateStride(PETSC_COMM_WORLD, n_kinetic_local, kinetic_lo, 1, &is_kinetic);
    ISCreateStride(PETSC_COMM_WORLD, n_fluid_local, fluid_lo, 1, &is_fluid);

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

    // See the comment in ConfigureSplitKinetic(): each rank must claim only
    // the part of each field that falls within its own locally-owned row
    // range of the operator matrix, not the field's full global size.
    Mat A;
    KSPGetOperators(this->ksp, &A, nullptr);
    PetscInt rstart, rend;
    MatGetOwnershipRange(A, &rstart, &rend);

    PetscInt fhot_lo = std::max(rstart, (PetscInt)0);
    PetscInt fhot_hi = std::min(rend, Nhot);
    PetscInt n_fhot_local = std::max(fhot_hi - fhot_lo, (PetscInt)0);

    PetscInt fre_lo = std::max(rstart, Nhot);
    PetscInt fre_hi = std::min(rend, Nhot + Nre);
    PetscInt n_fre_local = std::max(fre_hi - fre_lo, (PetscInt)0);

    PetscInt fluid_lo = std::max(rstart, Nhot + Nre);
    PetscInt fluid_hi = std::min(rend, Ntot);
    PetscInt n_fluid_local = std::max(fluid_hi - fluid_lo, (PetscInt)0);

    ISCreateStride(PETSC_COMM_WORLD, n_fhot_local, fhot_lo, 1, &is_fhot);
    ISCreateStride(PETSC_COMM_WORLD, n_fre_local, fre_lo, 1, &is_fre);
    ISCreateStride(PETSC_COMM_WORLD, n_fluid_local, fluid_lo, 1, &is_fluid);

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
 * Apply an ILU(1) preconditioner M to a sequential system, producing the
 * explicit preconditioned operator MA_seq = M^{-1} A_seq and right-hand side
 * Mb_seq = M^{-1} b_seq.
 *
 * A_seq, b_seq are left untouched; the results are placed in newly created
 * *MA_seq / *Mb_seq (the caller owns them and must MatDestroy/VecDestroy).
 *
 * M^{-1} A_seq is formed one column at a time by applying the preconditioner
 * to each column of a dense copy of A_seq -- PCApply has no sparse-matrix
 * form, only Vec-to-Vec -- and the result is then thresholded (entries with
 * |value| < dropTol are dropped) and packed back into a sparse matrix, since
 * M^{-1} generally fills in what was a sparse operator.
 *
 * A_seq:   sequential system matrix (PETSC_COMM_SELF).
 * b_seq:   sequential right-hand side, same size as A_seq.
 * MA_seq:  on return, holds the sparsified M^{-1} A_seq (SeqAIJ).
 * Mb_seq:  on return, holds M^{-1} b_seq.
 * dropTol: entries of M^{-1} A_seq smaller than this in magnitude are
 *          treated as fill-in noise and discarded when sparsifying.
 */
void MILU_PROVA::ApplyILUPreconditioning(
    Mat A_seq, Vec b_seq, Mat *MA_seq, Vec *Mb_seq, PetscReal dropTol)
{
    // ---- 1. ILU factorization ----
    KSP ksp_ilu;
    PC pc_ilu;
    KSPCreate(PETSC_COMM_SELF, &ksp_ilu);
    KSPSetOperators(ksp_ilu, A_seq, A_seq);
    KSPGetPC(ksp_ilu, &pc_ilu);
    PCSetType(pc_ilu, PCILU);
    PCFactorSetLevels(pc_ilu, 1);
    KSPSetFromOptions(ksp_ilu);

    KSPSetUp(ksp_ilu); // factorization happens here

    // ---- 2. Mb = M^{-1} b  and  MA = M^{-1} A ----
    VecDuplicate(b_seq, Mb_seq);
    PCApply(pc_ilu, b_seq, *Mb_seq);

    PetscInt n, nc;
    MatGetSize(A_seq, &n, &nc);

    // Dense copy of A, then X = M^{-1} A column by column
    Mat Ad, X;
    MatConvert(A_seq, MATSEQDENSE, MAT_INITIAL_MATRIX, &Ad);
    MatCreateSeqDense(PETSC_COMM_SELF, n, n, NULL, &X);
    for (PetscInt j = 0; j < n; j++)
    {
        Vec colA, colX;
        MatDenseGetColumnVecRead(Ad, j, &colA);
        MatDenseGetColumnVecWrite(X, j, &colX);
        PCApply(pc_ilu, colA, colX);
        MatDenseRestoreColumnVecWrite(X, j, &colX);
        MatDenseRestoreColumnVecRead(Ad, j, &colA);
    }

    // Sparsify X (drop |entry| < dropTol) into a SeqAIJ matrix
    const PetscScalar *xv;
    PetscInt lda;
    MatDenseGetLDA(X, &lda);
    MatDenseGetArrayRead(X, &xv); // column-major: X(i,j) = xv[i + j*lda]

    std::vector<PetscInt> nnz(n, 0);
    for (PetscInt j = 0; j < n; j++)
        for (PetscInt i = 0; i < n; i++)
            if (PetscAbsScalar(xv[i + j * lda]) >= dropTol)
                nnz[i]++;

    MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 0, nnz.data(), MA_seq);
    std::vector<PetscInt> cols;
    std::vector<PetscScalar> vals;
    for (PetscInt i = 0; i < n; i++)
    {
        cols.clear();
        vals.clear();
        for (PetscInt j = 0; j < n; j++)
        {
            PetscScalar v = xv[i + j * lda];
            if (PetscAbsScalar(v) >= dropTol)
            {
                cols.push_back(j);
                vals.push_back(v);
            }
        }
        MatSetValues(*MA_seq, 1, &i, (PetscInt)cols.size(), cols.data(), vals.data(), INSERT_VALUES);
    }
    MatDenseRestoreArrayRead(X, &xv);
    MatAssemblyBegin(*MA_seq, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*MA_seq, MAT_FINAL_ASSEMBLY);

    MatDestroy(&Ad);
    MatDestroy(&X);
    KSPDestroy(&ksp_ilu); // only now, after all PCApply calls
}

/**
 * Plain (unpreconditioned) GMRES on an already-distributed system Ax = b.
 *
 * Builds and destroys its own KSP -- independent of this->ksp and Invert()
 * -- for solving a system that has already been preconditioned explicitly
 * (e.g. by ApplyILUPreconditioning() on rank 0, then distributed), so no
 * second preconditioner is applied by default; PCNONE may still be
 * overridden from the command line via -pc_type.
 *
 * A, b: the distributed operator and right-hand side.
 * x:    solution vector. Contains the solution on return.
 * step, my_rank: only used for the status line, printed on rank 0.
 * gmresSetupStage, solveStage: PetscLogStage handles registered by the
 *   caller, so KSPSetUp/KSPSolve are attributed to the caller's own
 *   -log_view stages rather than an unnamed default stage.
 *
 * Returns the combined KSPSetUp + KSPSolve wall-clock time in seconds.
 */
double MILU_PROVA::Solve_GMRES(
    Mat A, Vec b, Vec x, int step, int my_rank,
    PetscLogStage gmresSetupStage, PetscLogStage solveStage)
{
    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, 100);
    KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1000);
    KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);

    // The system is already preconditioned by the rank-0 ILU,
    // so no second preconditioner by default (override with -pc_type)
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCNONE);
    KSPSetFromOptions(ksp);

    MPI_Barrier(PETSC_COMM_WORLD);
    double tS0 = MPI_Wtime();
    PetscLogStagePush(gmresSetupStage);
    KSPSetUp(ksp);
    PetscLogStagePop();
    double tSetup = MPI_Wtime() - tS0;

    MPI_Barrier(PETSC_COMM_WORLD);
    double t0 = MPI_Wtime();
    PetscLogStagePush(solveStage);
    KSPSolve(ksp, b, x);
    PetscLogStagePop();
    double tSolve = MPI_Wtime() - t0;

    PetscInt its;
    KSPConvergedReason reason;
    KSPGetIterationNumber(ksp, &its);
    KSPGetConvergedReason(ksp, &reason);
    if (my_rank == 0)
        std::cout << "step " << step << ": GMRES its = " << its
                   << ", " << (reason > 0 ? "converged" : "DIVERGED")
                   << " (reason " << reason << ")"
                   << ", setup " << tSetup << " s, solve " << tSolve << " s" << std::endl;

    KSPDestroy(&ksp);

    return tSetup + tSolve;
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