/**
 * Implementation of matrix inverter using FGMRES preconditioned by block
 * Jacobi over the radius, each block solved with GMRES + hypre/BoomerAMG --
 * the configuration benchmarked in iface/TestKineticHypreRadius.cpp
 * (variant 2b, options prefix radbjin_ there), here usable as a DREAM
 * linear solver.
 *
 * Ordering. Each kinetic unknown is stored radius-major, so the rows of
 * radius ir are
 *     f_hot: [ir*nh, (ir+1)*nh)               nh  = Nhot/Nr
 *     f_re:  [Nhot + ir*nre, Nhot+(ir+1)*nre) nre = Nre/Nr
 * which are not contiguous. The system is therefore permuted to radius-major
 * order (hot then re, per radius), with whole radii assigned to each rank,
 * so that every block is local to one rank. With more ranks than radii
 * (number of ranks a multiple of Nr, kinetic-only systems) each radius is
 * split over size/Nr ranks instead, and its block is solved by a parallel
 * BoomerAMG on that sub-communicator.
 *
 * Options (all also settable through PETSC_OPTIONS):
 *   -dream_rbj_radii_per_block <n>   radial cells per block (default 1)
 *   -dream_rbj_ksp_*                 outer FGMRES (rtol 1e-10, restart 100)
 *   -dream_rbj_sub_ksp_*             inner GMRES  (rtol 1e-6, max 500 its)
 *   -dream_rbj_sub_pc_hypre_*        BoomerAMG of the blocks
 */

#include <algorithm>
#include <vector>
#include <petscvec.h>
#include "FVM/config.h"
#include "FVM/FVMException.hpp"
#include "FVM/Matrix.hpp"
#include "FVM/Solvers/MIRadialBJ.hpp"

using namespace DREAM::FVM;

/**
 * Constructor.
 *
 * n:    Number of elements in solution vector.
 * Nhot: Number of f_hot rows (at offset 0).
 * Nre:  Number of f_re rows (at offset Nhot).
 * Nr:   Number of radial cells.
 */
MIRadialBJ::MIRadialBJ(const len_t n, const len_t Nhot, const len_t Nre, const len_t Nr)
    : xn(n), Nhot(Nhot), Nre(Nre), Nr(Nr) {

    if (Nr == 0 || Nhot % Nr != 0 || Nre % Nr != 0)
        throw FVMException(
            "MIRadialBJ: Nhot = " LEN_T_PRINTF_FMT " and Nre = " LEN_T_PRINTF_FMT
            " must both be multiples of Nr = " LEN_T_PRINTF_FMT ".", Nhot, Nre, Nr);

    KSPCreate(PETSC_COMM_WORLD, &this->ksp);
}

/**
 * Destructor.
 */
MIRadialBJ::~MIRadialBJ() {
    KSPDestroy(&this->ksp);
    if (this->scatter) VecScatterDestroy(&this->scatter);
    if (this->isPerm) ISDestroy(&this->isPerm);
    if (this->Ap) MatDestroy(&this->Ap);
    if (this->bP) VecDestroy(&this->bP);
    if (this->xP) VecDestroy(&this->xP);
    if (this->xPrevP) VecDestroy(&this->xPrevP);
}

/**
 * Radius-major permutation of the system. Rank p owns the radii
 * [p*Nr/size, (p+1)*Nr/size); for each, its nh f_hot rows followed by its
 * nre f_re rows. The fluid rows [Nhot+Nre, n), if any, are appended to the
 * last rank. With more ranks than radii each radius is instead split over
 * k = size/Nr consecutive ranks, rank p holding slice p%k of radius p/k.
 */
void MIRadialBJ::CreatePermutation() {
    PetscMPIInt rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const PetscInt nr = (PetscInt)this->Nr;
    const PetscInt Nk = (PetscInt)(this->Nhot + this->Nre);
    const PetscInt nh = (PetscInt)this->Nhot / nr, nre = (PetscInt)this->Nre / nr;
    const PetscInt nb = nh + nre;
    const PetscInt nFluid = (PetscInt)this->xn - Nk;

    std::vector<PetscInt> idx;
    if (size > nr) {
        if (size % nr != 0)
            throw FVMException(
                "MIRadialBJ: %d ranks is neither at most nor a multiple of Nr = %d.",
                (int)size, (int)nr);
        if (nFluid > 0)
            throw FVMException(
                "MIRadialBJ: more ranks (%d) than radii (%d) is only supported "
                "for kinetic-only systems (-dream_kinetic_only).", (int)size, (int)nr);

        const PetscInt k = size / nr;
        const PetscInt ir = rank / k, s = rank % k;
        const PetscInt j0 = s * nb / k, j1 = (s + 1) * nb / k;

        idx.reserve(j1 - j0);
        for (PetscInt j = j0; j < j1; j++)
            idx.push_back(j < nh ? ir * nh + j : this->Nhot + ir * nre + (j - nh));
        this->nLocalRadii = 0;
    } else {
        const PetscInt r0 = (PetscInt)rank * nr / size;
        const PetscInt r1 = (PetscInt)(rank + 1) * nr / size;

        idx.reserve((r1 - r0) * nb + (rank == size - 1 ? nFluid : 0));
        for (PetscInt ir = r0; ir < r1; ir++) {
            for (PetscInt i = 0; i < nh; i++)
                idx.push_back(ir * nh + i);
            for (PetscInt i = 0; i < nre; i++)
                idx.push_back(this->Nhot + ir * nre + i);
        }
        if (rank == size - 1)
            for (PetscInt i = Nk; i < (PetscInt)this->xn; i++)
                idx.push_back(i);
        this->nLocalRadii = r1 - r0;
    }

    ISCreateGeneral(PETSC_COMM_WORLD, (PetscInt)idx.size(), idx.data(),
                    PETSC_COPY_VALUES, &this->isPerm);
}

/**
 * One-time construction of the solver: outer FGMRES, block Jacobi over the
 * radius and, on every block, GMRES + BoomerAMG. Needs the permuted
 * operator Ap to be set, since the sub-solvers only exist after KSPSetUp().
 *
 * FGMRES rather than GMRES, since the inner Krylov solve makes the
 * preconditioner nonlinear.
 */
void MIRadialBJ::ConfigureSolver() {
#ifndef PETSC_HAVE_HYPRE
    throw FVMException(
        "MIRadialBJ: this PETSc build has no hypre support (PETSC_HAVE_HYPRE "
        "undefined) -- cannot use the boomeramg preconditioner.");
#endif

    PetscMPIInt rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    PetscOptionsGetInt(NULL, NULL, "-dream_rbj_radii_per_block", &this->radiiPerBlock, NULL);
    if (this->radiiPerBlock < 1)
        throw FVMException(
            "MIRadialBJ: -dream_rbj_radii_per_block must be >= 1 (got %d).",
            (int)this->radiiPerBlock);

    const PetscInt nr = (PetscInt)this->Nr;
    const PetscInt nb = (PetscInt)(this->Nhot + this->Nre) / nr;
    const PetscInt nFluid = (PetscInt)(this->xn - this->Nhot - this->Nre);
    const bool fluidHere = (nFluid > 0 && size <= nr && rank == size - 1);

    KSPSetOptionsPrefix(this->ksp, "dream_rbj_");
    KSPSetType(this->ksp, KSPFGMRES);
    KSPGMRESSetRestart(this->ksp, 100);
    KSPSetTolerances(this->ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1000);
    // Start from the previous timestep's solution (see Invert()).
    KSPSetInitialGuessNonzero(this->ksp, PETSC_TRUE);

    PC pc;
    KSPGetPC(this->ksp, &pc);
    PCSetType(pc, PCBJACOBI);
    if (size > nr) {
        // fewer blocks than ranks: each block spans several ranks
        if (nr % this->radiiPerBlock != 0)
            throw FVMException(
                "MIRadialBJ: with more ranks than radii, -dream_rbj_radii_per_block "
                "(%d) must divide Nr = %d.", (int)this->radiiPerBlock, (int)nr);
        PCBJacobiSetTotalBlocks(pc, nr / this->radiiPerBlock, NULL);
    } else {
        // blocks on this rank; the last one also takes the radii left over
        const PetscInt nblk = std::max(this->nLocalRadii / this->radiiPerBlock, (PetscInt)1);
        std::vector<PetscInt> lens(nblk, this->radiiPerBlock * nb);
        lens.back() = (this->nLocalRadii - (nblk - 1) * this->radiiPerBlock) * nb;
        if (fluidHere)
            lens.push_back(nFluid);
        PCBJacobiSetLocalBlocks(pc, (PetscInt)lens.size(), lens.data());
    }

    // Read last, so that anything supplied via PETSC_OPTIONS or the command
    // line overrides the defaults set above.
    KSPSetFromOptions(this->ksp);
    KSPSetUp(this->ksp);

    // Default sub-solvers (the sub-KSPs exist after KSPSetUp); the
    // -dream_rbj_sub_* options override them.
    PetscInt nsub, first;
    KSP *subksp;
    PCBJacobiGetSubKSP(pc, &nsub, &first, &subksp);
    for (PetscInt i = 0; i < nsub; i++) {
        PC subpc;
        KSPGetPC(subksp[i], &subpc);
        if (fluidHere && i == nsub - 1) {
            // fluid block: small, solved directly
            KSPSetType(subksp[i], KSPPREONLY);
            PCSetType(subpc, PCLU);
            continue;
        }
        KSPSetType(subksp[i], KSPGMRES);
        KSPGMRESSetRestart(subksp[i], 100);
        KSPSetTolerances(subksp[i], 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 500);
        KSPSetNormType(subksp[i], KSP_NORM_UNPRECONDITIONED);
        PCSetType(subpc, PCHYPRE);
        PCHYPRESetType(subpc, "boomeramg");
        KSPSetFromOptions(subksp[i]);
    }

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
void MIRadialBJ::Invert(Matrix *A, Vec *b, Vec *x) {
    // Permute to radius-major order. The sparsity pattern is fixed, so the
    // permuted matrix is only refilled with the new values after the first
    // call; the block AMG hierarchies are rebuilt on every solve.
    if (!this->isPerm)
        this->CreatePermutation();
    MatCreateSubMatrix(A->mat(), this->isPerm, this->isPerm,
                       this->Ap ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX, &this->Ap);

    if (!this->scatter) {
        MatCreateVecs(this->Ap, &this->xP, &this->bP);
        VecScatterCreate(*b, this->isPerm, this->bP, NULL, &this->scatter);
    }

    KSPSetOperators(this->ksp, this->Ap, this->Ap);
    if (!this->configured)
        this->ConfigureSolver();

    // old -> new order: bP[k] = b[isPerm[k]]. Invert() is called with
    // b == x, so b is read completely before x is written below.
    VecScatterBegin(this->scatter, *b, this->bP, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(this->scatter, *b, this->bP, INSERT_VALUES, SCATTER_FORWARD);

    // Seed with the previous solution (zero on the very first call).
    if (this->xPrevP)
        VecCopy(this->xPrevP, this->xP);
    else
        VecSet(this->xP, 0.0);

    this->errorcode = KSPSolve(this->ksp, this->bP, this->xP);

    KSPConvergedReason reason;
    KSPGetConvergedReason(this->ksp, &reason);

    if (reason < 0) {
        PetscInt its;
        KSPGetIterationNumber(this->ksp, &its);

        throw FVMException(
            "MIRadialBJ: linear solve did not converge after %d iterations "
            "(KSPConvergedReason %d: %s).",
            (int)its, (int)reason, KSPConvergedReasons[reason]);
    }

    if (!this->xPrevP)
        VecDuplicate(this->xP, &this->xPrevP);
    VecCopy(this->xP, this->xPrevP);

    // back to the old order
    VecScatterBegin(this->scatter, this->xP, *x, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(this->scatter, this->xP, *x, INSERT_VALUES, SCATTER_REVERSE);
}
