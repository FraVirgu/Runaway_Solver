/**
 * Standalone test solving ONLY the kinetic part (f_hot, f_re: rows/columns
 * [0, Nhot+Nre)) of the DREAM linear system, for a run with Nr > 1 radial
 * cells (e.g. examples/diff_grid_paper_kinetic_radial.py). As in
 * TestKineticHypre_split.cpp, all other unknowns are moved to the right-hand
 * side:
 *
 *     A_kk x_k = b_k - A_kf x_f
 *
 * (x_f from the reference solution -ref_solution if it exists, else 0; the
 * reference is also used for the relative error, printed as n/a without it.)
 *
 * Solver: GMRES + block Jacobi over the RADIUS: one block per radial cell,
 * holding the f_hot AND the f_re unknowns of that radius (the momentum-space
 * operator plus the hot/runaway interface), each block approximated by one
 * BoomerAMG V-cycle (preonly). The radial transport coupling between
 * neighbouring radii is left to the outer GMRES. Options prefix radbj_,
 * sub-solver options -radbj_sub_*.
 *
 * Ordering. Each kinetic unknown is stored radius-major, so the rows of
 * radius ir are
 *     f_hot: [ir*nh, (ir+1)*nh)              nh  = Nhot/Nr
 *     f_re:  [Nhot + ir*nre, Nhot+(ir+1)*nre) nre = Nre/Nr
 * which are not contiguous. For the block-Jacobi solves A_kk is therefore
 * permuted to radius-major order (hot then re, per radius), with whole radii
 * assigned to each rank, so that every block is local to one rank. With
 * more ranks than radii (number of ranks a multiple of Nr) each radius is
 * split over size/Nr ranks instead: one block per radius, solved by a
 * parallel BoomerAMG on its sub-communicator.
 *
 * Inputs: petsc_mat_np1_step1_iter1, petsc_rhs_np1_step1_iter1,
 * petsc_block_sizes.txt (Nhot, Nre, Nr) and the optional reference solution.
 * Options:
 *   -ref_solution <file>        reference solution (default petsc_solution_miilu)
 *   -scale none|diag|ruiz       row/column scaling of A_kk before solving
 *   -nr <Nr>                    number of radial cells, if petsc_block_sizes.txt
 *                               has no "Nr" line (older dumps)
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include <petscksp.h>
#include "DREAM/Init.h"

using namespace std;

static void LoadMat(const char *name, Mat *A)
{
    PetscViewer v;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &v);
    MatCreate(PETSC_COMM_WORLD, A);
    MatSetType(*A, MATAIJ);
    MatLoad(*A, v);
    PetscViewerDestroy(&v);
}

static void LoadVec(const char *name, Vec *x)
{
    PetscViewer v;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, name, FILE_MODE_READ, &v);
    VecCreate(PETSC_COMM_WORLD, x);
    VecLoad(*x, v);
    PetscViewerDestroy(&v);
}

enum class Scale
{
    None,
    Diag,
    Ruiz
};

static const char *ScaleName(Scale s)
{
    return s == Scale::None ? "none" : s == Scale::Diag ? "diag"
                                                        : "ruiz";
}

/**
 * Build the scaling vectors L (rows) and R (columns) for A' = L A R.
 *   none  L = R = 1
 *   diag  L = 1/diag(A), R = 1   (zero diagonal entries left unscaled)
 *   ruiz  5 sweeps of simultaneous row/column infinity-norm equilibration:
 *         L <- L / sqrt(max_j |a_ij|),  R <- R / sqrt(max_i |a_ij|)
 */
static void BuildScaling(Mat A, Scale mode, Vec L, Vec R)
{
    VecSet(L, 1.0);
    VecSet(R, 1.0);
    if (mode == Scale::None)
        return;

    if (mode == Scale::Diag)
    {
        MatGetDiagonal(A, L);
        VecAbs(L);
        const PetscScalar *d;
        PetscScalar *l;
        PetscInt n;
        VecGetLocalSize(L, &n);
        VecGetArrayRead(L, &d);
        VecGetArray(L, &l);
        for (PetscInt i = 0; i < n; i++)
            l[i] = d[i] > 0.0 ? 1.0 / d[i] : 1.0;
        VecRestoreArrayRead(L, &d);
        VecRestoreArray(L, &l);
        return;
    }

    // Ruiz
    Mat W, Wt = nullptr;
    MatDuplicate(A, MAT_COPY_VALUES, &W);
    Vec rmax, cmax;
    VecDuplicate(L, &rmax);
    VecDuplicate(R, &cmax);
    for (int it = 0; it < 5; it++)
    {
        MatGetRowMaxAbs(W, rmax, nullptr);
        if (Wt)
            MatTranspose(W, MAT_REUSE_MATRIX, &Wt);
        else
            MatTranspose(W, MAT_INITIAL_MATRIX, &Wt);
        MatGetRowMaxAbs(Wt, cmax, nullptr);

        for (Vec *v : {&rmax, &cmax})
        {
            PetscScalar *a;
            PetscInt n;
            VecGetLocalSize(*v, &n);
            VecGetArray(*v, &a);
            for (PetscInt i = 0; i < n; i++)
                a[i] = a[i] > 0.0 ? 1.0 / PetscSqrtReal(a[i]) : 1.0;
            VecRestoreArray(*v, &a);
        }
        MatDiagonalScale(W, rmax, cmax);
        VecPointwiseMult(L, L, rmax);
        VecPointwiseMult(R, R, cmax);
    }
    VecDestroy(&rmax);
    VecDestroy(&cmax);
    MatDestroy(&W);
    MatDestroy(&Wt);
}

/**
 * Relative residual and relative error of x on the ORIGINAL (unscaled,
 * unpermuted) system A0 x = b, printed as one result line. Without a
 * reference solution (x_ref = nullptr) the error is printed as n/a.
 */
static void Report(Mat A0, Vec b, Vec x, Vec x_ref, Scale mode, const string &label,
                   PetscInt its, KSPConvergedReason reason, double tSetup,
                   double tSolve, int my_rank)
{
    Vec r;
    VecDuplicate(b, &r);
    MatMult(A0, x, r);
    VecAXPY(r, -1.0, b);
    PetscReal resNorm, rhsNorm;
    VecNorm(r, NORM_2, &resNorm);
    VecNorm(b, NORM_2, &rhsNorm);
    PetscReal relRes = rhsNorm > 0 ? resNorm / rhsNorm : resNorm;

    char errStr[16] = "n/a";
    if (x_ref)
    {
        VecWAXPY(r, -1.0, x_ref, x);
        PetscReal errNorm, refNorm;
        VecNorm(r, NORM_2, &errNorm);
        VecNorm(x_ref, NORM_2, &refNorm);
        PetscReal relErr = refNorm > 0 ? errNorm / refNorm : errNorm;
        snprintf(errStr, sizeof(errStr), "%.2e", (double)relErr);
    }
    VecDestroy(&r);

    if (my_rank == 0)
    {
        printf("%-12s %-34s its %5d  %-9s setup %7.3f s  solve %8.3f s  "
               "res %.2e  err %s\n",
               ScaleName(mode), label.c_str(), (int)its,
               reason > 0 ? "converged" : "DIVERGED", tSetup, tSolve,
               (double)relRes, errStr);
        fflush(stdout); // stdout is a file in batch jobs: write the line now
    }
}

/**
 * Radius-major permutation of the kinetic block. Rank p owns the radii
 * [p*Nr/size, (p+1)*Nr/size); for each, its nh f_hot rows followed by its
 * nre f_re rows. The global index set lists, in new order, the OLD row of
 * every new row. *nlocalRadii is the number of radii (blocks) on this rank.
 *
 * With more ranks than radii (size > Nr, size a multiple of Nr) each radius
 * is instead split over k = size/Nr consecutive ranks, rank p holding slice
 * p%k of radius p/k; *nlocalRadii is then 0. The new row of (radius ir,
 * local index j) is still ir*(nh+nre) + j.
 */
static void CreateRadiusPermutation(PetscInt Nhot, PetscInt Nr, PetscInt nh,
                                    PetscInt nre, int my_rank, int size,
                                    IS *isPerm, PetscInt *nlocalRadii)
{
    vector<PetscInt> idx;
    if (size > Nr)
    {
        const PetscInt k = size / Nr, nb = nh + nre;
        const PetscInt ir = my_rank / k, s = my_rank % k;
        const PetscInt j0 = s * nb / k, j1 = (s + 1) * nb / k;

        idx.reserve(j1 - j0);
        for (PetscInt j = j0; j < j1; j++)
            idx.push_back(j < nh ? ir * nh + j : Nhot + ir * nre + (j - nh));
        *nlocalRadii = 0;
    }
    else
    {
        const PetscInt r0 = (PetscInt)my_rank * Nr / size;
        const PetscInt r1 = (PetscInt)(my_rank + 1) * Nr / size;

        idx.reserve((r1 - r0) * (nh + nre));
        for (PetscInt ir = r0; ir < r1; ir++)
        {
            for (PetscInt i = 0; i < nh; i++)
                idx.push_back(ir * nh + i);
            for (PetscInt i = 0; i < nre; i++)
                idx.push_back(Nhot + ir * nre + i);
        }
        *nlocalRadii = r1 - r0;
    }
    ISCreateGeneral(PETSC_COMM_WORLD, (PetscInt)idx.size(), idx.data(),
                    PETSC_COPY_VALUES, isPerm);
}

/**
 * Solve A0 x = rhs0 with block Jacobi over the radius, after scaling
 * (A' = L A0 R) and radius-major permutation (A'' = P A' P^T): outer GMRES,
 * each block approximated by one BoomerAMG V-cycle (preonly). Each block holds radiiPerBlock consecutive radii (the last block of a
 * rank also takes the radii left over). Residual and error are measured on
 * the ORIGINAL system. Returns the iteration count.
 *
 * With more ranks than radii (size > Nr) there are fewer blocks than ranks:
 * PETSc then gives each block a sub-communicator of the size/(Nr/
 * radiiPerBlock) consecutive ranks that own its rows, and the sub-solver
 * (BoomerAMG) runs in parallel on it. The outer Krylov method still runs
 * on all ranks.
 */
static PetscInt SolveRadialBlockJacobi(Mat A0, Vec rhs0, Vec xk_ref, PetscInt Nhot,
                                       PetscInt Nr, PetscInt nh, PetscInt nre,
                                       Scale mode, PetscInt radiiPerBlock,
                                       PetscInt maxits, int my_rank, int size)
{
    Mat As, Ap;
    Vec L, R, rhs, y, x, rhsP, yP;
    MatDuplicate(A0, MAT_COPY_VALUES, &As);
    MatCreateVecs(A0, &y, &rhs);
    VecDuplicate(rhs, &L);
    VecDuplicate(rhs, &R);
    VecDuplicate(rhs, &x);

    BuildScaling(A0, mode, L, R);
    MatDiagonalScale(As, L, R);
    VecPointwiseMult(rhs, L, rhs0);

    // ---- permute to radius-major order, whole radii per rank ----
    IS isPerm;
    PetscInt nlocal;
    CreateRadiusPermutation(Nhot, Nr, nh, nre, my_rank, size, &isPerm, &nlocal);
    MatCreateSubMatrix(As, isPerm, isPerm, MAT_INITIAL_MATRIX, &Ap);
    MatCreateVecs(Ap, &yP, &rhsP);

    // scatter old -> new order: rhsP[k] = rhs[isPerm[k]]
    VecScatter sc;
    VecScatterCreate(rhs, isPerm, rhsP, NULL, &sc);
    VecScatterBegin(sc, rhs, rhsP, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(sc, rhs, rhsP, INSERT_VALUES, SCATTER_FORWARD);
    VecSet(yP, 0.0);

    // ---- outer Krylov + block Jacobi, radiiPerBlock radii per block ----
    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOptionsPrefix(ksp, "radbj_");
    KSPSetOperators(ksp, Ap, Ap);
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, 100);
    KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, maxits);
    KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCBJACOBI);
    if (size > Nr)
        // fewer blocks than ranks: each block spans several ranks
        PCBJacobiSetTotalBlocks(pc, Nr / radiiPerBlock, NULL);
    else
    {
        const PetscInt nblk = max(nlocal / radiiPerBlock, (PetscInt)1); // blocks on this rank
        vector<PetscInt> lens(nblk, radiiPerBlock * (nh + nre));
        lens.back() = (nlocal - (nblk - 1) * radiiPerBlock) * (nh + nre);
        PCBJacobiSetLocalBlocks(pc, nblk, lens.data());
    }
    KSPSetFromOptions(ksp);

    MPI_Barrier(PETSC_COMM_WORLD);
    double t0 = MPI_Wtime();
    KSPSetUp(ksp);

    // Default sub-solvers (the sub-KSPs exist after KSPSetUp); the user's
    // -radbj_sub_* options override them.
    PetscInt nsub, first;
    KSP *subksp;
    PCBJacobiGetSubKSP(pc, &nsub, &first, &subksp);
    for (PetscInt i = 0; i < nsub; i++)
    {
        PC subpc;
        KSPGetPC(subksp[i], &subpc);
        KSPSetType(subksp[i], KSPPREONLY);
        PCSetType(subpc, PCHYPRE);
        PCHYPRESetType(subpc, "boomeramg");
        KSPSetFromOptions(subksp[i]);
    }
    KSPSetUpOnBlocks(ksp);
    MPI_Barrier(PETSC_COMM_WORLD);
    double tSetup = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    KSPSolve(ksp, rhsP, yP);
    MPI_Barrier(PETSC_COMM_WORLD);
    double tSolve = MPI_Wtime() - t0;

    PetscInt its;
    KSPConvergedReason reason;
    KSPGetIterationNumber(ksp, &its);
    KSPGetConvergedReason(ksp, &reason);

    // back to the old order, then x = R y
    VecScatterBegin(sc, yP, y, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(sc, yP, y, INSERT_VALUES, SCATTER_REVERSE);
    VecPointwiseMult(x, R, y);

    char lbl[64];
    snprintf(lbl, sizeof(lbl), "BJ(r,%d/blk) AMG V-cyc / GMRES", (int)radiiPerBlock);
    Report(A0, rhs0, x, xk_ref, mode, string(lbl), its, reason, tSetup, tSolve, my_rank);

    KSPDestroy(&ksp);
    VecScatterDestroy(&sc);
    ISDestroy(&isPerm);
    MatDestroy(&Ap);
    MatDestroy(&As);
    VecDestroy(&yP);
    VecDestroy(&rhsP);
    VecDestroy(&y);
    VecDestroy(&rhs);
    VecDestroy(&L);
    VecDestroy(&R);
    VecDestroy(&x);
    return its;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int my_rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    PETSC_COMM_WORLD = MPI_COMM_WORLD;
    dream_initialize();

#ifndef PETSC_HAVE_HYPRE
    if (my_rank == 0)
        cerr << "This PETSc build has no hypre support." << endl;
    dream_finalize();
    MPI_Finalize();
    return 1;
#endif

    const string matname = "petsc_mat_np1_step1_iter1";
    const string rhsname = "petsc_rhs_np1_step1_iter1";
    char refname[PETSC_MAX_PATH_LEN] = "petsc_solution_miilu";
    PetscOptionsGetString(NULL, NULL, "-ref_solution", refname, sizeof(refname), NULL);

    int haveFiles = 0, haveRef = 0;
    if (my_rank == 0)
    {
        haveFiles = (access(matname.c_str(), F_OK) == 0 &&
                     access(rhsname.c_str(), F_OK) == 0 &&
                     access("petsc_block_sizes.txt", F_OK) == 0);
        haveRef = (access(refname, F_OK) == 0);
    }
    MPI_Bcast(&haveFiles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&haveRef, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!haveFiles)
    {
        if (my_rank == 0)
            cerr << "Missing one of " << matname << ", " << rhsname
                 << ", petsc_block_sizes.txt in the current directory (they "
                    "are written by dreami)."
                 << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }

    Mat A;
    Vec b, xref = nullptr;
    LoadMat(matname.c_str(), &A);
    LoadVec(rhsname.c_str(), &b);
    if (haveRef)
        LoadVec(refname, &xref);

    PetscInt N;
    MatGetSize(A, &N, nullptr);

    PetscInt Nhot = 0, Nre = 0, Nr = 0;
    {
        ifstream f("petsc_block_sizes.txt");
        string name;
        PetscInt value;
        while (f >> name >> value)
        {
            if (name == "Nhot")
                Nhot = value;
            else if (name == "Nre")
                Nre = value;
            else if (name == "Nr")
                Nr = value;
        }
    }
    // -nr overrides (or supplies, for dumps without an "Nr" line)
    PetscOptionsGetInt(NULL, NULL, "-nr", &Nr, NULL);

    const PetscInt Nk = Nhot + Nre;
    if (Nk <= 0 || Nk > N)
    {
        if (my_rank == 0)
            cerr << "Invalid block sizes: Nhot+Nre = " << Nk << ", N = " << N << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }
    if (Nr <= 0 || Nhot % Nr != 0 || Nre % Nr != 0)
    {
        if (my_rank == 0)
            cerr << "Invalid number of radial cells Nr = " << Nr << " (Nhot " << Nhot
                 << ", Nre " << Nre << " must both be multiples of it). Give it "
                                       "with -nr if petsc_block_sizes.txt has no \"Nr\" line."
                 << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }
    const PetscInt nh = Nhot / Nr, nre = Nre / Nr;

    // ---- kinetic / fluid index sets, from each rank's own rows ----
    PetscInt rstart, rend;
    MatGetOwnershipRange(A, &rstart, &rend);
    PetscInt klo = rstart, khi = min(rend, Nk);
    PetscInt nk = max(khi - klo, (PetscInt)0);
    PetscInt flo = max(rstart, Nk), fhi = rend;
    PetscInt nf = max(fhi - flo, (PetscInt)0);

    IS isK, isF;
    ISCreateStride(PETSC_COMM_WORLD, nk, nk > 0 ? klo : 0, 1, &isK);
    ISCreateStride(PETSC_COMM_WORLD, nf, nf > 0 ? flo : 0, 1, &isF);

    const bool haveFluid = (Nk < N);
    const bool useFluid = haveFluid && haveRef;

    Mat Akk, Akf = nullptr;
    MatCreateSubMatrix(A, isK, isK, MAT_INITIAL_MATRIX, &Akk);
    if (useFluid)
        MatCreateSubMatrix(A, isK, isF, MAT_INITIAL_MATRIX, &Akf);

    Vec bk, xf = nullptr, xk_ref = nullptr;
    VecGetSubVector(b, isK, &bk);
    if (useFluid)
        VecGetSubVector(xref, isF, &xf);

    // rhs = b_k - A_kf x_f   (or b_k alone)
    Vec rhs;
    MatCreateVecs(Akk, NULL, &rhs);
    if (useFluid)
    {
        MatMult(Akf, xf, rhs);
        VecAYPX(rhs, -1.0, bk);
    }
    else
        VecCopy(bk, rhs);

    if (my_rank == 0)
    {
        cout << "Full system " << N << " x " << N << ", kinetic block "
             << Nk << " x " << Nk << " (Nhot " << Nhot << ", Nre " << Nre
             << "), " << size << " rank" << (size == 1 ? "" : "s") << endl;
        cout << "Nr = " << Nr << " radial blocks of " << nh + nre
             << " rows (f_hot " << nh << " + f_re " << nre << ")" << endl;
    }
    if (my_rank == 0 && haveFluid && !haveRef)
        cout << "No reference solution (" << refname << "): the fluid unknowns "
                                                        "are taken as 0, so the system solved is A_kk x_k = b_k."
             << endl;

    if (haveRef)
        VecGetSubVector(xref, isK, &xk_ref);

    char scaleStr[16] = "none";
    PetscOptionsGetString(NULL, NULL, "-scale", scaleStr, sizeof(scaleStr), NULL);
    Scale scale = Scale::None;
    if (string(scaleStr) == "diag")
        scale = Scale::Diag;
    else if (string(scaleStr) == "ruiz")
        scale = Scale::Ruiz;
    else if (string(scaleStr) != "none")
    {
        if (my_rank == 0)
            cerr << "Unknown -scale '" << scaleStr << "' (none|diag|ruiz)" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // GMRES + block Jacobi over the radius, one block per radius, each
    // approximated by one BoomerAMG V-cycle
    if (Nr < size && size % Nr != 0)
    {
        if (my_rank == 0)
            cout << "Skipping radial block Jacobi: Nr = " << Nr << " < "
                 << size << " ranks and " << size << " is not a multiple of Nr." << endl;
    }
    else
    {
        // With more ranks than radii each block is split over size/Nr ranks
        // and solved by a parallel BoomerAMG on them.
        if (my_rank == 0 && Nr < size)
            cout << "\nMulti-rank blocks: one block per radius, " << size / Nr
                 << " ranks per block\n" << endl;

        SolveRadialBlockJacobi(Akk, rhs, xk_ref, Nhot, Nr, nh, nre, scale,
                               1, 1000, my_rank, size);
    }

    VecRestoreSubVector(b, isK, &bk);
    if (useFluid)
        VecRestoreSubVector(xref, isF, &xf);
    if (haveRef)
        VecRestoreSubVector(xref, isK, &xk_ref);
    VecDestroy(&rhs);
    MatDestroy(&Akk);
    if (useFluid)
        MatDestroy(&Akf);
    ISDestroy(&isK);
    ISDestroy(&isF);
    MatDestroy(&A);
    VecDestroy(&b);
    if (haveRef)
        VecDestroy(&xref);

    dream_finalize();
    MPI_Finalize();
    return 0;
}
