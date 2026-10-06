/**
 * Standalone test solving ONLY the kinetic part (f_hot, f_re: rows/columns
 * [0, Nhot+Nre)) of the DREAM linear system with GMRES + hypre/BoomerAMG.
 * All other unknowns (fluid and scalar quantities) are taken as known and
 * moved to the right-hand side:
 *
 *     A_kk x_k = b_k - A_kf x_f
 *
 * If a reference solution of the FULL system exists (the file given by
 * -ref_solution, default petsc_solution_miilu, as written by testhypre), x_f
 * is taken from it and the solved x_k is compared with its kinetic part.
 * If it does not exist, nothing else needs to be run first: the fluid
 * unknowns are taken as 0 and the reference is a GMRES + BoomerAMG solve (diag
 * scaling, tight tolerance) of the same kinetic system, computed here. If the
 * matrix is kinetic-only (Nhot+Nre == N, written by
 * dreami with -dream_kinetic_only) there is no fluid block at all.
 *
 * One source, two executables (see iface/CMakeLists.txt):
 *   testkinetichypreserial    compiled with KINETIC_SERIAL: must be run with
 *                             `mpirun -n 1`, throws an error otherwise.
 *   testkinetichypreparallel  any rank count. A, b and the reference are
 *                             loaded distributed with MatLoad/VecLoad on
 *                             PETSC_COMM_WORLD (PETSc's own row split), the
 *                             kinetic/fluid index sets are built from each
 *                             rank's locally owned rows.
 *
 * Inputs: petsc_mat_serial_step1_iter1, petsc_rhs_serial_step1_iter1,
 * petsc_block_sizes.txt (Nhot, Nre) and the reference solution file.
 * Options (PETSC_OPTIONS / command line):
 *   -ref_solution <file>        reference solution (default petsc_solution_miilu)
 *   -scale none|diag|ruiz       row/column scaling of A_kk before AMG
 *   -sweep                      every scaling x a set of BoomerAMG settings,
 *                               one result line each (iteration cap 400)
 * Other PC/KSP options can be overridden through PETSC_OPTIONS as usual.
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


enum class Scale { None, Diag, Ruiz };

static const char *ScaleName(Scale s)
{
    return s == Scale::None ? "none" : s == Scale::Diag ? "diag" : "ruiz";
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
        VecShift(L, 0.0);
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
 * Solve A0 x = rhs0 with GMRES + BoomerAMG after scaling (A' = L A0 R,
 * rhs' = L rhs0, x = R x'), with extra PETSc options applied for this solve
 * only. Prints one result line; residual and error are measured on the
 * ORIGINAL (unscaled) kinetic system. Returns the iteration count.
 */
static PetscInt SolveConfig(Mat A0, Vec rhs0, Vec xk_ref, Scale mode,
                            const vector<pair<string, string>> &opts,
                            const string &label, PetscInt maxits, int my_rank)
{
    Mat As;
    Vec L, R, rhs, y, x, r;
    MatDuplicate(A0, MAT_COPY_VALUES, &As);
    MatCreateVecs(A0, &y, &rhs);
    VecDuplicate(rhs, &L);
    VecDuplicate(rhs, &R);
    VecDuplicate(rhs, &x);
    VecDuplicate(rhs, &r);

    BuildScaling(A0, mode, L, R);
    MatDiagonalScale(As, L, R);
    VecPointwiseMult(rhs, L, rhs0);
    VecSet(y, 0.0);

    for (auto &o : opts)
        PetscOptionsSetValue(NULL, o.first.c_str(), o.second.empty() ? NULL : o.second.c_str());

    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, As, As);
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, 100);
    KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, maxits);
    KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCHYPRE);
    PCHYPRESetType(pc, "boomeramg");
    KSPSetFromOptions(ksp);

    MPI_Barrier(PETSC_COMM_WORLD);
    double t0 = MPI_Wtime();
    KSPSetUp(ksp);
    MPI_Barrier(PETSC_COMM_WORLD);
    double tSetup = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    KSPSolve(ksp, rhs, y);
    MPI_Barrier(PETSC_COMM_WORLD);
    double tSolve = MPI_Wtime() - t0;

    PetscInt its;
    KSPConvergedReason reason;
    KSPGetIterationNumber(ksp, &its);
    KSPGetConvergedReason(ksp, &reason);

    VecPointwiseMult(x, R, y); // x = R y

    MatMult(A0, x, r);
    VecAXPY(r, -1.0, rhs0);
    PetscReal resNorm, rhsNorm;
    VecNorm(r, NORM_2, &resNorm);
    VecNorm(rhs0, NORM_2, &rhsNorm);
    PetscReal relRes = rhsNorm > 0 ? resNorm / rhsNorm : resNorm;

    VecWAXPY(r, -1.0, xk_ref, x);
    PetscReal errNorm, refNorm;
    VecNorm(r, NORM_2, &errNorm);
    VecNorm(xk_ref, NORM_2, &refNorm);
    PetscReal relErr = refNorm > 0 ? errNorm / refNorm : errNorm;

    if (my_rank == 0)
        printf("%-12s %-34s its %5d  %-9s setup %7.3f s  solve %8.3f s  "
               "res %.2e  err %.2e\n",
               ScaleName(mode), label.c_str(), (int)its,
               reason > 0 ? "converged" : "DIVERGED", tSetup, tSolve,
               (double)relRes, (double)relErr);

    for (auto &o : opts)
        PetscOptionsClearValue(NULL, o.first.c_str());

    KSPDestroy(&ksp);
    MatDestroy(&As);
    VecDestroy(&y);
    VecDestroy(&rhs);
    VecDestroy(&L);
    VecDestroy(&R);
    VecDestroy(&x);
    VecDestroy(&r);
    return its;
}

/**
 * Reference solution of A x = b with GMRES + hypre/BoomerAMG (the same solver
 * as the runs being tested), after row scaling with 1/|diag(A)| and to a tight
 * tolerance. Used when no reference solution file is available. The scaling
 * and the settings of this solve are fixed (under the prefix "amgref_"), so
 * that -scale and the other options only affect the runs under test.
 */
static void SolveAMGReference(Mat A0, Vec b, Vec x, int my_rank)
{
    Mat As;
    Vec L, R, rhs, y;
    MatDuplicate(A0, MAT_COPY_VALUES, &As);
    MatCreateVecs(A0, &y, &rhs);
    VecDuplicate(rhs, &L);
    VecDuplicate(rhs, &R);

    // A' = L A, rhs' = L b (R = 1 for the diagonal scaling, so x = y)
    BuildScaling(A0, Scale::Diag, L, R);
    MatDiagonalScale(As, L, R);
    VecPointwiseMult(rhs, L, b);
    VecSet(y, 0.0);

    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOptionsPrefix(ksp, "amgref_");
    KSPSetOperators(ksp, As, As);
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, 100);
    KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1000);
    KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCHYPRE);
    PCHYPRESetType(pc, "boomeramg");
    KSPSetFromOptions(ksp);

    MPI_Barrier(PETSC_COMM_WORLD);
    double t0 = MPI_Wtime();
    KSPSetUp(ksp);
    KSPSolve(ksp, rhs, y);
    MPI_Barrier(PETSC_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    PetscInt its;
    KSPConvergedReason reason;
    KSPGetIterationNumber(ksp, &its);
    KSPGetConvergedReason(ksp, &reason);
    if (my_rank == 0)
        printf("reference (GMRES + BoomerAMG, diag scaling, rtol 1e-12): its %d, %s, %.3f s\n",
               (int)its, reason > 0 ? "converged" : "DIVERGED", elapsed);

    VecPointwiseMult(x, R, y); // x = R y
    KSPDestroy(&ksp);
    MatDestroy(&As);
    VecDestroy(&y);
    VecDestroy(&rhs);
    VecDestroy(&L);
    VecDestroy(&R);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int my_rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef KINETIC_SERIAL
    if (size != 1)
    {
        if (my_rank == 0)
            cerr << "testkinetichypreserial: error: must be run with a single "
                    "MPI rank (mpirun -n 1), but was launched with "
                 << size << " ranks." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
#endif

    PETSC_COMM_WORLD = MPI_COMM_WORLD;
    dream_initialize();

#ifndef PETSC_HAVE_HYPRE
    if (my_rank == 0)
        cerr << "This PETSc build has no hypre support." << endl;
    dream_finalize();
    MPI_Finalize();
    return 1;
#endif

    const string matname = "petsc_mat_serial_step1_iter1";
    const string rhsname = "petsc_rhs_serial_step1_iter1";
    char refname[PETSC_MAX_PATH_LEN] = "petsc_solution_miilu";
    PetscOptionsGetString(NULL, NULL, "-ref_solution", refname, sizeof(refname), NULL);

    // The matrix, right-hand side and block sizes are required. The
    // reference solution is optional: without it, the reference is computed
    // here by solving the same kinetic system with GMRES + BoomerAMG.
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

    PetscInt Nhot = 0, Nre = 0;
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
        }
    }
    const PetscInt Nk = Nhot + Nre;
    if (Nk <= 0 || Nk > N)
    {
        if (my_rank == 0)
            cerr << "Invalid block sizes: Nhot+Nre = " << Nk << ", N = " << N << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }

    // ---- kinetic / fluid index sets, from each rank's own rows ----
    // (ISCreateStride's n is the rank-LOCAL size, so each rank claims only
    // the part of each field inside its ownership range.)
    PetscInt rstart, rend;
    MatGetOwnershipRange(A, &rstart, &rend);
    PetscInt klo = rstart, khi = min(rend, Nk);
    PetscInt nk = max(khi - klo, (PetscInt)0);
    PetscInt flo = max(rstart, Nk), fhi = rend;
    PetscInt nf = max(fhi - flo, (PetscInt)0);

    IS isK, isF;
    ISCreateStride(PETSC_COMM_WORLD, nk, nk > 0 ? klo : 0, 1, &isK);
    ISCreateStride(PETSC_COMM_WORLD, nf, nf > 0 ? flo : 0, 1, &isF);

    // Nk == N: kinetic-only system, no fluid block / coupling
    const bool haveFluid = (Nk < N);

    // The fluid unknowns x_f are only known when a reference solution of the
    // full system was loaded; otherwise they are taken as zero.
    const bool useFluid = haveFluid && haveRef;

    Mat Akk, Akf = nullptr;
    MatCreateSubMatrix(A, isK, isK, MAT_INITIAL_MATRIX, &Akk);
    if (useFluid)
        MatCreateSubMatrix(A, isK, isF, MAT_INITIAL_MATRIX, &Akf);

    Vec bk, xf = nullptr, xk_ref = nullptr;
    VecGetSubVector(b, isK, &bk);
    if (useFluid)
        VecGetSubVector(xref, isF, &xf);

    // rhs = b_k - A_kf x_f   (or b_k alone, see above)
    Vec rhs, x;
    MatCreateVecs(Akk, &x, &rhs);
    if (useFluid)
    {
        MatMult(Akf, xf, rhs);
        VecAYPX(rhs, -1.0, bk);
    }
    else
        VecCopy(bk, rhs);
    VecSet(x, 0.0);

    if (my_rank == 0)
        cout << "Full system " << N << " x " << N << ", kinetic block "
             << Nk << " x " << Nk << " (Nhot " << Nhot << ", Nre " << Nre
             << "), " << size << " rank" << (size == 1 ? "" : "s") << endl;
    if (my_rank == 0 && haveFluid && !haveRef)
        cout << "No reference solution (" << refname << "): the fluid unknowns "
                "are taken as 0, so the system solved is A_kk x_k = b_k." << endl;

    // Reference kinetic solution: the kinetic part of the loaded full-system
    // solution, or else a GMRES + BoomerAMG solve of this very system.
    if (haveRef)
        VecGetSubVector(xref, isK, &xk_ref);
    else
    {
        VecDuplicate(rhs, &xk_ref);
        SolveAMGReference(Akk, rhs, xk_ref, my_rank);
    }

    // ---- GMRES + hypre/BoomerAMG on the kinetic block ----
    // -scale none|diag|ruiz   row/column scaling applied before AMG
    // -sweep                  run every scaling x a set of hypre settings
    char scaleStr[16] = "none";
    PetscOptionsGetString(NULL, NULL, "-scale", scaleStr, sizeof(scaleStr), NULL);
    PetscBool sweep = PETSC_FALSE;
    PetscOptionsHasName(NULL, NULL, "-sweep", &sweep);

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

    if (!sweep)
    {
        SolveConfig(Akk, rhs, xk_ref, scale, {}, "default hypre", 1000, my_rank);
    }
    else
    {
        // iteration cap lowered so a bad configuration does not run for minutes
        const PetscInt cap = 400;
        const string P = "-pc_hypre_boomeramg_";
        typedef vector<pair<string, string>> Opts;
        vector<pair<string, Opts>> configs = {
            {"default", {}},
            {"strong_threshold 0.5", {{P + "strong_threshold", "0.5"}}},
            {"strong_threshold 0.7", {{P + "strong_threshold", "0.7"}}},
            {"Jacobi w=0.7", {{P + "relax_type_all", "Jacobi"},
                              {P + "relax_weight_all", "0.7"}}},
            {"l1scaled-Jacobi", {{P + "relax_type_all", "l1scaled-Jacobi"}}},
            {"agg_nl 1", {{P + "agg_nl", "1"}}},
            {"thr 0.5 + l1scaled-Jacobi", {{P + "strong_threshold", "0.5"},
                                           {P + "relax_type_all", "l1scaled-Jacobi"}}},
        };
        if (my_rank == 0)
            cout << "Sweep (max " << cap << " iterations per run):" << endl;
        for (Scale sc : {Scale::None, Scale::Diag, Scale::Ruiz})
            for (auto &c : configs)
                SolveConfig(Akk, rhs, xk_ref, sc, c.second, c.first, cap, my_rank);
    }

    VecRestoreSubVector(b, isK, &bk);
    if (useFluid)
        VecRestoreSubVector(xref, isF, &xf);
    if (haveRef)
        VecRestoreSubVector(xref, isK, &xk_ref);
    else
        VecDestroy(&xk_ref);
    VecDestroy(&x);
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
