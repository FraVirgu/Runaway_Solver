/**
 * Serial comparison of linear solvers on the kinetic block (f_hot, f_re:
 * rows/columns [0, Nhot+Nre)) of a DREAM system dumped by dreami with
 * -dream_save_system, e.g. a current-quench snapshot
 * (dream_settings_kinetic_CQ_*). As in TestKineticHypreRadius.cpp, the other
 * unknowns are moved to the right-hand side:
 *
 *     A_kk x_k = b_k - A_kf x_f
 *
 * (x_f from the reference solution if it exists, else 0).
 *
 * Solvers (outer rtol 1e-10 on the true residual, restart 100, zero initial
 * guess), each under its own options prefix:
 *
 *   bjamgeu  BJ(radii) + [GMRES + AMG/Euclid(1)] + FGMRES     prefix bjamgeu_
 *            block Jacobi over the radius (one block per radial cell,
 *            f_hot + f_re of that radius), each block SOLVED by an inner
 *            GMRES (rtol -inner_rtol, default 1e-6, max 500 its) +
 *            BoomerAMG with hypre's Euclid ILU(1) smoother; outer FGMRES.
 *            This is the configuration of fvm/Solvers/MIRadialBJ.cpp.
 *   bjamg    BJ(radii) + AMG + GMRES                          prefix bjamg_
 *            same blocks, each approximated by one BoomerAMG V-cycle
 *            (default hypre smoother), outer GMRES.
 *   bjilu    BJ(radii) + GMRES                                prefix bjilu_
 *            same blocks, each approximated by ILU(0) (PETSc's default
 *            block-Jacobi sub-solver), outer GMRES.
 *   amg      AMG + GMRES                                      prefix amg_
 *            one BoomerAMG V-cycle on the whole kinetic block, outer GMRES.
 *   ilu      ILU + GMRES                                      prefix ilu_
 *            ILU(0) on the whole kinetic block, outer GMRES.
 *   lu       LU (direct)                                      prefix lu_
 *            PETSc's serial LU factorization of the whole kinetic block
 *            (KSPPREONLY + PCLU, nested-dissection ordering); "setup" is
 *            the factorization, "solve" the triangular solves. Another
 *            package with -lu_pc_factor_mat_solver_type mumps|superlu.
 *
 * With one radial cell (Nr = 1) the BJ variants are the global ones (one
 * block): compare with -solvers lu,amg,ilu.
 *
 * Anything can be overridden on the command line under the prefix, e.g.
 *   -ilu_pc_factor_levels 1
 *   -bjilu_sub_pc_factor_levels 1
 *   -bjamg_sub_pc_hypre_boomeramg_smooth_type Euclid
 *   -amg_pc_hypre_boomeramg_smooth_type Euclid -amg_pc_hypre_boomeramg_eu_level 1
 *   -bjamgeu_ksp_monitor_true_residual
 *
 * The BJ variants run on the system permuted to radius-major order (hot then
 * re, per radius), since the f_hot/f_re rows of one radius are not
 * contiguous in the assembled matrix; residual and error are always measured
 * on the original, unpermuted kinetic system.
 *
 * Inputs (in the current directory): matrix, right-hand side,
 * petsc_block_sizes.txt (Nhot, Nre, Nr) and the optional reference solution.
 * Options:
 *   -mat <file>            matrix         (default petsc_mat_np1_step1_iter1)
 *   -rhs <file>            right-hand side (default petsc_rhs_np1_step1_iter1)
 *   -ref_solution <file>   reference      (default petsc_solution_np1_step1_iter1)
 *   -nr <Nr>               number of radial cells, if petsc_block_sizes.txt
 *                          has no "Nr" line (older dumps)
 *   -solvers a,b,...       subset to run, in order (default
 *                          bjamgeu,bjamg,bjilu,amg,ilu,lu)
 *   -maxits <n>            outer iteration limit (default 1000)
 *   -inner_rtol <tol>      inner GMRES tolerance of bjamgeu (default 1e-6)
 *
 * Serial only: run with a single MPI rank.
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
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

enum class Kind
{
    BJ_AMG_EUCLID_INNER, // BJ(radii) + [GMRES + AMG/Euclid(1)] + FGMRES
    BJ_AMG,              // BJ(radii) + AMG V-cycle + GMRES
    BJ_ILU,              // BJ(radii) + ILU + GMRES
    AMG,                 // AMG + GMRES
    ILU,                 // ILU + GMRES
    LU                   // direct LU
};

struct Variant
{
    const char *name;   // -solvers key
    const char *prefix; // options prefix
    const char *label;  // printed
    Kind kind;
};

static const Variant variants[] = {
    {"bjamgeu", "bjamgeu_", "BJ(r) + [GMRES+AMG/Euclid(1)] + FGMRES", Kind::BJ_AMG_EUCLID_INNER},
    {"bjamg", "bjamg_", "BJ(r) + AMG + GMRES", Kind::BJ_AMG},
    {"bjilu", "bjilu_", "BJ(r) + ILU + GMRES", Kind::BJ_ILU},
    {"amg", "amg_", "AMG + GMRES", Kind::AMG},
    {"ilu", "ilu_", "ILU + GMRES", Kind::ILU},
    {"lu", "lu_", "LU (direct)", Kind::LU},
};

static bool IsBJ(Kind k)
{
    return k == Kind::BJ_AMG_EUCLID_INNER || k == Kind::BJ_AMG || k == Kind::BJ_ILU;
}

/**
 * Kinetic system, in original and in radius-major (permuted) order.
 */
struct System
{
    Mat A;     // A_kk, original order
    Vec b;     // b_k - A_kf x_f
    Vec x_ref; // reference solution (nullptr: none)
    Mat Ap;    // A_kk, radius-major order
    Vec bP;    // rhs,  radius-major order
    VecScatter sc; // original -> radius-major (forward)
    PetscInt Nr, nb; // radial cells, rows per radius
};

/** Put a default into the options database unless the user gave one. */
static void SetDefaultOption(const string &name, const char *value)
{
    PetscBool set;
    PetscOptionsHasName(NULL, NULL, name.c_str(), &set);
    if (!set)
        PetscOptionsSetValue(NULL, name.c_str(), value);
}

/**
 * Relative residual and (if there is a reference) relative error of x on the
 * original kinetic system, printed as one result line.
 */
static void Report(const System &s, Vec x, const char *label, PetscInt its,
                   KSPConvergedReason reason, double tSetup, double tSolve)
{
    Vec r;
    VecDuplicate(s.b, &r);
    MatMult(s.A, x, r);
    VecAXPY(r, -1.0, s.b);
    PetscReal resNorm, rhsNorm;
    VecNorm(r, NORM_2, &resNorm);
    VecNorm(s.b, NORM_2, &rhsNorm);
    PetscReal relRes = rhsNorm > 0 ? resNorm / rhsNorm : resNorm;

    char errStr[32] = "-";
    if (s.x_ref)
    {
        VecWAXPY(r, -1.0, s.x_ref, x);
        PetscReal errNorm, refNorm;
        VecNorm(r, NORM_2, &errNorm);
        VecNorm(s.x_ref, NORM_2, &refNorm);
        snprintf(errStr, sizeof(errStr), "%.2e",
                 (double)(refNorm > 0 ? errNorm / refNorm : errNorm));
    }
    VecDestroy(&r);

    printf("%-40s its %5d  %-9s setup %8.3f s  solve %8.3f s  total %8.3f s  "
           "res %.2e  err %s\n",
           label, (int)its, reason > 0 ? "converged" : "DIVERGED", tSetup, tSolve,
           tSetup + tSolve, (double)relRes, errStr);
    if (reason <= 0)
        printf("%-40s   reason: %s\n", "", KSPConvergedReasons[reason]);
    fflush(stdout); // stdout is a file in batch jobs: write the line now
}

/**
 * Set up and solve with one variant, then report.
 */
static void RunVariant(const Variant &v, const System &s, PetscInt maxits,
                       PetscReal innerRtol)
{
    const bool bj = IsBJ(v.kind);
    const bool fgmres = (v.kind == Kind::BJ_AMG_EUCLID_INNER);
    Mat A = bj ? s.Ap : s.A;
    Vec b = bj ? s.bP : s.b;

    Vec y;
    VecDuplicate(b, &y);
    VecSet(y, 0.0);

    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOptionsPrefix(ksp, v.prefix);
    KSPSetOperators(ksp, A, A);
    KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
    if (v.kind == Kind::LU)
    {
        // one application of the factorization, no residual norm
        KSPSetType(ksp, KSPPREONLY);
        KSPSetNormType(ksp, KSP_NORM_NONE);
    }
    else
    {
        KSPSetType(ksp, fgmres ? KSPFGMRES : KSPGMRES);
        KSPGMRESSetRestart(ksp, 100);
        KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, maxits);
        // Right preconditioning everywhere, so that every variant stops on
        // the same (true, unpreconditioned) residual.
        KSPSetPCSide(ksp, PC_RIGHT);
        KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    }

    KSPGetPC(ksp, &pc);
    switch (v.kind)
    {
    case Kind::BJ_AMG_EUCLID_INNER:
    case Kind::BJ_AMG:
    case Kind::BJ_ILU:
    {
        PCSetType(pc, PCBJACOBI);
        vector<PetscInt> lens(s.Nr, s.nb); // one block per radius
        PCBJacobiSetLocalBlocks(pc, s.Nr, lens.data());
        break;
    }
    case Kind::AMG:
        PCSetType(pc, PCHYPRE);
        PCHYPRESetType(pc, "boomeramg");
        break;
    case Kind::ILU:
        PCSetType(pc, PCILU);
        break;
    case Kind::LU:
        PCSetType(pc, PCLU);
        break;
    }
    KSPSetFromOptions(ksp);

    // Euclid ILU(1) smoother of the block AMG, as in MIRadialBJ (the
    // options database is the only way to set it on BoomerAMG).
    const string pre = string("-") + v.prefix;
    if (v.kind == Kind::BJ_AMG_EUCLID_INNER)
    {
        SetDefaultOption(pre + "sub_pc_hypre_boomeramg_smooth_type", "Euclid");
        SetDefaultOption(pre + "sub_pc_hypre_boomeramg_eu_level", "1");
    }

    double t0 = MPI_Wtime();
    KSPSetUp(ksp);

    // Default block sub-solvers (the sub-KSPs exist after KSPSetUp); the
    // -<prefix>sub_* options override them.
    PetscInt nsub = 0, first;
    KSP *subksp = nullptr;
    if (bj)
    {
        PCBJacobiGetSubKSP(pc, &nsub, &first, &subksp);
        for (PetscInt i = 0; i < nsub; i++)
        {
            PC subpc;
            KSPGetPC(subksp[i], &subpc);
            if (v.kind == Kind::BJ_AMG_EUCLID_INNER)
            {
                KSPSetType(subksp[i], KSPGMRES);
                KSPGMRESSetRestart(subksp[i], 100);
                KSPSetTolerances(subksp[i], innerRtol, PETSC_DEFAULT, PETSC_DEFAULT, 500);
                KSPSetNormType(subksp[i], KSP_NORM_UNPRECONDITIONED);
            }
            else
                KSPSetType(subksp[i], KSPPREONLY);

            if (v.kind == Kind::BJ_ILU)
                PCSetType(subpc, PCILU);
            else
            {
                PCSetType(subpc, PCHYPRE);
                PCHYPRESetType(subpc, "boomeramg");
            }
            KSPSetFromOptions(subksp[i]);
        }
        KSPSetUpOnBlocks(ksp);
    }
    const double tSetup = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    KSPSolve(ksp, b, y);
    const double tSolve = MPI_Wtime() - t0;

    PetscInt its;
    KSPConvergedReason reason;
    KSPGetIterationNumber(ksp, &its);
    KSPGetConvergedReason(ksp, &reason);

    // back to the original order
    Vec x;
    VecDuplicate(s.b, &x);
    if (bj)
    {
        VecScatterBegin(s.sc, y, x, INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(s.sc, y, x, INSERT_VALUES, SCATTER_REVERSE);
    }
    else
        VecCopy(y, x);

    Report(s, x, v.label, its, reason, tSetup, tSolve);

    if (v.kind == Kind::BJ_AMG_EUCLID_INNER)
    {
        PetscInt innerIts = 0, maxBlk = 0, its_i;
        for (PetscInt i = 0; i < nsub; i++)
        {
            KSPGetTotalIterations(subksp[i], &its_i);
            innerIts += its_i;
            maxBlk = max(maxBlk, its_i);
        }
        printf("%-40s   inner GMRES iterations: %d total, %d on the worst radius "
               "(rtol %.1e)\n",
               "", (int)innerIts, (int)maxBlk, (double)innerRtol);
        fflush(stdout);
    }

    KSPDestroy(&ksp);
    VecDestroy(&y);
    VecDestroy(&x);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            cerr << "testkineticsolversserial runs on a single MPI rank (got "
                 << size << ")." << endl;
        MPI_Finalize();
        return 1;
    }

    PETSC_COMM_WORLD = MPI_COMM_WORLD;
    dream_initialize();

#ifndef PETSC_HAVE_HYPRE
    cerr << "This PETSc build has no hypre support." << endl;
    dream_finalize();
    MPI_Finalize();
    return 1;
#endif

    char matname[PETSC_MAX_PATH_LEN] = "petsc_mat_np1_step1_iter1";
    char rhsname[PETSC_MAX_PATH_LEN] = "petsc_rhs_np1_step1_iter1";
    char refname[PETSC_MAX_PATH_LEN] = "petsc_solution_np1_step1_iter1";
    PetscOptionsGetString(NULL, NULL, "-mat", matname, sizeof(matname), NULL);
    PetscOptionsGetString(NULL, NULL, "-rhs", rhsname, sizeof(rhsname), NULL);
    PetscOptionsGetString(NULL, NULL, "-ref_solution", refname, sizeof(refname), NULL);

    if (access(matname, F_OK) != 0 || access(rhsname, F_OK) != 0 ||
        access("petsc_block_sizes.txt", F_OK) != 0)
    {
        cerr << "Missing one of " << matname << ", " << rhsname
             << ", petsc_block_sizes.txt in the current directory (they are "
                "written by dreami with -dream_save_system)."
             << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }
    const bool haveRef = (access(refname, F_OK) == 0);

    PetscInt maxits = 1000;
    PetscReal innerRtol = 1e-6;
    PetscOptionsGetInt(NULL, NULL, "-maxits", &maxits, NULL);
    PetscOptionsGetReal(NULL, NULL, "-inner_rtol", &innerRtol, NULL);

    // ---- which solvers ----
    vector<const Variant *> toRun;
    {
        char *names[16];
        PetscInt n = 16;
        PetscBool set;
        PetscOptionsGetStringArray(NULL, NULL, "-solvers", names, &n, &set);
        if (!set)
            for (const Variant &v : variants)
                toRun.push_back(&v);
        else
        {
            for (PetscInt i = 0; i < n; i++)
            {
                const Variant *found = nullptr;
                for (const Variant &v : variants)
                    if (string(names[i]) == v.name)
                        found = &v;
                if (!found)
                {
                    cerr << "Unknown solver '" << names[i]
                         << "' in -solvers (bjamgeu|bjamg|bjilu|amg|ilu|lu)." << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                toRun.push_back(found);
                PetscFree(names[i]);
            }
        }
    }

    // ---- load ----
    Mat A;
    Vec b, xref = nullptr;
    LoadMat(matname, &A);
    LoadVec(rhsname, &b);
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
    PetscOptionsGetInt(NULL, NULL, "-nr", &Nr, NULL);

    const PetscInt Nk = Nhot + Nre;
    if (Nk <= 0 || Nk > N)
    {
        cerr << "Invalid block sizes: Nhot+Nre = " << Nk << ", N = " << N << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }
    if (Nr <= 0 || Nhot % Nr != 0 || Nre % Nr != 0)
    {
        cerr << "Invalid number of radial cells Nr = " << Nr << " (Nhot " << Nhot
             << ", Nre " << Nre << " must both be multiples of it). Give it with "
                                   "-nr if petsc_block_sizes.txt has no \"Nr\" line."
             << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }
    const PetscInt nh = Nhot / Nr, nre = Nre / Nr;

    // ---- kinetic block, fluid unknowns moved to the right-hand side ----
    const bool haveFluid = (Nk < N);
    const bool useFluid = haveFluid && haveRef;

    IS isK, isF;
    ISCreateStride(PETSC_COMM_WORLD, Nk, 0, 1, &isK);
    ISCreateStride(PETSC_COMM_WORLD, N - Nk, Nk, 1, &isF);

    System s;
    s.Nr = Nr;
    s.nb = nh + nre;
    MatCreateSubMatrix(A, isK, isK, MAT_INITIAL_MATRIX, &s.A);

    Vec bk;
    VecGetSubVector(b, isK, &bk);
    MatCreateVecs(s.A, NULL, &s.b);
    if (useFluid)
    {
        Mat Akf;
        Vec xf;
        MatCreateSubMatrix(A, isK, isF, MAT_INITIAL_MATRIX, &Akf);
        VecGetSubVector(xref, isF, &xf);
        MatMult(Akf, xf, s.b);
        VecAYPX(s.b, -1.0, bk); // b_k - A_kf x_f
        VecRestoreSubVector(xref, isF, &xf);
        MatDestroy(&Akf);
    }
    else
        VecCopy(bk, s.b);
    VecRestoreSubVector(b, isK, &bk);

    s.x_ref = nullptr;
    if (haveRef)
    {
        Vec xk;
        VecGetSubVector(xref, isK, &xk);
        VecDuplicate(s.b, &s.x_ref);
        VecCopy(xk, s.x_ref);
        VecRestoreSubVector(xref, isK, &xk);
    }

    // ---- radius-major permutation: isPerm[k] = old row of new row k ----
    IS isPerm;
    {
        vector<PetscInt> idx;
        idx.reserve(Nk);
        for (PetscInt ir = 0; ir < Nr; ir++)
        {
            for (PetscInt i = 0; i < nh; i++)
                idx.push_back(ir * nh + i);
            for (PetscInt i = 0; i < nre; i++)
                idx.push_back(Nhot + ir * nre + i);
        }
        ISCreateGeneral(PETSC_COMM_WORLD, Nk, idx.data(), PETSC_COPY_VALUES, &isPerm);
    }
    MatCreateSubMatrix(s.A, isPerm, isPerm, MAT_INITIAL_MATRIX, &s.Ap);
    MatCreateVecs(s.Ap, NULL, &s.bP);
    VecScatterCreate(s.b, isPerm, s.bP, NULL, &s.sc);
    VecScatterBegin(s.sc, s.b, s.bP, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(s.sc, s.b, s.bP, INSERT_VALUES, SCATTER_FORWARD);

    MatInfo info;
    MatGetInfo(s.A, MAT_LOCAL, &info);
    cout << "Matrix " << matname << ": full system " << N << " x " << N
         << ", kinetic block " << Nk << " x " << Nk << " (Nhot " << Nhot
         << ", Nre " << Nre << "), nnz " << (long long)info.nz_used << endl;
    cout << "Nr = " << Nr << " radial blocks of " << s.nb << " rows (f_hot "
         << nh << " + f_re " << nre << ")" << endl;
    if (haveRef)
        cout << "Reference solution: " << refname << endl;
    else
        cout << "No reference solution (" << refname << "): no error column"
             << (haveFluid ? ", and the fluid unknowns are taken as 0" : "")
             << "." << endl;
    cout << "Outer rtol 1e-10 (true residual), restart 100, max " << maxits
         << " iterations, serial" << endl
         << endl;

    for (const Variant *v : toRun)
        RunVariant(*v, s, maxits, innerRtol);

    VecScatterDestroy(&s.sc);
    ISDestroy(&isPerm);
    MatDestroy(&s.Ap);
    VecDestroy(&s.bP);
    MatDestroy(&s.A);
    VecDestroy(&s.b);
    if (s.x_ref)
        VecDestroy(&s.x_ref);
    ISDestroy(&isK);
    ISDestroy(&isF);
    MatDestroy(&A);
    VecDestroy(&b);
    if (xref)
        VecDestroy(&xref);

    dream_finalize();
    MPI_Finalize();
    return 0;
}
