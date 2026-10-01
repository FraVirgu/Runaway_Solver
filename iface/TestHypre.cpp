/**
 * Standalone test comparing dreami's two production linear solvers on the
 * same linear system:
 *
 *   1. DREAM::FVM::MIILU  (fvm/Solvers/MIILU.cpp)
 *   2. DREAM::FVM::MIAMG  (fvm/Solvers/MIAMG.cpp, hypre/BoomerAMG)
 *
 * Both are the actual production classes, called through Invert(), so this
 * test cannot drift out of sync with what dreami itself runs. Their PC/KSP
 * configuration is selected the same way as in dreami: via PETSC_OPTIONS /
 * the command line (e.g. -dream_split, -dream_amg_lag_pc).
 *
 * Serial only: must be launched with `mpirun -n 1` (or without mpirun);
 * with more than one rank the program throws an error and exits.
 *
 * The system is loaded from the same petsc_mat_serial_step1_iter1 /
 * petsc_rhs_serial_step1_iter1 files as Main.cpp. MIILU needs the Nhot/Nre
 * block sizes dreami used, read back from petsc_block_sizes.txt (written by
 * SolverLinearlyImplicit::initialize_internal).
 *
 * For each solver the true residual ||Ax - b|| / ||b|| is computed
 * independently of KSP's converged-reason flag, and the two solutions are
 * then compared: ||x_amg - x_miilu|| / ||x_miilu||.
 *
 * Separate executable (see iface/CMakeLists.txt, target testhypre).
 */

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <petscksp.h>
#include "DREAM/Init.h"
#include "FVM/Matrix.hpp"
#include "FVM/Solvers/MIILU.hpp"
#include "FVM/Solvers/MIAMG.hpp"

using namespace std;

/** True relative residual ||A x - b|| / ||b||. */
static PetscReal TrueResidual(Mat A, Vec x, Vec b)
{
    Vec r;
    VecDuplicate(b, &r);
    MatMult(A, x, r);
    VecAXPY(r, -1.0, b);

    PetscReal resNorm, bNorm;
    VecNorm(r, NORM_2, &resNorm);
    VecNorm(b, NORM_2, &bNorm);
    VecDestroy(&r);
    return (bNorm > 0.0) ? resNorm / bNorm : resNorm;
}

/**
 * Run one MatrixInverter on (A, b) -> x, print convergence, time and true
 * residual. Returns true if Invert() did not throw.
 */
static bool RunSolver(DREAM::FVM::MatrixInverter &inverter, Mat A, Vec b, Vec x,
                      PetscInt n, const char *label)
{
    const double tol = 1e-6;
    bool ok = true;

    DREAM::FVM::Matrix Aw(n, n, A); // wraps A, does not copy or own it
    Vec bcopy;
    VecDuplicate(b, &bcopy);
    VecCopy(b, bcopy);

    double t0 = MPI_Wtime();
    try
    {
        inverter.Invert(&Aw, &bcopy, &x);
    }
    catch (const DREAM::FVM::FVMException &ex)
    {
        cout << label << ": DIVERGED (" << ex.what() << ")" << endl;
        ok = false;
    }
    double elapsed = MPI_Wtime() - t0;

    PetscReal relRes = TrueResidual(A, x, b);
    cout << label << ": " << (ok ? "converged" : "failed")
         << ", time " << elapsed << " s"
         << ", ||A*x - b|| / ||b|| = " << relRes
         << (relRes <= tol ? "  (PASS)" : "  (FAIL)") << endl;

    VecDestroy(&bcopy);
    return ok;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1)
    {
        cerr << "TestHypre: error: must be run with a single MPI rank "
                "(mpirun -n 1), but was launched with "
             << size << " ranks." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    PETSC_COMM_WORLD = MPI_COMM_WORLD;
    dream_initialize();

    const int step = 1;
    string matname = "petsc_mat_serial_step" + to_string(step) + "_iter1";
    string rhsname = "petsc_rhs_serial_step" + to_string(step) + "_iter1";

    if (access(matname.c_str(), F_OK) != 0 || access(rhsname.c_str(), F_OK) != 0)
    {
        cerr << "Could not find " << matname << " / " << rhsname
             << " in the current directory -- run the serial simulation "
                "first to produce them."
             << endl;
        dream_finalize();
        MPI_Finalize();
        return 1;
    }

    // ---- load A, b ----
    Mat A;
    Vec b;
    PetscViewer viewer;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, matname.c_str(), FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetType(A, MATSEQAIJ);
    MatLoad(A, viewer);
    PetscViewerDestroy(&viewer);

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, rhsname.c_str(), FILE_MODE_READ, &viewer);
    VecCreate(PETSC_COMM_WORLD, &b);
    VecLoad(b, viewer);
    PetscViewerDestroy(&viewer);

    PetscInt N;
    MatGetSize(A, &N, nullptr);
    cout << "Loaded system: " << N << " x " << N << endl;

    // Nhot/Nre as dreami itself computed them for this matrix.
    PetscInt Nhot = 0, Nre = 0;
    {
        ifstream blockSizesFile("petsc_block_sizes.txt");
        string name;
        PetscInt value;
        while (blockSizesFile >> name >> value)
        {
            if (name == "Nhot")
                Nhot = value;
            else if (name == "Nre")
                Nre = value;
        }
        if (!blockSizesFile.eof() && blockSizesFile.fail())
            cerr << "Warning: could not parse petsc_block_sizes.txt -- "
                    "Nhot/Nre default to 0."
                 << endl;
    }

    Vec x_miilu, x_amg;
    VecDuplicate(b, &x_miilu);
    VecDuplicate(b, &x_amg);

    // ---- MIILU ----
    {
        DREAM::FVM::MIILU inverter((len_t)N, (len_t)Nhot, (len_t)Nre);
        RunSolver(inverter, A, b, x_miilu, N, "MIILU");
    }

    // MIILU registers its defaults (-pc_type ilu, -pc_factor_levels 0) in the
    // global PETSc options database, and MIAMG's KSPSetFromOptions() would
    // then pick them up and silently replace BoomerAMG by ILU. Remove them
    // unless the user supplied them explicitly for MIAMG via PETSC_OPTIONS
    // (in which case they were set before MIILU ran and are indistinguishable,
    // so pass MIAMG-specific overrides with -pc_hypre_* only).
    PetscOptionsClearValue(NULL, "-pc_type");
    PetscOptionsClearValue(NULL, "-pc_factor_levels");

    // ---- MIAMG ----
    {
        DREAM::FVM::MIAMG inverter((len_t)N);
        RunSolver(inverter, A, b, x_amg, N, "MIAMG");
    }

    // ---- save both solutions (PETSc binary, same format as dreami's
    // petsc_solution_final_step) ----
    auto SaveSolution = [](Vec x, const char *filename)
    {
        PetscViewer sv;
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &sv);
        VecView(x, sv);
        PetscViewerDestroy(&sv);
    };
    SaveSolution(x_miilu, "petsc_solution_miilu");
    SaveSolution(x_amg, "petsc_solution_amg");

    // ---- compare the two solutions ----
    {
        Vec diff;
        VecDuplicate(x_miilu, &diff);
        VecWAXPY(diff, -1.0, x_miilu, x_amg); // diff = x_amg - x_miilu

        PetscReal diffNorm, refNorm;
        VecNorm(diff, NORM_2, &diffNorm);
        VecNorm(x_miilu, NORM_2, &refNorm);
        PetscReal relError = (refNorm > 0.0) ? diffNorm / refNorm : diffNorm;

        const double tol = 1e-6;
        cout << "||x_amg - x_miilu|| / ||x_miilu|| = " << relError
             << (relError <= tol ? "  (MATCH)" : "  (MISMATCH)") << endl;
        VecDestroy(&diff);
    }

    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x_miilu);
    VecDestroy(&x_amg);

    dream_finalize();
    MPI_Finalize();
    return 0;
}
