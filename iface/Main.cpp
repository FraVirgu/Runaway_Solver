/**
 * General DREAM C++ interface.
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <H5Cpp.h>
#include <string>
#include <unistd.h>

// If "not debugging" is defined, then we're in
// debug mode and would like to active floating-point
// exceptions
#include <csignal>
#if !defined(NDEBUG) && defined(__linux__)
#include <fenv.h>
#endif

#include <softlib/SOFTLibException.h>

#include "DREAM/config.h"
#include "DREAM/Init.h"
#include "DREAM/IO.hpp"
#include "DREAM/QuitException.hpp"
#include "DREAM/Settings/Settings.hpp"
#include "DREAM/Settings/SFile.hpp"
#include "DREAM/Settings/SimulationGenerator.hpp"
#include "DREAM/Simulation.hpp"
#include "FVM/FVMException.hpp"
#include "FVM/Matrix.hpp"
#include "FVM/Solvers/MILU_prova.hpp"

using namespace std;

struct cmd_args
{
    bool display_settings = false;
    bool print_adas = false;
    bool splash = true;
    string
        input_filename;
};

void display_settings(DREAM::Settings *s = nullptr)
{
    if (s == nullptr)
        s = DREAM::SimulationGenerator::CreateSettings();

    cout << endl
         << "LIST OF DREAM SETTINGS" << endl;
    cout << "----------------------" << endl;
    s->DisplaySettings();
}

void display_adas(DREAM::Simulation *sim)
{
    sim->GetADAS()->PrintElements();
    cout << endl;
}

/**
 * Print the DREAMi command-line argument help.
 */
void print_help()
{
    cout << "Syntax: dreami INPUT [OPTIONS...]" << endl;
    cout << "   Run a thermal quench simulation according to the specifications made" << endl;
    cout << "   in the file 'INPUT'." << endl
         << endl;

    cout << "OPTIONS" << endl;
    cout << "  -a           Print list of elements in ADAS database." << endl;
    cout << "  -h           Print this help." << endl;
    cout << "  -l           List all available settings in DREAM." << endl;
    cout << "  -s           Do not show the splash screen." << endl;
}

/**
 * Parse command-line arguments.
 *
 * argc: Number of command-line arguments (including program name).
 * argv: List of command-line arguments.
 */
struct cmd_args *parse_args(int argc, char *argv[])
{
    char c;

    struct cmd_args *a = new struct cmd_args;
    a->display_settings = false;

    while ((c = getopt(argc, argv, "ahls")) != -1)
    {
        switch (c)
        {
        case 'a':
            a->print_adas = true;
            break;
        case 'h':
            print_help();
            break;
        case 'l':
            display_settings();
            break;
        case 's':
            a->splash = false;
            break;
        case '?':
            cout << "Unrecognized option: " << optopt << endl;
            return nullptr;
        }
    }

    if (optind + 1 < argc)
    {
        cout << "Too many trailing input arguments." << endl;
        return nullptr;
    }
    else if (optind == argc)
    {
        cout << "No input file specified." << endl;
        return nullptr;
    }

    a->input_filename = string(argv[optind]);
    return a;
}

void splash()
{
    dream_make_splash();
}

/**
 * Handle floating point exceptions.
 */
void sig_fpe(int)
{
    throw DREAM::FVM::FVMException("Floating-point error.");
}

/**
 * Handle a 'SIGQUIT' signal.
 */
void sig_quit(int)
{
    throw DREAM::QuitException("The user requested execution to stop.");
}

/**
 * Construct fake command-line arguments.
 */
char ***construct_fake_args(vector<string> &args, int &argc)
{
    argc = args.size();

    char ***argv = new char **;
    // +1: PETSc is apparently buggy...
    *argv = new char *[argc + 1];

    for (int i = 0; i < argc; i++)
    {
        size_t l = args[i].size();
        (*argv)[i] = new char[l + 1];
        args[i].copy((*argv)[i], l);
        (*argv)[i][l] = 0;
    }

    (*argv)[argc] = nullptr;

    return argv;
}

#include <vector>
#include <mpi.h>
#include <petscmat.h>

void DistributeMatrixFromRank0(Mat Source, Mat *A)
{
    PetscInt M = 0, N = 0;
    int my_rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    if (my_rank == 0)
        MatGetSize(Source, &M, &N);
    MPI_Bcast(&M, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&N, 1, MPIU_INT, 0, PETSC_COMM_WORLD);

    // ---- create the destination matrix's layout ----
    MatCreate(PETSC_COMM_WORLD, A);
    MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, M, N);
    MatSetFromOptions(*A);
    MatSetUp(*A); // allocate with a generic (unoptimized) preallocation

    PetscInt Istart, Iend;
    MatGetOwnershipRange(*A, &Istart, &Iend);

    // ---- gather every rank's row range ----
    std::vector<PetscInt> core_start(size), core_end(size);
    MPI_Allgather(&Istart, 1, MPIU_INT, core_start.data(), 1, MPIU_INT, PETSC_COMM_WORLD);
    MPI_Allgather(&Iend, 1, MPIU_INT, core_end.data(), 1, MPIU_INT, PETSC_COMM_WORLD);

    if (my_rank == 0)
    {
        std::vector<MPI_Request> requests;

        // Per destination rank r, pack rows [core_start[r], core_end[r]) as CSR:
        // rowptr (num_rows+1), cols (nnz), vals (nnz). Kept alive until Waitall.
        std::vector<std::vector<PetscInt>> rowptr(size);
        std::vector<std::vector<PetscInt>> cols(size);
        std::vector<std::vector<PetscScalar>> vals(size);

        for (int r = 0; r < size; r++)
        {
            PetscInt num_rows = core_end[r] - core_start[r];
            rowptr[r].resize(num_rows + 1, 0);

            // pass 1: build row pointer (counts) by reading the source rows
            for (PetscInt i = 0; i < num_rows; i++)
            {
                PetscInt global_row = core_start[r] + i;
                PetscInt ncols;
                MatGetRow(Source, global_row, &ncols, NULL, NULL);
                rowptr[r][i + 1] = rowptr[r][i] + ncols;
                MatRestoreRow(Source, global_row, &ncols, NULL, NULL);
            }

            PetscInt nnz = rowptr[r][num_rows];
            cols[r].resize(nnz);
            vals[r].resize(nnz);

            // pass 2: fill cols/vals
            for (PetscInt i = 0; i < num_rows; i++)
            {
                PetscInt global_row = core_start[r] + i;
                PetscInt ncols;
                const PetscInt *c;
                const PetscScalar *v;
                MatGetRow(Source, global_row, &ncols, &c, &v);
                for (PetscInt k = 0; k < ncols; k++)
                {
                    cols[r][rowptr[r][i] + k] = c[k];
                    vals[r][rowptr[r][i] + k] = v[k];
                }
                MatRestoreRow(Source, global_row, &ncols, &c, &v);
            }

            if (r == 0)
                continue; // rank 0 inserts its own part directly below, no MPI needed

            // Non-blocking sends: sizes first, then the three payload arrays
            PetscInt sizes[2] = {num_rows, nnz};
            requests.emplace_back();
            MPI_Isend(sizes, 2, MPIU_INT, r, 0, PETSC_COMM_WORLD, &requests.back());

            requests.emplace_back();
            MPI_Isend(rowptr[r].data(), num_rows + 1, MPIU_INT, r, 1, PETSC_COMM_WORLD, &requests.back());

            requests.emplace_back();
            MPI_Isend(cols[r].data(), nnz, MPIU_INT, r, 2, PETSC_COMM_WORLD, &requests.back());

            requests.emplace_back();
            MPI_Isend(vals[r].data(), nnz, MPIU_SCALAR, r, 3, PETSC_COMM_WORLD, &requests.back());
        }

        // ---- rank 0 inserts its own rows directly (no send needed) ----
        {
            PetscInt num_rows = core_end[0] - core_start[0];
            for (PetscInt i = 0; i < num_rows; i++)
            {
                PetscInt global_row = core_start[0] + i;
                PetscInt ncols = rowptr[0][i + 1] - rowptr[0][i];
                MatSetValues(*A, 1, &global_row, ncols,
                             &cols[0][rowptr[0][i]], &vals[0][rowptr[0][i]], INSERT_VALUES);
            }
        }

        // Wait for all sends to complete before the buffers go out of scope
        if (!requests.empty())
            MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    else
    {
        // ---- receive this rank's block ----
        PetscInt sizes[2];
        MPI_Recv(sizes, 2, MPIU_INT, 0, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
        PetscInt num_rows = sizes[0];
        PetscInt nnz = sizes[1];

        std::vector<PetscInt> rowptr(num_rows + 1);
        std::vector<PetscInt> cols(nnz);
        std::vector<PetscScalar> vals(nnz);

        MPI_Recv(rowptr.data(), num_rows + 1, MPIU_INT, 0, 1, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(cols.data(), nnz, MPIU_INT, 0, 2, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(vals.data(), nnz, MPIU_SCALAR, 0, 3, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);

        for (PetscInt i = 0; i < num_rows; i++)
        {
            PetscInt global_row = Istart + i;
            PetscInt ncols = rowptr[i + 1] - rowptr[i];
            MatSetValues(*A, 1, &global_row, ncols,
                         &cols[rowptr[i]], &vals[rowptr[i]], INSERT_VALUES);
        }
    }

    MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY); // collective, ALL ranks
    MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
}

/**
 * Distribute a vector that lives only on rank 0 (Source == NULL/unused on
 * every other rank) into a vector *b with the same parallel row layout as
 * Layout (typically x or b from MatCreateVecs on the already-distributed
 * matrix). Collective over PETSC_COMM_WORLD.
 *
 * Unlike DistributeMatrixFromRank0, this does not do its own point-to-point
 * MPI: *b is created with VecDuplicate so its ownership range matches
 * Layout, rank 0 hands PETSc every entry by its global index with
 * VecSetValues, and the collective VecAssemblyBegin/End (called by every
 * rank) is what moves each value to the rank that owns it.
 */
void DistributeVectorFromRank0(Vec Layout, Vec Source, Vec *b)
{
    int my_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);

    VecDuplicate(Layout, b); // *b gets Layout's parallel row layout

    if (my_rank == 0)
    {
        PetscInt n;
        VecGetSize(Source, &n);
        const PetscScalar *v;
        VecGetArrayRead(Source, &v);
        std::vector<PetscInt> idx(n);
        for (PetscInt i = 0; i < n; i++)
            idx[i] = i;
        VecSetValues(*b, n, idx.data(), v, INSERT_VALUES);
        VecRestoreArrayRead(Source, &v);
    }
    VecAssemblyBegin(*b); // collective, ALL ranks
    VecAssemblyEnd(*b);
}

/**
 * Program entry point.
 *
 * argc: Number of command-line arguments (including program name).
 * argv: List of command-line arguments.
 */
int main(int argc, char *argv[])
{
    int exit_code = 0;
    MPI_Init(&argc, &argv);
    int my_rank, size, my_sub_rank, sub_size;
    int color;
    MPI_Request ib_rq;
    MPI_Comm sub_comm, inter_comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (my_rank == 0)
        color = 0;
    else
        color = 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &sub_comm);
    MPI_Comm_size(sub_comm, &sub_size);
    MPI_Comm_rank(sub_comm, &my_sub_rank);

    DREAM::Simulation *sim = nullptr;

    // The code below can be used to make PETSc print a list of
    // citations to cite based on the current simulation
    /*int argc2;
    vector<string> args({"dreami", "-citations", "petsc-citations.txt"});
    char ***argv2 = construct_fake_args(args, argc2);
    dream_initialize(&argc2, argv2);*/

    // Initialize the DREAM library
    if (size == 1)
    {
        PETSC_COMM_WORLD = sub_comm;
        dream_initialize();
        // Allow the user to press Ctrl+\ or Ctrl+Y to quit the simulation early
        PetscPopSignalHandler();
        std::signal(SIGQUIT, sig_quit);

#if !defined(NDEBUG) && defined(__linux__)
        std::signal(SIGFPE, sig_fpe);
#endif

        // Parse command-line arguments
        struct cmd_args *a = parse_args(argc, argv);
        if (a == nullptr)
            return -1;

        // Make sure that anything written to stdio/stderr is
        // written immediately
        cout.setf(ios_base::unitbuf);

        if (a->splash)
            splash();

        cout << "commit " << DREAM_GIT_SHA1 << endl;

        // Except on NaN (but only in debug mode)
#if !defined(NDEBUG) && defined(__linux__)
        feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#endif

        try
        {

            cout << "sadsad" << endl;
            DREAM::Settings *settings = DREAM::SimulationGenerator::CreateSettings();
            cout << "asdsad" << endl;
            DREAM::SettingsSFile::LoadSettings(settings, a->input_filename);
            cout << "asdqw" << endl;
            sim = DREAM::SimulationGenerator::ProcessSettings(settings);

            if (a->print_adas)
                display_adas(sim);

            sim->Run();
            sim->Save();
        }
        catch (DREAM::QuitException &ex)
        {
            DREAM::IO::PrintInfo(ex.what());
            exit_code = 0;
        }
        catch (DREAM::FVM::FVMException &ex)
        {
            DREAM::IO::PrintError(ex.what());
            exit_code = 1;
        }
        catch (SOFTLibException &ex)
        {
            DREAM::IO::PrintError(ex.what());
            exit_code = 2;
        }
        catch (H5::FileIException &ex)
        {
            DREAM::IO::PrintError(ex.getDetailMsg().c_str());
            exit_code = 3;
        }

        dream_finalize();
        delete sim;
    }

    else
    {
        PETSC_COMM_WORLD = MPI_COMM_WORLD;
        dream_initialize();

        // Block sizes for the -dream_split index sets
        len_t blockNhot = 0, blockNre = 0, blockNtot = 0;
        {
            std::ifstream blockSizesFile("petsc_block_sizes.txt");
            if (!blockSizesFile)
                throw DREAM::FVM::FVMException(
                    "Could not open petsc_block_sizes.txt (run the serial "
                    "simulation first to produce it).");

            std::string label;
            blockSizesFile >> label >> blockNhot;
            blockSizesFile >> label >> blockNre;
            blockSizesFile >> label >> blockNtot;
        }

        DREAM::FVM::MILU_PROVA inverter(blockNtot, blockNhot, blockNre);

        int numTimeSteps = 100;
        {
            std::ifstream numStepsFile("petsc_num_timesteps.txt");
            if (numStepsFile)
                numStepsFile >> numTimeSteps;
        }

        double totalLoadTime = 0.0;  // read on rank 0
        double totalIluTime = 0.0;   // ILU setup on rank 0
        double totalPrecTime = 0.0;  // forming M^{-1}A and M^{-1}b on rank 0
        double totalDistTime = 0.0;  // distribution to all ranks
        double totalSolveTime = 0.0; // GMRES setup + solve

        // Log stages: registered by ALL ranks, outside any rank guard
        PetscLogStage loadStage, iluStage, precStage, distStage, gmresSetupStage, solveStage;
        PetscLogStageRegister("Load", &loadStage);
        PetscLogStageRegister("ILU setup", &iluStage);
        PetscLogStageRegister("Apply M^-1", &precStage);
        PetscLogStageRegister("Distribute", &distStage);
        PetscLogStageRegister("GMRES setup", &gmresSetupStage);
        PetscLogStageRegister("Solve", &solveStage);

        Mat A = NULL; // distributed matrix
        Vec b, x, final_sol;
        PetscViewer viewer;

        for (int step = 1; step <= numTimeSteps; step++)
        {
            string matname = "petsc_mat_serial_step" + to_string(step) + "_iter1";
            string rhsname = "petsc_rhs_serial_step" + to_string(step) + "_iter1";

            // Rank 0 decides whether this step exists, everybody follows
            // (avoids a deadlock if ranks disagree about the files).
            int haveStep = 0;
            if (my_rank == 0)
                haveStep = (access(matname.c_str(), F_OK) == 0 &&
                            access(rhsname.c_str(), F_OK) == 0);
            MPI_Bcast(&haveStep, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!haveStep)
                continue;

            MPI_Barrier(PETSC_COMM_WORLD);

            Mat MA_seq = NULL; // M^{-1}A, rank 0 only
            Vec Mb_seq = NULL; // M^{-1}b, rank 0 only
            Mat A = NULL;      // distributed M^{-1}A
            Vec b, x, final_sol;
            PetscViewer viewer;

            Mat A_seq;
            Vec b_seq;

            // =============== RANK 0: load, ILU, apply M^{-1} ===============
            MPI_Barrier(PETSC_COMM_WORLD);
            if (my_rank == 0)
            {

                // ---- 1. load A and b ----
                double tL0 = MPI_Wtime();
                PetscLogStagePush(loadStage);
                PetscViewerBinaryOpen(PETSC_COMM_SELF, matname.c_str(), FILE_MODE_READ, &viewer);
                MatCreate(PETSC_COMM_SELF, &A_seq);
                MatSetType(A_seq, MATSEQAIJ);
                MatLoad(A_seq, viewer);
                PetscViewerDestroy(&viewer);

                PetscViewerBinaryOpen(PETSC_COMM_SELF, rhsname.c_str(), FILE_MODE_READ, &viewer);
                VecCreate(PETSC_COMM_SELF, &b_seq);
                VecLoad(b_seq, viewer);
                PetscViewerDestroy(&viewer);
                PetscLogStagePop();
                totalLoadTime += MPI_Wtime() - tL0;

                // ---- 2+3. ILU factorization, then Mb = M^{-1} b and
                // MA = M^{-1} A (sparsified). ApplyILUPreconditioning does
                // not expose the factorization and apply times separately,
                // so both stages/timers cover the whole call.
                double tI0 = MPI_Wtime();
                PetscLogStagePush(iluStage);
                PetscLogStagePush(precStage);
                DREAM::FVM::MILU_PROVA::ApplyILUPreconditioning(A_seq, b_seq, &MA_seq, &Mb_seq);
                PetscLogStagePop();
                PetscLogStagePop();
                double tIluPrec = MPI_Wtime() - tI0;
                totalIluTime += tIluPrec;
                totalPrecTime += tIluPrec;

                MatDestroy(&A_seq);
                VecDestroy(&b_seq);
            }

            // =============== ALL RANKS: distribute M^{-1}A, M^{-1}b ===============
            MPI_Barrier(PETSC_COMM_WORLD);
            double tD0 = MPI_Wtime();
            PetscLogStagePush(distStage);

            DistributeMatrixFromRank0(MA_seq, &A); // collective
            if (my_rank == 0)
                MatDestroy(&MA_seq);

            Vec x_layout;
            MatCreateVecs(A, &x, &x_layout); // x and x_layout both get A's layout
            VecDestroy(&x_layout);

            DistributeVectorFromRank0(x, Mb_seq, &b); // collective
            if (my_rank == 0)
                VecDestroy(&Mb_seq);

            PetscLogStagePop();
            MPI_Barrier(PETSC_COMM_WORLD);
            totalDistTime += MPI_Wtime() - tD0;

            // ---- layout info (first step only) ----
            PetscInt row_start, row_end, M, N;
            MatGetOwnershipRange(A, &row_start, &row_end);
            MatGetSize(A, &M, &N);
            if (step == 1)
            {
                PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                                        "[rank %d] local rows [%d, %d)\n",
                                        my_rank, (int)row_start, (int)row_end);
                PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
                if (my_rank == 0)
                    cout << "[matrix] global size " << M << " x " << N << endl;
            }

            // Reference solution (only valid for the last step)
            bool haveFinalSol = (step == numTimeSteps);
            if (haveFinalSol)
            {
                PetscViewer viewer_final;
                PetscViewerBinaryOpen(PETSC_COMM_WORLD, "petsc_solution_final_step",
                                      FILE_MODE_READ, &viewer_final);
                VecDuplicate(x, &final_sol);
                VecLoad(final_sol, viewer_final);
                PetscViewerDestroy(&viewer_final);
            }

            // =============== GMRES on the distributed system ===============
            totalSolveTime += DREAM::FVM::MILU_PROVA::Solve_GMRES(
                A, b, x, step, my_rank, gmresSetupStage, solveStage);

            // =============== Compare with the serial solution ===============
            if (haveFinalSol)
            {
                const double tol = 1e-4;

                Vec diff;
                VecDuplicate(x, &diff);
                VecWAXPY(diff, -1.0, final_sol, x);

                PetscReal diffNorm, refNorm;
                VecNorm(diff, NORM_2, &diffNorm);
                VecNorm(final_sol, NORM_2, &refNorm);
                PetscReal relError = (refNorm > 0.0) ? diffNorm / refNorm : diffNorm;

                if (my_rank == 0)
                    cout << "step " << step << ": ||x_parallel - x_serial|| / ||x_serial|| = "
                         << relError << (relError <= tol ? " (MATCH)" : " (MISMATCH)") << endl;

                VecDestroy(&diff);
                VecDestroy(&final_sol);
            }

            VecDestroy(&x);
            VecDestroy(&b);
            MatDestroy(&A);
        } // end for

        if (my_rank == 0)
        {
            cout << endl;
            cout << "Total load (rank 0):                 " << totalLoadTime << " s" << endl;
            cout << "Total ILU setup (rank 0):            " << totalIluTime << " s" << endl;
            cout << "Total M^-1 A, M^-1 b (rank 0):       " << totalPrecTime << " s" << endl;
            cout << "Total distribution:                  " << totalDistTime << " s" << endl;
            cout << "Total GMRES setup + solve:           " << totalSolveTime << " s" << endl;
        }

        dream_finalize();
        MPI_Comm_free(&sub_comm);
        MPI_Finalize();
    }
    return 0;
}