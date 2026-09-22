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

        double totalLoadTime = 0.0; // matrix load (split-among-cores) time
        double totalSolveTime = 0.0;

        // Log stages: registered by ALL ranks, outside any rank guard
        PetscLogStage loadStage, solveStage;
        PetscLogStageRegister("Load", &loadStage);
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

            // ---- 1. Load the matrix in parallel: PETSc's MatLoad splits
            // the on-disk (serial) matrix directly among the ranks
            // according to the parallel layout it decides. -------------
            MPI_Barrier(PETSC_COMM_WORLD);
            double tLoad0 = MPI_Wtime();

            PetscLogStagePush(loadStage);
            PetscViewerBinaryOpen(PETSC_COMM_WORLD, matname.c_str(), FILE_MODE_READ, &viewer);
            MatCreate(PETSC_COMM_WORLD, &A);
            MatSetType(A, MATAIJ);
            MatLoad(A, viewer);
            PetscViewerDestroy(&viewer);
            PetscLogStagePop();

            MPI_Barrier(PETSC_COMM_WORLD);
            totalLoadTime += MPI_Wtime() - tLoad0;

            // ---- 2. Info on the layout (first step only) -------------------
            PetscInt row_start, row_end, M, N;
            MatGetOwnershipRange(A, &row_start, &row_end);
            MatGetSize(A, &M, &N);
            if (step == 1)
            {
                PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                                        "[rank %d] local rows [%d, %d)\n", my_rank, (int)row_start, (int)row_end);
                PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
                if (my_rank == 0)
                    cout << "[matrix] global size " << M << " x " << N << endl;
            }

            // ---- 3. Vectors with the same layout as A ---------------------
            MatCreateVecs(A, &x, &b);

            PetscViewerBinaryOpen(PETSC_COMM_WORLD, rhsname.c_str(), FILE_MODE_READ, &viewer);
            VecLoad(b, viewer);
            PetscViewerDestroy(&viewer);

            // The reference solution is only valid for the last step
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

            // ---- 4. Solve --------------------------------------------------
            DREAM::FVM::Matrix Awrap(M, N, A);

            MPI_Barrier(PETSC_COMM_WORLD);
            double t0 = MPI_Wtime();
            PetscLogStagePush(solveStage);
            inverter.Invert(&Awrap, &b, &x);
            PetscLogStagePop();
            totalSolveTime += MPI_Wtime() - t0;

            // ---- 5. Compare with the serial solution -----------------------
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
            cout << "Total matrix load (split-among-cores) time: " << totalLoadTime << " s" << endl;
            cout << "Total parallel solver time:                 " << totalSolveTime << " s" << endl;
        }

        dream_finalize();
        MPI_Comm_free(&sub_comm);
        MPI_Finalize();
    }
    return 0;
}
