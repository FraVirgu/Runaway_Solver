/**
 * Implementation of the linearly implicit solution method of the non-linear
 * equation system. With this solver, we first assume that the non-linear
 * function F(x) can be decomposed into
 *
 *   F(x) ~ M(t,x) x(t) + S(t,x),
 *
 * where M(t,x) is an operator matrix, S(t,x) is a vector and x(t) is the
 * unknown vector. Next, we assume that M(t,x) and S(t,x) vary slowly with time
 * such that
 *
 *   M(t_{n+1}, x_{n+1}) ~ M(t_n, x_n),
 *   S(t_{n+1}, x_{n+1}) ~ S(t_n, x_n),
 *
 * where x_{n+1} = x(t_{n+1}). With this approximation, we may write the
 * originally non-linear equation system "F(x) = 0" as the system of linear
 * equations
 *
 *   F(x_{n+1}) ~ M(x_n) x_{n+1} + S(x_n) = 0,
 *
 * which is solved by
 *
 *   x_{n+1} = -M(x_n)^{-1} S(x_n).
 *
 * Given the initial state 'x_n' of the system, we may thus straightforwardly
 * advance the system in time.
 */

#include <fstream>
#include <vector>
#include "DREAM/EquationSystem.hpp"
#include "DREAM/IO.hpp"
#include "DREAM/OutputGeneratorSFile.hpp"
#include "DREAM/Settings/OptionConstants.hpp"
#include "DREAM/Solver/SolverLinearlyImplicit.hpp"
#include <chrono>

using namespace DREAM;
using namespace std;

/**
 * Constructor.
 */
SolverLinearlyImplicit::SolverLinearlyImplicit(
    FVM::UnknownQuantityHandler *unknowns,
    vector<UnknownQuantityEquation *> *unknown_equations,
    EquationSystem *eqsys, const bool verbose,
    enum OptionConstants::linear_solver ls) : Solver(unknowns, unknown_equations, eqsys, verbose, ls)
{

    this->timeKeeper = new FVM::TimeKeeper("Solver linear");
    this->timerTot = this->timeKeeper->AddTimer("total", "Total time");
    this->timerRebuild = this->timeKeeper->AddTimer("rebuildtot", "Rebuild coefficients");
    this->timerMatrix = this->timeKeeper->AddTimer("matrix", "Construct matrix");
    this->timerInvert = this->timeKeeper->AddTimer("invert", "Invert matrix");
}

/**
 * Destructor.
 */
SolverLinearlyImplicit::~SolverLinearlyImplicit()
{
    delete this->matrix;
    delete this->timeKeeper;

    // VecDestroy(&this->petsc_S);
}

/**
 * Initialize the solver.
 */
void SolverLinearlyImplicit::initialize_internal(
    const len_t size, std::vector<len_t> &unknownIds)
{
    PetscBool kineticOnly = PETSC_TRUE;
    PetscOptionsHasName(NULL, NULL, "-dream_kinetic_only", &kineticOnly);
    if (kineticOnly)
    {
        this->initialize_kinetic_only(size, unknownIds);
        return;
    }

    this->matrix = new FVM::BlockMatrix();

    std::vector<len_t> fhot, fre, fluid;
    for (len_t id : nontrivial_unknowns)
    {
        const std::string &nm = unknowns->GetUnknown(id)->GetName();
        if (nm == OptionConstants::UQTY_F_HOT)
            fhot.push_back(id);
        else if (nm == OptionConstants::UQTY_F_RE)
            fre.push_back(id);
        else
            fluid.push_back(id);
    }

    // Ordered so that the assembled matrix has f_hot in rows [0, Nhot),
    // f_re in [Nhot, Nhot+Nre) and the fluid quantities thereafter. The
    // fieldsplit index sets are strides over these ranges.
    std::vector<len_t> ordered = fhot;
    ordered.insert(ordered.end(), fre.begin(), fre.end());
    ordered.insert(ordered.end(), fluid.begin(), fluid.end());

    for (len_t id : ordered)
    {
        UnknownQuantityEquation *eqn = this->unknown_equations->at(id);
        unknownToMatrixMapping[id] =
            matrix->CreateSubEquation(eqn->NumberOfElements(), eqn->NumberOfNonZeros(), id);
    }

    matrix->ConstructSystem();

    // Row counts, not quantity counts.
    this->Nhot = 0;
    for (len_t id : fhot)
        this->Nhot += this->unknown_equations->at(id)->NumberOfElements();

    this->Nre = 0;
    for (len_t id : fre)
        this->Nre += this->unknown_equations->at(id)->NumberOfElements();

    // Save the block sizes MIILU needs to reconstruct the same
    // -dream_split populations index sets (is_fhot/is_fre/is_fluid) when
    // replaying the dumped matrices through a separate (e.g. parallel)
    // solver process, which has no EquationSystem of its own to recompute
    // these from.
    // (Written by one rank only, as every rank runs this code.)
    PetscMPIInt mpiRank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &mpiRank);
    if (mpiRank == 0)
    {
        std::ofstream blockSizesFile("petsc_block_sizes.txt");
        blockSizesFile << "Nhot " << this->Nhot << "\n";
        blockSizesFile << "Nre " << this->Nre << "\n";
        blockSizesFile << "Ntot " << size << "\n";
    }

    this->SelectLinearSolver(size);

    // Same row layout as the matrix (distributed over PETSC_COMM_WORLD)
    MatCreateVecs(this->matrix->mat(), &this->petsc_S, NULL);
}
/**
 * Initialize the solver with a system that contains only the kinetic
 * unknowns, f_hot and f_re. Selected with -dream_kinetic_only.
 *
 * The matrix has f_hot in rows [0, Nhot) and f_re in [Nhot, Nhot+Nre)
 * and nothing else; its size is Nhot+Nre. The other unknowns (the fluid
 * and scalar quantities) take no part in the system at all: they are not
 * in the matrix, and the terms through which the kinetic equations depend
 * on them (the sources that are proportional to n_re, n_tot, n_i and
 * S_particle) are dropped, not moved to the right-hand side. The system
 * that is solved is therefore
 *
 *   A_kk x_k = b_k,
 *
 * where b_k only holds the terms of the kinetic equations that do not
 * depend on any unknown. The other unknowns are not evolved.
 *
 * Implementation: the solver's own list of non-trivial unknowns (and the
 * matrix size) is narrowed to the kinetic ones, so that BuildMatrix(),
 * Store() and RestoreSolution() only see them, and BuildMatrix() is told
 * to ignore operators applied to unknowns outside the matrix
 * (dropNonMatrixTerms). The EquationSystem's list, used for output, is
 * unaffected.
 *
 * Do not combine with -dream_split populations or kinetic: the fluid
 * block those use no longer exists.
 */
void SolverLinearlyImplicit::initialize_kinetic_only(
    const len_t, std::vector<len_t> &)
{
    this->matrix = new FVM::BlockMatrix();

    std::vector<len_t> fhot, fre;
    for (len_t id : nontrivial_unknowns)
    {
        const std::string &nm = unknowns->GetUnknown(id)->GetName();
        if (nm == OptionConstants::UQTY_F_HOT)
            fhot.push_back(id);
        else if (nm == OptionConstants::UQTY_F_RE)
            fre.push_back(id);
    }

    if (fhot.empty() && fre.empty())
        throw SolverException(
            "-dream_kinetic_only: the equation system contains no kinetic "
            "unknowns (f_hot, f_re).");

    // f_hot first, then f_re: the order of the matrix blocks, and also the
    // order in which unknowns->Store() reads the solution vector.
    std::vector<len_t> kinetic = fhot;
    kinetic.insert(kinetic.end(), fre.begin(), fre.end());

    // From here on, the solver only sees the kinetic unknowns, and terms
    // that depend on any other unknown are left out of the system.
    this->nontrivial_unknowns = kinetic;
    this->dropNonMatrixTerms = true;

    for (len_t id : kinetic)
    {
        UnknownQuantityEquation *eqn = this->unknown_equations->at(id);
        unknownToMatrixMapping[id] =
            matrix->CreateSubEquation(eqn->NumberOfElements(), eqn->NumberOfNonZeros(), id);
    }

    matrix->ConstructSystem();

    this->Nhot = 0;
    for (len_t id : fhot)
        this->Nhot += this->unknown_equations->at(id)->NumberOfElements();

    this->Nre = 0;
    for (len_t id : fre)
        this->Nre += this->unknown_equations->at(id)->NumberOfElements();

    const len_t sizeKinetic = this->Nhot + this->Nre;
    this->matrix_size = sizeKinetic;

    // Same file as in initialize_internal(), here with Ntot = Nhot+Nre.
    // (Written by one rank only, as every rank runs this code.)
    PetscMPIInt mpiRank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &mpiRank);
    if (mpiRank == 0)
    {
        std::ofstream blockSizesFile("petsc_block_sizes.txt");
        blockSizesFile << "Nhot " << this->Nhot << "\n";
        blockSizesFile << "Nre " << this->Nre << "\n";
        blockSizesFile << "Ntot " << sizeKinetic << "\n";
    }

    if (this->Verbose())
        cout << "Kinetic-only system: " << sizeKinetic << " unknowns (f_hot "
             << this->Nhot << ", f_re " << this->Nre << ")" << endl;

    this->SelectLinearSolver(sizeKinetic);

    // Same row layout as the matrix (distributed over PETSC_COMM_WORLD)
    MatCreateVecs(this->matrix->mat(), &this->petsc_S, NULL);
}

/**
 * Set the initial guess for the linear solver.
 *
 * guess: Initial guess. If 'nullptr', uses the previous
 *        solution as the initial guess.
 */
void SolverLinearlyImplicit::SetInitialGuess(const real_t * /*guess*/)
{
    /*if (guess != nullptr) {
        PetscScalar *x0;
        VecGetArray(petsc_sol, &x0);

        for (len_t i = 0; i < this->matrix_size; i++)
            x0[i] = guess[i];

        VecRestoreArray(petsc_sol, &x0);
    }*/
    // The initial guess is taken from the UnknownQuantityHandler,
    // and so this routine is not necessary...
}

/**
 * Solve the system of equations.
 *
 * t:  Time at which to solve the system.
 * dt: Time step to take.
 */
void SolverLinearlyImplicit::Solve(const real_t t, const real_t dt)
{
    this->nTimeStep++;

    this->timeKeeper->StartTimer(timerTot);

    bool extiter_conv = true;
    len_t iter = 0;
    do
    {
        iter++;

        if (iter > this->extiter_maxiter)
            throw SolverException(
                "Maximum number of iterations in external iterator reached.");

        if (!extiter_conv)
            unknowns->RestoreSolution(this->nontrivial_unknowns);

        this->timeKeeper->StartTimer(timerRebuild);
        RebuildTerms(t, dt);
        this->timeKeeper->StopTimer(timerRebuild);

        real_t *S;
        VecGetArray(petsc_S, &S);

        this->timeKeeper->StartTimer(timerMatrix);
        BuildMatrix(t, dt, matrix, S);
        this->timeKeeper->StopTimer(timerMatrix);

        matrix->View(
            FVM::Matrix::BINARY_MATLAB,
            "petsc_mat_serial_step" + std::to_string(this->nTimeStep) + "_iter" + std::to_string(iter));

        // Negate vector
        // We do this since in DREAM, we write the equation as
        //
        //   Mx + S = 0
        //
        // whereas PETSc solves the equation
        //
        //   Ax = b
        //
        // Thus, b = -S
        for (len_t i = 0; i < matrix->GetNRows(); i++)
            S[i] = -S[i];

        this->SaveDebugInfo(this->nTimeStep, matrix, S);

        VecRestoreArray(petsc_S, &S);

        {
            PetscViewer rhsViewer;
            std::string rhsname = "petsc_rhs_serial_step" + std::to_string(this->nTimeStep) + "_iter" + std::to_string(iter);
            PetscViewerBinaryOpen(PETSC_COMM_WORLD, rhsname.c_str(), FILE_MODE_WRITE, &rhsViewer);
            VecView(petsc_S, rhsViewer);
            PetscViewerDestroy(&rhsViewer);
        }

        // Scale the matrix and RHS (if -dream_scale is set)
        this->Scale(matrix, petsc_S);

        auto start = std::chrono::steady_clock::now();
        this->timeKeeper->StartTimer(timerInvert);
        inverter->Invert(matrix, &petsc_S, &petsc_S);
        this->timeKeeper->StopTimer(timerInvert);

        if (this->nTimeStep == 1)
        {
            auto end = std::chrono::steady_clock::now();
            cout << "iter_0:time " << std::chrono::duration<double>(end - start).count() << " s" << endl;
        }

        // Undo the scaling on the solution
        this->Unscale(petsc_S);

        // Store solution
        unknowns->Store(this->nontrivial_unknowns, petsc_S);

        // Call external iterator (if enabled)
        if (this->extiter != nullptr)
            extiter_conv = this->extiter->Solve(t, dt, this->nTimeStep);
    } while (!extiter_conv);

    {
        PetscViewer rhsViewer;
        std::string rhsname = "petsc_solution_final_step";
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, rhsname.c_str(), FILE_MODE_WRITE, &rhsViewer);
        VecView(petsc_S, rhsViewer);
        PetscViewerDestroy(&rhsViewer);
    }

    if (this->extiter)
        this->extiter_nIterations.push_back(iter);

    this->timeKeeper->StopTimer(timerTot);

    this->IterationFinished();
}

/**
 * Print timing information for this solver.
 */
void SolverLinearlyImplicit::PrintTimings()
{
    this->timeKeeper->PrintTimings(true, 0);
    printf("  %-*s  %3.4f s\n", 20, "Total invert time:", this->timeKeeper->GetSeconds(this->timerInvert));
    this->Solver::PrintTimings_rebuild();
}

/**
 * Save timing information to the given SFile object.
 *
 * sf:   SFile object to save timing information to.
 * path: Path in file to save timing information to.
 */
void SolverLinearlyImplicit::SaveTimings(SFile *sf, const string &path)
{
    this->timeKeeper->SaveTimings(sf, path);

    sf->CreateStruct(path + "/rebuild");
    this->Solver::SaveTimings_rebuild(sf, path + "/rebuild");
}

/**
 * Save debug information, if enabled.
 *
 * it:  Time step index.
 * mat: Linear operator matrix.
 * rhs: Right-hand side vector.
 */
void SolverLinearlyImplicit::SaveDebugInfo(len_t it, FVM::Matrix *mat, const real_t *rhs)
{
    if (this->savetimestep == it || this->savetimestep == 0)
    {
        string suffix = "_" + to_string(it);

        if (this->savematrix)
        {
            string matname;
            if (this->savetimestep == 0)
                matname = "petsc_mat" + suffix;
            else
                matname = "petsc_mat";

            mat->View(FVM::Matrix::BINARY_MATLAB, matname);
        }

        if (this->saverhs)
        {
            string rhsname;
            if (this->savetimestep == 0)
                rhsname = "rhs" + suffix + ".mat";
            else
                rhsname = "rhs.mat";

            SFile *sf = SFile::Create(rhsname, SFILE_MODE_WRITE);
            sf->WriteList("rhs", rhs, mat->GetNRows());
            sf->Close();
        }

        // Save full output?
        if (this->savesystem)
        {
            string outname = "debugout";
            if (this->savetimestep == 0)
                outname += suffix;
            outname += ".h5";

            OutputGeneratorSFile *outgen = new OutputGeneratorSFile(this->eqsys, outname, true);
            outgen->SaveCurrent();
            delete outgen;
        }

        if (this->printmatrixinfo)
            mat->PrintInfo();
    }
}

/**
 * Enable or disable debug mode (i.e. writing linear operator matrix
 * to file)
 *
 * printinfo:  If true, prints matrix debug info in every time step.
 * savematrix: If true, saves the linear operator matrix using a PETSc Viewer.
 * saverhs:    If true, saves the RHS vector to a MAT file.
 * timestep:   Index of time step for which to save the matrix. If '0', saves the
 *             matrix in all time steps.
 * iteration:  Index of iteration for which to save the matrix.
 * savesystem: If true, saves the full equation system, including grid information,
 *             to a proper DREAMOutput file. However, only the most recently obtained
 *             solution is saved.
 */
void SolverLinearlyImplicit::SetDebugMode(
    bool printinfo, bool savematrix, bool saverhs, int_t timestep,
    bool savesystem)
{
    this->printmatrixinfo = printinfo;
    this->savematrix = savematrix;
    this->saverhs = saverhs;
    this->savetimestep = timestep;
    this->savesystem = savesystem;
}

/**
 * Write basic data/statistics from this solver.
 *
 * sf:   SFile object to use for writing.
 * name: Name of group within file to store data in.
 */
void SolverLinearlyImplicit::WriteDataSFile(SFile *sf, const std::string &name)
{
    sf->CreateStruct(name);

    int32_t type = (int32_t)OptionConstants::SOLVER_TYPE_LINEARLY_IMPLICIT;
    sf->WriteList(name + "/type", &type, 1);

    if (this->extiter != nullptr)
    {
        sf->WriteList(name + "/iterations", this->extiter_nIterations.data(), this->extiter_nIterations.size());
    }
}
