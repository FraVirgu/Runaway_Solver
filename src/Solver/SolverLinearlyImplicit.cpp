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

#include <vector>
#include "DREAM/EquationSystem.hpp"
#include "DREAM/IO.hpp"
#include "DREAM/OutputGeneratorSFile.hpp"
#include "DREAM/Settings/OptionConstants.hpp"
#include "DREAM/Solver/SolverLinearlyImplicit.hpp"

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
    const len_t size, std::vector<len_t> &)
{

    cout << "Fino a qua ci arriva" << endl;
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

        /*

        NumberOfElements() — how many rows this quantity occupies. For f_hot on a 3D grid that's nr × np1 × np2; for a fluid quantity it's just nr.
        NumberOfNonZeros() — an estimate of non-zeros per row in this quantity's block row. This is purely a preallocation hint, not a constraint.

        CreateSubEquation:
        se.offset = this->next_subindex;   // where this block starts
        this->subeqs.push_back(se);        // this is the part of the GLOBAL matrix that we return
        this->next_subindex += n;          // bump the running cursor
        return (this->subeqs.size()-1);    // the block's index    // the part of the GLOBAL matrix above



        unknownToMatrixMapping[id] therefore translates DREAM unknown ID → block index

        */
    }

    matrix->ConstructSystem();
    /*
    PetscInt mSize = this->next_subindex;         // total rows = total cols
    PetscInt *nnz = new PetscInt[mSize];
    for (struct _subeq& s : this->subeqs) {
        PetscInt snnz = s.nnz;
        if (snnz > mSize) snnz = mSize;           // clamp
        for (PetscInt i = 0; i < s.n; i++)
            nnz[s.offset + i] = snnz;             // fan out per-block → per-row
    }
    this->Construct(mSize, mSize, 0, nnz);
    delete [] nnz;



    */

    // Row counts, not quantity counts.
    this->Nhot = 0;
    for (len_t id : fhot)
        this->Nhot += this->unknown_equations->at(id)->NumberOfElements();

    this->Nre = 0;
    for (len_t id : fre)
        this->Nre += this->unknown_equations->at(id)->NumberOfElements();

    this->SelectLinearSolver(size);

    MatCreateVecs(matrix->mat(), nullptr, &this->petsc_S);
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

        real_t *S = new real_t[matrix_size];

        this->timeKeeper->StartTimer(timerMatrix);
        BuildMatrix(t, dt, matrix, S);

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
        PetscInt rstart, rend;
        VecGetOwnershipRange(petsc_S, &rstart, &rend);

        real_t *Sloc;
        VecGetArray(petsc_S, &Sloc);

        for (PetscInt i = 0; i < rend - rstart; i++)
            Sloc[i] = -S[i + rstart]; // local index i  ←  global index i+rstart

        this->timeKeeper->StopTimer(timerMatrix);

        this->SaveDebugInfo(this->nTimeStep, matrix, S);

        VecRestoreArray(petsc_S, &Sloc);

        // Apply preconditioner (if enabled)
        this->Precondition(matrix, petsc_S);

        this->timeKeeper->StartTimer(timerInvert);
        inverter->Invert(matrix, &petsc_S, &petsc_S);
        this->timeKeeper->StopTimer(timerInvert);

        // Undo preconditioner (if enabled)
        this->UnPrecondition(petsc_S);

        // Store solution
        unknowns->Store(this->nontrivial_unknowns, petsc_S);

        // Call external iterator (if enabled)
        if (this->extiter != nullptr)
            extiter_conv = this->extiter->Solve(t, dt, this->nTimeStep);
    } while (!extiter_conv);

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
