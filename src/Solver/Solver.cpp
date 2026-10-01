/**
 * Implementation of common routines for the 'Solver' routines.
 */

#include <chrono>
#include <cstring>
#include <iostream>

#include <vector>
#include "DREAM/IO.hpp"
#include "DREAM/Solver/Solver.hpp"
#include "FVM/Equation/TermLog.hpp"
#include "DREAM/UnknownQuantityEquation.hpp"
#include "DREAM/EquationSystem.hpp"
#include "FVM/BlockMatrix.hpp"
#include "FVM/Equation/PrescribedParameter.hpp"
#include "FVM/UnknownQuantity.hpp"

// Linear solvers
#include "FVM/Solvers/MIGMRES.hpp"
#include "FVM/Solvers/MILU.hpp"
#include "FVM/Solvers/MIILU.hpp"
#include "FVM/Solvers/MIAMG.hpp"
#ifdef PETSC_HAVE_MKL_PARDISO
#include "FVM/Solvers/MIMKL.hpp"
#endif
#include "FVM/Solvers/MIMUMPS.hpp"
#include "FVM/Solvers/MISuperLU.hpp"

using namespace DREAM;
using namespace std;

/**
 * Constructor.
 */
Solver::Solver(
    FVM::UnknownQuantityHandler *unknowns,
    vector<UnknownQuantityEquation *> *unknown_equations,
    EquationSystem *eqsys,
    const bool verbose,
    enum OptionConstants::linear_solver lsolve,
    enum OptionConstants::linear_solver bksolve)
    : unknowns(unknowns), unknown_equations(unknown_equations), eqsys(eqsys),
      verbose(verbose), linearSolver(lsolve), backupSolver(bksolve)
{

    this->solver_timeKeeper = new FVM::TimeKeeper("Solver rebuild");
    this->timerTot = this->solver_timeKeeper->AddTimer("total", "Total time");
    this->timerCqh = this->solver_timeKeeper->AddTimer("collisionhandler", "Rebuild coll. handler");
    this->timerREFluid = this->solver_timeKeeper->AddTimer("refluid", "Rebuild RunawayFluid");
    this->timerSPIHandler = this->solver_timeKeeper->AddTimer("spihandler", "Rebuild SPIHandler");
    this->timerRebuildTerms = this->solver_timeKeeper->AddTimer("equations", "Rebuild terms");
}

/**
 * Destructor.
 */
Solver::~Solver()
{
    delete this->solver_timeKeeper;
    delete this->convChecker;

    if (this->eConvChecker != nullptr)
        delete this->eConvChecker;

    if (this->diag_prec != nullptr)
        delete this->diag_prec;
    if (this->scaleL != nullptr)
        VecDestroy(&this->scaleL);
    if (this->scaleR != nullptr)
        VecDestroy(&this->scaleR);
    if (this->extiter != nullptr)
        delete this->extiter;

    if (this->mainInverter != nullptr)
        delete this->mainInverter;
    if (this->backupInverter != nullptr)
        delete this->backupInverter;
}

/**
 * Build a jacobian matrix for the equation system.
 *
 * t:    Time to build the jacobian matrix for.
 * dt:   Length of time step to take.
 * mat:  Matrix to use for storing the jacobian.
 */
void Solver::BuildJacobian(const real_t, const real_t, FVM::BlockMatrix *jac)
{
    // Reset jacobian matrix
    jac->Zero();

    // Iterate over (non-trivial) unknowns (i.e. those which appear
    // in the matrix system), corresponding to blocks in F and
    // rows in the Jacobian matrix.
    for (len_t uqnId : nontrivial_unknowns)
    {
        UnknownQuantityEquation *eqn = unknown_equations->at(uqnId);
        map<len_t, len_t> &utmm = this->unknownToMatrixMapping;
        len_t matUqnId = utmm[uqnId];
        // Iterate over each equation term
        len_t operatorId = 0;
        for (auto it = eqn->GetOperators().begin(); it != eqn->GetOperators().end(); it++)
        {
            const real_t *x = unknowns->GetUnknownData(it->first);

            // "Differentiate with respect to the unknowns which
            // appear in the matrix"
            //   d (F_uqnId) / d x_derivId
            for (len_t derivId : nontrivial_unknowns)
            {
                len_t matDerivId = utmm[derivId];
                jac->SelectSubEquation(matUqnId, matDerivId);

                // - in the equation for                           x_uqnId
                // - differentiate the operator that is applied to x_it
                // - with respect to                               x_derivId
                it->second->SetJacobianBlock(it->first, derivId, jac, x);
            }

            operatorId++;
        }

        // printf("operatorId = " LEN_T_PRINTF_FMT "\n", operatorId);
    }
    jac->PartialAssemble();

    // Apply boundary conditions which overwrite elements
    for (len_t uqnId : nontrivial_unknowns)
    {
        UnknownQuantityEquation *eqn = unknown_equations->at(uqnId);
        map<len_t, len_t> &utmm = this->unknownToMatrixMapping;
        len_t matUqnId = utmm[uqnId];

        // Iterate over each equation
        for (auto it = eqn->GetOperators().begin(); it != eqn->GetOperators().end(); it++)
        {
            const real_t *x = unknowns->GetUnknownData(it->first);

            // "Differentiate with respect to the unknowns which
            // appear in the matrix"
            //   d (eqn_uqnId) / d x_derivId
            for (len_t derivId : nontrivial_unknowns)
            {
                len_t matDerivId = utmm[derivId];
                jac->SelectSubEquation(matUqnId, matDerivId);
                // For logic, see comment in the for-loop above
                it->second->SetJacobianBlockBC(it->first, derivId, jac, x);
            }
        }
    }
    jac->Assemble();
}

/**
 * Build a linear operator matrix for the equation system.
 *
 * t:    Time to build the jacobian matrix for.
 * dt:   Length of time step to take.
 * mat:  Matrix to use for storing the jacobian.
 * rhs:  Right-hand-side in equation.
 */
void Solver::BuildMatrix(const real_t, const real_t, FVM::BlockMatrix *mat, real_t *S)
{
    // Optional log of how the matrix and the RHS are built, with timings:
    //   -dream_log_matrix      log the first call
    //   -dream_log_matrix N    log the first N calls
    static PetscInt nLog = -1;
    static PetscInt nCalls = 0;
    if (nLog < 0)
    {
        PetscBool set = PETSC_FALSE;
        PetscOptionsHasName(NULL, NULL, "-dream_log_matrix", &set);
        nLog = 0;
        if (set)
        {
            nLog = 1;
            PetscOptionsGetInt(NULL, NULL, "-dream_log_matrix", &nLog, NULL);
        }
    }
    nCalls++;
    const bool log = (nCalls <= nLog);

    typedef std::chrono::steady_clock clk;
    auto ms = [](clk::time_point a, clk::time_point b)
    { return std::chrono::duration<double, std::milli>(b - a).count(); };
    clk::time_point tAll = clk::now(), t0 = tAll;

    if (log)
        printf("\n[matrix] call %d: %d x %d system, %d unknown(s) in the matrix\n",
               (int)nCalls, (int)matrix_size, (int)matrix_size, (int)nontrivial_unknowns.size());

    // Reset matrix and rhs
    mat->Zero();
    for (len_t i = 0; i < matrix_size; i++)
        S[i] = 0;
    if (log)
        printf("[matrix]  reset matrix and RHS %56s %9.3f ms\n", "", ms(t0, clk::now()));

    // Build matrix
    for (len_t uqnId : nontrivial_unknowns)
    {
        UnknownQuantityEquation *eqn = unknown_equations->at(uqnId);
        map<len_t, len_t> &utmm = this->unknownToMatrixMapping;
        len_t matUqnId = utmm[uqnId]; // selecting row
        if (log)
            printf("[matrix]  equation for %s (rows %d..%d): %s\n",
                   unknowns->GetUnknown(uqnId)->GetName().c_str(),
                   (int)mat->GetOffset(matUqnId),
                   (int)(mat->GetOffset(matUqnId) + unknowns->GetUnknown(uqnId)->NumberOfElements()) - 1,
                   eqn->GetDescription().c_str());

        for (auto it = eqn->GetOperators().begin(); it != eqn->GetOperators().end(); it++)
        {
            const char *colName = unknowns->GetUnknown(it->first)->GetName().c_str();
            t0 = clk::now();

            if (utmm.find(it->first) != utmm.end())
            {
                /*
                SelectSubEquation(2, 0)   →  rowOffset=6000, colOffset=0   (no matrix change)
                SetElement(5, 7, 1.3)     →  MatSetValue(mat, 6005, 7, 1.3)  (matrix changes here)
                */
                mat->SelectSubEquation(matUqnId, utmm[it->first]); //   utmm[it->first] selecting
                PetscInt vecoffs = mat->GetOffset(matUqnId);
                if (log)
                    printf("[matrix]    block (%s, %s) -> MATRIX\n",
                           unknowns->GetUnknown(uqnId)->GetName().c_str(), colName);
                FVM::TermLog::matrix() = log;
                it->second->SetMatrixElements(mat, S + vecoffs);
                FVM::TermLog::matrix() = false;

                // The unknown to which this operator should be applied is a
                // "trivial" unknown quantity, meaning it does not appear in the
                // equation system matrix. We therefore build it as part of the
                // RHS vector.
                //
                // (In kinetic-only mode these terms are dropped altogether:
                // see SolverLinearlyImplicit::initialize_kinetic_only().)
            }
            else if (!this->dropNonMatrixTerms)
            {
                PetscInt vecoffs = mat->GetOffset(matUqnId);
                const real_t *data = unknowns->GetUnknownData(it->first);
                if (log)
                    printf("[matrix]    block (%s, %s) -> RHS (%s is not in the matrix)\n",
                           unknowns->GetUnknown(uqnId)->GetName().c_str(), colName, colName);
                FVM::TermLog::matrix() = log;
                it->second->SetVectorElements(S + vecoffs, data);
                FVM::TermLog::matrix() = false;
            }
            else if (log)
                printf("[matrix]    block (%s, %s) -> DROPPED (kinetic-only: %s is not in the matrix)\n",
                       unknowns->GetUnknown(uqnId)->GetName().c_str(), colName, colName);

            if (log)
                printf("[matrix]      block total %56s %9.3f ms\n", "", ms(t0, clk::now()));
        }
    }

    t0 = clk::now();
    mat->Assemble();
    if (log)
    {
        printf("[matrix]  final assembly %62s %9.3f ms\n", "", ms(t0, clk::now()));
        printf("[matrix]  BuildMatrix total %59s %9.3f ms\n", "", ms(tAll, clk::now()));
    }
}

/**
 * Build a function vector for the equation system.
 *
 * t:   Time to build the function vector for.
 * dt:  Length of time step to take.
 * vec: Vector to store evaluated equations in.
 * jac: Associated jacobian matrix.
 */
void Solver::BuildVector(const real_t, const real_t, real_t *vec, FVM::BlockMatrix *jac)
{
    // Reset function vector
    for (len_t i = 0; i < matrix_size; i++)
        vec[i] = 0;

    for (len_t i = 0; i < nontrivial_unknowns.size(); i++)
    {
        len_t uqn_id = nontrivial_unknowns[i];
        len_t vecoffset = jac->GetOffset(i);
        unknown_equations->at(uqn_id)->SetVectorElements(vec + vecoffset, unknowns);
    }
}

/**
 * Calculate the 2-norm of the given vector separately for each
 * non-trivial unknown in the equation system. Thus, if there are
 * N non-trivial unknowns in the equation system, the output vector
 * 'retvec' will contain N elements, each holding the 2-norm of the
 * corresponding section of the input vector 'vec' (which is, for
 * example, a solution vector).
 *
 * vec:    Solution vector to calculate 2-norm of.
 * retvec: Vector which will contain result on return. The vector
 *         must have the same number of elements as there are
 *         non-trivial unknown quantities in the equation system.
 */
void Solver::CalculateNonTrivial2Norm(const real_t *vec, real_t *retvec)
{
    len_t offset = 0, i = 0;
    for (auto id : this->nontrivial_unknowns)
    {
        FVM::UnknownQuantity *uqn = this->unknowns->GetUnknown(id);
        const len_t N = uqn->NumberOfElements();

        retvec[i] = 0;
        for (len_t j = 0; j < N; j++)
            retvec[i] += vec[offset + j] * vec[offset + j];

        retvec[i] = sqrt(retvec[i]);

        offset += N;
        i++;
    }
}

/**
 * Initialize this solver.
 *
 * size:     Number of elements in full unknown vector.
 *           (==> jacobian is of size 'size-by-size').
 * unknowns: List of indices of unknowns to include in the
 *           function vectors/matrices.
 */
void Solver::Initialize(const len_t size, vector<len_t> &unknowns)
{
    this->matrix_size = size;

    // Copy list of non-trivial unknowns (those which will
    // appear in the matrices that are built later on)
    nontrivial_unknowns = unknowns;

    this->initialize_internal(size, unknowns);
}

/**
 * This method is called whenever the solver finishes an
 * iteration. It in turn calls all registered callback
 * functions.
 */
void Solver::IterationFinished()
{
    for (auto f : this->callbacks_iterationFinished)
        (*f)(this->eqsys->GetSimulation());
}

/**
 * Register a function to call whenever a solver iteration
 * has finished.
 */
void Solver::RegisterCallback_IterationFinished(
    iteration_finished_func_t f)
{
    this->callbacks_iterationFinished.push_back(f);
}

/**
 * Rebuild all equation terms in the equation system for
 * the specified time.
 *
 * t:  Time for which to rebuild the equation system.
 * dt: Length of time step to take next.
 */
void Solver::RebuildTerms(const real_t t, const real_t dt)
{
    // Optional log of what is rebuilt, with the time each part takes:
    //   -dream_log_rebuild      log the first call
    //   -dream_log_rebuild N    log the first N calls
    static PetscInt nLog = -1;
    static PetscInt nCalls = 0;
    if (nLog < 0)
    {
        PetscBool set = PETSC_FALSE;
        PetscOptionsHasName(NULL, NULL, "-dream_log_rebuild", &set);
        nLog = 0;
        if (set)
        {
            nLog = 1;
            PetscOptionsGetInt(NULL, NULL, "-dream_log_rebuild", &nLog, NULL);
        }
    }
    nCalls++;
    const bool log = (nCalls <= nLog);

    typedef std::chrono::steady_clock clk;
    auto ms = [](clk::time_point a, clk::time_point b)
    { return std::chrono::duration<double, std::milli>(b - a).count(); };

    if (log)
        printf("\n[rebuild] call %d: t = %g, dt = %g\n", (int)nCalls, (double)t, (double)dt);

    solver_timeKeeper->StartTimer(timerTot);

    clk::time_point t0 = clk::now();
    this->ionHandler->Rebuild();
    // Rebuild ionHandler, collision handlers and RunawayFluid
    if (log)
        printf("[rebuild]  1. IonHandler                       %9.3f ms\n", ms(t0, clk::now()));

    solver_timeKeeper->StartTimer(timerCqh);
    t0 = clk::now();
    if (this->cqh_hottail != nullptr)
        this->cqh_hottail->Rebuild();
    if (log)
        printf("[rebuild]  2. collision handler, hot-tail     %s%9.3f ms\n",
               this->cqh_hottail != nullptr ? "" : "(none) ", ms(t0, clk::now()));
    t0 = clk::now();
    if (this->cqh_runaway != nullptr)
        this->cqh_runaway->Rebuild();
    if (log)
        printf("[rebuild]  3. collision handler, runaway      %s%9.3f ms\n",
               this->cqh_runaway != nullptr ? "" : "(none) ", ms(t0, clk::now()));
    solver_timeKeeper->StopTimer(timerCqh);

    solver_timeKeeper->StartTimer(timerREFluid);
    t0 = clk::now();
    this->REFluid->Rebuild(t);
    if (log)
        printf("[rebuild]  4. RunawayFluid                     %9.3f ms\n", ms(t0, clk::now()));
    solver_timeKeeper->StopTimer(timerREFluid);

    solver_timeKeeper->StartTimer(timerRebuildTerms);
    // Update prescribed quantities and external unknowns
    if (log)
        printf("[rebuild]  5. predetermined unknowns (rebuilt and stored):\n");
    const len_t N = unknowns->Size();
    for (len_t i = 0; i < N; i++)
    {
        FVM::UnknownQuantity *uqty = unknowns->GetUnknown(i);
        UnknownQuantityEquation *eqn = unknown_equations->at(i);

        if (eqn->IsPredetermined())
        {
            t0 = clk::now();
            eqn->RebuildEquations(t, dt, unknowns);
            FVM::PredeterminedParameter *pp = eqn->GetPredetermined();
            uqty->Store(pp->GetData(), 0, true);
            if (log)
                printf("[rebuild]       %-12s (%s) %9.3f ms\n", uqty->GetName().c_str(),
                       eqn->GetDescription().c_str(), ms(t0, clk::now()));
        }
    }

    solver_timeKeeper->StartTimer(timerSPIHandler);
    if (this->SPI != nullptr)
    {
        t0 = clk::now();
        this->SPI->Rebuild(dt, t);
        if (log)
            printf("[rebuild]  6. SPIHandler                       %9.3f ms\n", ms(t0, clk::now()));
    }
    else if (log)
        printf("[rebuild]  6. SPIHandler                       (none)\n");
    solver_timeKeeper->StopTimer(timerSPIHandler);

    if (this->bootstrap != nullptr)
    {
        t0 = clk::now();
        this->bootstrap->Rebuild();
        if (log)
            printf("[rebuild]  7. BootstrapCurrent                 %9.3f ms\n", ms(t0, clk::now()));
    }
    else if (log)
        printf("[rebuild]  7. BootstrapCurrent                 (none)\n");

    if (log)
        printf("[rebuild]  8. equation operators of the %d non-trivial unknown(s):\n",
               (int)nontrivial_unknowns.size());
    for (len_t i = 0; i < nontrivial_unknowns.size(); i++)
    {
        len_t uqnId = nontrivial_unknowns[i];
        UnknownQuantityEquation *eqn = unknown_equations->at(uqnId);

        if (log)
            printf("[rebuild]       equation for %s: %s\n",
                   unknowns->GetUnknown(uqnId)->GetName().c_str(),
                   eqn->GetDescription().c_str());

        for (auto it = eqn->GetOperators().begin(); it != eqn->GetOperators().end(); it++)
        {
            if (log)
                printf("[rebuild]         operator on %-10s %s, ~%d nnz/row\n",
                       unknowns->GetUnknown(it->first)->GetName().c_str(),
                       it->second->HasTransientTerm() ? "(transient)" : "           ",
                       (int)it->second->GetNumberOfNonZerosPerRow());

            // The operator itself prints each of its terms, with timings
            FVM::TermLog::rebuild() = log;
            t0 = clk::now();
            it->second->RebuildTerms(t, dt, unknowns);
            FVM::TermLog::rebuild() = false;
            if (log)
                printf("[rebuild]           operator total %50s %9.3f ms\n", "", ms(t0, clk::now()));
        }
    }

    solver_timeKeeper->StopTimer(timerRebuildTerms);
    solver_timeKeeper->StopTimer(timerTot);
}

static const char *ScaleName(Solver::ScaleMode s)
{
    return s == Solver::ScaleMode::None ? "none" : s == Solver::ScaleMode::Diag ? "diag"
                                                                                : "ruiz";
}

/**
 * Build the scaling vectors L (rows) and R (columns) for A' = L A R.
 *   none  L = R = 1
 *   diag  L = 1/diag(A), R = 1   (zero diagonal entries left unscaled)
 *   ruiz  5 sweeps of simultaneous row/column infinity-norm equilibration:
 *         L <- L / sqrt(max_j |a_ij|),  R <- R / sqrt(max_i |a_ij|)
 */
void Solver::BuildScaling(Mat A, ScaleMode mode, Vec L, Vec R)
{
    VecSet(L, 1.0);
    VecSet(R, 1.0);
    if (mode == ScaleMode::None)
        return;

    if (mode == ScaleMode::Diag)
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
 * Scale the given matrix and RHS vector by row/column scaling,
 *
 *   A x = b   ->   (L A R) x' = L b,   x = R x'
 *
 * so that the rows (and, for Ruiz, the columns) of the matrix are on
 * comparable scales before it is handed to the linear solver. The
 * kinetic rows of the DREAM matrix differ by many orders of magnitude,
 * which AMG in particular does not tolerate (see Unscale() for the
 * matching back-transformation).
 *
 * The scaling is selected once, from -dream_scale (none|diag|ruiz,
 * default none), and recomputed on every call, since the matrix entries
 * change every timestep. The matrix is scaled in place.
 *
 * mat: Matrix to scale (in place).
 * rhs: Right-hand side vector to scale (in place).
 */
void Solver::Scale(FVM::Matrix *mat, Vec rhs)
{
    if (!this->scaleModeRead)
    {
        char name[16] = "none";
        PetscOptionsGetString(NULL, NULL, "-dream_scale", name, sizeof(name), NULL);

        if (strcmp(name, "none") == 0)
            this->scaleMode = ScaleMode::None;
        else if (strcmp(name, "diag") == 0)
            this->scaleMode = ScaleMode::Diag;
        else if (strcmp(name, "ruiz") == 0)
            this->scaleMode = ScaleMode::Ruiz;
        else
            throw SolverException(
                "Unrecognised -dream_scale '%s' (expected none, diag or ruiz).", name);

        this->scaleModeRead = true;
        if (this->verbose)
            cout << "Matrix scaling: " << ScaleName(this->scaleMode) << endl;
    }

    if (this->scaleMode == ScaleMode::None)
        return;

    Mat A = mat->mat();

    // L has the layout of the rows, R that of the columns.
    if (this->scaleL == nullptr)
        MatCreateVecs(A, &this->scaleR, &this->scaleL);

    BuildScaling(A, this->scaleMode, this->scaleL, this->scaleR);
    MatDiagonalScale(A, this->scaleL, this->scaleR);
    VecPointwiseMult(rhs, this->scaleL, rhs);
}

/**
 * Transform the solution of the scaled system back to the original
 * variables: x = R x'. Must be called after the linear solve whenever
 * Scale() was called before it.
 *
 * With -dream_scale diag, R = 1 and this is a no-op.
 */
void Solver::Unscale(Vec x)
{
    if (this->scaleMode == ScaleMode::None || this->scaleR == nullptr)
        return;

    VecPointwiseMult(x, this->scaleR, x);
}

/**
 * Print timing information for the 'Rebuild' stage of the solver.
 * This stage looks the same for all solvers, and so is conveniently
 * defined here in the base class.
 */
void Solver::PrintTimings_rebuild()
{
    this->solver_timeKeeper->PrintTimings(true, 0);
}

/**
 * Save timing information for the 'Rebuild' stage of the solver
 * to the given SFile object.
 */
void Solver::SaveTimings_rebuild(SFile *sf, const std::string &path)
{
    this->solver_timeKeeper->SaveTimings(sf, path);
}

/**
 * Select the linear solver to use.
 *
 * N: Number of rows (or columns) in matrix to invert.
 */
void Solver::SelectLinearSolver(const len_t N)
{
    if (this->linearSolver == this->backupSolver)
        throw SolverException(
            "The main and backup linear solvers may not be the same.");

    this->mainInverter = this->ConstructLinearSolver(N, this->linearSolver);
    this->inverter = this->mainInverter;

    if (this->backupSolver != OptionConstants::LINEAR_SOLVER_NONE)
        this->backupInverter = this->ConstructLinearSolver(N, this->backupSolver);
}

/**
 * Check if GMRES has converged.
 *
 * ksp:    KSP solver context.
 * it:     Iteration number.
 * rnorm:  Estimated 2-norm of preconditioned residual.
 * reason: (return) reason for convergence.
 * cctx:   Convergence context.
 */
/*PetscErrorCode CheckGMRESConverged(
    KSP ksp, PetscInt it, PetscReal, KSPConvergedReason *reason,
    void *cctx
) {
    Solver *solver = (Solver*)cctx;
    ConvergenceChecker *cc = solver->GetConvergenceChecker();

    Vec _x, _dx;
    VecCreateSeq(PETSC_COMM_WORLD, solver->GetMatrixSize(), &_x);

    KSPBuildSolution(ksp, _x, &_x);
    KSPBuildResidual(ksp, NULL, NULL, &_dx);

    real_t *x, *dx;
    VecGetArray(_x, &x);
    VecGetArray(_dx, &dx);

    bool conv = cc->IsConverged(x, dx, false);

    VecRestoreArray(_dx, &dx);
    VecRestoreArray(_x, &x);

    VecDestroy(&_dx);
    VecDestroy(&_x);

    printf("Converged? %s\n", conv?"yes":"no");

    if (conv)
        *reason = KSP_CONVERGED_RTOL_NORMAL;
    else
        *reason = KSP_CONVERGED_ITERATING;

    return 0;
}*/

FVM::MatrixInverter *Solver::ConstructLinearSolver(const len_t N, enum OptionConstants::linear_solver ls)
{
    if (ls == OptionConstants::LINEAR_SOLVER_GMRES)
    {
        // return new FVM::MIGMRES(N, nontrivial_unknowns, unknowns, &CheckGMRESConverged, this);
        return new FVM::MIGMRES(N, nontrivial_unknowns, unknowns, nullptr, nullptr);
    }
    else if (ls == OptionConstants::LINEAR_SOLVER_LU)
        return new FVM::MILU(N);
    else if (ls == OptionConstants::LINEAR_SOLVER_ILU)
        return new FVM::MIILU(N, this->Nhot, this->Nre);
    else if (ls == OptionConstants::LINEAR_SOLVER_AMG)
        return new FVM::MIAMG(N);
    else if (ls == OptionConstants::LINEAR_SOLVER_MKL)
    {
#ifdef PETSC_HAVE_MKL_PARDISO
        return new FVM::MIMKL(N);
#else
        throw SolverException(
            "Your version of PETSc does not include support for Intel MKL PARDISO. "
            "To use this linear solver you must recompile PETSc.");
#endif
    }
    else if (ls == OptionConstants::LINEAR_SOLVER_MUMPS)
    {
#ifdef PETSC_HAVE_MUMPS
        return new FVM::MIMUMPS(N);
#else
        throw SolverException(
            "Your version of PETSc does not include support for MUMPS. "
            "To use this linear solver you must recompile PETSc.");
#endif
    }
    else if (ls == OptionConstants::LINEAR_SOLVER_SUPERLU)
    {
#ifdef PETSC_HAVE_SUPERLU
        return new FVM::MISuperLU(N);
#else
        throw SolverException(
            "Your version of PETSc does not include support for SuperLU. "
            "To use this linear solver you must recompile PETSc.");
#endif
    }
    else
        throw SolverException(
            "Unrecognized linear solver specified: %d.", ls);
}

/**
 * Set the convergence checker to use for the linear solver.
 */
void Solver::SetConvergenceChecker(ConvergenceChecker *cc)
{
    if (this->convChecker != nullptr)
        delete this->convChecker;

    this->convChecker = cc;
}

/**
 * Set the external iterator to use.
 */
void Solver::SetExternalIterator(ExternalIterator *ei)
{
    if (this->extiter != nullptr)
        delete this->extiter;

    this->extiter = ei;
    this->extiter->SetVerbose(this->verbose);

    if (this->eConvChecker != nullptr)
        this->extiter->SetConvergenceChecker(this->eConvChecker);
}

/**
 * Set the external iterator to use.
 */
void Solver::SetExternalIteratorConvergenceChecker(ConvergenceChecker *cc)
{
    if (this->eConvChecker != nullptr)
        delete this->eConvChecker;

    this->eConvChecker = cc;

    if (this->extiter != nullptr)
        this->extiter->SetConvergenceChecker(this->eConvChecker);
}

/**
 * Set the preconditioner to use for solver. If 'nullptr', no
 * preconditioning is done.
 */
void Solver::SetPreconditioner(DiagonalPreconditioner *dp)
{
    if (this->diag_prec != nullptr)
        delete this->diag_prec;

    this->diag_prec = dp;
}

/**
 * Switch to using the backup inverter instead of the main inverter.
 */
void Solver::SwitchToBackupInverter()
{
    if (this->inverter == this->backupInverter)
        throw DREAMException("Backup matrix inverter failed with PETSc error: " INT_T_PRINTF_FMT, inverter->GetReturnCode());

    this->inverter = this->backupInverter;
}

/**
 * Switch to using the main inverter.
 */
void Solver::SwitchToMainInverter()
{
    this->inverter = this->mainInverter;
}

/**
 * Empty routine for writing solver data to output file.
 * This method should be overridden where needed.
 */
void Solver::WriteDataSFile(SFile *, const std::string &) {}
