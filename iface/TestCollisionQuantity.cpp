/**
 * Test of the distributed collision quantities and kinetic operators:
 * builds the simulation from a DREAM settings file exactly as dreami does,
 * does the first rebuild of the solver
 * (Solver::RebuildTermsTransientAdvectionDiffusion(): collision quantity
 * handlers, then transient, advection-diffusion and boundary terms, each
 * on the rows of this rank) and saves the result.
 *
 * Each rank only computes the radii containing its own rows of the
 * distribution function. The values of all ranks are collected on rank 0
 * (a radius shared by two ranks is computed by both, and the two values
 * are averaged) and written in global order, so that the files of a
 * serial and of a parallel run can be compared element by element with
 * compare_collision_quantities.py.
 *
 * For each kinetic grid (hottail, runaway) it saves, on the p1 and p2
 * flux grids,
 *   - the values of lnLambdaEE, lnLambdaEI, nuS, nuD and nuPar,
 *   - the partial derivatives of nuS, nuD and nuPar with respect to
 *     n_cold and n_i (used to build the Jacobian).
 * A radius that no rank has computed is saved as NaN.
 *
 * The memory file reports, per rank, the bytes stored for the collision
 * values, the transient weights, the advection interpolation coefficients
 * and the advection/diffusion coefficients of f_hot and f_re (against what
 * a serial run stored before they were distributed), the storage of the
 * PETSc matrix, and the resident set size of the process at the end of
 * each stage of the construction (recorded with RecordProcessMemory(), see
 * DREAM/Settings/ProcessMemory.hpp).
 *
 * Syntax:
 *   mpirun -n <N> ./testcollisionquantity dream_settings.h5
 *
 * Output (PETSc binary vectors, in the directory given by -cq_dir,
 * default the current directory):
 *   cq_np<N>_<grid>_<quantity>_<f1|f2>
 *   cq_np<N>_<grid>_<quantity>_d<unknown>_<f1|f2>
 * and cq_np<N>_memory.txt with the memory used by each rank.
 */

#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <H5Cpp.h>

#include <petscvec.h>
#include <softlib/SOFTLibException.h>

#include "DREAM/Init.h"
#include "DREAM/IO.hpp"
#include "DREAM/Settings/Settings.hpp"
#include "DREAM/Settings/SFile.hpp"
#include "DREAM/Settings/SimulationGenerator.hpp"
#include "DREAM/Simulation.hpp"
#include "DREAM/Settings/ProcessMemory.hpp"
#include "FVM/BlockMatrix.hpp"
#include "FVM/FVMException.hpp"
#include "FVM/Equation/AdvectionDiffusionTerm.hpp"
#include "FVM/Equation/Operator.hpp"

using namespace std;
using namespace DREAM;

/**
 * Collect on rank 0 a quantity stored as 'nMul' blocks of 'nr' radii of
 * 'n' elements each ([iMul][ir][i], flattened), of which every rank holds
 * the radii marked in 'owned'. Returns the full quantity on rank 0 (an
 * empty vector on the other ranks).
 */
static vector<real_t> GatherRadii(
    const real_t *data, const len_t nMul, const len_t nr, const len_t n,
    const vector<bool> &owned
) {
    const len_t N = nMul*nr*n;
    vector<real_t> sum(N, 0.0), cnt(nr, 0.0);
    for (len_t ir = 0; ir < nr; ir++) {
        if (!owned[ir])
            continue;
        cnt[ir] = 1;
        for (len_t m = 0; m < nMul; m++)
            for (len_t i = 0; i < n; i++)
                sum[(m*nr + ir)*n + i] = data[(m*nr + ir)*n + i];
    }

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    vector<real_t> sumAll(rank == 0 ? N : 0), cntAll(rank == 0 ? nr : 0);
    MPI_Reduce(sum.data(), sumAll.data(), (int)N, MPIU_REAL, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(cnt.data(), cntAll.data(), (int)nr, MPIU_REAL, MPI_SUM, 0, PETSC_COMM_WORLD);

    if (rank != 0)
        return vector<real_t>();

    for (len_t m = 0; m < nMul; m++)
        for (len_t ir = 0; ir < nr; ir++)
            for (len_t i = 0; i < n; i++) {
                real_t &v = sumAll[(m*nr + ir)*n + i];
                v = (cntAll[ir] > 0) ? v / cntAll[ir] : numeric_limits<real_t>::quiet_NaN();
            }

    return sumAll;
}

/**
 * Same as above, for a quantity stored as one array per radius (nullptr
 * for the radii which this rank has not computed).
 */
static vector<real_t> GatherRows(real_t *const* rows, const len_t nr, const len_t n) {
    vector<bool> owned(nr);
    vector<real_t> flat(nr*n, 0.0);
    for (len_t ir = 0; ir < nr; ir++) {
        owned[ir] = (rows[ir] != nullptr);
        if (owned[ir])
            for (len_t i = 0; i < n; i++)
                flat[ir*n + i] = rows[ir][i];
    }
    return GatherRadii(flat.data(), 1, nr, n, owned);
}

/**
 * Write 'x' (on rank 0 only) as a PETSc binary vector.
 */
static void Save(const string &name, const vector<real_t> &x) {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    Vec v;
    PetscViewer viewer;
    VecCreateSeqWithArray(PETSC_COMM_SELF, 1, (PetscInt)x.size(), x.data(), &v);
    PetscViewerBinaryOpen(PETSC_COMM_SELF, name.c_str(), FILE_MODE_WRITE, &viewer);
    VecView(v, viewer);
    PetscViewerDestroy(&viewer);
    VecDestroy(&v);
}

/**
 * Print, rank by rank, the radii at which this rank has computed the
 * collision quantities of the given handler.
 */
static void PrintOwnedRadii(const string &gridname, CollisionQuantityHandler *cqh, const len_t nr) {
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    real_t *const* rows = cqh->GetNuS()->GetValue_f1();
    len_t irlo = nr, irhi = 0;
    for (len_t ir = 0; ir < nr; ir++)
        if (rows[ir] != nullptr) {
            irlo = min(irlo, ir);
            irhi = ir+1;
        }

    for (int r = 0; r < size; r++) {
        if (r == rank) {
            if (irhi > irlo)
                printf("[%s] rank %d: radii [%d, %d) of %d\n", gridname.c_str(), rank, (int)irlo, (int)irhi, (int)nr);
            else
                printf("[%s] rank %d: no radius of %d\n", gridname.c_str(), rank, (int)nr);
            fflush(stdout);
        }
        MPI_Barrier(PETSC_COMM_WORLD);
    }
}

/**
 * Number of bytes allocated by this rank for the values of the collision
 * quantities of one kinetic grid (all radii if 'all' is true, i.e. what a
 * serial run allocates).
 */
static real_t CollisionValueBytes(CollisionQuantityHandler *cqh, FVM::Grid *grid, const bool all) {
    const len_t nr  = grid->GetNr();
    const len_t np1 = grid->GetMomentumGrid(0)->GetNp1();
    const len_t np2 = grid->GetMomentumGrid(0)->GetNp2();

    CollisionQuantity *quantities[] = {
        cqh->GetLnLambdaEE(), cqh->GetLnLambdaEI(),
        cqh->GetNuS(), cqh->GetNuD(), cqh->GetNuPar()
    };
    struct flux { FVM::fluxGridType type; len_t n; };
    const flux fluxGrids[] = {
        {FVM::FLUXGRIDTYPE_P1, (np1+1)*np2},
        {FVM::FLUXGRIDTYPE_P2, np1*(np2+1)}
    };

    real_t bytes = 0;
    for (CollisionQuantity *q : quantities)
        for (const flux &f : fluxGrids) {
            real_t *const* rows = q->GetValue(f.type);
            for (len_t ir = 0; ir < nr; ir++)
                if (all || rows[ir] != nullptr)
                    bytes += f.n*sizeof(real_t);
        }

    return bytes;
}

/**
 * Number of bytes allocated by this rank for the operator of a kinetic
 * equation (the diagonal block of unknown 'uqty'):
 *   trans:  weights of its transient terms,
 *   interp: interpolation coefficients of its advection-diffusion term,
 *   coeff:  advection (fr, f1, f2, f1pSqAtZero) and diffusion (drr, d11,
 *           d12, d21, d22) coefficients of its advection-diffusion term,
 * and (transAll, interpAll, coeffAll) what a serial run allocated for them
 * before they were distributed (all cells; the jacobian interpolation
 * coefficients always stored).
 */
static void OperatorBytes(
    EquationSystem *eqsys, const string &uqty,
    real_t &trans, real_t &transAll, real_t &interp, real_t &interpAll,
    real_t &coeff, real_t &coeffAll
) {
    trans = transAll = interp = interpAll = coeff = coeffAll = 0;
    FVM::UnknownQuantityHandler *u = eqsys->GetUnknownHandler();
    if (!u->HasUnknown(uqty))
        return;

    const len_t id = u->GetUnknownID(uqty);
    const FVM::Operator *op = eqsys->GetOperator(id, id);
    if (op == nullptr)
        return;

    const real_t nCells = (real_t)u->GetUnknown(id)->NumberOfElements();
    for (const FVM::LinearTransientTerm *t : op->GetTransientTerms()) {
        trans    += t->GetNumberOfStoredWeights()*sizeof(real_t);
        transAll += nCells*sizeof(real_t);
    }

    FVM::AdvectionDiffusionTerm *adt = op->GetAdvectionDiffusion();
    if (adt == nullptr)
        return;

    FVM::AdvectionInterpolationCoefficient *deltas[] = {
        adt->GetInterpolationCoeffR(), adt->GetInterpolationCoeff1(), adt->GetInterpolationCoeff2()
    };
    for (FVM::AdvectionInterpolationCoefficient *d : deltas) {
        // Per face: the coefficients and a pointer to them
        const real_t perFace = d->GetStencilSize()*sizeof(real_t) + sizeof(real_t*);
        interp    += d->GetNumberOfStoredFaces()*perFace*(d->StoresJacobian() ? 2 : 1);
        interpAll += d->GetNumberOfFaces()*perFace*2;
    }

    // XXX here we assume that all momentum grids are the same
    FVM::Grid *grid = u->GetUnknown(id)->GetGrid();
    const real_t
        nr  = grid->GetNr(),
        np1 = grid->GetMomentumGrid(0)->GetNp1(),
        np2 = grid->GetMomentumGrid(0)->GetNp2();
    const real_t nR = (nr+1)*np1*np2, n1 = nr*(np1+1)*np2, n2 = nr*np1*(np2+1);

    coeff = (adt->FVM::AdvectionTerm::GetNumberOfStoredCoefficients()
           + adt->FVM::DiffusionTerm::GetNumberOfStoredCoefficients())*sizeof(real_t);
    coeffAll = ((nR + n1 + n2 + nr*np2) + (nR + 2*n1 + 2*n2))*sizeof(real_t);
}

/**
 * Print, and save to '<prefix>memory.txt', the memory used by each rank:
 * the values of the collision quantities, the transient weights and the
 * interpolation coefficients of the kinetic operators (exact), and the
 * whole process (resident set size, current and peak).
 */
static void SaveMemory(const string &prefix, EquationSystem *eqsys) {
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    real_t hot = 0, re = 0, hotAll = 0, reAll = 0;
    if (eqsys->HasHotTailGrid()) {
        hot    = CollisionValueBytes(eqsys->GetHotTailCollisionHandler(), eqsys->GetHotTailGrid(), false);
        hotAll = CollisionValueBytes(eqsys->GetHotTailCollisionHandler(), eqsys->GetHotTailGrid(), true);
    }
    if (eqsys->HasRunawayGrid()) {
        re    = CollisionValueBytes(eqsys->GetRunawayCollisionHandler(), eqsys->GetRunawayGrid(), false);
        reAll = CollisionValueBytes(eqsys->GetRunawayCollisionHandler(), eqsys->GetRunawayGrid(), true);
    }

    // Operators of f_hot and f_re
    real_t trans = 0, transAll = 0, interp = 0, interpAll = 0, coeff = 0, coeffAll = 0;
    for (const string &uqty : {string(OptionConstants::UQTY_F_HOT), string(OptionConstants::UQTY_F_RE)}) {
        real_t t, tAll, d, dAll, c, cAll;
        OperatorBytes(eqsys, uqty, t, tAll, d, dAll, c, cAll);
        trans += t, transAll += tAll, interp += d, interpAll += dAll, coeff += c, coeffAll += cAll;
    }

    // Whole process
    real_t current, peak;
    GetProcessMemory(current, peak);

    // PETSc matrix of the solver (the rows of this rank)
    real_t matNz = 0, matNzUsed = 0, matBytes = 0, matInfoBytes = 0;
    if (FVM::BlockMatrix *bm = eqsys->GetSolver()->GetMatrix()) {
        MatInfo info;
        MatGetInfo(bm->mat(), MAT_LOCAL, &info);
        PetscInt rstart, nloc;
        bm->GetOwnershipRange(&rstart, &nloc);
        matNz = info.nz_allocated;
        matNzUsed = info.nz_used;
        matInfoBytes = info.memory;
        // AIJ storage: a value and a column index per allocated non-zero,
        // and about three integers per row (row start, lengths, diagonal)
        matBytes = matNz*(sizeof(PetscScalar) + sizeof(PetscInt)) + 3.0*nloc*sizeof(PetscInt);
    }

    const int NV = 11;
    real_t mine[NV] = {hot, re, trans, interp, coeff, current, peak, matNz, matNzUsed, matBytes, matInfoBytes};
    vector<real_t> all(rank == 0 ? NV*size : 0);
    MPI_Gather(mine, NV, MPIU_REAL, all.data(), NV, MPIU_REAL, 0, PETSC_COMM_WORLD);

    // Memory at each recorded stage of the construction (the same stages
    // on every rank): [stage][rank], current and peak
    const vector<ProcessMemoryRecord> &stages = ProcessMemoryLog();
    int nStages = (int)stages.size(), nStagesMin;
    MPI_Allreduce(&nStages, &nStagesMin, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD);
    vector<real_t> stageMine(2*nStagesMin);
    for (int k = 0; k < nStagesMin; k++) {
        stageMine[2*k]   = stages[k].current;
        stageMine[2*k+1] = stages[k].peak;
    }
    vector<real_t> stageAll(rank == 0 ? 2*nStagesMin*size : 0);
    MPI_Gather(stageMine.data(), 2*nStagesMin, MPIU_REAL, stageAll.data(), 2*nStagesMin, MPIU_REAL, 0, PETSC_COMM_WORLD);

    if (rank != 0)
        return;

    const real_t MB = 1024.0*1024.0;
    FILE *files[] = {stdout, fopen((prefix + "memory.txt").c_str(), "w")};
    for (FILE *f : files) {
        if (f == nullptr)
            continue;
        fprintf(f, "Memory per rank [MB] (%d rank%s)\n", size, size == 1 ? "" : "s");
        fprintf(f, "  collision values: exact, values of lnLambdaEE/EI, nuS, nuD, nuPar on the p1/p2 flux grids\n");
        fprintf(f, "  transient:        exact, weights of the transient terms of f_hot and f_re\n");
        fprintf(f, "  interpolation:    exact, advection interpolation coefficients (r, p1, p2) of f_hot and f_re\n");
        fprintf(f, "  adv/diff coeff.:  exact, advection and diffusion coefficients of f_hot and f_re\n");
        fprintf(f, "  process RSS:      whole process, current (/proc/self/statm) and peak (getrusage)\n");
        fprintf(f, "%6s  %14s  %14s  %14s  %12s  %14s  %16s  %12s  %12s\n",
            "rank", "coll. hottail", "coll. runaway", "coll. total", "transient", "interpolation",
            "adv/diff coeff.", "RSS", "peak RSS");
        real_t sum[5] = {0, 0, 0, 0, 0};
        for (int r = 0; r < size; r++) {
            const real_t *v = &all[NV*r];
            fprintf(f, "%6d  %14.3f  %14.3f  %14.3f  %12.3f  %14.3f  %16.3f  %12.1f  %12.1f\n",
                r, v[0]/MB, v[1]/MB, (v[0]+v[1])/MB, v[2]/MB, v[3]/MB, v[4]/MB, v[5]/MB, v[6]/MB);
            for (int k = 0; k < 5; k++)
                sum[k] += v[k];
        }
        fprintf(f, "%6s  %14.3f  %14.3f  %14.3f  %12.3f  %14.3f  %16.3f\n",
            "sum", sum[0]/MB, sum[1]/MB, (sum[0]+sum[1])/MB, sum[2]/MB, sum[3]/MB, sum[4]/MB);
        fprintf(f, "%6s  %14.3f  %14.3f  %14.3f  %12.3f  %14.3f  %16.3f   (one rank, before distribution)\n",
            "serial", hotAll/MB, reAll/MB, (hotAll+reAll)/MB, transAll/MB, interpAll/MB, coeffAll/MB);

        // Matrix
        fprintf(f, "\nPETSc matrix of the solver, rows of each rank\n");
        fprintf(f, "  storage: nz allocated x (value + column index) + 3 integers per row;"
                   " MatGetInfo memory: as reported by PETSc (0 if not tracked)\n");
        fprintf(f, "%6s  %14s  %14s  %14s  %16s\n", "rank", "nz allocated", "nz used", "storage [MB]", "MatGetInfo [MB]");
        real_t matSum = 0;
        for (int r = 0; r < size; r++) {
            const real_t *v = &all[NV*r];
            fprintf(f, "%6d  %14.0f  %14.0f  %14.3f  %16.3f\n", r, v[7], v[8], v[9]/MB, v[10]/MB);
            matSum += v[9];
        }
        fprintf(f, "%6s  %14s  %14s  %14.3f\n", "sum", "", "", matSum/MB);

        // Stages
        fprintf(f, "\nResident set size at the end of each stage of the construction [MB]\n");
        fprintf(f, "  min/max: over the ranks; added: largest increase on a rank since the previous stage\n");
        fprintf(f, "%-48s  %10s  %10s  %10s  %10s\n", "stage", "RSS min", "RSS max", "added", "peak max");
        for (int k = 0; k < nStagesMin; k++) {
            real_t lo = 0, hi = 0, added = 0, pk = 0;
            for (int r = 0; r < size; r++) {
                const real_t c = stageAll[2*nStagesMin*r + 2*k];
                const real_t prev = (k > 0 ? stageAll[2*nStagesMin*r + 2*(k-1)] : 0);
                lo = (r == 0 ? c : min(lo, c));
                hi = max(hi, c);
                added = max(added, c - prev);
                pk = max(pk, stageAll[2*nStagesMin*r + 2*k+1]);
            }
            fprintf(f, "%-48s  %10.1f  %10.1f  %10.1f  %10.1f\n",
                stages[k].stage.c_str(), lo/MB, hi/MB, added/MB, pk/MB);
        }
    }
    if (files[1] != nullptr) {
        fclose(files[1]);
        printf("Memory saved as %smemory.txt\n", prefix.c_str());
    }
}

/**
 * Save the collision quantities of one kinetic grid.
 */
static void SaveHandler(
    const string &prefix, const string &gridname, CollisionQuantityHandler *cqh,
    FVM::Grid *grid, EquationSystem *eqsys
) {
    const len_t nr  = grid->GetNr();
    const len_t np1 = grid->GetMomentumGrid(0)->GetNp1();
    const len_t np2 = grid->GetMomentumGrid(0)->GetNp2();
    const len_t nzs = eqsys->GetIonHandler()->GetNzs();

    PrintOwnedRadii(gridname, cqh, nr);

    struct flux { const char *name; FVM::fluxGridType type; len_t n1, n2, n; };
    const flux fluxGrids[] = {
        {"f1", FVM::FLUXGRIDTYPE_P1, np1+1, np2, (np1+1)*np2},
        {"f2", FVM::FLUXGRIDTYPE_P2, np1, np2+1, np1*(np2+1)}
    };

    struct qty { const char *name; CollisionQuantity *q; };
    const qty quantities[] = {
        {"lnLambdaEE", cqh->GetLnLambdaEE()},
        {"lnLambdaEI", cqh->GetLnLambdaEI()},
        {"nuS",        cqh->GetNuS()},
        {"nuD",        cqh->GetNuD()},
        {"nuPar",      cqh->GetNuPar()}
    };

    FVM::UnknownQuantityHandler *u = eqsys->GetUnknownHandler();
    struct unkn { const char *name; len_t id; len_t nMul; };
    const unkn unknowns[] = {
        {"ncold", u->GetUnknownID(OptionConstants::UQTY_N_COLD),      1},
        {"ni",    u->GetUnknownID(OptionConstants::UQTY_ION_SPECIES), nzs}
    };

    for (const flux &f : fluxGrids) {
        // Values
        for (const qty &q : quantities)
            Save(prefix + gridname + "_" + q.name + "_" + f.name,
                GatherRows(q.q->GetValue(f.type), nr, f.n));

        // Partial derivatives; the radii computed by this rank are those
        // at which the values are allocated
        vector<bool> owned(nr);
        real_t *const* rows = cqh->GetNuS()->GetValue(f.type);
        for (len_t ir = 0; ir < nr; ir++)
            owned[ir] = (rows[ir] != nullptr);

        // The partial derivatives are evaluated on the fly, cell by cell,
        // on the owned radii, in the layout (n*nr + ir)*N + j*n1 + i
        auto savePartial = [&](const string &name, const unkn &uq, auto partial) {
            vector<real_t> data(uq.nMul*nr*f.n, 0.0);
            for (len_t n = 0; n < uq.nMul; n++)
                for (len_t ir = 0; ir < nr; ir++) {
                    if (!owned[ir])
                        continue;
                    for (len_t j = 0; j < f.n2; j++)
                        for (len_t i = 0; i < f.n1; i++)
                            data[(n*nr + ir)*f.n + j*f.n1 + i] = partial(uq.id, f.type, ir, i, j, n);
                }
            Save(name, GatherRadii(data.data(), uq.nMul, nr, f.n, owned));
        };
        for (const unkn &uq : unknowns) {
            const string d = string("_d") + uq.name + "_" + f.name;
            savePartial(prefix + gridname + "_nuS" + d, uq,
                [&](len_t id, FVM::fluxGridType t, len_t ir, len_t i, len_t j, len_t n) {
                    return cqh->GetNuS()->GetUnknownPartialContribution(id, t, ir, i, j, n); });
            savePartial(prefix + gridname + "_nuD" + d, uq,
                [&](len_t id, FVM::fluxGridType t, len_t ir, len_t i, len_t j, len_t n) {
                    return cqh->GetNuD()->GetUnknownPartialContribution(id, t, ir, i, j, n); });
            savePartial(prefix + gridname + "_nuPar" + d, uq,
                [&](len_t id, FVM::fluxGridType t, len_t ir, len_t i, len_t j, len_t n) {
                    return cqh->GetNuPar()->GetUnknownPartialContribution(id, t, ir, i, j, n); });
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            cerr << "Syntax: testcollisionquantity INPUT" << endl;
        MPI_Finalize();
        return 1;
    }

    dream_initialize();
    RecordProcessMemory("test: MPI_Init + dream_initialize");

    char dir[PETSC_MAX_PATH_LEN] = ".";
    PetscOptionsGetString(NULL, NULL, "-cq_dir", dir, sizeof(dir), NULL);
    const string prefix = string(dir) + "/cq_np" + to_string(size) + "_";

    int exit_code = 0;
    Simulation *sim = nullptr;
    try {
        // Same construction as dreami
        Settings *settings = SimulationGenerator::CreateSettings();
        SettingsSFile::LoadSettings(settings, argv[1]);
        RecordProcessMemory("test: settings loaded");
        sim = SimulationGenerator::ProcessSettings(settings);

        EquationSystem *eqsys = sim->GetEquationSystem();
        Solver *solver = eqsys->GetSolver();
        if (solver->GetMatrix() == nullptr && rank == 0)
            cout << "The solver has no matrix: the collision quantities "
                    "keep all radii on every rank." << endl;

        // First rebuild done by the solver: the collision quantities and
        // the transient, advection-diffusion and boundary terms, each on
        // the rows of this rank. (The time step only enters the matrix,
        // not what is stored.)
        solver->RebuildTermsTransientAdvectionDiffusion(0, 1);
        RecordProcessMemory("test: first rebuild");

        SaveMemory(prefix, eqsys);

        if (eqsys->HasHotTailGrid())
            SaveHandler(prefix, "hottail", eqsys->GetHotTailCollisionHandler(),
                eqsys->GetHotTailGrid(), eqsys);
        if (eqsys->HasRunawayGrid())
            SaveHandler(prefix, "runaway", eqsys->GetRunawayCollisionHandler(),
                eqsys->GetRunawayGrid(), eqsys);

        if (rank == 0)
            cout << "Collision quantities saved as " << prefix << "*" << endl;
    } catch (FVM::FVMException &ex) {
        IO::PrintError(ex.what());
        exit_code = 1;
    } catch (SOFTLibException &ex) {
        IO::PrintError(ex.what());
        exit_code = 2;
    } catch (H5::FileIException &ex) {
        IO::PrintError(ex.getDetailMsg().c_str());
        exit_code = 3;
    }

    // The simulation owns PETSc objects, destroy them before PETSc
    delete sim;
    dream_finalize();
    MPI_Finalize();
    return exit_code;
}
