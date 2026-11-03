/**
 * Test of the distributed collision quantities: builds the simulation from
 * a DREAM settings file exactly as dreami does, reproduces the first
 * rebuild of the collision quantity handlers done by the solver
 * (Solver::RebuildTerms*(): SetCollisionQuantityLocalRange() followed by
 * CollisionQuantityHandler::Rebuild()) and saves the result.
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
 *     n_cold, n_i and T_cold (used to build the Jacobian).
 * A radius that no rank has computed is saved as NaN.
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
#include <sys/resource.h>
#include <unistd.h>

#include <petscvec.h>
#include <softlib/SOFTLibException.h>

#include "DREAM/Init.h"
#include "DREAM/IO.hpp"
#include "DREAM/Settings/Settings.hpp"
#include "DREAM/Settings/SFile.hpp"
#include "DREAM/Settings/SimulationGenerator.hpp"
#include "DREAM/Simulation.hpp"
#include "FVM/FVMException.hpp"

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
 * Print, and save to '<prefix>memory.txt', the memory used by each rank:
 * the values of the collision quantities (exact) and the whole process
 * (resident set size, current and peak).
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

    // Resident set size from Linux: current from /proc/self/statm (second
    // field, in pages), peak from getrusage (ru_maxrss, in kB).
    // (PetscMemoryGetCurrentUsage() is not used: without
    // PETSC_USE_PROCFS_FOR_SIZE it returns ru_maxrss, i.e. the peak.)
    real_t current = 0;
    if (FILE *statm = fopen("/proc/self/statm", "r")) {
        unsigned long size, resident;
        if (fscanf(statm, "%lu %lu", &size, &resident) == 2)
            current = (real_t)resident*sysconf(_SC_PAGESIZE);
        fclose(statm);
    }
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    const real_t peak = (real_t)usage.ru_maxrss*1024.0;

    real_t mine[4] = {hot, re, current, peak};
    vector<real_t> all(rank == 0 ? 4*size : 0);
    MPI_Gather(mine, 4, MPIU_REAL, all.data(), 4, MPIU_REAL, 0, PETSC_COMM_WORLD);
    if (rank != 0)
        return;

    const real_t MB = 1024.0*1024.0;
    FILE *files[] = {stdout, fopen((prefix + "memory.txt").c_str(), "w")};
    for (FILE *f : files) {
        if (f == nullptr)
            continue;
        fprintf(f, "Memory per rank [MB] (%d rank%s)\n", size, size == 1 ? "" : "s");
        fprintf(f, "  collision values: exact, values of lnLambdaEE/EI, nuS, nuD, nuPar on the p1/p2 flux grids\n");
        fprintf(f, "  process RSS:      whole process, current (/proc/self/statm) and peak (getrusage)\n");
        fprintf(f, "%6s  %14s  %14s  %14s  %12s  %12s\n",
            "rank", "coll. hottail", "coll. runaway", "coll. total", "RSS", "peak RSS");
        real_t sumHot = 0, sumRe = 0;
        for (int r = 0; r < size; r++) {
            const real_t *v = &all[4*r];
            fprintf(f, "%6d  %14.3f  %14.3f  %14.3f  %12.1f  %12.1f\n",
                r, v[0]/MB, v[1]/MB, (v[0]+v[1])/MB, v[2]/MB, v[3]/MB);
            sumHot += v[0];
            sumRe  += v[1];
        }
        fprintf(f, "%6s  %14.3f  %14.3f  %14.3f\n", "sum", sumHot/MB, sumRe/MB, (sumHot+sumRe)/MB);
        fprintf(f, "%6s  %14.3f  %14.3f  %14.3f   (all radii on one rank)\n",
            "serial", hotAll/MB, reAll/MB, (hotAll+reAll)/MB);
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

    char dir[PETSC_MAX_PATH_LEN] = ".";
    PetscOptionsGetString(NULL, NULL, "-cq_dir", dir, sizeof(dir), NULL);
    const string prefix = string(dir) + "/cq_np" + to_string(size) + "_";

    int exit_code = 0;
    Simulation *sim = nullptr;
    try {
        // Same construction as dreami
        Settings *settings = SimulationGenerator::CreateSettings();
        SettingsSFile::LoadSettings(settings, argv[1]);
        sim = SimulationGenerator::ProcessSettings(settings);

        EquationSystem *eqsys = sim->GetEquationSystem();
        Solver *solver = eqsys->GetSolver();
        if (solver->GetMatrix() == nullptr && rank == 0)
            cout << "The solver has no matrix: the collision quantities "
                    "keep all radii on every rank." << endl;

        // First rebuild of the collision quantities, as in
        // Solver::RebuildTerms*()
        solver->SetCollisionQuantityLocalRange();
        if (eqsys->GetHotTailCollisionHandler() != nullptr)
            eqsys->GetHotTailCollisionHandler()->Rebuild();
        if (eqsys->GetRunawayCollisionHandler() != nullptr)
            eqsys->GetRunawayCollisionHandler()->Rebuild();

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
