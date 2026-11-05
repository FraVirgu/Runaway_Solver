/**
 * Process settings and build a Simulation object.
 */

#include <iostream>
#include <string>
#include <petscsys.h>
#include "DREAM/ADAS.hpp"
#include "DREAM/AMJUEL.hpp"
#include "DREAM/EquationSystem.hpp"
#include "DREAM/IO.hpp"
#include "DREAM/NIST.hpp"
#include "DREAM/OutputGeneratorSFile.hpp"
#include "DREAM/Settings/Settings.hpp"
#include "DREAM/Settings/SimulationGenerator.hpp"
#include "DREAM/Settings/ProcessMemory.hpp"
#include "FVM/Grid/Grid.hpp"
#include "FVM/Grid/RadialGrid.hpp"
#include "FVM/Interpolator3D.hpp"

using namespace DREAM;

/**
 * Free the initial distribution functions given in the settings
 * ('eqsys/f_hot/init/...' and 'eqsys/f_re/init/...'): they are only used
 * to set the initial values of f_hot and f_re when the equation system is
 * constructed, but are given on their own (r, p, xi) grid, which can be
 * much larger than the simulation grid, and would otherwise stay in
 * memory on every rank for the whole run.
 *
 * The settings are emptied and marked unused, so that they are not saved
 * to the output either (its 'settings' group then has no initial
 * distribution). Run with -dream_keep_init_settings to keep them.
 */
static void FreeInitialDistributionSettings(Settings *s)
{
    PetscBool keep = PETSC_FALSE;
    PetscOptionsHasName(NULL, NULL, "-dream_keep_init_settings", &keep);
    if (keep)
        return;

    for (auto const& [name, set] : s->GetSettings()) {
        if (name.find("f_hot/init/") == std::string::npos &&
            name.find("f_re/init/") == std::string::npos)
            continue;

        if (set->type == Settings::SETTING_TYPE_REAL_ARRAY)
            delete [] (real_t*)set->value;
        else if (set->type == Settings::SETTING_TYPE_INT_ARRAY)
            delete [] (int_t*)set->value;
        else
            continue;

        set->value = nullptr;
        for (len_t i = 0; i < set->ndims; i++)
            set->dims[i] = 0;
        set->used = false;
    }
}

/**
 * Process the given settings and construct a
 * simulation object.
 *
 * s: Settings specifying how to construct the simulation.
 */
Simulation *SimulationGenerator::ProcessSettings(Settings *s)
{
    const real_t t0 = 0;
    // Construct grids
    enum OptionConstants::momentumgrid_type ht_type, re_type;
    fprintf(stderr, "[process] start\n");
    RecordProcessMemory("process: start");
    FVM::Grid *scalarGrid = ConstructScalarGrid();
    fprintf(stderr, "[process] scalar grid\n");
    RecordProcessMemory("process: scalar grid");
    FVM::Grid *fluidGrid = ConstructRadialGrid(s);
    fprintf(stderr, "[process] radial grid\n");
    RecordProcessMemory("process: radial grid");
    FVM::Grid *hottailGrid = ConstructHotTailGrid(s, fluidGrid->GetRadialGrid(), &ht_type);
    fprintf(stderr, "[process] hot-tail grid\n");
    RecordProcessMemory("process: hot-tail grid");

    scalarGrid->Rebuild(t0);
    fluidGrid->Rebuild(t0);
    if (hottailGrid)
        hottailGrid->Rebuild(t0);
    fprintf(stderr, "[process] grids rebuilt\n");
    RecordProcessMemory("process: grids rebuilt");

    // The runaway grid depends on the hot-tail grid (if it exists)
    FVM::Grid *runawayGrid = ConstructRunawayGrid(s, fluidGrid->GetRadialGrid(), hottailGrid, &re_type);
    fprintf(stderr, "[process] runaway grid\n");
    RecordProcessMemory("process: runaway grid");
    if (runawayGrid)
        runawayGrid->Rebuild(t0);
    fprintf(stderr, "[process] runaway grid rebuilt\n");
    RecordProcessMemory("process: runaway grid rebuilt");

    // Load ADAS database
    ADAS *adas = LoadADAS(s);
    fprintf(stderr, "[process] ADAS\n");
    RecordProcessMemory("process: ADAS");
    // Load NIST database
    NIST *nist = LoadNIST(s);
    fprintf(stderr, "[process] NIST\n");
    RecordProcessMemory("process: NIST");
    // Load AMJUEL database
    AMJUEL *amjuel = LoadAMJUEL(s);
    fprintf(stderr, "[process] AMJUEL\n");
    RecordProcessMemory("process: AMJUEL");
    // Construct equation system
    EquationSystem *eqsys = ConstructEquationSystem(
        s, scalarGrid, fluidGrid, ht_type, hottailGrid, re_type, runawayGrid,
        adas, nist, amjuel);
    fprintf(stderr, "[process] equation system\n");
    RecordProcessMemory("process: equation system");

    // The initial values are set: the initial distributions are not needed
    FreeInitialDistributionSettings(s);
    RecordProcessMemory("process: initial distributions freed");
    // Set up simulation
    Simulation *sim = new Simulation();
    sim->SetADAS(adas);
    sim->SetNIST(nist);
    sim->SetAMJUEL(amjuel);
    sim->SetEquationSystem(eqsys);

    LoadOutput(s, sim);
    fprintf(stderr, "[process] output\n");
    RecordProcessMemory("process: output");

    IO::Init(sim);

    return sim;
}

/**
 * Convert an 'enum OptionConstants::momentumgrid_type' to
 * an 'enum Interpolator3D::momentumgrid_type'.
 */
enum FVM::Interpolator3D::momentumgrid_type SimulationGenerator::GetInterp3DMomentumGridType(
    enum OptionConstants::momentumgrid_type type)
{
    switch (type)
    {
    case OptionConstants::MOMENTUMGRID_TYPE_PXI:
        return FVM::Interpolator3D::GRID_PXI;
    case OptionConstants::MOMENTUMGRID_TYPE_PPARPPERP:
        return FVM::Interpolator3D::GRID_PPARPPERP;

    default:
        throw SettingsException(
            "%s:%d: Requested to translate unrecognized momentum grid type: %d.",
            __FILE__, __LINE__, type);
    }
}

/**
 * Load ADAS database.
 */
ADAS *SimulationGenerator::LoadADAS(Settings *s)
{
    enum OptionConstants::adas_interp_type intp =
        (enum OptionConstants::adas_interp_type)s->GetInteger("atomic/adas_interpolation");

    switch (intp)
    {
    case OptionConstants::ADAS_INTERP_BILINEAR:
        return new ADAS(gsl_interp2d_bilinear);
    case OptionConstants::ADAS_INTERP_BICUBIC:
        return new ADAS(gsl_interp2d_bicubic);

    default:
        throw SettingsException(
            "%s:%d: Unrecognized GSL interpolation method: %d.",
            __FILE__, __LINE__, intp);
    }
}

/**
 * Load NIST database (total ion binding and ionization energies).
 */
NIST *SimulationGenerator::LoadNIST(Settings *)
{
    return new NIST();
}

/**
 * Load AMJUEL database (ionization/recombination/radiation rates
 * for hydrogen isotopes, including coefficients for Lyman opaque conditions).
 */
AMJUEL *SimulationGenerator::LoadAMJUEL(Settings *)
{
    return new AMJUEL();
}

/**
 * Load output settings.
 */
void SimulationGenerator::LoadOutput(Settings *s, Simulation *sim)
{
    std::string filename = s->GetString("output/filename");

    sim->SetOutputGenerator(new OutputGeneratorSFile(
        sim->GetEquationSystem(), filename));
}
