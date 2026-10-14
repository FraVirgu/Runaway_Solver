/**
 * Process settings and build a Simulation object.
 */

#include <iostream>
#include "DREAM/ADAS.hpp"
#include "DREAM/AMJUEL.hpp"
#include "DREAM/EquationSystem.hpp"
#include "DREAM/IO.hpp"
#include "DREAM/NIST.hpp"
#include "DREAM/OutputGeneratorSFile.hpp"
#include "DREAM/Settings/Settings.hpp"
#include "DREAM/Settings/SimulationGenerator.hpp"
#include "FVM/Grid/Grid.hpp"
#include "FVM/Grid/RadialGrid.hpp"
#include "FVM/Interpolator3D.hpp"

using namespace DREAM;

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
    FVM::Grid *scalarGrid = ConstructScalarGrid();
    fprintf(stderr, "[process] scalar grid\n");
    FVM::Grid *fluidGrid = ConstructRadialGrid(s);
    fprintf(stderr, "[process] radial grid\n");
    FVM::Grid *hottailGrid = ConstructHotTailGrid(s, fluidGrid->GetRadialGrid(), &ht_type);
    fprintf(stderr, "[process] hot-tail grid\n");

    scalarGrid->Rebuild(t0);
    fluidGrid->Rebuild(t0);
    if (hottailGrid)
        hottailGrid->Rebuild(t0);
    fprintf(stderr, "[process] grids rebuilt\n");

    // The runaway grid depends on the hot-tail grid (if it exists)
    FVM::Grid *runawayGrid = ConstructRunawayGrid(s, fluidGrid->GetRadialGrid(), hottailGrid, &re_type);
    fprintf(stderr, "[process] runaway grid\n");
    if (runawayGrid)
        runawayGrid->Rebuild(t0);
    fprintf(stderr, "[process] runaway grid rebuilt\n");

    // Load ADAS database
    ADAS *adas = LoadADAS(s);
    fprintf(stderr, "[process] ADAS\n");
    // Load NIST database
    NIST *nist = LoadNIST(s);
    fprintf(stderr, "[process] NIST\n");
    // Load AMJUEL database
    AMJUEL *amjuel = LoadAMJUEL(s);
    fprintf(stderr, "[process] AMJUEL\n");
    // Construct equation system
    EquationSystem *eqsys = ConstructEquationSystem(
        s, scalarGrid, fluidGrid, ht_type, hottailGrid, re_type, runawayGrid,
        adas, nist, amjuel);
    fprintf(stderr, "[process] equation system\n");
    // Set up simulation
    Simulation *sim = new Simulation();
    sim->SetADAS(adas);
    sim->SetNIST(nist);
    sim->SetAMJUEL(amjuel);
    sim->SetEquationSystem(eqsys);

    LoadOutput(s, sim);
    fprintf(stderr, "[process] output\n");

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
