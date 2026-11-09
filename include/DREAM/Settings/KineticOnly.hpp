#ifndef _DREAM_SETTINGS_KINETIC_ONLY_HPP
#define _DREAM_SETTINGS_KINETIC_ONLY_HPP

/**
 * Helpers for the kinetic-only mode (PETSc option -dream_kinetic_only).
 *
 * In this mode the solver matrix contains the kinetic unknowns f_hot and
 * f_re only. The other fluid and scalar unknowns (n_hot, n_re, n_tot, j_hot,
 * j_re, j_ohm, j_tot, I_p, n_i, ...) keep their real DREAM equation, which
 * is used only to initialise them from the input; they are left out of the
 * matrix (EquationSystem::ProcessSystem()) and keep that initial value.
 * psi_p, psi_edge and S_particle, whose DREAM initialisation is not
 * available here, are given the constant 0 instead. See
 * SimulationGenerator::ConstructEquations() and
 * SolverLinearlyImplicit::initialize_kinetic_only().
 */

#include <string>
#include <petscsys.h>
#include "DREAM/EquationSystem.hpp"
#include "DREAM/EqsysInitializer.hpp"
#include "FVM/Equation/ConstantParameter.hpp"
#include "FVM/Equation/PrescribedParameter.hpp"
#include "FVM/Equation/Operator.hpp"
#include "FVM/Grid/Grid.hpp"

namespace DREAM
{
    /**
     * Returns true if -dream_kinetic_only was given.
     */
    inline bool KineticOnlyMode()
    {
        PetscBool set = PETSC_FALSE;
        PetscOptionsHasName(NULL, NULL, "-dream_kinetic_only", &set);
        return set == PETSC_TRUE;
    }

    /**
     * Give the unknown 'id', which lives on the grid 'grid', the constant
     * value 'value' (the same in every cell) instead of an equation.
     *
     * The equation consists of a single predetermined (constant) term, so
     * the unknown is "trivial": it does not enter the solver matrices. It
     * is initialised by evaluating that equation.
     */
    inline void SetConstantEquation(
        EquationSystem *eqsys, const len_t id, FVM::Grid *grid,
        const real_t value, const std::string &desc = "Constant")
    {
        FVM::Operator *op = new FVM::Operator(grid);
        op->AddTerm(new FVM::ConstantParameter(grid, value));
        eqsys->SetOperator(id, id, op, desc);

        eqsys->initializer->AddRule(
            id, EqsysInitializer::INITRULE_EVAL_EQUATION);
    }

    /**
     * Same as above, but with a value for each cell of the grid.
     */
    inline void SetConstantEquation(
        EquationSystem *eqsys, const len_t id, FVM::Grid *grid,
        const real_t *values, const std::string &desc = "Constant")
    {
        FVM::PrescribedParameter *pp = new FVM::PrescribedParameter(grid);
        real_t t0 = 0;
        pp->SetData(1, &t0, const_cast<real_t *>(values), true);

        FVM::Operator *op = new FVM::Operator(grid);
        op->AddTerm(pp);
        eqsys->SetOperator(id, id, op, desc);

        eqsys->initializer->AddRule(
            id, EqsysInitializer::INITRULE_EVAL_EQUATION);
    }
}

#endif /*_DREAM_SETTINGS_KINETIC_ONLY_HPP*/
