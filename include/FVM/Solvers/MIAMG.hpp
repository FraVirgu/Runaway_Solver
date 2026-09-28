#ifndef _DREAM_FVM_MATRIX_INVERTER_AMG_HPP
#define _DREAM_FVM_MATRIX_INVERTER_AMG_HPP

#include <petscksp.h>
#include "FVM/config.h"
#include "FVM/MatrixInverter.hpp"

namespace DREAM::FVM {
    /**
     * Matrix inverter using GMRES preconditioned by hypre's BoomerAMG,
     * applied to the whole system with no field splitting -- isolates the
     * AMG preconditioner path on its own, separate from MILU_PROVA's
     * fieldsplit/ILU experiments.
     */
    class MIAMG : public MatrixInverter {
    private:
        len_t xn;

        bool configured = false;

        // -dream_amg_lag_pc: number of Invert() calls that share one
        // PCSetUp before the AMG hierarchy is rebuilt. 1 (default) rebuilds
        // on every call; see the comment on ConfigureSolver().
        PetscInt lagPC = 1;
        PetscInt callsSinceSetup = 0;

        void ConfigureSolver();

    public:
        MIAMG(const len_t n);
        ~MIAMG();

        virtual void Invert(Matrix*, Vec*, Vec*) override;
    };
}

#endif/*_DREAM_FVM_MATRIX_INVERTER_AMG_HPP*/
