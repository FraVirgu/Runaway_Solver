#ifndef _DREAM_FVM_MATRIX_INVERTER_RADIAL_BJ_HPP
#define _DREAM_FVM_MATRIX_INVERTER_RADIAL_BJ_HPP

#include <petscksp.h>
#include "FVM/config.h"
#include "FVM/MatrixInverter.hpp"

namespace DREAM::FVM {
    /**
     * Matrix inverter using FGMRES preconditioned by block Jacobi over the
     * RADIUS: one block per group of -dream_rbj_radii_per_block radial cells
     * (default 1), holding the f_hot AND the f_re unknowns of those radii.
     * Each block is solved with an inner GMRES + hypre/BoomerAMG; the radial
     * coupling between blocks is left to the outer FGMRES. This is variant
     * 2b of iface/TestKineticHypreRadius.cpp, wired into DREAM.
     *
     * The f_hot/f_re rows of one radius are not contiguous in the assembled
     * matrix, so the system is permuted to radius-major order before every
     * solve. Fluid rows (if any, i.e. without -dream_kinetic_only) form one
     * extra block on the last rank, solved directly with LU.
     */
    class MIRadialBJ : public MatrixInverter {
    private:
        len_t xn;   // total number of rows in the assembled system
        len_t Nhot; // rows belonging to f_hot, at offset 0
        len_t Nre;  // rows belonging to f_re,  at offset Nhot
        len_t Nr;   // number of radial cells

        // -dream_rbj_radii_per_block: radial cells per Jacobi block
        PetscInt radiiPerBlock = 1;
        // radii owned by this rank (0 when each radius spans several ranks)
        PetscInt nLocalRadii = 0;

        bool configured = false;

        // Radius-major permutation: isPerm[k] is the old row of new row k
        IS isPerm = nullptr;
        Mat Ap = nullptr;
        VecScatter scatter = nullptr;
        Vec bP = nullptr, xP = nullptr;
        // Previous (permuted) solution, used as the nonzero initial guess
        Vec xPrevP = nullptr;

        void CreatePermutation();
        void ConfigureSolver();

    public:
        MIRadialBJ(const len_t n, const len_t Nhot, const len_t Nre, const len_t Nr=1);
        ~MIRadialBJ();

        virtual void Invert(Matrix*, Vec*, Vec*) override;
    };
}

#endif/*_DREAM_FVM_MATRIX_INVERTER_RADIAL_BJ_HPP*/
