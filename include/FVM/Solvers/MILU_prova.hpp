#ifndef _DREAM_FVM_MATRIX_INVERTER_LU_PROVA_HPP
#define _DREAM_FVM_MATRIX_INVERTER_LU_PROVA_HPP

#include <petscksp.h>
#include "FVM/config.h"
#include "FVM/MatrixInverter.hpp"
#include <cstring>
namespace DREAM::FVM
{
    /**
     * Iterative solver for the DREAM equation system.
     *
     * The system is solved by flexible GMRES, preconditioned by one of three
     * configurations selected at runtime through -dream_split:
     *
     *   none         no splitting; -pc_type governs the whole system
     *   kinetic      kinetic block separated from fluid and scalar
     *   populations  f_hot, f_re and fluid separated  (default)
     *
     * The splits exploit the structure of the assembled matrix: the
     * distribution functions occupy the leading rows and carry essentially
     * the whole system, whereas the fluid and scalar quantities are moments
     * and contribute O(N_r) rows each. Separating them keeps the dense moment
     * rows, which couple one fluid unknown to an entire kinetic block, out of
     * the preconditioner applied to the kinetic blocks.
     */
    class MILU_PROVA : public MatrixInverter
    {
    private:
        len_t xn;   // total number of rows in the assembled system
        len_t Nhot; // rows belonging to f_hot, at offset 0
        len_t Nre;  // rows belonging to f_re,  at offset Nhot

        // The preconditioner is built once and reused: the matrix entries
        // change every timestep but its sparsity pattern does not, and
        // re-issuing PCFieldSplitSetIS on a configured preconditioner appends
        // splits rather than replacing them.
        bool configured = false;

        void ConfigureSolver();
        void ConfigureOuter();
        void ConfigureMonolithic();
        void ConfigureSplitKinetic();
        void ConfigureSplitPopulations();

    public:
        MILU_PROVA(const len_t n, len_t Nhot, len_t Nre);
        ~MILU_PROVA();

        virtual void Invert(Matrix *, Vec *, Vec *) override;

        static void ApplyILUPreconditioning(
            Mat A_seq, Vec b_seq, Mat *MA_seq, Vec *Mb_seq,
            PetscReal dropTol = 1e-8);

        static double Solve_GMRES(
            Mat A, Vec b, Vec x, int step, int my_rank,
            PetscLogStage gmresSetupStage, PetscLogStage solveStage);
    };
}

#endif /*_DREAM_FVM_MATRIX_INVERTER_LU_PROVA_HPP*/