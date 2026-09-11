#ifndef _DREAM_FVM_MATRIX_INVERTER_LU_PROVA_HPP
#define _DREAM_FVM_MATRIX_INVERTER_LU_PROVA_HPP

#include <petscksp.h>
#include "FVM/config.h"
#include "FVM/MatrixInverter.hpp"

namespace DREAM::FVM
{
    class MILU_PROVA : public MatrixInverter
    {
    private:
        len_t xn;

    public:
        MILU_PROVA(const len_t);
        ~MILU_PROVA();

        virtual void Invert(Matrix *, Vec *, Vec *) override;
    };
}

#endif /*_DREAM_FVM_MATRIX_INVERTER_LU_HPP*/
