#ifndef _DREAM_FVM_EQUATION_LINEAR_TRANSIENT_TERM_HPP
#define _DREAM_FVM_EQUATION_LINEAR_TRANSIENT_TERM_HPP

#include <limits>
#include "FVM/config.h"
#include "FVM/Equation/DiagonalTerm.hpp"
#include "FVM/Grid/Grid.hpp"
#include "FVM/Matrix.hpp"


namespace DREAM::FVM {
    class LinearTransientTerm : public DiagonalTerm {
    private:
        real_t dt;

        // ID of differentiated quantity
        len_t unknownId;
        // Differentiated quantity at the previous time step: the 'nloc'
        // elements owned by this rank, the first of which is element
        // 'ifirst' of the quantity (so xn[k] is element ifirst+k)
        real_t *xn;
        len_t nloc = 0, ifirst = 0;

        // Matrix rows owned by this rank, and the first matrix row of the
        // differentiated quantity. The defaults select the whole quantity.
        PetscInt rstart = 0, rend = std::numeric_limits<PetscInt>::max(), rowOffset = 0;
    protected:
        // The part of the differentiated quantity owned by this rank (valid
        // once 'Rebuild()' has been called)
        len_t GetFirstLocalElement() const { return this->ifirst; }
        len_t GetNumberOfLocalElements() const { return this->nloc; }

        virtual bool TermDependsOnUnknowns() override {return false;}
        virtual bool AddWeightsJacobian(const len_t, const len_t, Matrix*, const real_t*) override {return false;}
    public:
        LinearTransientTerm(Grid*, const len_t);
        
        void SetLocalRange(const PetscInt rs, const PetscInt re, const PetscInt off)
            { rstart = rs; rend = re; rowOffset = off; }

        virtual void Rebuild(const real_t, const real_t dt, UnknownQuantityHandler *uqty) override;
        virtual void SetMatrixElements(Matrix*, real_t*) override;
        virtual void SetVectorElements(real_t*, const real_t*) override;
    };
}

#endif/*_DREAM_FVM_EQUATION_LINEAR_TRANSIENT_TERM_HPP*/
