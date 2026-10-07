/**
 * Implementation of an Euler backward transient term.
 *
 * df   f_{n+1} - f_n
 * -- ~ -------------
 * dt        dt
 *
 */


#ifndef _DREAM_FVM_TRANSIENT_TERM_HPP
#define _DREAM_FVM_TRANSIENT_TERM_HPP

#include "FVM/Equation/LinearTransientTerm.hpp"

namespace DREAM::FVM {
    class TransientTerm : public LinearTransientTerm {
    private:
        real_t scaleFactor;
    protected:
        // Only the elements owned by this rank are set (and used)
        virtual void SetWeights() override {
            const len_t iEnd = GetFirstLocalElement() + GetNumberOfLocalElements();
            for(len_t i = GetFirstLocalElement(); i < iEnd; i++)
                weights[i] = scaleFactor;
        }
    public:
        TransientTerm(Grid* g, const len_t unknownId, real_t scaleFactor = 1.0) 
            : LinearTransientTerm(g,unknownId), scaleFactor(scaleFactor) {SetName("TransientTerm");} 
    };
}

#endif/*_DREAM_FVM_TRANSIENT_TERM_HPP*/
