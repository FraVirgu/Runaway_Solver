#ifndef _DREAM_EQUATIONS_COLLISION_QUANTITY_HANDLER_HPP
#define _DREAM_EQUATIONS_COLLISION_QUANTITY_HANDLER_HPP

namespace DREAM { class CollisionQuantityHandler; }
#include "DREAM/Equations/RunawayFluid.hpp"
#include "DREAM/Equations/ParallelDiffusionFrequency.hpp"
#include <gsl/gsl_interp2d.h>

namespace DREAM {
    class CollisionQuantityHandler{
    private:
        FVM::UnknownQuantityHandler *unknowns = nullptr;
        IonHandler *ionHandler = nullptr;

        CollisionQuantity::collqty_settings *collQtySettings;

        CoulombLogarithm *lnLambdaEE;
        CoulombLogarithm *lnLambdaEI;
        SlowingDownFrequency *nuS;
        PitchScatterFrequency *nuD;
        ParallelDiffusionFrequency *nuPar;
    public:
        CollisionQuantityHandler(FVM::Grid *g, FVM::UnknownQuantityHandler *u, IonHandler *ih,  
                enum OptionConstants::momentumgrid_type mgtype,  CollisionQuantity::collqty_settings *cqset);
        ~CollisionQuantityHandler();

        void gridRebuilt();

        void Rebuild();

        // Restrict all collision quantities to the matrix rows [rs, re)
        // owned by this rank, where 'off' is the first matrix row of the
        // distribution function on this grid. The quantities read each
        // other's values, so they must all have the same range.
        void SetLocalRange(const PetscInt rs, const PetscInt re, const PetscInt off) {
            lnLambdaEE->SetLocalRange(rs, re, off);
            lnLambdaEI->SetLocalRange(rs, re, off);
            nuS->SetLocalRange(rs, re, off);
            nuD->SetLocalRange(rs, re, off);
            nuPar->SetLocalRange(rs, re, off);
        }

        SlowingDownFrequency* GetNuS(){return nuS;}
        PitchScatterFrequency* GetNuD(){return nuD;}
        ParallelDiffusionFrequency* GetNuPar(){return nuPar;}
        CoulombLogarithm* GetLnLambdaEE(){return lnLambdaEE;}
        CoulombLogarithm* GetLnLambdaEI(){return lnLambdaEI;}

        CollisionQuantity::collqty_settings* GetCollisionQuantitySettings()
            {return collQtySettings;}
    };

}


#endif/*_DREAM_EQUATIONS_COLLISION_QUANTITY_HANDLER_HPP*/

    


