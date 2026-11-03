/**
 * Implementation of a class from which the Coulomb logarithms and collision frequencies are derived.
 */

#include <algorithm>
#include "DREAM/Equations/CollisionQuantity.hpp"
//#include "DREAM/Constants.hpp"
//#include "DREAM/Settings/OptionConstants.hpp"
//#include "FVM/UnknownQuantityHandler.hpp"
#include "DREAM/NotImplementedException.hpp"

using namespace DREAM;

/**
 * Constructor.
 */
CollisionQuantity::CollisionQuantity(FVM::Grid *g, FVM::UnknownQuantityHandler *u, IonHandler *ih,  
                enum OptionConstants::momentumgrid_type mgtype,  struct collqty_settings *cqset){
    mg = g->GetMomentumGrid(0);
    rGrid = g->GetRadialGrid();

    collQtySettings = cqset;
    ionHandler = ih;
    unknowns = u;

    // Various settings that appear in many places in the calculations
    isPXiGrid = (mgtype==OptionConstants::MOMENTUMGRID_TYPE_PXI);
    isPartiallyScreened = (collQtySettings->collfreq_type==OptionConstants::COLLQTY_COLLISION_FREQUENCY_TYPE_PARTIALLY_SCREENED);
    isNonScreened = (collQtySettings->collfreq_type==OptionConstants::COLLQTY_COLLISION_FREQUENCY_TYPE_NON_SCREENED);
    isNonlinear = (collQtySettings->nonlinear_mode == OptionConstants::EQTERM_NONLINEAR_MODE_NON_REL_ISOTROPIC);
    isBrems = (collQtySettings->bremsstrahlung_mode != OptionConstants::EQTERM_BREMSSTRAHLUNG_MODE_NEGLECT);
    // ID of quantities that contribute to collision frequencies
    id_ncold = unknowns->GetUnknownID(OptionConstants::UQTY_N_COLD);
    id_ni    = unknowns->GetUnknownID(OptionConstants::UQTY_ION_SPECIES);
    id_Tcold = unknowns->GetUnknownID(OptionConstants::UQTY_T_COLD);

    /**
     * Set buildOnlyF1F2=false if quantities need to be evaluated on the distribution 
     * and radial flux grids. For now hardcoded to true because it isn't expected to 
     * be needed. In fact the calculation on the radial flux grid is not supported,
     * since we do not yet interpolate in the unknown quantities.
     */
    buildOnlyF1F2 = true;

    /**
     * This is the ad-hoc k-parameter appearing in Linneas paper that sets the transition
     * from superthermal limits to p->0. For now hardcoded to the recommended value k=5.
     */
    kInterpolate = 5;
}

/**
 * Destructor.
 */
CollisionQuantity::~CollisionQuantity(){
    DeallocateCollisionQuantities();
}

/**
 * Rebuilds collision quantities; the quantities are split into 
 * multiple partial contributions, and only those that change are
 * rebuilt. 
 */
void CollisionQuantity::Rebuild(){
    if (gridRebuilt){ // Reallocate and rebuild everything
        nr  = rGrid->GetNr();
        nZ  = ionHandler->GetNZ();
        nzs = ionHandler->GetNzs();
        np1 = mg->GetNp1();
        np2 = mg->GetNp2();
        if(isPXiGrid)
            np2_store = 1;
        else
            np2_store = mg->GetNp2();
        SetLocalRadii();
        // We should Deallocate before we update nr, nZ etc. so that deletions occurs with parameters of previous iteration
        AllocateCollisionQuantities(); 
        
        RebuildConstantTerms();
        RebuildPlasmaDependentTerms();
        AssembleQuantity();
        gridRebuilt = false;
    } else if (parametersHaveChanged()){
        RebuildPlasmaDependentTerms();
        AssembleQuantity();
    }    
}


/** 
 * Assembles the collision frequency from the various partial contributions, on all grids.
 */
void CollisionQuantity::AssembleQuantity(){
    if(!buildOnlyF1F2){
        AssembleQuantity(collisionQuantity,nr,np1,np2,FVM::FLUXGRIDTYPE_DISTRIBUTION);
        AssembleQuantity(collisionQuantity_fr,nr /* +1 this case should be treated properly somehow*/,np1,np2,FVM::FLUXGRIDTYPE_RADIAL);
    }
    AssembleQuantity(collisionQuantity_f1,nr,np1+1,np2,FVM::FLUXGRIDTYPE_P1);
    AssembleQuantity(collisionQuantity_f2,nr,np1,np2+1,FVM::FLUXGRIDTYPE_P2);
}


/** 
 * Returns true if any unknown quantities that affect collision quantities have changed. 
 */
bool CollisionQuantity::parametersHaveChanged(){
    return unknowns->HasChanged(id_ncold) || unknowns->HasChanged(id_Tcold) || unknowns->HasChanged(id_ni);
}


/**
 * Evaluate quantity at p (detailed calculation 
 * implemented in derived classes)
 */
real_t CollisionQuantity::evaluateAtP(len_t ir, real_t p){
    return evaluateAtP(ir, p, collQtySettings);
}
/**
 * Evaluate derivative of quantity at p w.r.t. unknown derivId 
 * (detailed calculation implemented in derived classes)
 */
real_t CollisionQuantity::evaluatePartialAtP(len_t ir, real_t p, len_t derivId, len_t n){
    return evaluatePartialAtP(ir, p, derivId, n, collQtySettings);
}


/**
 * Allocate quantities
 */
void CollisionQuantity::AllocateCollisionQuantities(){
    DeallocateCollisionQuantities();

    if(!buildOnlyF1F2){
        AllocateCollisionQuantity(collisionQuantity,nr,np1,np2);
        AllocateCollisionQuantity(collisionQuantity_fr,nr+1,np1,np2);
    }
    AllocateCollisionQuantity(collisionQuantity_f1,nr,np1+1,np2);
    AllocateCollisionQuantity(collisionQuantity_f2,nr,np1,np2+1);


    AllocatePartialQuantities();
}


/**
 * Allocate quantity on one grid. The pointer table spans all 'nr' radii,
 * so that the quantity is indexed with the global radial index, but only
 * the radii [irlo, irhi) owned by this rank are allocated (the others
 * are nullptr).
 *
 * NOTE: on the radial flux grid (nr+1 radii) the upper face 'irhi' of
 * the owned cells is not allocated. It is not needed as long as
 * buildOnlyF1F2 = true.
 */
void CollisionQuantity::AllocateCollisionQuantity(real_t **&collisionQuantity, len_t nr, len_t np1, len_t np2){
    collisionQuantity = new real_t*[nr];
    for(len_t ir = 0; ir<nr; ir++)
        collisionQuantity[ir] = (ir>=irlo && ir<irhi) ? new real_t[np1*np2] : nullptr;
}


/**
 * Set the range of radii [irlo, irhi) containing matrix rows owned by
 * this rank (all radii unless 'SetLocalRange()' has been called). Both
 * are 0 if the rank owns no row of the quantity.
 */
void CollisionQuantity::SetLocalRadii(){
    irlo = irhi = 0;

    len_t i1lo, i1hi, i2lo, i2hi;
    bool found = false;
    for(len_t ir = 0; ir<nr; ir++){
        if(GetLocalRange(ir, i1lo, i1hi, i2lo, i2hi)){
            if(!found){
                irlo = ir;
                found = true;
            }
            irhi = ir+1;
        }
    }
}


/**
 * Translate the matrix rows owned by this rank (as given to
 * 'SetLocalRange()') into the cells they cover at radius 'ir'.
 *
 * The cells of a radius are ordered as 'j*np1 + i' (i = index in the
 * first momentum coordinate, j = index in the second) and the radii
 * follow each other, so a rank owns a contiguous run of them which may
 * start and end in the middle of a line of constant j. The range
 * returned is the smallest box [i1lo,i1hi) x [i2lo,i2hi) containing
 * that run: all of np1 when the run spans several j, only the owned
 * part of the line when it stays within a single j.
 *
 * Unlike 'FVM::EquationTerm::GetLocalRange()', the momentum grid is
 * assumed to be the same at all radii (see 'mg'), so radius 'ir' starts
 * at cell ir*np1*np2.
 *
 * To cover the faces of these cells on a flux grid, extend the
 * corresponding upper bound by one (i < i1hi+1 or j < i2hi+1).
 *
 * ir:         Radial index.
 * i1lo, i1hi: On return, range of the first momentum index.
 * i2lo, i2hi: On return, range of the second momentum index.
 *
 * RETURNS false if this rank owns no cell at this radius (the ranges
 * are then empty).
 */
bool CollisionQuantity::GetLocalRange(
    const len_t ir, len_t &i1lo, len_t &i1hi, len_t &i2lo, len_t &i2hi
) const {
    const PetscInt n1 = (PetscInt)np1;
    const PetscInt ncells = n1*(PetscInt)np2;

    // Index in the quantity of the first cell at this radius
    const PetscInt first = (PetscInt)ir * ncells;

    // Owned cells at this radius, [lo, hi) in the 'j*np1 + i' ordering
    const PetscInt lo = std::max<PetscInt>(0, rstart - rowOffset - first);
    const PetscInt hi = std::min<PetscInt>(ncells, rend - rowOffset - first);

    if (lo >= hi) {
        i1lo = i1hi = i2lo = i2hi = 0;
        return false;
    }

    i2lo = (len_t)(lo / n1);
    i2hi = (len_t)((hi-1) / n1) + 1;

    if (i2hi - i2lo == 1) {
        i1lo = (len_t)(lo % n1);
        i1hi = (len_t)((hi-1) % n1) + 1;
    } else {
        i1lo = 0;
        i1hi = np1;
    }

    return true;
}


/**
 * Deallocate quantity on one grid
 */
void CollisionQuantity::DeallocateCollisionQuantity(real_t **&collisionQuantity, len_t nr){
    if(collisionQuantity != nullptr){
        for(len_t ir = 0; ir<nr; ir++)
            delete [] collisionQuantity[ir];
        delete [] collisionQuantity;
    }
}


/**
 * Deallocate quantities
 */
void CollisionQuantity::DeallocateCollisionQuantities(){
    if(!buildOnlyF1F2){
        DeallocateCollisionQuantity(collisionQuantity,nr);
        DeallocateCollisionQuantity(collisionQuantity_fr,nr+1);
    }
    DeallocateCollisionQuantity(collisionQuantity_f1,nr);
    DeallocateCollisionQuantity(collisionQuantity_f2,nr);    

    this->collisionQuantity = nullptr;
    this->collisionQuantity_f1 = nullptr;
    this->collisionQuantity_f2 = nullptr;
    this->collisionQuantity_fr = nullptr;
}
