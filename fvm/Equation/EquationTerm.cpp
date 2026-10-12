/**
 * Implementation of the 'EquationTerm' base class.
 */

#include <algorithm>
#include "FVM/Equation/EquationTerm.hpp"
#include "FVM/Grid/Grid.hpp"

using namespace DREAM::FVM;


/**
 * Constructor.
 */
EquationTerm::EquationTerm(Grid *rg)
    : grid(rg) {
    
    this->AllocateMemory();
}

/**
 * Destructor.
 */
EquationTerm::~EquationTerm() {
    this->DeallocateMemory();
}

/**
 * Allocate memory for cached grid sizes.
 */
void EquationTerm::AllocateMemory() {
    DeallocateMemory();
    this->nr = this->grid->GetNr();

    this->n1 = new len_t[nr];
    this->n2 = new len_t[nr];

    for (len_t i = 0; i < nr; i++) {
        this->n1[i] = this->grid->GetMomentumGrid(i)->GetNp1();
        this->n2[i] = this->grid->GetMomentumGrid(i)->GetNp2();
    }
}

/**
 * De-allocate memory for the cached grid sizes.
 */
void EquationTerm::DeallocateMemory() {
    if (n2 != nullptr)
        delete [] this->n2;

    if (n1 != nullptr)
        delete [] this->n1;
}

/**
 * Function called when any of the grids have been re-built.
 */
bool EquationTerm::GridRebuilt() {
    this->AllocateMemory();

    return true;
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
bool EquationTerm::GetLocalRange(
    const len_t ir, len_t &i1lo, len_t &i1hi, len_t &i2lo, len_t &i2hi
) const {
    // Index in the quantity of the first cell at this radius
    PetscInt first = 0;
    for (len_t k = 0; k < ir; k++)
        first += (PetscInt)(n1[k]*n2[k]);

    // Owned cells at this radius, [lo, hi) in the 'j*np1 + i' ordering
    const PetscInt np1 = (PetscInt)n1[ir];
    const PetscInt lo = std::max<PetscInt>(0, rstart - rowOffset - first);
    const PetscInt hi = std::min<PetscInt>(np1*(PetscInt)n2[ir], rend - rowOffset - first);

    if (lo >= hi) {
        i1lo = i1hi = i2lo = i2hi = 0;
        return false;
    }

    i2lo = (len_t)(lo / np1);
    i2hi = (len_t)((hi-1) / np1) + 1;

    if (i2hi - i2lo == 1) {
        i1lo = (len_t)(lo % np1);
        i1hi = (len_t)((hi-1) % np1) + 1;
    } else {
        i1lo = 0;
        i1hi = (len_t)np1;
    }

    return true;
}

/**
 * Returns true if derivId is in the derivIdsJacobian vector.
 *
 * derivId:    ID of unknown quantity to differentiate w.r.t.
 * nMultiples: OUTPUT. If not 'nullptr', will contain the number
 *             of multiples of the quantity with respect to which
 *             the derivative was taken.
 */
bool EquationTerm::HasJacobianContribution(len_t derivId, len_t *nMultiples) {
    bool hasDerivIdContribution = false;
    for(len_t i_deriv = 0; i_deriv < derivIdsJacobian.size(); i_deriv++){
        if (derivId == derivIdsJacobian[i_deriv]){
            if(nMultiples != nullptr)
                *nMultiples = derivNMultiplesJacobian[i_deriv];
            hasDerivIdContribution = true;
        }
    }
    return hasDerivIdContribution;
}

