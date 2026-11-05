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
 * Returns the lines j in [jlo, jhi) of index 'ir' of the flux grid
 * 'fgType' whose coefficients this rank stores: those containing faces of
 * the cells it owns (see 'GetLocalRange()'), as whole lines.
 *
 *   FLUXGRIDTYPE_P1:     lines of the cells at radius ir
 *   FLUXGRIDTYPE_P2:     same, plus the upper xi face of the last line
 *   FLUXGRIDTYPE_RADIAL: face ir is the outer face of the cells at ir-1
 *                        and the inner face of those at ir. The outer
 *                        boundary (ir = nr) is stored whole, since the
 *                        transport boundary conditions read all of it.
 *
 * On a fluid grid (one cell per radius) all lines are stored, since the
 * coefficients of fluid quantities are read at any radius.
 */
void EquationTerm::GetStoredLines(
    fluxGridType fgType, const len_t ir, len_t &jlo, len_t &jhi
) const {
    // XXX here we assume that all momentum grids are the same
    const len_t np2 = n2[std::min(ir, nr-1)] + (fgType == FLUXGRIDTYPE_P2);

    if (grid->GetNCells() == nr) {
        jlo = 0, jhi = np2;
        return;
    }

    len_t i1lo, i1hi, i2lo, i2hi;
    jlo = jhi = 0;
    if (fgType == FLUXGRIDTYPE_RADIAL) {
        jlo = np2;
        for (len_t k = (ir > 0 ? ir-1 : 0); k <= ir && k < nr; k++)
            if (GetLocalRange(k, i1lo, i1hi, i2lo, i2hi))
                jlo = std::min(jlo, i2lo), jhi = std::max(jhi, i2hi);

        if (jlo >= jhi)
            jlo = jhi = 0;
        else if (ir == nr)
            jlo = 0, jhi = np2;
    } else if (GetLocalRange(ir, i1lo, i1hi, i2lo, i2hi)) {
        jlo = i2lo;
        jhi = i2hi + (fgType == FLUXGRIDTYPE_P2);
    }
}

/**
 * Allocate, in one block set to 0, the lines j in [jlo[k], jhi[k]) of
 * 'lineLength[k]' elements of each row k < nRows, and point rows[k] to
 * element (j=0, i=0) of that row, so that rows[k][j*lineLength[k] + i]
 * can be used with the global j on the stored lines (as PETSc does for
 * the local part of a distributed array). rows[k] is nullptr for the rows
 * with no stored line.
 *
 * RETURNS the block, to be deleted by the caller (nullptr if empty).
 */
real_t *EquationTerm::AllocateLines(
    real_t **rows, const len_t nRows, const std::vector<len_t> &lineLength,
    const std::vector<len_t> &jlo, const std::vector<len_t> &jhi
) {
    len_t N = 0;
    for (len_t k = 0; k < nRows; k++)
        N += (jhi[k]-jlo[k]) * lineLength[k];

    real_t *data = (N > 0 ? new real_t[N]() : nullptr);

    len_t off = 0;
    for (len_t k = 0; k < nRows; k++) {
        if (jlo[k] == jhi[k]) {
            rows[k] = nullptr;
            continue;
        }
        rows[k] = data + off - jlo[k]*lineLength[k];
        off += (jhi[k]-jlo[k]) * lineLength[k];
    }

    return data;
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

