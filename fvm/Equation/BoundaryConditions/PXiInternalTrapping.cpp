/**
 * This boundary condition ensures that the trapped region is handled correctly
 * in kinetic simulations using toroidal geometry. The B.C. resets the equation
 * at f(xi0) corresponding -xi_T <= xi0 < 0 so that this region is exactly
 * governed by the equation f(-xi0) = f(xi0) at 0 < -xi0 <= xi_T.
 * NOTE: Only the diagonal is reset, and the implementations of individual
 *       other equation terms must make sure that other elements are not set
 *
 * Distributed version (Runaway_Solver). The rows of the matrix are split
 * over the ranks in contiguous ranges, which in general cut a radius, so
 * that the rows of the line jp (just inside +xi_T) and the face jm (at
 * -xi_T), whose flux they receive, are on different ranks. Each rank only
 * stores the coefficients of the cells it owns, so the coefficients on
 * that face (and on the negative trapped radial faces) are read from
 * AdvectionDiffusionTerm::GetTrapping*(), where they are collected on every
 * rank by GatherTrappingBoundaryCoefficients() after each rebuild.
 *
 *   matrix (and its rhs): only the rows owned by this rank are set. The rhs
 *                         passed by the solver only holds those rows.
 *   function vector:      all rows are set, as the vector is complete on
 *                         every rank and so are the gathered coefficients.
 */

#include <cmath>
#include <functional>
#include <vector>
#include "FVM/config.h"
#include "FVM/Equation/AdvectionDiffusionTerm.hpp"
#include "FVM/Equation/BoundaryConditions/PXiInternalTrapping.hpp"
#include "FVM/FVMException.hpp"


using namespace DREAM::FVM;
using namespace DREAM::FVM::BC;
using namespace std;


/**
 * Rows [lo, hi) of the unknown currently selected in 'mat' (indices
 * relative to the unknown, as used by the boundary condition) owned by
 * this rank.
 */
static void OwnedRows(Matrix *mat, PetscInt &lo, PetscInt &hi) {
    PetscInt rstart, nloc;
    mat->GetOwnershipRange(&rstart, &nloc);
    const PetscInt off = mat->GetRowOffset();
    lo = rstart - off;
    hi = rstart + nloc - off;
}


/**
 * Constructor.
 */
PXiInternalTrapping::PXiInternalTrapping(
    Grid *g, Operator *oprtr
) : PXiAdvectionDiffusionBoundaryCondition(g, oprtr) {

    SetName("PXiInternalTrapping");

    this->LocateTrappedRegion();
}

/**
 * Destructor.
 */
PXiInternalTrapping::~PXiInternalTrapping() {
    this->DeallocateTrappedIndices();
}

/**
 * Deallocate memory for the 'trappedNegXi_indices' array.
 */
void PXiInternalTrapping::DeallocateTrappedIndices() {
    if (this->trappedNegXi_indices == nullptr)
        return;

    len_t nr = this->grid->GetNr();
    for (len_t ir = 0; ir < nr; ir++) {
        delete [] trappedNegXi_indices[ir];
		delete [] trappedNegXiRadial_indices[ir];

		delete [] trappedPosXi_indices[ir];
		delete [] trappedPosXiRadial_indices[ir];
	}

    delete [] trappedNegXi_indices;
	delete [] trappedPosXi_indices;
	delete [] trappedNegXiRadial_indices;
	delete [] trappedPosXiRadial_indices;
    delete [] nTrappedNegXi_indices;
	delete [] nTrappedNegXiRadial_indices;

    delete [] rowsToReset;

    this->trappedNegXi_indices = nullptr;
	this->trappedPosXi_indices = nullptr;
	this->trappedNegXiRadial_indices = nullptr;
	this->trappedPosXiRadial_indices = nullptr;
    this->nTrappedNegXi_indices = nullptr;
    this->nTrappedNegXiRadial_indices = nullptr;
    this->rowsToReset = nullptr;
}

/**
 * Method which is called after the grid has been rebuilt.
 */
bool PXiInternalTrapping::GridRebuilt() {
    this->PXiAdvectionDiffusionBoundaryCondition::GridRebuilt();
    this->LocateTrappedRegion();
    return true;
}

/**
 * Construct the array 'trappedNegXi_indices', which contains indices to the
 * points of xi0 which are located inside the trapped region, and which have
 * xi0 < 0 (i.e. those points which should be copied from the points with
 * xi0 > 0).
 *
 * The indices are collected as they are found (in increasing order), rather
 * than assumed contiguous from the first one, so that the radial list is
 * exactly AdvectionDiffusionTerm::GetTrappingKm(ir) (same predicate, same
 * order) and entry q of one is entry q of the other.
 */
void PXiInternalTrapping::LocateTrappedRegion() {
    if (this->trappedNegXi_indices != nullptr)
        this->DeallocateTrappedIndices();

    len_t nr = this->grid->GetNr();
    this->trappedNegXi_indices  = new PetscInt*[nr];
    this->trappedPosXi_indices  = new PetscInt*[nr];
    this->nTrappedNegXi_indices = new PetscInt[nr];
    this->trappedNegXiRadial_indices  = new PetscInt*[nr];
    this->trappedPosXiRadial_indices  = new PetscInt*[nr];
    this->nTrappedNegXiRadial_indices = new PetscInt[nr];

    this->nRowsToReset = 0;
    for (len_t ir = 0; ir < nr; ir++) {
        auto mg = this->grid->GetMomentumGrid(ir);
        const real_t *xi0   = mg->GetP2();
        const real_t *xi0_f = mg->GetP2_f();
        const real_t dxi0   = mg->GetDp2(0);
        const PetscInt nXi  = mg->GetNp2();
        const PetscInt nP   = mg->GetNp1();

        // Find the closest mirrored xi0 (i.e. -xi0) on the grid...
        // (allow for both monotonic increase and decrease in xi0)
        auto mirror = [&](const PetscInt jn) -> PetscInt {
            const real_t negXI0 = -xi0[jn];
            for (PetscInt j = 0; j < nXi; j++)
                if ((dxi0 > 0 && xi0_f[j+1] > negXI0) || (dxi0 < 0 && xi0_f[j+1] < negXI0))
                    return j;
            return nXi-1;
        };

        vector<PetscInt> negXi, negXiRadial;
        for (PetscInt j = 0; j < nXi; j++){
            if (this->grid->IsNegativePitchTrappedIgnorableCell(ir,j))
                negXi.push_back(j);
            if (this->grid->IsNegativePitchTrappedIgnorableRadialFluxCell(ir,j))
                negXiRadial.push_back(j);
        }

        // XI INDICES
        this->nTrappedNegXi_indices[ir] = (PetscInt)negXi.size();
        this->trappedNegXi_indices[ir] = new PetscInt[negXi.size()];
        this->trappedPosXi_indices[ir] = new PetscInt[negXi.size()];
        for (len_t i = 0; i < negXi.size(); i++) {
            this->trappedNegXi_indices[ir][i] = negXi[i];
            this->trappedPosXi_indices[ir][i] = mirror(negXi[i]);
        }
        this->nRowsToReset += nP*negXi.size();

        // R INDICES
        this->nTrappedNegXiRadial_indices[ir] = (PetscInt)negXiRadial.size();
        this->trappedNegXiRadial_indices[ir] = new PetscInt[negXiRadial.size()];
        this->trappedPosXiRadial_indices[ir] = new PetscInt[negXiRadial.size()];
        for (len_t i = 0; i < negXiRadial.size(); i++) {
            this->trappedNegXiRadial_indices[ir][i] = negXiRadial[i];
            this->trappedPosXiRadial_indices[ir][i] = mirror(negXiRadial[i]);
        }
    }

    // create large index vector for rows to reset
    this->rowsToReset = new PetscInt[nRowsToReset];
    len_t offset=0;
    for (len_t ir = 0, it = 0; ir < nr; ir++) {
        len_t nP = grid->GetNp1(ir);
        len_t nXi = grid->GetNp2(ir);
        for(PetscInt j=0; j<nTrappedNegXi_indices[ir]; j++){
            len_t indXi = offset + nP*trappedNegXi_indices[ir][j];
            for (len_t i = 0; i < nP; i++, it++)
                rowsToReset[it] = indXi + i;
        }
        offset += nP*nXi;
    }
}

/**
 * Rebuild this boundary condition (not used).
 */
bool PXiInternalTrapping::Rebuild(const real_t, UnknownQuantityHandler*) { return false; }

/**
 * Add elements to the Jacobian.
 */
bool PXiInternalTrapping::AddToJacobianBlock(
    const len_t uqtyId, const len_t derivId, Matrix *jac, const real_t *x
) {
    bool contributes = false;
    if (uqtyId == derivId){
        interp_mode = AdvectionInterpolationCoefficient::AD_INTERP_MODE_JACOBIAN;
        this->AddToMatrixElements(jac, nullptr);
        contributes = true;
    }
    // Handle derivatives of coefficients (we assume that the coefficients
    // do not depend on the distribution functions...)
    // NOTE: the derivatives of the coefficients are not gathered (see
    // _addElements()), and the base class sets all rows of 'jac'.
    else
        contributes |=
            this->PXiAdvectionDiffusionBoundaryCondition::AddPartialJacobianContributions(
                uqtyId, derivId, jac, x, true
            );

    return contributes;
}

/**
 * Add elements to the linear operator matrix (only the rows owned by this
 * rank).
 */
void PXiInternalTrapping::AddToMatrixElements(Matrix *mat, real_t*) {
    const real_t *const* Ar  = this->oprtr->GetAdvectionCoeffR();
    const real_t *const* Ax  = this->oprtr->GetAdvectionCoeff2();
    const real_t *const* Drr = this->oprtr->GetDiffusionCoeffRR();
    const real_t *const* Dxp = this->oprtr->GetDiffusionCoeff21();
    const real_t *const* Dxx = this->oprtr->GetDiffusionCoeff22();

    PetscInt lo, hi;
    OwnedRows(mat, lo, hi);

    this->_addElements([mat,lo,hi](const len_t I, const len_t J, const real_t V) {
        if ((PetscInt)I >= lo && (PetscInt)I < hi)
            mat->SetElement(I, J, V);
    }, Ar, Ax, Drr, Dxp, Dxx);
}

/**
 * Add elements to the function vector (all rows: the vector is complete on
 * every rank).
 */
void PXiInternalTrapping::AddToVectorElements_c(
    real_t *vec, const real_t *f,
    const real_t *const* dfr, const real_t *const*,
    const real_t *const* df2, const real_t *const* ddrr,
    const real_t *const*, const real_t *const*,
    const real_t *const* dd21, const real_t *const* dd22,
    jacobian_interp_mode set_mode
) {
    this->interp_mode = AdvectionInterpolationCoefficient::AD_INTERP_MODE_FULL;
    this->_addElements([&vec,&f](const len_t I, const len_t J, const real_t V) {
        vec[I] += V*f[J];
    }, dfr, df2, ddrr, dd21, dd22, set_mode);
}

/**
 * Internal routine for adding fluxes to matrix or vector. The purpose
 * of this routine is to add the flux from f(-xi_T^-}) to f(+xi_T^+),
 * i.e. from just _outside_ the trapped region at negative xi0, to just
 * _inside_ the trapped region at positive xi0.
 *
 * The coefficients on the faces at -xi_T (and the negative trapped radial
 * faces) are those gathered on every rank by the advection-diffusion term
 * when the arrays given are the operator's own coefficients (matrix and
 * function vector). Otherwise (derivatives of the coefficients, for the
 * jacobian) the arrays given are read, and only the interpolation deltas
 * come from the gathered values.
 */
void PXiInternalTrapping::_addElements(
    function<void(const len_t, const len_t, const real_t)> f,
    const real_t *const* cAr, const real_t *const* cAx, const real_t *const* cDrr,
    const real_t *const* cDxp, const real_t *const* cDxx,
    jacobian_interp_mode set_mode
) {
    if (cAr == nullptr && cAx == nullptr && cDrr == nullptr && cDxp == nullptr && cDxx == nullptr)
        return;

    const AdvectionDiffusionTerm *adt = this->oprtr->GetAdvectionDiffusion();
    if (adt == nullptr || !adt->HasTrappingCoefficients())
        throw FVMException(
            "PXiInternalTrapping: the coefficients on the trapped-passing boundary "
            "have not been gathered (AdvectionDiffusionTerm::GatherTrappingBoundaryCoefficients())."
        );

    // The operator's own coefficients are taken from the gathered values
    auto isOwn = [](const real_t *const* c, const real_t *const* own) {
        return (c == nullptr || c == own);
    };
    const bool fromGather =
        isOwn(cAr,  this->oprtr->GetAdvectionCoeffR())  &&
        isOwn(cAx,  this->oprtr->GetAdvectionCoeff2())  &&
        isOwn(cDrr, this->oprtr->GetDiffusionCoeffRR()) &&
        isOwn(cDxp, this->oprtr->GetDiffusionCoeff21()) &&
        isOwn(cDxx, this->oprtr->GetDiffusionCoeff22());

    const bool jacDeltas = (interp_mode == AdvectionInterpolationCoefficient::AD_INTERP_MODE_JACOBIAN);
    const len_t nr = this->grid->GetNr();

    len_t offset = 0;
    for (len_t ir = 0; ir < nr; ir++) {
        auto mg = this->grid->GetMomentumGrid(ir);
        const len_t np  = mg->GetNp1();
        const len_t nxi = mg->GetNp2();

        const real_t
            *Vp     = this->grid->GetVp(ir),
            *Vp_f2  = this->grid->GetVp_f2(ir),
            *dp_f   = mg->GetDp1(),
            *dxi0   = mg->GetDp2(),
            *dxi0_f = mg->GetDp2_f(),
            *dr     = this->grid->GetRadialGrid()->GetDr(),
            *dr_f   = this->grid->GetRadialGrid()->GetDr_f();

        const real_t *deltaRadialFlux = adt->GetRadialJacobianInterpolationCoeffs();

        // The gathered values must describe the same faces as this boundary condition
        if (this->nTrappedNegXi_indices[ir] > 0 &&
            adt->GetTrappingJm(ir) != (len_t)this->trappedNegXi_indices[ir][0])
            throw FVMException(
                "PXiInternalTrapping: the gathered face at -xi_T (j = " LEN_T_PRINTF_FMT
                ") differs from the trapped region of the boundary condition (j = " LEN_T_PRINTF_FMT
                ") at ir = " LEN_T_PRINTF_FMT ".",
                adt->GetTrappingJm(ir), (len_t)this->trappedNegXi_indices[ir][0], ir);
        if (adt->GetTrappingKm(ir).size() != (len_t)this->nTrappedNegXiRadial_indices[ir])
            throw FVMException(
                "PXiInternalTrapping: " LEN_T_PRINTF_FMT " gathered negative trapped radial lines, "
                LEN_T_PRINTF_FMT " in the boundary condition, at ir = " LEN_T_PRINTF_FMT ".",
                (len_t)adt->GetTrappingKm(ir).size(), (len_t)this->nTrappedNegXiRadial_indices[ir], ir);

        if(this->nTrappedNegXi_indices[ir] && (set_mode != JACOBIAN_SET_LOWER && set_mode != JACOBIAN_SET_UPPER)) {
            // indices indicating in which cells to mirror pitch fluxes
            len_t jm = trappedNegXi_indices[ir][0];
            len_t jp = trappedPosXi_indices[ir][0];
            AdvectionInterpolationCoefficient *delta2 = oprtr->GetInterpolationCoeff2();

            // Iterate over all p and set the fluxes...
            for (len_t i = 0; i < np; i++) {
                const len_t idxm = jm*np + i;
                const len_t idxp = jp*np + i;
                const real_t *g = adt->GetTrappingXi(ir, i);

                // XI ADVECTION
                real_t S_i;
                if (cAx != nullptr) {
                    const real_t Ax = fromGather ? g[AdvectionDiffusionTerm::TRAP_F2] : cAx[ir][idxm];
                    S_i = Ax * Vp_f2[idxm] / (Vp[idxp]*dxi0[jp]);
                    const real_t *delta = adt->GetTrappingXiDelta(ir, i, jacDeltas);
                    for (len_t n, k = delta2->GetKmin(jm, &n); k <= delta2->GetKmax(jm, nxi); k++, n++)
                        f(offset+idxp, offset+k*np+i, -S_i * delta[n]);
                }

                // XI-XI DIFFUSION
                if (jm > 0 && cDxx != nullptr) {
                    const real_t Dxx = fromGather ? g[AdvectionDiffusionTerm::TRAP_D22] : cDxx[ir][idxm];
                    S_i = Dxx * Vp_f2[idxm] / (Vp[idxp]*dxi0[jp]*dxi0_f[jm-1]);

                    f(offset+idxp, offset+jm*np+i,     +S_i);
                    f(offset+idxp, offset+(jm-1)*np+i, -S_i);
                }

                // XI-P DIFFUSION
                if (jm > 0 && (i > 0 && i < np-1) && cDxp != nullptr) {
                    const real_t Dxp = fromGather ? g[AdvectionDiffusionTerm::TRAP_D21] : cDxp[ir][idxm];
                    S_i = Dxp * Vp_f2[idxm] / (Vp[idxp]*dxi0[jp]*(dp_f[i]+dp_f[i-1]));

                    f(offset+idxp, offset+(jm-1)*np+i+1, +S_i);
                    f(offset+idxp, offset+jm*np+i+1,     +S_i);
                    f(offset+idxp, offset+(jm-1)*np+i-1, -S_i);
                    f(offset+idxp, offset+jm*np+i-1,     -S_i);
                }
            }
        }

        const real_t *Vp_fr = this->grid->GetVp_fr(ir);
        AdvectionInterpolationCoefficient *deltar = oprtr->GetInterpolationCoeffR();

        // unlike the xi fluxes, here there may be multiple radial fluxes to move for
        // each radius, so we sum over all pitch cells that may be mirrored
        // (entry q is line AdvectionDiffusionTerm::GetTrappingKm(ir)[q])
        for(len_t q=0; q<(len_t)nTrappedNegXiRadial_indices[ir]; q++){
            // pitch indices
            len_t km = trappedNegXiRadial_indices[ir][q];
            len_t kp = trappedPosXiRadial_indices[ir][q];
            // Iterate over all p and set the fluxes...
            for (len_t i = 0; i < np; i++) {
                const len_t idxm = km*np + i;
                const len_t idxp = kp*np + i;
                const real_t *g = adt->GetTrappingR(ir, q, i);

                // R ADVECTION
                real_t S_i;
                if (cAr != nullptr) {
                    const real_t Ar = fromGather ? g[AdvectionDiffusionTerm::TRAP_FR] : cAr[ir][idxm];
                    S_i = Ar * Vp_fr[idxm] / (Vp[idxp]*dr[ir]);

                    // Handle any radial interpolation
                    if (set_mode == JACOBIAN_SET_LOWER)
                        S_i *= 1 - deltaRadialFlux[ir];
                    else if (set_mode == JACOBIAN_SET_CENTER)
                        S_i *= deltaRadialFlux[ir];
                    else if (set_mode == JACOBIAN_SET_UPPER)
                        S_i = 0;

                    if (S_i) { // often we will not have radial fluxes, and can skip these calculations
                        const real_t *delta = adt->GetTrappingRDelta(ir, q, i, jacDeltas);
                        for (len_t n, k = deltar->GetKmin(ir, &n); k <= deltar->GetKmax(ir, nr); k++, n++)
                            f(offset+idxp, offset+(k-ir)*np*nxi + idxm, -S_i * delta[n]);
                    }
                }

                // R-R DIFFUSION
                if (ir > 0 && cDrr != nullptr) {
                    const real_t Drr = fromGather ? g[AdvectionDiffusionTerm::TRAP_DRR] : cDrr[ir][idxm];
                    S_i = Drr * Vp_fr[idxm] / (Vp[idxp]*dr[ir]*dr_f[ir-1]);

                    // Handle any radial interpolation
                    if (set_mode == JACOBIAN_SET_LOWER)
                        S_i *= 1 - deltaRadialFlux[ir];
                    else if (set_mode == JACOBIAN_SET_CENTER)
                        S_i *= deltaRadialFlux[ir];
                    else if (set_mode == JACOBIAN_SET_UPPER)
                        S_i = 0;

                    if(S_i){
                        f(offset+idxp, offset + idxm,        +S_i);
                        f(offset+idxp, offset-np*nxi + idxm, -S_i);
                    }
                }
            }
        }
        offset += np*nxi;
    }
}

/**
 * Reset the rows of the jacobian corresponding to -xi_T <= xi0 < 0.
 */
bool PXiInternalTrapping::SetJacobianBlock(
    const len_t uqtyId, const len_t derivId, Matrix *jac, const real_t*
) {
    if (uqtyId == derivId)
        this->SetMatrixElements(jac, nullptr);

    return (uqtyId == derivId);
}

/**
 * Reset the rows of the matrix corresponding to -xi_T <= xi0 < 0
 * and set f(xi0) = f(-xi0) in this region.
 *
 * Only the rows owned by this rank are set: 'rhs' only holds those
 * (rhs[I] is row I of this unknown, for the owned I).
 */
void PXiInternalTrapping::SetMatrixElements(
    Matrix *mat, real_t *rhs
) {
    PetscInt lo, hi;
    OwnedRows(mat, lo, hi);

    // Rows to reset owned by this rank
    vector<PetscInt> owned;
    owned.reserve(nRowsToReset);
    for (len_t it = 0; it < nRowsToReset; it++)
        if (rowsToReset[it] >= lo && rowsToReset[it] < hi)
            owned.push_back(rowsToReset[it]);

    // override diagonal and reset rhs in trapped region
    mat->SetDiagonalConstant((PetscInt)owned.size(), owned.data(), -1.0);
    if(rhs != nullptr)
        for (PetscInt row : owned)
            rhs[row] = 0;

    const len_t nr = this->grid->GetNr();
    len_t offset = 0;
    // Iterate over all rows with -xi_T <= xi0 < 0.
    for (len_t ir = 0; ir < nr; ir++) {
        // NOTE: Must be 'INSERT_VALUES' since we called 'SetDiagonalConstant(...)'
        // above and didn't call 'PartialAssemble()' after...
        offset += this->_setElements(
            ir, offset,
            [mat,lo,hi](const len_t I, const len_t J, const real_t v) {
                if ((PetscInt)I >= lo && (PetscInt)I < hi)
                    mat->SetElement(I, J, v, INSERT_VALUES);
            }
        );
    }
}

/**
 * Reset the elements of the vector corresponding to -xi_T <= xi0 < 0
 * and set f(xi0) = f(-xi0) in this region (all rows: the vector is
 * complete on every rank).
 */
void PXiInternalTrapping::SetVectorElements(real_t *vec, const real_t *f) {
    const len_t nr = this->grid->GetNr();
    len_t offset = 0;

    // Reset elements corresponding to -xi_T <= xi0 < 0
    for(len_t it=0; it<nRowsToReset; it++)
        vec[rowsToReset[it]] = -f[rowsToReset[it]];

    // Iterate over all rows with -xi_T <= xi0 < 0.
    for (len_t ir = 0; ir < nr; ir++) {
        offset += this->_setElements(
            ir, offset,
            [&vec,&f](const len_t I, const len_t J, const real_t v) { vec[I] += v*f[J]; }
        );
    }
}

/**
 * Internal routine used for setting f(xi0) = f(-xi0) in the region
 * xi_T <= xi0 < 0.
 *
 * offset: Offset of the current radial coordinate.
 * f:      Function which sets the element.
 */
len_t PXiInternalTrapping::_setElements(
    const len_t ir, const len_t offset,
    function<void(const len_t, const len_t, const real_t)> f
) {
    PetscInt nidcs = this->nTrappedNegXi_indices[ir];
    PetscInt *idcs = this->trappedNegXi_indices[ir];
    // Set the row to f(xi0) = f(-xi0)
    auto mg = this->grid->GetMomentumGrid(ir);
    const len_t np  = mg->GetNp1();
    const len_t nxi = mg->GetNp2();
    const real_t *xi0 = mg->GetP2();

    for (PetscInt j = 0; j < nidcs; j++) {
        const len_t J  = idcs[j];
        const len_t pJ = this->trappedPosXi_indices[ir][j];

        // interpolate in the direction of the closest grid point
        int_t interpolationDirection;
        if( xi0[pJ] > - xi0[J])
            interpolationDirection = -1;
        else
            interpolationDirection = 1;

        // Interpolation coefficient (xi0[J] is negative)
        real_t delta =
            (pJ == nxi-1) ?
                1.0 :
                (-xi0[J] - xi0[pJ]) / (xi0[pJ+interpolationDirection] - xi0[pJ]);
        // set nearly vanishing elements to identically zero to reduce nnz
        if(fabs(1-delta)<100*realeps)
            delta=1.0;
        else if (fabs(delta)<100*realeps)
            delta=0.0;

        // Avoid overwriting diagonal
        if (pJ+interpolationDirection == J)
            delta = 0;

        for (len_t i = 0; i < np; i++) {
            f(offset + J*np + i, offset + pJ*np + i, 1-delta);

            if (pJ+1 < nxi-1 && delta != 0)
                f(offset + J*np + i, offset + (pJ+interpolationDirection)*np + i, delta);
        }
    }

    return np*nxi;
}
